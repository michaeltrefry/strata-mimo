"""Plan a two-slot MiMo smoke campaign; no network unless --run is explicit.

    python tools/mimo/serve_slots_smoke.py --case long-prefix
    python tools/mimo/serve_slots_smoke.py --run --base-url http://127.0.0.1:8096 --case all

--run requires an isolated, loaded loopback test server, never production. Does not
start, stop, reload or configure a model. 'all' retains the five short cases; heavy
long-prefix/matched campaigns are separately opt-in. Stream gaps measure TEXT
updates, not native token ITL. No third-party packages.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import threading
import uuid
import time
import urllib.parse
import urllib.request

if __package__:
    from . import long_turn_smoke as long_turn
else:
    # Direct and isolated (-I) invocations may only import this trusted script directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import long_turn_smoke as long_turn


def fetch(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return long_turn.open_local(request, timeout=600)


def request(base, model, name, prompt, barrier=None, streaming=True, cancel_after=None, started=None):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": streaming, "max_tokens": 256, "temperature": 0, "reasoning_effort": "low"}
    if barrier:
        barrier.wait(timeout=30)
    began = time.perf_counter()
    first = None
    text, reasoning, ids, updates = [], [], set(), []
    usage = None
    cancelled = False
    with fetch(base, "/v1/chat/completions", payload) as response:
        if not streaming:
            body = json.load(response)
            choice = body["choices"][0]["message"]
            text.append(choice.get("content") or "")
            reasoning.append(choice.get("reasoning_content") or "")
            usage = body.get("usage")
            ids.add(body["id"])
        else:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                event = json.loads(value)
                if event.get("error"):
                    raise RuntimeError(f"{name}: {event['error']}")
                if event.get("id"):
                    ids.add(event["id"])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content, thought = delta.get("content") or "", delta.get("reasoning_content") or ""
                    if content or thought:
                        updates.append(time.perf_counter())
                        if first is None:
                            first = updates[-1]
                            if started:
                                started.set()
                    text.append(content)
                    reasoning.append(thought)
                if cancel_after and sum(map(len, text)) + sum(map(len, reasoning)) >= cancel_after:
                    cancelled = True
                    break
    ended = time.perf_counter()
    content, thought = "".join(text), "".join(reasoning)
    if not content and not thought:
        raise AssertionError(f"{name}: no generated text")
    if len(ids) != 1:
        raise AssertionError(f"{name}: response identity changed: {ids}")
    return {"name": name, "began": began, "first": first, "ended": ended,
            "wall_s": round(ended - began, 3), "first_text_s": round(first - began, 3) if first else None,
            "cancelled": cancelled, "response_id": next(iter(ids)), "usage": usage,
            "max_text_gap_s": round(max((b - a for a, b in zip(updates, updates[1:])), default=0.0), 3),
            "content": content, "reasoning": thought, "updates": updates}


PROMPTS = (
    "Write a complete Python implementation of alpha_cache, a bounded thread-safe LRU cache. "
    "Include a detailed docstring, type hints, and eight unittest cases. Return code, not a summary.",
    "Write a complete Python implementation of beta_counter, a thread-safe sliding-window rate limiter. "
    "Include a detailed docstring, type hints, and eight unittest cases. Return code, not a summary.",
)


def pair(base, model, case):
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(request, base, model, "A", PROMPTS[0], barrier)
        b = pool.submit(request, base, model, "B", PROMPTS[1], barrier,
                        case != "mixed", 96 if case == "cancel" else None)
        results = [a.result(timeout=600), b.result(timeout=600)]
    if results[0]["response_id"] == results[1]["response_id"]:
        raise AssertionError("the two requests shared a response identity")
    if case == "overlap":
        if max(r["first"] for r in results) >= min(r["ended"] for r in results):
            raise AssertionError("both streams did not produce text before either completed")
    if case == "cancel" and not results[1]["cancelled"]:
        raise AssertionError("cancellation fixture did not close its client stream")
    return results


def prefill(base, model):
    started = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(request, base, model, "A", PROMPTS[0], started=started)
        if not started.wait(timeout=180):
            raise AssertionError("established stream never produced text")
        # A distinct, several-thousand-token prompt. Context admission remains the server's responsibility.
        context = "\n".join(f"def fixture_{i}(value): return value + {i}" for i in range(400))
        b = pool.submit(request, base, model, "B", context + "\n\n" + PROMPTS[1])
        results = [a.result(timeout=600), b.result(timeout=600)]
    if results[0]["ended"] <= results[1]["began"]:
        raise AssertionError("long prompt was not submitted while A was active")
    if not any(results[1]["began"] < t < results[1]["first"] for t in results[0]["updates"]):
        raise AssertionError("established stream did not progress while the long request awaited its first text")
    return results


def queued_cancel(base, model):
    started = [threading.Event(), threading.Event()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(request, base, model, "A", PROMPTS[0], started=started[0])
        b = pool.submit(request, base, model, "B", PROMPTS[1], started=started[1])
        if not all(event.wait(timeout=180) for event in started):
            raise AssertionError("both active streams must start before the queued request")
        payload = {"model": model, "stream": True, "max_tokens": 128, "reasoning_effort": "low",
                   "messages": [{"role": "user", "content": "Write a complete gamma_queue implementation."}]}
        with fetch(base, "/v1/chat/completions", payload) as response:
            # The API sends its assistant-role envelope before waiting for an inference lease.
            while True:
                line = response.readline()
                if not line:
                    raise AssertionError("queued stream closed before its response envelope")
                if line.startswith(b"data:"):
                    break
            deadline = time.monotonic() + 10
            while True:
                with fetch(base, "/status") as status:
                    queued = json.load(status).get("queued", 0)
                if queued:
                    if a.done() or b.done():
                        raise AssertionError("queue fixture lost an active request before cancellation")
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("third request never entered the admission queue")
                time.sleep(0.05)
        # Closing the queued client's socket must remove its admission without disturbing A/B.
        deadline = time.monotonic() + 10
        while True:
            with fetch(base, "/status") as status:
                queued = json.load(status).get("queued", 0)
            if not queued:
                break
            if time.monotonic() > deadline:
                raise AssertionError("disconnected queued request did not leave the admission queue")
            time.sleep(0.05)
        return [a.result(timeout=600), b.result(timeout=600)]


def main(argv=None, client_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8096")
    parser.add_argument("--case", choices=("overlap", "cancel", "mixed", "prefill", "queue", "all",
                                          "long-prefix", "matched"), default="overlap")
    parser.add_argument("--run", action="store_true", help="explicitly enable HTTP/inference on the chosen server")
    parser.add_argument("--allow-busy", action="store_true", help="override initial busy refusal; does not disable evidence checks")
    parser.add_argument("--output", type=Path, help="new JSON artifact; existing paths are never overwritten")
    parser.add_argument("--target-tokens", type=int, nargs="+", choices=long_turn.TARGETS, default=list(long_turn.TARGETS))
    parser.add_argument("--output-tokens", type=int, default=1024, help="long cases only; >=512 requested, >256 actually generated")
    parser.add_argument("--repetitions", type=int, default=3, help="matched warmup/solo/competing repetitions; >=2")
    parser.add_argument("--max-text-gap-s", type=float, default=10.0, help="mixed-case streamed-text stall limit, NOT token ITL")
    parser.add_argument("--scheduler-label", default="unspecified", help="evidence label only; never changes server configuration")
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    url = urllib.parse.urlsplit(base)
    if (url.scheme not in ("http", "https") or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username or url.password or url.path or url.query or url.fragment):
        parser.error("use an explicitly chosen loopback test server origin without credentials/path/query")
    if args.output_tokens < 512 or args.repetitions < 2 or not long_turn.number(args.max_text_gap_s) or args.max_text_gap_s <= 0:
        parser.error("require output-tokens >=512, repetitions >=2 and a finite positive text-gap limit")
    plan = {"case": args.case, "base_url": base, "run": args.run,
            "scheduler_label": args.scheduler_label, "allow_busy_override": args.allow_busy,
            "repetitions": args.repetitions, "max_streamed_text_gap_s": args.max_text_gap_s,
            "scenario_plan": long_turn.plan_scenarios(args.target_tokens, args.output_tokens)}
    if not args.run:
        print(json.dumps({"dry_run": True, "network_contacted": False, **plan}, indent=2))
        return 0
    output_path = args.output or Path(f"mimo-smoke-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:12]}.json")
    # Reserve artifact BEFORE even status HTTP; a typo cannot overwrite evidence.
    with output_path.open("x", encoding="utf-8") as artifact:
        report = {**plan, "passed": False, "performance_validated": False}
        try:
            client = (client_factory or long_turn.Client)(base)
            if args.case == "long-prefix":
                report.update(long_turn.run_long_prefix(client, targets=args.target_tokens,
                                                       output_tokens=args.output_tokens, allow_busy=args.allow_busy, evidence=report))
            elif args.case == "matched":
                report.update(long_turn.run_matched(client, output_tokens=args.output_tokens,
                                                   repetitions=args.repetitions, allow_busy=args.allow_busy,
                                                   max_text_gap_s=args.max_text_gap_s, evidence=report))
            else:
                initial = long_turn.prepare_run(client, args.allow_busy)
                model = initial["engine"]["model"]
                cases = ("overlap", "mixed", "cancel", "prefill", "queue") if args.case == "all" else (args.case,)
                reports = []
                report.update(initial=initial, results=reports)
                for case in cases:
                    if not args.allow_busy and not long_turn.idle(client.metrics()):
                        raise RuntimeError("concurrent client detected before short case")
                    if case == "prefill":
                        results = prefill(base, model)
                    elif case == "queue":
                        results = queued_cancel(base, model)
                    else:
                        results = pair(base, model, case)
                    wall = max(r["ended"] for r in results) - min(r["began"] for r in results)
                    count = sum((r["usage"] or {}).get("completion_tokens", 0) for r in results)
                    reports.append({"case": case, "passed": True, "wall_s": round(wall, 3),
                                    "completed_output_tokens": count,
                                    "aggregate_completed_tok_s": round(count / wall, 2) if count and wall else None,
                                    "requests": results})
                report["after"] = long_turn.settled(client)
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            json.dump(report, artifact, indent=2, ensure_ascii=False)
            artifact.write("\n")
    print(json.dumps({"passed": True, "performance_validated": False, "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    main()
