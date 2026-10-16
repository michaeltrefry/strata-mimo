"""Drive a loaded, isolated two-slot MiMo HTTP server; never starts/stops a model.

    python tools/mimo/serve_slots_smoke.py --base-url http://127.0.0.1:8096 --case all

Uses the public OpenAI-compatible API. Reports wall latency and overlap, not GPU service time.
The caller must choose the test instance, not a production endpoint. No third-party packages.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.parse
import urllib.request


def fetch(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=600)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--case", choices=("overlap", "cancel", "mixed", "prefill", "queue", "all"), default="overlap")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urllib.parse.urlsplit(base).hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("use an explicitly chosen loopback test server")
    with fetch(base, "/slots") as response:
        slots = json.load(response)
    if not isinstance(slots, list) or len(slots) != 2:
        raise AssertionError(f"test server does not advertise two slots: {slots}")
    with fetch(base, "/v1/models") as response:
        model = json.load(response)["data"][0]["id"]
    cases = ("overlap", "mixed", "cancel", "prefill", "queue") if args.case == "all" else (args.case,)
    reports = []
    for case in cases:
        if case == "prefill":
            results = prefill(base, model)
        elif case == "queue":
            results = queued_cancel(base, model)
        else:
            results = pair(base, model, case)
        wall = max(r["ended"] for r in results) - min(r["began"] for r in results)
        count = sum((r["usage"] or {}).get("completion_tokens", 0) for r in results)
        summary = {"case": case, "passed": True, "wall_s": round(wall, 3),
                   "completed_output_tokens": count,
                   "aggregate_completed_tok_s": round(count / wall, 2) if count and wall else None}
        reports.append({**summary, "requests": results})
        print(json.dumps({**summary,
                          "requests": [{k: v for k, v in r.items() if k not in ("content", "reasoning", "began", "first", "ended", "updates")}
                                       for r in results]}, ensure_ascii=False), flush=True)
    print(json.dumps({"passed": True, "model": model, "results": reports}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
