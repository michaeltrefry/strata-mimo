"""Opt-in long-turn HTTP orchestration and pure, fail-closed evidence checks.

Counting and generation both use the Anthropic-compatible rendering path. Count
results are preparation evidence, not native evidence: prompt_total from the
completed native request must agree. No code here starts or reconfigures a server.
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
import math
import statistics
import threading
import time
import urllib.parse
import urllib.request
import uuid


TARGETS = (25000, 50000, 77000, 100000)
REWIND_KINDS = {"repeat", "divergent", "repeat-after-branch", "solo", "competing"}
CONTINUATION_KINDS = {"append", "tool"}
REPLAY_KINDS = REWIND_KINDS | CONTINUATION_KINDS
RESTORE_REASONS = {"restored_checkpoint"}
DIAGNOSTICS = ("prefix_reason", "raw_lcp", "boundary_keep", "written_end", "resume",
               "checkpoint_boundary", "checkpoint_reason", "native_prepared",
               "prompt_service_ms", "decode_service_ms", "prompt_expert_hits", "prompt_expert_uses",
               "decode_expert_hits", "decode_expert_uses", "checkpoint_capture_ms", "checkpoint_restore_ms",
               "checkpoint_count", "checkpoint_bytes", "checkpoint_metadata_bytes", "checkpoint_captures",
               "checkpoint_restores", "checkpoint_boundaries", "command_id", "slot")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def idle(metrics):
    live = metrics.get("live", {})
    return (live.get("state") == "idle" and live.get("queued") == 0
            and live.get("active", 0) == 0
            and not any(s.get("busy") for s in live.get("slots", [])))


def prior_slot_boundaries(before, slot):
    """Inventory at the most recent completed use of this physical slot, BEFORE work."""
    prior = next((r for r in before.get("requests", []) if r.get("slot") == slot), None)
    return copy.deepcopy(prior.get("checkpoint_boundaries")) if prior is not None else None


def reuse_evidence(before, native, kind, counted):
    # An unchanged prompt independently guarantees all [0,N) tokens match its
    # earlier checkpoints. Edits/normalized continuations use the native exact
    # LCP as an explicit conservative eligibility bound, NOT chosen checkpoint C.
    unchanged = kind in {"repeat", "solo", "competing"}
    return {"prior_checkpoint_boundaries": prior_slot_boundaries(before, native.get("slot")),
            "known_matching_prefix_tokens": counted if unchanged else native.get("raw_lcp"),
            "matching_prefix_source": "unchanged_rendered_prompt" if unchanged else "native_exact_lcp"}


def verify_long_prefix_evidence(records, *, max_text_gap_s=10.0, require_overlap=True):
    """Return issues without I/O. Missing native facts are UNKNOWN, never zero.

    Each record has client timestamps/text, counted_prompt_tokens, requested_output,
    and native history under 'native'. The best eligible anchor lower bound comes
    from BEFORE-slot inventory plus explicit matching-prefix evidence, never from
    selected checkpoint C and never a fixed 4K interval. Live-tail continuation
    may retain live state; an aged rewind must restore a checkpoint at A=C-1.
    prompt_read counts fresh processed tokens; any API boundary allowance must be
    explicitly stated on the record (currently this harness always uses zero).
    SSE updates are text chunks, not native token timestamps or inter-token latency.
    """
    issues = []
    if not number(max_text_gap_s) or max_text_gap_s <= 0:
        return ["invalid streamed-text gap limit"]
    if not records:
        return ["no request evidence"]
    for record in records:
        if not isinstance(record, dict):
            issues.append("invalid request evidence record")
            continue
        label = record.get("name", "unnamed")
        def fail(message):
            issues.append(f"{label}: {message}")
        native = record.get("native")
        if not isinstance(native, dict):
            fail("UNKNOWN native history")
            continue
        for key in DIAGNOSTICS:
            if native.get(key) is None:
                fail(f"UNKNOWN {key}")
        for key in ("raw_lcp", "boundary_keep", "written_end", "resume", "checkpoint_boundary",
                    "prompt_service_ms", "decode_service_ms", "checkpoint_capture_ms",
                    "checkpoint_restore_ms", "checkpoint_count", "checkpoint_bytes",
                    "checkpoint_metadata_bytes", "checkpoint_captures", "checkpoint_restores",
                    "prompt_expert_hits", "prompt_expert_uses", "decode_expert_hits", "decode_expert_uses",
                    "reused", "prompt_read", "prompt_total", "engine_generated"):
            if not number(native.get(key)) or native[key] < 0:
                fail(f"UNKNOWN/invalid numeric {key}")
        if native.get("native_prepared") is not True:
            fail("native preparation not confirmed")
        boundaries = native.get("checkpoint_boundaries")
        if not isinstance(boundaries, list) or len(boundaries) > 8 or any(
                not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in boundaries):
            fail("UNKNOWN/invalid bounded checkpoint inventory")
        if number(native.get("checkpoint_count")) and not 0 <= native["checkpoint_count"] <= 8:
            fail("checkpoint count exceeds bounded policy")
        if number(native.get("checkpoint_bytes")) and native["checkpoint_bytes"] > 512 * 1024 * 1024:
            fail("checkpoint bytes exceed aggregate budget")
        if not isinstance(native.get("slot"), int) or isinstance(native.get("slot"), bool) or native["slot"] not in (0, 1):
            fail("UNKNOWN/invalid physical slot")
        if not isinstance(native.get("command_id"), (int, str)) or not str(native["command_id"]):
            fail("UNKNOWN command identity")
        n = native.get("prompt_total")
        counted = record.get("counted_prompt_tokens")
        if not number(counted) or counted <= 0 or counted != n:
            fail("count endpoint and actual native rendered prompt length disagree/UNKNOWN")
        if number(n) and number(native.get("resume")) and number(native.get("prompt_read")):
            if native["prompt_read"] != n - native["resume"]:
                fail("completed prompt processed-token accounting is not N-resume")
        target = record.get("target_tokens")
        if target is not None and (not number(target) or not number(n) or not target <= n <= target + 128):
            fail("actual rendered prefix outside target tolerance [target, target+128]")
        if not number(record.get("requested_output")) or record["requested_output"] < 512:
            fail("request must budget at least 512 output tokens")
        generated = native.get("engine_generated")
        if not number(generated) or generated <= 256:
            fail("actual native generation did not age the SWA ring (>256 required)")
        if native.get("finish") != "length":
            fail("early stop/incomplete completion cannot pass the long-output gate")
        if record.get("stream_complete") is not True or record.get("stop_reason") != "max_tokens":
            fail("missing complete bounded-length SSE terminal evidence")
        output_usage = (record.get("usage") or {}).get("output_tokens")
        if record.get("thinking_disabled") is not True:
            fail("output-count alignment requires reasoning disabled")
        if not number(output_usage) or output_usage != generated:
            fail("final API output usage and native generated count disagree/UNKNOWN")
        if output_usage != record.get("requested_output"):
            fail("bounded-length completion did not exhaust its requested output budget")
        if not record.get("content"):
            fail("missing actual assistant content")
        if record.get("kind") in REPLAY_KINDS:
            prior = record.get("prior_checkpoint_boundaries")
            matched = record.get("known_matching_prefix_tokens")
            if not isinstance(prior, list) or any(not number(c) or c <= 0 for c in prior):
                fail("UNKNOWN prior checkpoint inventory")
                prior = []
            if not number(matched) or not number(n) or not 0 <= matched <= n:
                fail("UNKNOWN explicit matching-prefix eligibility")
                matched = 0
            if record.get("before") is not None and prior != prior_slot_boundaries(record["before"], native.get("slot")):
                fail("prior inventory disagrees with before-request same-slot evidence")
            source = record.get("matching_prefix_source")
            if source not in {"unchanged_rendered_prompt", "native_exact_lcp"}:
                fail("UNKNOWN matching-prefix eligibility source")
            if source == "native_exact_lcp" and matched != native.get("raw_lcp"):
                fail("explicit eligibility disagrees with native exact LCP")
            if record.get("kind") in {"repeat", "solo", "competing"} and (
                    matched != n or source != "unchanged_rendered_prompt"):
                fail("unchanged repeat must use its full known rendered prefix for eligibility")
            # Lower bound is independently derived from BEFORE inventory and
            # known matching tokens. A selected small C cannot weaken this gate.
            anchors = [c - 1 for c in prior if number(n) and 1 < c <= n and c - 1 <= matched]
            best = max(anchors, default=0)
            resume = native.get("resume")
            if not number(resume) or resume <= 0 or native.get("reused") != resume:
                fail("no useful actual prefix reuse")
            elif number(n):
                if resume < best:
                    fail("best eligible prior anchor not reused")
                if not number(native.get("prompt_read")) or native["prompt_read"] > n - max(best, resume):
                    fail("excess fresh work: expected at most N-A, not full-prefix replay")
                if not number(native.get("raw_lcp")) or resume > native["raw_lcp"]:
                    fail("unsafe stale-branch restore: resume exceeds exact LCP")
                if not number(native.get("written_end")) or resume > native["written_end"]:
                    fail("unsafe restore beyond global KV write watermark")
            if record.get("processed_boundary_allowance", 0) != 0:
                fail("unsupported processed-token boundary allowance")
            live_continuation = (record.get("kind") in CONTINUATION_KINDS
                                 and native.get("prefix_reason") == "retained_live")
            if live_continuation:
                if native.get("resume") != native.get("boundary_keep"):
                    fail("live continuation did not use the actual retained live boundary")
            else:
                c = native.get("checkpoint_boundary")
                a = c - 1 if number(c) else None
                if best <= 0:
                    fail("no independently eligible useful retained checkpoint anchor")
                if not number(a) or a <= 0 or a != resume:
                    fail("selected checkpoint resume is not A=C-1")
                if native.get("prefix_reason") not in RESTORE_REASONS:
                    fail("aged-ring replay lacks checkpoint restoration reason")
                if native.get("checkpoint_reason") != "restored" or not number(native.get("checkpoint_restores")) or native["checkpoint_restores"] <= 0:
                    fail("checkpoint restore not confirmed")
        # Validate service facts independently of wall throughput.
        if number(native.get("decode_service_ms")) and native["decode_service_ms"] <= 0:
            fail("no charged decode service")
        for phase in ("prompt", "decode"):
            hits, uses = native.get(phase + "_expert_hits"), native.get(phase + "_expert_uses")
            if number(hits) and number(uses) and hits > uses:
                fail(f"invalid {phase} expert hit/use counts")
        if require_overlap and record.get("kind") == "competing":
            peer = record.get("peer") or {}
            began, first, ended = peer.get("began"), peer.get("first"), peer.get("ended")
            own_first, own_end = record.get("first"), record.get("ended")
            if not all(number(x) for x in (began, first, ended, own_first, own_end)):
                fail("UNKNOWN peer timing")
            else:
                overlap_end = min(first, own_end)
                if not own_first <= began < overlap_end or not began < first <= ended:
                    fail("no established decode / peer prefill overlap")
                updates = [t for t in record.get("updates", []) if number(t) and began < t < overlap_end]
                if not updates:
                    fail("decoder made no text progress during peer prefill")
                # Measure only while BOTH decode and peer prefill are active. Once decode finishes,
                # a continuing prompt read is not a stalled response. Include silence at both edges.
                edges = [began] + sorted(updates) + [overlap_end]
                if max(b - a for a, b in zip(edges, edges[1:])) > max_text_gap_s:
                    fail("excess streamed-text stall during peer prefill (not token ITL)")
            pn = peer.get("native") or {}
            for key in DIAGNOSTICS:
                if pn.get(key) is None:
                    fail(f"UNKNOWN peer {key}")
            if pn.get("native_prepared") is not True:
                fail("peer native preparation not confirmed")
            if (peer.get("stream_complete") is not True or peer.get("stop_reason") != "max_tokens"
                    or peer.get("thinking_disabled") is not True or pn.get("finish") != "length"
                    or (peer.get("usage") or {}).get("output_tokens") != pn.get("engine_generated")
                    or (peer.get("usage") or {}).get("output_tokens") != record.get("requested_output")):
                fail("incomplete/misaligned peer terminal evidence")
            if pn.get("command_id") == native.get("command_id") or pn.get("slot") == native.get("slot"):
                fail("peer command/slot identity not isolated")
            if not number(pn.get("prompt_read")) or pn["prompt_read"] < 27000:
                fail("no full 28K peer prefill progress (peer may already be cached)")
            if not number(pn.get("prompt_service_ms")) or pn["prompt_service_ms"] <= 0:
                fail("no peer prompt service evidence")
            if not number(pn.get("engine_generated")) or pn["engine_generated"] <= 0:
                fail("no peer completion progress")
            if pn.get("prompt_total") != peer.get("counted_prompt_tokens"):
                fail("peer actual rendered prompt count mismatch")
            if not number(pn.get("prompt_total")) or not 28000 <= pn["prompt_total"] <= 28128:
                fail("peer not actually 28K rendered tokens")
    return issues


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("redirect refused: smoke traffic must stay on the explicit loopback origin")


def open_local(request, timeout=600):
    url = urllib.parse.urlsplit(request.full_url)
    require(url.scheme in ("http", "https") and url.hostname in ("localhost", "127.0.0.1", "::1")
            and not url.username and not url.password, "loopback HTTP endpoint required")
    # Environment proxies and redirects must not silently send this workload elsewhere.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    return opener.open(request, timeout=timeout)


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")

    def json(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with open_local(req, timeout=600) as response:
            return json.load(response)

    def metrics(self):
        return self.json("/metrics")

    def count(self, model, messages):
        result = self.json("/v1/messages/count_tokens", payload(model, messages, 512, False))
        count = result.get("input_tokens")
        require(isinstance(count, int) and count > 0, "missing rendered token count")
        return count

    def request_messages(self, model, messages, max_tokens=512, started=None):
        return request_messages(self.base, model, messages, max_tokens, started)


def payload(model, messages, max_tokens, stream=True):
    return {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": stream, "temperature": 0, "thinking": {"type": "disabled"}}


def request_messages(base, model, messages, max_tokens=512, started=None):
    """Collect actual assistant blocks and SSE TEXT-update timestamps, not tokens."""
    req = urllib.request.Request(base + "/v1/messages",
                                 data=json.dumps(payload(model, messages, max_tokens)).encode(),
                                 headers={"Content-Type": "application/json"})
    began = time.perf_counter()
    updates, blocks, usage = [], {}, {}
    response_id, stop_reason = None, None
    saw_final_delta, saw_message_stop, final_output = False, False, None
    with open_local(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:].strip())
            kind = event.get("type")
            require(not saw_message_stop, "unexpected SSE event after message_stop")
            if kind == "error":
                raise RuntimeError(str(event.get("error")))
            if kind == "message_start":
                message = event["message"]
                response_id = message["id"]
                usage.update(message.get("usage") or {})
            elif kind == "content_block_start":
                blocks[event["index"]] = copy.deepcopy(event["content_block"])
            elif kind == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    block = blocks.setdefault(event["index"], {"type": "text", "text": ""})
                    block["text"] = block.get("text", "") + delta["text"]
                    updates.append(time.perf_counter())
                    if started:
                        started.set()
                elif delta.get("type") == "thinking_delta":
                    raise RuntimeError("thinking unexpectedly enabled: cannot normalize a text-only continuation")
            elif kind == "message_delta":
                require(not saw_final_delta, "duplicate final message_delta")
                saw_final_delta = True
                final_usage = event.get("usage") or {}
                final_output = final_usage.get("output_tokens")
                usage.update(final_usage)
                stop_reason = event.get("delta", {}).get("stop_reason")
            elif kind == "message_stop":
                require(saw_final_delta, "message_stop without final message_delta")
                saw_message_stop = True
    ended = time.perf_counter()
    content = [blocks[i] for i in sorted(blocks)]
    require(response_id and any(b.get("text") for b in content), "missing assistant text/identity")
    require(saw_final_delta and saw_message_stop, "truncated SSE: missing final message_delta/message_stop")
    require(isinstance(final_output, int) and not isinstance(final_output, bool) and final_output > 256,
            "missing/short final output-token usage")
    require(stop_reason == "max_tokens", "early stop cannot pass a bounded-length campaign")
    require(final_output == max_tokens, "final output usage did not exhaust the bounded output budget")
    return {"began": began, "first": updates[0] if updates else None, "ended": ended,
            "wall_s": ended - began, "content": content, "updates": updates,
            "usage": usage, "response_id": response_id, "stop_reason": stop_reason,
            "stream_complete": True, "thinking_disabled": True,
            "max_text_gap_s": max((b - a for a, b in zip(updates, updates[1:])), default=0)}


def plan_scenarios(targets=TARGETS, output_tokens=512):
    require(output_tokens >= 512, "long-turn output budget must be >=512")
    return {"targets_rendered_tokens": list(targets), "requested_output_tokens": output_tokens,
            "minimum_actual_native_output": 257, "rendered_count_tolerance": 128,
            "turns_per_target": ["seed", "repeat", "append", "tool", "divergent", "repeat-after-branch"],
            "matched": {"decoder_rendered_tokens": 77000, "peer_rendered_tokens": 28000,
                        "order": ["warmup", "solo", "competing"],
                        "text_gaps_are_not_token_itl": True},
            "checkpoint_gate": "best retained matching anchor A; fresh work <= N-A",
            "performance_validation": "pending matched baseline/current repeated same-hardware evidence"}


def counted_prefix(client, model, target, marker):
    """Find a bounded synthetic prefix by real rendered counts, never word estimates."""
    instruction = ("\nWrite a complete, very detailed Python module with docstrings and many tests. "
                   "Continue for at least 2000 output tokens; do not summarize.\n")
    def messages(rows):
        return [{"role": "user", "content": marker + "\n" + "".join(
            f"Record {i:07d}: blue square, green triangle, red circle.\n" for i in range(rows)) + instruction}]
    low, high = 0, 128
    count = client.count(model, messages(high))
    for _ in range(16):
        if count >= target:
            break
        low, high = high, high * 2
        require(high <= 2 * target + 1024, "count endpoint cannot reach target within bounded fixture size")
        count = client.count(model, messages(high))
    require(count >= target, "could not bracket requested rendered token count")
    for _ in range(24):
        if low + 1 >= high:
            break
        mid = (low + high) // 2
        if client.count(model, messages(mid)) < target:
            low = mid
        else:
            high = mid
    result = messages(high)
    actual = client.count(model, result)
    require(target <= actual <= target + 128, "token count target cannot be reached within tolerance")
    return result, actual


def continued_messages(seed, response, tool=False):
    """Append the ACTUAL assistant output, then a real user or normalized tool turn."""
    result = copy.deepcopy(seed)
    blocks = copy.deepcopy(response["content"])
    require(bool(blocks), "cannot continue without actual assistant content")
    if tool:
        # A deliberately supplied tool exchange, not a claim that the model called it.
        blocks.append({"type": "tool_use", "id": "smoke_tool_1", "name": "inspect_fixture", "input": {}})
        result.append({"role": "assistant", "content": blocks})
        result.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "smoke_tool_1", "content": "Fixture inspection passed."},
            {"type": "text", "text": "Now expand every test and explain each invariant in detail."}]})
    else:
        result += [{"role": "assistant", "content": blocks},
                   {"role": "user", "content": "Continue with many additional tests and detailed explanations."}]
    return result


def fresh_history(before, after, expected):
    total0 = before.get("totals", {}).get("requests")
    total1 = after.get("totals", {}).get("requests")
    require(isinstance(total0, int) and total1 == total0 + expected,
            "missing history or concurrent client detected by request totals")
    old = {json.dumps(r, sort_keys=True) for r in before.get("requests", [])}
    fresh = [r for r in after.get("requests", []) if json.dumps(r, sort_keys=True) not in old]
    require(len(fresh) == expected, "ambiguous/missing new request history")
    return fresh


def settled(client):
    for _ in range(100):
        metrics = client.metrics()
        if idle(metrics):
            return metrics
        time.sleep(.1)
    raise RuntimeError("test server did not return to idle")


def prepare_run(client, allow_busy=False):
    initial = client.metrics()
    require(initial.get("live", {}).get("state") in ("idle", "reading", "generating"),
            "requires an already-loaded test server; never load/restart one through this harness")
    require(allow_busy or idle(initial), "server busy/unknown; use an isolated idle endpoint (--allow-busy overrides refusal)")
    require(len(client.json("/slots")) == 2, "requires exactly two slots")
    engine = initial.get("engine", {})
    require(engine.get("model"), "missing loaded model identity")
    require(number(engine.get("max_context")), "UNKNOWN context capacity")
    return initial


def check_headroom(count, output, max_context):
    require(count + output + 64 <= max_context, "insufficient context headroom for counted prompt + bounded output")


def run_long_prefix(client, *, targets=TARGETS, output_tokens=512, allow_busy=False, evidence=None):
    plan_scenarios(targets, output_tokens)
    initial = prepare_run(client, allow_busy)
    model, context = initial["engine"]["model"], initial["engine"]["max_context"]
    records = []
    evidence = evidence if evidence is not None else {}
    evidence.update(initial=initial, records=records, after=None)
    for target in targets:
        seed, seed_count = counted_prefix(client, model, target, "long-smoke-" + uuid.uuid4().hex)
        history = seed
        previous = None
        for kind in ("seed", "repeat", "append", "tool", "divergent", "repeat-after-branch"):
            if kind == "repeat":
                history = seed
            elif kind in ("append", "tool"):
                history = continued_messages(history, previous, tool=kind == "tool")
            elif kind == "divergent":
                history = copy.deepcopy(seed)
                history[0]["content"] = history[0]["content"].replace(
                    "Write a complete, very detailed Python module", "Write an entirely different, detailed Rust module")
            elif kind == "repeat-after-branch":
                history = seed
            counted = seed_count if kind in ("seed", "repeat", "repeat-after-branch") else client.count(model, history)
            check_headroom(counted, output_tokens, context)
            before = client.metrics()
            require(allow_busy or idle(before), "concurrent client detected before request")
            evidence["pending"] = {"name": f"{target}-{kind}", "before": before,
                                   "counted_prompt_tokens": counted}
            response = client.request_messages(model, history, output_tokens)
            evidence["pending"]["response"] = response
            after = settled(client)
            evidence["pending"]["after"] = after
            native = fresh_history(before, after, 1)[0]
            record = {**response, "name": f"{target}-{kind}", "kind": kind,
                      "target_tokens": target if kind in ("seed", "repeat", "divergent", "repeat-after-branch") else None,
                      "counted_prompt_tokens": counted, "requested_output": output_tokens,
                      "aged_ring": previous is not None, "native": native,
                      "before": before, "after": after,
                      **reuse_evidence(before, native, kind, counted)}
            records.append(record)
            evidence.pop("pending", None)
            evidence["after"] = after
            previous = response
            issues = verify_long_prefix_evidence([record])
            require(not issues, "; ".join(issues))
    return evidence


def matched_summary(records):
    """Report comparable wall/service observations; this is not a scheduler A/B claim."""
    observations = []
    for record in records:
        native = record["native"]
        generated = native["engine_generated"]
        delivered = (record.get("usage") or {}).get("output_tokens")
        wall = record["ended"] - record["began"]
        service = native["decode_service_ms"]
        peer = record.get("peer")
        observations.append({"name": record["name"], "kind": record["kind"],
                             "wall_s": wall,
                             "api_output_tokens": delivered, "actual_native_output_tokens": generated,
                             "delivered_wall_tok_s": delivered / wall if number(delivered) and wall > 0 else None,
                             "native_wall_tok_s": generated / wall if wall > 0 else None,
                             "charged_decode_service_tok_s": generated * 1000 / service if service > 0 else None,
                             "prompt_service_ms": native["prompt_service_ms"],
                             "decode_service_ms": service,
                             "peer_first_text_s": peer["first"] - peer["began"] if peer else None,
                             "max_streamed_text_gap_s": max((b - a for a, b in zip(
                                 record.get("updates", []), record.get("updates", [])[1:])), default=0),
                             "checkpoint_capture_ms": native["checkpoint_capture_ms"],
                             "checkpoint_restore_ms": native["checkpoint_restore_ms"],
                             "checkpoint_bytes": native["checkpoint_bytes"],
                             "warmth": record["warmth"]})
    solo = [r["delivered_wall_tok_s"] for r in observations if r["kind"] == "solo" and number(r["delivered_wall_tok_s"])]
    mixed = [r["delivered_wall_tok_s"] for r in observations if r["kind"] == "competing" and number(r["delivered_wall_tok_s"])]
    solo_median = statistics.median(solo) if solo else None
    mixed_median = statistics.median(mixed) if mixed else None
    return {"observations": observations, "solo_median_wall_tok_s": solo_median,
            "competing_median_wall_tok_s": mixed_median,
            "competing_to_solo_delivery_ratio": mixed_median / solo_median if solo_median and mixed_median is not None else None,
            "text_gaps_are_not_token_itl": True,
            "comparison_scope": "same endpoint/model in one campaign; hardware/config snapshots attached; "
                                "no baseline scheduler run or 100 tok/s assertion"}


def run_matched(client, *, output_tokens=1024, repetitions=3, allow_busy=False, max_text_gap_s=10, evidence=None):
    require(repetitions >= 2, "matched evidence requires at least two repetitions")
    require(output_tokens >= 512, "output budget must be >=512")
    initial = prepare_run(client, allow_busy)
    model, context = initial["engine"]["model"], initial["engine"]["max_context"]
    decoder, n = counted_prefix(client, model, 77000, "matched-decoder-" + uuid.uuid4().hex)
    check_headroom(n, output_tokens, context)
    records = []
    evidence = evidence if evidence is not None else {}
    evidence.update(initial=initial, records=records)
    for iteration in range(repetitions):
        # A new leading identity prevents a repeated peer becoming a cached 1-token prefill.
        peer, pn = counted_prefix(client, model, 28000, "matched-peer-" + uuid.uuid4().hex)
        check_headroom(pn, output_tokens, context)
        for kind in ("warmup", "solo", "competing"):
            before = client.metrics()
            require(allow_busy or idle(before), "concurrent client detected")
            evidence["pending"] = {"name": f"matched-{iteration}-{kind}", "before": before}
            peer_result = None
            if kind != "competing":
                response = client.request_messages(model, decoder, output_tokens)
            else:
                started = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    running = pool.submit(client.request_messages, model, decoder, output_tokens, started)
                    require(started.wait(timeout=180), "decoder never produced established text")
                    peer_result = client.request_messages(model, peer, output_tokens)
                    response = running.result(timeout=600)
            evidence["pending"].update(response=response, peer=peer_result)
            after = settled(client)
            evidence["pending"]["after"] = after
            histories = fresh_history(before, after, 2 if peer_result else 1)
            selected = [r for r in histories if r.get("prompt_total") == n]
            require(len(selected) == 1, "cannot identify decoder native history by actual prompt count")
            record = {**response, "name": f"matched-{iteration}-{kind}", "kind": kind,
                      "target_tokens": 77000, "counted_prompt_tokens": n,
                      "requested_output": output_tokens, "native": selected[0],
                      "before": before, "after": after,
                      **reuse_evidence(before, selected[0], kind, n)}
            if peer_result:
                peers = [r for r in histories if r.get("prompt_total") == pn]
                require(len(peers) == 1, "cannot identify peer native history")
                require(peers[0].get("command_id") is not None
                        and peers[0]["command_id"] != selected[0].get("command_id")
                        and peers[0].get("slot") in (0, 1)
                        and peers[0]["slot"] != selected[0].get("slot")
                        and peer_result["response_id"] != response["response_id"],
                        "peer and decoder identities/physical slots not isolated")
                record["peer"] = {**peer_result, "native": peers[0], "counted_prompt_tokens": pn}
            records.append(record)
            evidence.pop("pending", None)
            evidence["after"] = after
            issues = verify_long_prefix_evidence([record], require_overlap=True, max_text_gap_s=max_text_gap_s)
            require(not issues, "; ".join(issues))
            # Warmth is observed (reused/restore diagnostics), never inferred from run order.
            record["warmth"] = {k: selected[0].get(k) for k in ("reused", "prefix_reason", "checkpoint_boundary")}
    evidence.update(performance_validated=False, matched_summary=matched_summary(records),
                    note="Current scheduler evidence only; needs a matched baseline campaign. "
                         "No 100 tok/s claim. Text-update gaps are not token ITL.")
    return evidence
