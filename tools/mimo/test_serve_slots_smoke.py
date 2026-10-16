"""CPU-only tests: no endpoint, model, GPU, background server or downloads.

    python -m unittest tools.mimo.test_serve_slots_smoke
    python tools/mimo/test_serve_slots_smoke.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

try:
    from . import long_turn_smoke as smoke
    from . import serve_slots_smoke as cli
except ImportError:
    import long_turn_smoke as smoke
    import serve_slots_smoke as cli


def native_record(n=25000, a=24999, command=1, slot=0):
    return {"time": command, "command_id": command, "slot": slot,
            "prompt_total": n, "prompt_tokens": n, "reused": a, "prompt_read": n - a,
            "engine_generated": 512, "finish": "length", "native_prepared": True,
            "prefix_reason": "restored_checkpoint", "raw_lcp": n, "boundary_keep": n - 2,
            "written_end": n + 512, "resume": a, "checkpoint_boundary": a + 1,
            "checkpoint_reason": "restored", "checkpoint_boundaries": [a + 1],
            "prompt_service_ms": 5.0, "decode_service_ms": 400.0,
            "prompt_expert_hits": 1, "prompt_expert_uses": 2,
            "decode_expert_hits": 20, "decode_expert_uses": 30,
            "checkpoint_capture_ms": 1.0, "checkpoint_restore_ms": 2.0,
            "checkpoint_count": 1, "checkpoint_bytes": 27197440,
            "checkpoint_metadata_bytes": 100000, "checkpoint_captures": 1,
            "checkpoint_restores": 1}


def replay_record():
    return {"name": "repeat", "kind": "repeat", "target_tokens": 25000,
            "counted_prompt_tokens": 25000, "requested_output": 512, "aged_ring": True,
            "content": [{"type": "text", "text": "actual generated assistant output"}],
            "native": native_record(), "began": 0, "first": 1, "ended": 20,
            "prior_checkpoint_boundaries": [4096, 25000], "known_matching_prefix_tokens": 25000,
            "matching_prefix_source": "unchanged_rendered_prompt",
            "stream_complete": True, "thinking_disabled": True,
            "stop_reason": "max_tokens", "usage": {"output_tokens": 512},
            "updates": [1, 2, 4, 6, 9, 12, 19]}


def competing_record():
    record = replay_record()
    record.update(kind="competing", target_tokens=77000, counted_prompt_tokens=77000,
                  prior_checkpoint_boundaries=[4096, 77000], known_matching_prefix_tokens=77000,
                  native=native_record(77000, 76999))
    peer = native_record(28000, 0, 2, 1)
    peer.update(prompt_read=28000, prompt_service_ms=200, prefix_reason="empty_slot")
    record["peer"] = {"began": 3, "first": 8, "ended": 22,
                      "stream_complete": True, "thinking_disabled": True,
                      "stop_reason": "max_tokens", "usage": {"output_tokens": 512},
                      "counted_prompt_tokens": 28000, "native": peer}
    return record


class FakeClient:
    """Records requests and provides scripted native facts; never touches urllib."""
    def __init__(self, base="", *, busy=False, context=131072, generated=None):
        self.history = []
        self.calls = []
        self.busy = busy
        self.context = context
        self.generated = generated
        self.lock = threading.Lock()

    def metrics(self):
        with self.lock:
            return {"engine": {"model": "fake-mimo", "max_context": self.context},
                    "live": {"state": "generating" if self.busy else "idle", "queued": 0,
                             "active": int(self.busy)},
                    "totals": {"requests": len(self.history)},
                    "requests": copy.deepcopy(self.history[::-1]),
                    "hardware_static": {"gpu": "fake CPU-only test device"}}

    def json(self, path):
        if path == "/slots":
            return [{"id": 0}, {"id": 1}]
        raise AssertionError(f"unexpected path {path}")

    def count(self, model, messages):
        text = json.dumps(messages)
        # Deterministic fake tokenizer, deliberately unlike character/word estimates.
        return 20 + 13 * len(re.findall(r"Record \d{7}", text)) + 500 * sum(
            m["role"] == "assistant" for m in messages)

    def request_messages(self, model, messages, max_tokens=512, started=None):
        with self.lock:
            self.calls.append(copy.deepcopy(messages))
            n = self.count(model, messages)
            peer = "matched-peer-" in json.dumps(messages)
            prior = next((r for r in self.history[::-1] if r["slot"] == int(peer)), None)
            a = 0 if prior is None or peer else max(
                (c - 1 for c in prior["checkpoint_boundaries"] if c <= n), default=0)
            native = native_record(n, a, len(self.history) + 1, int(peer))
            native["checkpoint_boundaries"] = [4096, n]
            native["checkpoint_count"] = 2
            generated = max_tokens if self.generated is None else self.generated
            native["engine_generated"] = generated
            if a == 0:
                native.update(prefix_reason="empty_slot", checkpoint_reason="none", checkpoint_restores=0)
            self.history.append(native)
        if started:
            started.set()
        return {"response_id": f"fake-{native['command_id']}", "content": [{"type": "text", "text": "Actual answer."}],
                "began": 3 if peer else 0, "first": 8 if peer else 1,
                "ended": 22 if peer else 20, "wall_s": 19 if peer else 20,
                "updates": [8, 10, 15] if peer else [1, 2, 4, 6, 9, 12, 19],
                "usage": {"output_tokens": generated}, "stop_reason": "max_tokens",
                "stream_complete": True, "thinking_disabled": True}


class ValidatorTests(unittest.TestCase):
    def test_valid_prompt_end_and_geometric_anchors(self):
        record = replay_record()
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))
        # Geometric spacing does NOT promise <=4K fresh tokens for arbitrary edits.
        record["native"].update(checkpoint_boundary=8192, reused=8191, resume=8191,
                                prompt_read=16809, checkpoint_boundaries=[8192])
        record["prior_checkpoint_boundaries"] = [4096, 8192]
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))

    def test_divergent_exact_lcp_can_restore_boundary_one_beyond_it(self):
        record = replay_record()
        record.update(kind="divergent", known_matching_prefix_tokens=8191,
                      matching_prefix_source="native_exact_lcp", prior_checkpoint_boundaries=[8192])
        record["native"].update(checkpoint_boundary=8192, resume=8191, reused=8191,
                                raw_lcp=8191, boundary_keep=8190, prompt_read=16809,
                                checkpoint_boundaries=[8192])
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))
        record["prior_checkpoint_boundaries"] = [4096, 8192]
        record["native"].update(checkpoint_boundary=4096, resume=4095, reused=4095, prompt_read=20905)
        self.assertIn("best eligible prior anchor", "; ".join(smoke.verify_long_prefix_evidence([record])))

    def test_checkpoint_beyond_target_length_is_not_an_eligible_anchor(self):
        record = replay_record()
        record["prior_checkpoint_boundaries"] = [25000, 25001]
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))

    def test_valid_live_assistant_and_tool_continuation(self):
        for kind in ("append", "tool"):
            with self.subTest(kind=kind):
                record = replay_record()
                record.update(kind=kind, target_tokens=None, counted_prompt_tokens=26000,
                              known_matching_prefix_tokens=25512, matching_prefix_source="native_exact_lcp")
                record["native"] = native_record(26000, 25510)
                record["native"].update(prefix_reason="retained_live", boundary_keep=25510,
                                        raw_lcp=25512, checkpoint_boundary=0,
                                        checkpoint_reason="not_needed", checkpoint_restores=0)
                self.assertEqual([], smoke.verify_long_prefix_evidence([record]))
                record["native"]["resume"] -= 1
                record["native"]["reused"] -= 1
                record["native"]["prompt_read"] += 1
                self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_wrong_small_selected_anchor_cannot_weaken_before_inventory_gate(self):
        record = replay_record()
        record["native"].update(checkpoint_boundary=4096, resume=4095, reused=4095,
                                prompt_read=20905)
        self.assertIn("best eligible prior anchor", "; ".join(smoke.verify_long_prefix_evidence([record])))
        record["known_matching_prefix_tokens"] = 4096
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_explicit_divergent_eligibility_not_selected_checkpoint(self):
        record = replay_record()
        record.update(kind="divergent", known_matching_prefix_tokens=10000,
                      matching_prefix_source="native_exact_lcp", prior_checkpoint_boundaries=[4096, 8192, 25000])
        record["native"].update(checkpoint_boundary=8192, resume=8191, reused=8191,
                                raw_lcp=10000, prompt_read=16809)
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))
        record["native"].update(checkpoint_boundary=4096, resume=4095, reused=4095, prompt_read=20905)
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_api_native_count_and_terminal_alignment(self):
        for key, value in (("stream_complete", False), ("thinking_disabled", False),
                           ("stop_reason", "end_turn"), ("usage", {"output_tokens": 511})):
            with self.subTest(key=key):
                record = replay_record()
                record[key] = value
                self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_live_boundary_is_not_checkpoint_resume(self):
        record = replay_record()
        record["native"]["boundary_keep"] = 12000
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))

    def test_checkpoint_boundary_can_be_lcp_plus_one(self):
        record = replay_record()
        record["native"]["raw_lcp"] = 24999
        self.assertEqual([], smoke.verify_long_prefix_evidence([record]))

    def test_reused_zero_mutation(self):
        record = replay_record()
        record["native"]["reused"] = 0
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_full_prefix_read_mutation(self):
        record = replay_record()
        record["native"]["prompt_read"] = 25000
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_every_missing_diagnostic_fails_closed(self):
        for key in smoke.DIAGNOSTICS:
            with self.subTest(key=key):
                record = replay_record()
                del record["native"][key]
                self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_missing_actual_count_is_not_an_estimate(self):
        for key in ("prompt_total", "engine_generated", "reused", "prompt_read"):
            record = replay_record()
            del record["native"][key]
            self.assertTrue(smoke.verify_long_prefix_evidence([record]))
        record = replay_record()
        record["counted_prompt_tokens"] += 1
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_256_actual_output_even_if_1024_requested_fails(self):
        record = replay_record()
        record["requested_output"] = 1024
        record["native"]["engine_generated"] = 256
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_early_stop_is_not_a_long_output_pass(self):
        record = replay_record()
        record["native"].update(finish="stop", engine_generated=300)
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_stale_branch_restore_mutations(self):
        for key, value in (("raw_lcp", 10000), ("written_end", 10000),
                           ("checkpoint_reason", "identity_mismatch"),
                           ("prefix_reason", "retained_live"), ("checkpoint_restores", 0)):
            with self.subTest(key=key):
                record = replay_record()
                record["kind"] = "repeat-after-branch"
                record["native"][key] = value
                self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_no_fixed_interval_bound_or_unknown_zero(self):
        self.assertTrue(smoke.verify_long_prefix_evidence([]))
        record = replay_record()
        record["native"]["decode_service_ms"] = None
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))
        record["native"]["decode_service_ms"] = float("nan")
        self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_bounded_checkpoint_policy(self):
        for key, value in (("checkpoint_count", 9), ("checkpoint_bytes", 513 * 1024 * 1024),
                           ("checkpoint_boundaries", list(range(1, 10)))):
            record = replay_record()
            record["native"][key] = value
            self.assertTrue(smoke.verify_long_prefix_evidence([record]))

    def test_overlap_good_and_mutations(self):
        self.assertEqual([], smoke.verify_long_prefix_evidence([competing_record()], require_overlap=True))
        for change in ("no-overlap", "no-peer-progress", "no-decoder-progress", "excess-stall", "cached-peer"):
            with self.subTest(change=change):
                record = competing_record()
                if change == "no-overlap":
                    record["peer"]["began"] = 21
                elif change == "no-peer-progress":
                    record["peer"]["native"]["prompt_service_ms"] = 0
                elif change == "no-decoder-progress":
                    record["updates"] = [1, 2, 9, 12]
                elif change == "excess-stall":
                    record["peer"]["first"] = 19
                    record["updates"] = [1, 2, 4]
                else:
                    record["peer"]["native"]["prompt_read"] = 1
                self.assertTrue(smoke.verify_long_prefix_evidence([record], require_overlap=True))

    def test_completed_decoder_does_not_stall_until_peer_first_text(self):
        record = competing_record()
        record["peer"].update(first=80, ended=90)
        self.assertEqual([], smoke.verify_long_prefix_evidence([record], require_overlap=True))

    def test_decoder_completion_still_bounds_its_actual_active_silence(self):
        record = competing_record()
        record["peer"].update(first=80, ended=90)
        record["updates"] = [1, 2, 4]
        issues = smoke.verify_long_prefix_evidence([record], require_overlap=True)
        self.assertTrue(any("excess streamed-text stall" in issue for issue in issues))

    def test_peer_arriving_after_decoder_finished_has_no_overlap(self):
        record = competing_record()
        record["peer"].update(began=21, first=80, ended=90)
        issues = smoke.verify_long_prefix_evidence([record], require_overlap=True)
        self.assertTrue(any("no established decode / peer prefill overlap" in issue for issue in issues))

    def test_stall_includes_silent_endpoints(self):
        record = competing_record()
        record["updates"] = [7.9]
        self.assertTrue(smoke.verify_long_prefix_evidence([record], require_overlap=True, max_text_gap_s=2))


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden in CPU tests"))
        self.network.start()
        self.addCleanup(self.network.stop)
        opener = mock.patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("network forbidden in CPU tests"))
        opener.start()
        self.addCleanup(opener.stop)

    def test_plan_has_all_required_targets(self):
        plan = smoke.plan_scenarios()
        self.assertEqual([25000, 50000, 77000, 100000], plan["targets_rendered_tokens"])
        self.assertGreaterEqual(plan["requested_output_tokens"], 512)

    def test_dry_run_all_cases_never_instantiates_client_or_opens_artifact(self):
        for case in ("all", "overlap", "mixed", "cancel", "prefill", "queue", "long-prefix", "matched"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
                path = Path(tmp) / "absent.json"
                factory = mock.Mock(side_effect=AssertionError("dry-run constructed HTTP client"))
                self.assertEqual(0, cli.main(["--case", case, "--output", str(path)], client_factory=factory))
                factory.assert_not_called()
                self.assertFalse(path.exists())
                self.assertFalse(json.loads(out.getvalue())["network_contacted"])

    def test_isolated_script_dry_run_unreachable_loopback_no_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "must-not-exist.json"
            result = subprocess.run([sys.executable, "-I", str(Path(cli.__file__).resolve()),
                                     "--base-url", "http://127.0.0.1:1", "--case", "long-prefix",
                                     "--output", str(path)], cwd=tmp, capture_output=True, text=True, timeout=15)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertFalse(json.loads(result.stdout)["network_contacted"])
            self.assertFalse(path.exists())

    def test_remote_urls_are_refused_before_network(self):
        for url in ("http://example.org", "file://localhost/foo", "http://localhost@evil.test", "http://127.0.0.1/path"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["--base-url", url])

    def test_existing_artifact_refused_before_client_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.json"
            path.write_text("keep", encoding="utf-8")
            factory = mock.Mock(side_effect=AssertionError("unexpected client"))
            with self.assertRaises(FileExistsError):
                cli.main(["--run", "--output", str(path)], client_factory=factory)
            factory.assert_not_called()
            self.assertEqual("keep", path.read_text(encoding="utf-8"))

    def test_busy_and_unknown_refused_without_override(self):
        with self.assertRaisesRegex(RuntimeError, "busy"):
            smoke.prepare_run(FakeClient(busy=True))
        self.assertTrue(smoke.prepare_run(FakeClient(busy=True), allow_busy=True))
        self.assertFalse(smoke.idle({}))

    def test_four_targets_and_actual_assistant_tool_continuations(self):
        client = FakeClient()
        result = smoke.run_long_prefix(client)
        self.assertEqual(24, len(result["records"]))
        self.assertEqual([], smoke.verify_long_prefix_evidence(result["records"]))
        append = client.calls[2]
        self.assertEqual("Actual answer.", append[1]["content"][0]["text"])
        tool = client.calls[3]
        self.assertEqual("tool_use", tool[-2]["content"][-1]["type"])
        self.assertEqual("tool_result", tool[-1]["content"][0]["type"])
        self.assertNotEqual(client.calls[0], client.calls[4])
        self.assertEqual(client.calls[0], client.calls[5])

    def test_context_headroom_refuses_before_generation(self):
        client = FakeClient(context=25001)
        with self.assertRaisesRegex(RuntimeError, "headroom"):
            smoke.run_long_prefix(client, targets=(25000,))
        self.assertEqual([], client.calls)

    def test_short_generation_failure_retains_evidence_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure.json"
            with self.assertRaisesRegex(RuntimeError, "generation"):
                cli.main(["--run", "--case", "long-prefix", "--target-tokens", "25000", "--output", str(path)],
                         client_factory=lambda base: FakeClient(generated=256))
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(result["passed"])
            self.assertIn("initial", result)
            self.assertIn("after", result)
            self.assertEqual(256, result["records"][0]["native"]["engine_generated"])

    def test_matched_fake_campaign_and_new_peer_each_repetition(self):
        client = FakeClient()
        result = smoke.run_matched(client, repetitions=2)
        self.assertEqual(6, len(result["records"]))
        self.assertFalse(result["performance_validated"])
        self.assertEqual([], smoke.verify_long_prefix_evidence(result["records"], require_overlap=True))
        peers = [call for call in client.calls if "matched-peer-" in json.dumps(call)]
        self.assertEqual(2, len(peers))
        self.assertNotEqual(peers[0], peers[1])

    def test_history_totals_detect_unrelated_traffic(self):
        before = FakeClient().metrics()
        after = copy.deepcopy(before)
        after["totals"]["requests"] = 2
        with self.assertRaisesRegex(RuntimeError, "concurrent"):
            smoke.fresh_history(before, after, 1)

    def test_proxy_and_redirect_guard_without_network(self):
        import urllib.request
        request = urllib.request.Request("http://127.0.0.1:8096/metrics")
        opener = mock.Mock()
        with mock.patch("urllib.request.build_opener", return_value=opener) as build:
            smoke.open_local(request)
            handlers = build.call_args.args
            self.assertEqual({}, handlers[0].proxies)
            self.assertIsInstance(handlers[1], smoke.NoRedirect)
        with self.assertRaisesRegex(RuntimeError, "redirect"):
            smoke.NoRedirect().redirect_request(request, None, 302, "Found", {}, "https://example.org")
        with self.assertRaisesRegex(RuntimeError, "loopback"):
            smoke.open_local(urllib.request.Request("https://example.org/metrics"))

    def test_anthropic_sse_collects_actual_text_and_usage(self):
        events = [
            {"type": "message_start", "message": {"id": "msg-fixture", "usage": {"input_tokens": 25000}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Actual "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "answer"}},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 512}},
            {"type": "message_stop"}]
        data = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        started = threading.Event()
        with mock.patch.object(smoke, "open_local", return_value=io.BytesIO(data)) as opened:
            response = smoke.request_messages("http://127.0.0.1:8096", "fake", [{"role": "user", "content": "hello"}], started=started)
        self.assertTrue(started.is_set())
        self.assertEqual([{"type": "text", "text": "Actual answer"}], response["content"])
        self.assertEqual(2, len(response["updates"]))  # text updates, emphatically NOT 512 token timestamps
        self.assertEqual(512, response["usage"]["output_tokens"])
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual({"type": "disabled"}, body["thinking"])
        self.assertEqual(512, body["max_tokens"])

    def test_truncated_and_nonterminal_sse_variants_fail(self):
        prefix = [
            {"type": "message_start", "message": {"id": "msg-fixture", "usage": {"output_tokens": 512}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Actual answer"}}]
        final = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 512}}
        stop = {"type": "message_stop"}
        variants = {
            "EOF_after_text": prefix,
            "missing_message_stop": prefix + [final],
            "missing_final_delta": prefix + [stop],
            "missing_final_usage": prefix + [{"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}, stop],
            "missing_final_reason": prefix + [{"type": "message_delta", "usage": {"output_tokens": 512}}, stop],
            "early_stop": prefix + [{"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 512}}, stop]}
        for name, events in variants.items():
            with self.subTest(name=name):
                data = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
                with mock.patch.object(smoke, "open_local", return_value=io.BytesIO(data)), self.assertRaises(RuntimeError):
                    smoke.request_messages("http://127.0.0.1:1", "fake", [{"role": "user", "content": "hello"}])

    def test_count_and_inference_payloads_use_identical_rendering_inputs(self):
        messages = [{"role": "user", "content": "hello"}]
        with mock.patch.object(smoke, "open_local", return_value=io.BytesIO(b'{"input_tokens": 42}')) as opened:
            self.assertEqual(42, smoke.Client("http://localhost:8096").count("fake", messages))
        count_body = json.loads(opened.call_args.args[0].data)
        inference_body = smoke.payload("fake", messages, 1024)
        for key in ("model", "messages", "thinking", "temperature"):
            self.assertEqual(count_body[key], inference_body[key])

    def test_counted_prefix_never_uses_word_estimates(self):
        client = FakeClient()
        messages, n = smoke.counted_prefix(client, "fake", 77000, "marker")
        self.assertEqual(n, client.count("fake", messages))
        self.assertTrue(77000 <= n <= 77128)


if __name__ == "__main__":
    unittest.main()
