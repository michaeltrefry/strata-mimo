"""GPU-free tagged protocol, backpressure, lease and HTTP regressions."""
import contextlib
import json
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.mimo_transport import _Handle
from serve.server import (ByteTokenizer, EngineDied, MiMoEngine, Service, StrataEngine,
                          resolve_slots, slot_engine_args, serve, openai_chunks, anthropic_events)


FAKE = r'''
import sys
mode = sys.argv[1]
def out(s):
    print(s, flush=True)
def done(c, token=1, finish='length'):
    out(f'DONE {c} {token} 3 {c}.0 {c}0.0 {finish} 0 0 0 0 0 0 0 0 3')
out('INFO kv_bytes=123 slots=2')
out('READY 4096 stop' + ('' if mode == 'old' else ' multiplex=1 slots=2'))
active = []
for line in sys.stdin:
    p = line.split()
    if not p: continue
    if p[0] == 'QUIT': break
    if p[0] == 'STOP':
        c = int(p[1]); done(c, finish='cancel')
        if mode == 'cancel':
            for peer in active:
                if peer != c:
                    out(f'T {peer} 66'); done(peer)
        continue
    if p[0] != 'GEN': continue
    c, slot = int(p[1]), int(p[2])
    active.append(c)
    if mode.startswith('bad:'):
        if len(active) == 2: out(mode[4:].replace('@', str(active[0])))
    elif mode == 'eof':
        if len(active) == 2: break
    elif mode == 'pair':
        if len(active) == 2:
            a,b = active
            out(f'PP {a} 2 3 1.0 2.0')
            out(f'T {a} 65'); out(f'T {b} 66')
            done(b); done(a)
    elif mode == 'cancel':
        out(f'T {c} 65')
    elif mode == 'overflow':
        if slot == 0:
            out(f'T {c} 65')
        else:
            for i in range(2000): out(f'T {active[0]} 65')
            out(f'T {c} 66'); done(c)
    elif mode == 'stale':
        if c > 1:
            out(f'T {c-1} 99'); done(c-1)
        out(f'T {c} 65'); done(c)
    elif mode == 'wrap':
        out(f'T {c} {65 if len(active) == 1 else 66}')
        if len(active) > 1: done(c)
    elif mode == 'hold':
        out(f'T {c} 65')
    else:
        out(f'T {c} 65'); done(c)
'''


@contextlib.contextmanager
def fake_engine(mode='pair', lazy=False):
    original = subprocess.Popen
    def spawn(argv, **kwargs):
        return original([sys.executable, '-u', '-c', FAKE, mode], **kwargs)
    with mock.patch('serve.mimo_transport.subprocess.Popen', side_effect=spawn):
        engine = MiMoEngine('fake-mimo', ['--slots', '2'], lazy=lazy)
        engine.stop_timeout_s = 0.5
        try:
            yield engine
        finally:
            engine.close()


def service(engine):
    return Service(engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))


def wait_for(test, predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= deadline:
            test.fail('timed out waiting for deterministic engine state')
        threading.Event().wait(0.005)


class Transport(unittest.TestCase):
    def collect(self, engine, slot, results, errors, cancel=None):
        try:
            with engine.bind_slot(slot):
                results[slot] = list(engine.generate([1, 2, 3], 4, {}, cancel or threading.Event()))
                results[str(slot)] = engine.last
        except Exception as e:
            errors.append(e)

    def pair(self, engine):
        results, errors = {}, []
        workers = [threading.Thread(target=self.collect, args=(engine, i, results, errors)) for i in range(2)]
        for t in workers: t.start()
        for t in workers: t.join(5)
        self.assertFalse(any(t.is_alive() for t in workers))
        return results, errors

    def test_interleaved_reverse_done_and_request_timings(self):
        with fake_engine() as e:
            results, errors = self.pair(e)
            self.assertEqual(errors, [])
            self.assertEqual([t for t in results[0] if t is not None], [65])
            self.assertEqual([t for t in results[1] if t is not None], [66])
            self.assertNotEqual(results['0']['decode_ms'], results['1']['decode_ms'])
            self.assertIsNone(e.last)          # no shared completion object on monitoring threads
            self.assertEqual(e.info['kv_bytes'], 123)

    def test_targeted_cancel_does_not_stop_peer(self):
        with fake_engine('cancel') as e:
            results, errors = {}, []
            cancel = threading.Event()
            workers = [threading.Thread(target=self.collect, args=(e, i, results, errors, cancel if i == 0 else None))
                       for i in range(2)]
            for t in workers: t.start()
            wait_for(self, lambda: len(e._pending) == 2)
            cancel.set()
            for t in workers: t.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(results['0']['finish'], 'cancel')
            self.assertEqual(results['1']['finish'], 'length')
            self.assertIn(66, results[1])

    def test_slow_consumer_overflow_is_targeted_and_drained(self):
        with fake_engine('overflow') as e:
            e.token_queue_limit = 4
            with e.bind_slot(0):
                gen = e.generate([1], 3000, {}, threading.Event())
                self.assertEqual(next(gen), 65)
                results, errors = {}, []
                peer = threading.Thread(target=self.collect, args=(e, 1, results, errors))
                peer.start(); peer.join(5)
                self.assertEqual(errors, [])
                self.assertEqual(results[1], [66])
                with self.assertRaisesRegex(ValueError, 'overflow'):
                    list(gen)
                gen.close()
                self.assertTrue(e.alive())

    def test_blocked_stop_writer_cannot_block_stdout_dispatch(self):
        with fake_engine('overflow') as e:
            e.token_queue_limit = 4
            entered, release = threading.Event(), threading.Event()
            original = e.proc.stdin
            class BlockStop:
                def write(self, line):
                    if line.startswith('STOP '):
                        entered.set()
                        release.wait(5)
                    return original.write(line)
                def flush(self): original.flush()
                def close(self): original.close()
            e.proc.stdin = BlockStop()
            with e.bind_slot(0):
                gen = e.generate([1], 3000, {}, threading.Event())
                self.assertEqual(next(gen), 65)
                results, errors = {}, []
                peer = threading.Thread(target=self.collect, args=(e, 1, results, errors))
                try:
                    peer.start()
                    self.assertTrue(entered.wait(5))
                    peer.join(2)
                    self.assertFalse(peer.is_alive(), 'reader stalled behind STOP writer')
                    self.assertEqual(errors, [])
                    self.assertEqual(results[1], [66])
                finally:
                    release.set()
                    gen.close()
                    peer.join(5)

    def test_stale_ids_cannot_complete_new_occupant(self):
        with fake_engine('stale') as e:
            results, errors = {}, []
            for _ in range(3):
                self.collect(e, 0, results, errors)
                self.assertEqual(results[0], [65])
            self.assertEqual(errors, [])
            self.assertEqual(e._next_command, 4)

    def test_eof_fails_both_and_restart_has_new_identity(self):
        with fake_engine('eof') as e:
            results, errors = self.pair(e)
            self.assertEqual(len(errors), 2)
            self.assertTrue(all(isinstance(x, EngineDied) for x in errors))
            generation, proc = e._generation, e.proc
            e.restart()
            self.assertGreater(e._generation, generation)
            e._fail(proc, generation, 'stale EOF')
            self.assertTrue(e.alive())

    def test_malformed_protocol_fails_all_without_waiting_for_watchdog(self):
        bad_lines = ['DONE @ nope', 'DONE nope 1 2 3 4 stop', 'WAT @ x', 'T @ -1',
                     'PP @ 1 2 3 nan', 'RESUME @ nope', 'ERR @', 'T 999999 65', 'DONE']
        for line in bad_lines:
            with self.subTest(line=line), fake_engine('bad:' + line) as e:
                results, errors = self.pair(e)
                self.assertEqual(len(errors), 2)
                self.assertTrue(all(isinstance(x, EngineDied) and 'protocol failure' in str(x) for x in errors))
                self.assertFalse(e.alive())
                self.assertEqual(e._pending, {})

    def test_watchdog_uses_configured_quantum_and_bounded_peer_progress(self):
        for args in (['--prefill-quantum', '16384'], ['--prefill-quantum=16384']):
            e = MiMoEngine('missing', args, lazy=True)
            e.silence_s = 10
            a, b = _Handle(1, 0, 4), _Handle(2, 1, 4)
            a.heard = b.heard = 100.0
            e._process_heard = 100.0
            base = 10 + 2 * 16384 / 50
            self.assertAlmostEqual(e._request_deadline(a, [a, b]), 100 + base)
            # A valid peer event grants scheduling slack, but an infinite stream from that
            # peer cannot postpone this request beyond two complete expected rounds.
            e._process_heard = 100 + base - 1
            self.assertGreater(e._request_deadline(a, [a, b]), 100 + base)
            e._process_heard = 100000.0
            self.assertAlmostEqual(e._request_deadline(a, [a, b]), 100 + 2 * base)
            # Measured slow prefill also widens complete-chunk allowances, not only the CLI size.
            b.prefill_tok_s_mean = 5.0
            self.assertGreater(e._round_allowance([a, b]), 3 * 16384 / 5)
            self.assertGreaterEqual(max(e.stop_timeout_s, e._round_allowance([a, b])), e._round_allowance([a, b]))

    def test_unsupported_ready_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'rebuild'):
            with fake_engine('old'):
                pass

    def test_close_consumer_drains_without_future_iteration(self):
        with fake_engine('hold') as e:
            with e.bind_slot(0):
                gen = e.generate([1], 4, {}, threading.Event())
                self.assertEqual(next(gen), 65)
                gen.close()
                self.assertEqual(e.last['finish'], 'cancel')
                wait_for(self, lambda: not e._pending)

    def test_strict_capacity_and_legacy_default(self):
        self.assertEqual(resolve_slots({}), 1)
        for bad in (True, False, 0, 3, 2.0, '2', None):
            with self.assertRaises(ValueError): resolve_slots({'slots': bad})
        cfg = {'family': 'mimo', 'args': ['--slots', '2', '--slots=2']}
        self.assertEqual(slot_engine_args(cfg, 2), ['--slots', '2'])
        self.assertEqual(slot_engine_args({'args': []}, 1), [])
        with self.assertRaises(ValueError): slot_engine_args({'args': []}, 2)
        with self.assertRaises(ValueError): slot_engine_args({'family': 'mimo', 'args': ['--slots', '2']}, 1)
        e = StrataEngine('missing', [], lazy=True)
        self.assertEqual(service(e).capacity, 1)


class ServiceSlots(unittest.TestCase):
    def test_two_http_requests_active_and_own_timings(self):
        with fake_engine() as e:
            svc = service(e)
            httpd = serve(svc, port=0)
            base = f'http://127.0.0.1:{httpd.server_address[1]}'
            results, errors = [], []
            def request():
                try:
                    body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8,
                            'reasoning_effort': 'none'}
                    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                                 headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=5) as r: results.append(json.load(r))
                except Exception as x: errors.append(x)
            try:
                threads = [threading.Thread(target=request) for _ in range(2)]
                for t in threads: t.start()
                for t in threads: t.join(7)
                self.assertFalse(any(t.is_alive() for t in threads))
                self.assertEqual(errors, [])
                self.assertEqual(sorted(r['choices'][0]['message']['content'] for r in results), ['A', 'B'])
                self.assertNotEqual(results[0]['timings']['predicted_ms'], results[1]['timings']['predicted_ms'])
                with urllib.request.urlopen(base + '/props') as r: self.assertEqual(json.load(r)['total_slots'], 2)
                with urllib.request.urlopen(base + '/slots') as r: self.assertEqual(len(json.load(r)), 2)
                self.assertEqual(svc.v1_status()['concurrency']['serving'], 2)
                self.assertEqual(svc.totals['requests'], 2)
                self.assertEqual(svc.status['active'], 0)
            finally:
                httpd.shutdown(); httpd.server_close()

    def test_queued_cancel_never_submits_and_busy_unload(self):
        with fake_engine('hold') as e:
            svc = service(e)
            gens = [svc.run([1], False, None, 4, {}, threading.Event()) for _ in range(2)]
            # Generators must execute on distinct threads: request state is thread-local.
            started = [threading.Event(), threading.Event()]
            release = threading.Event()
            def hold(i):
                try:
                    next(gens[i]); started[i].set(); release.wait(5)
                finally: gens[i].close()
            workers = [threading.Thread(target=hold, args=(i,)) for i in range(2)]
            for w in workers: w.start()
            try:
                for event in started: self.assertTrue(event.wait(5))
                self.assertEqual(svc.unload(), 'busy')
                self.assertEqual(svc.v1_status()['activity']['in_flight'], 2)
                svc.max_queued_requests = 1
                cancel = threading.Event()
                result = []
                queued = threading.Thread(target=lambda: result.extend(svc.run([2], False, None, 4, {}, cancel)))
                queued.start()
                wait_for(self, lambda: svc.status['queued'] == 1)
                try:
                    with self.assertRaisesRegex(ValueError, 'queue is full'):
                        list(svc.run([3], False, None, 4, {}, threading.Event()))
                    self.assertEqual(svc.status['queued'], 1)
                    self.assertEqual(e._next_command, 3)
                    self.assertEqual(sum(s['is_processing'] for s in svc.slot_snapshot()), 2)
                finally:
                    cancel.set()
                queued.join(5)
                self.assertFalse(queued.is_alive())
                self.assertEqual(e._next_command, 3)
                self.assertEqual(result[-1][1]['finish'], 'cancel')
                self.assertEqual(sum(s['is_processing'] for s in svc.slot_snapshot()), 2)
            finally:
                release.set()
                for w in workers: w.join(5)
            self.assertEqual(svc.status['active'], 0)

    def test_idle_slot_prefix_affinity_is_a_hint_not_cross_slot_reuse(self):
        with fake_engine('single') as e:
            svc = service(e)
            with mock.patch.object(e, 'bind_slot', wraps=e.bind_slot) as bind:
                for ids in ([11, 12], [99, 98], [11, 12, 13]):
                    items = list(svc.run(ids, False, None, 4, {}, threading.Event()))
                    self.assertEqual(items[-1][0], 'done')
                self.assertEqual([call.args[0] for call in bind.call_args_list], [0, 1, 0])
            self.assertEqual(sorted(svc._free_slots), [0, 1])

    def test_concurrent_lazy_requests_share_one_load(self):
        with fake_engine('pair', lazy=True) as e:
            svc = service(e)
            results, errors = [], []
            def run():
                try:
                    results.append(list(svc.run([1], False, None, 4, {}, threading.Event())))
                except Exception as x:
                    errors.append(x)
            with mock.patch.object(e, 'restart', wraps=e.restart) as restart:
                workers = [threading.Thread(target=run) for _ in range(2)]
                for w in workers: w.start()
                for w in workers: w.join(5)
                self.assertFalse(any(w.is_alive() for w in workers))
                self.assertEqual(errors, [])
                self.assertEqual(restart.call_count, 1)
                self.assertEqual(len(results), 2)

    def test_reasoning_wrap_keeps_lease_and_uses_new_command(self):
        with fake_engine('wrap') as e:
            svc = service(e)
            ids = svc.tok.encode('<think>', parse_special=True)
            items = list(svc.run(ids, True, None, 300, {'reasoning_budget_tokens': 1}, threading.Event()))
            self.assertEqual(items[-1][0], 'done')
            self.assertEqual(e._next_command, 3)
            self.assertEqual(svc.totals['requests'], 1)
            self.assertEqual(svc.totals['decode_ms'], 30.0)
            self.assertEqual(svc.history[-1]['engine_generated'], 2)
            self.assertEqual(len(svc._free_slots), 2)
            self.assertEqual(e._pending, {})

    def test_overflow_releases_lease_even_while_consumer_is_suspended(self):
        with fake_engine('overflow') as e:
            e.token_queue_limit = 4
            svc = service(e)
            first_ready, finish_first = threading.Event(), threading.Event()
            errors = []
            def suspended():
                gen = svc.run([1], False, None, 3000, {}, threading.Event())
                try:
                    next(gen)
                    first_ready.set()
                    finish_first.wait(5)
                except Exception as x:
                    errors.append(x)
                finally:
                    gen.close()
            first = threading.Thread(target=suspended)
            first.start()
            try:
                self.assertTrue(first_ready.wait(5))
                peer_items = list(svc.run([2], False, None, 4, {}, threading.Event()))
                self.assertEqual(peer_items[-1][0], 'done')
                wait_for(self, lambda: len(svc._free_slots) == 2)
                self.assertTrue(first.is_alive())   # no further next()/close() drove the cleanup
                self.assertEqual(svc.status['active'], 0)
            finally:
                finish_first.set(); first.join(5)
            self.assertEqual(errors, [])
            self.assertEqual(sorted(svc._free_slots), [0, 1])

    def test_api_wrapper_close_reaches_service(self):
        for wrapper in ('openai', 'anthropic'):
            with self.subTest(wrapper=wrapper), fake_engine('hold') as e:
                svc = service(e)
                source = (openai_chunks(svc, {}, [1], False, None, 4, threading.Event()) if wrapper == 'openai'
                          else anthropic_events(svc, {}, [1], False, None, 4, threading.Event()))
                next(source)              # role/message_start precedes engine admission
                next(source)              # first model token
                source.close()
                self.assertEqual(svc.status['active'], 0)
                self.assertEqual(e._pending, {})


if __name__ == '__main__':
    unittest.main()
