"""Tagged, text-only MiMo transport. The stdout dispatcher never waits for a client or stdin.

A slot is cache identity; a monotonically increasing command ID is request identity. Cancelled
handles remain registered until native acknowledgement (or process death), including after close.
"""
from __future__ import annotations

import collections
import contextlib
import math
import queue
import subprocess
import threading
import time


# These names are request-local, never process-wide INFO facts. Missing diagnostics from
# older tagged engines stay unknown, including across reasoning-wrap commands.
PREFIX_REASONS = frozenset(('empty_slot', 'no_common_prefix', 'boundary_replay_from_zero',
                            'ring_history_missing', 'retained_live', 'restored_checkpoint'))
CHECKPOINT_REASONS = frozenset(('not_needed', 'no_checkpoint', 'prefix_mismatch',
                                'global_prefix_overwritten', 'slot_reset', 'ring_coverage',
                                'budget', 'host_allocation', 'disabled', 'restored'))
CHECKPOINT_SKIP_REASONS = CHECKPOINT_REASONS | frozenset(('none',))
PREFIX_FIELDS = ('slot', 'raw_lcp', 'boundary_keep', 'written_end', 'resume', 'prefix_reason',
                 'checkpoint_boundary', 'checkpoint_id', 'checkpoint_reason')
SERVICE_SUM_FIELDS = ('prompt_service_ms', 'decode_service_ms', 'prompt_expert_hits',
                      'prompt_expert_uses', 'decode_expert_hits', 'decode_expert_uses',
                      'checkpoint_capture_ms', 'checkpoint_restore_ms', 'checkpoint_captures',
                      'checkpoint_restores', 'fresh_prompt_tokens')
INVENTORY_FIELDS = ('checkpoint_bytes', 'checkpoint_metadata_bytes', 'checkpoint_count',
                    'checkpoint_boundaries')
DIAGNOSTIC_FIELDS = (*PREFIX_FIELDS, 'command_id', 'native_prepared', 'live_written_end', 'checkpoint_skip_reason',
                     *SERVICE_SUM_FIELDS, *INVENTORY_FIELDS)
RATE_FIELDS = ('prompt_service_tok_per_sec', 'decode_service_tok_per_sec',
               'prompt_hit_rate', 'decode_hit_rate')


def service_metrics(last):
    """Active native service rates, not elapsed request/API throughput."""
    for phase, count in (('prompt', last.get('fresh_prompt_tokens')), ('decode', last.get('generated'))):
        ms = last.get(phase + '_service_ms')
        uses, hits = last.get(phase + '_expert_uses'), last.get(phase + '_expert_hits')
        last[phase + '_service_tok_per_sec'] = count * 1000 / ms if count is not None and ms else None
        last[phase + '_hit_rate'] = hits / uses if hits is not None and uses else None
    last['hit_rate_scope'] = 'combined_prompt_decode'
    return last


class _Handle:
    def __init__(self, command, slot, limit):
        self.command, self.slot, self.limit = command, slot, limit
        self.cv = threading.Condition()
        self.tokens = collections.deque()
        self.progress = self.prefill_tok_s_mean = None
        self.last = {}
        self.diag = dict.fromkeys(DIAGNOSTIC_FIELDS)
        self.input_ids = ()
        self.native_tokens = collections.deque()
        self.native_count = 0
        self.terminal = False
        self.error = None
        self.cancelled = False
        self.stop_at = None
        self.heard = time.monotonic()
        self.progress_version = 0


class MultiplexEngine:
    """Mixin using the legacy engine's sampling/diagnostic helpers, not its pipe reader."""
    capacity = 2
    token_queue_limit = 256
    stop_timeout_s = 300.0   # a complete bounded prefill chunk can still take seconds on a slower PC
    shutdown_timeout_s = 20.0
    failure_type = RuntimeError
    stuck_type = RuntimeError
    silent_type = RuntimeError

    def __init__(self, exe, args, cwd=None, log=None, env=None, lazy=False):
        self.spawn = (exe, list(args), cwd, log, env)
        self.log_path = log
        self.proc = self.log = self.pump = None
        self.ended = self.unloaded = True
        self.max_context = int(args[args.index('--max-context') + 1]) if '--max-context' in args else 4096
        self.can_stop = True
        self.info = {'slots': 2}
        self.silent_note = None
        self._local = threading.local()
        self._registry_lock = threading.RLock()
        self._pending = {}
        self._next_command = 1
        self._generation = 0
        self._slot_hints = {}
        self.prefill_max_wait_ms = 250
        self.decode_service_percent = 80
        self.prefill_quantum = 2048
        for i, arg in enumerate(args):
            if arg == '--prefill-quantum' and i + 1 < len(args):
                self.prefill_quantum = int(args[i + 1])
            elif arg.startswith('--prefill-quantum='):
                self.prefill_quantum = int(arg.split('=', 1)[1])
        for flag, attr, low, high in (('--prefill-max-wait-ms', 'prefill_max_wait_ms', 1, 60000),
                                      ('--decode-service-percent', 'decode_service_percent', 1, 99)):
            for i, arg in enumerate(args):
                if arg == flag or arg.startswith(flag + '='):
                    value = arg.split('=', 1)[1] if '=' in arg else args[i + 1] if i + 1 < len(args) else ''
                    try:
                        value = int(value)
                    except ValueError:
                        raise ValueError(f'{flag} needs an integer in {low}..{high}') from None
                    if not low <= value <= high:
                        raise ValueError(f'{flag} needs an integer in {low}..{high}')
                    setattr(self, attr, value)
        if self.prefill_quantum <= 0:
            raise ValueError('MiMo prefill quantum must be positive')
        self._process_heard = time.monotonic()
        if not lazy:
            self.restart()

    @property
    def last(self):
        return getattr(self._local, 'last', None)

    @property
    def progress(self):
        h = getattr(self._local, 'handle', None)
        return h.progress if h else None

    @property
    def prefill_tok_s_mean(self):
        h = getattr(self._local, 'handle', None)
        return h.prefill_tok_s_mean if h else None

    @contextlib.contextmanager
    def bind_slot(self, slot, on_failure=None):
        self._local.slot, self._local.last = slot, None
        self._local.on_failure = on_failure
        try:
            yield
        finally:
            self._local.handle = None
            self._local.slot = None
            self._local.on_failure = None

    def _info(self, line):
        for field in line.split()[1:]:
            k, sep, v = field.partition('=')
            if sep:
                self.info[k] = int(v) if v.lstrip('-').isdigit() else v

    def _request_info(self, line):
        fields = line.split()[1:]
        if not any(f.partition('=')[0] == 'request_diag_v' for f in fields):
            return False
        values = {}
        for field in fields:
            key, sep, value = field.partition('=')
            if not sep or not value or key in values:
                raise ValueError('malformed reserved request diagnostic')
            values[key] = value
        if values.get('request_diag_v') != '1':
            raise ValueError('unsupported request diagnostic version')
        command = values.get('command_id', '')
        if not command.isdigit() or int(command) <= 0:
            raise ValueError('diagnostic command ID must be positive')
        command = int(command)
        with self._registry_lock:
            if command >= self._next_command:
                raise ValueError('diagnostic refers to a command ID never submitted')
            h = self._pending.get(command)
        if h is None:
            return True                         # issued stale IDs cannot mutate a successor
        kind = values.get('diag_kind')
        allowed = {'prefix': PREFIX_FIELDS, 'timing': ('native_prepared', 'live_written_end', *SERVICE_SUM_FIELDS, *INVENTORY_FIELDS),
                   'cache': INVENTORY_FIELDS}.get(kind)
        if allowed is None:
            raise ValueError('unknown request diagnostic kind')
        slot = values.get('slot', '')
        if slot not in ('0', '1') or int(slot) != h.slot:
            raise ValueError('diagnostic slot does not match command')
        parsed = {'slot': int(slot), 'command_id': command}
        for key, value in values.items():
            if key in ('request_diag_v', 'command_id', 'diag_kind', 'slot'):
                continue
            if key not in allowed and key != 'checkpoint_skip_reason':
                raise ValueError('unknown request diagnostic field: ' + key)
            if key in ('prefix_reason', 'checkpoint_reason', 'checkpoint_skip_reason'):
                enums = (PREFIX_REASONS if key == 'prefix_reason' else
                         CHECKPOINT_SKIP_REASONS if key == 'checkpoint_skip_reason' else CHECKPOINT_REASONS)
                if value not in enums:
                    raise ValueError('unknown ' + key)
                parsed[key] = value
            elif key == 'checkpoint_boundaries':
                parts = [] if value == 'none' else value.split(',')
                if len(parts) > 8 or not all(p.isdigit() and 0 < int(p) <= self.max_context for p in parts):
                    raise ValueError('invalid checkpoint boundaries')
                boundaries = tuple(map(int, parts))
                if list(boundaries) != sorted(set(boundaries)):
                    raise ValueError('checkpoint boundaries must be sorted and unique')
                parsed[key] = boundaries
            elif key.endswith('_ms'):
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError('diagnostic metrics must be finite and nonnegative')
                parsed[key] = number
            else:
                if not value.isdigit():
                    raise ValueError('diagnostic count must be a nonnegative integer')
                parsed[key] = int(value)
                if key == 'native_prepared':
                    if parsed[key] not in (0, 1):
                        raise ValueError('native_prepared must be 0 or 1')
                    parsed[key] = bool(parsed[key])
                if key == 'checkpoint_count' and parsed[key] > 8:
                    raise ValueError('checkpoint count exceeds bound')
        for key in ('raw_lcp', 'boundary_keep', 'resume', 'checkpoint_boundary'):
            if key in parsed and parsed[key] > len(h.input_ids):
                raise ValueError('diagnostic prompt position exceeds input')
        if any(parsed.get(key, 0) > self.max_context for key in ('written_end', 'live_written_end')):
            raise ValueError('diagnostic write watermark exceeds context')
        if parsed.get('prefix_reason') == 'restored_checkpoint':
            boundary, resume, checkpoint = (parsed.get(k) for k in ('checkpoint_boundary', 'resume', 'checkpoint_id'))
            if boundary is not None and resume is not None and resume != boundary - 1:
                raise ValueError('restored checkpoint resume must replay its boundary token')
            if checkpoint is not None and checkpoint <= 0:
                raise ValueError('restored checkpoint needs a positive checkpoint ID')
        with h.cv:
            h.diag.update(parsed)
            for phase in ('prompt', 'decode'):
                hits, uses = h.diag[phase + '_expert_hits'], h.diag[phase + '_expert_uses']
                if hits is not None and uses is not None and hits > uses:
                    raise ValueError('expert hits exceed uses')
            h.heard = self._process_heard = time.monotonic()
            h.cv.notify_all()
        return True

    def slot_hints(self):
        """Immutable, terminal-confirmed estimates; native exact validation is authoritative."""
        with self._registry_lock:
            return {slot: dict(hint) for slot, hint in self._slot_hints.items()
                    if hint['generation'] == self._generation}

    def _confirm_hint(self, h):
        if h.error is not None or h.last.get('finish') == 'error':
            self._slot_hints.pop(h.slot, None)
            return
        if h.diag['native_prepared'] == 0:
            return                              # cancelled before any native state was touched
        if h.native_count != h.last.get('generated'):
            self._slot_hints.pop(h.slot, None)
            return                              # uncertain journal, never guess accepted drafts
        read = h.last.get('prompt_read')
        if read is None:
            self._slot_hints.pop(h.slot, None)
            return                              # old DONE groups cannot confirm a partial prompt
        end = min(len(h.input_ids), (h.last.get('reused') or 0) + read)
        tokens = (h.input_ids[:end] + tuple(h.native_tokens))[:self.max_context]
        self._slot_hints[h.slot] = {'generation': self._generation, 'command_id': h.command,
                                   'tokens': tokens,
                                   'checkpoint_boundaries': h.diag['checkpoint_boundaries'],
                                   'live_written_end': h.diag['live_written_end']}

    def restart(self):
        self.close()
        self.silent_note = None
        exe, args, cwd, log, env = self.spawn
        self.ended, self.unloaded = True, False
        self._generation += 1
        generation = self._generation
        self._writes = None
        self.log = open(log, 'a', encoding='utf-8') if log else subprocess.DEVNULL
        try:
            proc = subprocess.Popen([exe, '--serve', *args], cwd=cwd, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=self.log, text=True,
                                    encoding='utf-8', bufsize=1, env=env)
            self.proc = proc
            from serve.winjob import contain
            contain(proc)
            for line in proc.stdout:
                if line.startswith('INFO '):
                    if not self._request_info(line):
                        self._info(line)
                elif line.startswith('READY '):
                    f = line.split()
                    if not {'stop', 'multiplex=1', 'slots=2'}.issubset(f[2:]):
                        raise RuntimeError('slots=2 requires a MiMo engine advertising multiplex=1 slots=2 stop; rebuild the engine')
                    self.max_context = int(f[1])
                    if self.max_context <= 0:
                        raise RuntimeError('invalid MiMo per-slot context')
                    break
            else:
                raise RuntimeError('two-slot MiMo engine exited before READY; rebuild with --slots support')
            self.info['slots'] = 2
            if self.info.get('engine'):
                self.info['version'] = str(self.info['engine'])
            self.ended = False
            self._process_heard = time.monotonic()
            writes = queue.Queue()
            self._writes = writes
            self.pump = threading.Thread(target=self._reader, args=(proc, generation), daemon=True)
            self.pump.start()
            self._writer_thread = threading.Thread(target=self._writer, args=(proc, generation, writes), daemon=True)
            self._writer_thread.start()
            threading.Thread(target=self._watch, args=(proc, generation), daemon=True).start()
        except BaseException:
            self.close()
            raise

    def _fail(self, proc, generation, message):
        with self._registry_lock:
            if self.proc is not proc or generation != self._generation:
                return
            self.ended = True
            handles = list(self._pending.values())
            self._pending.clear()
            self._slot_hints.clear()
        for h in handles:
            with h.cv:
                h.error = self.failure_type(message)
                h.terminal = True
                h.cv.notify_all()
            self._notify_failure(h)

    @staticmethod
    def _notify_failure(h):
        callback = getattr(h, 'on_failure', None)
        if callback and h.error is not None:
            h.on_failure = None
            # Lease cleanup must outlive a consumer suspended in a socket write; keep it off
            # the sole stdout reader as well as off the possibly blocked stdin writer.
            threading.Thread(target=callback, daemon=True).start()

    def _writer(self, proc, generation, writes):
        try:
            while True:
                line = writes.get()
                if line is None:
                    proc.stdin.close()        # detached native input readers must see EOF on Windows
                    return
                proc.stdin.write(line + '\n')
                proc.stdin.flush()
        except (OSError, ValueError):
            self._fail(proc, generation, 'MiMo engine command pipe closed')

    @staticmethod
    def _validate_payload(kind, payload):
        fields = payload.split()
        if kind in ('T', 'RESUME'):
            if len(fields) != 1 or not fields[0].isdigit():
                raise ValueError(f'{kind} needs one nonnegative integer')
        elif kind == 'PP':
            if len(fields) not in (2, 3, 4) or not all(x.isdigit() for x in fields[:2]):
                raise ValueError('PP needs prompt position, total and optional timing/rate')
            if int(fields[0]) > int(fields[1]):
                raise ValueError('PP position exceeds total')
            if any(not math.isfinite(float(x)) or float(x) < 0 for x in fields[2:]):
                raise ValueError('PP timing/rate must be finite and nonnegative')
        elif kind == 'DONE':
            if len(fields) < 5 or len(fields) in (6, 7, 9, 11, 12):
                raise ValueError('truncated DONE payload')
            if not all(x.isdigit() for x in fields[:2]) or fields[4] not in ('stop', 'length', 'cancel', 'error'):
                raise ValueError('invalid DONE counts/finish')
            for i, value in enumerate(fields):
                if i == 4:
                    continue
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise ValueError('DONE metrics must be finite and nonnegative')
        elif kind == 'ERR' and not payload.strip():
            raise ValueError('ERR needs an explanation')

    def _abort_protocol(self, proc, generation, message):
        with self._registry_lock:
            if self.proc is not proc or generation != self._generation:
                return
            self.silent_note = 'The engine was stopped for an invalid protocol record: ' + message
            try:
                proc.kill()
            except OSError:
                pass
            self._fail(proc, generation, 'MiMo protocol failure (process-wide): ' + message)

    def _reader(self, proc, generation):
        try:
            for line in proc.stdout:
                if line.startswith('INFO '):
                    if self.proc is proc and generation == self._generation:
                        if not self._request_info(line):
                            self._info(line)
                    continue
                if not line.strip():
                    continue
                fields = line.strip().split(maxsplit=2)
                if len(fields) < 2 or fields[0] not in ('RESUME', 'PP', 'T', 'DONE', 'ERR'):
                    raise ValueError('unknown or untagged event: ' + line.strip()[:160])
                if not fields[1].isdigit() or int(fields[1]) <= 0:
                    raise ValueError('event command ID must be a positive integer')
                kind, command = fields[0], int(fields[1])
                payload = fields[2] if len(fields) > 2 else ''
                with self._registry_lock:
                    if self.proc is not proc or generation != self._generation:
                        return
                    if command >= self._next_command:
                        raise ValueError('event refers to a command ID never submitted')
                    h = self._pending.get(command)
                if h is None:
                    continue                       # already-issued stale/duplicate ID; never attach to a slot
                self._validate_payload(kind, payload)
                overflow = False
                with h.cv:
                    h.heard = self._process_heard = time.monotonic()
                    if kind == 'T':
                        h.native_count += 1
                        if len(h.native_tokens) < self.max_context:
                            h.native_tokens.append(int(payload))
                        if not h.cancelled:
                            if len(h.tokens) >= h.limit:
                                h.error = ValueError('MiMo token queue overflow: this request was cancelled (slow consumer)')
                                overflow = True
                            else:
                                h.tokens.append(int(payload))
                    elif kind == 'PP':
                        p = payload.split()
                        h.progress = (int(p[0]), int(p[1]))
                        h.prefill_tok_s_mean = float(p[3]) if len(p) >= 4 else None
                        h.progress_version += 1
                    elif kind == 'DONE':
                        result = type('DoneResult', (), {})()
                        self._parse_done_into(result, 'DONE ' + payload)
                        h.last = service_metrics(dict(result.last, **h.diag))
                        with self._registry_lock:
                            self._confirm_hint(h)
                        h.terminal = True
                    elif kind == 'ERR':
                        h.error = ValueError(payload)
                        with self._registry_lock:
                            self._slot_hints.pop(h.slot, None)
                        h.terminal = True
                    h.cv.notify_all()
                if overflow:
                    self._cancel(h)
                if h.terminal:
                    with self._registry_lock:
                        self._pending.pop(command, None)
                    self._notify_failure(h)
        except (ValueError, IndexError, OverflowError) as e:
            self._abort_protocol(proc, generation, str(e))
        except OSError:
            pass
        finally:
            self._fail(proc, generation, 'MiMo engine stdout closed (process-wide failure)')

    def _cancel(self, h):
        with self._registry_lock:
            handles = list(self._pending.values())
        with h.cv:
            if h.terminal or h.cancelled:
                return
            h.cancelled = True
            h.tokens.clear()
            h.stop_grace = max(self.stop_timeout_s, self._round_allowance(handles))
            h.stop_at = time.monotonic()
        self._writes.put(f'STOP {h.command}')   # unbounded control queue; never write a pipe on the reader

    def _round_allowance(self, handles):
        """Two cooperative slots can each own a complete prefill quantum before control returns."""
        quantum_s = self.prefill_quantum / 50.0
        allowances = []
        for h in handles:
            rate = h.prefill_tok_s_mean
            allowances.append(max(quantum_s, 3.0 * self.prefill_quantum / rate) if rate else quantum_s)
        # Retain an allowance for a peer that just completed and left the registry.
        return (sum(sorted(allowances, reverse=True)[:self.capacity]) +
                max(0, self.capacity - len(allowances)) * quantum_s +
                self.prefill_max_wait_ms / 1000.0)

    def _request_deadline(self, h, handles):
        budget = float(self.silence_s or 0) + self._round_allowance(handles)
        # Healthy peer output may mean this handle is waiting for its turn. Grant at most
        # one extra round since OUR last event; a forever-chatty peer cannot hide starvation.
        return min(h.heard + 2 * budget, max(h.heard + budget, self._process_heard + budget))

    def _watch(self, proc, generation):
        while self.proc is proc and generation == self._generation and not self.ended:
            with self._registry_lock:
                handles = list(self._pending.values())
            now = time.monotonic()
            for h in handles:
                if h.stop_at is not None:
                    h.stop_grace = max(h.stop_grace, self._round_allowance(handles))
                if h.stop_at is not None and now - h.stop_at > h.stop_grace:
                    self.silent_note = 'MiMo STOP was not acknowledged; process-wide escalation (all slots failed)'
                    self._fail(proc, generation, self.silent_note)
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return
                silence = float(self.silence_s or 0)
                if silence and not h.cancelled and now > self._request_deadline(h, handles):
                    with h.cv:
                        h.error = self.silent_type('MiMo request timed out; targeted STOP sent')
                    self._cancel(h)
            time.sleep(0.05)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if embeddings:
            raise ValueError('two-slot MiMo is text-only (GENI/images are unsupported)')
        slot = getattr(self._local, 'slot', None)
        if slot not in (0, 1):
            raise RuntimeError('MiMo generation requires a slot lease')
        with self._registry_lock:
            if not self.alive():
                raise self.failure_type('MiMo engine is not running')
            if any(h.slot == slot and not h.terminal for h in self._pending.values()):
                raise RuntimeError('MiMo slot still draining')
            command = self._next_command
            self._next_command += 1
            h = _Handle(command, slot, self.token_queue_limit)
            h.input_ids = tuple(int(t) for t in ids)[:self.max_context]
            h.diag.update(command_id=command, slot=slot)
            h.on_failure = getattr(self._local, 'on_failure', None)
            self._pending[command] = h
            self._local.handle = h
            previous = self._local.last
            self._writes.put(f'GEN {command} {slot} {int(max_new)}{self.sampling_keys(sampling or {})} ' +
                             ','.join(str(int(t)) for t in ids))
        version = 0
        heartbeat = time.monotonic()
        try:
            while True:
                if cancel.is_set():
                    self._cancel(h)
                    return
                with h.cv:
                    if h.error is not None:
                        raise h.error
                    if h.tokens:
                        token = h.tokens.popleft()
                    elif h.terminal:
                        return
                    else:
                        h.cv.wait(0.1)
                        if h.tokens or h.terminal or h.error:
                            continue
                        token = None
                        if version == h.progress_version and time.monotonic() - heartbeat < 10:
                            # Short waits also notice disconnects/cancel without blocking either reader.
                            continue
                        version = h.progress_version
                heartbeat = time.monotonic()
                yield token
        finally:
            self._cancel(h)
            # Transport-owned watchdog/drain works even when no generator is consuming. A reasoning
            # wrap cannot reuse this cache until native has acknowledged the previous command.
            with h.cv:
                while not h.terminal:
                    h.cv.wait(0.1)
                if h.last:
                    last = dict(h.last)
                    if previous:
                        for key in ('generated', 'prompt_ms', 'decode_ms', 'drafts_accepted', 'drafts_offered',
                                    'hits', 'lookups', 'ram_blobs', 'file_blobs', 'file_mb', 'prompt_read'):
                            if key in previous or key in last:
                                last[key] = previous.get(key, 0) + last.get(key, 0)
                        for key in SERVICE_SUM_FIELDS:
                            a, b = previous.get(key), last.get(key)
                            last[key] = a + b if a is not None and b is not None else None
                        decisions = (*PREFIX_FIELDS, 'command_id', 'native_prepared')
                        first = previous.get('prefix_decisions', ({k: previous.get(k) for k in decisions},))
                        last['prefix_decisions'] = (*first, {k: last.get(k) for k in decisions})
                        prepared = previous.get('native_prepared'), last.get('native_prepared')
                        last['native_prepared'] = 1 if 1 in prepared else None if None in prepared else 0
                        # The top-level decision belongs to the original prompt. Later wrap
                        # decisions remain separate rather than claiming one prefix explained all work.
                        for key in PREFIX_FIELDS:
                            last[key] = previous.get(key)
                        last['command_ids'] = (*previous.get('command_ids', (previous.get('command_id'),)),
                                               last.get('command_id'))
                        last['command_id'] = previous.get('command_id')
                        service_metrics(last)
                        # Cache accounting is for the client's original prompt, not our injected wrap-up.
                        last['reused'] = previous.get('reused', 0)
                        last['prompt_tokens'] = previous.get('prompt_tokens', last.get('prompt_tokens'))
                    self._local.last = last
                else:
                    self._local.last = previous

    def slot_progress(self):
        with self._registry_lock:
            handles = list(self._pending.values())
        result = {}
        for h in handles:
            with h.cv:
                result[h.slot] = {"command_id": h.command, "progress": h.progress,
                                  "prefill_tok_s_mean": h.prefill_tok_s_mean,
                                  "draining": h.cancelled and not h.terminal,
                                  **service_metrics(dict(h.diag, generated=h.native_count))}
        return result

    def wait_slot(self, slot):
        while True:
            with self._registry_lock:
                handles = [h for h in self._pending.values() if h.slot == slot]
            if not handles:
                return
            for h in handles:
                self._cancel(h)
                with h.cv:
                    if not h.terminal:
                        h.cv.wait(0.1)

    def close(self):
        proc = self.proc
        if proc is None:
            return
        generation = self._generation
        self._fail(proc, generation, 'MiMo engine closed (process-wide failure)')
        writes = getattr(self, '_writes', None)
        if writes:
            writes.put('QUIT')
            writes.put(None)
        try:
            if writes is None:             # startup failed before capability negotiation
                proc.kill()
            proc.wait(timeout=self.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=self.shutdown_timeout_s)
                except subprocess.TimeoutExpired:
                    raise self.stuck_type('MiMo engine is still releasing memory; not starting a replacement') from None
        # Do not close a TextIO pipe while another thread is blocked inside it. Process death
        # unblocks both workers; join them before closing their handles.
        if self.pump and self.pump is not threading.current_thread():
            self.pump.join(timeout=2)
        writer = getattr(self, '_writer_thread', None)
        if writer and writer is not threading.current_thread():
            writer.join(timeout=2)
        for pipe in (proc.stdout, proc.stdin):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass
        if hasattr(self.log, 'close'):
            self.log.close()
        self.proc = None
