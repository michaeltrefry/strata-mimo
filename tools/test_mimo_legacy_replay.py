#!/usr/bin/env python3
"""CPU regression of the actual legacy mimo_serve.cpp, not a protocol reimplementation.

Compile separately (from the worktree root), for example:
  c++ -std=c++17 -O2 -pthread -Itests/mimo/legacy_fake -Iinclude \
      src/program/mimo_serve.cpp -o /tmp/mimo-legacy-fake
  python tools/test_mimo_legacy_replay.py --engine /tmp/mimo-legacy-fake

MSVC: cl /nologo /std:c++17 /EHsc /O2 /Itests/mimo/legacy_fake /Iinclude \
          src/program/mimo_serve.cpp /Fe:mimo-legacy-fake.exe

No compiler automation, model, GPU, server, or production/CMake changes. The override
implements only the decoder surface consumed by this translation unit (including its
scheduler instantiation); the real serve.hpp and sampling.hpp are compiled unchanged.
The oracle below uses unbounded position-addressed history and always starts fresh.
It shares the toy hash equations, NOT the fake's ring indexing or prefix-reuse policy.
This tests replay correctness, not real model numerics, CUDA graphs, batched prefill,
stochastic sampling, or multiplexed scheduling. All requests use greedy sampling.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

VOCAB = 32749
MASK = (1 << 32) - 1
MAX_CONTEXT = 4096
NEW = 32


def mix(a, b):
    x = a ^ ((b + 0x9E3779B9 + ((a << 6) & MASK) + (a >> 2)) & MASK)
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK
    return x ^ (x >> 15)


def token(h):
    return 1 + h % (VOCAB - 1)


def embedding(t, p):
    return mix(t + 17, p)


class Reference:
    """Two causal layers with 128-wide attention; no wrapping physical storage."""
    def __init__(self):
        self.keys = [{}, {}]
        self.mtp = {}
        self.hidden = []

    @staticmethod
    def attend(keys, p, current):
        h = mix(0x12345678, p)
        for j in range(max(0, p - 127), p + 1):
            # Only the unwritten MTP p=0 sentinel is absent on a fresh sequence.
            h = mix(h, current if j == p else keys.get(j, 0))
        return h

    def preview(self, t, p):
        h0 = self.attend(self.keys[0], p, embedding(t, p))
        return self.attend(self.keys[1], p, mix(h0, 0xABCDEF01))

    def forward(self, ts, p):
        # Sequential causal evaluation is independent of the fake's layer-major
        # windows; valid fixed-eight windows have exactly these same results.
        self.hidden = []
        for i, t in enumerate(ts):
            at = p + i
            h = embedding(t, at)
            for layer in range(2):
                self.keys[layer][at] = h
                h = self.attend(self.keys[layer], at, h)
                if layer == 0:
                    h = mix(h, 0xABCDEF01)
            self.hidden.append(h)
        return [token(h) for h in self.hidden]

    def mtp_forward(self, ts, p):
        for i, t in enumerate(ts):
            self.mtp[p + i] = mix(embedding(t, p + i), self.hidden[i])
        drafts = []
        for i, t in enumerate(ts):
            at = p + i
            h = self.attend(self.mtp, at, self.mtp[at])
            drafts.append(token(self.preview(t, at) if h % 3 == 0 else h))
        return drafts

    def generate(self, prompt, mtp):
        first = draft = None
        for p in range(0, len(prompt), 8):
            ts = prompt[p:p + 8]
            first = self.forward(ts, p)[-1]
            if mtp:
                shifted = prompt[p + 1:p + len(ts) + 1]
                if len(shifted) < len(ts):
                    shifted = shifted + [first]
                draft = self.mtp_forward(shifted, p + 1)[-1]
        out, accepted, offered = [], 0, 0
        p = len(prompt)
        while len(out) < NEW:
            if mtp and len(out) + 1 < NEW:
                nxt = self.forward([first, draft], p)
                offered += 1
                out.append(first)
                if nxt[0] == draft:
                    accepted += 1
                    out.append(draft)
                    first = nxt[1]
                    shifted = [draft, first]
                else:
                    first = nxt[0]
                    shifted = [first]
                draft = self.mtp_forward(shifted, p + 1)[-1]
                p += len(shifted)
            else:
                nxt = self.forward([first], p)
                out.append(first)
                first = nxt[0]
                p += 1
        return out, accepted, offered


@dataclass
class Result:
    tokens: list
    accepted: int
    offered: int
    resume: int
    done: list


class Engine:
    def __init__(self, path, mtp, timeout):
        self.timeout = timeout
        self.stderr = tempfile.TemporaryFile(mode="w+t")
        args = [str(path), "--serve", "--native", "unused-fake.gguf",
                "--max-context", str(MAX_CONTEXT), "--eos-ids", "0"]
        if mtp:
            args += ["--mtp", "unused-fake-mtp.gguf"]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr, text=True, bufsize=1)
        self.lines = queue.Queue()

        def read():
            try:
                for line in self.proc.stdout:
                    self.lines.put(line.rstrip("\r\n"))
            finally:
                self.lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        try:
            infos = []
            while True:
                line = self.line()
                if line.startswith("INFO "):
                    infos.append(line)
                    continue
                assert line.split() == ["READY", str(MAX_CONTEXT), "stop"], line
                break
            assert any(f"mtp={int(mtp)}" in line.split() for line in infos), infos
        except BaseException:
            self.close()
            raise

    def line(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise AssertionError("fake engine timed out waiting for protocol output") from None
        if line is None:
            self.stderr.seek(0)
            raise AssertionError("fake engine exited early: " + self.stderr.read())
        assert not line.startswith("ERR"), line
        return line

    def generate(self, prompt):
        # Never pipeline GEN or close stdin before DONE: Input::eof is a stop
        # signal in some legacy revisions, and this is the untagged protocol.
        self.proc.stdin.write(f"GEN {NEW} temperature=0 " + ",".join(map(str, prompt)) + "\n")
        self.proc.stdin.flush()
        tokens, resumes = [], []
        while True:
            line = self.line()
            fields = line.split()
            assert fields, line
            if fields[0] == "RESUME":
                assert len(fields) == 2, "RESUME must be untagged: " + line
                resumes.append(int(fields[1]))
            elif fields[0] == "PP":
                assert len(fields) == 5, "PP must be untagged: " + line
                assert int(fields[2]) == len(prompt), line
            elif fields[0] == "T":
                assert len(fields) == 2, "T must be untagged: " + line
                tokens.append(int(fields[1]))
            elif fields[0] == "DONE":
                d = fields[1:]
                assert len(d) == 14, "DONE must have 14 untagged fields: " + line
                assert len(resumes) == 1, resumes
                assert int(d[0]) == len(tokens) == NEW and int(d[1]) == len(prompt), line
                assert d[4] == "length", line
                assert float(d[2]) >= 0 and float(d[3]) >= 0, line
                return Result(tokens, int(d[5]), int(d[6]), resumes[0], d)
            else:
                raise AssertionError("unexpected legacy event: " + line)

    def close(self):
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=self.timeout)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self.proc.kill()
                    self.proc.wait()
        finally:
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.stderr.close()


def prompt_tokens(n, salt=0):
    return [token(mix(i + 31, salt + 913)) for i in range(n)]


def check(engine, prompt, mtp, name, expected_start):
    got = engine.generate(prompt)
    want_tokens, want_accepted, want_offered = Reference().generate(prompt, mtp)
    # Numeric checks deliberately precede RESUME assertions. An old keep-128
    # branch must fail on outputs/acceptance, not merely its claimed replay start.
    assert got.tokens == want_tokens, (
        f"{name}: greedy tokens differ from fresh reference\n"
        f"got {got.tokens}\nwant {want_tokens} (RESUME {got.resume})")
    assert (got.accepted, got.offered) == (want_accepted, want_offered), (
        f"{name}: draft acceptance differs from fresh reference: "
        f"got {got.accepted}/{got.offered}, want {want_accepted}/{want_offered} "
        f"(RESUME {got.resume})")
    assert 0 <= got.accepted <= got.offered, got
    assert (got.offered > 0) == mtp, got
    if expected_start == "reuse":
        assert got.resume > 0, f"{name}: valid non-MTP reuse was disabled"
    else:
        assert got.resume == expected_start, f"{name}: RESUME {got.resume}, want {expected_start}"
    assert int(got.done[7]) == got.resume, got.done
    assert int(got.done[13]) == len(prompt) - got.resume, got.done
    assert all(int(got.done[i]) >= 0 for i in (8, 9, 10, 11)), got.done
    assert float(got.done[12]) >= 0, got.done
    return prompt + got.tokens, got.accepted


def run_mode(path, mtp, timeout):
    mode = "MTP" if mtp else "noMTP"
    total_accepted = 0
    # Each scenario has its own live long session, so the far/near/identical/
    # append labels refer to known histories, not accidents of earlier cases.
    scenarios = ["far rewind", "near rewind", "identical shorter", "append"]
    if not mtp:
        scenarios += ["last live start", "first expired start"]
    for scenario in scenarios:
        engine = Engine(path, mtp, timeout)
        try:
            seed = prompt_tokens(1400)
            session, accepted = check(engine, seed, mtp, mode + " seed", 0)
            total_accepted += accepted
            if scenario == "far rewind":
                # common=400 >128, live=1432 >1024; old start=400-128
                # cannot rebuild layer-two keys before the 417-token prompt ends.
                prompt = seed[:400] + prompt_tokens(17, 47)
                start = 0
            elif scenario == "near rewind":
                prompt = session[:1390] + prompt_tokens(8, 77)
                start = 0 if mtp else "reuse"
            elif scenario == "identical shorter":
                prompt = seed[:1392]
                start = 0 if mtp else "reuse"
            elif scenario == "append":
                prompt = session + prompt_tokens(11, 99)
                start = 0 if mtp else "reuse"
            else:
                keep = len(session) - (129 if scenario == "last live start" else 130)
                prompt = session[:keep] + prompt_tokens(8, 181)
                start = keep if scenario == "last live start" else 0
            _, accepted = check(engine, prompt, mtp, mode + " " + scenario, start)
            total_accepted += accepted
        finally:
            engine.close()
    if not mtp:
        # Reusing a shorter prompt shrinks session, but cannot resurrect ring keys
        # overwritten by the original long request. A second rewind must use the
        # write high-water mark, not the shortened session length, as its bound.
        engine = Engine(path, False, timeout)
        try:
            seed = prompt_tokens(1400)
            check(engine, seed, False, "shrink-chain seed", 0)  # written end 1432
            shorter = seed[:1310] + prompt_tokens(7, 311)
            session, _ = check(engine, shorter, False, "first safe shortening", 1310)
            keep = len(session) - 129  # 1220: looks live using session, not written end
            check(engine, session[:keep] + prompt_tokens(17, 419), False,
                  "second rewind after shortening", 0)
        finally:
            engine.close()
    if mtp:
        # A wrapped MTP row 0 is read as the position-zero sentinel on full replay.
        # Short prompts retain its influence in the first draft gate. Re-seed before
        # each one so removing reset_sequence cannot hide behind a previous replay.
        for salt in range(12):
            engine = Engine(path, True, timeout)
            try:
                check(engine, prompt_tokens(1100, salt), True, "sentinel seed", 0)
                _, accepted = check(engine, prompt_tokens(9 + salt, 200 + salt), True,
                                    f"MTP wrapped sentinel {salt}", 0)
                total_accepted += accepted
            finally:
                engine.close()
        assert total_accepted > 0, "fixture never exercised an accepted draft"
    print(f"PASS {mode}: far/near rewind, identical shorter, append" +
          (", wrapped MTP sentinel; accepted drafts=" + str(total_accepted) if mtp
           else ", ring boundaries, chained shortening"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, type=Path, help="already compiled fake mimo_serve executable")
    parser.add_argument("--timeout", type=float, default=30, help="seconds per protocol line/process exit")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mtp-only", action="store_true")
    modes.add_argument("--no-mtp-only", action="store_true")
    args = parser.parse_args()
    path = args.engine.resolve(strict=True)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.mtp_only:
        run_mode(path, False, args.timeout)
    if not args.no_mtp_only:
        run_mode(path, True, args.timeout)


if __name__ == "__main__":
    main()
