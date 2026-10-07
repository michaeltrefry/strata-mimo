"""Talk to `strata-mimo --serve` over its stdin/stdout protocol, as serve/server.py does: two requests, the second
continuing the first conversation (prefix reuse) with sampling.  Prints the protocol lines' summary and timings.

usage: python serve_smoke.py <strata-mimo.exe> <model.gguf> <mtp.gguf> <profile> <prompt.i32>
"""
import subprocess
import sys
import time

import numpy as np


def main() -> int:
    exe, model, mtp, prof, prompt = sys.argv[1:6]
    p = subprocess.Popen([exe, "--serve", "--native", model, "--mtp", mtp, "--expert-profile", prof,
                          "--max-context", "65536"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open("serve_smoke.log", "w"), text=True, encoding="utf-8", bufsize=1)
    t = time.time()
    for line in p.stdout:
        line = line.strip()
        if line.startswith("INFO") or line.startswith("READY"):
            print(line)
        if line.startswith("READY"):
            break
    print(f"ready after {time.time() - t:.1f} s")
    ids = np.fromfile(prompt, np.int32).tolist()

    def gen(ids, max_new, keys=""):
        p.stdin.write(f"GEN {max_new}{keys} {','.join(map(str, ids))}\n")
        p.stdin.flush()
        out, t0 = [], time.time()
        for line in p.stdout:
            f = line.split()
            if f[0] == "T":
                out.append(int(f[1]))
            elif f[0] in ("RESUME", "PP"):
                if f[0] == "RESUME" or len(out) == 0:
                    print("  ", line.strip())
            elif f[0] in ("DONE", "ERR"):
                print("  ", line.strip())
                break
        dt = time.time() - t0
        print(f"  {len(out)} tokens in {dt:.2f} s wall")
        return out

    print("request 1: greedy, 300 tokens")
    a = gen(ids, 300)
    print("request 2: the conversation so far + 300 more, temperature 0.7 top_p 0.95 seed 7")
    b = gen(ids + a, 300, " temperature=0.7 top_p=0.95 top_k=40 seed=7")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=60)
    np.array(a + b, np.int32).tofile("serve_smoke_out.i32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
