"""Compare Strata-MiMo's intermediate tensors with llama.cpp's (tools/mimo/mimo-dump.cpp) on the same tokens.

Both directories hold raw little-endian <name>.f32 / <name>.i32 files named as llama.cpp names them ("l_out-17",
"ffn_moe_topk-3", "result_output", ...) and a tokens.i32.  For every tensor present in both: relative L2 error
||a - b|| / ||b|| and the cosine per token (worst token reported); for ffn_moe_topk-L the share of tokens whose expert
set differs; for result_output the top-1 / top-5 agreement and the mean KL(ref || engine).

usage: python mimo_parity.py <engine dump dir> <llama.cpp dump dir> [--tokens N]
"""
import argparse
import pathlib
import re
import sys

import numpy as np


def load(d: pathlib.Path, name: str, n_tok: int):
    for ext, dt in ((".f32", np.float32), (".i32", np.int32)):
        p = d / (name + ext)
        if p.exists():
            a = np.fromfile(p, dtype=dt)
            return a.reshape(n_tok, -1) if a.size % n_tok == 0 else None
    return None


def layer_key(name: str):
    m = re.match(r"(.*)-(\d+)$", name)
    return (int(m.group(2)), m.group(1)) if m else (10**6, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("reference")
    ap.add_argument("--tokens", type=int, default=0, help="compare only the first N tokens")
    ap.add_argument("--max-rel", type=float, default=0.05, help="flag tensors whose worst-token rel error exceeds this")
    args = ap.parse_args()
    eng, ref = pathlib.Path(args.engine), pathlib.Path(args.reference)
    t_ref = np.fromfile(ref / "tokens.i32", dtype=np.int32)
    t_eng = np.fromfile(eng / "tokens.i32", dtype=np.int32)
    n = min(len(t_ref), len(t_eng))
    if args.tokens:
        n = min(n, args.tokens)
    if not np.array_equal(t_ref[:n], t_eng[:n]):
        print("token sequences differ: the dumps are not comparable")
        return 1
    names = sorted({p.stem for p in eng.iterdir() if p.suffix in (".f32", ".i32") and p.stem != "tokens"}, key=layer_key)
    flagged = 0
    print(f"{n} tokens; {'tensor':<26} {'rel L2 (mean / worst tok)':>28} {'cos (worst tok)':>16}")
    for name in names:
        a = load(eng, name, len(t_eng))
        b = load(ref, name, len(t_ref))
        if a is None or b is None or a.shape[1] != b.shape[1]:
            continue
        a, b = a[:n], b[:n]
        if name.startswith("ffn_moe_topk"):
            diff = sum(set(x) != set(y) for x, y in zip(a.tolist(), b.tolist()))
            print(f"  {name:<26} expert sets differ on {diff} of {n} tokens")
            continue
        if name == "result_output":
            top1 = np.mean(a.argmax(1) == b.argmax(1))
            top5 = np.mean([len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) / 5 for x, y in zip(a, b)])
            la = a - a.max(1, keepdims=True)
            lb = b - b.max(1, keepdims=True)
            la -= np.log(np.exp(la).sum(1, keepdims=True))
            lb -= np.log(np.exp(lb).sum(1, keepdims=True))
            kl = np.sum(np.exp(lb) * (lb - la), axis=1)
            print(f"  {name:<26} top-1 same {top1 * 100:.2f}%, top-5 overlap {top5 * 100:.2f}%, "
                  f"KL mean {kl.mean():.5f} / max {kl.max():.4f}")
            continue
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        rel = np.linalg.norm(a64 - b64, axis=1) / np.maximum(np.linalg.norm(b64, axis=1), 1e-30)
        cos = np.sum(a64 * b64, axis=1) / np.maximum(np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1), 1e-30)
        mark = ""
        if rel.max() > args.max_rel:
            mark = f"   <-- worst token {int(rel.argmax())}"
            flagged += 1
        print(f"  {name:<26} {rel.mean():12.2e} / {rel.max():9.2e} {cos.min():16.6f}{mark}")
    print(f"{flagged} tensor(s) above {args.max_rel} relative error on some token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
