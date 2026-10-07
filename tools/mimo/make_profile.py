"""Build a MiMo expert-usage profile from moe-trace files (tools/mimo/moe-trace.cpp).

The decode driver keeps the most-used experts in VRAM; this file says which ones those are.  Output format
("MIMOPROF" v1, little-endian): magic[8], u32 version=1, u32 n_layer, u32 n_expert, u64 tokens, then
f32 score[n_layer][n_expert] (the share of that layer's choices that went to the expert; 0 for a dense layer).

usage: python make_profile.py <out.prof> <trace> [<trace>:<weight> ...]
       a trace's weight (default 1) scales its tokens' contribution, e.g. corpus_prompts.trace:0.25
"""
import struct
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from analyze_trace import load  # noqa: E402

N_LAYER, N_EXPERT = 48, 256


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out = sys.argv[1]
    score = np.zeros((N_LAYER, N_EXPERT), np.float64)
    tokens = 0
    for arg in sys.argv[2:]:
        path, weight = arg, 1.0
        head, sep, tail = arg.rpartition(":")
        if sep and tail.replace(".", "", 1).isdigit():   # "<trace>:<weight>" (a drive letter's colon is not one)
            path, weight = head, float(tail)
        _, ids = load(path)                       # [T, 47, 8], MoE layers 1..47
        tokens += len(ids)
        for l in range(ids.shape[1]):
            c = np.bincount(ids[:, l, :].ravel(), minlength=N_EXPERT).astype(np.float64)
            score[l + 1] += weight * c / c.sum()
        print(f"{path}: {len(ids)} tokens, weight {weight}")
    with open(out, "wb") as f:
        f.write(b"MIMOPROF")
        f.write(struct.pack("<IIIQ", 1, N_LAYER, N_EXPERT, tokens))
        f.write(score.astype(np.float32).tobytes())
    flat = np.sort(score[1:].ravel())[::-1]
    share = np.cumsum(flat) / flat.sum()
    print(f"wrote {out}: top 6,600 experts hold {share[6599] * 100:.1f}% of the profile's choices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
