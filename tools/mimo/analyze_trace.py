"""Expert-routing skew for MiMo-V2.6-Flash-RL from moe-trace files.

For each trace: split documents into profile (even doc ids) and test (odd) halves, pick the globally most-used
experts from the profile half (any layer), and report what share of the test half's expert choices they serve.
Also reports a per-layer LRU cache of the same total size, and the speed the hit rate implies.
"""
import struct, sys
import numpy as np

N_LAYER_MOE, N_EXP, K = 47, 256, 8
TOTAL = N_LAYER_MOE * N_EXP                    # 12,032 experts
EXPERT_MIB = 12.75                             # 3 x 4096x2048 MXFP4 matrices
BUDGET_EXPERTS = int(82.3 * 1024 / EXPERT_MIB) # 256K context, 8-bit KV


def load(path):
    b = open(path, "rb").read()
    nl, k = struct.unpack_from("<II", b, 12)
    dt = np.dtype([("doc", "<u4"), ("pos", "<u4"), ("L", [("ids", "<i2", (k,)), ("w", "<f4", (k,))], (nl,))])
    a = np.frombuffer(b, dtype=dt, offset=20, count=(len(b) - 20) // dt.itemsize)
    return a["doc"], a["L"]["ids"][:, 1:, :].astype(np.int64)   # drop dense layer 0 -> [T, 47, 8]


def counts(ids):
    c = np.zeros((N_LAYER_MOE, N_EXP), np.int64)
    for l in range(N_LAYER_MOE):
        c[l] = np.bincount(ids[:, l, :].ravel(), minlength=N_EXP)
    return c


def static_hit(profile, test, n_cache):
    order = np.argsort(profile.ravel())[::-1][:n_cache]
    resident = np.zeros(TOTAL, bool)
    resident[order] = True
    return (test.ravel() * resident).sum() / test.sum(), resident.reshape(N_LAYER_MOE, N_EXP)


def lru_hit(ids, n_cache):
    per_layer = n_cache // N_LAYER_MOE
    from collections import OrderedDict
    hits = total = 0
    for l in range(N_LAYER_MOE):
        resident = OrderedDict()
        for row in ids[:, l, :].tolist():
            for e in row:
                total += 1
                if e in resident:
                    hits += 1
                    resident.move_to_end(e)
                else:
                    if len(resident) >= per_layer:
                        resident.popitem(last=False)
                    resident[e] = None
    return hits / total


def speed(hit, cpu_gbs=120.0, gpu_gbs=1600.0, dense_gb=6.5, overhead_ms=1.0):
    """Decode estimate, no drafting: GPU (dense + hit experts) and CPU (missed experts) run in parallel."""
    exp_gb = N_LAYER_MOE * K * EXPERT_MIB / 1024 * 1.073741824   # GB per token
    gpu_ms = (dense_gb + exp_gb * hit) / gpu_gbs * 1000
    cpu_ms = exp_gb * (1 - hit) / cpu_gbs * 1000
    return 1000 / (max(gpu_ms, cpu_ms) + overhead_ms)


def report(name, path):
    doc, ids = load(path)
    prof, test = ids[doc % 2 == 0], ids[doc % 2 == 1]
    cp, ct = counts(prof), counts(test)
    print(f"\n== {name}: {len(ids)} tokens ({len(prof)} profile / {len(test)} test), {len(np.unique(doc))} docs")
    share = np.sort(counts(ids).ravel())[::-1].cumsum() / counts(ids).sum()
    print("  top-N experts' share of ALL choices (in-sample): " +
          ", ".join(f"{p}%: {share[int(TOTAL * p / 100) - 1] * 100:.1f}%" for p in (10, 25, 40, 50, 55, 60, 70)))
    print("  never chosen in this trace:", int((counts(ids) == 0).sum()), "of", TOTAL)
    print("  cache size  | static profile hit (out-of-sample) | est. tok/s @120 GB/s CPU | @90 GB/s")
    for frac in (0.40, 0.50, BUDGET_EXPERTS / TOTAL, 0.60, 0.70):
        n = int(TOTAL * frac)
        h, _ = static_hit(cp, ct, n)
        print(f"  {n:5d} ({frac * 100:4.1f}%) | {h * 100:5.1f}% | {speed(h):5.1f} | {speed(h, cpu_gbs=90):5.1f}")
    # per-layer balance of the chosen budget
    h, res = static_hit(cp, ct, BUDGET_EXPERTS)
    per = res.sum(1)
    print(f"  at budget: resident experts per layer min {per.min()} / median {int(np.median(per))} / max {per.max()}")
    # how much a dynamic per-layer LRU of the same size would get (test half, in order)
    sub = test[: min(len(test), 20000)]
    print(f"  per-layer LRU, same size, first {len(sub)} test tokens: {lru_hit(sub, BUDGET_EXPERTS) * 100:.1f}%")
    # two-token verify windows (MTP n=1): share of a window's unique experts that are resident
    pairs = np.concatenate([test[0:-1:2], test[1::2]], axis=2) if len(test) > 1 else test
    uniq = [len(set(r)) for r in pairs[:5000].reshape(-1, 2 * K)]
    print(f"  2-token windows: {np.mean(uniq):.2f} unique experts per layer (of 16 slots)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        report(name, path)
