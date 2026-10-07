"""Compare llama-server generation speed with and without MTP drafting on GPU 0."""
import json, os, subprocess, sys, time, urllib.request

SNAP = r"E:\huggingface\hub\models--ggml-org--MiMo-V2.6-Flash-RL-GGUF\snapshots\a5d1269fdcf9c3346934411f700038e9cd9d5102"
MODEL = SNAP + r"\MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf"
MTP = SNAP + r"\mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf"
SERVER = r"D:\repos\llama.cpp\build\bin\llama-server.exe"
OUT = os.path.dirname(os.path.abspath(__file__))
PORT = 8091

BASE = [SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "24", "-fa", "on", "-c", "16384",
        "-np", "1", "-t", "24", "--host", "127.0.0.1", "--port", str(PORT), "--jinja"]
CONFIGS = {
    "none": [],
    "mtp-n1": ["--spec-type", "draft-mtp", "-md", MTP, "--spec-draft-n-max", "1"],
    "mtp-n2": ["--spec-type", "draft-mtp", "-md", MTP, "--spec-draft-n-max", "2"],
    "mtp-n3": ["--spec-type", "draft-mtp", "-md", MTP, "--spec-draft-n-max", "3"],
}
PROMPTS = [
    "Write a Python function that merges overlapping intervals, with a short explanation.",
    "Explain how a B-tree differs from a binary search tree.",
    "Write a short story (about 400 words) about a lighthouse keeper who finds a radio.",
    "Write a C++ class for a thread-safe bounded queue with blocking push and pop.",
]


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def run(name, extra):
    log = open(os.path.join(OUT, f"spec_{name}.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen(BASE + extra, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                return {"config": name, "error": f"server exited {p.returncode}, see spec_{name}.log"}
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                if time.time() - t0 > 1200:
                    return {"config": name, "error": "server not ready after 20 min"}
                time.sleep(2)
        post({"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64, "temperature": 0})
        rows = []
        for pr in PROMPTS:
            d = post({"messages": [{"role": "user", "content": pr}], "max_tokens": 512, "temperature": 0})
            t = d["timings"]
            rows.append({"n": t["predicted_n"], "tps": t["predicted_per_second"],
                         "draft": t.get("draft_n"), "accepted": t.get("draft_n_accepted")})
        n = sum(r["n"] for r in rows)
        secs = sum(r["n"] / r["tps"] for r in rows)
        dr = sum(r["draft"] or 0 for r in rows)
        ac = sum(r["accepted"] or 0 for r in rows)
        return {"config": name, "tokens": n, "tps": round(n / secs, 2),
                "per_prompt_tps": [round(r["tps"], 2) for r in rows],
                "accept_rate": round(ac / dr, 3) if dr else None}
    finally:
        p.kill()
        p.wait()
        log.close()
        time.sleep(5)


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    for name in names:
        res = run(name, CONFIGS[name])
        print(json.dumps(res), flush=True)
