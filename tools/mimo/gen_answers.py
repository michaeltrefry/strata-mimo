"""Generate MiMo's own answers (with reasoning) to varied prompts, for routing traces of decode-like text.

Writes answers.txt: one document per prompt, separated by <<<DOC>>> lines, in the model's raw chat format
(prompt rendered by the server's chat template + the raw generated text).
"""
import concurrent.futures as cf, json, os, subprocess, time, urllib.request

SNAP = r"E:\huggingface\hub\models--ggml-org--MiMo-V2.6-Flash-RL-GGUF\snapshots\a5d1269fdcf9c3346934411f700038e9cd9d5102"
MODEL = SNAP + r"\MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf"
SERVER = r"D:\repos\llama.cpp\build\bin\llama-server.exe"
OUT = os.path.dirname(os.path.abspath(__file__))
PORT = 8092

CODE = ["a thread-safe LRU cache in C++", "a Rust function that parses an INI file", "a Python asyncio web crawler with a rate limit",
        "a CUDA kernel that sums a large float array", "a TypeScript debounce hook for React", "a SQL query for monthly active users from an events table",
        "a Go HTTP middleware that logs request latency", "a bash script that rotates log files", "a Python CLI with argparse that renames photos by EXIF date",
        "a C# class that retries an HTTP call with exponential backoff", "unit tests for a Python function that validates email addresses",
        "a Dockerfile for a FastAPI app with a non-root user"]
FIX = ["Why does this Python code print the same lambda result three times: fs=[lambda: i for i in range(3)]; print([f() for f in fs])",
       "My C++ program crashes when I erase from a std::vector inside a range-for loop. Why, and how do I fix it?",
       "My React component re-renders forever after I added a useEffect that sets state. What is going on?",
       "A Postgres query got 50x slower after the table grew to 20M rows. How should I investigate?",
       "Git says 'detached HEAD'. What happened and how do I get my commits back onto main?",
       "My CUDA kernel gives different results each run. List likely causes."]
EXPLAIN = ["how a mixture-of-experts language model routes tokens", "the difference between TCP and QUIC", "how garbage collection works in Go",
           "what a B-tree is and why databases use it", "how speculative decoding speeds up LLM inference", "how HTTPS certificates are validated",
           "the CAP theorem with a practical example", "how a CPU branch predictor works", "what Rust's borrow checker prevents",
           "how GPU memory bandwidth limits LLM token generation", "how vector databases do approximate nearest neighbour search",
           "the idea behind Kalman filters"]
WRITE = ["a short story about a lighthouse keeper who finds a radio", "a polite email declining a meeting invitation",
         "a product description for a mechanical keyboard", "a 6-line poem about autumn in a city", "a summary of the causes of World War I",
         "a weekly meal plan for a vegetarian on a budget", "advice for someone starting their first job as a software engineer",
         "a cover letter opening paragraph for a data analyst role", "a comparison of renting vs buying a home",
         "a travel itinerary for three days in Kyoto"]
AGENT = ["You are a coding agent. The user says: 'Add a --dry-run flag to our deploy script and make sure the tests pass.' Describe the exact steps and commands you would run, then show the patch.",
         "You are an agent with tools read_file, write_file, run_command. Plan how you would find and fix a failing test named test_parse_dates in a Python repo. Show the tool calls as JSON.",
         "Review this diff for bugs: `- if (n > 0) { free(p); }` `+ if (n >= 0) { free(p); p = NULL; }`. Explain your reasoning.",
         "You are a shell assistant. Write a PowerShell command that finds the 20 largest files under D:\\repos and prints size in GB.",
         "Given a JSON API that returns paginated results with a 'next' cursor, write a Python generator that yields all items and handles 429 errors.",
         "Refactor this function to be testable and explain each change: def main(): data = open('/etc/app.conf').read(); r = requests.get(URL); print(process(data, r.json()))"]

PROMPTS = ([f"Write {c}." for c in CODE] + FIX + [f"Explain {e}." for e in EXPLAIN] + [f"Write {w}." for w in WRITE] + AGENT
           + [f"Explain {e} to a 12-year-old." for e in EXPLAIN[:6]] + [f"Write {c}, then explain the design choices." for c in CODE[:8]])


def call(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def one(prompt):
    rendered = call("/apply-template", {"messages": [{"role": "user", "content": prompt}]})["prompt"]
    d = call("/completion", {"prompt": rendered, "n_predict": 1200, "temperature": 0.7, "top_p": 0.95, "cache_prompt": False})
    return rendered + d["content"], d["timings"]["predicted_n"]


if __name__ == "__main__":
    log = open(os.path.join(OUT, "gen_server.log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen([SERVER, "-m", MODEL, "-ngl", "99", "-ncmoe", "24", "-fa", "on", "-c", "32768", "-np", "4",
                          "-t", "24", "--host", "127.0.0.1", "--port", str(PORT), "--jinja"], stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        while True:
            if p.poll() is not None:
                raise SystemExit(f"server exited {p.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        t0, total, docs = time.time(), 0, []
        with cf.ThreadPoolExecutor(4) as ex:
            for i, (text, n) in enumerate(ex.map(one, PROMPTS)):
                docs.append(text)
                total += n
                print(f"{i + 1}/{len(PROMPTS)} prompts, {total} tokens generated, {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(OUT, "answers.txt"), "w", encoding="utf-8") as f:
            f.write("\n<<<DOC>>>\n".join(docs))
        print("DONE", flush=True)
    finally:
        p.kill()
        p.wait()
