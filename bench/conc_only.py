#!/usr/bin/env python3
# Prefill (TTFT) at several prompt sizes with cache-busting random text, then aggregate decode
# at 1/4/8 concurrent streams. OpenAI-compatible server on PORT. Reports tokens/s.
import json, random, sys, time, threading, urllib.request
port = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.8"
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
WORDS = ("system memory kernel thread socket packet buffer index table column vector matrix tensor "
         "gradient cache queue stack heap pointer schema token stream batch latency bandwidth driver").split()

def stream(body, timeout=1800):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; n = 0; pt = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]; pt = d["usage"]["prompt_tokens"]
            ch = d.get("choices")
            if ch and (ch[0]["delta"].get("content") or ch[0]["delta"].get("reasoning_content") or ch[0]["delta"].get("reasoning")):
                now = time.perf_counter(); first = first or now; last = now
    return t0, first, last, n, pt

def body(prompt, max_tokens, seed):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}

print("=== prefill: unique random prompts (prefix cache cannot hit), max_tokens=1, 2 reps each, best shown")
for target in (1024, 4096, 8192, 32768, 65536):
    if True: continue
    best = None
    for rep in range(2):
        rnd = random.Random(target * 7 + rep)
        text = " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(int(target * 0.55)))
        t0, first, last, n, pt = stream(body(text + "\nReply with OK.", 1, rep))
        ttft = first - t0
        res = (pt, ttft, pt / ttft)
        if best is None or res[2] > best[2]: best = res
        print(f"  target {target:6d}: prompt_tokens {pt:6d}  ttft {ttft:7.2f}s  prefill {pt/ttft:8.1f} tok/s", flush=True)
    print(f"  pp{best[0]}: {best[2]:.1f} tok/s (best of 2)", flush=True)

print("=== decode: aggregate at N concurrent streams, thinking off, 400 tokens each, temp 0")
prompts = ["Explain in detail how a %s works, in about 600 words." % t for t in
           ("refrigerator", "jet engine", "transistor", "suspension bridge", "lithium battery",
            "hydraulic press", "GPS receiver", "nuclear reactor")]
for conc in (1, 4, 8):
    results = [None] * conc
    def run(i):
        results[i] = stream(body(prompts[i], 400, 100 + i))
    th = [threading.Thread(target=run, args=(i,)) for i in range(conc)]
    T0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; T1 = time.perf_counter()
    per = [(r[3] - 1) / (r[2] - r[1]) for r in results]
    total = sum(r[3] for r in results)
    print(f"  conc {conc}: per-stream decode {min(per):6.1f}-{max(per):6.1f} tok/s  aggregate {total/(T1-T0):7.1f} tok/s over the wall ({total} tok in {T1-T0:.1f}s)", flush=True)
