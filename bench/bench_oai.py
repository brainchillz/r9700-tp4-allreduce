#!/usr/bin/env python3
# Engine-neutral benchmark over the OpenAI streaming API: same prompts as bench.py.
# Measures TTFT (prompt tok/s = prompt_tokens / TTFT) and decode tok/s from the stream wall clock.
# Usage: URL=http://127.0.0.1:8092 MODEL=name bench_oai.py [label] [runs]
import json, sys, time, urllib.request, os
URL = os.environ.get("URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("MODEL", "")
label = sys.argv[1] if len(sys.argv) > 1 else "run"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
here = os.path.dirname(os.path.abspath(__file__))
long_doc = open(os.path.join(here, "longprompt.txt")).read()

def chat(messages, n, tag):
    body = {"messages": messages, "max_tokens": n, "stream": True, "temperature": 0.7, "seed": 42,
            "stream_options": {"include_usage": True}, "cache_prompt": False}
    if MODEL: body["model"] = MODEL
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None; chunks = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                if t_first is None: t_first = time.time()
                chunks += 1
    t_end = time.time()
    pt = (usage or {}).get("prompt_tokens", 0); ct = (usage or {}).get("completion_tokens", 0)
    ttft = (t_first or t_end) - t0
    dec = ct / (t_end - t_first) if t_first and t_end > t_first else 0
    print(f"{label:14s} {tag:9s} pp={pt:5d}tok ttft={ttft:5.2f}s ({pt/ttft if ttft else 0:7.1f} t/s) | "
          f"tg={ct:4d}tok {dec:6.2f} t/s | wall={t_end-t0:5.1f}s", flush=True)

sys_msg = {"role": "system", "content": "You are a helpful assistant. /no_think"}
for i in range(runs):
    chat([sys_msg, {"role": "user", "content": "Here is a CLI help text:\n\n" + long_doc + "\n\nSummarize the ten most important options in a bulleted list."}], 300, "long-pp")
    chat([sys_msg, {"role": "user", "content": "Write a detailed, original 400-word short story about a lighthouse keeper who discovers a message in a bottle. Do not use lists."}], 300, "short-pp")
    chat([sys_msg, {"role": "user", "content": "Write a Python module implementing an LRU cache class with get/put, TTL expiry, and unit tests using pytest. Include docstrings."}], 400, "code")
