"""Plain generation throughput: 200-token completions on varied prompts, at
several concurrencies, completion tokens per second of wall time.

usage: gen-bench.py   (PORT overrides 8010, as in smoke.sh)"""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8010')}"
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Explain how a bicycle stays upright, in plain language.",
           "Describe a morning in a busy fish market.",
           "Give a recipe for a simple lentil soup, with steps.",
           "Write a letter from a robot to its inventor.",
           "Summarize the causes of the French Revolution.",
           "Describe the rules of chess to a child.",
           "Write a dialogue between two astronauts repairing a satellite."]


def one(i):
    body = {"model": "dgemma", "max_tokens": 200,
            "messages": [{"role": "user", "content": f"({i}) {PROMPTS[i % len(PROMPTS)]}"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["usage"]["completion_tokens"], time.time() - t0


for _ in range(2):
    one(0)
print(f"{'conc':>4} {'reqs':>4} {'tok/s':>8} {'req/s':>6} {'mean tok':>8} {'mean s':>7}")
for conc, n in ((1, 8), (4, 16), (8, 32), (16, 48), (32, 64)):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    toks = sum(r[0] for r in res)
    print(f"{conc:4d} {n:4d} {toks / wall:8.1f} {n / wall:6.2f} {toks / n:8.1f} {sum(r[1] for r in res) / n:7.2f}", flush=True)
