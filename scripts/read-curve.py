"""Structured-read throughput: a unique three-question state per request, a
fixed number of clients, 15 s per point, after one warm-up round per point.

usage: read-curve.py [secs]   (STRUCTURED_PORT overrides 8011, as in smoke.sh)"""
import json, os, random, sys, threading, time, urllib.request

S = f"http://127.0.0.1:{os.environ.get('STRUCTURED_PORT', '8011')}"
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 15
WORDS = "server outage billing invoice refund login crash slow demo customer urgent deadline".split()


def body(samples, rnd):
    ticket = " ".join(rnd.choice(WORDS) for _ in range(20)) + f" #{rnd.randrange(10**9)}"
    return {"model": "dgemma", "state": {"ticket": ticket}, "samples": samples, "questions": {
        "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
        "team": {"type": "choice", "instructions": "Which team owns this?",
                 "criteria": {"billing": None, "outage": "service down", "feature": None}},
        "tone": {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "furious"]}}}


def post(b):
    req = urllib.request.Request(S + "/v1/systemone", data=json.dumps(b).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def run(n, samples):
    lat, stop, lock = [], time.time() + SECS, threading.Lock()

    def w(k):
        rnd = random.Random(k * 7919 + samples)
        while time.time() < stop:
            t = time.time()
            post(body(samples, rnd))
            with lock:
                lat.append(time.time() - t)

    ts = [threading.Thread(target=w, args=(k,)) for k in range(n)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    el = time.time() - t0
    lat.sort()
    return len(lat) / el, lat[len(lat) // 2], lat[int(len(lat) * 0.95)]


# warm every client count once, so first-use compiles land outside the timing
for n in (1, 8, 16, 32):
    for s in (1, 4):
        threads = [threading.Thread(target=post, args=(body(s, random.Random(i)),)) for i in range(n)]
        [t.start() for t in threads]
        [t.join() for t in threads]
print(f"{'samples':>7} {'clients':>7} {'req/s':>7} {'decisions/s':>11} {'p50 s':>6} {'p95 s':>6}")
for samples, clients in ((1, 1), (1, 8), (1, 16), (1, 32), (4, 1), (4, 32)):
    rps, p50, p95 = run(clients, samples)
    print(f"{samples:7d} {clients:7d} {rps:7.2f} {rps * 3:11.1f} {p50:6.2f} {p95:6.2f}", flush=True)
