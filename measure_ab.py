#!/usr/bin/env python3
"""Averaged 3-way bake-off probe. Codegen (partial SA), novel (SA sits out), copy (SA fires)."""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"


def comp(p, n):
    b = json.dumps({"model": "GLM-5.2", "prompt": p, "max_tokens": n,
                    "temperature": 0, "seed": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/completions", data=b,
        headers={"Content-Type": "application/json"}), timeout=600))
    return r["usage"]["completion_tokens"], time.time() - t


CG = "Write a Python class for an LRU cache with get and put methods, type hints, and full docstrings.\n"
NV = "Write a short story about a lighthouse keeper who discovers a hidden cave beneath the rocks.\n"
CP = "Repeat this exact line eighty times, one per line:\nThe quick brown fox jumps over the lazy dog number seven.\n"

for nm, p, n, k in [("CODEGEN", CG, 350, 4), ("NOVEL", NV, 300, 4), ("COPY", CP, 900, 2)]:
    xs = [a / dt for a, dt in (comp(p, n) for _ in range(k))]
    print(f"  {nm:8s} x{k}: {[round(x) for x in xs]}  avg={sum(xs)/len(xs):.0f} t/s")
