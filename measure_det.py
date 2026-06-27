#!/usr/bin/env python3
"""Lossless gate v2 -- for a NON-deterministic stack (base doesn't match itself on near-ties).
Uses FORCED/repetitive prompts where the base IS deterministic (and where SA fires).
Lossless ==> SA reproduces these byte-identically to the base, run after run.
Usage: measure_det.py <url> <label>
"""
import hashlib
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "x"

# Forced/deterministic: exact repetition + strict sequences. No near-ties => base is stable =>
# any divergence is a REAL spec-decode error. Repetition also makes SA fire (the copy path).
# All in the proven-deterministic [2] mold: "output this exact <fixed distinctive content> N times,
# one per line". Forces long deterministic repetition (no stop near-ties) AND makes SA fire (copy).
PROMPTS = [
    "Output this exact list eight times, one per line:\nalpha, beta, gamma, delta, epsilon, zeta\n",
    "Output this exact line ten times, one per line:\nThe answer is forty-two and the secret code is XK9-ZULU.\n",
    "Output this exact CSV row twelve times, one per line:\nid,name,value,status,timestamp,region\n",
    "Output these exact words nine times, one copy per line:\nred green blue yellow purple orange cyan\n",
]


def model_id():
    return json.load(urllib.request.urlopen(URL + "/v1/models", timeout=30))["data"][0]["id"]


def complete(mid, prompt, n=300):
    body = json.dumps({"model": mid, "prompt": prompt, "max_tokens": n,
                       "temperature": 0, "seed": 0, "stream": False}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["text"], r.get("usage", {}).get("completion_tokens", -1)


def main():
    mid = model_id()
    print(f"url={URL} label={LABEL} model={mid}")
    allh = hashlib.sha256()
    for i, p in enumerate(PROMPTS):
        text, ntok = complete(mid, p)
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        allh.update(text.encode())
        print(f"[{i}] tokens={ntok} sha={h}  HEAD {repr(text[:70])}")
        with open(f"/tmp/det_{LABEL}_{i}.txt", "w") as f:
            f.write(text)
    print(f"COMBINED[{LABEL}]={allh.hexdigest()[:20]}")


if __name__ == "__main__":
    main()
