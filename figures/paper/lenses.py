#!/usr/bin/env python
"""The three lenses of the case study, from the traced runs (run_repeat_eval.py --case_trace 1).

For every committed position of one response:
  anchor  F1  deep-band (L25-32) attention mass on the +-5 neighbours, as served at the commit step
  age     F2  steps since the cached state of the position was last recomputed (mean over the 32 layers)
  E_ctx   F3  deep-layer entropy of the committed neighbours (bits) at the commit step
Usage:  python lenses.py B        (tag of the case: results/lladav/case_study/cs_<tag>_<arm>/)
"""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "server_mirror", "results", "lladav", "case_study")
PREFIX = "cs2"          # round 2 of the traced runs carries the paired measures
sys.path.insert(0, os.path.join(HERE, "..", "server_mirror", "scripts")); import repeat_metrics as RM
ARMS = ["van", "cache", "ctar", "dar", "ctev", "full"]

def load(tag, arm, M=128):
    d = os.path.join(R, f"{PREFIX}_{tag}_{arm}")
    t = json.load(open(glob.glob(os.path.join(d, "trace_*.json"))[0]))
    out = json.loads(open(os.path.join(d, "outputs.jsonl")).readline())
    S = dict(step=np.full(M, -1), anchor=np.full(M, np.nan), E=np.full(M, np.nan), age=np.full(M, np.nan),
             conf=np.full(M, np.nan), text=out["text"],
             ids=np.array((out["ids"][-M:] + [126081] * M)[:M]))   # some runs return M-1 tokens
    # F1, paired: routing stored by the cache vs routing re-formed by CTAR, same rows and steps
    pr = t.get("ctar_pair")
    if pr:
        n = np.array(pr["n"]); ok = n > 0
        S["pair_old"] = np.where(ok, np.array(pr["old"]) / np.maximum(n, 1), np.nan)
        S["pair_new"] = np.where(ok, np.array(pr["new"]) / np.maximum(n, 1), np.nan)
    # F3, paired: what raw confidence would have committed vs what was committed, steps where they differ
    S["alt"] = [(r["step"], r["alt_E"], r["E_ctx"], r["alt_pos"], r["pos"]) for r in t["rows"]
                if r.get("alt_pos") is not None and r["alt_pos"] != r["pos"]]
    S["nsteps"] = len(t["rows"])
    fwd_of = {}
    for r in t["rows"]:
        p = r["pos"]; S["step"][p] = r["step"]; fwd_of[p] = r["fwd"]; S["conf"][p] = r["conf"]
        if r["anchor"] is not None: S["anchor"][p] = r["anchor"]
        if r["E_ctx"] is not None: S["E"][p] = r["E_ctx"]
    if t["refresh"]:                                   # age of the cached state at the commit
        L = 32; last = np.zeros((L, M), int); commits = sorted(fwd_of.items(), key=lambda kv: kv[1])
        by_f = {}
        for p, f in commits: by_f.setdefault(f, []).append(p)
        ages = {}
        for f in range(1, t["n_forwards"] + 1):
            part = t["refresh"].get(str(f))
            for l in range(L):
                idx = None if part is None else part.get(str(l))
                if idx is None: last[l, :] = f             # periodic full recompute
                elif idx: last[l, idx] = f
            for p in by_f.get(f, []): ages[p] = float(np.mean(f - last[:, p]))
        for p, a in ages.items(): S["age"][p] = a
    elif arm == "van":
        S["age"][:] = 0.0
    return S

def repeat_mask(ids, eot=126081):
    """positions (in the raw 128-token suffix) that continue an adjacent identical content token"""
    m = np.zeros(len(ids), bool); prev = None
    for i, t in enumerate(ids):
        if t == eot: break
        if t in RM.LAYOUT_IDS: continue
        if prev is not None and t == prev: m[i] = True
        prev = t
    return m

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "B"
    D = {a: load(tag, a) for a in ARMS if os.path.isdir(os.path.join(R, f"{PREFIX}_{tag}_{a}"))}
    rep = repeat_mask(D["cache"]["ids"]); span = np.where(rep)[0]
    print(f"case {tag}: the cache repeats at {rep.sum()} positions, span {span.min()}-{span.max()}" if rep.any() else "no repeats")
    lo, hi = (span.min(), span.max()) if rep.any() else (0, 127)
    sel = np.zeros(128, bool); sel[max(lo - 1, 0):hi + 1] = True
    print(f"{'arm':6s} | anchor (F1)  all / looped span | age (F2)  all / span | E_ctx (F3)  all / span | repeats")
    for a, S in D.items():
        f = lambda v, m=None: (np.nanmean(v[m]) if m is not None else np.nanmean(v))
        print(f"{a:6s} | {f(S['anchor']):6.3f} / {f(S['anchor'], sel):6.3f}          | {f(S['age']):5.2f} / {f(S['age'], sel):5.2f}     | "
              f"{f(S['E']):6.2f} / {f(S['E'], sel):6.2f}        | {int(repeat_mask(S['ids']).sum())}")
