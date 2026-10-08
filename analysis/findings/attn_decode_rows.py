#!/usr/bin/env python
"""Attention of every suffix position at its decode moment, deep band (2026-09-30, for the CTAR before/after figure).

For one image and one recorded arm: row p of the output is the attention of query position p over the 128 suffix keys
at the step at which p was committed, averaged over the heads and over layers 25-32. Rows are therefore comparable
across arms although the arms commit in different orders. Also stored: the decode step and the token of every position.
usage: attn_decode_rows.py <root> <stem> <out.npz> <arm> [<arm> ...]      root = attn_multi100 or attn_case
"""
import sys, os, numpy as np
ROOT, STEM, OUT, ARMS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
IF = "/data/zhaoqiyan/autodl-tmp/information_flow"
LO, HI, M = 24, 32, 128
res = {}
for arm in ARMS:
    d = os.path.join(IF, ROOT, arm, STEM)
    A = np.full((M, M), np.nan, np.float32); step_of = np.full(M, -1); tok = np.full(M, -1)
    for t in range(M):
        z = np.load(os.path.join(d, "step_%d.npz" % t))
        ti = z["transfer_index"][-M:]; ids = z["token_ids"][-M:]
        rows = np.where(ti)[0]
        if not len(rows): continue
        q = z["quantized_attentions"][LO:HI, 0][:, :, rows, :].astype(np.float32)      # layers, heads, rows, keys
        a = (q - float(z["zero_point"])) * float(z["scale"])
        A[rows] = a.mean(axis=(0, 1)); step_of[rows] = t; tok[rows] = ids[rows]
    res[arm + "_attn"] = A; res[arm + "_step"] = step_of; res[arm + "_tok"] = tok
    rep = np.zeros(M, bool); rep[1:] = (tok[1:] == tok[:-1]) & (tok[1:] >= 0)
    w5 = np.array([np.nansum(A[p, max(0, p - 5):p + 6]) - A[p, p] for p in range(M)])
    print("%-12s committed %d positions | repeat positions %d | row sum %.3f | attention on the +-5 context tokens: all %.3f, repeat %.3f"
          % (arm, int((step_of >= 0).sum()), int(rep.sum()), float(np.nanmean(np.nansum(A, 1))), float(np.nanmean(w5)),
             float(np.nanmean(w5[rep])) if rep.any() else float("nan")), flush=True)
np.savez_compressed(OUT, **res)
print("->", OUT)
