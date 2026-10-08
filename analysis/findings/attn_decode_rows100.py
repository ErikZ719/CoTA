#!/usr/bin/env python
"""Attention of every suffix position at its decode moment, deep band, for the 100 recorded images (2026-09-30).
The batch form of attn_decode_rows.py: one 128 x 128 matrix per image and arm (row p = attention of query p over the
suffix keys at the step at which p was committed, mean over heads and layers 25-32), plus decode step and token.
usage: attn_decode_rows100.py <out.npz> <arm> [<arm> ...]        (CPU only; ADR_WORKERS processes)
"""
import sys, os, json, numpy as np
from multiprocessing import Pool
IF = "/data/zhaoqiyan/autodl-tmp/information_flow"; ROOT = IF + "/attn_multi100"
OUT, ARMS = sys.argv[1], sys.argv[2:]
LO, HI, M = 24, 32, 128
STEMS = [os.path.splitext(os.path.basename(f))[0] for f in json.load(open(IF + "/coco100_seed0.json"))["files"]]
def one(job):
    arm, stem = job; d = os.path.join(ROOT, arm, stem)
    A = np.full((M, M), np.nan, np.float32); step_of = np.full(M, -1, np.int16); tok = np.full(M, -1, np.int64)
    for t in range(M):
        z = np.load(os.path.join(d, "step_%d.npz" % t))
        ti = z["transfer_index"][-M:]; rows = np.where(ti)[0]
        if not len(rows): continue
        q = z["quantized_attentions"][LO:HI, 0][:, :, rows, :].astype(np.float32)
        A[rows] = ((q - float(z["zero_point"])) * float(z["scale"])).mean(axis=(0, 1)); step_of[rows] = t; tok[rows] = z["token_ids"][-M:][rows]
    return arm, stem, A.astype(np.float16), step_of, tok
if __name__ == "__main__":
    jobs = [(a, s) for a in ARMS for s in STEMS]
    res = {}
    with Pool(int(os.environ.get("ADR_WORKERS", "16"))) as pool:
        for k, (arm, stem, A, st, tk) in enumerate(pool.imap_unordered(one, jobs), 1):
            res.setdefault(arm, {})[stem] = (A, st, tk)
            if k % 25 == 0: print("%d / %d" % (k, len(jobs)), flush=True)
    out = dict(stems=np.array(STEMS))
    for arm in ARMS:
        out[arm + "_attn"] = np.stack([res[arm][s][0] for s in STEMS]); out[arm + "_step"] = np.stack([res[arm][s][1] for s in STEMS])
        out[arm + "_tok"] = np.stack([res[arm][s][2] for s in STEMS])
    np.savez_compressed(OUT, **out); print("->", OUT, flush=True)
