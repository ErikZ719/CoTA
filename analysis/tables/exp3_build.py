#!/usr/bin/env python
"""Aggregate the three main-text experiments (results/lladav/exp3/<cfg>_128_{0,250}) with the Table VII protocol
(tab5_build.py: content tokens, cut at the first end-of-text token {126081, 126348}, ARR/SRR averaged over the 500
responses, MRL pooled). Reference rows (B = 128, window 26-30) are read from results/table5_cells.json.
Run on the server in the llada-v env: python scripts/analysis/exp3_build.py [--json out]
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts")
import repeat_metrics as RM

E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
FS = set(json.load(open(E + "/data/coco500_final.json"))["files"])
EOT = {126081, 126348}


def load(pat):
    d = {}
    for p in sorted(glob.glob("%s/results/%s/outputs.jsonl" % (E, pat))):
        for l in open(p):
            r = json.loads(l)
            if r["image"] in FS:
                m = RM.sample_metrics(r["ids"], EOT)
                m["_runs"] = RM.run_lengths(RM.strip_layout(RM.trim_at_eot(r["ids"], EOT)))
                d[r["image"]] = m
    return d


def cell(d):
    v = list(d.values())
    runs = [x for m in v for x in m["_runs"]]
    f = lambda k: float(np.mean([x[k] for x in v]))
    return dict(n=len(v), arr=100 * f("arr"), srr=100 * float(np.mean([x["mrl"] >= 2 for x in v])),
                mrl=float(max(runs)) if runs else 0.0, arl=float(np.mean(runs)) if runs else 0.0,
                d2=f("distinct_2"), len=f("len_trimmed"))


CFGS = (["ng2", "ng3"] + ["dcB%d" % b for b in (16, 32, 64)] + ["cppB%d" % b for b in (16, 32, 64)] + ["vanB%d" % b for b in (16, 32, 64)]
        + ["cppL%s" % w for w in ("1_8", "9_24", "25_32", "22_32", "30_32")] + ["ctevL%s" % w for w in ("1_8", "9_24", "25_32", "22_32", "30_32", "26_30")])
out = {}
for c in CFGS:
    d = load("lladav/exp3/%s_128_*" % c)
    if not d:
        continue
    out[c] = cell(d)
    done = "" if out[c]["n"] == 500 else "   (partial: %d/500)" % out[c]["n"]
    print("%-10s n=%3d  ARR %5.2f  SRR %5.1f  MRL %4.0f  ARL %4.2f  d2 %.3f  len %5.1f%s" % (
        c, out[c]["n"], out[c]["arr"], out[c]["srr"], out[c]["mrl"], out[c]["arl"], out[c]["d2"], out[c]["len"], done))
T = json.load(open(E + "/results/table5_cells.json"))
ref = {}
for key in ("LLaDA-V|Vanilla|128", "LLaDA-V|dLLM-Cache|128", "LLaDA-V|dLLM-Cache+CoTA|128", "LLaDA-V|dLLM-Cache+CoTA++|128"):
    c = T[key]["cell"] if "cell" in T.get(key, {}) else T.get(key)
    if c:
        ref[key] = c; print("%-32s ARR %5.2f  SRR %5.1f  MRL %4.0f" % (key, c["arr"], c["srr"], c["mrl"]))
out["_reference"] = ref
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
