#!/usr/bin/env python
'''Compare variants of one component on the 500-image list at L=128, with the metric code of Table V / the component grid.
usage: variant_compare.py <name>=<glob>[,<glob>] ... --ref <name>     (globs relative to results/)
Prints one line per configuration and, for every configuration, the paired comparison with the reference over images.'''
import json, glob, sys, os
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
import numpy as np
from scipy import stats
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"; EOT = 126081
FILES = json.load(open(E + "/data/coco500_final.json"))["files"]; FS = set(FILES)
def load(pats):
    d = {}
    for pat in pats:
        for p in sorted(glob.glob(f"{E}/results/{pat}/outputs.jsonl")):
            for l in open(p):
                r = json.loads(l)
                if r["image"] in FS:
                    m = RM.sample_metrics(r["ids"], EOT); m["_runs"] = RM.run_lengths(RM.strip_layout(RM.trim_at_eot(r["ids"], EOT))); d[r["image"]] = m
    return d
args = sys.argv[1:]; ref = args[args.index("--ref") + 1]; specs = [a for a in args[:args.index("--ref")]]
D = {}
for s in specs:
    k, g = s.split("=", 1); D[k] = load(g.split(","))
for k, d in D.items():
    v = list(d.values()); runs = [x for m in v for x in m["_runs"]]; f = lambda q: float(np.mean([x[q] for x in v]))
    print("%-16s n=%3d  ARR=%6.3f  SRR=%5.1f  MRL=%3.0f  seq-rep-4=%.4f  distinct-2=%.4f  len=%.1f" % (k, len(v), 100 * f("arr"), 100 * float(np.mean([x["mrl"] >= 2 for x in v])), max(runs) if runs else 0, f("seq_rep_4"), f("distinct_2"), f("len_trimmed")))
print("\npaired over images, per-response ARR, against", ref)
for k, d in D.items():
    if k == ref: continue
    ims = [i for i in FILES if i in d and i in D[ref]]
    a = np.array([d[i]["arr"] for i in ims]); b = np.array([D[ref][i]["arr"] for i in ims])
    ra = np.array([d[i]["mrl"] >= 2 for i in ims]); rb = np.array([D[ref][i]["mrl"] >= 2 for i in ims])
    p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    n10, n01 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    pm = stats.binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    print("  %-16s n=%3d  ARR lower in %3d, higher in %3d, Wilcoxon p=%.3g | repeats only here %3d, only in %s %3d, sign test p=%.3g" % (k, len(ims), int((a < b).sum()), int((a > b).sum()), p, n10, ref, n01, pm))
