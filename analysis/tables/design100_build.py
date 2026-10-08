#!/usr/bin/env python
"""Design tables of the Discussion (2026-10-01): LLaDA-V, dLLM-Cache, L=128, the first 100 images of coco500_final,
every variant alone (results/lladav/design100/<tag>). Metrics as tab5_build.py (content tokens, cut at the first
end-of-text token, ARR/SRR over responses, MRL pooled). Reference rows: the cache and the components of Table X on the
same 100 images (their 500-image runs restricted to the first 100). Prints one line per row; --json writes them."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
FILES = json.load(open(E + "/data/coco500_final.json"))["files"][:100]; FS = set(FILES)
EOT = {126081, 126348}

def load(pats):
    d = {}
    for pat in pats:
        for p in sorted(glob.glob("%s/results/%s/outputs.jsonl" % (E, pat))):
            for l in open(p):
                r = json.loads(l)
                if r["image"] in FS:
                    m = RM.sample_metrics(r["ids"], EOT); m["_runs"] = RM.run_lengths(RM.strip_layout(RM.trim_at_eot(r["ids"], EOT)))
                    m["_rep4"] = RM.seq_rep_n(RM.trim_at_eot(r["ids"], EOT), 4); d[r["image"]] = m
    return d

def cell(d):
    v = list(d.values()); runs = [x for m in v for x in m["_runs"]]
    f = lambda k: float(np.mean([x[k] for x in v]))
    return dict(n=len(v), arr=100 * f("arr"), srr=100 * float(np.mean([x["mrl"] >= 2 for x in v])), mrl=float(max(runs)) if runs else 0.0,
                rep4=f("_rep4"), d2=f("distinct_2"))

P = json.load(open(E + "/results/table5_provenance.json"))["cells"]
REF = {"cache": [r["dir"] for r in P["LLaDA-V|dLLM-Cache|128"]["runs"]],
       "vanilla": [r["dir"] for r in P["LLaDA-V|Vanilla|128"]["runs"]],
       "ctar_grid": ["lladav/grid/ctar128_*"], "dar_grid": ["lladav/grid/dar128_*"]}
TAGS = ["ctar_bias", "ctar_mult", "ctar_gated", "ctar_restore", "ctar_sharpen", "ctar_vold", "ctar_band1_8", "ctar_bandall", "ctar_ctar",
        "dar_r2", "dar_r4", "dar_r8", "dar_r16", "dar_r4w1", "dar_r4w3", "dar_r4w5", "dar_imm6", "dar_anc6", "dar_ent6", "dar_bal222", "dar_immanc", "dar_imment",
        "ctev_self", "ctev_veto", "ctev_l0125", "ctev_l025", "ctev_l05", "ctev_l1", "ctev_w2", "ctev_w10", "ctev_L1_8", "ctev_L9_24", "ctev_L25_32", "ctev_Lall"]
out = {}
for k, pats in REF.items():
    d = load(pats); out[k] = cell(d)
for t in TAGS:
    d = load(["lladav/design100/" + t])
    if d: out[t] = cell(d)
for k, c in out.items():
    print("%-12s n=%3d  ARR %5.2f  SRR %5.1f  MRL %4.0f  rep4 %.3f  d2 %.3f%s" % (k, c["n"], c["arr"], c["srr"], c["mrl"], c["rep4"], c["d2"], "" if c["n"] == 100 else "  (partial)"))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
