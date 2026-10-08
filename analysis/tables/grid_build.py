"""Component grid of Sec. VI-D (LLaDA-V + dLLM-Cache, 500 COCO images, L = 64 / 128 / 512): the 2^3 subsets of
{DAR, CTAR, CTEV}. Same metric code and image list as Table V (tab5_build.py). Five subsets come from
results/lladav/grid (queue of 2026-09-28); none, CTAR+CTEV (= CoTA) and the full stack (= CoTA++) are the Table V runs."""
import json, glob, sys, os, time, hashlib
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
import numpy as np
from scipy import stats
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"; EOT = 126081
FILES = json.load(open(E + "/data/coco500_final.json"))["files"]; FS = set(FILES)
def load(pats):
    d = {}; src = []
    for pat in pats:
        for p in sorted(glob.glob(f"{E}/results/{pat}/outputs.jsonl")):
            if "_oldcode" in p: continue
            src.append(os.path.relpath(os.path.dirname(p), E + "/results"))
            for l in open(p):
                r = json.loads(l)
                if r["image"] in FS:
                    m = RM.sample_metrics(r["ids"], EOT)
                    m["_runs"] = RM.run_lengths(RM.strip_layout(RM.trim_at_eot(r["ids"], EOT)))
                    m["_rep3"] = RM.seq_rep_n(RM.trim_at_eot(r["ids"], EOT), 3)
                    d[r["image"]] = m
    return d, src
def cell(d):
    v = list(d.values()); runs = [x for m in v for x in m["_runs"]]; f = lambda k: float(np.mean([x[k] for x in v]))
    return dict(n=len(v), arr=100 * f("arr"), srr=100 * float(np.mean([x["mrl"] >= 2 for x in v])),
                mrl=float(max(runs)) if runs else 0.0, arl=float(np.mean(runs)) if runs else 0.0,
                p95=float(np.percentile(runs, 95)) if runs else 0.0,
                d1=f("distinct_1"), d2=f("distinct_2"), r3=f("_rep3"), r4=f("seq_rep_4"), len=f("len_trimmed"))
SPEC = {
 "vanilla":   {512: ["lladav/l512van/*", "lladav/l512t/van_*", "lladav/t5/van512_*"], 128: ["lladav/baseline_L128", "lladav/baseline_L128_b2"], 64: ["lladav/baseline_L64", "lladav/baseline_L64_b2"]},
 "none":      {512: ["lladav/headline/off_*", "lladav/l512x/off_*", "lladav/l512t/off_*", "lladav/t5/off512_*"], 128: ["lladav/dllm_cache_L128", "lladav/dllm_cache_L128_b2"], 64: ["lladav/dllm_cache_L64", "lladav/dllm_cache_L64_b2"]},
 "DAR":       {L: [f"lladav/grid/dar{L}_*"] for L in (512, 128, 64)},
 "CTAR":      {L: [f"lladav/grid/ctar{L}_*"] for L in (512, 128, 64)},
 "CTEV":      {L: [f"lladav/grid/ctev{L}_*"] for L in (512, 128, 64)},
 "DAR+CTAR":  {L: [f"lladav/grid/darctar{L}_*"] for L in (512, 128, 64)},
 "DAR+CTEV":  {L: [f"lladav/grid/darctev{L}_*"] for L in (512, 128, 64)},
 "CTAR+CTEV": {512: ["lladav/gf/ctarth_ctev_*", "lladav/t5/cota512_*"], 128: ["lladav/t5/cota128_*"], 64: ["lladav/t5/cota64_*"]},
 "all":       {512: ["lladav/gf/fullth_*", "lladav/t5/cotapp512_*"], 128: ["lladav/t5/cotapp128_*"], 64: ["lladav/t5/cotapp64_*"]},
}
data = {}; out = {}
for k, per in SPEC.items():
    for L, pats in per.items():
        d, src = load(pats); data[(k, L)] = d; c = cell(d); c["runs"] = src; out[f"{k}|{L}"] = c
        print(f"{k:10s} L={L:3d} n={c['n']:3d}  ARR={c['arr']:7.3f}  SRR={c['srr']:5.1f}  MRL={c['mrl']:5.0f}  ARL={c['arl']:.2f}  95p={c['p95']:.1f}  seq-rep-4={c['r4']:.3f}  len={c['len']:.0f}")
# paired tests over images: against the plain cache and against the vanilla model (ARR per response)
print("\npaired Wilcoxon over images on the per-response ARR")
for L in (64, 128, 512):
    base = data[("none", L)]; van = data[("vanilla", L)]
    for k in ("DAR", "CTAR", "CTEV", "DAR+CTAR", "DAR+CTEV", "CTAR+CTEV", "all"):
        d = data[(k, L)]; ims = [i for i in FILES if i in d and i in base and i in van]
        a = np.array([d[i]["arr"] for i in ims]); b = np.array([base[i]["arr"] for i in ims]); v = np.array([van[i]["arr"] for i in ims])
        pc = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        pv = stats.wilcoxon(a, v).pvalue if np.any(a != v) else 1.0
        out[f"{k}|{L}"].update(p_vs_cache=float(pc), p_vs_vanilla=float(pv), n_paired=len(ims),
                               better_than_cache=int((a < b).sum()), worse_than_cache=int((a > b).sum()))
        print(f"  L={L:3d} {k:10s} n={len(ims)}  vs cache p={pc:.1e} (lower in {int((a<b).sum())}, higher in {int((a>b).sum())})   vs vanilla p={pv:.2g}")
meta = dict(built=time.strftime("%Y-%m-%d %H:%M"), images="data/coco500_final.json", code_sha1={p: hashlib.sha1(open(os.path.join(E, p), "rb").read()).hexdigest()[:12] for p in ("scripts/repeat_metrics.py", "scripts/run_repeat_eval.py")})
json.dump(dict(meta=meta, cells=out), open(E + "/results/grid_cells.json", "w"), indent=1)
print("saved results/grid_cells.json")
