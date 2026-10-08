"""Table V rows (content-token metrics). Prints LaTeX cell triples ARR / SRR / MRLmax per (model, method, L)."""
import json,glob,sys,os
os.environ.setdefault("HF_HOME","/data/zhaoqiyan/autodl-tmp/hf_cache"); os.environ["HF_HUB_OFFLINE"]="1"
sys.path.insert(0,"/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
import numpy as np
E="/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
FILES=json.load(open(E+"/data/coco500_final.json"))["files"]; FS=set(FILES)
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True)
MM_EOT={i for i in (tk.eos_token_id, tk.convert_tokens_to_ids("<|eot_id|>")) if isinstance(i,int) and i>=0}
print("MMaDA terminators", MM_EOT)
def load(pats, eot):
    d={}
    for pat in pats:
        for p in sorted(glob.glob(f"{E}/results/{pat}/outputs.jsonl")):
            if "_oldcode" in p: continue          # archived pre-fix SlowFast runs
            for l in open(p):
                r=json.loads(l)
                if r["image"] in FS:
                    m=RM.sample_metrics(r["ids"],eot)
                    m["_runs"]=RM.run_lengths(RM.strip_layout(RM.trim_at_eot(r["ids"],eot)))
                    m["_rep3"]=RM.seq_rep_n(RM.trim_at_eot(r["ids"],eot),3)
                    d[r["image"]]=m
    return d
def cell(d):
    """Adjacent-run statistics (runs pooled over all responses) and phrase-level statistics."""
    if len(d)<500: return None, len(d)
    v=list(d.values())
    runs=[x for m in v for x in m["_runs"]]
    f=lambda k: float(np.mean([x[k] for x in v]))
    c=dict(arr=100*f("arr"), srr=100*float(np.mean([x["mrl"]>=2 for x in v])),
           mrl=float(max(runs)) if runs else 0.0, arl=float(np.mean(runs)) if runs else 0.0,
           p95=float(np.percentile(runs,95)) if runs else 0.0,
           d1=f("distinct_1"), d2=f("distinct_2"), r3=f("_rep3"), r4=f("seq_rep_4"), len=f("len_trimmed"))
    return c, 500
# L=512 of MMaDA and LaViDa: plan-2 runs of 2026-09-23 (fixed refresh budget 12 tokens/step, LaViDa in 32-token
# blocks) under results/*/t5v2/. The 2026-09-20 single-block runs under */t5/*512_* are superseded (moved to _superseded).
SPEC={
 ("LLaDA-V","Vanilla"):   {512:["lladav/l512van/*","lladav/l512t/van_*","lladav/t5/van512_*"],128:["lladav/baseline_L128","lladav/baseline_L128_b2"],64:["lladav/baseline_L64","lladav/baseline_L64_b2"]},
 ("LLaDA-V","dLLM-Cache"):{512:["lladav/headline/off_*","lladav/l512x/off_*","lladav/l512t/off_*","lladav/t5/off512_*"],128:["lladav/dllm_cache_L128","lladav/dllm_cache_L128_b2"],64:["lladav/dllm_cache_L64","lladav/dllm_cache_L64_b2"]},
 ("LLaDA-V","dLLM-Cache+CoTA"):{512:["lladav/gf/ctarth_ctev_*","lladav/t5/cota512_*"],128:["lladav/t5/cota128_*"],64:["lladav/t5/cota64_*"]},
 ("LLaDA-V","dLLM-Cache+CoTA++"):{512:["lladav/gf/fullth_*","lladav/t5/cotapp512_*"],128:["lladav/t5/cotapp128_*"],64:["lladav/t5/cotapp64_*"]},
 ("LLaDA-V","SlowFast"):  {512:["lladav/t5/sf512_*"],128:["lladav/slowfast_L128","lladav/slowfast_L128_b2"],64:["lladav/slowfast_L64","lladav/slowfast_L64_b2"]},
 ("LLaDA-V","SlowFast+CoTA"):{L:[f"lladav/t5/sfcota{L}_*"] for L in (512,128,64)},
 ("LLaDA-V","SlowFast+CoTA++"):{512:["lladav/sfdiag512/sfd512_ctardarrg","lladav/sfguard/g512_*"],
                                128:["lladav/sfguard/g128_*"],64:["lladav/sfguard/g64_*"]},
 ("LaViDa","Vanilla"):       {512:["lavida/t5v2/van512_*"],128:["lavida/t5/van128_*"],64:["lavida/t5/van64_*"]},
 ("LaViDa","dLLM-Cache"):    {512:["lavida/t5v2/dc512_*"],128:["lavida/t5/dc128_*"],64:["lavida/t5/dc64_*"]},
 ("LaViDa","dLLM-Cache+CoTA"):   {512:["lavida/t5v2/cota512_*"],128:["lavida/t5/cota128_*"],64:["lavida/t5/cota64_*"]},
 ("LaViDa","dLLM-Cache+CoTA++"): {512:["lavida/t5v2/cpp512_*"],128:["lavida/t5/cpp128_*"],64:["lavida/t5/cpp64_*"]},
 ("MMaDA","Vanilla"):     {512:["mmada/t5/van512_*"],128:["mmada/mmada_v2_baseline_L128","mmada/supplement_262376/baseline_L128"],64:["mmada/t5/van64_*"]},
 ("MMaDA","dLLM-Cache"):  {512:["mmada/t5v2/off512_*"],128:["mmada/mmada_v2_dllm_cache_L128","mmada/supplement_262376/dllm_cache_L128"],64:["mmada/t5/off64_*"]},
 ("MMaDA","SlowFast"):    {512:["mmada/t5v2/sf512_*"],128:["mmada/mmada_v2_slowfast_L128","mmada/supplement_262376/slowfast_L128"],64:["mmada/t5/sf64_*"]},
 ("MMaDA","dLLM-Cache+CoTA"):   {512:["mmada/t5v2/cota512_*"],128:["mmada/t5/cota128_*"],64:["mmada/t5/cota64_*"]},
 ("MMaDA","dLLM-Cache+CoTA++"): {512:["mmada/t5v2/cotapp512_*"],128:["mmada/t5/cotapp128_*"],64:["mmada/t5/cotapp64_*"]},
 ("MMaDA","SlowFast+CoTA"):     {512:["mmada/t5v2/sfcota512_*"],128:["mmada/t5/sfcota128_*"],64:["mmada/t5/sfcota64_*"]},
 ("MMaDA","SlowFast+CoTA++"):   {512:["mmada/t5v2/sfcotapp512_*"],128:["mmada/t5/sfcotapp128_*"],64:["mmada/t5/sfcotapp64_*"]},
 ("LaViDa","SlowFast"):         {512:["lavida/t5v2/sf512_*"],128:["lavida/t5/sf128_*"],64:["lavida/t5/sf64_*"]},
 ("LaViDa","SlowFast+CoTA"):    {512:["lavida/t5v2/sfcota512_*"],128:["lavida/t5/sfcota128_*"],64:["lavida/t5/sfcota64_*"]},
 ("LaViDa","SlowFast+CoTA++"):  {512:["lavida/t5v2/sfcpp512_*"],128:["lavida/t5/sfcpp128_*"],64:["lavida/t5/sfcpp64_*"]},
}
out={}
for (m,meth),per in SPEC.items():
    # 2026-09-30: LLaDA-V is cut at <|eot_id|> (126348) as well as <|endoftext|> (126081). Its generate() already stops at
    # <|eot_id|>, but the SlowFast sampler returns the whole suffix, which continues past the end of the turn.
    eot = MM_EOT if m in ("MMaDA","LaViDa") else {126081, 126348}
    for L,p in per.items():
        c,n=cell(load(p,eot)); out[f"{m}|{meth}|{L}"]={"n":n,"cell":c}
        print(f"{m:8s} {meth:20s} L={L:3d} n={n:3d} ", " ".join(f"{k}={v:.3f}" for k,v in c.items()) if c else "--")
json.dump(out, open(E+"/results/table5_cells.json","w"), indent=1)


# --- provenance: which runs produced each cell, and with which code ----------------------------
import hashlib, time
CODE = ["scripts/run_repeat_eval.py", "scripts/run_repeat_eval_mmada.py", "scripts/run_repeat_eval_lavida.py",
        "scripts/repeat_metrics.py", "scripts/mmada_cotapp.py", "scripts/analysis/tab5_build.py",
        "../../LLaDA-V/train/llava/hooks/sf_cotapp.py", "../../LLaDA-V/train/llava/hooks/cache_hook_LLaDA_V.py",
        "../../LLaDA-V/train/llava/model/language_model/modeling_llada.py"]
def _sha(p):
    try: return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
    except OSError: return None
def _runs_of(pats):
    out = []
    for pat in pats:
        for p in sorted(glob.glob(f"{E}/results/{pat}/outputs.jsonl")):
            if "_oldcode" in p: continue
            d = os.path.dirname(p); meta = os.path.join(d, "run_meta.json")
            m = json.load(open(meta)) if os.path.exists(meta) else None
            out.append(dict(dir=os.path.relpath(d, E + "/results"),
                            lines=sum(1 for _ in open(p)),
                            finished=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
                            meta=m))
    return out
prov = dict(built=time.strftime("%Y-%m-%d %H:%M"),
            metric="content tokens (layout ids stripped), responses truncated at the first EOT; "
                   "ARR/SRR averaged over responses, MRL/ARL/95pRL pooled over all runs of the 500 responses",
            images=dict(file="data/coco500_final.json", n=len(FILES),
                        sha1=hashlib.sha1(json.dumps(FILES, sort_keys=True).encode()).hexdigest()[:12]),
            code={c: _sha(os.path.join(E, c)) for c in CODE},
            cells={})
for (m, meth), per in SPEC.items():
    for L, pats in per.items():
        k = f"{m}|{meth}|{L}"
        prov["cells"][k] = dict(patterns=pats, n=out[k]["n"], complete=bool(out[k]["cell"]),
                                cell=out[k]["cell"], runs=_runs_of(pats))
json.dump(prov, open(E + "/results/table5_provenance.json", "w"), indent=1)
print("provenance ->", E + "/results/table5_provenance.json",
      sum(1 for c in prov["cells"].values() if c["complete"]), "complete cells")
