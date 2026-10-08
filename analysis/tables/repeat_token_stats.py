#!/usr/bin/env python
"""Which tokens repeat (2026-09-30, for the word cloud). Reads the runs of the cached rows of Table VI, restricted to
the 500 images, with the metric code of the table: content tokens, response cut at the first end-of-text token, a run
is a maximal sequence of identical tokens of length >= 2. Writes nothing but its own json.
usage: repeat_token_stats.py <LLaDA-V|LaViDa|MMaDA> <out.json>      (MMaDA needs the mmada environment)"""
import json, glob, sys, os, collections
os.environ.setdefault("HF_HOME", "/data/zhaoqiyan/autodl-tmp/hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
sys.path.insert(0, E + "/scripts"); import repeat_metrics as RM
from transformers import AutoTokenizer
MODEL, OUT = sys.argv[1], sys.argv[2]
FS = set(json.load(open(E + "/data/coco500_final.json"))["files"])
SPEC = {   # copied from scripts/analysis/tab5_build.py (the cached rows, no component)
 ("LLaDA-V", "dLLM-Cache"): {512: ["lladav/headline/off_*", "lladav/l512x/off_*", "lladav/l512t/off_*", "lladav/t5/off512_*"], 128: ["lladav/dllm_cache_L128", "lladav/dllm_cache_L128_b2"], 64: ["lladav/dllm_cache_L64", "lladav/dllm_cache_L64_b2"]},
 ("LLaDA-V", "SlowFast"): {512: ["lladav/t5/sf512_*"], 128: ["lladav/slowfast_L128", "lladav/slowfast_L128_b2"], 64: ["lladav/slowfast_L64", "lladav/slowfast_L64_b2"]},
 ("LaViDa", "dLLM-Cache"): {512: ["lavida/t5v2/dc512_*"], 128: ["lavida/t5/dc128_*"], 64: ["lavida/t5/dc64_*"]},
 ("LaViDa", "SlowFast"): {512: ["lavida/t5v2/sf512_*"], 128: ["lavida/t5/sf128_*"], 64: ["lavida/t5/sf64_*"]},
 ("MMaDA", "dLLM-Cache"): {512: ["mmada/t5v2/off512_*"], 128: ["mmada/mmada_v2_dllm_cache_L128", "mmada/supplement_262376/dllm_cache_L128"], 64: ["mmada/t5/off64_*"]},
 ("MMaDA", "SlowFast"): {512: ["mmada/t5v2/sf512_*"], 128: ["mmada/mmada_v2_slowfast_L128", "mmada/supplement_262376/slowfast_L128"], 64: ["mmada/t5/sf64_*"]},
}
if MODEL == "MMaDA":
    tk = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True)
else:
    tk = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
if MODEL == "LLaDA-V":
    EOT = {126081, 126348}          # <|endoftext|> and <|eot_id|>: the SlowFast hook returns the suffix beyond the end of turn
else:
    mm = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True) if MODEL == "MMaDA" else None
    EOT = ({i for i in (mm.eos_token_id, mm.convert_tokens_to_ids("<|eot_id|>")) if isinstance(i, int) and i >= 0}
           if mm is not None else {126081, 126348})
def runs_of(ids):
    out, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[i]: j += 1
        if j > i: out.append((ids[i], j - i + 1))
        i = j + 1
    return out
res = {}
for (m, backend), per in SPEC.items():
    if m != MODEL: continue
    for L, pats in per.items():
        seen, runs, nresp = set(), [], 0
        for pat in pats:
            for p in sorted(glob.glob(f"{E}/results/{pat}/outputs.jsonl")):
                if "_oldcode" in p: continue
                for l in open(p):
                    r = json.loads(l)
                    if r["image"] not in FS: continue
                    if r["image"] in seen: runs = [x for x in runs if x[0] != r["image"]]
                    seen.add(r["image"])
                    ids = RM.strip_layout(RM.trim_at_eot(r["ids"], EOT))
                    runs += [(r["image"], t, n) for t, n in runs_of(ids)]
        by = collections.defaultdict(lambda: dict(runs=0, positions=0, responses=set(), longest=0))
        for img, t, n in runs:
            b = by[t]; b["runs"] += 1; b["positions"] += n - 1; b["responses"].add(img); b["longest"] = max(b["longest"], n)
        toks = [dict(id=int(t), piece=tk.decode([t]), runs=b["runs"], positions=b["positions"], responses=len(b["responses"]), longest=b["longest"])
                for t, b in sorted(by.items(), key=lambda kv: -kv[1]["runs"])]
        res[f"{m}|{backend}|{L}"] = dict(n_responses=len(seen), n_runs=len(runs), n_repeating=len({x[0] for x in runs}), tokens=toks)
        print(f"{m:8s} {backend:10s} L={L:3d} responses={len(seen)} with a run={len({x[0] for x in runs})} runs={len(runs)} distinct tokens={len(toks)} | top:", [(x['piece'], x['runs']) for x in toks[:8]], flush=True)
json.dump(res, open(OUT, "w"), indent=0)
