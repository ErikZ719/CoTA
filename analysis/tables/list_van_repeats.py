#!/usr/bin/env python
"""Every image on which the uncached LLaDA-V repeats (Table VI runs), with what dLLM-Cache and CoTA++ do on it."""
import glob, json, os, sys
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"; EOT = {126081, 126348}
HARMLESS = {11: ",", 13: ".", 268: " the", 259: " a", 301: " and", 300: " of"}
L = int(sys.argv[1]) if len(sys.argv) > 1 else 128
P = json.load(open(E + "/results/table5_provenance.json"))["cells"]
M = os.environ.get("MODEL", "LLaDA-V"); B = os.environ.get("BACKEND", "dLLM-Cache")
ARMS = {"van": "%s|Vanilla|%d" % (M, L), "cache": "%s|%s|%d" % (M, B, L), "cpp": "%s|%s+CoTA++|%d" % (M, B, L)}
def load(cell):
    d = {}
    for r in P[cell]["runs"]:
        for p in sorted(glob.glob("%s/results/%s/outputs.jsonl" % (E, r["dir"]))):
            for l in open(p):
                x = json.loads(l); d.setdefault(x["image"], x["ids"])
    return d
def runs(ids):
    c = RM.strip_layout(RM.trim_at_eot(ids, EOT)); out = []; i = 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]: j += 1
        if j > i: out.append((c[i], j - i + 1, i))
        i = j + 1
    return out, c
D = {a: load(c) for a, c in ARMS.items()}
os.environ.setdefault("HF_HOME", "/data/zhaoqiyan/autodl-tmp/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-V")
nm = lambda t: HARMLESS.get(t, tok.decode([t]))
for im in sorted(set(D["van"]) & set(D["cache"]) & set(D["cpp"])):
    rv, cv = runs(D["van"][im]); rc, cc = runs(D["cache"][im]); rp, cp = runs(D["cpp"][im])
    if not rv: continue
    mv = max(r[1] for r in rv); mc = max([r[1] for r in rc], default=0); mp = max([r[1] for r in rp], default=0)
    print("%s  van %d %s | cache %d %s | cpp %d %s" % (im[-10:-4], mv, [(nm(t), n) for t, n, s in rv][:4], mc, [(nm(t), n) for t, n, s in rc][:2], mp, [(nm(t), n) for t, n, s in rp][:4]))
