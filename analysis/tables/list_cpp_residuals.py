#!/usr/bin/env python
"""Every image on which CoTA++ still leaves a non-digit repeat (Table VI runs), with the cache run on the same image."""
import glob, json, os, sys
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"; EOT = {126081, 126348}
L = int(sys.argv[1]) if len(sys.argv) > 1 else 128; M = os.environ.get("MODEL", "LLaDA-V"); B = os.environ.get("BACKEND", "dLLM-Cache")
P = json.load(open(E + "/results/table5_provenance.json"))["cells"]
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
pc = lambda t: tok.decode([t])
for im in sorted(set(D["van"]) & set(D["cache"]) & set(D["cpp"])):
    rp, cp = runs(D["cpp"][im])
    rp = [r for r in rp if not pc(r[0]).strip().isdigit()]
    if not rp: continue
    rv, cv = runs(D["van"][im]); rc, cc = runs(D["cache"][im])
    mv = max([r[1] for r in rv], default=0); mc = max([r[1] for r in rc], default=0)
    tcache = max(rc, key=lambda r: r[1]) if rc else None
    print("\n== %s  van %d | cache %d x %r | cpp %s" % (im[-10:-4], mv, mc, pc(tcache[0]) if tcache else "", [(pc(t), n) for t, n, s in rp]))
    for t, n, s in rp[:3]:
        print("   cpp  ...%s..." % tok.decode(cp[max(0, s - 12):s + n + 10]).replace("\n", " "))
    if tcache:
        t, n, s = tcache; print("   cache...%s..." % tok.decode(cc[max(0, s - 10):s + min(n, 8) + 6]).replace("\n", " "))
