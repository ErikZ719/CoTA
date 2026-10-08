#!/usr/bin/env python
"""Candidates for Fig. 10(b) (user, 2026-10-01): LLaDA-V repeats a little on its own (a short harmless run such as ",," or
"the the"), dLLM-Cache makes it much worse, CoTA++ brings it back to a short harmless run. Table VI runs, by image."""
import glob, json, os, sys
sys.path.insert(0, "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts"); import repeat_metrics as RM
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"; EOT = {126081, 126348}
HARMLESS = {11: ",", 13: ".", 268: " the", 259: " a", 301: " and", 300: " of"}
L = int(sys.argv[1]) if len(sys.argv) > 1 else 128
P = json.load(open(E + "/results/table5_provenance.json"))["cells"]
ARMS = {"van": "LLaDA-V|Vanilla|%d" % L, "cache": "LLaDA-V|dLLM-Cache|%d" % L, "cpp": "LLaDA-V|dLLM-Cache+CoTA++|%d" % L}
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
imgs = set(D["van"]) & set(D["cache"]) & set(D["cpp"]); cands = []
for im in imgs:
    rv, cv = runs(D["van"][im]); rc, cc = runs(D["cache"][im]); rp, cp = runs(D["cpp"][im])
    mv = max([r[1] for r in rv], default=0); mc = max([r[1] for r in rc], default=0); mp = max([r[1] for r in rp], default=0)
    CMIN = int(os.environ.get("CMIN", 10)); PMAX = int(os.environ.get("PMAX", 2)); VMAX = int(os.environ.get("VMAX", 3))
    if not (2 <= mv <= VMAX and mc >= CMIN and 2 <= mp <= PMAX): continue
    if os.environ.get("HARM", "1") == "1" and (not all(r[0] in HARMLESS for r in rv) or not all(r[0] in HARMLESS for r in rp)): continue
    cands.append((mc, im, mv, mp, rv, rc, rp))
cands.sort(reverse=True)
print("L=%d images %d candidates %d" % (L, len(imgs), len(cands)))
try:
    os.environ.setdefault("HF_HOME", "/data/zhaoqiyan/autodl-tmp/hf_cache"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-V")
    dec = lambda ids: tok.decode(ids)
except Exception as e:
    print("no tokenizer:", e); dec = lambda ids: str(ids)
for mc, im, mv, mp, rv, rc, rp in cands[:12]:
    print("\n== %s  van %d  cache %d  cpp %d" % (im, mv, mc, mp))
    for a, rr in (("van", rv), ("cache", rc), ("cpp", rp)):
        c = runs(D[a][im])[1]
        for t, n, s in rr[:3]:
            print("   %-5s run %3d x %-6r | ...%s..." % (a, n, HARMLESS.get(t, t), dec(c[max(0, s - 10):s + min(n, 6) + 6]).replace("\n", " ")))
json.dump([dict(image=im, van=mv, cache=mc, cpp=mp) for mc, im, mv, mp, *_ in cands], open(E + "/results/case_b_candidates_%d.json" % L, "w"), indent=1)
