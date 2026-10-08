"""Dump, for every Table V cell, the response text that the metrics were computed on (truncated at the first
end-of-text token), one jsonl per cell under results/table5_texts/. Input of the LLM-judge column of Table VI.
The cell -> runs mapping is read from results/table5_provenance.json (exact directories, no globs).
Run in the mmada env (same as tab5_build.py)."""
import json, os, sys, re
os.environ.setdefault("HF_HOME", "/data/zhaoqiyan/autodl-tmp/hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
sys.path.insert(0, E + "/scripts"); import repeat_metrics as RM
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True)
MM_EOT = {i for i in (tk.eos_token_id, tk.convert_tokens_to_ids("<|eot_id|>")) if isinstance(i, int) and i >= 0}
FS = set(json.load(open(E + "/data/coco500_final.json"))["files"])
PROV = json.load(open(E + "/results/table5_provenance.json")); prov = PROV["cells"]
out = E + "/results/table5_texts"; os.makedirs(out, exist_ok=True)
same = tot = 0; index = {}
for key, c in prov.items():
    m, meth, L = key.split("|")
    eot = MM_EOT if m in ("MMaDA", "LaViDa") else {126081, 126348}   # 2026-09-30: see tab5_build.py
    d = {}
    for run in c["runs"]:                      # same order as the builder: later runs overwrite earlier ones
        p = "%s/results/%s/outputs.jsonl" % (E, run["dir"])
        for l in open(p):
            r = json.loads(l)
            if r["image"] in FS:
                ids = RM.trim_at_eot(r["ids"], eot)
                txt = tk.decode(ids, skip_special_tokens=True).strip()
                tot += 1; same += (txt == r["text"].strip())
                d[r["image"]] = dict(image=r["image"], text=txt, n_tok=len(ids), run=run["dir"])
    name = re.sub(r"[^A-Za-z0-9]+", "_", key.replace("++", "pp")).strip("_")
    with open("%s/%s.jsonl" % (out, name), "w") as f:
        for im in sorted(d):
            f.write(json.dumps(d[im], ensure_ascii=False) + "\n")
    index[key] = dict(file=name + ".jsonl", n=len(d))
# the stamp lets judge_table6.py refuse a summary whose texts do not belong to the current provenance
json.dump(dict(provenance_built=PROV["built"], runs={k: [r["dir"] for r in c["runs"]] for k, c in prov.items()}, cells=index),
          open(out + "/INDEX.json", "w"), indent=1)
print("cells", len(index), "| rows", tot, "| decoded == stored text:", same,
      "| incomplete:", [k for k, v in index.items() if v["n"] != 500])
