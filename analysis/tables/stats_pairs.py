#!/usr/bin/env python
"""Paired statistics behind the Section VI-B claims, computed from the local mirror.

Reads `table5_provenance.json` (which names the run directories behind every Table V cell), loads the
per-image responses from `../server_mirror/results/`, and compares the configurations of one
(model, backend, length) against each other on the same images:

  ARR        paired Wilcoxon signed-rank, two-sided, plus a bootstrap 95% interval on the mean drop
  seq-rep-4  the same, to catch repetition displaced to the phrase level rather than removed
  SRR        McNemar exact test on the responses that changed status
  collapse   responses whose longest content-token run reaches 10 tokens

Nothing here touches the server or the GPUs. Refresh the mirror first:
    bash information_flow/results/sync_from_server.sh
Then: /opt/anaconda3/bin/python information_flow/results/table5/stats_pairs.py
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.join(HERE, "..", "server_mirror")
sys.path.insert(0, os.path.join(MIRROR, "scripts"))
import repeat_metrics as RM  # noqa: E402

EOT = {"LLaDA-V": 126081, "MMaDA": {126081, 126348}, "LaViDa": {126081, 126348}}
LENS = [512, 128, 64]
# within one backend, the ladder we want tested
LADDERS = [("Vanilla", "dLLM-Cache"), ("dLLM-Cache", "dLLM-Cache+CoTA"),
           ("dLLM-Cache", "dLLM-Cache+CoTA++"), ("dLLM-Cache+CoTA", "dLLM-Cache+CoTA++"),
           ("Vanilla", "SlowFast"), ("SlowFast", "SlowFast+CoTA"),
           ("SlowFast", "SlowFast+CoTA++"), ("SlowFast+CoTA", "SlowFast+CoTA++")]


def load_cell(runs, eot, keep=None):
    """image -> per-response metrics, for every run directory of one cell.

    `keep` restricts to the evaluation set: the older baseline runs cover a larger pool, and mixing
    those extra images in would quietly change every mean.
    """
    d = {}
    for r in runs:
        p = os.path.join(MIRROR, "results", r["dir"], "outputs.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                       # half-written last line of a running job
            if keep is not None and rec["image"] not in keep:
                continue
            ids = RM.trim_at_eot(rec["ids"], eot)
            runs_ = RM.run_lengths(RM.strip_layout(ids))
            d[rec["image"]] = dict(arr=RM.sample_metrics(rec["ids"], eot)["arr"],
                                   r4=RM.seq_rep_n(ids, 4),
                                   mrl=max(runs_) if runs_ else 0)
    return d


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(a, b):
    """b against a on their shared images. Positive drop = b repeats less."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 30:
        return None
    out = {"n": len(keys)}
    for k in ("arr", "r4"):
        xa = np.array([a[i][k] for i in keys], float)
        xb = np.array([b[i][k] for i in keys], float)
        d = xa - xb
        p = stats.wilcoxon(xa, xb).pvalue if np.any(d != 0) else 1.0
        lo, hi = boot_ci(d)
        out[k] = dict(mean_a=float(xa.mean()), mean_b=float(xb.mean()),
                      drop=float(d.mean()), ci=[lo, hi], p=float(p))
    ra = np.array([a[i]["mrl"] >= 2 for i in keys])
    rb = np.array([b[i]["mrl"] >= 2 for i in keys])
    n01, n10 = int((~ra & rb).sum()), int((ra & ~rb).sum())
    p = stats.binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    out["srr"] = dict(a=float(ra.mean()), b=float(rb.mean()), fixed=n10, broken=n01, p=float(p))
    out["collapse"] = dict(a=int(sum(a[i]["mrl"] >= 10 for i in keys)),
                           b=int(sum(b[i]["mrl"] >= 10 for i in keys)))
    return out


prov = json.load(open(os.path.join(HERE, "table5_provenance.json")))

# The evaluation set itself. data/coco500_final.json is the authority; when the mirror does not carry
# it, rebuild the set from a Table V run that was launched on it (any complete t5 cell covers it
# exactly), and say so in the report.
EVAL_FILE = os.path.join(MIRROR, "data", "coco500_final.json")
if os.path.exists(EVAL_FILE):
    EVAL = set(json.load(open(EVAL_FILE))["files"]); EVAL_SRC = "data/coco500_final.json"
else:
    EVAL, EVAL_SRC = None, None
    for key, c in prov["cells"].items():
        if c["complete"] and all(r["dir"].startswith(("lladav/t5/", "mmada/t5/", "lavida/t5/")) for r in c["runs"]):
            imgs = set()
            for r in c["runs"]:
                p_ = os.path.join(MIRROR, "results", r["dir"], "outputs.jsonl")
                imgs |= {json.loads(l)["image"] for l in open(p_) if l.strip().endswith("}")}
            if len(imgs) == 500:
                EVAL, EVAL_SRC = imgs, f"reconstructed from {key}"
                break
if EVAL is None or len(EVAL) != 500:
    sys.exit("evaluation set unavailable: sync data/coco500_final.json into the mirror first")

cells, report = {}, {}
for key, c in prov["cells"].items():
    model = key.split("|")[0]
    if c["runs"]:
        cells[key] = load_cell(c["runs"], EOT[model], keep=EVAL)

for model in ["LLaDA-V", "MMaDA", "LaViDa"]:
    for L in LENS:
        for lo, hi in LADDERS:
            a, b = cells.get(f"{model}|{lo}|{L}"), cells.get(f"{model}|{hi}|{L}")
            if not a or not b:
                continue
            r = compare(a, b)
            if r:
                report[f"{model}|L{L}|{lo} -> {hi}"] = r

json.dump(report, open(os.path.join(HERE, "stats_pairs.json"), "w"), indent=1)

md = ["# Paired statistics behind Section VI-B", "",
      f"Generated by `stats_pairs.py` from the mirror (cells built {prov['built']}).",
      f"Evaluation set: {EVAL_SRC}, {len(EVAL)} images. Every comparison is restricted to it.",
      "Each row compares two configurations on the images both of them produced.",
      "ARR and seq-rep-4 carry the paired Wilcoxon p and a bootstrap 95% interval on the mean drop.",
      "SRR fixed/broken count the responses that stopped/started repeating, tested with McNemar's exact test.",
      "", "| Comparison | n | ARR a→b (drop [95% CI], p) | seq-rep-4 a→b (p) | SRR a→b (fixed/broken, p) | collapsed a→b |",
      "|---|---|---|---|---|---|"]
for k, r in report.items():
    arr, r4, srr = r["arr"], r["r4"], r["srr"]
    md.append(
        f"| {k} | {r['n']} | {100*arr['mean_a']:.2f} → {100*arr['mean_b']:.2f} "
        f"({100*arr['drop']:+.2f} [{100*arr['ci'][0]:.2f}, {100*arr['ci'][1]:.2f}], p={arr['p']:.1e}) "
        f"| {r4['mean_a']:.3f} → {r4['mean_b']:.3f} (p={r4['p']:.1e}) "
        f"| {100*srr['a']:.1f} → {100*srr['b']:.1f} ({srr['fixed']}/{srr['broken']}, p={srr['p']:.1e}) "
        f"| {r['collapse']['a']} → {r['collapse']['b']} |")
md.append("")
open(os.path.join(HERE, "STATS.md"), "w").write("\n".join(md))
print("STATS.md written:", len(report), "comparisons")
