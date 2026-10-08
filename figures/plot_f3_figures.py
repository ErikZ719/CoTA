#!/usr/bin/env python
"""F3 figures for SS4.4 (and the intro exhibit it replaces).

Fig A  fig-f3-entropy-curves.pdf  -- replaces entropy_summary.pdf (intro Fig.4):
       per-layer decision-step entropy, three groups over 60 COCO images
       (baseline-normal / cache-normal / cache-repeat), bootstrap 95% CI bands,
       deep window L26-30 shaded. Cache-normal hugs baseline; cache-repeat
       fails to converge in deep layers.
Fig B  fig-f3-quartile.pdf  -- new SS4.4 exhibit: deep-layer entropy (L26-30)
       by quarter of the decoding trajectory, repeat vs normal, points + 95% CI,
       significance markers (ns/***) from f3_by_decoding_step.json.

Data: results/F3/f3_curves_multisample.json, f3_quartile_rows.json (server),
      results/independence/f3_by_decoding_step.json (p-values).
Style: STIX serif, house palette. Run with /opt/anaconda3/bin/python.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.7,
})
BASE = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results"
RED, BLUE, ORANGE, DARK = "#C44E52", "#8FA9C9", "#DD8452", "#444444"

# ---------------- Fig A: representative per-position curves ----------------
# Protocol = the ORIGINAL Fig.3 protocol, verified point-by-point (2026-09-08):
# cross-layer entropy at the FINAL decoding step (t=127), reference sample.
# Blue: vanilla normal positions (collapse inside the deep window).
# Red: formal repeat positions of the cached run (persistently high to L31).
E = np.load(BASE + "/F3/dllm_cache_G128_test.npz")["entropy_bits"].astype(np.float32)
Eb = np.load(BASE + "/F3/baseline_G128_test.npz")["entropy_bits"].astype(np.float32)
T = 127
REP_IDS = [46, 89, 113, 101]     # cached run, formal criterion ("the","the","the","the")
NORM_IDS = [36, 106, 23, 57]     # vanilla run (incl. 36 " of", the Fig.3(a) case)
layers = np.arange(1, 33)
fig, ax = plt.subplots(figsize=(4.7, 3.4))
ax.axvspan(26, 30, color="#EEEEEE", zorder=0)
ax.annotate("deep window\nL26\u201330", (25.4, 4.3), ha="right", fontsize=9.5, color="#777777")
for i in NORM_IDS:
    ax.plot(layers, Eb[T, 1:33, i], color=BLUE, lw=1.4, alpha=0.95,
            label="Vanilla (normal)" if i == NORM_IDS[0] else None, zorder=2)
for i in REP_IDS:
    ax.plot(layers, E[T, 1:33, i], color=RED, lw=1.9,
            label="$+$dLLM-Cache (repeat)" if i == REP_IDS[0] else None, zorder=3)
ax.set_xlabel("Layer $\\ell$", fontsize=13)
ax.set_ylabel("Entropy (bits)", fontsize=13)
ax.set_xlim(1, 32); ax.set_ylim(0, 16.5)
ax.tick_params(labelsize=11)
ax.legend(fontsize=10, frameon=False, loc="lower left", handlelength=1.4, borderaxespad=0.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(BASE + "/F3/fig-f3-entropy-curves.pdf", bbox_inches="tight")
fig.savefig(BASE + "/F3/fig-f3-entropy-curves.png", dpi=200, bbox_inches="tight")

# ---------------- Fig B: deep entropy by trajectory quarter ----------------
rows = json.load(open(BASE + "/F3/f3_quartile_rows.json"))
P = json.load(open(BASE + "/independence/f3_by_decoding_step.json"))
QLAB = ["1st", "2nd", "3rd", "4th"]
QKEY = ["0-31", "32-63", "64-95", "96-127"]
rng = np.random.default_rng(0)
def boot_ci(x, n=5000):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return m, m - lo, hi - m
fig2, bx = plt.subplots(figsize=(7.2, 2.75))
for rep, c, lab, mk in ((False, BLUE, "Normal", "o"), (True, RED, "Repeat", "s")):
    ms, los, his = [], [], []
    for q in range(4):
        vals = [r["deep"] for r in rows if r["q"] == q and r["rep"] == rep]
        m, l, h = boot_ci(vals)
        ms.append(m); los.append(l); his.append(h)
    bx.errorbar(range(4), ms, yerr=[los, his], color=c, marker=mk, markersize=5.5,
                linewidth=1.8, capsize=3, elinewidth=0.9, label=lab, zorder=3)
tops = []
for rep in (False, True):
    for q in range(4):
        vals = [r["deep"] for r in rows if r["q"] == q and r["rep"] == rep]
        m, l, h = boot_ci(vals)
        if rep or not tops or True:
            pass
for q in range(4):
    hi_top = 0.0
    for rep in (False, True):
        vals = [r["deep"] for r in rows if r["q"] == q and r["rep"] == rep]
        m, l, h = boot_ci(vals)
        hi_top = max(hi_top, m + h)
    pv = P[QKEY[q]]["p"]; nrep = P[QKEY[q]]["n_rep"]
    star = "n.s." if pv >= 0.05 else ("$*$$*$$*$" if pv < 1e-3 else "$*$$*$")
    bx.annotate(star, (q, hi_top + 0.22), ha="center", fontsize=11, color=DARK)
    bx.annotate("$n_{\\mathrm{rep}}$=%d" % nrep, (q, 9.15), ha="center", fontsize=9.5, color="#777777")
bx.set_xticks(range(4)); bx.set_xticklabels(QLAB, fontsize=12)
bx.set_xlabel("Quarter of the decoding trajectory", fontsize=13)
bx.set_ylabel("Deep-layer entropy (bits)", fontsize=13)
bx.set_ylim(8.8, 14.0); bx.set_xlim(-0.4, 3.4)
bx.tick_params(axis="y", labelsize=11)
bx.legend(fontsize=11, frameon=False, loc="upper left", handlelength=1.6)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
fig2.savefig(BASE + "/F3/fig-f3-quartile.pdf", bbox_inches="tight")
fig2.savefig(BASE + "/F3/fig-f3-quartile.png", dpi=200, bbox_inches="tight")
print("saved both F3 figures")
