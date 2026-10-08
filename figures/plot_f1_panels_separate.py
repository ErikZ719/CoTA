#!/usr/bin/env python
"""Panels (b), (c), (d) of the merged F1 figure as three separate files, for assembly in
PowerPoint (fig-f1-combined.pptx). No panel letters or titles: the captions are added in
the slide so that every label in the figure shares one font.

  b_regional.{pdf,png}   mean d_w5 (cached - vanilla), repeat vs distant region, per band
  c_trajectory.{pdf,png} mean d_w5 per band across decoding-step quartiles
  d_dose.{pdf,png}       deep-band d_w5 vs neighbours committed since last recompute

Data: results/F1/f1_multisample_raw.npz, results/F1/f1_dose.npz. Each panel is 3.6 x 2.9 in
so that at the slide scale (three across 16.4 cm) 11 pt here renders as ~7.5 pt.
Run with /opt/anaconda3/bin/python.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})
ROOT = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/F1"
OUT = f"{ROOT}/panels"
import os; os.makedirs(OUT, exist_ok=True)
D = np.load(f"{ROOT}/f1_multisample_raw.npz")
Z = np.load(f"{ROOT}/f1_dose.npz")
BANDS = ["shallow", "mid", "deep"]
BAND_LABEL = {"shallow": "Shallow\n(L1–8)", "mid": "Middle\n(L9–24)", "deep": "Deep\n(L25–32)"}
RED, BLUE, ORANGE, GRAY = "#C44E52", "#8FA9C9", "#DD8452", "#666666"
FS = dict(tick=10.5, label=12, legend=10.5, annot=11.5, small=9.5)
SIZE = (3.6, 2.9)

rng = np.random.default_rng(0)
def boot_ci(x, n=10000):
    m = x.mean()
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return m, m - lo, hi - m

def finish(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=FS["tick"])

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------- (b) regional ----------------
fig, ax = plt.subplots(figsize=SIZE)
w = 0.34
for j, band in enumerate(BANDS):
    reg, far = D[band + "__reg"], D[band + "__far"]
    mr, lr, hr = boot_ci(reg); mf, lf, hf = boot_ci(far)
    ax.bar(j - w/2, mr, w, color=RED, edgecolor="black", linewidth=0.6, zorder=3,
           label="Repeat region" if j == 0 else None)
    ax.bar(j + w/2, mf, w, color=BLUE, edgecolor="black", linewidth=0.6, zorder=3,
           label="Distant region" if j == 0 else None)
    ax.errorbar([j - w/2, j + w/2], [mr, mf], yerr=[[lr, lf], [hr, hf]],
                fmt="none", ecolor="black", elinewidth=0.9, capsize=2.8, zorder=4)
    ax.annotate(r"$%.1f\times$" % (mr / mf), (j, 0.004), ha="center", va="bottom",
                fontsize=FS["annot"], color=RED, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([BAND_LABEL[b] for b in BANDS], fontsize=FS["tick"])
ax.set_ylabel(r"$\Delta w_{5}$  (cached $-$ vanilla)", fontsize=FS["label"])
ax.set_ylim(-0.088, 0.017)
ax.legend(fontsize=FS["legend"], frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02),
          handlelength=1.2, handletextpad=0.5)
finish(ax); save(fig, "b_regional")

# ---------------- (c) trajectory ----------------
fig, ax = plt.subplots(figsize=SIZE)
BINS = [(0, 32), (32, 64), (64, 96), (96, 128)]
STYLE = {"shallow": (BLUE, "o", "Shallow"), "mid": (ORANGE, "^", "Middle"), "deep": (RED, "s", "Deep")}
for band in BANDS:
    st, d = D[band + "__step"], D[band + "__d"]
    ms, los, his = [], [], []
    for lo, hi in BINS:
        m = (st >= lo) & (st < hi)
        mm, l, h = boot_ci(d[m]); ms.append(mm); los.append(l); his.append(h)
    c, mk, lab = STYLE[band]
    ax.errorbar(range(4), ms, yerr=[los, his], color=c, marker=mk, markersize=4.8,
                linewidth=2.0, capsize=2.8, elinewidth=0.9, label=lab,
                zorder=4 if band == "deep" else 3)
ax.set_xticks(range(4)); ax.set_xticklabels(["1st", "2nd", "3rd", "4th"], fontsize=FS["tick"])
ax.set_xlabel("Quarter of the decoding trajectory", fontsize=FS["label"])
ax.set_ylabel(r"$\Delta w_{5}$", fontsize=FS["label"])
ax.legend(fontsize=FS["legend"], frameon=False, loc="lower left", handlelength=1.6)
finish(ax); save(fig, "c_trajectory")

# ---------------- (d) dose, split by region ----------------
fig, ax = plt.subplots(figsize=SIZE)
late, dw, rep = Z["late"], Z["dw5"], Z["repeat"]
groups = [late == 0, late == 1, late == 2, late == 3, late >= 4]
for mask, color, mk, lab in [(rep == 1, RED, "s", "Repeat region (%.2f missed on avg.)" % late[rep == 1].mean()),
                             (rep == 0, BLUE, "o", "Distant region (%.2f missed on avg.)" % late[rep == 0].mean())]:
    ms, los, his = zip(*[boot_ci(dw[g & mask]) for g in groups])
    ax.errorbar(range(5), ms, yerr=[los, his], color=color, marker=mk, markersize=5.0,
                linewidth=2.0, capsize=2.8, elinewidth=0.9, label=lab, zorder=4)
ax.plot(range(5), [dw[g].mean() for g in groups], color=GRAY, linestyle="--", linewidth=1.3,
        label="All positions", zorder=3)
ax.set_xticks(range(5)); ax.set_xticklabels(["0", "1", "2", "3", r"$\geq$4"], fontsize=FS["tick"])
ax.set_xlabel("Anchors missed since last recompute", fontsize=FS["label"])
ax.set_ylabel(r"$\Delta w_{5}$  (deep band)", fontsize=FS["label"])
ax.set_ylim(-0.165, 0.03)
ax.legend(fontsize=FS["small"], frameon=False, loc="lower left", handlelength=1.6)
finish(ax); save(fig, "d_dose")
print("saved panels b/c/d to", OUT)
