#!/usr/bin/env python
"""Panels (b), (c), (d) of the merged F1 figure from the per-position table of f1_positions100.py.

Differences from plot_f1_panels_separate.py (the 20-image version):
  * one image set for all three panels: the images whose cached response repeats (content-token criterion);
  * every interval resamples IMAGES (cluster bootstrap), not positions;
  * panel (b) keeps the ratio annotation (user's choice, 2026-09-28). On 94 images the ratios are
    3.1x [2.5, 3.9], 2.8x [2.2, 3.7] and 7.6x [3.3, 40.3] (95% intervals over images): the lower bounds
    are solid, the upper bound of the deep band is not, so the text reports the gap and its interval too.
usage: plot_f1_panels100.py <positions.npz> <outdir>     Run with /opt/anaconda3/bin/python.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})
SRC, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
z = np.load(SRC, allow_pickle=True); R = z["data"]; c = {k: i for i, k in enumerate(z["cols"])}
img = R[:, c["img"]].astype(int); reg = R[:, c["reg_con"]] == 1
K = [i for i in np.unique(img) if reg[img == i].any() and (~reg[img == i]).any()]
sel = np.isin(img, K); P = R[sel]; pim = img[sel]; preg = reg[sel]
idx = {i: np.where(pim == i)[0] for i in K}
BANDS = ["shallow", "mid", "deep"]
BAND_LABEL = {"shallow": "Shallow\n(L1–8)", "mid": "Middle\n(L9–24)", "deep": "Deep\n(L25–32)"}
RED, BLUE, ORANGE, GRAY = "#C44E52", "#8FA9C9", "#DD8452", "#666666"
FS = dict(tick=10.5, label=12, legend=10.5, annot=11.0, small=9.5)
SIZE = (3.6, 2.9)
rng = np.random.default_rng(0)
DRAWS = [np.concatenate([idx[i] for i in rng.choice(K, len(K))]) for _ in range(4000)]

def ci(d, mask):
    """mean over the masked positions, with an interval that resamples images"""
    m = d[mask].mean()
    bs = np.array([d[r][mask[r]].mean() if mask[r].any() else np.nan for r in DRAWS])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
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

D = {b: P[:, c["w5c_" + b]] - P[:, c["w5b_" + b]] for b in BANDS}

# ---------------- (b) regional ----------------
fig, ax = plt.subplots(figsize=SIZE); w = 0.34; lows = []
for j, band in enumerate(BANDS):
    mr, lr, hr = ci(D[band], preg); mf, lf, hf = ci(D[band], ~preg); lows.append(mr - lr)
    ax.bar(j - w/2, mr, w, color=RED, edgecolor="black", linewidth=0.6, zorder=3, label="Repeat region" if j == 0 else None)
    ax.bar(j + w/2, mf, w, color=BLUE, edgecolor="black", linewidth=0.6, zorder=3, label="Distant region" if j == 0 else None)
    ax.errorbar([j - w/2, j + w/2], [mr, mf], yerr=[[lr, lf], [hr, hf]], fmt="none", ecolor="black",
                elinewidth=0.9, capsize=2.8, zorder=4)
    ax.annotate(r"$%.1f\times$" % (mr / mf), (j, 0.004), ha="center", va="bottom", fontsize=FS["annot"] + 0.5,
                color=RED, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels([BAND_LABEL[b] for b in BANDS], fontsize=FS["tick"])
ax.set_ylabel(r"$\Delta w_{5}$  (cached $-$ vanilla)", fontsize=FS["label"])
ax.set_ylim(min(lows) * 1.12, 0.017)
ax.legend(fontsize=FS["legend"], frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.02),
          handlelength=1.2, handletextpad=0.5)
finish(ax); save(fig, "b_regional")

# ---------------- (c) trajectory ----------------
fig, ax = plt.subplots(figsize=SIZE)
BINS = [(0, 32), (32, 64), (64, 96), (96, 128)]; st = P[:, c["step"]]
STYLE = {"shallow": (BLUE, "o", "Shallow"), "mid": (ORANGE, "^", "Middle"), "deep": (RED, "s", "Deep")}
for band in BANDS:
    ms, los, his = zip(*[ci(D[band], (st >= lo) & (st < hi)) for lo, hi in BINS])
    col, mk, lab = STYLE[band]
    ax.errorbar(range(4), ms, yerr=[los, his], color=col, marker=mk, markersize=4.8, linewidth=2.0,
                capsize=2.8, elinewidth=0.9, label=lab, zorder=4 if band == "deep" else 3)
ax.set_xticks(range(4)); ax.set_xticklabels(["1st", "2nd", "3rd", "4th"], fontsize=FS["tick"])
ax.set_xlabel("Quarter of the decoding trajectory", fontsize=FS["label"])
ax.set_ylabel(r"$\Delta w_{5}$", fontsize=FS["label"])
ax.legend(fontsize=FS["legend"], frameon=False, loc="lower left", handlelength=1.6)
finish(ax); save(fig, "c_trajectory")

# ---------------- (d) missed anchors, split by region ----------------
fig, ax = plt.subplots(figsize=SIZE)
late, dw = P[:, c["late"]], D["deep"]
groups = [late == 0, late == 1, late == 2, late == 3, late >= 4]; lows = []
for mask, color, mk, lab in [(preg, RED, "s", "Repeat region (%.2f missed on avg.)" % late[preg].mean()),
                             (~preg, BLUE, "o", "Distant region (%.2f missed on avg.)" % late[~preg].mean())]:
    ms, los, his = zip(*[ci(dw, g & mask) for g in groups]); lows += [m - l for m, l in zip(ms, los)]
    ax.errorbar(range(5), ms, yerr=[los, his], color=color, marker=mk, markersize=5.0, linewidth=2.0,
                capsize=2.8, elinewidth=0.9, label=lab, zorder=4)
ax.plot(range(5), [dw[g].mean() for g in groups], color=GRAY, linestyle="--", linewidth=1.3,
        label="All positions", zorder=3)
ax.set_xticks(range(5)); ax.set_xticklabels(["0", "1", "2", "3", r"$\geq$4"], fontsize=FS["tick"])
ax.set_xlabel("Missed anchors", fontsize=FS["label"])
ax.set_ylabel(r"$\Delta w_{5}$  (deep band)", fontsize=FS["label"])
ax.set_ylim(min(lows) * 1.45, 0.03)
ax.legend(fontsize=FS["small"], frameon=False, loc="lower left", handlelength=1.6)
finish(ax); save(fig, "d_dose")
print("images %d (with a repeat), positions %d -> panels in %s" % (len(K), len(P), OUT))
