#!/usr/bin/env python
"""F3 figure on the 100-image sample (94 images that repeat under dLLM-Cache), single column, two panels.

(a) per-layer entropy at the end of generation, mean over positions: vanilla, cached normal, cached repeat
    (bands: 95% intervals over images); deep window L26-30 shaded
(b) repetition rate by decile of the context score (mean deep-window entropy of the committed neighbours
    within +-5 at the decode moment); error bars: 95% intervals over images; dashed: overall rate

Designed at print size (8.8 cm wide): font sizes below are the sizes on paper.
Data: results/F3/f3_100_positions.npz, results/F3/f3_100_stats.json. Run with /opt/anaconda3/bin/python.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
})
B = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/F3/"
z = np.load(B + "f3_100_positions.npz", allow_pickle=True); D = z["data"]; cols = list(z["cols"]); c = lambda k: D[:, cols.index(k)]
img = c("img").astype(int); rep = c("rep").astype(int)
FIN = D[:, cols.index("fin_00"):cols.index("fin_32") + 1]; BFIN = D[:, cols.index("bfin_00"):cols.index("bfin_32") + 1]
has = np.array([rep[img == k].sum() > 0 for k in range(100)]); ids = np.where(has)[0]; keep = has[img]
S = json.load(open(B + "f3_100_stats.json"))
rng = np.random.default_rng(0)
idx = {k: np.where(img == k)[0] for k in ids}
def band(M, mask):
    v = []
    for _ in range(1000):
        m = np.concatenate([idx[k] for k in rng.choice(ids, len(ids))]); m = m[mask[m]]
        v.append(M[m].mean(0))
    return np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0)
RED, BLUE, DARK, GREY, VAN = "#C44E52", "#8FA9C9", "#333333", "#6B6B6B", "#5C7FA8"
FS = dict(label=7.5, tick=6.8, key=6.6, cap=8.5, ann=6.6)
CM = 1 / 2.54
W, Ht = 8.8 * CM, 3.95 * CM
fig = plt.figure(figsize=(W, Ht))
y0, h = 0.50, 0.90
xa, wa = 0.33, 1.20
xb, wb = xa + wa + 0.42, 1.43
ax = fig.add_axes([xa / W, y0 / Ht, wa / W, h / Ht]); bx = fig.add_axes([xb / W, y0 / Ht, wb / W, h / Ht])

# ---- (a)
L = np.arange(33); sl = slice(20, 33)
groups = (("Vanilla", BFIN, keep, VAN, (0, (3, 2)), 0.9), ("Cached, normal", FIN, keep & (rep == 0), BLUE, "-", 1.2), ("Cached, repeat", FIN, keep & (rep == 1), RED, "-", 1.2))
ax.axvspan(25.5, 30.5, color="#ECECEC", zorder=0, linewidth=0)
for name, M, mask, col, ls, lw in groups:
    mu = M[mask].mean(0); lo, hi = band(M, mask)
    ax.fill_between(L[sl], lo[sl], hi[sl], color=col, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(L[sl], mu[sl], color=col, linestyle=ls, linewidth=lw, zorder=3, label=name)
ax.set_xlim(20, 32); ax.set_ylim(0, 16.5); ax.set_xticks([20, 24, 28, 32]); ax.set_yticks([0, 5, 10, 15])
ax.set_xlabel(r"Layer $\ell$", fontsize=FS["label"], labelpad=1.5); ax.set_ylabel("Entropy (bits)", fontsize=FS["label"], labelpad=2)
ax.text(28.0, 16.1, "deep window", fontsize=FS["ann"] - 0.4, color=GREY, ha="center", va="top")
ax.axhline(7.57, color=GREY, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)          # half of the plateau (15.1 bits)
ax.text(20.25, 7.95, "half of plateau", fontsize=FS["ann"] - 0.4, color=GREY, ha="left", va="bottom")
ax.legend(fontsize=FS["key"], frameon=False, loc="lower left", bbox_to_anchor=(-0.03, -0.03), handlelength=1.5, handletextpad=0.4, labelspacing=0.15, borderpad=0.2)

# ---- (b)
r = np.array(S["risk"]["rate"]) * 100; lo = np.array(S["risk"]["lo"]) * 100; hi = np.array(S["risk"]["hi"]) * 100; base = S["risk"]["base"] * 100
x = np.arange(1, 11); colb = [BLUE] * 8 + [RED] * 2
bx.bar(x, r, 0.78, color=colb, edgecolor=DARK, linewidth=0.35, zorder=3)
bx.errorbar(x, r, yerr=[r - lo, hi - r], fmt="none", ecolor=DARK, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=4)
bx.axhline(base, color=GREY, linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)
bx.text(0.55, base + 1.2, r"overall $%.1f\%%$" % base, fontsize=FS["ann"], color=GREY, ha="left", va="bottom")
ratio = S["risk"]["top2"] / S["risk"]["rest8"]
bx.plot([8.62, 8.62, 10.38, 10.38], [hi[9] + 1.2, hi[9] + 2.4, hi[9] + 2.4, hi[9] + 1.2], color=RED, linewidth=0.6, zorder=4, clip_on=False)
bx.text(9.35, hi[9] + 2.9, r"$%.1f\times$ the rest" % ratio, fontsize=FS["ann"], color=RED, ha="center", va="bottom")
bx.set_xlim(0.35, 10.65); bx.set_ylim(0, 49); bx.set_xticks(x); bx.set_yticks([0, 10, 20, 30, 40])
bx.set_xlabel("Decile of context entropy", fontsize=FS["label"], labelpad=1.5); bx.set_ylabel("Repetition rate (%)", fontsize=FS["label"], labelpad=2)
for a in (ax, bx):
    a.tick_params(labelsize=FS["tick"], pad=1.5)
    for s in ("top", "right"): a.spines[s].set_visible(False)
for xx, ww, s in ((xa, wa, "(a) End of generation"), (xb, wb, "(b) Decode moment")):
    fig.text((xx + ww / 2) / W, 0.075 / Ht, s, ha="center", va="center", fontsize=FS["cap"])
out = sys.argv[1] if len(sys.argv) > 1 else B + "fig-f3-combined-100"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("saved %s  %.2f x %.2f cm ; ratio %.2f ; base %.2f" % (out, W / CM, Ht / CM, ratio, base))
