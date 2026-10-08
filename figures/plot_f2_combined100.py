#!/usr/bin/env python
"""Merged F2 figure (Fig. 7 of the paper, full width, 17 cm), v2: four equal square panels on an even pitch.

(a) staleness at the decode moment, 94 of 100 COCO images (946 repeat vs 10,992 normal positions)
(b) staleness of each position up to its decode, reference sample
(c) KV states as read at layer 22 (attention received by each position), vanilla, reference sample
(d) the same under dLLM-Cache

Layout rules
  * the four axes boxes have the same size (square) and the same distance from one another
  * every panel carries its key in a header strip of the same height: legend (a), colour bars (b), (c)+(d)
  * panel titles sit below, on one baseline, as in Fig. 6
  * heat maps are upsampled by an integer factor, so that a PDF viewer cannot blur them
Designed at print size: font sizes below are the sizes on paper.
Data: F2/delta_at_decode_100.json, F2/delta_field_reference.json, F2/received_attention_layer22.json
Run with /opt/anaconda3/bin/python.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
})
BASE = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/F2"
J = json.load(open(BASE + "/delta_field_reference.json"))
F = np.array(J["field"]); us = {int(k): v for k, v in J["decode_step"].items()}; reps = sorted(J["repeats"])
T, I = F.shape
H = json.load(open(BASE + "/delta_at_decode_100.json"))
D = json.load(open(BASE + "/received_attention_layer22.json"))
A0 = np.array(D["baseline"]); A1 = np.array(D["dllm_cache"])

RED, BLUE, DARK, GREY = "#C44E52", "#8FA9C9", "#333333", "#6B6B6B"
FS = dict(label=8, tick=7, key=7, cap=9, ann=7)
UP = 8                                                     # integer upsampling of the heat maps
CM = 1 / 2.54
W = 17.0 * CM
BOX = 1.25                                                 # side of every axes box, inches
RIGHT = 0.05
GUT = (W - RIGHT - 4 * BOX) / 4                            # gutter left of every box
Y0 = 0.475                                                 # bottom of the boxes
HEAD = 0.385                                               # header strip above the boxes
Ht = Y0 + BOX + HEAD
X = [GUT + k * (BOX + GUT) for k in range(4)]
fig = plt.figure(figsize=(W, Ht))
ax_a, ax_b, ax_c, ax_d = [fig.add_axes([x / W, Y0 / Ht, BOX / W, BOX / Ht]) for x in X]
KEY_Y = Y0 + BOX + 0.165                                   # baseline of the keys in the header strip
BAR_H = 0.055

def up(a):
    return np.repeat(np.repeat(a, UP, axis=0), UP, axis=1)

# ---- (a) staleness at the decode moment
R = np.array(H["repeat"]); N = np.array(H["normal"]); vals = np.arange(7); w = 0.38
ax_a.bar(vals - w / 2, [(N == v).mean() for v in vals], w, color=BLUE, edgecolor=DARK, linewidth=0.35, zorder=3)
ax_a.bar(vals + w / 2, [(R == v).mean() for v in vals], w, color=RED, edgecolor=DARK, linewidth=0.35, zorder=3)
ax_a.axvline(3.5, color=GREY, linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)
ax_a.text(3.62, 0.485, r"$\Delta{>}\tau$", fontsize=FS["ann"], color=GREY, ha="left", va="top")
ax_a.text(6.45, 0.405, r"Repeat $%.0f\%%$" % (100 * (R > 3).mean()), fontsize=FS["ann"], color=RED, ha="right", va="top")
ax_a.text(6.45, 0.345, r"Normal $%.0f\%%$" % (100 * (N > 3).mean()), fontsize=FS["ann"], color="#5C7FA8", ha="right", va="top")
ax_a.set_xlabel(r"$\Delta$ at the decode moment", fontsize=FS["label"], labelpad=2)
ax_a.set_ylabel("Fraction of positions", fontsize=FS["label"], labelpad=2.5)
ax_a.set_xticks(vals); ax_a.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax_a.set_ylim(0, 0.52); ax_a.set_xlim(-0.65, 6.65)
ax_a.yaxis.grid(True, color="#E4E4E4", linewidth=0.45, zorder=0); ax_a.set_axisbelow(True)
fig.legend(handles=[Patch(facecolor=BLUE, edgecolor=DARK, linewidth=0.35, label="Normal"),
                    Patch(facecolor=RED, edgecolor=DARK, linewidth=0.35, label="Repeat")],
           fontsize=FS["key"], frameon=False, ncol=2, loc="center",
           bbox_to_anchor=((X[0] + BOX / 2) / W, (KEY_Y + BAR_H / 2) / Ht),
           handlelength=1.1, handleheight=0.75, handletextpad=0.4, columnspacing=1.4, borderpad=0)

# ---- (b) staleness up to each decode
ramp = LinearSegmentedColormap.from_list(
    "stale", ["#FFFFFF", "#E3E9F1", "#C5D2E2", "#9FB4CD", "#7391B2", "#4E6E93", "#33506F"], N=7)
norm = BoundaryNorm(np.arange(-0.5, 7.5, 1), ramp.N)
rgba = ramp(norm(F)); post = np.zeros_like(F, dtype=bool)
for i, t in us.items(): post[t + 1:, i] = True
rgba[post] = 1.0 - (1.0 - rgba[post]) * 0.12
ax_b.imshow(up(rgba), aspect="auto", origin="upper", extent=[-0.5, I - 0.5, T - 0.5, -0.5], interpolation="nearest")
nx = [i for i in us if i not in reps]; rx = [i for i in us if i in reps]
ax_b.scatter(nx, [us[i] for i in nx], s=1.3, c=DARK, marker="o", linewidths=0, alpha=0.85, zorder=3)
ax_b.scatter(rx, [us[i] for i in rx], s=15, facecolor=RED, edgecolor="white", linewidths=0.55, marker="o", zorder=4)
hnd = [Line2D([], [], marker="o", color="none", markerfacecolor=DARK, markeredgewidth=0, markersize=2.4, label="Normal"),
       Line2D([], [], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markeredgewidth=0.5, markersize=4.3, label="Repeat")]
ax_b.legend(handles=hnd, fontsize=FS["key"], frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.01),
            handletextpad=0.0, labelspacing=0.2, borderpad=0.25, handlelength=1.3)

# ---- (c), (d) received attention
ramp2 = LinearSegmentedColormap.from_list(
    "attn", ["#FFFFFF", "#DDE5EF", "#AFC3D9", "#7B9BBB", "#4E6E93", "#2C4763", "#16283C"])
vmax = 8.0
for ax, A in ((ax_c, A0), (ax_d, A1)):
    im = ax.imshow(up(A * 1e3), aspect="auto", cmap=ramp2, vmin=0, vmax=vmax, origin="upper",
                   extent=[-0.5, 127.5, 127.5, -0.5], interpolation="nearest")

for ax in (ax_b, ax_c, ax_d):
    ax.set_xlim(-0.5, 127.5); ax.set_ylim(127.5, -0.5)
    ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel("Suffix position $i$", fontsize=FS["label"], labelpad=2)
    ax.set_ylabel("Decoding step $t$", fontsize=FS["label"], labelpad=2.5)
for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.tick_params(labelsize=FS["tick"], pad=1.8)
    for s in ax.spines.values(): s.set_linewidth(0.6); s.set_color(DARK)
for ax in (ax_b, ax_d):                                    # repeat positions of the cached run
    ax.plot(reps, [1.0 + 0.042] * len(reps), transform=ax.get_xaxis_transform(), linestyle="none", marker="v",
            markersize=3.3, markerfacecolor=RED, markeredgewidth=0, clip_on=False, zorder=5)

# ---- keys of (b) and (c)+(d): horizontal colour bars in the header strip
def key_bar(x_left, width, mappable, ticks, label, ticklabels=None):
    cax = fig.add_axes([x_left / W, KEY_Y / Ht, width / W, BAR_H / Ht])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=FS["key"] - 0.5, pad=0.8, length=1.6, width=0.4)
    if ticklabels: cb.ax.set_xticklabels(ticklabels)
    cb.outline.set_linewidth(0.4)
    fig.text((x_left - 0.05) / W, (KEY_Y + BAR_H / 2) / Ht, label, fontsize=FS["key"], ha="right", va="center")
bw = 0.56
key_bar(X[1] + 0.61, bw, plt.cm.ScalarMappable(cmap=ramp, norm=norm), [0, 2, 4, 6], r"Staleness $\Delta_i^{t}$")
bw2 = 1.05; xc = X[2] + (X[3] + BOX - X[2]) / 2
lab = r"Attention received ($\times10^{-3}$)"
key_bar(xc + 0.03, bw2, plt.cm.ScalarMappable(cmap=ramp2, norm=Normalize(0, vmax)), [0, 2, 4, 6, 8], lab,
        ticklabels=["0", "2", "4", "6", r"$\geq$8"])

for x, s in zip(X, ("(a) Staleness at decode", "(b) Staleness up to decode", "(c) KV states, vanilla", "(d) KV states, cached")):
    fig.text((x + BOX / 2) / W, 0.075 / Ht, s, ha="center", va="center", fontsize=FS["cap"])

fig.savefig(BASE + "/fig-f2-combined-100.pdf", dpi=600); fig.savefig(BASE + "/fig-f2-combined-100.png", dpi=300)
print("saved fig-f2-combined-100  %.2f x %.2f cm ; box %.2f in, gutter %.3f in" % (W / CM, Ht / CM, BOX, GUT))
