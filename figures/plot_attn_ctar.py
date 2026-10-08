#!/usr/bin/env python
"""Attention at the decode moment, before and after CTAR (full width, 17 cm): four equal square panels, as Fig. 7.

(a) vanilla   (b) dLLM-Cache   (c) + CTAR     the image of Fig. 10(a). Row i is the attention of position i over the suffix
                                               at the step at which i is committed, mean over heads and layers 25-32, so
                                               that the three maps are comparable although they commit in different orders
(d) mean attention against the offset from the position, at the positions that repeat under the cache, 100 images
Measure: as in Section IV-B (scripts/f1_positions100.py): every row is normalised over the 128 suffix keys, and the
         anchoring w5 is the share that falls within +-5 of the position, the position included.

Data: attn_decode/caseA_118929.npz and attn_decode/seed0_100.npz (analysis/attn_decode_rows{,100}.py on the server, from
      the recordings information_flow/attn_case and attn_multi100; arms baseline, dllm_cache, ctar_th)
Designed at print size: font sizes below are the sizes on paper. Run with /opt/anaconda3/bin/python.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
})
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "attn_decode")
C = np.load(os.path.join(D, "caseA_118929.npz")); Z = np.load(os.path.join(D, "seed0_100.npz"))
M, EOT, LAYOUT, W5 = 128, (126081, 126348), (198, 220, 197, 256, 262144), 5
ZOOM = (int(sys.argv[sys.argv.index("--zoom") + 1]), int(sys.argv[sys.argv.index("--zoom") + 2])) if "--zoom" in sys.argv else None
ARMS = [("baseline", "Vanilla"), ("dllm_cache", "dLLM-Cache"), ("ctar_th", "+CTAR")]


def rownorm(A):
    """rows normalised over the suffix keys (the measure of Section IV-B)"""
    A = np.nan_to_num(np.asarray(A, np.float64)); s = A.sum(axis=-1, keepdims=True)
    return np.where(s > 0, A / np.where(s > 0, s, 1), np.nan)


def repeats(tok):
    """positions that repeat the content token before them, up to the first end-of-text token"""
    tok = np.asarray(tok); end = np.where(np.isin(tok, EOT))[0]; end = int(end[0]) if len(end) else len(tok)
    rep = np.zeros(len(tok), bool); prev = None
    for p in range(end):
        if tok[p] in LAYOUT: continue
        if prev is not None and tok[p] == prev: rep[p] = True
        prev = tok[p]
    return rep


# ---- (d): profile at the positions that repeat under the cache, the same positions in every arm
OFF = np.arange(-15, 16)
rep100 = np.array([repeats(t) for t in Z["dllm_cache_tok"]])
prof, w5 = {}, {}
for arm, _ in ARMS:
    A = rownorm(Z[arm + "_attn"]); P = np.full((len(A), len(OFF)), np.nan); Wm = np.full(len(A), np.nan)
    for i in range(len(A)):
        ps = np.where(rep100[i])[0]
        if not len(ps): continue
        rows = np.full((len(ps), len(OFF)), np.nan)
        for k, p in enumerate(ps):
            for j, o in enumerate(OFF):
                if 0 <= p + o < M: rows[k, j] = A[i, p, p + o]
        P[i] = np.nanmean(rows, axis=0)
        Wm[i] = np.nanmean([np.nansum(A[i, p, max(0, p - W5):p + W5 + 1]) for p in ps])
    prof[arm] = np.nanmean(P, axis=0); w5[arm] = Wm
allw = {a: np.concatenate([[np.nansum(rownorm(Z[a + "_attn"][i])[p, max(0, p - W5):p + W5 + 1]) for p in np.where(rep100[i])[0]]
                           for i in range(len(rep100))]) for a, _ in ARMS}
S = dict(images_with_repeats=int((rep100.sum(1) > 0).sum()), repeat_positions=int(rep100.sum()),
         w5_mean_over_images={a: float(np.nanmean(v)) for a, v in w5.items()},
         w5_mean_over_positions={a: float(np.nanmean(v)) for a, v in allw.items()},
         case=dict(repeat_positions={a: int(repeats(C[a + "_tok"]).sum()) for a, _ in ARMS}))
print(json.dumps(S, indent=1))

RED, BLUE, DARK, GREY = "#C44E52", "#4C72B0", "#333333", "#6B6B6B"
FS = dict(label=8, tick=7, key=7, cap=9, ann=7)
UP = 8
up = lambda a: np.repeat(np.repeat(a, UP, axis=0), UP, axis=1)
rep_case = np.where(repeats(C["dllm_cache_tok"]))[0]

if "--single" in sys.argv:
    # 2026-09-30 (user): single column, the cached map and the map with CTAR only, viridis as in Figs. 2 and 6
    W = 3.5; BOX = 1.36; GUT = 0.30; X0 = 0.42; Y0 = 0.44; HEAD = 0.40
    Ht = Y0 + BOX + HEAD
    fig = plt.figure(figsize=(W, Ht))
    AX = [fig.add_axes([(X0 + k * (BOX + GUT)) / W, Y0 / Ht, BOX / W, BOX / Ht]) for k in range(2)]
    VMAX = 12.0
    for ax, (arm, name) in zip(AX, ARMS[1:]):
        A = np.nan_to_num(rownorm(C[arm + "_attn"])) * 1e2
        ax.imshow(up(A), aspect="auto", cmap="viridis", vmin=0, vmax=VMAX, origin="upper", extent=[-0.5, M - 0.5, M - 0.5, -0.5], interpolation="nearest")
        if ZOOM:
            ax.set_xlim(ZOOM[0] - 0.5, ZOOM[1] + 0.5); ax.set_ylim(ZOOM[1] + 0.5, ZOOM[0] - 0.5)
            tk = [t for t in range(0, M, 20) if ZOOM[0] <= t <= ZOOM[1]]; ax.set_xticks(tk); ax.set_yticks(tk)
        else:
            ax.set_xlim(-0.5, M - 0.5); ax.set_ylim(M - 0.5, -0.5); ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120])
        ax.set_xlabel("Suffix position $j$ (key)", fontsize=FS["label"], labelpad=2)
        ax.tick_params(labelsize=FS["tick"], pad=1.8)
        for s_ in ax.spines.values(): s_.set_linewidth(0.6); s_.set_color(DARK)
        if arm == "dllm_cache" and len(rep_case):
            rc = [r for r in rep_case if (ZOOM is None or ZOOM[0] <= r <= ZOOM[1])]
            ax.plot([0.965] * len(rc), rc, transform=ax.get_yaxis_transform(), linestyle="none", marker="<", markersize=2.6,
                    markerfacecolor=RED, markeredgewidth=0, clip_on=False, zorder=5)
    AX[0].set_ylabel("Suffix position $i$ (query)", fontsize=FS["label"], labelpad=2.5)
    AX[1].tick_params(labelleft=False)
    cax = fig.add_axes([(X0 + 0.30) / W, (Y0 + BOX + 0.10) / Ht, 1.50 / W, 0.055 / Ht])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(0, VMAX)), cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12])
    cax.xaxis.set_ticks_position("top"); cb.ax.tick_params(labelsize=FS["key"] - 0.5, pad=0.8, length=1.6, width=0.4)
    cb.ax.set_xticklabels(["0", "4", "8", r"$\geq$12"]); cb.outline.set_linewidth(0.4)
    fig.text((X0 + 0.30 + 0.75) / W, (Y0 + BOX + 0.31) / Ht, r"Share of suffix attention ($\times10^{-2}$)", fontsize=FS["key"], ha="center", va="center")
    fig.legend(handles=[Line2D([], [], marker="<", color="none", markerfacecolor=RED, markeredgewidth=0, markersize=3.6)], labels=["Repeat"],
               fontsize=FS["key"], frameon=False, loc="center left", bbox_to_anchor=((X0 + 0.30 + 1.50 + 0.12) / W, (Y0 + BOX + 0.10 + 0.028) / Ht),
               handletextpad=0.1, borderpad=0, handlelength=1.0)
    for k, s_ in enumerate(("(a) dLLM-Cache", "(b) +CTAR")):
        fig.text((X0 + k * (BOX + GUT) + BOX / 2) / W, 0.07 / Ht, s_, ha="center", va="center", fontsize=FS["cap"])
    out = os.path.join(D, "fig-attn-ctar-1col")
    fig.savefig(out + ".pdf", dpi=600); fig.savefig(out + ".png", dpi=300)
    json.dump(S, open(out + ".json", "w"), indent=1); print("->", out + ".pdf"); sys.exit(0)

CM = 1 / 2.54
W = 17.0 * CM
BOX, RIGHT = 1.25, 0.05
GUT = (W - RIGHT - 4 * BOX) / 4
Y0, HEAD = 0.475, 0.385
Ht = Y0 + BOX + HEAD
X = [GUT + k * (BOX + GUT) for k in range(4)]
fig = plt.figure(figsize=(W, Ht))
AX = [fig.add_axes([x / W, Y0 / Ht, BOX / W, BOX / Ht]) for x in X]
KEY_Y, BAR_H = Y0 + BOX + 0.165, 0.055
ramp = LinearSegmentedColormap.from_list("attn", ["#FFFFFF", "#DDE5EF", "#AFC3D9", "#7B9BBB", "#4E6E93", "#2C4763", "#16283C"])
VMAX = 12.0                                                # x 10^-2, share of the suffix attention
for ax, (arm, name) in zip(AX[:3], ARMS):
    A = np.nan_to_num(rownorm(C[arm + "_attn"])) * 1e2
    ax.imshow(up(A), aspect="auto", cmap=ramp, vmin=0, vmax=VMAX, origin="upper", extent=[-0.5, M - 0.5, M - 0.5, -0.5],
              interpolation="nearest")
    if ZOOM:
        ax.set_xlim(ZOOM[0] - 0.5, ZOOM[1] + 0.5); ax.set_ylim(ZOOM[1] + 0.5, ZOOM[0] - 0.5)
        tk = [t for t in range(0, M, 20) if ZOOM[0] <= t <= ZOOM[1]]; ax.set_xticks(tk); ax.set_yticks(tk)
    else:
        ax.set_xlim(-0.5, M - 0.5); ax.set_ylim(M - 0.5, -0.5); ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel("Suffix position $j$ (key)", fontsize=FS["label"], labelpad=2)
    ax.set_ylabel("Suffix position $i$ (query)", fontsize=FS["label"], labelpad=2.5)
    if arm == "dllm_cache" and len(rep_case):              # repeat positions of the cached run
        rc = [r for r in rep_case if (ZOOM is None or ZOOM[0] <= r <= ZOOM[1])]
        ax.plot([0.965] * len(rc), rc, transform=ax.get_yaxis_transform(), linestyle="none", marker="<",
                markersize=2.6, markerfacecolor=RED, markeredgewidth=0, clip_on=False, zorder=5)
ax = AX[3]
ax.axvspan(-W5 - 0.5, W5 + 0.5, color="#EDEDED", lw=0, zorder=0)
for (arm, name), col, ls, z in zip(ARMS, (GREY, RED, BLUE), ((0, (4, 2)), "-", "-"), (2, 3, 4)):
    y = prof[arm].copy() * 1e2
    ax.plot(OFF, y, color=col, lw=0.95, ls=ls, zorder=z, marker="o", markersize=1.6, markeredgewidth=0)
ax.set_xlim(OFF[0], OFF[-1]); ax.set_xticks([-10, 0, 10]); ax.set_ylim(0, None)
ax.set_xlabel("Offset $j-i$", fontsize=FS["label"], labelpad=2)
ax.set_ylabel(r"Share of attention ($\times10^{-2}$)", fontsize=FS["label"], labelpad=2.5)
ax.grid(axis="y", color="#E3E3E3", lw=0.4, zorder=0); ax.set_axisbelow(True)
for a_ in AX:
    a_.tick_params(labelsize=FS["tick"], pad=1.8)
    for s in a_.spines.values(): s.set_linewidth(0.6); s.set_color(DARK)
# keys
cax = fig.add_axes([(X[1] + BOX / 2 - 0.30) / W, KEY_Y / Ht, 1.05 / W, BAR_H / Ht])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ramp, norm=Normalize(0, VMAX)), cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12])
cax.xaxis.set_ticks_position("top"); cb.ax.tick_params(labelsize=FS["key"] - 0.5, pad=0.8, length=1.6, width=0.4)
cb.ax.set_xticklabels(["0", "4", "8", r"$\geq$12"]); cb.outline.set_linewidth(0.4)
fig.text((X[1] + BOX / 2 - 0.35) / W, (KEY_Y + BAR_H / 2) / Ht, r"Share of suffix attention ($\times10^{-2}$)", fontsize=FS["key"], ha="right", va="center")
fig.legend(handles=[Line2D([], [], marker="<", color="none", markerfacecolor=RED, markeredgewidth=0, markersize=3.6)], labels=["Repeat"],
           fontsize=FS["key"], frameon=False, loc="center left", bbox_to_anchor=((X[1] + BOX / 2 + 0.80) / W, (KEY_Y + BAR_H / 2) / Ht),
           handletextpad=0.1, borderpad=0, handlelength=1.0)
fig.legend(handles=[Line2D([0], [0], color=c, lw=1.0, ls=ls) for c, ls in ((GREY, (0, (4, 2))), (RED, "-"), (BLUE, "-"))],
           labels=[n for _, n in ARMS], fontsize=FS["key"], frameon=False, ncol=3, loc="center",
           bbox_to_anchor=((X[3] + BOX / 2 - 0.26) / W, (KEY_Y + BAR_H / 2) / Ht), handlelength=1.1, handletextpad=0.3,
           columnspacing=0.5, borderpad=0)
for x, s in zip(X, ("(a) Vanilla", "(b) dLLM-Cache", "(c) +CTAR", "(d) At repeat positions")):
    fig.text((x + BOX / 2) / W, 0.075 / Ht, s, ha="center", va="center", fontsize=FS["cap"])
out = os.path.join(D, "fig-attn-ctar" + ("-zoom" if ZOOM else ""))
fig.savefig(out + ".pdf", dpi=600); fig.savefig(out + ".png", dpi=300)
json.dump(S, open(out + ".json", "w"), indent=1); print("->", out + ".pdf")
