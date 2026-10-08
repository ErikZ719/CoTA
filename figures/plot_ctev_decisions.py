#!/usr/bin/env python
"""CTEV: where confidence alone would commit, and where CTEV commits instead (single column, two stacked panels).

Top    the repeat rate of the positions committed at each level of context entropy, under dLLM-Cache (risk curve)
Bottom at the decisions that CTEV changes, the context entropy of the position that the confidence alone would have
       committed against that of the position CTEV commits (same step, same response)

Both panels share the x axis, so that the shift of the bottom panel can be read against the risk of the top one.
Data: decoding traces (run_repeat_eval.py --case_trace 1), LLaDA-V with dLLM-Cache, L = 128, the first 100 images of
      the evaluation set: lens100/ctev_iso/tr_cache (top) and tr_ctev (bottom). Positions without a committed
      context token (context entropy 0 by definition) are left out of both panels.
Designed at print size: font sizes below are the sizes on paper. Run with /opt/anaconda3/bin/python.
"""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
})
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "lens100")
M, EOT, LAYOUT = 128, (126081, 126348), {198, 220, 197, 256, 262144}


def commits(arm):
    """(E_ctx, repeat, alt_E or nan) of every content commit"""
    out = []
    for f in sorted(glob.glob(os.path.join(R, arm, "trace_*.json"))):
        t = json.load(open(f)); rows = sorted(t["rows"], key=lambda r: r["step"])
        tok = np.full(M, -1)
        for r in rows: tok[r["pos"]] = r["tok"]
        end = next((k for k in range(M) if tok[k] in EOT), M)
        rep = np.zeros(M, bool); prev = None
        for p in range(end):
            if tok[p] in LAYOUT: continue
            if prev is not None and tok[p] == prev: rep[p] = True
            prev = tok[p]
        for r in rows:
            if tok[r["pos"]] in EOT or r["pos"] >= end: continue
            changed = r.get("alt_pos") is not None and r["alt_pos"] != r["pos"]
            out.append((r["E_ctx"], float(rep[r["pos"]]), r["alt_E"] if changed else np.nan))
    return np.array(out, float)


C = commits("ctev_iso/tr_cache"); V = commits("ctev_iso/tr_ctev")
C = C[C[:, 0] > 0]
chg = V[~np.isnan(V[:, 2])]; chg = chg[(chg[:, 0] > 0) & (chg[:, 2] > 0)]
LO, HI, STEP = 4.0, 16.0, 1.0
edges = np.arange(LO, HI + STEP, STEP); ctr = (edges[:-1] + edges[1:]) / 2
rate, n_in = np.full(len(ctr), np.nan), np.zeros(len(ctr), int)
for i in range(len(ctr)):
    m = (C[:, 0] > edges[i]) & (C[:, 0] <= edges[i + 1]); n_in[i] = m.sum()
    if m.sum() >= 50: rate[i] = 100 * C[m, 1].mean()
S = dict(cache_commits=int(len(C)), cache_repeat_rate=float(100 * C[:, 1].mean()),
         bins=[dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=int(n_in[i]), repeat_rate=(None if np.isnan(rate[i]) else float(rate[i]))) for i in range(len(ctr))],
         changed=int(len(chg)), committed_mean=float(chg[:, 0].mean()), confidence_alone_mean=float(chg[:, 2].mean()),
         lower=float((chg[:, 0] < chg[:, 2]).mean()))
top = C[:, 0] > 12.0
S["above_12_bits"] = dict(share_of_cache_commits=float(100 * top.mean()), repeat_rate=float(100 * C[top, 1].mean()),
                          repeat_rate_below=float(100 * C[~top, 1].mean()),
                          confidence_alone=float(100 * (chg[:, 2] > 12).mean()), ctev=float(100 * (chg[:, 0] > 12).mean()))
print(json.dumps(S, indent=1))

RED, BLUE, DARK, GREY = "#C44E52", "#4C72B0", "#333333", "#6B6B6B"
FS = dict(label=8, tick=7, key=7, cap=9, ann=7)
W = 3.5
BW, X0 = 2.85, 0.52
COMPACT = "--compact" in sys.argv   # 2026-10-01 (user): shorter panels and a one-row legend, to give the text room
H1, H2, GAP, Y0, HEAD = (0.72, 0.82, 0.10, 0.36, 0.20) if COMPACT else (0.95, 1.05, 0.12, 0.36, 0.34)
Ht = Y0 + H2 + GAP + H1 + HEAD
fig = plt.figure(figsize=(W, Ht))
ax1 = fig.add_axes([X0 / W, (Y0 + H2 + GAP) / Ht, BW / W, H1 / Ht])
ax2 = fig.add_axes([X0 / W, Y0 / Ht, BW / W, H2 / Ht])
# top: risk curve
ax1.bar(ctr, rate, width=STEP * 0.86, color=RED, edgecolor=DARK, linewidth=0.3, zorder=3)
ax1.set_ylabel("Repeat rate (%)", fontsize=FS["label"], labelpad=2.5); ax1.set_ylim(0, max(np.nanmax(rate) * 1.15, 1))
ax1.tick_params(labelsize=FS["tick"], pad=1.5, labelbottom=False)
ax1.text(0.02, 0.95, "dLLM-Cache: repeat rate of the position\ncommitted, by its context entropy",
         transform=ax1.transAxes, fontsize=FS["ann"], color=DARK, ha="left", va="top", linespacing=1.15)
# bottom: the two distributions at the decisions CTEV changes
bins = np.arange(LO, HI + 0.5, 0.5)
for v, col, z in ((chg[:, 2], RED, 3), (chg[:, 0], BLUE, 4)):
    h, e = np.histogram(np.clip(v, LO, HI - 1e-6), bins=bins); h = 100 * h / len(v)
    ax2.stairs(h, e, color=col, lw=0.9, zorder=z); ax2.stairs(h, e, color=col, alpha=0.18, fill=True, zorder=z - 2, lw=0)
    ax2.axvline(v.mean(), color=col, lw=0.7, ls=(0, (4, 3)), zorder=z)
ax2.set_ylabel("Decisions (%)", fontsize=FS["label"], labelpad=2.5)
ax2.set_xlabel("Context entropy at the decode moment (bits)", fontsize=FS["label"], labelpad=2)
ax2.tick_params(labelsize=FS["tick"], pad=1.5)
MEAN_NOTE = "--mean-note" in sys.argv   # 2026-10-01: name the dashed lines in the panel so the caption can drop it
ax2.text(0.02, 0.95, "the $%s$ decisions that CTEV changes" % format(len(chg), ",").replace(",", "{,}") + ("\n(dashed: means)" if MEAN_NOTE else ""),
         transform=ax2.transAxes, fontsize=FS["ann"], color=DARK, ha="left", va="top", linespacing=1.15)
for ax in (ax1, ax2):
    ax.set_xlim(LO, HI); ax.set_xticks(np.arange(4, 17, 2))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E3E3E3", lw=0.4, zorder=0); ax.set_axisbelow(True)
fig.legend(handles=[Line2D([0], [0], color=RED, lw=1.0), Line2D([0], [0], color=BLUE, lw=1.0)],
           labels=(["Committed by confidence alone", "Committed by CTEV"] if COMPACT else ["Position the confidence alone would commit", "Position CTEV commits"]), loc="lower left",
           bbox_to_anchor=(X0 / W - 0.005, (Y0 + H2 + GAP + H1 + 0.04) / Ht), ncol=2 if COMPACT else 1, frameon=False, fontsize=FS["key"],
           handlelength=1.4, handletextpad=0.4, labelspacing=0.25, columnspacing=1.0, borderpad=0, borderaxespad=0)
out = os.path.join(R, "fig-ctev-decisions")
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); json.dump(S, open(out + ".json", "w"), indent=1)
print("->", out + ".pdf")
