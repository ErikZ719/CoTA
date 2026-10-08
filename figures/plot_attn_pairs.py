#!/usr/bin/env python
"""Attention at the decode moment, cached against treated, three model/backend pairs (single column, 2026-09-30).

Row 1  LLaDA-V + dLLM-Cache and + CTAR, L = 128, the image of Fig. 10(a)      attn_decode/caseA_118929.npz (validated recorder)
Row 2  LaViDa  + dLLM-Cache and + CoTA++, L = 128, image 479129 (run of 89)   rows_probe/lavida_dc128_{cache,cpp}.npz
Row 3  LLaDA-V + SlowFast   and + CoTA++, L = 512, image 462371 (run of 68)   rows_probe/lladav_sf512b_{cache,cpp}.npz
Rows 2-3 come from scripts/exp3_2026-09-30/decode_rows_recorder.py (run_repeat_eval*.py --decode_rows), which agrees
with the validated recorder to 5e-4 on row 1's image. Row i of a map is the attention of position i over the suffix at
the forward it was decoded from, mean over heads and layers 25-32, normalised over the suffix keys (Section IV-B).
Each row of the figure shows a window of 60 (L = 128) or 80 (L = 512) positions around the repeated run; the window is
the same for both maps of a row. Triangles: positions that repeat under the cache.
Designed at print size; run with /opt/anaconda3/bin/python.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
})
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..")
EOT, LAYOUT = (126081, 126348), (198, 220, 197, 256, 262144)
MASK = 126336


def rownorm(A):
    A = np.nan_to_num(np.asarray(A, np.float64)); s = A.sum(axis=-1, keepdims=True)
    return np.where(s > 0, A / np.where(s > 0, s, 1), np.nan)


def repeats(tok):
    tok = np.asarray(tok); end = np.where(np.isin(tok, EOT) | (tok == MASK))[0]; end = int(end[0]) if len(end) else len(tok)
    rep = np.zeros(len(tok), bool); prev = None
    for p in range(end):
        if tok[p] in LAYOUT: continue
        if prev is not None and tok[p] == prev: rep[p] = True
        prev = tok[p]
    return rep


def longest_run(rep):
    best = (0, 0); s = None
    for p in range(len(rep) + 1):
        if p < len(rep) and rep[p]:
            if s is None: s = p - 1
        elif s is not None:
            if p - s > best[1] - best[0]: best = (s, p)
            s = None
    return best


C = np.load(os.path.join(R, "attn_decode", "caseA_118929.npz"))
Z100 = np.load(os.path.join(R, "attn_decode", "seed0_100.npz"))
def probe(name):
    z = np.load(os.path.join(R, "rows_probe", name + ".npz")); return z["rows"].astype(np.float64), z["ids"]
def seed(stem, arm):
    i = next(k for k, s_ in enumerate(Z100["stems"]) if s_.endswith(stem)); return Z100[arm + "_attn"][i].astype(np.float64), Z100[arm + "_tok"][i]
# 2026-10-01 (user): every right panel is CTAR alone; images chosen for a visible loss of anchoring under the cache.
#   VARIANT selects the row set: "v1" = the 2026-09-30 figure (118929 / 479129 CoTA++ / 462371 CoTA++);
#   "v2" = 131969 (seed0 recording) / LaViDa 479129 CTAR / LLaDA-V SlowFast 462371 CTAR; windows: 0 = full suffix.
VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "v2"
LV = sys.argv[sys.argv.index("--lavida") + 1] if "--lavida" in sys.argv else "lavida_dc128"          # or lavida_dc128b (497330)
SF = sys.argv[sys.argv.index("--slowfast") + 1] if "--slowfast" in sys.argv else "lladav_sf512b"     # or lladav_sf512 (479129)
R1 = sys.argv[sys.argv.index("--row1") + 1] if "--row1" in sys.argv else "131969"
NROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 3        # 2 = drop the SlowFast row (user, 2026-10-01)
CBAR_TITLE = "--no-cbar-title" not in sys.argv
if VARIANT == "v1":
    ROWS = [("LLaDA-V + dLLM-Cache", "+CTAR", (C["dllm_cache_attn"], C["dllm_cache_tok"]), (C["ctar_th_attn"], C["ctar_th_tok"]), 60),
            ("LaViDa + dLLM-Cache", "+CoTA++", probe("lavida_dc128_cache"), probe("lavida_dc128_cpp"), 60),
            ("LLaDA-V + SlowFast", "+CoTA++", probe("lladav_sf512b_cache"), probe("lladav_sf512b_cpp"), 80)]
    WIN = {0: (44, 103)}
else:
    ROWS = [("LLaDA-V + dLLM-Cache", "+CTAR", seed(R1, "dllm_cache"), seed(R1, "ctar_th"), 0),
            ("LaViDa + dLLM-Cache", "+CTAR", probe(LV + "_cache"), probe(LV + "_ctar"), 0),
            ("LLaDA-V + SlowFast", "+CTAR", probe(SF + "_cache"), probe(SF + "_ctar"), 80)]
    WIN = {}
ROWS = ROWS[:NROWS]
for k, (_, _, (Ac, tc), _, wl) in enumerate(ROWS):
    if k in WIN: continue
    M = len(tc)
    if wl == 0: WIN[k] = (0, M - 1); continue
    rep = repeats(tc); s_, e_ = longest_run(rep)
    lo = max(0, min(s_ - wl // 4, M - wl)); WIN[k] = (lo, lo + wl - 1)

RED, DARK = "#C44E52", "#333333"
FS = dict(label=8, tick=7, key=7, cap=8.5)
UP = 8
up = lambda a: np.repeat(np.repeat(a, UP, axis=0), UP, axis=1)
SIDE = "--side" in sys.argv   # 2026-10-01 (user): titles above the panels, colorbar and Repeat marker on the right side
if SIDE:
    W = 3.5; BOX = 1.22; GUT = 0.24; X0 = 0.42; BOT = 0.33; TIT = 0.17; ROWH = BOX + BOT + TIT; HEAD = 0.02; CAPY = 0.0
else:
    W = 3.5; BOX = 1.30; GUT = 0.30; X0 = 0.42; HEAD = 0.40 if CBAR_TITLE else 0.30; ROWH = BOX + 0.46; CAPY = 0.08; BOT = 0.46; TIT = 0.0
Ht = HEAD + len(ROWS) * ROWH
fig = plt.figure(figsize=(W, Ht))
VMAX = 12.0
S = {}
for k, (lc, lt, (Ac, tc), (At, tt), wl) in enumerate(ROWS):
    y0 = Ht - HEAD - (k + 1) * ROWH + BOT
    lo, hi = WIN[k]; M = len(tc)
    rep = np.where(repeats(tc))[0]; rep_t = np.where(repeats(tt))[0]
    S["row%d" % (k + 1)] = dict(window=[int(lo), int(hi)], L=int(M), repeat_positions_cached=int(len(rep)), repeat_positions_treated=int(len(rep_t)),
                                longest_run_cached=int(np.diff(longest_run(repeats(tc)))[0]))
    for j, (A, tok, name) in enumerate(((Ac, tc, lc), (At, tt, lt))):
        ax = fig.add_axes([(X0 + j * (BOX + GUT)) / W, y0 / Ht, BOX / W, BOX / Ht])
        Z = np.nan_to_num(rownorm(A)) * 1e2
        sub = Z[lo:hi + 1, lo:hi + 1]
        ax.imshow(up(sub), aspect="auto", cmap="viridis", vmin=0, vmax=VMAX, origin="upper",
                  extent=[lo - 0.5, hi + 0.5, hi + 0.5, lo - 0.5], interpolation="nearest")
        ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(hi + 0.5, lo - 0.5)
        step = 40 if (hi - lo) >= 120 else 20
        tk = [t for t in range(0, M, step) if lo <= t <= hi]; ax.set_xticks(tk); ax.set_yticks(tk)
        ax.set_xlabel("Suffix position $j$ (key)", fontsize=FS["label"], labelpad=2)
        ax.tick_params(labelsize=FS["tick"], pad=1.8)
        for s_ in ax.spines.values(): s_.set_linewidth(0.6); s_.set_color(DARK)
        if j == 0:
            ax.set_ylabel("Suffix position $i$ (query)", fontsize=FS["label"], labelpad=2.5)
            rc = [r for r in rep if lo <= r <= hi]
            if rc:
                ax.plot([0.965] * len(rc), rc, transform=ax.get_yaxis_transform(), linestyle="none", marker="<", markersize=2.6,
                        markerfacecolor=RED, markeredgewidth=0, clip_on=False, zorder=5)
        else:
            ax.tick_params(labelleft=False)
        if SIDE: fig.text((X0 + j * (BOX + GUT) + BOX / 2) / W, (y0 + BOX + 0.085) / Ht, "(%s) %s" % ("abcdef"[2 * k + j], name), ha="center", va="center", fontsize=FS["cap"])
        else: fig.text((X0 + j * (BOX + GUT) + BOX / 2) / W, (y0 - 0.46 + CAPY) / Ht, "(%s) %s" % ("abcdef"[2 * k + j], name), ha="center", va="center", fontsize=FS["cap"])
if SIDE:
    xr = X0 + 2 * BOX + GUT; ybot = Ht - HEAD - len(ROWS) * ROWH + BOT; ytop = Ht - HEAD - ROWH + BOT + BOX
    cax = fig.add_axes([(xr + 0.09) / W, ybot / Ht, 0.07 / W, (ytop - ybot) / Ht])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(0, VMAX)), cax=cax, orientation="vertical", ticks=[0, 4, 8, 12])
    cb.ax.tick_params(labelsize=FS["key"] - 0.5, pad=1.0, length=1.6, width=0.4); cb.ax.set_yticklabels(["0", "4", "8", r"$\geq$12"]); cb.outline.set_linewidth(0.4)
else:
    cax = fig.add_axes([(X0 + 0.30) / W, (Ht - HEAD + 0.10) / Ht, 1.50 / W, 0.055 / Ht])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(0, VMAX)), cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12])
    cax.xaxis.set_ticks_position("top"); cb.ax.tick_params(labelsize=FS["key"] - 0.5, pad=0.8, length=1.6, width=0.4)
    cb.ax.set_xticklabels(["0", "4", "8", r"$\geq$12"]); cb.outline.set_linewidth(0.4)
if CBAR_TITLE: fig.text((X0 + 0.30 + 0.75) / W, (Ht - HEAD + 0.31) / Ht, r"Share of suffix attention ($\times10^{-2}$)", fontsize=FS["key"], ha="center", va="center")
if SIDE:
    fig.legend(handles=[Line2D([], [], marker="<", color="none", markerfacecolor=RED, markeredgewidth=0, markersize=3.6)], labels=["Repeat"],
               fontsize=FS["key"], frameon=False, loc="center", bbox_to_anchor=((xr + 0.09 + 0.035 + 0.06) / W, (ytop + 0.085) / Ht),
               handletextpad=0.1, borderpad=0, handlelength=1.0)
else:
    fig.legend(handles=[Line2D([], [], marker="<", color="none", markerfacecolor=RED, markeredgewidth=0, markersize=3.6)], labels=["Repeat"],
               fontsize=FS["key"], frameon=False, loc="center left", bbox_to_anchor=((X0 + 0.30 + 1.50 + 0.12) / W, (Ht - HEAD + 0.10 + 0.028) / Ht),
               handletextpad=0.1, borderpad=0, handlelength=1.0)
out = os.path.join(R, "attn_decode", "fig-attn-pairs-1col" + ("" if VARIANT == "v2" else "-" + VARIANT))
fig.savefig(out + ".pdf", dpi=600); fig.savefig(out + ".png", dpi=300)
json.dump(S, open(out + ".json", "w"), indent=1); print(json.dumps(S, indent=1)); print("->", out + ".pdf")
