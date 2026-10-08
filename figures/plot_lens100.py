#!/usr/bin/env python
"""What each component changes at the decode moment (full width, 17 cm): three panels, one per axis.

(a) CTAR, routing        attention that a position puts on its context tokens (+-5, layers 25-32): the routing stored in
                         the cache against the routing re-formed by CTAR, same position and same step (paired)
(b) DAR, timeliness      staleness of the KV state of a position when it is committed, along the decoding steps
(c) CTEV, consolidation  context entropy of the position that is committed against that of the position that the
                         confidence alone would commit, at the steps where CTEV changes the decision (paired)

Data: decoding traces of run_repeat_eval.py --case_trace 1, LLaDA-V with dLLM-Cache, L = 128, the first 100 images of
      the evaluation set. Single components: lens100/lens100/tr_{ctar,dar}, lens100/ctev_iso/tr_ctev; cache: tr_cache.
      (mirrors of results/lladav/{lens100,ctev_iso} on the server)
Options: --ctev_all   panel (c) also counts the positions without any committed context token (context entropy 0)
         --dar_only   the single-column figure of the paper: panel (b) alone (user, 2026-09-30: CTAR has its own figure,
                      the CTEV panel is left out)
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
ARM_DIR = {"cache": "ctev_iso/tr_cache", "ctev": "ctev_iso/tr_ctev", "ctar": "lens100/tr_ctar", "dar": "lens100/tr_dar"}
M, LAYERS, EOT = 128, 32, (126081, 126348)
CTEV_ALL = "--ctev_all" in sys.argv
DAR_ONLY = "--dar_only" in sys.argv


def traces(arm):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, ARM_DIR[arm], "trace_*.json")))]


def ages(t):
    """staleness of every commit: forwards since the state of the position was last recomputed, mean over layers"""
    rows = sorted(t["rows"], key=lambda r: r["step"]); age = np.full(M, np.nan)
    last = np.zeros((LAYERS, M), int); by_f = {}
    for r in rows: by_f.setdefault(r["fwd"], []).append(r)
    for fwd in range(1, t["n_forwards"] + 1):
        part = t["refresh"].get(str(fwd))
        for l in range(LAYERS):
            idx = None if part is None else part.get(str(l))
            if idx is None: last[l, :] = fwd                         # periodic full recompute
            elif idx: last[l, idx] = fwd
        for r in by_f.get(fwd, []):
            if r["tok"] not in EOT and r["step"] < M: age[r["step"]] = float(np.mean(fwd - last[:, r["pos"]]))
    return age


T = {a: traces(a) for a in ARM_DIR}
print({a: len(v) for a, v in T.items()})
# ---- (a) paired routing under CTAR
old, new = [], []
for t in T["ctar"]:
    pr = t["ctar_pair"]; n = np.array(pr["n"]); ok = n > 0
    old += list(np.array(pr["old"])[ok] / n[ok]); new += list(np.array(pr["new"])[ok] / n[ok])
old, new = np.array(old), np.array(new); gain = new - old
# ---- (b) staleness along the decoding steps
AGE = {a: np.array([ages(t) for t in T[a]]) for a in ("cache", "dar")}
# ---- (c) decisions changed by CTEV
E_c, E_a = [], []
for t in T["ctev"]:
    for r in t["rows"]:
        if r["tok"] in EOT or r.get("alt_pos") is None or r["alt_pos"] == r["pos"]: continue
        E_c.append(r["E_ctx"]); E_a.append(r["alt_E"])
E_c, E_a = np.array(E_c), np.array(E_a)
n_changed = len(E_c); n_commits = sum(1 for t in T["ctev"] for r in t["rows"] if r["tok"] not in EOT)
keep = np.ones(len(E_c), bool) if CTEV_ALL else (E_c > 0) & (E_a > 0)
E_c, E_a = E_c[keep], E_a[keep]
S = dict(images=len(T["cache"]),
         ctar=dict(positions=int(len(old)), stored=float(old.mean()), reformed=float(new.mean()), raised=float((gain > 0).mean()),
                   gain_mean=float(gain.mean()), gain_median=float(np.median(gain))),
         dar=dict(cache_mean=float(np.nanmean(AGE["cache"])), dar_mean=float(np.nanmean(AGE["dar"])),
                  cache_fresh=float(np.nanmean(AGE["cache"] == 0)), dar_fresh=float(np.nanmean(AGE["dar"] == 0)),
                  cache_max=float(np.nanmax(np.nanmean(AGE["cache"], axis=0))), dar_max=float(np.nanmax(np.nanmean(AGE["dar"], axis=0)))),
         ctev=dict(commits=int(n_commits), changed=int(n_changed), share_changed=n_changed / n_commits, shown=int(keep.sum()),
                   with_context_only=not CTEV_ALL, committed=float(E_c.mean()), confidence_alone=float(E_a.mean()),
                   lower=float((E_c < E_a).mean())))
print(json.dumps(S, indent=1))

RED, BLUE, DARK, GREY = "#C44E52", "#4C72B0", "#333333", "#6B6B6B"
FS = dict(label=8, tick=7, key=7, cap=9, ann=7)
CM = 1 / 2.54
if DAR_ONLY:                                               # single column: the staleness panel alone, no title line
    COMPACT = "--compact" in sys.argv   # 2026-10-01 (user): shorter box, legend dropped (the annotations name the lines)
    W = 3.5; BW, BH = 2.85, (0.95 if COMPACT else 1.25); Y0, HEAD = 0.36, (0.06 if COMPACT else 0.20); Ht = Y0 + BH + HEAD
    fig = plt.figure(figsize=(W, Ht)); ax = fig.add_axes([0.52 / W, Y0 / Ht, BW / W, BH / Ht])
    steps = np.arange(M)
    for a, col in (("cache", RED), ("dar", BLUE)):
        n = np.sum(~np.isnan(AGE[a]), axis=0); mu = np.where(n >= 10, np.nanmean(AGE[a], axis=0), np.nan)
        ax.plot(steps, mu, color=col, lw=0.9, zorder=3 if a == "cache" else 4)
    ax.set_xlim(0, M - 1); ax.set_xticks([0, 32, 64, 96, 127]); ax.set_ylim(-0.25, 6.4); ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("Decoding step $t$", fontsize=FS["label"], labelpad=2)
    ax.set_ylabel(r"Staleness $\Delta$ (steps)" if COMPACT else r"Staleness $\Delta$ at decode (steps)", fontsize=FS["label"], labelpad=2.5)
    ax.tick_params(labelsize=FS["tick"], pad=1.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color="#E3E3E3", lw=0.4, zorder=0); ax.set_axisbelow(True)
    ax.text(0.02, 0.965, "dLLM-Cache: mean $%.2f$" % S["dar"]["cache_mean"], transform=ax.transAxes, fontsize=FS["ann"], color=RED, ha="left", va="top")
    ax.text(0.02, 0.845, "+DAR: mean $%.3f$, zero at $%.0f\\%%$" % (S["dar"]["dar_mean"], 100 * S["dar"]["dar_fresh"]), transform=ax.transAxes, fontsize=FS["ann"], color=BLUE, ha="left", va="top")
    if not COMPACT: fig.legend(handles=[Line2D([0], [0], color=RED, lw=1.0), Line2D([0], [0], color=BLUE, lw=1.0)], labels=["dLLM-Cache", "+DAR"],
               loc="lower left", bbox_to_anchor=(0.50 / W, (Y0 + BH + 0.04) / Ht), ncol=2, frameon=False, fontsize=FS["key"],
               handlelength=1.4, handletextpad=0.4, columnspacing=0.9, borderpad=0, borderaxespad=0)
    out = os.path.join(R, "fig-dar-staleness")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); json.dump(S["dar"], open(out + ".json", "w"), indent=1)
    print("->", out + ".pdf"); sys.exit(0)
W = 17.0 * CM
BW, BH = 1.78, 1.18                                        # axes box, inches
RIGHT, Y0, HEAD = 0.06, 0.475, 0.30
GUT = (W - RIGHT - 3 * BW) / 3
Ht = Y0 + BH + HEAD
fig = plt.figure(figsize=(W, Ht))
AX = [fig.add_axes([(GUT + k * (BW + GUT)) / W, Y0 / Ht, BW / W, BH / Ht]) for k in range(3)]


def dress(ax, xlab, ylab, title, handles, labels):
    ax.set_xlabel(xlab, fontsize=FS["label"], labelpad=2); ax.set_ylabel(ylab, fontsize=FS["label"], labelpad=2.5)
    ax.tick_params(labelsize=FS["tick"], pad=1.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E3E3E3", lw=0.4, zorder=0); ax.set_axisbelow(True)
    x0 = ax.get_position().x0 * W
    fig.legend(handles=handles, labels=labels, loc="lower left", bbox_to_anchor=((x0 - 0.02) / W, (Y0 + BH + 0.035) / Ht),
               ncol=len(labels), frameon=False, fontsize=FS["key"], handlelength=1.4, handletextpad=0.4, columnspacing=0.9,
               borderpad=0, borderaxespad=0)
    fig.text((x0 + BW / 2) / W, 0.04 / Ht, title, fontsize=FS["cap"], ha="center", va="bottom")


# (a)
ax = AX[0]
lim = 0.06; bins = np.linspace(-lim, lim, 41); g = np.clip(gain, -lim + 1e-6, lim - 1e-6)
h, e = np.histogram(g, bins=bins); h = h / len(g); c = (e[:-1] + e[1:]) / 2
ax.bar(c[c < 0], h[c < 0], width=e[1] - e[0], color=RED, edgecolor=DARK, linewidth=0.25, zorder=3)
ax.bar(c[c > 0], h[c > 0], width=e[1] - e[0], color=BLUE, edgecolor=DARK, linewidth=0.25, zorder=3)
ax.axvline(0, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=2)
ax.set_xlim(-lim, lim); ax.set_xticks([-0.05, 0, 0.05]); ax.set_xticklabels(["$-0.05$", "0", "$+0.05$"])
ax.text(0.98, 0.95, "raised at $%.0f\\%%$" % (100 * S["ctar"]["raised"]), transform=ax.transAxes, fontsize=FS["ann"], color=BLUE, ha="right", va="top")
ax.text(0.98, 0.83, "of the positions", transform=ax.transAxes, fontsize=FS["ann"], color=BLUE, ha="right", va="top")
dress(ax, "Change in attention on context tokens", "Fraction of positions", "(a) Routing, stored and re-formed",
      [Patch(fc=RED, ec=DARK, lw=0.25), Patch(fc=BLUE, ec=DARK, lw=0.25)], ["Lowered", "Raised by CTAR"])
# (b)
ax = AX[1]; steps = np.arange(M)
for a, col in (("cache", RED), ("dar", BLUE)):
    n = np.sum(~np.isnan(AGE[a]), axis=0); mu = np.where(n >= 10, np.nanmean(AGE[a], axis=0), np.nan)
    ax.plot(steps, mu, color=col, lw=0.9, zorder=3 if a == "cache" else 4)
ax.set_xlim(0, M - 1); ax.set_xticks([0, 40, 80, 120]); ax.set_ylim(-0.25, 6.4); ax.set_yticks([0, 2, 4, 6])
dress(ax, "Decoding step $t$", r"Staleness $\Delta$ at decode", "(b) Staleness at the decode moment",
      [Line2D([0], [0], color=RED, lw=1.0), Line2D([0], [0], color=BLUE, lw=1.0)], ["dLLM-Cache", "+DAR"])
# (c)
ax = AX[2]
lo_, hi_ = (0.0 if CTEV_ALL else 4.0), 16.0; bins = np.linspace(lo_, hi_, 33 if CTEV_ALL else 25)
for v, col, z in ((E_a, RED, 3), (E_c, BLUE, 4)):
    h, e = np.histogram(np.clip(v, lo_, hi_ - 1e-6), bins=bins); h = h / len(v)
    ax.stairs(h, e, color=col, lw=0.9, zorder=z, fill=False); ax.stairs(h, e, color=col, alpha=0.18, fill=True, zorder=z - 2, lw=0)
    ax.axvline(v.mean(), color=col, lw=0.7, ls=(0, (4, 3)), zorder=z)
ax.set_xlim(lo_, hi_); ax.set_xticks([4, 8, 12, 16] if not CTEV_ALL else [0, 4, 8, 12, 16])
dress(ax, "Context entropy at decode (bits)", "Fraction of decisions", "(c) Context entropy under CTEV",
      [Line2D([0], [0], color=RED, lw=1.0), Line2D([0], [0], color=BLUE, lw=1.0)], ["Confidence alone", "CTEV"])
out = os.path.join(R, "fig-lens100" + ("-all" if CTEV_ALL else ""))
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
json.dump(S, open(out + ".json", "w"), indent=1); print("->", out + ".pdf")
