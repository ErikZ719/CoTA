#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Word cloud of the tokens that repeat under an approximate cache (single column, drawn at its printed size).

Input : information_flow/results/repeat_tokens/{LLaDA-V,LaViDa,MMaDA}.json  (scripts/repeat_token_stats.py on the server:
        the cached rows of Table VI, 500 images, three lengths, two backends; a run is a maximal sequence of identical
        content tokens of length >= 2, responses cut at the first end-of-text or end-of-turn token)
Weight: number of runs a token opens, pooled over the 18 cells. A run counts once whatever its length, so that a single
        response that loops for 400 tokens does not decide the picture. Font size grows with log(weight).
Layout: own placement (Archimedean spiral, bounding boxes from the glyph outlines), all words horizontal, Times New Roman.
        Punctuation marks are drawn as key caps. Two-letter word pieces are left out.
Output: information_flow/results/repeat_tokens/fig-repeat-wordcloud.{pdf,png} and the table of weights (.json).

    /opt/anaconda3/bin/python information_flow/results/scripts/plot_repeat_wordcloud.py [--models LLaDA-V,LaViDa,MMaDA] [--top 110]
"""
import json, os, sys, math, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["pdf.fonttype"] = 42
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "repeat_tokens")
def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
MODELS = opt("--models", "LLaDA-V,LaViDa,MMaDA").split(",")
TOP = int(opt("--top", "110"))
W, H = 3.5, float(opt("--height", "1.95"))          # inches: \columnwidth
LEG = 0.17                                           # legend strip at the bottom, inches
S_MIN, S_MAX = 5.6, float(opt("--smax", "33"))       # font size on paper, points
KEY_W, KEY_H = 0.66, 0.92                            # key cap of a punctuation mark, in units of the font size
FONT = "Times New Roman"
COL = {"function": "1F3864", "punct": "7F7F7F", "content": "C00000"}
FUNCTION = set("""the a an and or of to in on at by for with from as is are was were be been being it its this that these those there
which who whose what their his her he she they them we you i not no so than then also into over under up down out off near other
one two three some each both all any more most very can could will would may might has have had do does did if while but""".split())


def rgb(h): return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def label_of(piece):
    """(shown text, class). Tokens that differ only in case or leading space are merged."""
    s = piece.strip()
    if not s: return None, None
    if not any(ch.isalnum() for ch in s):
        return (s, "punct") if len(s) == 1 else (None, None)
    low = s.lower()
    if s.isalpha() and len(s) <= 2 and low not in FUNCTION:
        return None, None                              # two-letter word pieces say nothing on their own
    if not piece.startswith(" ") and s.isalpha() and s.islower() and len(s) <= 5 and low not in FUNCTION:
        return "-" + s, "content"                      # a word piece (no leading space)
    return low, ("function" if low in FUNCTION else "content")


weights, cls, cells, total = collections.Counter(), {}, 0, 0
for m in MODELS:
    J = json.load(open(os.path.join(D, m + ".json")))
    for key, c in J.items():
        cells += 1; total += c["n_runs"]
        for t in c["tokens"]:
            lab, k = label_of(t["piece"])
            if lab is None: continue
            weights[lab] += t["runs"]; cls.setdefault(lab, k)
shown = sum(weights.values())
top = weights.most_common(TOP)
share = {k: sum(v for w, v in weights.items() if cls[w] == k) / shown for k in COL}
print("cells %d, runs %d (%d in the classes shown), distinct tokens %d" % (cells, total, shown, len(weights)))
print("share of the runs: function words %.1f%%, punctuation %.1f%%, content words %.1f%%" % (100 * share["function"], 100 * share["punct"], 100 * share["content"]))
print("top 12:", [(w, v, "%.1f%%" % (100 * v / shown)) for w, v in top[:12]])
lo, hi = math.log(top[-1][1]), math.log(top[0][1])
size = lambda v: S_MIN + (S_MAX - S_MIN) * ((math.log(v) - lo) / (hi - lo)) ** 1.35

FP = {k: FontProperties(family=FONT, weight="bold" if k == "content" else "normal") for k in COL}
def extent(text, pt, k):
    if k == "punct":
        return 0.0, 0.0, KEY_W * pt / 72, KEY_H * pt / 72
    e = TextPath((0, 0), text, size=pt, prop=FP[k]).get_extents()
    return e.x0 / 72, e.y0 / 72, e.width / 72, e.height / 72

placed, boxes, missed = [], [], []
PAD = 0.012
def free(b):
    x0, y0, x1, y1 = b
    if x0 < 0.02 or y0 < LEG + 0.02 or x1 > W - 0.02 or y1 > H - 0.02: return False
    return all(x1 < a0 or x0 > a1 or y1 < b0 or y0 > b1 for a0, b0, a1, b1 in boxes)
rng = np.random.RandomState(int(opt("--seed", "7")))
for text, v in top:
    k = cls[text]; pt = size(v); ok = False
    while pt >= S_MIN - 0.01 and not ok:
        ex0, ey0, ew, eh = extent(text, pt, k)
        t = 0.0; ph = rng.uniform(0, 2 * math.pi)
        while t < 70:
            r = 0.0105 * t; cx = W / 2 + 2.05 * r * math.cos(t + ph); cy = LEG + (H - LEG) / 2 + r * math.sin(t + ph)
            b = (cx - ew / 2 - PAD, cy - eh / 2 - PAD, cx + ew / 2 + PAD, cy + eh / 2 + PAD)
            if free(b):
                boxes.append(b); placed.append((text, pt, k, cx - ew / 2 - ex0, cy - eh / 2 - ey0, v)); ok = True; break
            t += 0.1
        pt -= 0.6
    if not ok: missed.append(text)

fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
for text, pt, k, x, y, v in placed:
    if k == "punct":
        w_, h_ = KEY_W * pt / 72, KEY_H * pt / 72
        ax.add_patch(FancyBboxPatch((x + 0.05 * w_, y + 0.05 * h_), 0.90 * w_, 0.90 * h_,
                                    boxstyle="round,pad=0,rounding_size=%.4f" % (0.16 * w_), fc=rgb("F2F2F2"), ec=rgb(COL[k]), lw=0.5))
        ax.text(x + w_ / 2, y + 0.34 * h_, text, fontsize=0.95 * pt, family=FONT, color=rgb("404040"), ha="center", va="baseline",
                fontweight="bold")
        continue
    ax.text(x, y, text, fontsize=pt, family=FONT, color=rgb(COL[k]), ha="left", va="baseline",
            fontweight="bold" if k == "content" else "normal")
# legend: the three classes and their share of the runs
items = [("function", "function words"), ("punct", "punctuation"), ("content", "content words")]
lab = ["%s %.0f%%" % (n, 100 * share[k]) for k, n in items]
wid = [0.10 + TextPath((0, 0), t, size=6.6, prop=FontProperties(family=FONT)).get_extents().width / 72 for t in lab]
x = (W - sum(wid) - 0.16 * (len(lab) - 1)) / 2
for (k, n), t, w_ in zip(items, lab, wid):
    ax.add_patch(Rectangle((x, 0.055), 0.07, 0.07, fc=rgb(COL[k]), ec="none"))
    ax.text(x + 0.10, 0.09, t, fontsize=6.6, family=FONT, color=rgb("000000"), ha="left", va="center")
    x += w_ + 0.16
out = os.path.join(D, "fig-repeat-wordcloud" + opt("--suffix", ""))
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=400)
json.dump(dict(models=MODELS, cells=cells, runs=total, runs_shown=shown, share=share,
               top=[dict(token=w, runs=v, share=v / shown, cls=cls[w]) for w, v in top]),
          open(out + ".json", "w"), indent=1, ensure_ascii=False)
print("placed %d of %d words (left out for lack of room: %d), smallest %.1f pt -> %s.pdf" % (len(placed), len(top), len(missed), min(p[1] for p in placed), out))
