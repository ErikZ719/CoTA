#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Failure-case figure (Discussion, 2026-10-01), single column, drawn at its printed size, in the style of Fig. 10.

Image COCO_val2014_000000001779 (a corner with two street signs), LLaDA-V at L=512, the Table VI runs:
    LLaDA-V  /  + dLLM-Cache  /  + dLLM-Cache + CoTA++
A token that repeats the one before it is marked red (the criterion of ARR and MRL). A run made of the digits of a
number that is written in the image is marked blue instead: it is what the uncached model also produces, and it is
all that CoTA++ leaves. Data: information_flow/results/case_study/fail_cases.json (ids decoded token by token).

    /opt/anaconda3/bin/python _tools/make_fail_fig.py [--preview]      -> images/fig-failure-case.pdf (+ .png)
"""
import os, re, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from PIL import Image

warnings.filterwarnings("ignore")
plt.rcParams["pdf.fonttype"] = 42
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.dirname(HERE)
CS = os.path.join(PAPER, "..", "..", "information_flow", "results", "case_study")
PREVIEW = "--preview" in sys.argv
CASE = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else "hydrant"
# signs: 001779 at L=512 (the residual is the digits of the signs); hydrant: 536426 at L=128 (the cache breaks into a run of
# 61 word pieces, CoTA++ leaves one word piece attached to its own end)
CASES = {"signs": dict(image="COCO_val2014_000000001779.jpg", L=512, prefix="", focus_words=("5100", "3700"), crop=None),
         "hydrant": dict(image="COCO_val2014_000000536426.jpg", L=128, prefix="hyd_", focus_words=("hydrantrant",), crop=(0, 60, 375, 341))}
CC = CASES[CASE]; IMAGE = CC["image"]
LAYOUT = {198, 220, 197, 256, 262144}
L = CC["L"]

FONT = "Times New Roman"
INK, RULE, NOTE, RED, RED_BG, HEAD_BG = "000000", "595959", "404040", "C00000", "FBE3E3", "F2F2F2"
BLUE, BLUE_BG = "1F4E9C", "DCE6F5"
GREEN = "1E7B34"
FRAME_PT, RULE_PT = 0.9, 0.6
MARGIN = 0.05
W = 3.5
BODY_PT, HEAD_PT, LH = 7.6, 8.0, 0.128
PAD, N_LINES = 0.06, 6
CW_ = W - 2 * MARGIN
TEXT_W = CW_ - 2 * PAD - 0.02
STYLE = {"rep": (RED, RED_BG), "num": (BLUE, BLUE_BG)}


def rgb(h): return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
_FP = {(b, i): FontProperties(family=FONT, weight="bold" if b else "normal", style="italic" if i else "normal")
       for b in (0, 1) for i in (0, 1)}
_TW = {}
def tw(s, size, bold=False, italic=False):
    k = (s, size, bold, italic)
    if k not in _TW:
        fp = _FP[(int(bold), int(italic))]
        if s.strip():
            _TW[k] = TextPath((0, 0), s, size=size, prop=fp).get_extents().width / 72.0
        else:
            _TW[k] = (TextPath((0, 0), "x x", size=size, prop=fp).get_extents().width
                      - TextPath((0, 0), "xx", size=size, prop=fp).get_extents().width) / 72.0
    return _TW[k]
def ww_(word, size): return sum(tw(t, size, st is not None) for t, st in word)


def token_words(pieces, ids, keep_words=8, keep_pieces=4):
    """Words of a response from its tokens. A token in a run of identical content tokens is 'rep'; if the run is
    made of a digit token it is 'num'. Runs longer than keep + 2 are shortened to '... (xn)'."""
    end = next((k for k, p in enumerate(pieces) if p.startswith("<|")), len(pieces))
    pieces, ids = pieces[:end], ids[:end]
    content = [k for k, t in enumerate(ids) if t not in LAYOUT]
    run_of, i = {}, 0
    while i < len(content):
        j = i
        while j + 1 < len(content) and ids[content[j + 1]] == ids[content[i]]: j += 1
        if j > i:
            for k in content[i:j + 1]: run_of[k] = (content[i], j - i + 1)
        i = j + 1
    segs, k = [], 0
    while k < len(pieces):
        if k in run_of and run_of[k][0] == k:
            n = run_of[k][1]; p = pieces[k]; last = [q for q in content if q >= k][n - 1]
            st = "num" if p.strip().isdigit() else "rep"
            if p.startswith(" "):
                shown = n if n <= keep_words + 2 else keep_words
                segs += [(p, st)] * shown
                if shown < n: segs += [(" …", st), (" (×%d)" % n, st)]
            else:
                shown = n if n <= keep_pieces + 2 else keep_pieces
                segs.append((p * shown + ("…" if shown < n else ""), st))
                if shown < n: segs.append((" (×%d)" % n, st))
            k = last + 1
        else:
            segs.append((pieces[k].replace("\n", " "), None)); k += 1
    words, cur = [], []
    for t, st in segs:
        for part in re.split(r"(\s+)", t):
            if not part: continue
            if part.isspace():
                if cur: words.append(cur); cur = []
            elif cur and cur[-1][1] == st: cur[-1] = (cur[-1][0] + part, st)
            else: cur.append((part, st))
    if cur: words.append(cur)
    return words


ELLIPSIS = [("[…]", None)]
def wrap(ws, width, size, max_lines):
    lines, cur, cw, sp = [], [], 0.0, tw(" ", size)
    for word in ws:
        w_ = ww_(word, size)
        if cur and cw + sp + w_ > width:
            lines.append(cur); cur, cw = [], 0.0
            if len(lines) == max_lines: break
        cur.append(word); cw += (sp if len(cur) > 1 else 0) + w_
    else:
        if cur: lines.append(cur)
        return lines, False
    last = lines[-1]
    while last and sum(ww_(a, size) for a in last) + sp * len(last) + tw("…", size) > width: last.pop()
    last.append([("…", None)]); return lines, True
def n_lines(ws, width, size): return len(wrap(ws, width, size, 10 ** 6)[0])
def fit(ws, width, size, max_lines, focus=None):
    """the opening of the text, then '[...]', then the words [focus[0], focus[1]) in full, within max_lines"""
    if focus is None or n_lines(ws[:focus[1]], width, size) <= max_lines:
        return wrap(ws, width, size, max_lines)[0]
    a, b = focus
    ends = [k + 1 for k in range(a) if ws[k][-1][0].endswith((".", "!", "?", ":"))]
    for cands in (sorted(ends, reverse=True), range(a - 1, 3, -1)):
        for h_ in cands:
            if n_lines(ws[:h_] + [ELLIPSIS] + ws[a:b], width, size) <= max_lines:
                return wrap(ws[:h_] + [ELLIPSIS] + ws[a:], width, size, max_lines)[0]
    return wrap(ws, width, size, max_lines)[0]


# ------------------------------------------------------------------ data
D = json.load(open(os.path.join(CS, "fail_cases.json")))
def find(ws, needle):
    return [k for k, w_ in enumerate(ws) if needle in "".join(t for t, _ in w_)]
def stats(ids):
    """longest run of non-digit content tokens, and whether digit runs exist"""
    end = next((k for k, t in enumerate(ids) if t in (126081, 126348)), len(ids)); ids = ids[:end]
    content = [t for t in ids if t not in LAYOUT]
    best, cur, prev, digit = 0, 0, None, False
    for t in content:
        cur = cur + 1 if t == prev else 1; prev = t
        if cur >= 2:
            if PIECE[t].strip().isdigit(): digit = True
            else: best = max(best, cur)
    return best, digit
PIECE = {}
PX = CC["prefix"]
for arm in (PX + "van_%d" % L, PX + "cache_%d" % L, PX + "cpp_%d" % L):
    for i, p in zip(D[arm]["ids"], D[arm]["pieces"]): PIECE[i] = p
rows = []
for arm, name in ((PX + "van_%d" % L, "LLaDA-V"), (PX + "cache_%d" % L, "+ dLLM-Cache"), (PX + "cpp_%d" % L, "+ dLLM-Cache + CoTA++")):
    ws = token_words(D[arm]["pieces"], D[arm]["ids"])
    if "cache" in arm:
        hit = [k for k, w_ in enumerate(ws) if any("×" in t for t, _ in w_)]
        k0 = hit[0] if hit else 0
        first = k0
        while first - 1 >= 0 and any(st == "rep" for _, st in ws[first - 1]): first -= 1
        focus = (max(first - 3, 0), min(k0 + 2, len(ws)))
    else:
        hs = [find(ws, w) for w in CC["focus_words"]]; hs = [h[0] for h in hs if h]
        focus = (max(min(hs) - 6, 0), max(hs) + 2) if hs else None
    lines = fit(ws, TEXT_W, BODY_PT, N_LINES, focus)
    best, digit = stats(D[arm]["ids"])
    rows.append((name, lines, best, digit))

# ------------------------------------------------------------------ draw
IMG = Image.open(os.path.join(CS, IMAGE)).convert("RGB")
if CC["crop"]: IMG = IMG.crop(CC["crop"])
ANY_DIGIT = any(d for _, _, _, d in rows)
IMG_W = 1.30; IMG_H = IMG_W * IMG.size[1] / IMG.size[0]
HEAD_H, GAP = 0.165, 0.055
BOX = HEAD_H + N_LINES * LH + 0.05
H = 2 * MARGIN + IMG_H + GAP + 3 * BOX + 2 * GAP
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
Z = [1]
def z():
    Z[0] += 1; return Z[0]
def frame(x, y, w, h, lc=RULE, lw=FRAME_PT, fc=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=rgb(fc) if fc else "none", ec=rgb(lc) if lc else "none", lw=lw, zorder=z()))
def rule(x1, y1, x2, y2, color=RULE, lw=RULE_PT):
    ax.plot([x1, x2], [y1, y2], color=rgb(color), lw=lw, zorder=z(), solid_capstyle="butt")
def label(x, y, w, h, s, size, color=INK, bold=False, italic=False, align="left"):
    tx = {"left": x, "right": x + w, "center": x + w / 2}[align]
    ax.text(tx, y + h / 2, s, fontsize=size, color=rgb(color), family=FONT, ha=align, va="center",
            fontweight="bold" if bold else "normal", fontstyle="italic" if italic else "normal", zorder=z())
def words(x, y, lines):
    sp = tw(" ", BODY_PT)
    for i, ln in enumerate(lines):
        cx, cy = x, y + i * LH
        for word in ln:
            for q, (t, st) in enumerate(word):
                w_ = tw(t, BODY_PT, st is not None)
                if st is not None:
                    l_ = 0.006 if q == 0 else 0.0; r_ = 0.006 if q == len(word) - 1 else 0.0
                    ax.add_patch(Rectangle((cx - l_, cy + 0.012), w_ + l_ + r_, LH - 0.02, fc=rgb(STYLE[st][1]), ec="none", zorder=z()))
                ax.text(cx, cy + LH / 2, t, fontsize=BODY_PT, family=FONT, va="center", ha="left", zorder=z(),
                        color=rgb(STYLE[st][0] if st is not None else INK), fontweight="bold" if st is not None else "normal")
                cx += w_
            cx += sp
def swatch(x, y, w, text, color, bg):
    ax.add_patch(Rectangle((x, y + 0.012), w, LH - 0.02, fc=rgb(bg), ec="none", zorder=z()))
    label(x + 0.012, y, w, LH, text, BODY_PT, color=color, bold=True)

X0, Y0 = MARGIN, MARGIN
ax.imshow(np.asarray(IMG), extent=(X0, X0 + IMG_W, Y0 + IMG_H, Y0), zorder=z(), aspect="auto"); frame(X0, Y0, IMG_W, IMG_H)
tx0, tw0 = X0 + IMG_W + 0.10, CW_ - IMG_W - 0.10
label(tx0, Y0 + 0.015, tw0, 0.14, "Instruction", HEAD_PT, bold=True)
label(tx0, Y0 + 0.150, tw0, 0.14, "“Please describe the image in detail.”", BODY_PT, italic=True)
ly = Y0 + IMG_H - (2 * LH + 0.02 if ANY_DIGIT else LH + 0.005)
swatch(tx0, ly, 0.205, "the", RED, RED_BG); label(tx0 + 0.255, ly, tw0 - 0.255, LH, "token repeating the one before it", BODY_PT, color=NOTE)
if ANY_DIGIT:
    ly2 = ly + LH + 0.01
    swatch(tx0, ly2, 0.205, "00", BLUE, BLUE_BG); label(tx0 + 0.255, ly2, tw0 - 0.255, LH, "digits of a number written in the image", BODY_PT, color=NOTE)
y = Y0 + IMG_H + GAP
for name, lines, best, digit in rows:
    frame(X0, y, CW_, HEAD_H, lc=None, fc=HEAD_BG); frame(X0, y, CW_, BOX); rule(X0, y + HEAD_H, X0 + CW_, y + HEAD_H)
    label(X0 + PAD, y, 2.2, HEAD_H, name, HEAD_PT, bold=True)
    if best >= 2: note, col = "longest repeated run: %d tokens" % best, RED
    elif digit: note, col = "repeats: digits of the signs only", BLUE
    else: note, col = "no repetition", GREEN
    label(X0 + CW_ - PAD - 2.0, y, 2.0, HEAD_H, note, BODY_PT, color=col, italic=True, bold=(best < 2), align="right")
    words(X0 + PAD, y + HEAD_H + 0.025, lines)
    y += BOX + GAP
base = os.path.join(HERE, "preview_fail_" + CASE) if PREVIEW else os.path.join(PAPER, "images", "fig-failure-case")
fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=400)
print("written: %s.pdf/.png (%.2f x %.2f in)" % (base, W, H))
for name, lines, best, digit in rows: print(name, "lines", len(lines), "longest non-digit run", best, "digit runs", digit)
