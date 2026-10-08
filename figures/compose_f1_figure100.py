#!/usr/bin/env python
"""Fallback composer for the merged F1 figure that does not need PowerPoint.

Same layout as build_f1_slide.py (17 cm wide): row 1 = panels/a_row.png with caption (a),
row 2 = panels/{b,c,d}_*.png with captions (b)-(d). Captions are set in the same serif
family as the panels. Writes results/F1/fig-f1-combined.pdf (vector text, raster panels).
Run with /opt/anaconda3/bin/python.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
ROOT = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/F1"
P = f"{ROOT}/panels"
CM = 1 / 2.54
W = 17.0 * CM; M = 0.25 * CM; CAP = 10          # width, side margin (in), caption pt

a = mpimg.imread(f"{P}/a_row.png")
bcd = [mpimg.imread(f"{ROOT}/panels100/{n}.png") for n in ("b_regional", "c_trajectory", "d_dose")]
aw = W - 2 * M; ah = aw * a.shape[0] / a.shape[1]
gap = 0.35 * CM; pw = (aw - 2 * gap) / 3; ph = max(pw * im.shape[0] / im.shape[1] for im in bcd)
cap_h = 0.55 * CM; pad = 0.15 * CM
H = pad + ah + 0.02 * CM + cap_h + ph + 0.02 * CM + cap_h + pad

fig = plt.figure(figsize=(W, H))
def place(im, x, y_top, w):                       # y_top measured from the top, in inches
    h = w * im.shape[0] / im.shape[1]
    ax = fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H]); ax.imshow(im); ax.axis("off")
    return h
def caption(text, x, y_top, w):
    fig.text((x + w / 2) / W, 1 - (y_top + 0.28 * CM) / H, text, ha="center", va="center", fontsize=CAP)

y = pad
h = place(a, M, y, aw); y += h + 0.02 * CM
caption("(a) Within-step information flow of vanilla LLaDA-V: each query anchors on its nearby context tokens", M, y, aw)
y += cap_h
caps = ["(b) Regional loss of anchoring", "(c) Compounding along decoding", "(d) Loss versus missed anchors"]
for k, (im, c) in enumerate(zip(bcd, caps)):
    x = M + k * (pw + gap); place(im, x, y, pw)
for k, c in enumerate(caps):
    caption(c, M + k * (pw + gap), y + ph + 0.02 * CM, pw)
fig.savefig(f"{ROOT}/fig-f1-combined-100.pdf", dpi=400); fig.savefig(f"{ROOT}/fig-f1-combined-100.png", dpi=200)
print("composed fig-f1-combined-100.pdf  %.1f x %.1f cm" % (W / CM, H / CM))
