#!/usr/bin/env python
"""Panel (a) of the merged F1 figure: within-step attention of vanilla LLaDA-V.

Source: information_flow/token-level/baseline/layer{L}/step{t}.jpg (3.4k x 3.4k renders,
128 x 128 query-key window, per-panel colour scale). The heatmap body is cropped from each
render and re-plotted with the paper's fonts; the colour scale is per panel, so one
qualitative viridis bar (low -> high) serves the row.

Outputs (results/F1/panels/): a_row.{pdf,png} (four maps + bar) and a_L{L}_S{t}.png singles.
Run with /opt/anaconda3/bin/python.
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.8,
})
SRC = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/token-level/baseline"
OUT = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/information_flow/results/F1/panels"
PANELS = [(1, 0), (22, 0), (1, 127), (22, 127)]      # (layer 1-indexed, step 0-indexed)

def body(path):
    """Crop the heatmap square: the largest block of rows/cols dominated by viridis pixels."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    vir = (b > 60) & (r < 120) & (g < 200) & ~((r > 200) & (g > 200) & (b > 200))   # viridis-ish, not white
    col = vir.mean(axis=0) > 0.6
    row = vir.mean(axis=1) > 0.6
    cs = np.where(col)[0]; rs = np.where(row)[0]
    # the colorbar is a narrow viridis strip to the right: keep the widest contiguous column run
    runs = []; s = cs[0]
    for a, b2 in zip(cs[:-1], cs[1:]):
        if b2 != a + 1: runs.append((s, a)); s = b2
    runs.append((s, cs[-1]))
    c0, c1 = max(runs, key=lambda t: t[1] - t[0])
    r0, r1 = rs[0], rs[-1]
    return im[r0:r1 + 1, c0:c1 + 1].astype(np.uint8)

crops = [body(f"{SRC}/layer{L}/step{t}.jpg") for L, t in PANELS]
for (L, t), c in zip(PANELS, crops):
    Image.fromarray(c).save(f"{OUT}/a_L{L}_S{t+1}.png")
    print(f"layer {L} step {t+1}: body {c.shape[1]}x{c.shape[0]} px")

fig = plt.figure(figsize=(12.2, 3.6))
gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.035, right=0.958, top=0.87, bottom=0.11)
axes = []
for k, ((L, t), c) in enumerate(zip(PANELS, crops)):
    ax = fig.add_subplot(gs[0, k]); axes.append(ax)
    ax.imshow(c, interpolation="lanczos", aspect="equal")
    ax.set_title(f"Layer {L}, Step {t + 1}", fontsize=13, pad=5)
    ax.set_xticks([0, c.shape[1] // 2, c.shape[1] - 1]); ax.set_xticklabels(["1", "64", "128"], fontsize=10)
    ax.set_xlabel("Key token", fontsize=11.5, labelpad=1.5)
    if k == 0:
        ax.set_yticks([0, c.shape[0] // 2, c.shape[0] - 1]); ax.set_yticklabels(["1", "64", "128"], fontsize=10)
        ax.set_ylabel("Query token", fontsize=11.5, labelpad=1.5)
    else:
        ax.set_yticks([])
    ax.tick_params(length=2.5, width=0.7)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
# colour bar hugging the last map
fig.canvas.draw()
pos = axes[-1].get_position()
cax = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.011, pos.height])
grad = np.linspace(1, 0, 256).reshape(-1, 1)
cax.imshow(grad, cmap=cm.viridis, aspect="auto")
cax.set_xticks([]); cax.set_yticks([0, 255]); cax.set_yticklabels(["high", "low"], fontsize=10)
cax.yaxis.tick_right(); cax.tick_params(length=0, pad=2)
cax.set_ylabel("Attention weight", fontsize=11, rotation=270, labelpad=13); cax.yaxis.set_label_position("right")
for sp in cax.spines.values():
    sp.set_linewidth(0.8)
fig.savefig(f"{OUT}/a_row.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/a_row.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
print("saved a_row.{pdf,png}")
