#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CoTA++ method framework figure (TPAMI, double-column width).

(a) Inference pipeline of a cached dMLLM, left to right: image and instruction
    -> tokens -> token-token attention -> approximate K/V cache -> confidence
    decoding, looping back for the next denoising step.
(b) The three CoTA++ components, each editing one structure at the site marked
    by the matching circled number in (a).

Tokens are drawn as small squares throughout. Palette follows Fig.5 v4.
Run:  /opt/anaconda3/bin/python make_framework_fig.py
Out:  images/fig-framework.pdf  (+ .png preview)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

BLUE, AMBER, RED = '#3D5A98', '#C0872F', '#D9534F'
BLUE_F, AMBER_F, RED_F = '#DCE3EF', '#F4E7CF', '#F7DEDD'
GRAY, LGRAY, INK = '#8A8A8A', '#C9C9C9', '#242424'
VIS, INS, CMT = '#C6D2E3', '#E2DCD1', '#8F9EB8'
cmap_blue = LinearSegmentedColormap.from_list('cb', ['#FFFFFF', '#B9C7E0', BLUE])

W, H = 170.0, 98.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect('equal'); ax.axis('off')

SQ, GP = 2.05, 0.28


def box(x, y, w, h, fc='none', ec=GRAY, lw=0.6, r=0.8, z=2, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=%g' % r,
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))


def sq(x, y, fc='white', ec=GRAY, lw=0.45, z=4, s=SQ, ls='-'):
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=ec, lw=lw, zorder=z,
                           linestyle=ls))


def txt(x, y, s, size=6.0, color=INK, ha='center', va='center', weight='normal',
        style='normal', z=7):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z,
            fontweight=weight, fontstyle=style)


def arrow(x0, y0, x1, y1, color=GRAY, lw=0.85, z=5, style='-|>', ms=5.5,
          conn='arc3,rad=0', ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=color, zorder=z,
                                 connectionstyle=conn, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def pin(x, y, label, color, z=10, r=2.0, fs=5.2):
    ax.add_patch(Circle((x, y), r, fc='white', ec=color, lw=1.0, zorder=z))
    txt(x, y - 0.05, label, size=fs, color=color, weight='bold', z=z + 1)


def brace(x0, x1, y, label, color=GRAY, size=5.2, drop=0.85):
    ax.plot([x0, x0, x1, x1], [y, y - drop, y - drop, y], color=color, lw=0.5,
            zorder=4)
    txt((x0 + x1) / 2, y - drop - 1.7, label, size=size, color=color)


# ====================================================== (a) INFERENCE FLOW ==
txt(3.0, 94.6, '(a)', size=8.2, weight='bold', ha='left')
for cx, lab in [(15.5, 'input'), (56.0, 'tokens'), (93.2, 'attention'),
                (120.8, 'approximate cache'), (152.0, 'confidence decode')]:
    txt(cx, 94.6, lab, size=6.3, color=GRAY, style='italic')

# ---- 1. image + instruction ------------------------------------------------
img = mpimg.imread('images/framework-input.jpg')
ax.imshow(img, extent=(5.0, 26.0, 70.5, 91.5), zorder=3, aspect='auto',
          interpolation='bilinear')
ax.add_patch(Rectangle((5.0, 70.5), 21.0, 21.0, fc='none', ec=GRAY, lw=0.6,
                       zorder=4))
box(3.0, 61.6, 25.0, 6.4, fc='#FAFAFA', ec=LGRAY, lw=0.6, r=0.9, z=3)
txt(15.5, 65.9, 'Please describe the', size=5.6, color='#3A3A3A')
txt(15.5, 63.3, 'image in detail.', size=5.6, color='#3A3A3A')

# ---- 2. tokens -------------------------------------------------------------
NV, NI, NM = 8, 3, 8
seq_x, seq_y = 34.0, 77.6
x = seq_x
for _ in range(NV):
    sq(x, seq_y, fc=VIS, ec=BLUE, lw=0.4); x += SQ + GP
vis_end = x - GP
for _ in range(NI):
    sq(x, seq_y, fc=INS, ec='#9A9186', lw=0.4); x += SQ + GP
ins_end = x - GP
for i in range(NM):
    if i < 2:
        sq(x, seq_y, fc=CMT, ec='#6E7B92', lw=0.4)
    else:
        sq(x, seq_y, fc='white', ec=LGRAY, lw=0.45, ls=(0, (1.3, 1.0)))
    x += SQ + GP
seq_end = x - GP
lab_y = seq_y + SQ + 1.9
txt((seq_x + vis_end) / 2, lab_y, 'visual', size=5.5, color=BLUE)
txt((vis_end + GP + ins_end) / 2, lab_y, 'instr.', size=5.5, color='#9A9186')
txt((ins_end + GP + seq_end) / 2, lab_y, 'masked suffix', size=5.5, color=GRAY)
arrow(26.8, 80.5, 33.0, 79.2, color=GRAY, conn='arc3,rad=0.10')
arrow(28.8, 64.8, 33.0, 77.4, color=GRAY, conn='arc3,rad=-0.22')

# ---- 3. token-token attention ----------------------------------------------
n = 13
ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
rng = np.random.RandomState(5)
A = np.exp(-((ii - jj) ** 2) / (2 * 1.30 ** 2)) + 0.03 * rng.rand(n, n)
A /= A.sum(1, keepdims=True)
ax_, ay_, asz = 84.0, 71.0, 18.2
for k in (2, 1):
    ax.add_patch(Rectangle((ax_ + k * 1.35, ay_ + k * 1.35), asz, asz, fc='white',
                           ec=LGRAY, lw=0.55, zorder=3))
ax.imshow(A, cmap=cmap_blue, aspect='auto', zorder=4,
          extent=(ax_, ax_ + asz, ay_, ay_ + asz), interpolation='nearest')
ax.add_patch(Rectangle((ax_, ay_), asz, asz, fc='none', ec=BLUE, lw=0.8, zorder=5))
txt(ax_ + asz / 2, ay_ - 2.4, r'$A=\mathrm{softmax}(QK^{\top}\!/\sqrt{d})$ per layer',
    size=5.7, color='#3A3A3A')
txt(ax_ + asz / 2, ay_ - 5.1, r'deep band $\ell\in[25,32]$', size=5.7, color=BLUE,
    weight='bold')
arrow(seq_end + 1.6, 78.6, ax_ - 1.6, 79.6, color=GRAY)

# ---- 4. approximate K/V cache ----------------------------------------------
cx0, cy0, ncol = 108.5, 76.4, 11
box(cx0 - 1.5, cy0 - 1.6, ncol * (SQ + GP) + 2.7, 2 * (SQ + GP) + 3.0,
    fc='white', ec=LGRAY, lw=0.6, r=0.9, z=2)
stale = [0, 3, 1, 0, 5, 2, 0, 4, 1, 6, 2, 1, 0, 2, 5, 0, 3, 1, 4, 0, 2, 6]
for i, s in enumerate(stale[:2 * ncol]):
    r_, c_ = divmod(i, ncol)
    g = 1.0 - min(s, 6) / 7.5
    sq(cx0 + c_ * (SQ + GP), cy0 + (1 - r_) * (SQ + GP),
       fc=(g * 0.76 + 0.24, g * 0.78 + 0.22, g * 0.82 + 0.18), ec='white', lw=0.35)
txt(cx0 + ncol * (SQ + GP) / 2 - 0.3, cy0 - 3.4,
    r'reuse; refresh $\lfloor\alpha M\rfloor$ per step', size=5.7, color='#3A3A3A')
txt(cx0 + ncol * (SQ + GP) / 2 - 0.3, cy0 - 6.1, 'darker = staler', size=5.5,
    color=GRAY)
arrow(ax_ + asz + 1.6, 79.6, cx0 - 3.0, 79.6, color=GRAY)

# ---- 5. confidence decoding ------------------------------------------------
dx0, dtop = 138.0, 87.0
for i, v in enumerate([20.0, 16.6, 13.2, 9.9, 7.0]):
    yy = dtop - i * 2.85
    ax.add_patch(Rectangle((dx0 + 2.4, yy), v, 1.95,
                           fc='#8494AE' if i < 2 else '#BAC3D0', ec='none',
                           zorder=4))
ax.plot([dx0 + 2.0, dx0 + 2.0], [dtop - 4 * 2.85 - 0.4, dtop + 2.3], color=LGRAY,
        lw=0.5, zorder=3)
cut = dtop - 1.45 * 2.85 + 1.95
ax.plot([dx0 + 1.2, dx0 + 23.4], [cut, cut], color=RED, lw=0.6,
        ls=(0, (2.0, 1.5)), zorder=6)
txt(dx0 + 24.2, cut, r'top-$k$', size=5.5, color=RED, ha='left')
for i in range(5):
    sq(dx0 + 2.4 + i * (SQ + GP), 71.6,
       fc=CMT if i < 2 else 'white', ec='#6E7B92' if i < 2 else LGRAY, lw=0.45,
       ls='-' if i < 2 else (0, (1.3, 1.0)))
txt(dx0 + 14.6, 71.6 + SQ / 2, 'newly committed', size=5.5, color=GRAY, ha='left')
arrow(cx0 + ncol * (SQ + GP) + 1.0, 79.6, dx0 - 1.0, 79.6, color=GRAY)

# ---- recurrence ------------------------------------------------------------
ax.plot([dx0 + 3.0, dx0 + 3.0, 56.0], [70.4, 62.4, 62.4], color=GRAY, lw=0.75,
        ls=(0, (2.6, 2.0)), zorder=4)
arrow(56.0, 62.4, 56.0, 76.4, color=GRAY, lw=0.75, ms=5.5, ls=(0, (2.6, 2.0)))
txt(102.0, 60.6, r'next denoising step $t\!+\!1$', size=5.8, color=GRAY,
    style='italic')

# ---- pins ------------------------------------------------------------------
pin(ax_ + asz + 1.9, ay_ + asz + 3.4, '1', BLUE)
pin(cx0 + ncol * (SQ + GP) - 0.4, cy0 + 2 * (SQ + GP) + 1.0, '2', AMBER)
pin(dx0 - 0.2, dtop + 3.6, '3', RED)


# ========================================================== (b) COMPONENTS ==
txt(3.0, 55.4, '(b)', size=8.2, weight='bold', ha='left')

PW, PGAP, PX0, PY0, PY1, HDR = 52.0, 5.0, 4.0, 8.0, 51.6, 6.2


def panel(ix, tag, name, full, color, fillc):
    px = PX0 + ix * (PW + PGAP)
    box(px, PY0, PW, PY1 - PY0, fc='white', ec=color, lw=0.85, r=1.3, z=2)
    box(px, PY1 - HDR, PW, HDR, fc=fillc, ec=color, lw=0.85, r=1.3, z=3)
    ax.add_patch(Rectangle((px + 0.45, PY1 - HDR), PW - 0.9, 1.2, fc=fillc,
                           ec='none', zorder=4))
    pin(px + 4.4, PY1 - HDR / 2, tag, color, z=6, r=1.95)
    txt(px + 8.2, PY1 - HDR / 2, name, size=8.2, color=color, weight='bold',
        ha='left', z=7)
    txt(px + PW - 2.2, PY1 - HDR / 2, full, size=5.4, color=color, ha='right',
        z=7, style='italic')
    return px


# --------------------------------------------------------------- 1. CTAR ----
p0 = panel(0, '1', 'CTAR', 'F1 $\\cdot$ routing', BLUE, BLUE_F)
m = 13
ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
rng = np.random.RandomState(7)
A_new = np.exp(-((ii - jj) ** 2) / (2 * 1.25 ** 2)) + 0.03 * rng.rand(m, m)
A_new /= A_new.sum(1, keepdims=True)
A_old = (0.5 * np.exp(-((ii - jj - 3.5) ** 2) / (2 * 4.0 ** 2))
         + 0.5 * rng.rand(m, m))
A_old /= A_old.sum(1, keepdims=True)
hy, hs = 25.6, 16.0
for k, (Am, lab, ec) in enumerate([(A_old, 'replayed', GRAY),
                                   (A_new, 're-anchored', BLUE)]):
    hx = p0 + (4.0 if k == 0 else 32.0)
    ax.imshow(Am, cmap=cmap_blue, vmin=0, vmax=A_new.max() * 0.92, aspect='auto',
              zorder=3, extent=(hx, hx + hs, hy, hy + hs), interpolation='nearest')
    ax.add_patch(Rectangle((hx, hy), hs, hs, fc='none', ec=ec, lw=0.7, zorder=4))
    txt(hx + hs / 2, hy - 2.4, lab, size=5.7, color=GRAY if k == 0 else BLUE)
arrow(p0 + 21.0, hy + hs / 2, p0 + 30.8, hy + hs / 2, color=BLUE, lw=1.0, ms=6)
txt(p0 + 26.0, hy + hs / 2 + 2.2, 'recompute', size=5.4, color=BLUE)
txt(p0 + PW / 2, 17.6, r'$A_i=\mathrm{softmax}(q_iK^{\top}\!/\sqrt{d})$,   '
                      r'$q_i=W_Q\mathrm{LN}(h_i)$', size=6.6)
txt(p0 + PW / 2, 12.6, r'All $M$ suffix positions, deep band only.', size=6.3)

# ---------------------------------------------------------------- 2. DAR ----
p1 = panel(1, '2', 'DAR', 'F2 $\\cdot$ timeliness', AMBER, AMBER_F)
nc, S2 = 10, 4.0
rx, rowA, rowB = p1 + 5.0, 35.4, 26.8
txt(rx, rowA + S2 + 2.0, r'backend set $S^{(t)}$  (by drift)', size=5.8,
    ha='left', color='#3A3A3A')
for i in range(nc):
    sq(rx + i * (S2 + 0.5), rowA, fc='#DEE0E4', ec='white', lw=0.5, s=S2)
for i in range(nc):
    xx = rx + i * (S2 + 0.5)
    if i < 2:
        sq(xx, rowB, fc=AMBER, ec='white', lw=0.5, s=S2)
    elif i >= nc - 2:
        sq(xx, rowB, fc='#F6F6F6', ec=LGRAY, lw=0.5, s=S2, ls=(0, (1.2, 1.0)))
        ax.plot([xx + 0.8, xx + S2 - 0.8], [rowB + 0.8, rowB + S2 - 0.8],
                color=LGRAY, lw=0.6, zorder=6)
        ax.plot([xx + 0.8, xx + S2 - 0.8], [rowB + S2 - 0.8, rowB + 0.8],
                color=LGRAY, lw=0.6, zorder=6)
    else:
        sq(xx, rowB, fc='#DEE0E4', ec='white', lw=0.5, s=S2)
arrow(rx + 4.2, rowA - 0.4, rx + 4.2, rowB + S2 + 0.4, color=AMBER, lw=0.9, ms=5)
arrow(rx + nc * (S2 + 0.5) - 5.2, rowA - 0.4, rx + nc * (S2 + 0.5) - 5.2,
      rowB + S2 + 0.4, color=LGRAY, lw=0.9, ms=5)
txt(rx + 7.6, (rowA + rowB + S2) / 2, 'admitted', size=5.5, color=AMBER, ha='left')
txt(rx + nc * (S2 + 0.5) - 7.6, (rowA + rowB + S2) / 2, 'dropped', size=5.5,
    color=GRAY, ha='right')
txt(rx, rowB - 2.6, r'CoTA++ set $R^{(t)}$', size=5.8, ha='left', color=AMBER)
txt(p1 + PW / 2, 17.6, r'$R^{(t)}=\mathcal{I}^{(t)}\cup\mathrm{Drop}'
                       r'(\vert\mathcal{I}^{(t)}\backslash S^{(t)}\vert,S^{(t)})$',
    size=6.6)
txt(p1 + PW / 2, 12.6, r'Same budget $b$, given to imminent positions.', size=6.3)

# --------------------------------------------------------------- 3. CTEV ----
p2 = panel(2, '3', 'CTEV', 'F3 $\\cdot$ consolidation', RED, RED_F)
bh2, bg2, ty = 2.5, 1.0, 39.8
lv, rv = [17.0, 14.6, 12.4, 10.0, 7.6], [14.6, 12.4, 11.0, 10.0, 7.6]


def bars(x0, vals, red_i, tag):
    for r, v in enumerate(vals):
        yy = ty - r * (bh2 + bg2)
        ax.add_patch(Rectangle((x0 + 2.2, yy), v, bh2,
                               fc=RED if r == red_i else '#C9CCD3', ec='none',
                               zorder=4))
        txt(x0 + 1.6, yy + bh2 / 2, '%d' % (r + 1), size=5.0, color=GRAY,
            ha='right')
        if r == red_i and tag:
            txt(x0 + 3.2, yy + bh2 / 2, r'high $\bar{E}_{\mathrm{ctx}}$', size=5.0,
                color='white', ha='left', z=8)
    ax.plot([x0 + 1.9, x0 + 1.9], [ty - 4 * (bh2 + bg2) - 0.4, ty + bh2 + 0.4],
            color=LGRAY, lw=0.5, zorder=3)


lcx, rcx = p2 + 2.6, p2 + 27.6
bars(lcx, lv, 0, True)
bars(rcx, rv, 2, False)
arrow(p2 + 23.4, ty - 5.0, p2 + 28.4, ty - 5.0, color=RED, lw=1.0, ms=6)
txt(p2 + 25.9, ty - 2.6, 'defer', size=5.4, color=RED)
txt(lcx + 10.5, 23.2, r'by $c_{(i)}$', size=5.8, color='#3A3A3A')
txt(rcx + 10.5, 23.2, r'by $\mathrm{Score}(i)$', size=5.8, color=RED)
txt(p2 + PW / 2, 17.6,
    r'$\mathrm{Score}(i)=c_{(i)}-\lambda\,\bar{E}_{\mathrm{ctx}}(i)$', size=6.8)
txt(p2 + PW / 2, 12.6, 'Defers a token, never forbids it.', size=6.3)

out = 'images/fig-framework'
fig.savefig(out + '.pdf', dpi=600)
fig.savefig(out + '.png', dpi=300)
print('written:', out + '.pdf / .png')
