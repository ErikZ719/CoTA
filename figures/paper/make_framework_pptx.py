#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CoTA++ framework figure — editable PowerPoint slide, modern-teaser styling.

Structure (unchanged, it is the argument of the figure):
    one large DECODING TRAJECTORY canvas — suffix position on x, denoising step
    on y — on which the paper's three axes are literally three objects:
        routing        within one step   ->  a ROW
        timeliness     across steps      ->  a COLUMN
        consolidation  at the commit     ->  a CELL

Styling: sans throughout, saturated-but-soft palette, rounded cards on a tinted
ground, soft shadows, gradient fills, hairline connectors instead of arrows.

Run:  /opt/anaconda3/bin/python make_framework_pptx.py
Out:  CoTA++_framework.pptx
"""
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

# ------------------------------------------------------------------ palette --
def C(h):
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def T(h):
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


INDIGO, INDIGO_A, INDIGO_B = C('4C6FFF'), 'EEF2FF', 'DCE4FF'
AMBER,  AMBER_A,  AMBER_B = C('E8930C'), 'FFF6E5', 'FFEBC7'
CORAL,  CORAL_A,  CORAL_B = C('F2415F'), 'FFECF0', 'FFDBE3'
TEAL,   TEAL_A,   TEAL_B = C('0BA88B'), 'E6FAF5', 'CFF3EA'
VIOLET = C('8B5CF6')
INK, BODY, MUTE = C('0F172A'), C('47556A'), C('94A3B8')
HAIR, GROUND, CARD = C('E6EAF0'), C('F5F7FB'), C('FFFFFF')
SLATE_D, SLATE_M = C('1E293B'), C('64748B')
WHITE = C('FFFFFF')
SANS = 'Helvetica Neue'
FS = 1.50          # font multiplier: slide pt -> legible at 17 cm on paper

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
SH = slide.shapes


# ------------------------------------------------------------------ helpers --
def shp(kind, x, y, w, h, fc=None, lc=None, lw=0.8, dash=None, adj=None):
    s = SH.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if adj is not None:
        try:
            s.adjustments[0] = adj
        except Exception:
            pass
    if fc is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fc
    if lc is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = lc; s.line.width = Pt(lw)
        if dash:
            ln = s.line._get_or_add_ln()
            ln.append(ln.makeelement(qn('a:prstDash'), {'val': dash}))
    s.shadow.inherit = False
    return s


def rect(x, y, w, h, **kw):
    return shp(MSO_SHAPE.RECTANGLE, x, y, w, h, **kw)


def card(x, y, w, h, radius=0.06, **kw):
    return shp(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, adj=radius, **kw)


def oval(x, y, w, h, **kw):
    return shp(MSO_SHAPE.OVAL, x, y, w, h, **kw)


def gradfill(s, c1, c2, ang=5400000):
    spPr = s._element.spPr
    for tag in ('a:solidFill', 'a:noFill', 'a:gradFill', 'a:blipFill',
                'a:pattFill', 'a:grpFill'):
        el = spPr.find(qn(tag))
        if el is not None:
            spPr.remove(el)
    gf = spPr.makeelement(qn('a:gradFill'), {'rotWithShape': '1'})
    gsLst = gf.makeelement(qn('a:gsLst'), {})
    for pos, h in ((0, c1), (100000, c2)):
        gs = gf.makeelement(qn('a:gs'), {'pos': str(pos)})
        gs.append(gf.makeelement(qn('a:srgbClr'), {'val': h}))
        gsLst.append(gs)
    gf.append(gsLst)
    gf.append(gf.makeelement(qn('a:lin'), {'ang': str(ang), 'scaled': '0'}))
    spPr.insert_element_before(gf, 'a:ln', 'a:effectLst', 'a:scene3d', 'a:sp3d',
                               'a:extLst')
    return s


def shadow(s, blur=9, dist=3, alpha=11):
    spPr = s._element.spPr
    old = spPr.find(qn('a:effectLst'))
    if old is not None:
        spPr.remove(old)
    fx = spPr.makeelement(qn('a:effectLst'), {})
    sh = fx.makeelement(qn('a:outerShdw'),
                        {'blurRad': str(int(blur * 12700)),
                         'dist': str(int(dist * 12700)), 'dir': '5400000',
                         'rotWithShape': '0'})
    clr = fx.makeelement(qn('a:srgbClr'), {'val': '0F172A'})
    clr.append(fx.makeelement(qn('a:alpha'), {'val': str(int(alpha * 1000))}))
    sh.append(clr); fx.append(sh)
    spPr.insert_element_before(fx, 'a:scene3d', 'a:sp3d', 'a:extLst')
    return s


def text(x, y, w, h, runs, size=9, color=None, align=PP_ALIGN.LEFT, bold=False,
         italic=False, anchor=MSO_ANCHOR.MIDDLE, spacing=1.0, spc=None, rot=None,
         font=SANS):
    tb = SH.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if rot is not None:
        tb.rotation = rot
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    if isinstance(runs, str):
        runs = [(runs, color, bold, italic, None, None)]
    elif isinstance(runs, tuple):
        runs = [runs]
    for item in runs:
        t, c, b = item[0], item[1], item[2]
        it = item[3] if len(item) > 3 else italic
        base = item[4] if len(item) > 4 else None
        hl = item[5] if len(item) > 5 else None
        r = p.add_run(); r.text = t
        r.font.size = Pt(size * FS); r.font.name = font
        r.font.bold = b; r.font.italic = it
        r.font.color.rgb = c if c is not None else (color or BODY)
        rPr = r.font._rPr
        if base == 'sub':
            rPr.set('baseline', '-25000')
        elif base == 'sup':
            rPr.set('baseline', '30000')
        if spc:
            rPr.set('spc', str(spc))
        if hl is not None:
            el = rPr.makeelement(qn('a:highlight'), {})
            el.append(rPr.makeelement(qn('a:srgbClr'), {'val': hl}))
            rPr.insert_element_before(
                el, 'a:uLnTx', 'a:uLn', 'a:uFillTx', 'a:uFill', 'a:latin',
                'a:ea', 'a:cs', 'a:sym', 'a:hlinkClick', 'a:hlinkMouseOver',
                'a:rtl', 'a:extLst')
    return tb


def kick(x, y, w, s, color=MUTE, size=7.5):
    return text(x, y, w, 0.16, s.upper(), size=size, color=color, spc=120,
                bold=True)


def line(x1, y1, x2, y2, color=None, lw=0.75, dash=None, head=None):
    cn = SH.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                          Inches(x2), Inches(y2))
    cn.line.color.rgb = color or C('C7CFDB')
    cn.line.width = Pt(lw)
    ln = cn.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn('a:prstDash'), {'val': dash}))
    if head:
        ln.append(ln.makeelement(qn('a:tailEnd'),
                                 {'type': head, 'w': 'sm', 'len': 'med'}))
    return cn


def curve(pts, color, lw=1.6):
    b = SH.build_freeform(Inches(pts[0][0]), Inches(pts[0][1]))
    b.add_line_segments([(Inches(x), Inches(y)) for x, y in pts[1:]], close=False)
    s = b.convert_to_shape()
    s.fill.background(); s.line.color.rgb = color; s.line.width = Pt(lw)
    ln = s.line._get_or_add_ln()
    ln.append(ln.makeelement(qn('a:round'), {}))
    s.shadow.inherit = False
    return s


def ramp(v, c0, c1):
    v = max(0.0, min(1.0, v))
    return RGBColor(*[int(c0[k] + v * (c1[k] - c0[k])) for k in range(3)])


def grid(x, y, s, mat, c0=T('EEF1F6'), c1=T('4C6FFF')):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rect(x + j * s, y + i * s, s, s, fc=ramp(mat[i, j], c0, c1),
                 lc=WHITE, lw=0.25)


def IT(t, c=None): return (t, c, False, True, None, None)
def PL(t, c=None): return (t, c, False, False, None, None)
def BD(t, c=None): return (t, c, True, False, None, None)
def SUP(t, c=None): return (t, c, False, False, 'sup', None)
def SUB(t, c=None): return (t, c, False, False, 'sub', None)
def HL(t, c, h): return (t, c, True, False, None, h)


L, R = 0.34, 12.99
rect(0, 0, 13.333, 7.5, fc=GROUND, lc=None)          # tinted ground

# ================================================================== HEADER ===
oval(L, 0.20, 0.15, 0.15, fc=INDIGO, lc=None)
text(L + 0.26, 0.13, 3.0, 0.30, 'CoTA++', size=17, color=INK, bold=True)
text(L + 1.90, 0.16, 8.0, 0.26,
     'diagnosing and treating the Repeat Curse on three axes', size=10,
     color=MUTE)

# ============================================================ LEFT: INPUT ====
shadow(card(L, 0.66, 1.58, 3.26, radius=0.05, fc=CARD, lc=None))
kick(L + 0.16, 0.82, 1.3, 'input')
SH.add_picture('images/framework-input.jpg', Inches(L + 0.16), Inches(1.06),
               Inches(1.26), Inches(1.26))
card(L + 0.16, 1.06, 1.26, 1.26, radius=0.05, fc=None, lc=C('E3E8F0'), lw=0.8)
shadow(gradfill(card(L + 0.16, 2.46, 1.26, 0.60, radius=0.14, fc=CARD,
                     lc=None), TEAL_A, TEAL_B), blur=5, dist=2, alpha=9)
text(L + 0.24, 2.50, 1.10, 0.52, '"Describe the image\nin detail."', size=8.5,
     color=C('0A7A65'), align=PP_ALIGN.CENTER, spacing=1.08)
text(L + 0.16, 3.14, 1.30, 0.18, 'LLaDA-V', size=9, color=INK, bold=True)
text(L + 0.16, 3.33, 1.30, 0.18, '+ dLLM-Cache', size=9, color=AMBER, bold=True)
text(L + 0.16, 3.54, 1.30, 0.20,
     [IT('M'), PL(' = '), IT('N'), PL(' = 128')], size=8.5, color=MUTE)

# ====================================================== HERO: THE CANVAS =====
FX, FY = 2.52, 1.16
NC, NR = 32, 14
CW, CH = 0.186, 0.150
FW, FH = NC * CW, NR * CH
shadow(card(2.10, 0.66, 6.98, 3.26, radius=0.05, fc=CARD, lc=None))
kick(FX, 0.82, 3.6, 'the decoding trajectory')

rng = np.random.RandomState(17)
order = np.argsort(rng.rand(NC) * 0.95 + np.linspace(0, 1, NC))
commit_step = np.empty(NC, dtype=int)
for rank, col in enumerate(order):
    commit_step[col] = int(rank * (NR - 2) / (NC - 1)) + 1
refresh = rng.rand(NR, NC) < 0.26
REPEAT_COLS = [19, 24, 29]
for j in REPEAT_COLS:
    c = commit_step[j]
    for i in range(max(0, c - 5), c + 1):
        refresh[i, j] = False
stale = np.zeros((NR, NC), dtype=int)
for j in range(NC):
    run = 0
    for i in range(NR):
        run = 0 if refresh[i, j] else run + 1
        stale[i, j] = run

for i in range(NR):
    for j in range(NC):
        x, y = FX + j * CW, FY + i * CH
        if i > commit_step[j]:
            rect(x, y, CW, CH, fc=C('FAFBFD'), lc=C('F1F3F7'), lw=0.25)
        elif i == commit_step[j]:
            rect(x, y, CW, CH, fc=CORAL if j in REPEAT_COLS else SLATE_D,
                 lc=WHITE, lw=0.35)
        else:
            rect(x, y, CW, CH, fc=ramp(min(stale[i, j], 5) / 5.0,
                                       T('EDF0F5'), T('7C8BA3')), lc=WHITE,
                 lw=0.25)
rect(FX, FY, FW, FH, fc=None, lc=C('DDE3EC'), lw=0.8)
text(FX, FY + FH + 0.07, FW, 0.17, 'suffix position', size=7.5, color=MUTE)
text(1.30, FY + FH / 2 - 0.09, 1.40, 0.18, 'denoising step  t', size=7.5,
     color=MUTE, align=PP_ALIGN.CENTER, rot=270)

lg = FX + FW - 2.82
for k, (lab, fc) in enumerate([('fresh', ramp(0.0, T('EDF0F5'), T('7C8BA3'))),
                               ('stale', ramp(1.0, T('EDF0F5'), T('7C8BA3'))),
                               ('commit', SLATE_D), ('repeat', CORAL)]):
    rect(lg + k * 0.70, FY + FH + 0.30, 0.11, 0.11, fc=fc, lc=C('DDE3EC'),
         lw=0.35)
    text(lg + k * 0.70 + 0.16, FY + FH + 0.26, 0.50, 0.18, lab, size=7.5,
         color=MUTE)

ROW_I, COL_J = 6, 24
CELL = (commit_step[29], 29)


def badge(cx, cy, n, col, d=0.24):
    shadow(oval(cx - d / 2, cy - d / 2, d, d, fc=col, lc=WHITE, lw=1.2),
           blur=5, dist=1, alpha=22)
    text(cx - d / 2, cy - d / 2 + 0.015, d, d - 0.03, n, size=8.5, color=WHITE,
         bold=True, align=PP_ALIGN.CENTER)


rect(FX - 0.12, FY + ROW_I * CH, 0.08, CH, fc=INDIGO, lc=None)
rect(FX + FW + 0.04, FY + ROW_I * CH, 0.08, CH, fc=INDIGO, lc=None)
rect(FX - 0.03, FY + ROW_I * CH - 0.03, FW + 0.06, CH + 0.06, fc=None,
     lc=INDIGO, lw=2.0)
rect(FX + COL_J * CW, FY - 0.12, CW, 0.08, fc=AMBER, lc=None)
rect(FX + COL_J * CW, FY + FH + 0.04, CW, 0.08, fc=AMBER, lc=None)
rect(FX + COL_J * CW - 0.03, FY - 0.03, CW + 0.06, FH + 0.06, fc=None, lc=AMBER,
     lw=2.0)
oval(FX + CELL[1] * CW - 0.08, FY + CELL[0] * CH - 0.08, CW + 0.16, CH + 0.16,
     fc=None, lc=CORAL, lw=1.6)

badge(FX + FW + 0.24, FY + ROW_I * CH + CH / 2, '1', INDIGO)
badge(FX + COL_J * CW + CW / 2, FY - 0.22, '2', AMBER)
badge(FX + CELL[1] * CW + CW + 0.22, FY + CELL[0] * CH + CH / 2, '3', CORAL)

# ======================================================== RIGHT: THE TEXT ====
OX = 9.28
kick(OX + 0.18, 0.82, 3.4, 'what comes out')
shadow(gradfill(card(OX, 1.02, R - OX, 1.36, radius=0.05, fc=CARD, lc=None),
                CORAL_A, CORAL_B))
rect(OX + 0.18, 1.16, 0.10, 0.10, fc=CORAL, lc=None)
text(OX + 0.36, 1.10, 2.4, 0.20, 'the Repeat Curse', size=9, color=C('B02444'),
     bold=True)
text(OX + 0.18, 1.36, R - OX - 0.36, 0.92,
     [PL('… towards '), HL('the the', C('B02444'), 'FFC4CF'), PL(' horizon. '),
      HL('The The', C('B02444'), 'FFC4CF'), PL(' sky '),
      HL('is is', C('B02444'), 'FFC4CF'), PL(' a soft blue.')], size=9.5,
     color=C('4A2B34'),
     anchor=MSO_ANCHOR.TOP, spacing=1.16)

shadow(gradfill(card(OX, 2.54, R - OX, 1.38, radius=0.05, fc=CARD, lc=None),
                TEAL_A, TEAL_B))
rect(OX + 0.18, 2.68, 0.10, 0.10, fc=TEAL, lc=None)
text(OX + 0.36, 2.62, 2.4, 0.20, '+ CoTA++', size=9, color=C('07705C'),
     bold=True)
text(OX + 0.18, 2.88, R - OX - 0.36, 0.72,
     '… towards the horizon.', size=9.5, color=C('204A42'),
     anchor=MSO_ANCHOR.TOP, spacing=1.16)

# ================================================== BOTTOM: THE THREE ========
BY, BH, PW = 4.06, 3.12, 4.10
PXS = [L, L + 4.28, L + 8.56]
ACC = [INDIGO, AMBER, CORAL]
TINT = [(INDIGO_A, INDIGO_B), (AMBER_A, AMBER_B), (CORAL_A, CORAL_B)]
NAMES = [('CTAR', 'Context-Token Attention Re-anchoring'),
         ('DAR', 'Decode-Aware Refresh'),
         ('CTEV', 'Context-Token Entropy-Guided Voting')]
OBJ = [('a ROW', 'routing within one step'),
       ('a COLUMN', 'timeliness across steps'),
       ('a CELL', 'consolidation at the commit')]

for k in range(3):
    px = PXS[k]
    shadow(card(px, BY, PW, BH, radius=0.045, fc=CARD, lc=None))
    gradfill(card(px, BY, PW, 0.66, radius=0.13, fc=CARD, lc=None), TINT[k][0],
             TINT[k][1])
    badge(px + 0.36, BY + 0.33, str(k + 1), ACC[k], d=0.28)
    text(px + 0.62, BY + 0.20, 1.6, 0.30, NAMES[k][0], size=14.5, color=ACC[k],
         bold=True)
    text(px + 0.20, BY + 0.82, PW - 0.40, 0.18,
         [BD(OBJ[k][0], ACC[k]), ('    ' + OBJ[k][1], MUTE, False, False, None,
                                  None)], size=8)

VZ, EQ, TL = BY + 1.14, BY + 2.40, BY + 2.76

# ------------------------------------------------------------- 1 · CTAR -----
px = PXS[0]
m, gs = 9, 0.112
ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
rr = np.random.RandomState(7)
An = np.exp(-((ii - jj) ** 2) / (2 * 0.92 ** 2)); An /= An.max()
Ao = 0.40 * np.exp(-((ii - jj - 2.4) ** 2) / (2 * 2.9 ** 2)) + 0.46 * rr.rand(m, m)
Ao /= Ao.max()
grid(px + 0.34, VZ, gs, Ao, T('EEF1F6'), T('9AA6B8'))
rect(px + 0.34, VZ, gs * m, gs * m, fc=None, lc=C('DDE3EC'), lw=0.8)
grid(px + 2.66, VZ, gs, An, T('EEF2FF'), T('4C6FFF'))
rect(px + 2.66, VZ, gs * m, gs * m, fc=None, lc=INDIGO, lw=1.0)
text(px + 0.22, VZ + gs * m + 0.06, 1.36, 0.17, 'cached',
     size=8.5, color=MUTE, align=PP_ALIGN.CENTER)
text(px + 2.54, VZ + gs * m + 0.06, 1.36, 0.17, 're-anchored',
     size=8.5, color=INDIGO, align=PP_ALIGN.CENTER)
line(px + 1.62, VZ + gs * m / 2, px + 2.60, VZ + gs * m / 2, color=INDIGO,
     lw=1.2, head='triangle')
text(px + 1.52, VZ + gs * m / 2 - 0.24, 1.18, 0.17, 'recompute', size=8.5,
     color=INDIGO, align=PP_ALIGN.CENTER)
gradfill(card(px + 0.20, EQ, PW - 0.40, 0.32, radius=0.16, fc=CARD, lc=None),
         INDIGO_A, INDIGO_B)
text(px + 0.26, EQ, PW - 0.52, 0.32,
     [IT('A'), SUB('i'), PL(' = softmax('), IT('q'), SUB('i'), IT('K'),
      SUP('T'), PL(' / √'), IT('d'), PL('),   '), IT('q'), SUB('i'),
      PL(' = '), IT('W'), SUB('Q'), PL('LN('), IT('h'), SUB('i'), PL(')')],
     size=8.0,
     color=C('27356B'), align=PP_ALIGN.CENTER)
text(px + 0.20, TL, PW - 0.40, 0.24,
     [BD('9.9×', INDIGO), PL('  anchoring lost at repeats')],
     size=9, color=BODY)

# -------------------------------------------------------------- 2 · DAR -----
px = PXS[1]
cs, cg, nc = 0.272, 0.048, 11
rx = px + 0.30
text(px + 0.20, VZ - 0.06, 2.2, 0.17, [PL('by drift  '), IT('S'), SUP('(t)')],
     size=8.5, color=MUTE)
for i in range(nc):
    card(rx + i * (cs + cg), VZ + 0.16, cs, cs, radius=0.22, fc=C('EEF1F6'),
         lc=None)
for i in range(nc):
    xx = rx + i * (cs + cg)
    if i < 2:
        card(xx, VZ + 0.76, cs, cs, radius=0.22, fc=AMBER, lc=None)
    elif i >= nc - 2:
        card(xx, VZ + 0.76, cs, cs, radius=0.22, fc=C('F7F8FA'), lc=C('E1E5EC'),
             lw=0.7, dash='sysDash')
        line(xx + 0.06, VZ + 0.82, xx + cs - 0.06, VZ + 0.76 + cs - 0.06,
             color=C('D3D8E0'), lw=0.8)
        line(xx + 0.06, VZ + 0.76 + cs - 0.06, xx + cs - 0.06, VZ + 0.82,
             color=C('D3D8E0'), lw=0.8)
    else:
        card(xx, VZ + 0.76, cs, cs, radius=0.22, fc=C('EEF1F6'), lc=None)
line(rx + 0.13, VZ + 0.48, rx + 0.13, VZ + 0.72, color=AMBER, lw=1.2,
     head='triangle')
line(rx + (nc - 1.5) * (cs + cg), VZ + 0.48, rx + (nc - 1.5) * (cs + cg),
     VZ + 0.72, color=C('C7CFDB'), lw=1.2, head='triangle')
text(rx + 0.26, VZ + 0.52, 1.44, 0.17, 'imminent', size=8.5, color=AMBER)
text(rx + (nc - 4.8) * (cs + cg), VZ + 0.52, 1.26, 0.17, 'least drift', size=8.5,
     color=MUTE, align=PP_ALIGN.RIGHT)
text(px + 0.20, VZ + 1.04, 3.6, 0.17,
     [BD('decode-aware  ', AMBER), IT('R', AMBER), SUP('(t)', AMBER),
      ('    same size', MUTE, False, False, None, None)], size=8.5)
gradfill(card(px + 0.20, EQ, PW - 0.40, 0.32, radius=0.16, fc=CARD, lc=None),
         AMBER_A, AMBER_B)
text(px + 0.26, EQ, PW - 0.52, 0.32,
     [IT('R'), SUP('(t)'), PL(' = '), IT('ℐ'), SUP('(t)'), PL(' ∪ Drop(|'),
      IT('ℐ'), SUP('(t)'), PL('∖'), IT('S'), SUP('(t)'), PL('|, '), IT('S'),
      SUP('(t)'), PL(')')], size=7.5, color=C('6B4B0C'),
     align=PP_ALIGN.CENTER)
text(px + 0.20, TL, PW - 0.40, 0.24,
     [BD('2×', AMBER), PL('  repeats decode on stale state')], size=9,
     color=BODY)

# ------------------------------------------------------------- 3 · CTEV -----
px = PXS[2]
ex, ey, ew, eh = px + 0.30, VZ + 0.02, 1.46, 0.94
rect(ex + (25 / 31) * ew, ey, (6 / 31) * ew, eh, fc=C('EEF2FF'), lc=None)
line(ex, ey + eh, ex + ew, ey + eh, color=C('DDE3EC'), lw=0.7)
line(ex, ey + eh, ex, ey, color=C('DDE3EC'), lw=0.7)
lay = np.arange(1, 33)
nrm = np.where(lay < 24, 12.6 - 0.03 * lay, 12.0 * np.exp(-(lay - 23) / 2.6) + 0.9)
rep = 13.1 + 0.35 * np.sin(lay / 3.0)
sc = lambda v: ey + eh - (v / 14.0) * eh
curve([(ex + (l - 1) / 31 * ew, sc(v)) for l, v in zip(lay, rep)], CORAL, 1.8)
curve([(ex + (l - 1) / 31 * ew, sc(v)) for l, v in zip(lay, nrm)], INDIGO, 1.8)
text(ex + ew - 0.68, ey + 0.01, 0.66, 0.16, 'repeat', size=8, color=CORAL,
     bold=True, align=PP_ALIGN.RIGHT)
text(ex + ew - 0.68, ey + eh - 0.18, 0.66, 0.16, 'normal', size=8, color=INDIGO,
     bold=True, align=PP_ALIGN.RIGHT)
text(ex, ey + eh + 0.06, ew, 0.16, 'entropy vs. layer', size=8, color=MUTE,
     align=PP_ALIGN.CENTER)
bx0 = ex + ew + 0.34
for cc, (vals, red_i) in enumerate([([1.00, 0.84, 0.70, 0.56], 0),
                                    ([0.84, 0.70, 0.62, 0.56], 2)]):
    ox = bx0 + cc * 1.04
    for r_, v in enumerate(vals):
        card(ox + 0.13, ey + r_ * 0.222, v, 0.155, radius=0.40,
             fc=CORAL if r_ == red_i else C('DCE1E9'), lc=None)
        text(ox - 0.04, ey + r_ * 0.222 - 0.01, 0.14, 0.18, str(r_ + 1), size=6.5,
             color=MUTE, align=PP_ALIGN.RIGHT)
    line(ox + 0.11, ey - 0.02, ox + 0.11, ey + 0.90, color=C('E1E5EC'), lw=0.7)
text(bx0 - 0.06, ey + eh + 0.06, 1.00, 0.16, [PL('by  '), IT('c'), SUB('(i)')],
     size=7, color=MUTE, align=PP_ALIGN.CENTER)
text(bx0 + 0.98, ey + eh + 0.06, 1.04, 0.16,
     [PL('by  Score(', CORAL), IT('i', CORAL), PL(')', CORAL)], size=7,
     align=PP_ALIGN.CENTER)
line(bx0 + 0.96, ey + 0.40, bx0 + 1.06, ey + 0.40, color=CORAL, lw=1.2,
     head='triangle')
gradfill(card(px + 0.20, EQ, PW - 0.40, 0.32, radius=0.16, fc=CARD, lc=None),
         CORAL_A, CORAL_B)
text(px + 0.26, EQ, PW - 0.52, 0.32,
     [PL('Score('), IT('i'), PL(') = '), IT('c'), SUB('(i)'), PL(' − '),
      IT('λ'), PL(' · '), IT('Ē'), SUB('ctx'), PL('('), IT('i'), PL(')')],
     size=8.5, color=C('8A2540'), align=PP_ALIGN.CENTER)
text(px + 0.20, TL, PW - 0.40, 0.24,
     [BD('+3.6 bits', CORAL), PL('  entropy at a repeat commit')],
     size=9, color=BODY)

prs.save('_tools/CoTA++_framework.pptx')
print('written: CoTA++_framework.pptx')
