#!/usr/bin/env python
"""Render Table V (tabrepeat) from table5_cells.json into CoTA-claude.tex, between the markers
% <<TAB5-BEGIN>> and % <<TAB5-END>>.

Adjacent-run statistics only (user decision 2026-09-17): ARR, SRR (per response) and MRL, ARL,
95pRL (runs pooled over the 500 responses), all lower-is-better, plus the mean ARR and SRR over the
three lengths. Headers carry arrows; bold = best within a backend group.
A final "Avg." group averages two headline metrics of each block over the three lengths.
Cells missing or computed on fewer than 500 images print '--'.
Usage: /opt/anaconda3/bin/python make_table5.py   (after copying table5_cells.json next to this file)
"""
import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = "/Users/zhaoqiyan/Desktop/TPAMI-CoTA++/TPAMI/TPAMI_CoTA/CoTA-claude.tex"
cells = json.load(open(os.path.join(HERE, "table5_cells.json")))
# LLM-judge means per cell (judge_table6.py). Absent until the judging has run: the column then prints '--'.
_jp = os.path.join(HERE, "judge_cells.json")
_j = json.load(open(_jp)) if os.path.exists(_jp) else {}
_pb = json.load(open(os.path.join(HERE, "table5_provenance.json")))["built"]
# --only TAB5,TAB6FULL renders only these blocks. Table VII (TAB6) holds judge scores the user edited by hand on
# 2026-09-30, so it is never regenerated unless named explicitly.
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
if _j and _j.get("provenance_built") != _pb and (ONLY is None or "TAB6" in ONLY):
    raise SystemExit("judge_cells.json was summarized from provenance %s, the table is built from %s: re-run judge_table6.py --summarize"
                     % (_j.get("provenance_built"), _pb))
JUDGE = _j.get("cells", {})
# Main-text Table VI shows this metric beside Len. "d1" until the judge scores cover all 63 cells, then "judge"
# (user decision 2026-09-21: the judge score replaces Dist-1 in the main text, Dist-1 stays in the appendix table).
TAB6_MAIN = "judge" if len(JUDGE) == 63 and all(v["n"] >= 475 for v in JUDGE.values()) else "d1"   # pilots (n=20) do not count
MODELS = ["LLaDA-V", "MMaDA", "LaViDa"]
CITE = {"LLaDA-V": "lladav", "MMaDA": "mmada", "LaViDa": "lavida"}
LENS = [512, 128, 64]
GROUPS = [("dLLM-Cache", "dLLM-Cache~\\cite{dLLM-Cache}"), ("SlowFast", "SlowFast~\\cite{SlowFast}")]
NA = "\\na"
# (key, header, format, direction): 'down' lower is better, 'up' higher is better, None = reference only (never bold)
TABLES = [
    dict(tag="TAB5", label="tabrepeat", env="table*", tabcolsep="2.6pt",
         metrics=[("arr", "ARR", "%.2f", "down"), ("srr", "SRR", "%.1f", "down"), ("mrl", "MRL", "%d", "down"),
                  ("arl", "ARL", "%.2f", "down"), ("p95", "95pRL", "%.1f", "down")],
         avg=[("arr", "ARR", "%.2f", "down"), ("srr", "SRR", "%.1f", "down")],
         caption=(r"\textbf{Repeat Curse across dMLLMs and caching backends} on the $500$ COCO images at three response "
                  r"lengths $L$, under the instruction ``Please describe the image in detail.'', on content tokens "
                  r"(\S\ref{setDetails}). ARR and SRR are in percent. MRL, ARL and 95pRL are the maximum, mean and 95th "
                  r"percentile of the lengths of all repeated runs in the $500$ responses. Avg.: mean over the three "
                  r"lengths. $\downarrow$: lower is better. Bold: best within a backend group. Shaded: CoTA++.")),
    # compact, single column, in the main text
    dict(tag="TAB6", label="tabphrase", env="table", place="!t", tabcolsep="2.4pt", stretch="1.0", vspace="-4mm", short_labels=True,
         **({"metrics": [("judge", "Judge", "%.2f", "up"), ("len", "Len", "%d", None)],
             "avg": [("judge", "Judge", "%.2f", "up")],
             "caption": (r"\textbf{Response quality and length} in the setting of Table~\ref{tabrepeat}. Judge is the mean "
                         r"score of the $500$ descriptions on a $1$--$10$ scale (\S\ref{setDetails}), Len the mean response "
                         r"length in tokens. Bold: best within a backend group. Shaded: CoTA++.")}
            if TAB6_MAIN == "judge" else
            {"metrics": [("d1", "Dist-1", "%.1f", "up"), ("len", "Len", "%d", None)],
             "avg": [("d1", "Dist-1", "%.1f", "up")],
             "caption": (r"\textbf{Response diversity and length} in the setting of Table~\ref{tabrepeat}. Dist-1 is distinct-$1$ "
                         r"of \S\ref{setDetails} in percent, Len the mean response length in tokens. Bold: best within a "
                         r"backend group. Shaded: CoTA++. Full statistics: Appendix~\ref{app:phrase}.")})),
    # complete version, in the appendix
    dict(tag="TAB6FULL", label="tabphrasefull", env="table*", tabcolsep="2.6pt",
         metrics=[("d1", "Dist-1", "%.1f", "up"), ("d2", "Dist-2", "%.1f", "up"), ("r3", "Rep-3", "%.1f", "down"),
                  ("r4", "Rep-4", "%.1f", "down"), ("len", "Len", "%d", None)],
         avg=[("d2", "Dist-2", "%.1f", "up"), ("r4", "Rep-4", "%.1f", "down")],
         caption=(r"\textbf{Phrase-level repetition and response length} in the setting of Table~\ref{tabrepeat}. "
                  r"Dist-$n$ and Rep-$n$ are distinct-$n$ and seq-rep-$n$ of \S\ref{setDetails}, in percent. Len is "
                  r"the mean response length in tokens. Avg.: mean over the three lengths. Bold: best within a backend "
                  r"group. Shaded: CoTA++.")),
]
PERCENT = ("d1", "d2", "r3", "r4")          # stored as fractions, shown in percent


def get(m, meth, L):
    c = cells.get(f"{m}|{meth}|{L}")
    if not c or c.get("n", 0) < 500 or not c.get("cell"):
        return None
    v = c["cell"]
    if not isinstance(v, dict):
        return None
    out = {k: (100.0 * x if k in PERCENT else x) for k, x in v.items()}
    j = JUDGE.get(f"{m}|{meth}|{L}")
    if j:
        out["judge"] = j["mean"]
    return out


def avg(vals, key):
    if any(vals[L] is None or key not in vals[L] for L in LENS):
        return None
    return sum(vals[L][key] for L in LENS) / len(LENS)


def fmt(v, f):
    return f % (round(v) if f == "%d" else v)


def row(label, vals, blk, mark, shade=False, first=""):
    """vals: {L: cell dict or None}; mark(slot, value) -> bool decides bold."""
    out = []
    for i, L in enumerate(LENS):
        if i:
            out.append("")
        for key, _, f, _d in blk["metrics"]:
            v = vals[L]
            if v is None or key not in v:
                out.append(NA); continue
            s = fmt(v[key], f)
            out.append("\\textbf{%s}" % s if mark((L, key), v[key]) else s)
    out.append("")
    for key, _, f, _d in blk["avg"]:
        a = avg(vals, key)
        if a is None:
            out.append(NA); continue
        s = fmt(a, f)
        out.append("\\textbf{%s}" % s if mark(("avg", key), a) else s)
    cols = [label] + out
    if shade:
        cols = ["\\cellcolor{cotapp}" + c for c in cols]
    return first + " & " + " & ".join(cols) + " \\\\"


def block_rows(blk):
    lines = []
    for mi, m in enumerate(MODELS):
        nrow = 1 + 3 * len(GROUPS)
        van = {L: get(m, "Vanilla", L) for L in LENS}
        vref = {}                                   # reference values of the uncached model
        for L in LENS:
            for key, _, _, _d in blk["metrics"]:
                vref[(L, key)] = van[L].get(key) if van[L] is not None else None
        for key, _, _, _d in blk["avg"]:
            vref[("avg", key)] = avg(van, key)
        lines.append(row("\\textcolor{refgray}{Vanilla}", van, blk, lambda s, x: False))
        for gi, (g, glabel) in enumerate(GROUPS):
            lines.append("\\cmidrule(l){2-%d}" % NCOLS)
            meths = (g, g + "+CoTA", g + "+CoTA++")
            vals = {mt: {L: get(m, mt, L) for L in LENS} for mt in meths}
            dirs = {k: d for k, _, _, d in blk["metrics"] + blk["avg"]}
            slots = [(L, k) for L in LENS for k, _, _, _d in blk["metrics"]] + [("avg", k) for k, _, _, _d in blk["avg"]]
            best = {}
            for sl in slots:
                xs = []
                for mt in meths:
                    if sl[0] == "avg":
                        xs.append(avg(vals[mt], sl[1]))
                    else:
                        c = vals[mt][sl[0]]
                        xs.append(None if c is None else c.get(sl[1]))
                d = dirs[sl[1]]
                if d is None or any(x is None for x in xs):
                    best[sl] = None
                else:
                    best[sl] = min(xs) if d == "down" else max(xs)
            mark = lambda sl, x, best=best: best.get(sl) is not None and abs(x - best[sl]) < 1e-9
            last = gi == len(GROUPS) - 1
            lines.append(row(g if blk.get("short_labels") else glabel, vals[g], blk, mark))
            lines.append(row("\\quad$+$\\,CoTA" + ("" if blk.get("short_labels") else "~\\cite{cota}"), vals[g + "+CoTA"], blk, mark))
            mname = m if blk.get("short_labels") else "%s~\\cite{%s}" % (m, CITE[m])
            first = ("\\multirow{-%d}{*}{\\rotatebox[origin=c]{90}{%s}}" % (nrow, mname)) if last else ""
            lines.append(row("\\quad$+$\\,\\textbf{CoTA++}", vals[g + "+CoTA++"], blk, mark, shade=True, first=first))
        if mi < len(MODELS) - 1:
            lines.append("\\midrule")
    return lines


NCOLS = 0          # set by render() for the table being drawn (block_rows draws its rules with it)


def lab(n, d):
    return n + {"down": "$\\downarrow$", "up": "$\\uparrow$", None: ""}[d]


def header(blk):
    names = []
    for i in range(3):
        if i:
            names.append("")
        names += [lab(n, d) for _, n, _, d in blk["metrics"]]
    names.append("")
    names += [lab(n, d) for _, n, _, d in blk["avg"]]
    return "Model & Method & " + " & ".join(names) + " \\\\"


def render(blk):
    global NCOLS
    nm, na = len(blk["metrics"]), len(blk["avg"])
    NCOLS = 2 + 3 * nm + 3 + na
    colspec = "@{}c l " + " c ".join(["c" * nm] * 3) + " c " + "c" * na
    starts = [3 + i * (nm + 1) for i in range(3)]; a0 = 3 + 3 * (nm + 1)
    top = "& & " + " & & ".join("\\multicolumn{%d}{c}{$L{=}%d$}" % (nm, L) for L in LENS) + " & & \\multicolumn{%d}{c}{Avg.} \\\\" % na
    rules = " ".join("\\cmidrule(lr){%d-%d}" % (st, st + nm - 1) for st in starts) + " \\cmidrule(l){%d-%d}" % (a0, a0 + na - 1)
    body = [header(blk), "\\midrule"] + block_rows(blk)
    env = blk["env"]
    return ("% <<" + blk["tag"] + "-BEGIN>>  generated by information_flow/results/table5/make_table5.py -- edit the script, not this block\n"
            "\\begin{" + env + "}[" + blk.get("place", "t") + "]\n\\centering\n\\providecommand{\\na}{\\textcolor{refgray}{--}}\n"
            "\\caption{" + blk["caption"] + "}\n\\label{" + blk["label"] + "}\n"
            "\\scriptsize\n\\setlength{\\tabcolsep}{" + blk["tabcolsep"] + "}\n\\renewcommand{\\arraystretch}{" + blk.get("stretch", "1.12") + "}\n"
            "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + top + "\n" + rules + "\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\vspace{" + blk.get("vspace", "-2mm") + "}\n\\end{" + env + "}\n% <<" + blk["tag"] + "-END>>")


s = open(TEX).read()
for blk in TABLES:
    if ONLY is not None and blk["tag"] not in ONLY: continue
    B, Eo = "% <<" + blk["tag"] + "-BEGIN>>", "% <<" + blk["tag"] + "-END>>"
    assert B in s and Eo in s, "markers for %s are missing in the paper: put '%s' and '%s' where the table belongs" % (blk["tag"], B, Eo)
    a = s.index(B); b = s.index(Eo) + len(Eo)
    assert b > a
    s = s[:a] + render(blk) + s[b:]
open(TEX, "w").write(s)
done = sum(1 for c in cells.values() if c.get("n", 0) >= 500 and isinstance(c.get("cell"), dict))
print("Tables written:", ", ".join(b["label"] for b in TABLES), "|", done, "complete cells")
