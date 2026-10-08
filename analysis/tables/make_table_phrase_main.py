#!/usr/bin/env python
"""Main-text phrase-level table (single column): Dist-2 and Rep-4 at the three lengths and their averages.

It is cut out of the appendix table (the <<TAB6FULL>> block of CoTA-claude.tex), cell by cell, so that the two
tables cannot disagree: same numbers, same bold marks. It reads the tex and prints the block; it writes nothing.

    /opt/anaconda3/bin/python make_table_phrase_main.py > table_phrase_main.tex
"""
import io, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "..", "..", "..", "TPAMI", "TPAMI_CoTA", "CoTA-claude.tex")
CAPTION = (r"\textbf{Phrase-level repetition} in the setting of Table~\ref{tabrepeat}. D-2 and R-4 are distinct-$2$ and "
           r"seq-rep-$4$ (\S\ref{setDetails}), in percent. Bold: best within a backend group. Shaded: CoTA++.")

def main():
    tex = io.open(TEX, encoding="utf-8").read()
    blk = re.search(r"% <<TAB6FULL-BEGIN>>.*?% <<TAB6FULL-END>>", tex, flags=re.S).group(0)
    body = blk.split(r"\midrule", 1)[1].rsplit(r"\bottomrule", 1)[0]
    out = [r"% <<TAB6MAIN-BEGIN>>  cut out of the <<TAB6FULL>> block by information_flow/results/table5/make_table_phrase_main.py",
           r"\begin{table}[!t]", r"\centering", r"\caption{%s}" % CAPTION, r"\label{tabphrasemain}", r"\scriptsize",
           r"\setlength{\tabcolsep}{2.4pt}", r"\renewcommand{\arraystretch}{1.0}",
           r"\begin{tabular}{@{}c l cc c cc c cc c cc}", r"\toprule",
           r"& & \multicolumn{2}{c}{$L{=}512$} & & \multicolumn{2}{c}{$L{=}128$} & & \multicolumn{2}{c}{$L{=}64$} & & \multicolumn{2}{c}{Avg.} \\",
           r"\cmidrule(lr){3-4} \cmidrule(lr){6-7} \cmidrule(lr){9-10} \cmidrule(l){12-13}",
           r"Model & Method & D-2$\uparrow$ & R-4$\downarrow$ &  & D-2$\uparrow$ & R-4$\downarrow$ &  & D-2$\uparrow$ & R-4$\downarrow$ &  & D-2$\uparrow$ & R-4$\downarrow$ \\",
           r"\midrule"]
    for line in body.strip().split("\n"):
        line = line.strip()
        if line.startswith(r"\cmidrule"): out.append(r"\cmidrule(l){2-13}"); continue
        if line.startswith(r"\midrule"): out.append(r"\midrule"); continue
        assert line.endswith(r"\\"), line
        c = [x.strip() for x in line[:-2].split("&")]
        assert len(c) == 22, (len(c), line)
        shade = r"\cellcolor{cotapp}" in line
        method = re.sub(r"~\\cite\{[^}]*\}", "", c[1])
        model = re.sub(r"~\\cite\{[^}]*\}", "", c[0])
        keep = [c[3], c[5], None, c[9], c[11], None, c[15], c[17], None, c[20], c[21]]
        cells = [model, method] + [(r"\cellcolor{cotapp}" if shade else "") if k is None else k for k in keep]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\vspace{-4mm}", r"\end{table}", r"% <<TAB6MAIN-END>>"]
    print("\n".join(out))

if __name__ == "__main__":
    main()
