#!/usr/bin/env python
"""Turn table5_provenance.json into PROVENANCE.md: one row per Table V cell, naming the runs behind it.

Usage: /opt/anaconda3/bin/python make_provenance_md.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
prov = json.load(open(os.path.join(HERE, "table5_provenance.json")))
MODELS = ["LLaDA-V", "MMaDA", "LaViDa"]
METHODS = ["Vanilla", "dLLM-Cache", "dLLM-Cache+CoTA", "dLLM-Cache+CoTA++",
           "SlowFast", "SlowFast+CoTA", "SlowFast+CoTA++"]
LENS = [512, 128, 64]


def components(meta):
    """Read the component switches back out of a run's stored configuration."""
    if not meta:
        return ""
    bits = []
    ctae = (meta.get("ctae") or {}).get("mode")
    if ctae and ctae != "off":
        x = meta.get("ctarx") or {}
        bits.append(f"CTAR({ctae},theta={x.get('theta')},w={x.get('w')})")
    dar = meta.get("dar") or {}
    r = dar.get("r") if isinstance(dar, dict) else None
    if r:
        bits.append(f"DAR(r={r},{dar.get('mode', 'legacy')})")
    ctev = meta.get("ctev") or {}
    if ctev.get("mode") and ctev.get("mode") != "off":
        bits.append(f"CTEV({ctev['mode']},lam={ctev.get('lambda')})")
    return " + ".join(bits) if bits else "none"


rows, gaps = [], []
for m in MODELS:
    for meth in METHODS:
        for L in LENS:
            c = prov["cells"].get(f"{m}|{meth}|{L}")
            if c is None:
                continue
            runs = c["runs"]
            where = "<br>".join(f"`{r['dir']}` ({r['lines']}, {r['finished']})" for r in runs) or "—"
            comp = components(runs[0]["meta"]) if runs else ""
            cell = c["cell"]
            val = f"{cell['arr']:.2f} / {cell['srr']:.1f} / {cell['mrl']:.0f}" if cell else "pending"
            rows.append(f"| {m} | {meth} | {L} | {c['n']}/500 | {val} | {comp} | {where} |")
            if not c["complete"]:
                gaps.append(f"- {m} {meth} L={L}: {c['n']}/500 images")

md = [
    "# Table V provenance",
    "",
    f"Generated from `table5_provenance.json` (built {prov['built']} on the server).",
    "Regenerate with `bash information_flow/results/sync_from_server.sh`.",
    "",
    f"- Metric: {prov['metric']}",
    f"- Image list: `{prov['images']['file']}`, {prov['images']['n']} images, sha1 `{prov['images']['sha1']}`",
    "- Values below are ARR / SRR / MRL; the full cell (with ARL, 95pRL, distinct-n, seq-rep-n, length)",
    "  is in `table5_cells.json`.",
    "",
    "## Code behind the runs",
    "",
    "| file | sha1 |",
    "|---|---|",
] + [f"| `{k}` | `{v}` |" for k, v in prov["code"].items()] + [
    "",
    "## Cells",
    "",
    "| Model | Method | L | Images | ARR/SRR/MRL | Components | Result directories (lines, finished) |",
    "|---|---|---|---|---|---|---|",
] + rows + [
    "",
    "## Still pending",
    "",
] + (gaps or ["- none"]) + [""]

open(os.path.join(HERE, "PROVENANCE.md"), "w").write("\n".join(md))
print("PROVENANCE.md written:", len(rows), "cells,", len(gaps), "pending")
