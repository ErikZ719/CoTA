#!/usr/bin/env python
"""Write results/INDEX.md: every run directory, what it is, and whether it feeds Table V.

Regenerate after new runs:  python scripts/analysis/make_results_index.py
Table V membership comes from results/table5_provenance.json (rerun tab5_build.py first)."""
import json, os, glob, time
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"; R = f"{E}/results"
PURPOSE = {
 "lladav/t5": "Table V runs on LLaDA-V. NOTE sfcotapp512_* is the three-component stack on SlowFast: kept for the appendix, not read by the table (the table reads sfguard/).",
 "lladav/sfguard": "SlowFast with CTAR + DAR + run guard (no CTEV). This IS the LLaDA-V SlowFast+CoTA++ row of Table V.",
 "lladav/sfdiag512": "SlowFast L=512 diagnostics on images 0-47: port identity, single components, fix variants. sfd512_ctardarrg completes the Table V cell.",
 "lladav/sfport": "SlowFast port validation (identity and component coverage).",
 "lladav/gf": "Component grid, L=512, images 0-47. ctarth_ctev_* and fullth_* complete Table V cells.",
 "lladav/gf128": "Component grid, L=128, images 60-259.",
 "lladav/kstep": "More than one token per step (k=2, 4) and DAR ranking-mode checks.",
 "lladav/darmode": "DAR ranking modes (legacy / score / masked) on SlowFast.",
 "lladav/bs2": "Budget stress with the monitor (E_s 21/49, alpha 0.10).",
 "lladav/bs": "Budget stress, first version (no monitor).",
 "lladav/ablation": "Cache-component ablation (e.g. alpha=1).",
 "lladav/th1": "CTAR monitor, theta=1.",
 "lladav/l512van": "Uncached L=512 slice, images 0-47. Feeds Table V.",
 "lladav/headline": "dLLM-Cache L=512 slice. Feeds Table V.",
 "lladav/l512x": "dLLM-Cache L=512 slice. Feeds Table V.",
 "lladav/l512t": "L=512 held-out slice from the abandoned dev/test protocol.",
 "lladav/lat": "Timing. Only solo runs are valid.", "lladav/lat2": "Timing. Only solo runs are valid.",
 "lladav/cost": "Timing. Only solo runs are valid.", "lladav/pareto": "Timing / quality trade-off.",
 "lladav/pareto512": "Timing / quality trade-off at L=512.",
 "mmada/t5": "Table V runs on MMaDA (blocks of 8, one token per step, cache 20/10/0.10).",
 "mmada/port": "Port validation. verify_sfcotapp64 is the single-writer check of a doubly written shard.",
 "mmada/supplement_262376": "Replacement images for the pool runs. L=128 entries feed Table V.",
 "lavida/t5": "Table V runs on LaViDa (alpha 0.10).",
 "lavida/port": "SlowFast-on-LaViDa port validation.",
 "_superseded": "Runs replaced after a code fix. Never report these.",
}
POOL = "1,000-image pool run (seeds 42/43). Table V reads its 500-image subset; ALWAYS filter to data/coco500_final.json."
prov = json.load(open(f"{R}/table5_provenance.json"))
feeds = {}
for key, c in prov["cells"].items():
    for r in c["runs"]:
        feeds.setdefault(r["dir"], []).append(key.replace("|", " / "))

def comp(meta):
    if not meta: return ""
    m = meta.get("sf_components") or meta.get("components") or {}
    bits = []
    ctae = (meta.get("ctae") or {}).get("mode") or m.get("ctae_mode")
    if m.get("ctar") or (ctae and ctae not in ("off", None) and ctae.startswith("reroute")): bits.append("CTAR")
    elif ctae and ctae != "off": bits.append(f"CTAE({ctae})")
    dar = meta.get("dar") or {}
    r = (dar.get("r") if isinstance(dar, dict) else 0) or m.get("dar_r") or 0
    if r: bits.append(f"DAR r={r}")
    ctev = (meta.get("ctev") or {}).get("mode") or m.get("ctev_mode")
    if ctev and ctev != "off": bits.append("CTEV")
    if m.get("sf_repguard"): bits.append(f"guard {m['sf_repguard']}")
    if m.get("sf_thresh") == "raw": bits.append("raw-threshold")
    return " + ".join(bits) or "none"

groups = {}
for p in sorted(glob.glob(f"{R}/**/outputs.jsonl", recursive=True)):
    d = os.path.dirname(p); rel = os.path.relpath(d, R); parts = rel.split("/")
    g = "_superseded" if parts[0] == "_superseded" else "/".join(parts[:2]) if len(parts) > 2 else parts[0]
    meta = {}
    try: meta = json.load(open(f"{d}/run_meta.json"))
    except Exception: pass
    gen = meta.get("gen_kwargs") or meta.get("gen") or {}
    L = gen.get("max_new_tokens") or gen.get("gen_length") or meta.get("length") or ""
    groups.setdefault(g, []).append(dict(rel=rel, n=sum(1 for _ in open(p)), done=os.path.exists(f"{d}/summary.json"),
        when=time.strftime("%m-%d", time.localtime(os.path.getmtime(p))), mode=meta.get("mode", ""), L=L,
        comp=comp(meta), feeds=feeds.get(rel, [])))

out = ["# Result directories", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/analysis/make_results_index.py`. Do not edit by hand.",
       "", "**In Table V** marks a run that the builder reads. Anything without the mark is not in the paper's main table,",
       "whatever its directory name suggests. Runs under `_superseded/` were replaced after a code fix.", ""]
for g in sorted(groups, key=lambda x: (x == "_superseded", x)):
    runs = groups[g]; nfeed = sum(1 for r in runs if r["feeds"])
    why = PURPOSE.get(g) or (POOL if any(k in g for k in ("baseline_L", "dllm_cache_L", "slowfast_L", "mmada_v2_")) else "Exploration before the Table V pipeline (see RESULTS_REGISTRY.md).")
    out += [f"## `{g}`  ({len(runs)} runs, {nfeed} in Table V)", "", why, "", "| run | images | L | backend | components | finished | in Table V |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        out.append(f"| `{r['rel']}` | {r['n']}{'' if r['done'] else ' (unfinished)'} | {r['L']} | {r['mode']} | {r['comp']} | {r['when']} | {'<br>'.join(r['feeds'])} |")
    out.append("")
open(f"{R}/INDEX.md", "w").write("\n".join(out))
print("INDEX.md:", sum(len(v) for v in groups.values()), "runs in", len(groups), "groups;",
      sum(1 for v in groups.values() for r in v if r["feeds"]), "feed Table V")
