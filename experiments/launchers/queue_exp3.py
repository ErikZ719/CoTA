#!/usr/bin/env python
"""Job lines for the three main-text experiments (2026-09-30), LLaDA-V, L = 128, the 500 images of Table VII in two
shards of 250, appended to jobs/sfbench.txt before END (the running scheduler re-reads the file).

  (a) n-gram penalty baseline: dLLM-Cache + no-repeat 2-gram / 3-gram             results/lladav/exp3/ng{2,3}_128_*
  (b) block-length sweep 16/32/64 (128 = existing cells): cache, CoTA++, vanilla   dcB*, cppB*, vanB*
  (c) CTEV layer window with the full CoTA++ stack (26-30 = existing cell)         cppL<lo>_<hi>_128_*
      and, last, CTEV alone (cache + CTEV) over the same windows                   ctevL<lo>_<hi>_128_*
CoTA++ = the Table VII configuration: --dar_r 4 --dar_mode legacy --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31
--ctar_theta 1 --ctev_mode ctx --ctev_lambda 0.25.
"""
import io, os, sys

E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
JOBF = E + "/jobs/sfbench.txt"
BASE = "PY -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json --length 128 --limit 250"
CPP = "--dar_r 4 --dar_mode legacy --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --ctev_mode ctx --ctev_lambda 0.25"
WINDOWS = ["1-8", "9-24", "25-32", "22-32", "30-32"]          # 26-30 is the existing CoTA++ cell

def job(name, extra):
    lines = []
    for off in (0, 250):
        out = "lladav/exp3/%s_128_%d" % (name, off)
        lines.append("x3_%s_%d %s/summary.json %s --offset %d --out $E/results/%s %s" % (name, off, out, BASE, off, out, extra))
    return lines

L = ["# 2026-09-30 12:10 (user): three main-text experiments replacing appendix items 10-12 (queue_exp3.py). LLaDA-V, L=128."]
L += ["# (a) n-gram penalty baseline under dLLM-Cache"]
for n in (2, 3):
    L += job("ng%d" % n, "--mode dllm_cache --ngram %d" % n)
L += ["# (b) block-length sweep: cache and CoTA++ (B = 128 are the Table VII cells)"]
for B in (16, 32, 64):
    L += job("dcB%d" % B, "--mode dllm_cache --block %d" % B)
    L += job("cppB%d" % B, "--mode dllm_cache --block %d %s" % (B, CPP))
L += ["# (c) CTEV layer window with the full CoTA++ stack (26-30 = Table VII cell)"]
for w in WINDOWS:
    L += job("cppL%s" % w.replace("-", "_"), "--mode dllm_cache --ctev_layers %s %s" % (w, CPP))
L += ["# (b') uncached model at the same block lengths"]
for B in (16, 32, 64):
    L += job("vanB%d" % B, "--mode baseline --block %d" % B)
L += ["# (c') CTEV alone (cache + CTEV, no CTAR/DAR) over the windows, diagnostic"]
for w in WINDOWS + ["26-30"]:
    L += job("ctevL%s" % w.replace("-", "_"), "--mode dllm_cache --ctev_mode ctx --ctev_lambda 0.25 --ctev_layers %s" % w)

if __name__ == "__main__":
    text = "\n".join(L) + "\n"
    if "--print" in sys.argv:
        print(text); sys.exit(0)
    src = io.open(JOBF).read()
    if "queue_exp3.py" in src:
        sys.exit("already queued")
    assert src.rstrip().endswith("END"), "job file does not end with END"
    head = src.rstrip()[:-3]
    io.open(JOBF, "w").write(head + text + "END\n")
    print("queued", sum(1 for l in L if not l.startswith("#")), "jobs into", JOBF)
