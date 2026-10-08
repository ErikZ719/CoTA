"""Table VI production: four configurations x benchmarks, released generation settings, 1 process per job."""
import sys
JF = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/jobs/ch6_resume.txt"
TASKS = sys.argv[1:] or ["chartqa_local", "mme_local", "mmbench_en_dev_local", "docvqa_val_local"]
L = open(JF).read().rstrip("\n").split("\n")
have = {l.split(None, 1)[0] for l in L if l.strip() and not l.startswith("#")}
new = []
for t in TASKS:                                   # task-major order: a benchmark completes across all four rows at once
    for c in ("van", "cache", "cota", "cpp"):
        n = f"bench_{c}_{t}"
        if n not in have:
            new.append(f"{n} bench/{c}/{t}/DONE bash $E/scripts/launchers/bench_eval.sh {c} {t}")
i = 0
while i < len(L) and (not L[i].strip() or L[i].startswith("#")): i += 1
while i < len(L) and L[i].startswith(("bsmoke", "bench_")): i += 1      # after what is already queued
L[i:i] = new; open(JF, "w").write("\n".join(L) + "\n"); print("queued", len(new), "jobs:", TASKS)
