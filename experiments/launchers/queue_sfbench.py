#!/usr/bin/env python
"""General benchmarks under SlowFast (2026-09-30): writes jobs/sfbench.txt for scripts/sched_cmd.py.
Three models x three configurations (SlowFast, + CoTA, + CoTA++) x the benchmarks of Table VIII, one process per GPU.
DocVQA is run for LLaDA-V only (not reported for MMaDA and LaViDa, as in Table VIII). Task-major order, so that a
benchmark completes across the nine rows at once; LLaVA-Bench first, since its answers still have to be judged.
Start:  COTA_MAXPER=1 COTA_FREE_MIB=4000 nohup python -u scripts/sched_cmd.py jobs/sfbench.txt sfbench > logs/sfbench_driver.log 2>&1 &
"""
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
TASKS = ["chartqa_local", "mmstar_local", "mme_local", "mmbench_en_dev_local", "mathvista_testmini_cot_local",
         "mathverse_testmini_vision_only_local", "mathverse_testmini_vision_intensive_local",
         "mathverse_testmini_vision_dominant_local", "seedbench_local", "docvqa_val_local"]
CFGS = ["sf", "sfcota", "sfcpp"]
L = ["# general benchmarks under SlowFast, queued 2026-09-30 by scripts/launchers/queue_sfbench.py"]
for m in ("lladav", "lavida", "mmada"):
    for c in CFGS:
        L.append(f"sfb_lb_{m}_{c} llavabench_sf/{m}/{c}/DONE bash $E/scripts/launchers/lb_sf_one.sh {m} {c}")
for t in TASKS:
    for c in CFGS:
        L.append(f"sfb_lladav_{c}_{t} bench_sf/{c}/{t}/DONE bash $E/scripts/launchers/bench_eval_sf.sh {c} {t}")
    if t.startswith("docvqa"):
        continue
    for c in CFGS:
        L.append(f"sfb_lavida_{c}_{t} bench_sf_lavida/{c}/{t}/DONE bash $E/scripts/launchers/bench_eval_model_sf.sh lavida {c} {t}")
    for c in CFGS:
        L.append(f"sfb_mmada_{c}_{t} bench_sf_mmada_mixcot/{c}/{t}/DONE bash $E/scripts/launchers/bench_eval_model_sf.sh mmada {c} {t}")
L.append("END")
open(E + "/jobs/sfbench.txt", "w").write("\n".join(L) + "\n")
print("queued", len(L) - 2, "jobs in jobs/sfbench.txt")
