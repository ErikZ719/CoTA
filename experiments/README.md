# Experiment manifests

The job files are the exact command lines the scheduler ran (`eval/scheduler/sched_cmd.py`, one job per line,
`END` closes the queue; `PY` / `PYMM` stand for the llada-v / mmada interpreters, `$E` for the experiment tree).
Lines starting with `#` are superseded or finished; `_r2` lines are re-runs after a code fix.

| File | Experiment |
|---|---|
| `jobs/ch6_resume.txt` | Repeat-Curse evaluation: every (model, backend, method, L) cell, in 1–4 shards (`--limit/--offset`) |
| `jobs/_archive/grid_final.txt` | the component grid (every subset of CTAR / DAR / CTEV) |
| `launchers/timing_quick64.sh`, `timing_suite.sh`, `timing_vanilla.sh`, `timing_vie.sh`, `timing_bands.sh` | efficiency: latency, throughput and memory (one job alone on one GPU) |
| `jobs/design100.txt`, `jobs/ctar_design.txt`, `jobs/ctev_design100.txt` | design alternatives of CTAR, DAR and CTEV, 100 images, L = 128 |
| `launchers/queue_exp3.py`, `smoke_exp3.sh` | decoding-time alternatives: n-gram penalty (`--ngram 2/3`), block lengths (`--block 16/32/64`), CTEV layer windows (`--ctev_layers`) |
| `jobs/sfbench.txt`, `launchers/queue_sfbench.py`, `queue_bench.py` | general benchmarks under SlowFast and the LLaDA-V benchmark queue |
| `jobs/_archive/budget_stress.txt`, `band.txt`, `th1.txt`, `es49_power.txt`, `arms_power.txt` | studies that shaped the method (budget pressure, CTAR band, θ=1 monitor, E_s = 49, routing arms) |
| `launchers/apply_exp3_patch.py`, `apply_ctevcache_patch.py`, `apply_rows_patch*.py` | the patch scripts that added `--ngram/--block/--ctev_layers`, the CTEV entropy cache and `--decode_rows` to the drivers; already applied in the shipped files, kept as documentation of each option |
| `jobs/COTA_GPUS`, `jobs/README_ch6_queue.txt` | scheduler configuration and queue notes |
