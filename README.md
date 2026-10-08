# CoTA++: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective

CoTA++ is a training-free, plug-and-play framework that mitigates the *Repeat Curse* of cached diffusion
multimodal LLMs (dMLLMs). It extends [CoTA](https://openreview.net/pdf?id=mOz9jVYxsD) (*Context Tokens are Anchors*,
ICLR 2026).

Approximate KV caching (dLLM-Cache, SlowFast sampling) makes dMLLMs repeat themselves: under a cache, most long
descriptions contain a repeated run. CoTA++ traces this to three disruptions of the context-token information flow
and counteracts each where it originates:

| Axis | Finding | Component | What it does | Switch |
|---|---|---|---|---|
| Routing, within a step | F1: a position about to commit anchors less on its context tokens | **CTAR** | re-forms the attention routing of the deep band (layers 25–32) against the current keys when a context anchor has changed | `--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1` |
| Timeliness, across steps | F2: the similarity-based refresh is not staleness-aware | **DAR** | reserves part of the backend's refresh budget for the positions the decoder is about to commit | `--dar_r 4 --dar_mode legacy` |
| Consolidation, across layers | F3: repetitions commit from contexts whose deep-layer entropy has not converged | **CTEV** | discounts the commit confidence by the context entropy read at layers 26–30 | `--ctev_mode ctx --ctev_lambda 0.25` |

The three components act at different stages of one decoding step and compose with the backend as it is.
**CoTA** (the conference method) = CTAR + CTEV; **CoTA++** = CTAR + DAR + CTEV. Supported models: LLaDA-V-8B,
MMaDA-8B, LaViDa-8B; supported caching backends: dLLM-Cache and SlowFast sampling.

---

## 1. Repository layout

```
CoTA/
├── README.md                this file
├── LICENSE                  Apache-2.0
├── relocate.sh              rewrites the machine-specific roots inside the scripts (see §3)
├── environment/             pip freezes of the three environments + ENVIRONMENTS.md
├── llada_v/                 drop-in files for ML-GSAI/LLaDA-V: the components, the SlowFast port,
│                            the attention/entropy recorders, the lmms-eval wrappers  -> llada_v/README.md
├── mmada/                   the components for MMaDA-8B and LaViDa-8B (one file) + the SlowFast stack for MMaDA
├── eval/                    Repeat-Curse evaluation drivers for the three models, the metrics,
│                            the decode-row recorder, LLaVA-Bench generation, and the idle-GPU scheduler
├── benchmarks/              general multimodal benchmarks through lmms-eval
├── judge/                   LLM-as-judge scorers (description quality, LLaVA-Bench, MathVista/MathVerse)
├── analysis/
│   ├── tables/              aggregators that turn run directories into result tables
│   ├── findings/            per-position tables and statistics of the three findings F1–F3
│   ├── recording/           launchers for attention / entropy / decode-row recording
│   └── probes/              validation probes (routing fidelity, regression of the default path)
├── figures/                 plotting scripts (figures/paper: case-study and framework figures)
├── experiments/             the exact command lines of every experiment (job manifests and launchers)
└── data/                    the evaluation image lists
```

Nothing here is a package: the scripts run in place against the upstream repositories of §2.1 and an experiment
tree of run directories (§3). They were written for one machine and keep its absolute paths; `relocate.sh`
rewrites them in one pass.

---

## 2. Setup

### 2.1 Upstream repositories

The components are implemented inside the model repositories, not around them. Clone the pinned commits:

| Repository | Commit | Used for |
|---|---|---|
| [ML-GSAI/LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) | `f8b02ce` (2026-03-23) | LLaDA-V-8B, the dLLM-Cache hook, lmms-eval |
| [maomaocun/dLLM-cache](https://github.com/maomaocun/dLLM-cache) | `17235bf` | the MMaDA cache hook (`dllm_cache/hooks/cache_hook_MMaDA.py`) and `demo_MMada_mmu_cache.py` |
| [Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA) | `3cdb870` | MMaDA-8B-Base (repeat-curse evaluation) and MMaDA-8B-MixCoT (benchmarks) |
| [jacklishufan/LaViDa](https://github.com/jacklishufan/LaViDa) | `24220c0` | LaViDa-8B |
| [LiangrunFlora/Slow-Fast-Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling) | `e12b9de` | reference implementation of the SlowFast sampler (ported, not imported) |

Only LLaDA-V is modified. Everything for MMaDA and LaViDa lives in `mmada/` and the drivers, which load the
upstream code and insert a few lines at run time, so those repositories stay untouched.

```bash
ROOT=/path/to/workspace            # the directory that will hold the five repositories
cd $ROOT
git clone https://github.com/ML-GSAI/LLaDA-V.git && (cd LLaDA-V && git checkout f8b02ce)
git clone https://github.com/maomaocun/dLLM-cache.git && (cd dLLM-cache && git checkout 17235bf)
git clone https://github.com/Gen-Verse/MMaDA.git MMaDA-official && (cd MMaDA-official && git checkout 3cdb870)
git clone https://github.com/jacklishufan/LaViDa.git && (cd LaViDa && git checkout 24220c0)
git clone https://github.com/LiangrunFlora/Slow-Fast-Sampling.git && (cd Slow-Fast-Sampling && git checkout e12b9de)
bash /path/to/CoTA/llada_v/apply.sh $ROOT/LLaDA-V    # copies the drop-in files (prints what it overwrites)
```

### 2.2 Environments

Three Python environments are used; `environment/pip_freeze_*.txt` are their exact package lists and
`environment/ENVIRONMENTS.md` says how they were built.

| Environment | Python | torch | transformers | Runs |
|---|---|---|---|---|
| `llada-v` | 3.10 | 2.6.0+cu124 | 4.39.3 | LLaDA-V drivers, lmms-eval, recorders |
| `mmada` | 3.10 | 2.6.0+cu124 | 4.46.3 | MMaDA drivers, `tab5_build.py` (it is the only one that loads the MMaDA tokenizer) |
| `lavida` (venv over `mmada`) | 3.10 | 2.6.0+cu124 | 4.50.3 | LaViDa drivers |

The plotting scripts and the table renderer run on any machine with `matplotlib`, `numpy`, `scipy`, `python-pptx`
and `Pillow`.

### 2.3 Models, data, benchmarks

* Checkpoints (Hugging Face): `GSAI-ML/LLaDA-V`, `Gen-Verse/MMaDA-8B-Base`, `Gen-Verse/MMaDA-8B-MixCoT`,
  `jacklishufan/lavida-llada-v1.0-instruct`, plus the vision towers they pull (`google/siglip2-so400m-patch14-384`,
  `google/siglip-so400m-patch14-384`, `showlab/magvitv2`). Set `HF_HOME` and run offline once they are cached.
* COCO val2014 images (`http://images.cocodataset.org/zips/val2014.zip`), expected under `$COTA_ROOT/coco2014/val2014/`.
  The evaluation list is `data/coco500_final.json` (500 images; how it was constructed is in the file's `criterion`
  field), the diagnosis sample is `data/coco100_seed0.json` (100 images, 94 of which repeat under dLLM-Cache).
  The judge uses the five human captions of each image from the COCO annotations (Karpathy split).
* General benchmarks: ChartQA, DocVQA, MMStar, MME, SEED-Bench, MMBench, LLaVA-Bench (in-the-wild), MathVista,
  MathVerse as parquet files, fetched with `benchmarks/bench_download.sh` / `benchmarks/ms_download.py` and
  registered as `*_local` tasks (`llada_v/eval/lmms-eval/lmms_eval/tasks/*`).

---

## 3. Paths

Every script was written for one machine, whose roots are
`/data/zhaoqiyan/autodl-tmp/{LLaDA-V,dLLM-cache,LaViDa,MMaDA-official,hf_cache,datasets,coco2014,information_flow}`
and `/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval` (the experiment tree: `scripts/`, `results/`, `jobs/`, `logs/`).
Older scripts carry the same tree under `/root/autodl-tmp` or `/home/<user>/zhaoqiyan/autodl-tmp`; the plotting
scripts use `/opt/anaconda3/bin/python`. `relocate.sh` rewrites all of them:

```bash
COTA_ROOT=/path/to/workspace COTA_EXP=/path/to/workspace/experiments/repeat_eval COTA_PY=python bash relocate.sh
```

The experiment tree is the one the drivers write into: `$COTA_EXP/results/<model>/<group>/<run>/`
(`outputs.jsonl`, `run_meta.json`, `summary.json` per run), `$COTA_EXP/data/` (copy `data/*.json` there),
`$COTA_EXP/jobs/`, `$COTA_EXP/logs/`.

---

## 4. Running the method

### 4.1 One image (LLaDA-V, with attention maps)

`llada_v/train/generate_demo.py` decodes one image with dLLM-Cache and the components, and can render the
per-layer attention maps (`VISUALIZE_ATTENTION`, `VIS_EVERY_LAYER` at the top of the file).

### 4.2 Repeat-Curse evaluation

The drivers decode a list of images with the prompt *"Please describe the image in detail."*, write one line per
image to `outputs.jsonl` and the repetition metrics to `summary.json`. Every component is a flag that defaults to off,
so `--mode dllm_cache` with no component is the backend as released.

```bash
E=$COTA_EXP
# LLaDA-V, dLLM-Cache, L = 128
python eval/run_repeat_eval.py --images $E/data/coco500_final.json --mode dllm_cache --length 128 \
    --out $E/results/lladav/t5/dc128                                             # the cache alone
python eval/run_repeat_eval.py --images $E/data/coco500_final.json --mode dllm_cache --length 128 \
    --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 \
    --ctev_mode ctx --ctev_lambda 0.25 --out $E/results/lladav/t5/cota128        # + CoTA
python eval/run_repeat_eval.py --images $E/data/coco500_final.json --mode dllm_cache --length 128 \
    --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 \
    --dar_r 4 --dar_mode legacy --ctev_mode ctx --ctev_lambda 0.25 \
    --out $E/results/lladav/t5/cotapp128                                         # + CoTA++
# LLaDA-V, SlowFast: --mode slowfast; on this backend CoTA++ uses CTAR + DAR + the run guard, without CTEV
python eval/run_repeat_eval.py --images $E/data/coco500_final.json --mode slowfast --length 128 \
    --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --dar_r 4 --dar_mode legacy \
    --sf_repguard 0.25 --out $E/results/lladav/sfguard/g128
# MMaDA-8B-Base (blocks of 8, one token per step, cache 20/10/0.10); run inside the mmada environment
python eval/run_repeat_eval_mmada.py --images $E/data/coco500_final.json --cache_pi 20 --cache_gi 10 --cache_tr 0.1 \
    --mode dllm_cache --length 128 --steps 128 --block 8 --ctae_mode reroute_q --ctar_theta 1 \
    --dar_r 4 --dar_mode legacy --ctev_mode ctx --ctev_lambda 0.25 --out $E/results/mmada/t5/cotapp128
# LaViDa-8B (cache 25/7/0.10; blocks of 32 at L = 512); run from the LaViDa checkout with the lavida environment
python eval/run_repeat_eval_lavida.py --mode dllm_cache --cache_tr 0.10 --length 128 \
    --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --dar_r 4 --dar_mode legacy \
    --ctev_mode ctx --ctev_lambda 0.25 --out $E/results/lavida/t5/cpp128
```

`--limit N --offset K` shard a list across GPUs (a 500-image cell is 1–4 shards, pooled by the aggregator). The
command line of every evaluation cell is in `experiments/jobs/ch6_resume.txt`; the other experiments have their own
manifests (`experiments/README.md`).

Default settings and the flags that set them:

| Setting | LLaDA-V | MMaDA | LaViDa |
|---|---|---|---|
| dLLM-Cache intervals (prompt / suffix) and share α | 25 / 7 / 0.25 | 20 / 10 / 0.10 (0.0235 at L=512) | 25 / 7 / 0.10 (0.0235 at L=512) |
| decoding | one token per step, single block | blocks of 8 | single block; blocks of 32 at L=512 |
| CTAR | band 25–32 (`--stitch_lo 24 --stitch_hi 31`, 0-indexed), θ=1, w=5 | same | same |
| DAR | r=4, `legacy` ranking | same | same |
| CTEV | λ=0.25, window ±5, layers 26–30 | same | same |

Metrics (`eval/repeat_metrics.py`): ARR (adjacent repetition rate), SRR (share of responses with a repeat), MRL,
ARL and 95pRL (maximum, mean and 95th percentile of the repeated-run lengths), on content tokens (layout ids
`{198, 220, 197, 256, 262144}` stripped, response truncated at its first end-of-text token), plus seq-rep-n and
distinct-n. `--decode_rows DIR` additionally records the decode-moment attention rows of layers 25–32.

### 4.3 Many runs: the scheduler

`eval/scheduler/sched_cmd.py <jobfile> <logname>` launches the lines of a job file on idle GPUs, one job per
line, re-reading the file before every launch, and stops at `END`. A line is

```
<name> <done-marker relative to $E/results> <command ...>
```

with `PY`/`PYMM` standing for the llada-v / mmada interpreters and `CWD=<dir>` as an optional prefix. Restart it only
with `eval/scheduler/restart_sched.py`, which parks the lines of jobs that are still running so they are not launched
twice. `COTA_GPUS`, `COTA_MAXPER` and `COTA_FREE_MIB` set which cards it may use and how many jobs share one; one
LLaDA-V job per A100-80G is the safe setting (a job holds 29–43 GB at any length).

### 4.4 General benchmarks

`benchmarks/bench_eval.sh <van|cache|cota|cpp> <task>` evaluates LLaDA-V through lmms-eval with the released
generation settings of each task; the backend and the components reach the lmms-eval model through environment
variables that the launcher sets (`COTA_BACKEND`, `COTA_CTAE`, `COTA_CTAR_THETA`, `COTA_DAR_R`, `COTA_DAR_MODE`,
`COTA_CTEV_LAMBDA`, cache intervals `COTA_EP`/`COTA_ES`/`COTA_ALPHA`). `bench_eval_model.sh <lavida|mmada> ...`
does the same for the other two models through the wrappers `lavida_cota.py` / `mmada_cota.py`;
`bench_eval_sf.sh` and `bench_eval_model_sf.sh` are the SlowFast variants; `chain_mixcot.sh` runs MMaDA-8B-MixCoT.
MathVista and MathVerse answers are scored afterwards with `judge/score_math_from_samples.py`, LLaVA-Bench with
`eval/llavabench/` + `judge/score_llavabench.py`.

### 4.5 Response quality (LLM judge)

`judge/judge_table6.py` rates every description of every evaluation cell on a 1–10 scale against the five human
captions of the image, with the prompt of `judge/score_coco_desc.py`. It reads the texts that
`analysis/tables/tab5_dump_texts.py` writes (the exact strings the metrics were computed on) and a captions file
built from the COCO annotations (`{image file name: [five captions]}`). The judge model and endpoint are two
constants at the top of the file (an OpenAI-compatible chat endpoint, temperature 0). **The API key is read from the
environment variable `JUDGE_API_KEY` and is never written anywhere.** `judge/pairwise_coco_desc.py` is the
order-swapped pairwise protocol: each pair is judged twice with A/B swapped and counts only when both verdicts agree.

---

## 5. Experiments

Each experiment is a set of runs (its command lines are in `experiments/`) and an aggregator that reads the run
directories by exact path and prints or writes the result:

| Experiment | Produce the runs with | Aggregate / render with |
|---|---|---|
| Repeat Curse across models, backends and lengths; response quality and length | `experiments/jobs/ch6_resume.txt` | `analysis/tables/tab5_build.py` (mmada env) → `judge/judge_table6.py` → `analysis/tables/make_table5.py`; paired tests `analysis/tables/stats_pairs.py` |
| Phrase-level repetition (seq-rep-n, distinct-n) | same runs | `analysis/tables/make_table_phrase_main.py` |
| Nine general benchmarks | `benchmarks/*.sh`, `experiments/jobs/sfbench.txt` | lmms-eval result files; `judge/score_math_from_samples.py`, `judge/score_llavabench.py` |
| Component grid (every subset of CTAR / DAR / CTEV) | `experiments/jobs/_archive/grid_final.txt` | `analysis/tables/grid_build.py` |
| Efficiency (latency, throughput, memory) | `experiments/launchers/timing_quick64.sh` (one job per idle GPU) | `summary.json` of each run (`gen_seconds_mean`, `peak_mem_gib`) |
| Design alternatives of each component (100 images) | `experiments/jobs/design100.txt`, `ctar_design.txt`, `ctev_design100.txt` | `analysis/tables/design100_build.py --json` |
| Decoding-time alternatives: no-repeat n-gram penalty, block-wise decoding | `experiments/launchers/queue_exp3.py` (`--ngram`, `--block`) | `analysis/tables/exp3_build.py`; judge `judge/judge_extra.py` |
| Attention maps of the uncached and cached model | `llada_v/train/generate_demo.py` with `VISUALIZE_ATTENTION`, `attention_recorder.py` | — |
| Findings F1 / F2 / F3 (anchoring loss, staleness at the decode moment, context entropy) | recorders `llada_v/train/{f1_attn_multisample,f1f3_multisample,f3_layer_entropy}.py` via `analysis/recording/*.sh` | per-position tables `analysis/findings/{f1,f2f3,f3}_positions100.py`; statistics `f1_stats100.py`, `f2_stats100.py`, `f3_stats100.py`, `joint_stats100.py`; figures `figures/plot_f1_panel_a.py` + `plot_f1_panels_separate.py` + `compose_f1_figure100.py`, `plot_f2_combined100.py`, `plot_f3_combined100.py` |
| Case study (uncached / cached / cached + CoTA++ responses side by side) | the evaluation runs | `figures/paper/make_case_fig_2col.py` (responses exported with `analysis/tables/find_case_b.py`) |
| Attention at the decode moment, with and without CTAR | `--decode_rows` runs, `analysis/recording/probe_rows*.sh`, `mech_case.sh` | `analysis/findings/attn_decode_rows100.py`; `figures/plot_attn_pairs.py --variant v2 --rows 2 --no-cbar-title --side` |
| Staleness with and without DAR; context entropy with and without CTEV | `--case_trace 1` runs (`analysis/recording/ctev_iso_trace.sh`) | `figures/plot_lens100.py --dar_only --compact`; `figures/plot_ctev_decisions.py --mean-note --compact` |
| Which tokens repeat (word cloud) | the cached evaluation runs | `analysis/tables/repeat_token_stats.py` → `figures/plot_repeat_wordcloud.py` |
| Framework figure | — | `figures/paper/make_framework_fig.py` |

Working rules the pipeline enforces: a cell is reported only when all 500 images are present; re-running a cell means
moving the old run out of the globbed directory *and* commenting out its job line; every aggregation is restricted to
`data/coco500_final.json`; backend-specific code stays behind a flag that defaults to off, and a regression run shows
the default path unchanged before a new result is trusted.

---

## 6. Where the components are implemented

| File | Contents |
|---|---|
| `llada_v/train/llava/hooks/cache_hook_LLaDA_V.py` | the dLLM-Cache hook of LLaDA-V with the attention-side additions: **CTAR** (`set_ctarx`, `_stitch_step`: re-forms the routing of the rows the context-anchor monitor selects, query from the current state against the stored keys/values, explicit softmax), **DAR** (`set_dar`, `publish_scores`: the reserved positions are forced into the refresh set), the conference CTAE gains, and the CTEV entropy cache (`ctevc_*`) |
| `llada_v/train/llava/model/language_model/modeling_llada.py` | `generate_with_embeds`: the decoding loop; **CTEV** (`_ctev_deep_entropy_bits`, modes `self` / `ctx` / `ctx_gated`), DAR wiring, the n-gram penalty baseline (`_no_repeat_ngram_bans`), `--block`, `--ctev_layers` |
| `llada_v/train/llava/hooks/sf_cotapp.py`, `slowfast_cache.py`, `slowfast_hook.py` | the SlowFast sampler ported to LLaDA-V, its evolved cache, and the components on top of it (`--mode slowfast`) |
| `mmada/mmada_cotapp.py` | the same three components for MMaDA and LaViDa, inserted into the upstream hook at load time (`dar_select`, `ctar_apply`, `ctev` score); with every component off both backends reproduce the original code token for token |
| `mmada/sf_mmada_stack.py` | SlowFast sampler + evolved cache for MMaDA, self-contained |
| `llada_v/eval/lmms-eval/lmms_eval/models/{llava_onevision_llada,lavida_cota,mmada_cota}.py` | the benchmark wrappers; same decoding as the drivers, configured by environment variables |
| `llada_v/train/llava/model/language_model/utils/attention_recorder.py` | per-step attention recording without touching the model code |
| `eval/decode_rows_recorder.py` | decode-moment attention rows, sampler-agnostic |

`analysis/probes/probe_fidelity.py` checks that CTAR's re-derived routing equals the model's own on rows both
compute; `analysis/probes/regress.sh` checks that the default path is unchanged after a code change.

---

## 7. License and citation

This repository is released under the [Apache License 2.0](LICENSE). It builds on ML-GSAI/LLaDA-V (whose files under
`llada_v/` are modified copies), dLLM-cache (Apache-2.0), MMaDA (MIT), LaViDa (Apache-2.0) and Slow-Fast-Sampling;
use of the code must also follow the licenses of those projects.

```bibtex
@inproceedings{zhao2026context,
  title     = {Context Tokens are Anchors: Understanding the Repeat Curse in dMLLMs from an Information Flow Perspective},
  author    = {Zhao, Qiyan and Zhang, Xiaofeng and Chang, Shuochen and Chen, Qianyu and Yuan, Xiaosong and Chen, Xuhang and Liu, Luoqi and Zhang, Jiajun and Zhang, Xu-Yao and Wang, Da-Han},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
