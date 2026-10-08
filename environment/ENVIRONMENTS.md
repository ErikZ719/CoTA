# Environments

Three Python 3.10 environments on the experiment machine (8× A100-SXM4-80GB, driver 535.129.03, CUDA 12.4).
The `pip_freeze_*.txt` files are their exact package lists on 2026-10-08.

| Environment | How it was built | Key versions | Used by |
|---|---|---|---|
| `llada-v` (conda) | LLaDA-V's `train/init_env.sh` with the pins of `llada_v/pyproject.toml.diff`, then `pip install -e eval/lmms-eval` | torch 2.6.0+cu124, transformers 4.39.3 | `eval/run_repeat_eval.py`, `eval/run_llavabench*.py`, lmms-eval, the recorders, `sched_cmd.py` |
| `mmada` (conda) | MMaDA's requirements plus dLLM-cache | torch 2.6.0+cu124, transformers 4.46.3 | `eval/run_repeat_eval_mmada.py`, `analysis/tables/tab5_build.py` (needs the MMaDA tokenizer for the end-of-text ids) |
| `lavida` (venv over `mmada`) | LaViDa's requirements on top of the mmada environment | torch 2.6.0+cu124, transformers 4.50.3 | `eval/run_repeat_eval_lavida.py`, run from the LaViDa checkout |

Environment variables the scripts expect: `HF_HOME` / `HUGGINGFACE_HUB_CACHE` (model cache), `HF_HUB_OFFLINE=1`
once the checkpoints are cached, `CUDA_VISIBLE_DEVICES` (one GPU per driver process). The benchmark launchers also
set `LLADA_CONF_GEN_ONLY=1` (suffix-only confidence softmax, bit-identical outputs, lower memory) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Memory: an LLaDA-V job holds 29–43 GB at any length (the float64 softmax of the attention in `modeling_llada.py`
peaks at 3.7 GB), so one job per 80 GB card; attention recording is CPU-bound but still holds 35–42 GB.

The figure scripts and the table renderer ran on macOS with Python 3.12: `numpy`, `scipy`, `matplotlib`, `Pillow`,
`python-pptx` (framework figure), `wordcloud` (word cloud).
