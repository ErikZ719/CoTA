# Drop-in files for ML-GSAI/LLaDA-V

Target: [ML-GSAI/LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) at the commit in `UPSTREAM_COMMIT` (`f8b02ce`,
2026-03-23). `apply.sh <clone>` copies everything below into the clone, keeping a `.upstream` copy of each file it
overwrites. `pyproject.toml.diff` shows the only dependency change (torch 2.6.0 / torchvision 0.21.0,
transformers pinned to 4.39.3).

| Path | Status | Contents |
|---|---|---|
| `train/llava/hooks/cache_hook_LLaDA_V.py` | modified (+679 lines) | dLLM-Cache hook with CTAR (`set_ctarx`, `_stitch_step`, `publish_mask`), DAR (`set_dar`, `publish_scores`, `publish_entropy`, `reset_dar`), the conference CTAE gains (`set_ctae`), the anchoring probe used by the F1 analysis (`set_anchor`, `get_anchor_w5`), the CTEV entropy cache (`ctevc_*`) |
| `train/llava/model/language_model/modeling_llada.py` | modified (+306 lines) | `generate_with_embeds`: CTEV (`self` / `ctx` / `ctx_gated`), DAR wiring, semi-AR blocks (`block_length`), the no-repeat n-gram baseline, the CTEV layer window, decode-row and trace recording |
| `train/llava/hooks/sf_cotapp.py` | new | SlowFast sampling on LLaDA-V with the components (`--mode slowfast`); generated at import from the cache hook plus the evolved-cache delta |
| `train/llava/hooks/slowfast_cache.py` | new | SlowFast-compatible dLLM-Cache stack (`SFFeatureCache`, tolerates truncated forwards) |
| `train/llava/hooks/slowfast_hook.py` | new | the Slow-Fast-Sampling sampler ported to LLaDA-V's multimodal prefix |
| `train/llava/model/language_model/utils/attention_recorder.py` | new | per-step attention recording around `generate_with_embeds` (attention maps, the F1 recordings) |
| `train/generate_demo.py` | modified | single-image demo with dLLM-Cache, the components, and optional per-layer attention rendering |
| `train/f1_attn_multisample.py`, `train/f1f3_multisample.py`, `train/f3_layer_entropy.py` | new | recorders of the diagnosis: effective attention (F1), per-layer entropy + staleness (F2/F3), logit-lens entropy at every step (F3); configured by environment variables documented in each file |
| `eval/lmms-eval/lmms_eval/models/llava_onevision_llada.py` | modified | the LLaDA-V lmms-eval model with the backend and components taken from `COTA_*` environment variables |
| `eval/lmms-eval/lmms_eval/models/lavida_cota.py`, `mmada_cota.py`, `__init__.py` | new / modified | LaViDa and MMaDA lmms-eval models with the same decoding as the Repeat-Curse drivers; registered in `__init__.py` |
| `eval/lmms-eval/lmms_eval/tasks/*/*_local.yaml`, `tasks/{mathverse,mathvista,mmstar}/utils.py` | new / modified | the nine benchmark tasks read from local parquet files (`benchmarks/make_local_tasks.py` generates the yaml); the utils accept raw image bytes |

The hook keeps the upstream behaviour when every component is off: `--mode dllm_cache` without flags is dLLM-Cache
as released, which `analysis/probes/regress.sh` verifies.
