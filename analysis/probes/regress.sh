#!/bin/bash
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
source $Z/miniconda3/etc/profile.d/conda.sh; conda activate llada-v
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
cd $Z/autodl-tmp/LLaDA-V/train
CUDA_VISIBLE_DEVICES=${2:-0} python -u $E/scripts/run_repeat_eval.py \
  --images $E/data/coco500_final.json --out $1 --length 128 --limit 6 \
  --mode dllm_cache --cache_tr 0.25 --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 \
  --device cuda:0 "${@:3}"
