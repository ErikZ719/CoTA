#!/bin/bash
# Attention recording for the mechanism analysis on the 100-image seed-0 sample (2026-09-28).
# usage: mech100.sh <tag> <baseline|dllm_cache> [E_s] [dar_r] [dar_w] [ctae] [band_lo] [band_hi] [theta]
# Images already recorded under attn_multi100/<tag>/ are skipped by the collector. Writes <tag>.DONE when all 100 exist.
set -u
Z=/data/zhaoqiyan/autodl-tmp; IF=$Z/information_flow
TAG=$1; MODE=$2
export F1A_TAG=$TAG F1A_MODE=$MODE F1A_OUT=$IF/attn_multi100
export F1A_GI=${3:-7} F1A_DAR_R=${4:-0} F1A_DAR_W=${5:-0} F1A_CTAE=${6:-off} F1A_BAND_LO=${7:-24} F1A_BAND_HI=${8:-31} F1A_CTAR_THETA=${9:-0}
export F1A_IMAGES=$(/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python -c "import json;print(','.join(json.load(open('$IF/coco100_seed0.json'))['files']))")
cd $Z/LLaDA-V/train
/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python -u f1_attn_multisample.py
n=$(ls -d $IF/attn_multi100/$TAG/*/ 2>/dev/null | while read d; do [ -f "$d/step_127.npz" ] && echo x; done | wc -l)
echo "[mech100] $TAG complete images: $n/100"
[ "$n" -eq 100 ] && date > $IF/attn_multi100/$TAG.DONE
