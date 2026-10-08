#!/bin/bash
# Attention recording for the case-study images (2026-09-30), for the "attention before and after CTAR" figure.
# usage: mech_case.sh <image stem> [<image stem> ...]      e.g. mech_case.sh COCO_val2014_000000118929
# Four arms per image, with the recorder and the settings of mech100.sh: baseline, dllm_cache, ctar_th (CTAR, theta=1,
# layers 25-32), dar (r=4). Output: information_flow/attn_case/<arm>/<stem>/step_*.npz, then attn_case/DONE.
set -u
Z=/data/zhaoqiyan/autodl-tmp; IF=$Z/information_flow; PY=/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python
export HF_HOME=$Z/hf_cache HUGGINGFACE_HUB_CACHE=$Z/hf_cache/hub HF_HUB_OFFLINE=1
export F1A_OUT=$IF/attn_case; mkdir -p $F1A_OUT
IMGS=""; for s in "$@"; do IMGS="$IMGS,$Z/coco2014/val2014/$s.jpg"; done; export F1A_IMAGES=${IMGS#,}
cd $Z/LLaDA-V/train
rec() { # tag mode dar_r ctae theta
  F1A_TAG=$1 F1A_MODE=$2 F1A_GI=7 F1A_DAR_R=$3 F1A_DAR_W=0 F1A_CTAE=$4 F1A_BAND_LO=24 F1A_BAND_HI=31 F1A_CTAR_THETA=$5 $PY -u f1_attn_multisample.py || exit 3
}
rec baseline   baseline   0 off       0
rec dllm_cache dllm_cache 0 off       0
rec ctar_th    dllm_cache 0 reroute_q 1
rec dar        dllm_cache 4 off       0
n=0; for a in baseline dllm_cache ctar_th dar; do for s in "$@"; do [ -f $F1A_OUT/$a/$s/step_127.npz ] && n=$((n+1)); done; done
echo "[mech_case] complete recordings: $n / $(( 4 * $# ))"
[ "$n" -eq $(( 4 * $# )) ] && date > $F1A_OUT/DONE
