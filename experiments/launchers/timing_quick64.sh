#!/bin/bash
# Quick exclusive timing at L=64 (2026-10-01, user): vanilla, dLLM-Cache, +DAR, +CTAR, +CTEV, CoTA++ (+ CoTA++ with the
# entropy cache, for the text), 20 images, one configuration at a time on one idle A100 (other processes counted).
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval; cd $E
PY=$Z/miniconda3/envs/llada-v/bin/python
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
G=${1:-7}; export CUDA_VISIBLE_DEVICES=$G
OUT=$E/results/lladav/timing_2026-10-01/64; mkdir -p $OUT logs
LOG=$E/logs/timing_quick64.log
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G)
CT="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1"
CV="--ctev_mode ctx --ctev_lambda 0.25"
run() { tag=$1; shift
  o1=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
  $PY -u scripts/run_repeat_eval.py --images $E/data/coco500_final.json --mode $MODE --length 64 --limit 20 --out $OUT/$tag "$@" > $E/logs/timing_64_$tag.log 2>&1
  rc=$?; o2=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
  s=$($PY -c "import json;s=json.load(open('$OUT/$tag/summary.json'));print(s['gen_seconds_mean'],s['peak_mem_gib'],s.get('ctev_cache'))" 2>/dev/null)
  echo "[$(date '+%m-%d %H:%M:%S')] done $tag rc=$rc others_before=$o1 others_after=$o2 gen_s/img peak_gib ctev: $s" >> $LOG
}
MODE=baseline   run vanilla
MODE=dllm_cache run cache
MODE=dllm_cache run dar      --dar_r 4 --dar_mode legacy
MODE=dllm_cache run ctar     $CT
MODE=dllm_cache run ctev     $CV
MODE=dllm_cache run cotapp   --dar_r 4 --dar_mode legacy $CT $CV
MODE=dllm_cache run cotapp_c --dar_r 4 --dar_mode legacy $CT $CV --ctev_cache 1
echo "QUICK64-COMPLETE $(date)" >> $LOG
