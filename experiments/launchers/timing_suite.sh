#!/bin/bash
# Exclusive wall-clock / throughput / memory suite for the efficiency table (2026-10-01).
# One configuration at a time on one otherwise idle A100; before and after each run the card's other compute
# processes are counted into the log so that non-exclusive runs can be identified and discarded.
# LLaDA-V, dLLM-Cache 25/7/0.25. L=128: 20 images; L=512: 12 images (the protocol of the earlier lat2 timings).
# usage: timing_suite.sh <gpu>      results: results/lladav/timing_2026-10-01/<L>/<tag>/summary.json
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval; cd $E
PY=$Z/miniconda3/envs/llada-v/bin/python
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
G=${1:-7}; export CUDA_VISIBLE_DEVICES=$G
OUT=$E/results/lladav/timing_2026-10-01; mkdir -p $OUT logs
LOG=$E/logs/timing_suite.log
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G)
CT="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1"
CV="--ctev_mode ctx --ctev_lambda 0.25"
run() { L=$1; N=$2; tag=$3; shift 3
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UUID")
  echo "[$(date '+%m-%d %H:%M:%S')] start L=$L $tag (other procs on gpu $G before start: $others)" >> $LOG
  $PY -u scripts/run_repeat_eval.py --images $E/data/coco500_final.json --mode $MODE --length $L --limit $N \
     --out $OUT/$L/$tag "$@" > $E/logs/timing_${L}_$tag.log 2>&1
  rc=$?
  others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UUID")
  s=$($PY -c "import json;s=json.load(open('$OUT/$L/$tag/summary.json'));print(s['gen_seconds_mean'],s['peak_mem_gib'],s.get('ctev_cache'))" 2>/dev/null)
  echo "[$(date '+%m-%d %H:%M:%S')] done  L=$L $tag rc=$rc (other procs after: $others) gen_s/img peak_gib ctev: $s" >> $LOG
}
suite() { L=$1; N=$2
  MODE=baseline   run $L $N vanilla
  MODE=dllm_cache run $L $N cache
  MODE=dllm_cache run $L $N dar        --dar_r 4 --dar_mode legacy
  MODE=dllm_cache run $L $N ctar       $CT
  MODE=dllm_cache run $L $N ctev       $CV
  MODE=dllm_cache run $L $N ctev_c     $CV --ctev_cache 1
  MODE=dllm_cache run $L $N dar_ctar   --dar_r 4 --dar_mode legacy $CT
  MODE=dllm_cache run $L $N cotapp     --dar_r 4 --dar_mode legacy $CT $CV
  MODE=dllm_cache run $L $N cotapp_c   --dar_r 4 --dar_mode legacy $CT $CV --ctev_cache 1
  MODE=slowfast   run $L $N slowfast
}
suite 128 20
suite 512 12
echo "TIMING-SUITE-COMPLETE $(date)" >> $LOG
