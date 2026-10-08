#!/bin/bash
# Wall-clock attribution for Sec. VI-E, LLaDA-V + dLLM-Cache, L=512, alpha=0.25, 12 images (2026-09-26, hgx030).
# The published table must come from one card in one session, so all eight rows are re-measured here, including
# the two that were never run (+CTEV alone and the full CoTA++ stack). Timings are only valid with nothing else
# of ours on the machine, so the script waits for the MixCoT chain and the LLaVA-Bench jobs to finish first.
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
OUT=$E/results/lladav/lat_vie; LOG=$E/logs/timing_vie.log
mkdir -p $OUT $(dirname $LOG)
exec > >(tee -a $LOG) 2>&1
echo "=== timing_vie queued $(date); waiting for an idle machine"
while pgrep -f "chain_mixcot|run_llavabench_|bench_eval_model" > /dev/null; do sleep 120; done
echo "=== machine idle, starting $(date) on $(hostname)"
source $Z/miniconda3/etc/profile.d/conda.sh; conda activate llada-v
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $Z/autodl-tmp/LLaDA-V/train
G=${COTA_GPUS:-0}
CT="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31"
EV="--ctev_mode ctx --ctev_lambda 0.25"
run() { tag=$1; shift
  rm -rf $OUT/$tag
  CUDA_VISIBLE_DEVICES=$G python -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json \
    --out $OUT/$tag --length 512 --offset 0 --limit 12 --mode dllm_cache --cache_tr 0.25 \
    --device cuda:0 "$@" > $E/logs/lat_vie_$tag.log 2>&1
  rc=$?
  s=$(python3 -c "
import json;d=json.load(open('$OUT/$tag/summary.json'));print('%.1f'%(d['wall_seconds']/12))" 2>/dev/null || echo NA)
  echo "[$(date '+%H:%M')] $tag rc=$rc  ${s}s/image"; }
run cache                                                     # dLLM-Cache
run dar                  --dar_r 4                            # +DAR
run dar_ctar_nomon       --dar_r 4 $CT                        # +DAR+CTAR, no monitor
run dar_ctar             --dar_r 4 $CT --ctar_theta 1         # +DAR+CTAR
run dar_ctar_all_nomon   --dar_r 4 --ctae_mode reroute_q --stitch_lo 0 --stitch_hi 31
run dar_ctar_all         --dar_r 4 --ctae_mode reroute_q --stitch_lo 0 --stitch_hi 31 --ctar_theta 1
run ctev                 $EV                                  # +CTEV alone
run cotapp               --dar_r 4 $CT --ctar_theta 1 $EV     # full CoTA++
echo "=== TIMING-VIE-COMPLETE $(date)"
