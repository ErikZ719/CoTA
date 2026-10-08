#!/bin/bash
# Exclusive wall-clock for the CTAR band candidates: one job at a time on one GPU, nothing
# else of ours running. L=512, E_s=7, alpha=.25, 12 images. Only these timings may be cited.
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
source $Z/miniconda3/etc/profile.d/conda.sh; conda activate llada-v
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
cd $Z/autodl-tmp/LLaDA-V/train
G=${COTA_GPUS%%,*}
CT="--ctae_mode reroute_q"
run() { tag=$1; shift
  CUDA_VISIBLE_DEVICES=$G python -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json \
    --out $E/results/lladav/lat2/$tag --length 512 --offset 0 --limit 12 --mode dllm_cache --cache_tr 0.25 \
    --device cuda:0 "$@" > $E/logs/lat2_$tag.log 2>&1
  echo "$tag rc=$? $(grep -oE '[0-9.]+s/img' $E/logs/lat2_$tag.log | tail -1)"; }
run off
run dar            --dar_r 4
run dar_ctar_deep  --dar_r 4 $CT --stitch_lo 24 --stitch_hi 31
run dar_ctar_th1   --dar_r 4 $CT --stitch_lo 24 --stitch_hi 31 --ctar_theta 1
run dar_ctar_sh    --dar_r 4 $CT --stitch_lo 0 --stitch_hi 7
run dar_ctar_all   --dar_r 4 $CT --stitch_lo 0 --stitch_hi 31
run dar_ctar_all_th1 --dar_r 4 $CT --stitch_lo 0 --stitch_hi 31 --ctar_theta 1
echo TIMING-COMPLETE
