#!/bin/bash
# 2026-09-29: does the value CTEV gives to a position without committed neighbours matter?
# L=128, 500 images (two shards), CTEV alone and the full stack, --ctev_empty mean | max.
# References with the released behaviour (zero): lladav/grid/ctev128_{0,250} and lladav/t5/cotapp128_{0,250}.
# Starts once the two earlier batches have finished; one job per GPU. A regression run (8 images, zero) comes first.
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval; PY=$Z/miniconda3/envs/llada-v/bin/python
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
until [ -f $E/results/lladav/darmask/ALL_DONE ] && [ -f $E/results/lladav/ctev_iso/TRACES_DONE ]; do sleep 60; done
mkdir -p $E/logs/ctev_empty $E/results/lladav/ctev_empty; cd $Z/autodl-tmp/LLaDA-V/train
CTAR="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1"; DAR="--dar_r 4 --dar_mode legacy"; CTEV="--ctev_mode ctx --ctev_lambda 0.25"
g=0
for cfg in ctev full; do for em in mean max; do for off in 0 250; do
  t=${cfg}_${em}_128_${off}; out=$E/results/lladav/ctev_empty/$t
  if [ "$cfg" = full ]; then A="$CTAR $DAR $CTEV"; else A="$CTEV"; fi
  if [ ! -f $out/summary.json ]; then
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json --mode dllm_cache --length 128 --limit 250 --offset $off --out $out $A --ctev_empty $em > $E/logs/ctev_empty/$t.log 2>&1 &
    echo "$(date +%H:%M) started $t on gpu $g"
  fi
  g=$((g+1))
done; done; done
wait
# regression: the default must reproduce the released outputs token for token
out=$E/results/lladav/ctev_empty/regress_zero_8
CUDA_VISIBLE_DEVICES=0 $PY -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json --mode dllm_cache --length 128 --limit 8 --offset 0 --out $out $CTEV --ctev_empty zero > $E/logs/ctev_empty/regress_zero_8.log 2>&1
date > $E/results/lladav/ctev_empty/ALL_DONE
