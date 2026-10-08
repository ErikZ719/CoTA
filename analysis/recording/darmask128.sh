#!/bin/bash
# 2026-09-29: DAR alone, L=128, 500 images. Reserved set drawn from the positions still masked after the commit
# (--dar_mode masked; with CTEV off the published score is the raw confidence), r=4 and r=3.
# Reference: lladav/grid/dar128_{0,250} (--dar_mode legacy --dar_r 4, reserved set drawn from the positions masked at t-1).
# One job per GPU. usage: darmask128.sh
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval; PY=$Z/miniconda3/envs/llada-v/bin/python
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
mkdir -p $E/logs/darmask $E/results/lladav/darmask; cd $Z/autodl-tmp/LLaDA-V/train
g=0
for r in 4 3; do for off in 0 250; do
  out=$E/results/lladav/darmask/darm${r}_128_${off}
  if [ -f $out/summary.json ]; then echo "skip $out"; else
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json --mode dllm_cache --length 128 --limit 250 --offset $off --out $out --dar_r $r --dar_mode masked > $E/logs/darmask/darm${r}_128_${off}.log 2>&1 &
    echo "started darm${r}_128_${off} on gpu $g"
  fi
  g=$((g+1))
done; done
wait
date > $E/results/lladav/darmask/ALL_DONE
