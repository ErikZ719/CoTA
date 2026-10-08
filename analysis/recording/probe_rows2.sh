#!/bin/bash
# second SlowFast pair for Fig. 12 (2026-09-30): image 462371 (run of 68 under SlowFast at L=512, full-length clean response with CoTA++)
set -u
E=/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval; cd $E
export HF_HOME=/data/zhaoqiyan/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=/data/zhaoqiyan/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${1:-3}
PY=/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python
IMG=$E/data/probe_462371.json
CPP_SF="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --dar_r 4 --dar_mode legacy --sf_repguard 0.25"
O=$E/results/lladav/rows_probe
until grep -q "PROBE-ROWS-DONE" $E/logs/rows_lv128_cpp.log 2>/dev/null; do sleep 20; done
$PY -u scripts/run_repeat_eval.py --images $IMG --mode slowfast --length 512 --decode_rows $O/sf512b_cache --out $O/sf512b_cache > logs/rows_sf512b_cache.log 2>&1
$PY -u scripts/run_repeat_eval.py --images $IMG --mode slowfast --length 512 $CPP_SF --decode_rows $O/sf512b_cpp --out $O/sf512b_cpp > logs/rows_sf512b_cpp.log 2>&1
echo PROBE-ROWS2-DONE >> logs/rows_sf512b_cpp.log
