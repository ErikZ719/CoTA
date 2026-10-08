#!/bin/bash
# Decode-moment attention rows for the two new Fig. 12 pairs (2026-09-30), image COCO_val2014_000000479129:
#   LLaDA-V + SlowFast at L=512 (run of 232 tokens under SlowFast; 58-token clean response with CoTA++), arms
#   slowfast and slowfast + CoTA++ in the Table VII guard configuration;
#   LaViDa + dLLM-Cache at L=128 (run of 89 tokens; clean with CoTA++), arms cache and cache + CoTA++ (Table VII config).
# Output: results/lladav/rows_probe/<arm>/rows_<stem>.npz and results/lavida/rows_probe/<arm>/rows_<stem>.npz
# usage: probe_rows.sh <gpu>
set -u
E=/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval; cd $E
export HF_HOME=/data/zhaoqiyan/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=/data/zhaoqiyan/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${1:-3}
PY=/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python
PYLV=/data/zhaoqiyan/venvs/lavida/bin/python
IMG=$E/data/probe_479129.json
CPP_SF="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --dar_r 4 --dar_mode legacy --sf_repguard 0.25"
CPP_LV="--dar_r 4 --dar_mode legacy --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --ctev_mode ctx --ctev_lambda 0.25"
O=$E/results/lladav/rows_probe; mkdir -p $O logs
$PY -u scripts/run_repeat_eval.py --images $IMG --mode slowfast --length 512 --decode_rows $O/sf512_cache --out $O/sf512_cache > logs/rows_sf512_cache.log 2>&1
$PY -u scripts/run_repeat_eval.py --images $IMG --mode slowfast --length 512 $CPP_SF --decode_rows $O/sf512_cpp --out $O/sf512_cpp > logs/rows_sf512_cpp.log 2>&1
O=$E/results/lavida/rows_probe; mkdir -p $O
cd /data/zhaoqiyan/autodl-tmp/LaViDa
$PYLV -u $E/scripts/run_repeat_eval_lavida.py --images $IMG --mode dllm_cache --cache_tr 0.10 --length 128 --decode_rows $O/dc128_cache --out $O/dc128_cache > $E/logs/rows_lv128_cache.log 2>&1
$PYLV -u $E/scripts/run_repeat_eval_lavida.py --images $IMG --mode dllm_cache --cache_tr 0.10 --length 128 $CPP_LV --decode_rows $O/dc128_cpp --out $O/dc128_cpp > $E/logs/rows_lv128_cpp.log 2>&1
echo PROBE-ROWS-DONE >> $E/logs/rows_lv128_cpp.log
