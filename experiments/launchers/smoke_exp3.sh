#!/bin/bash
# Smoke tests of the EXP3 options on one GPU (2026-09-30): regression of the plain cache run and of the CoTA++ run
# (must reproduce the ids of the existing runs), then --ngram 2, --block 16 (CoTA++), --ctev_layers 1-8 (CoTA++).
# usage: smoke_exp3.sh <gpu>      logs: logs/smoke_exp3_*.log, end marker SMOKE-DONE in logs/smoke_exp3_regcpp.log
set -u
E=/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval; cd $E
PY=/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python
export HF_HOME=/data/zhaoqiyan/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=/data/zhaoqiyan/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${1:-3}
C="--images $E/data/coco500_final.json --mode dllm_cache --length 128 --limit 3"
CPP="--dar_r 4 --dar_mode legacy --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --ctev_mode ctx --ctev_lambda 0.25"
O=results/_smoke/exp3; rm -rf $O; mkdir -p $O logs
$PY -u scripts/run_repeat_eval.py $C --out $O/reg_cache > logs/smoke_exp3_reg.log 2>&1
$PY -u scripts/run_repeat_eval.py $C --ngram 2 --out $O/ngram2 > logs/smoke_exp3_ngram2.log 2>&1
$PY -u scripts/run_repeat_eval.py $C $CPP --block 16 --out $O/cpp_b16 > logs/smoke_exp3_b16.log 2>&1
$PY -u scripts/run_repeat_eval.py $C $CPP --ctev_layers 1-8 --out $O/cpp_l1_8 > logs/smoke_exp3_l18.log 2>&1
$PY -u scripts/run_repeat_eval.py $C $CPP --out $O/reg_cpp > logs/smoke_exp3_regcpp.log 2>&1
echo SMOKE-DONE >> logs/smoke_exp3_regcpp.log
