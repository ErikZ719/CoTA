#!/bin/bash
# CTAR-only arms for the Fig. 11 pairs (2026-10-01, user: rows (c)-(f) must show CTAR, not CoTA++), plus a second LaViDa
# candidate image. LaViDa + dLLM-Cache (L=128, cache_tr 0.10): 479129 ctar; 497330 cache + ctar.
# LLaDA-V + SlowFast (L=512): 462371 ctar; 479129 ctar.   usage: probe_rows3.sh <gpu>
set -u
E=/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval; cd $E
export HF_HOME=/data/zhaoqiyan/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=/data/zhaoqiyan/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${1:-7}
PY=/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python
PYLV=/data/zhaoqiyan/venvs/lavida/bin/python
CT="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1"
python3 - <<"PYEOF"
import json
E="/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
s=json.load(open(E+"/data/coco500_final.json"))
for stem in ["COCO_val2014_000000497330"]:
    d=dict(s); d["files"]=[stem+".jpg"]; json.dump(d, open(E+"/data/probe_%s.json"%stem[-6:],"w"))
PYEOF
O=$E/results/lladav/rows_probe; mkdir -p $O logs
$PY -u scripts/run_repeat_eval.py --images $E/data/probe_462371.json --mode slowfast --length 512 $CT --decode_rows $O/sf512b_ctar --out $O/sf512b_ctar > logs/rows_sf512b_ctar.log 2>&1
$PY -u scripts/run_repeat_eval.py --images $E/data/probe_479129.json --mode slowfast --length 512 $CT --decode_rows $O/sf512_ctar --out $O/sf512_ctar > logs/rows_sf512_ctar.log 2>&1
O=$E/results/lavida/rows_probe; mkdir -p $O
cd /data/zhaoqiyan/autodl-tmp/LaViDa
$PYLV -u $E/scripts/run_repeat_eval_lavida.py --images $E/data/probe_479129.json --mode dllm_cache --cache_tr 0.10 --length 128 $CT --decode_rows $O/dc128_ctar --out $O/dc128_ctar > $E/logs/rows_lv128_ctar.log 2>&1
$PYLV -u $E/scripts/run_repeat_eval_lavida.py --images $E/data/probe_497330.json --mode dllm_cache --cache_tr 0.10 --length 128 --decode_rows $O/dc128b_cache --out $O/dc128b_cache > $E/logs/rows_lv128b_cache.log 2>&1
$PYLV -u $E/scripts/run_repeat_eval_lavida.py --images $E/data/probe_497330.json --mode dllm_cache --cache_tr 0.10 --length 128 $CT --decode_rows $O/dc128b_ctar --out $O/dc128b_ctar > $E/logs/rows_lv128b_ctar.log 2>&1
echo PROBE-ROWS3-DONE >> $E/logs/rows_lv128b_ctar.log
