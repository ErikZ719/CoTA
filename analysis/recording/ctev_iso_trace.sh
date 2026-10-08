#!/bin/bash
# 2026-09-29: decoding traces for the question "does CTEV favour positions without any committed neighbour?"
# L=128, first 100 images of coco500_final, --case_trace 1 (analysis only: logs, per commit, position, confidence, score and
# context entropy; with CTEV off the entropy is logged with a zero penalty and the decoding is untouched).
# One job per GPU, GPUs 4-7.
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval; PY=$Z/miniconda3/envs/llada-v/bin/python
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
mkdir -p $E/logs/ctev_iso $E/results/lladav/ctev_iso; cd $Z/autodl-tmp/LLaDA-V/train
COMMON="--images $E/data/coco500_final.json --length 128 --limit 100 --offset 0 --case_trace 1"
CTAR="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1"; DAR="--dar_r 4 --dar_mode legacy"; CTEV="--ctev_mode ctx --ctev_lambda 0.25"
run() { # gpu tag args...
  g=$1; t=$2; shift 2; out=$E/results/lladav/ctev_iso/$t
  [ -f $out/summary.json ] && { echo "skip $t"; return; }
  CUDA_VISIBLE_DEVICES=$g nohup $PY -u $E/scripts/run_repeat_eval.py $COMMON --out $out "$@" > $E/logs/ctev_iso/$t.log 2>&1 &
  echo "started $t on gpu $g"
}
run 4 tr_van   --mode baseline
run 5 tr_cache --mode dllm_cache
run 6 tr_ctev  --mode dllm_cache $CTEV
run 7 tr_full  --mode dllm_cache $CTAR $DAR $CTEV
wait
date > $E/results/lladav/ctev_iso/TRACES_DONE
