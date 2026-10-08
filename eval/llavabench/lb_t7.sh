#!/bin/bash
# LLaVA-Bench (in the wild) for the MMaDA and LaViDa rows of Table VII (2026-09-26, hgx030).
# LLaDA-V already has this column (results/llavabench, run_llavabench.py); these two runners are derived from the
# validated Table V runners by make_lb_runners.py, so all three models share one decoding path per model.
# Decoding: L=512, one token per step. MMaDA's released MMVet setting (512/256/128, two tokens per step) collapses
# the answers to 3-15 tokens, so MMaDA keeps its Table V granularity (steps 512, block 8); LaViDa keeps block 32.
# Cache: the Table VII settings, LaViDa 25/7/0.10 and MMaDA 20/10/0.10.
# One cell per GPU, all eight at once. Answers are judged afterwards with code-final/eval/score_llavabench.py.
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
OUTROOT=$E/results/llavabench_t7; LOGD=$E/logs/llavabench_t7
mkdir -p $OUTROOT $LOGD
export HF_HOME=$Z/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LLADA_CONF_GEN_ONLY=1

comp_flags() {                       # $1 = van|cache|cota|cpp
  case $1 in
    van)   echo "--mode baseline" ;;
    cache) echo "--mode dllm_cache" ;;
    cota)  echo "--mode dllm_cache --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --ctev_mode ctx --ctev_lambda 0.25" ;;
    cpp)   echo "--mode dllm_cache --ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1 --ctev_mode ctx --ctev_lambda 0.25 --dar_r 4 --dar_mode legacy" ;;
  esac
}

g=0
for MODEL in mmada lavida; do
  for CFG in van cache cota cpp; do
    OUT=$OUTROOT/$MODEL/$CFG
    if [ -f $OUT/summary.json ]; then echo "[skip] $MODEL/$CFG"; g=$((g+1)); continue; fi
    rm -rf $OUT
    F=$(comp_flags $CFG)
    if [ $MODEL = mmada ]; then
      CUDA_VISIBLE_DEVICES=$g nohup $Z/miniconda3/envs/mmada/bin/python -u $E/scripts/run_llavabench_mmada.py \
        --ckpt Gen-Verse/MMaDA-8B-MixCoT --resolution 512 --length 512 --steps 512 --block 8 \
        --cache_pi 20 --cache_gi 10 --cache_tr 0.10 $F --out $OUT \
        > $LOGD/${MODEL}_${CFG}.log 2>&1 &
    else
      ( cd $Z/autodl-tmp/LaViDa && CUDA_VISIBLE_DEVICES=$g nohup $Z/venvs/lavida/bin/python -u $E/scripts/run_llavabench_lavida.py \
        --length 512 --block 32 --cache_pi 25 --cache_gi 7 --cache_tr 0.10 $F --out $OUT \
        > $LOGD/${MODEL}_${CFG}.log 2>&1 & )
    fi
    echo "[launch] $MODEL/$CFG on GPU $g"
    g=$((g+1))
  done
done
echo "launched; watch $LOGD"
