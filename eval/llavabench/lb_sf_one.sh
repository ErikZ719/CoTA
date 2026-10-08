#!/bin/bash
# LLaVA-Bench (in the wild) under SlowFast, one cell (2026-09-30): lb_sf_one.sh <lladav|lavida|mmada> <sf|sfcota|sfcpp>
# Decoding as in lb_t7.sh (Table VIII): L=512; LLaDA-V one block, LaViDa block 32, MMaDA-8B-MixCoT at resolution 512 with
# steps 512 and block 8. Backend and components as in the SlowFast rows of Table VI. GPU from CUDA_VISIBLE_DEVICES.
# Output: results/llavabench_sf/<model>/<config>/answers.jsonl, DONE only when the runner exits cleanly.
# Answers are judged afterwards with code-final/eval/score_llavabench.py (the judge key never touches this machine's disk).
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
MODEL=$1; CFG=$2
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LLADA_CONF_GEN_ONLY=1
OUT=$E/results/${LB_ROOT:-llavabench_sf}/$MODEL/$CFG; mkdir -p $OUT; rm -f $OUT/DONE
CTAR="--ctae_mode reroute_q --stitch_lo 24 --stitch_hi 31 --ctar_theta 1"
case $MODEL-$CFG in
  lladav-sf)     F="" ;;
  lladav-sfcota) F="$CTAR --ctev_lambda 0.25" ;;
  lladav-sfcpp)  F="$CTAR --dar_r 4 --dar_mode legacy --sf_repguard 0.25" ;;      # the Table VI row (lladav/sfguard)
  *-sf)          F="--mode slowfast" ;;
  *-sfcota)      F="--mode slowfast $CTAR --ctev_mode ctx --ctev_lambda 0.25" ;;
  *-sfcpp)       F="--mode slowfast $CTAR --ctev_mode ctx --ctev_lambda 0.25 --dar_r 4 --dar_mode legacy" ;;
  *) echo "unknown cell $MODEL $CFG"; exit 2 ;;
esac
echo "[lb_sf_one] model=$MODEL config=$CFG flags=$F out=$OUT"
case $MODEL in
  lladav) cd $Z/autodl-tmp/LLaDA-V/train && $Z/miniconda3/envs/llada-v/bin/python -u $E/scripts/run_llavabench_sf.py --length 512 $F --out $OUT "${@:3}" ;;
  lavida) cd $Z/autodl-tmp/LaViDa && $Z/venvs/lavida/bin/python -u $E/scripts/run_llavabench_lavida.py --length 512 --block 32 \
            --cache_pi 25 --cache_gi 7 --cache_tr 0.10 $F --out $OUT "${@:3}" ;;
  mmada)  $Z/miniconda3/envs/mmada/bin/python -u $E/scripts/run_llavabench_mmada.py --ckpt Gen-Verse/MMaDA-8B-MixCoT --resolution 512 \
            --length 512 --steps 512 --block 8 --cache_pi 20 --cache_gi 10 --cache_tr 0.10 $F --out $OUT "${@:3}" ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
rc=$?
if [ $rc -eq 0 ] && [ -s $OUT/answers.jsonl ]; then date '+%F %T' > $OUT/DONE; exit 0; fi
echo "lb_sf_one: $MODEL/$CFG failed (rc=$rc)"; exit 3
