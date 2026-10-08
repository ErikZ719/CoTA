#!/bin/bash
# Table VII, MMaDA rows, re-run on MMaDA-8B-MixCoT at resolution 512 (2026-09-26, hgx030, 8x A100-80GB).
# MMaDA releases an eval config for MixCoT only; MMaDA-8B-Base, used until now, has no released protocol,
# which is why its scores sat far below the published ones. The Base results are kept under results/bench_mmada/,
# these go to results/bench_mmada_mixcot/, so Table V (which stays on Base) and Table VII never share a directory.
# One cell at a time, each 8-way data parallel. Resumable: a cell with a DONE file is skipped.
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
export BENCH_ROOT=bench_mmada_mixcot MMADA_CKPT=Gen-Verse/MMaDA-8B-MixCoT MMADA_RES=512
export NPROC=${NPROC:-8} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
RUN=$E/scripts/launchers/bench_eval_model_mixcot.sh          # frozen copy: never edited while jobs run
LOGD=$E/logs/mixcot; mkdir -p $LOGD
TASKS="mmstar_local chartqa_local mme_local mmbench_en_dev_local seedbench_local docvqa_val_local \
       mathvista_testmini_cot_local mathverse_testmini_vision_dominant_local \
       mathverse_testmini_vision_intensive_local mathverse_testmini_vision_only_local"
echo "=== chain_mixcot start $(date)"
for TASK in $TASKS; do
  for CFG in van cache cota cpp; do
    OUT=$E/results/$BENCH_ROOT/$CFG/$TASK
    if [ -f $OUT/DONE ]; then echo "[skip] $CFG/$TASK"; continue; fi
    echo "[$(date '+%m-%d %H:%M')] RUN $CFG/$TASK"
    bash $RUN mmada $CFG $TASK > $LOGD/${CFG}_${TASK}.log 2>&1
    rc=$?
    echo "[$(date '+%m-%d %H:%M')] $CFG/$TASK rc=$rc $([ -f $OUT/DONE ] && echo OK || echo FAILED)"
  done
done
echo "=== chain_mixcot done $(date)"
echo "cells with DONE: $(find $E/results/$BENCH_ROOT -name DONE 2>/dev/null | wc -l)/40"
