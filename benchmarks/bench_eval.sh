#!/bin/bash
# One Table VI cell: bench_eval.sh <config> <task> [extra lmms-eval args, e.g. --limit 24]
#   config: van | cache | cota | cpp      (uncached, dLLM-Cache, +CoTA = CTAR(theta=1)+CTEV, +CoTA++ = +DAR r=4 legacy)
#   task  : lmms-eval task name; generation settings follow LLaDA-V's released eval/scripts/evaluate.sh
# GPU comes from CUDA_VISIBLE_DEVICES (set by the scheduler). Output: results/bench/<config>/<task>/, DONE on success.
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
CFG=$1; TASK=$2; shift 2
source $Z/miniconda3/etc/profile.d/conda.sh; conda activate $Z/miniconda3/envs/llada-v   # by path: a same-named env elsewhere must never win (2026-09-23, loguru failure on hgx030)
export CUDA_HOME=${CUDA_HOME:-/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts/launchers/cuda_home_stub}   # DeepSpeed (imported under accelerate) runs nvcc --version; no toolkit on dgx056
export LLADA_CONF_GEN_ONLY=1   # suffix-only confidence softmax in modeling_llada (bit-identical outputs, verified 2026-09-23; needed on 40 GB cards)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 40 GB cards (dgx056): the math tasks peaked at 37-41 GB with fragmentation on 80 GB cards
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset COTA_BACKEND COTA_CTAE COTA_CTAR_THETA COTA_DAR_R COTA_DAR_MODE COTA_CTEV_LAMBDA
case $CFG in
  van)   export COTA_BACKEND=none ;;
  cache) export COTA_BACKEND=dllm_cache ;;
  cota)  export COTA_BACKEND=dllm_cache COTA_CTAE=reroute_q COTA_CTAR_THETA=1 COTA_CTEV_LAMBDA=0.25 ;;
  cpp)   export COTA_BACKEND=dllm_cache COTA_CTAE=reroute_q COTA_CTAR_THETA=1 COTA_CTEV_LAMBDA=0.25 COTA_DAR_R=4 COTA_DAR_MODE=legacy ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
case $TASK in
  chartqa*)                  GK='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":16,"block_length":16,"gen_steps":16,"stopping_criteria":["\n"],"think_mode":"no_think"}' ;;
  docvqa_val*|infovqa_val*)  GK='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":32,"block_length":32,"gen_steps":16,"think_mode":"no_think"}' ;;
  mathvista*)                GK='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":96,"block_length":96,"gen_steps":48,"think_mode":"think"}' ;;
  mathverse*)                GK='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":64,"block_length":64,"gen_steps":32,"think_mode":"think"}' ;;
  *)                         GK='{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}' ;;
esac
# GPT-based scorers (MathVista, MathVerse, MMBench fallback) cannot reach a gateway from here: fail fast,
# keep the logged responses, and score them later on the machine that has the judge key.
export OPENAI_API_URL=http://127.0.0.1:9/v1/chat/completions OPENAI_API_KEY=none
OUT=$E/results/${BENCH_ROOT:-bench}/$CFG/$TASK; mkdir -p $OUT; rm -f $OUT/DONE; touch $OUT/.start   # BENCH_ROOT=bench_smoke for trial runs
echo "[bench_eval] config=$CFG task=$TASK gen=$GK"
cd $Z/autodl-tmp/LLaDA-V/eval/lmms-eval
# NPROC=k (k>1): data-parallel over the k GPUs in CUDA_VISIBLE_DEVICES via accelerate (rank 0 writes the results)
if [ "${NPROC:-1}" -gt 1 ]; then LAUNCH="accelerate launch --num_processes ${NPROC} --main_process_port $((20000 + RANDOM % 20000)) -m lmms_eval"; else LAUNCH="python -m lmms_eval"; fi
$LAUNCH --model llava_onevision_llada \
  --model_args pretrained=GSAI-ML/LLaDA-V,conv_template=llava_llada,model_name=llava_llada \
  --gen_kwargs "$GK" --tasks $TASK --batch_size 1 --log_samples --log_samples_suffix $TASK --output_path $OUT "$@"
rc=$?
# lmms-eval returns 0 even when the evaluation raised: trust the results file, not the exit code
if [ $rc -eq 0 ] && [ -n "$(find $OUT -name "*results*.json" -newer $OUT/.start 2>/dev/null | head -1)" ]; then
  date '+%F %T' > $OUT/DONE; exit 0
fi
echo "bench_eval: no results file for $CFG/$TASK (rc=$rc)"; exit 3
