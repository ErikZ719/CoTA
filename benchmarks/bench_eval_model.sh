#!/bin/bash
# One Table VII cell for MMaDA or LaViDa: bench_eval_model.sh <lavida|mmada> <van|cache|cota|cpp> <task> [extra lmms-eval args]
# Same harness, tasks, prompts and scorers as the LLaDA-V rows (LLaDA-V/eval/lmms-eval with the *_local tasks), through
# the wrappers lmms_eval/models/{lavida_cota,mmada_cota}.py. GPU(s) from CUDA_VISIBLE_DEVICES; NPROC=k for accelerate.
# Output: results/bench_<model>/<config>/<task>/, DONE only when a results json exists.
#
# Generation settings (2026-09-24):
#   LaViDa : the per-task max_new_tokens of LaViDa's own lmms-eval fork (MME 16, ChartQA 16, DocVQA 32, MMBench 100,
#            MathVista 100, MathVerse 100; MMStar/SEED 16 as MME instead of its 256-token wrapper default), block = min(128, L), one token per step,
#            no prefix cache (as in Table V). Cache 25/7/0.10 (Table V).
#   MMaDA  : the released VLMEvalKit configs of MMaDA (default 2/2/2; MathVista 96/96/48; MathVerse 256/128/32);
#            DocVQA 32/32/16 and ChartQA 16/16/16 have no released setting and take LLaDA-V's. Cache 20/10/0.10 (Table V).
#            2026-09-26: the checkpoint is MMaDA-8B-MixCoT at resolution 512 (MMADA_CKPT / MMADA_RES), the only
#            configuration MMaDA releases an eval config for; set BENCH_ROOT to keep runs of different checkpoints apart.
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
MODEL=$1; CFG=$2; TASK=$3; shift 3
case $MODEL in
  lavida) PY=$Z/venvs/lavida/bin/python; WRAPPER=lavida_cota; PI=25; GI=7; TR=0.10; MARGS="" ;;
  mmada)  PY=$Z/miniconda3/envs/mmada/bin/python; WRAPPER=mmada_cota; PI=20; GI=10; TR=0.10
          # MMaDA releases a VLMEvalKit config for MixCoT only, at resolution 512 (evaluation/VLMEvalKit/vlmeval/config.py);
          # MMaDA-8B-Base has no released eval protocol, which is why its scores were far below the paper's.
          MMADA_CKPT=${MMADA_CKPT:-Gen-Verse/MMaDA-8B-MixCoT}; MMADA_RES=${MMADA_RES:-512}
          MARGS="pretrained=$MMADA_CKPT,resolution=$MMADA_RES" ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
export PYTHONPATH=$Z/autodl-tmp/LLaDA-V/eval/lmms-eval${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_HOME=${CUDA_HOME:-$E/scripts/launchers/cuda_home_stub}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COTA_CACHE_PI=$PI COTA_CACHE_GI=$GI COTA_CACHE_TR=$TR
unset COTA_BACKEND COTA_CTAE COTA_CTAR_THETA COTA_DAR_R COTA_DAR_MODE COTA_CTEV_MODE COTA_CTEV_LAMBDA GEN_L GEN_STEPS GEN_BLOCK
case $CFG in
  van)   export COTA_BACKEND=none ;;
  cache) export COTA_BACKEND=dllm_cache ;;
  cota)  export COTA_BACKEND=dllm_cache COTA_CTAE=reroute_q COTA_CTAR_THETA=1 COTA_CTEV_MODE=ctx COTA_CTEV_LAMBDA=0.25 ;;
  cpp)   export COTA_BACKEND=dllm_cache COTA_CTAE=reroute_q COTA_CTAR_THETA=1 COTA_CTEV_MODE=ctx COTA_CTEV_LAMBDA=0.25 COTA_DAR_R=4 COTA_DAR_MODE=legacy ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
# per-task generation settings
if [ $MODEL = lavida ]; then
  case $TASK in
    mme*)                     export GEN_L=16 ;;
    chartqa*)                 export GEN_L=16 ;;
    docvqa_val*)              export GEN_L=32 ;;
    mmbench*)                 export GEN_L=16 ;;         # MCQ: one option letter. LaViDa's fork says 100, which costs
                                                         # 9 h per configuration here and changes no answer (as for MMStar and SEED)
    mathvista*)               export GEN_L=100 ;;
    mathverse*)               export GEN_L=100 ;;
    llava*)                   export GEN_L=512 ;;        # LLaVA-Bench: the Table VIII length
    *)                        export GEN_L=16 ;;         # MMStar, SEED, MMBench-type answers: LaViDa's MME length. Its wrapper
                                                         # default of 256 tokens per MCQ item would cost ~2 days per SEED configuration
  esac
  export GEN_BLOCK=$(( GEN_L < 128 ? GEN_L : 128 )) GEN_STEPS=$GEN_L
else
  case $TASK in
    mathvista*)               export GEN_L=96  GEN_STEPS=96  GEN_BLOCK=48 ;;
    mathverse*)               export GEN_L=256 GEN_STEPS=128 GEN_BLOCK=32 ;;
    docvqa_val*)              export GEN_L=32  GEN_STEPS=32  GEN_BLOCK=16 ;;
    chartqa*)                 export GEN_L=16  GEN_STEPS=16  GEN_BLOCK=16 ;;
    llava*)                   export GEN_L=512 GEN_STEPS=256 GEN_BLOCK=128 ;;   # LLaVA-Bench: MMaDA's released long-form (MMVet) setting
    *)                        export GEN_L=2   GEN_STEPS=2   GEN_BLOCK=2 ;;
  esac
fi
GK='{"temperature":0,"max_new_tokens":'$GEN_L'}'
# GPT-based scorers (MathVista, MathVerse) cannot reach a gateway from here: fail fast, keep the samples, score later
export OPENAI_API_URL=http://127.0.0.1:9/v1/chat/completions OPENAI_API_KEY=none
OUT=$E/results/${BENCH_ROOT:-bench_$MODEL}/$CFG/$TASK; mkdir -p $OUT; rm -f $OUT/DONE; touch $OUT/.start
echo "[bench_eval_model] model=$MODEL config=$CFG task=$TASK L=$GEN_L steps=$GEN_STEPS block=$GEN_BLOCK backend=${COTA_BACKEND} margs=${MARGS:-none} out=$OUT"
cd $Z/autodl-tmp/LLaDA-V/eval/lmms-eval
if [ "${NPROC:-1}" -gt 1 ]; then LAUNCH="$PY -m accelerate.commands.launch --num_processes ${NPROC} --main_process_port $((20000 + RANDOM % 20000)) -m lmms_eval"; else LAUNCH="$PY -m lmms_eval"; fi
$LAUNCH --model $WRAPPER ${MARGS:+--model_args "$MARGS"} --gen_kwargs "$GK" --tasks $TASK --batch_size 1 \
  --log_samples --log_samples_suffix $TASK --output_path $OUT "$@"
rc=$?
if [ $rc -eq 0 ] && [ -n "$(find $OUT -name "*results*.json" -newer $OUT/.start 2>/dev/null | head -1)" ]; then
  date '+%F %T' > $OUT/DONE; exit 0
fi
echo "bench_eval_model: no results file for $MODEL/$CFG/$TASK (rc=$rc)"; exit 3
