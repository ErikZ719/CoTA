#!/bin/bash
# F1 multi-sample attention collection (20 images x 2 modes).
# Same seed/order as run_multisample.sh, so images line up with the entropy data.
set -eo pipefail
mkdir -p /root/autodl-tmp/information_flow/attn_multi
LOG=/root/autodl-tmp/information_flow/attn_multi/run.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) F1 attention collection start ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llada-v
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
cd /root/autodl-tmp/LLaDA-V/train
export F1A_OUT=/root/autodl-tmp/information_flow/attn_multi F1A_N=20 F1A_SEED=0
F1A_MODE=dllm_cache python -u f1_attn_multisample.py
F1A_MODE=baseline   python -u f1_attn_multisample.py
echo "=== $(date) F1_ATTN_DONE ==="
