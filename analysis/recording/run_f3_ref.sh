#!/bin/bash
set -eo pipefail
LOG=/root/autodl-tmp/information_flow/f3_entropy/run_ref.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) F3 reference-sample run start ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llada-v
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
cd /root/autodl-tmp/LLaDA-V/train
F3_MODE=baseline python -u f3_layer_entropy.py
F3_MODE=dllm_cache python -u f3_layer_entropy.py
echo "=== $(date) F3_REF_DONE ==="
