#!/bin/bash
# Multi-sample validation for F1 regional effect + F3 independence.
# Waits for any running F3 batch to finish, then runs cache and baseline passes.
set -eo pipefail
LOG=/root/autodl-tmp/information_flow/multisample/run.log
mkdir -p /root/autodl-tmp/information_flow/multisample
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) multisample start ==="
while pgrep -f 'f3_layer_entropy.py' > /dev/null; do echo "waiting for f3 batch..."; sleep 60; done
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llada-v
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
cd /root/autodl-tmp/LLaDA-V/train
export F1F3_OUT=/root/autodl-tmp/information_flow/multisample
export F1F3_N=60 F1F3_SEED=0
F3_MODE=dllm_cache python -u f1f3_multisample.py
F3_MODE=baseline  python -u f1f3_multisample.py
echo "=== $(date) MULTISAMPLE_DONE ==="
