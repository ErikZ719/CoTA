#!/bin/bash
set -eo pipefail
LOG=/root/autodl-tmp/information_flow/f3_entropy/run_coco.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) F3 COCO-100 run start ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llada-v
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
cd /root/autodl-tmp/LLaDA-V/train
IMGS=$(python - <<'PY'
import os, random
files = sorted(f for f in os.listdir('/root/autodl-tmp/coco2014/val2014') if f.lower().endswith(('.jpg','.jpeg','.png')))
random.seed(0)
picks = random.sample(files, 100)
print(','.join('/root/autodl-tmp/coco2014/val2014/' + f for f in picks))
PY
)
export F3_OUT=/root/autodl-tmp/information_flow/f3_entropy/coco100
F3_MODE=dllm_cache F3_IMAGES="$IMGS" python -u f3_layer_entropy.py
F3_MODE=baseline F3_IMAGES="$IMGS" python -u f3_layer_entropy.py
echo "=== $(date) F3_COCO_DONE ==="
