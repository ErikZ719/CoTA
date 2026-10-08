#!/bin/bash
# Uncached baseline for Sec. VI-E: the same 12 images, L=512, one idle card, so the CoTA++ overhead
# measured against dLLM-Cache can be put next to the speedup caching buys in the first place.
set -u
Z=/data/zhaoqiyan; E=$Z/autodl-tmp/experiments/repeat_eval
OUT=$E/results/lladav/lat_vie/vanilla
source $Z/miniconda3/etc/profile.d/conda.sh; conda activate llada-v
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $Z/autodl-tmp/LLaDA-V/train
rm -rf $OUT
CUDA_VISIBLE_DEVICES=0 python -u $E/scripts/run_repeat_eval.py --images $E/data/coco500_final.json \
  --out $OUT --length 512 --offset 0 --limit 12 --mode baseline --device cuda:0 \
  > $E/logs/lat_vie_vanilla.log 2>&1
python3 -c "
import json;d=json.load(open(\"$OUT/summary.json\"));print(\"vanilla %.1f s/image\"%(d[\"wall_seconds\"]/12))" | tee -a $E/logs/timing_vie.log
