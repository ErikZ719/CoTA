#!/bin/bash
# Download the benchmark datasets of Table VI through the mirror, one after another, with retries.
# Log: experiments/repeat_eval/logs/bench_download.log.  Verify afterwards with a real load (a broken
# .incomplete file can be promoted to a full one by an interrupted download).
Z=/data/zhaoqiyan
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export HF_HOME=$Z/autodl-tmp/hf_cache HUGGINGFACE_HUB_CACHE=$Z/autodl-tmp/hf_cache/hub
PY=$Z/miniconda3/envs/llada-v/bin
for repo in Lin-Chen/MMStar CaraJ/MathVerse-lmmseval AI4Math/MathVista; do   # the large sets come from ModelScope (ms_download.py)
  for try in 1 2 3 4 5; do
    echo "[$(date '+%m-%d %H:%M')] $repo (try $try)"
    if $PY/python -m huggingface_hub.commands.huggingface_cli download --repo-type dataset "$repo" --max-workers 8 >/dev/null 2>>$Z/autodl-tmp/experiments/repeat_eval/logs/bench_download.err; then
      echo "[$(date '+%m-%d %H:%M')] $repo OK  $(du -sh $HUGGINGFACE_HUB_CACHE/datasets--${repo//\//--} 2>/dev/null | cut -f1)"; break
    fi
    sleep 20
  done
done
echo "[$(date '+%m-%d %H:%M')] DOWNLOADS-FINISHED"
