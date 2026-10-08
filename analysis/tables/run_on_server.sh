#!/bin/bash
# Server-side launcher of judge_table6.py. The key is read from STDIN (one line) into the environment of this
# process only: it never appears on a command line, in a file or in a log.
#   printf '%s\n' "$KEY" | ssh hgx030 'bash /data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/judge_t6/run_on_server.sh [args]'
D=/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval
read -r JUDGE_API_KEY; export JUDGE_API_KEY
export T6_SCORER=$D/judge_t6 T6_TEXTS=$D/results/table5_texts T6_CAPS=$D/judge_t6/coco500_captions.json \
       T6_EVAL=$D/data/coco500_final.json T6_CACHE=$D/judge_t6/cache.jsonl T6_OUT=$D/judge_t6/judge_cells.json
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY        # the server reaches the gateway directly
cd $D/judge_t6
if [ "$1" = "--detach" ]; then shift
  setsid nohup /data/zhaoqiyan/miniconda3/envs/llada-v/bin/python -u judge_table6.py "$@" > run_full.log 2> run_full.err < /dev/null &
  echo "detached pid $!"
else
  /data/zhaoqiyan/miniconda3/envs/llada-v/bin/python -u judge_table6.py "$@"
fi
