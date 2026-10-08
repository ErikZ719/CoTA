#!/bin/bash
# Judge the n-gram baseline runs after the nine-cell re-judge has finished and the ng shards are complete (2026-09-30).
# The key is read from STDIN into this process's environment only.
#   printf '%s\n' "$KEY" | ssh hgx030 'bash .../judge_t6/run_extra_after.sh --detach'
D=/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval
read -r JUDGE_API_KEY; export JUDGE_API_KEY
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
PY=/data/zhaoqiyan/miniconda3/envs/mmada/bin/python
run() {
  until grep -q "^judged" $D/judge_t6/run_full.log 2>/dev/null; do sleep 300; done
  until [ -f $D/results/lladav/exp3/ng2_128_0/summary.json ] && [ -f $D/results/lladav/exp3/ng2_128_250/summary.json ] \
     && [ -f $D/results/lladav/exp3/ng3_128_0/summary.json ] && [ -f $D/results/lladav/exp3/ng3_128_250/summary.json ]; do sleep 300; done
  cd $D/judge_t6
  $PY -u judge_extra.py --label ng2 --runs 'lladav/exp3/ng2_128_*' --label ng3 --runs 'lladav/exp3/ng3_128_*' --workers 2
  echo EXTRA-SCORING-COMPLETE
}
if [ "$1" = "--detach" ]; then
  setsid nohup bash -c "$(declare -f run); D=$D; PY=$PY; run" > $D/judge_t6/extra_scoring.log 2>&1 < /dev/null &
  echo "detached pid $!"
else
  run
fi
