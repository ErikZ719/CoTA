#!/bin/bash
# Launcher for score_math_from_samples.py on dgx056. The key is read from STDIN into the environment of this
# process only: never on a command line, in a file or in a log. T6_BENCH_ROOT picks the model's results root.
#   printf '%s\n' "$KEY" | ssh <loaner-machine> 'T6_BENCH_ROOT=bench_lavida bash .../judge_t6/run_math_scoring.sh --detach --workers 6'
D=/home/<user>/zhaoqiyan/autodl-tmp/experiments/repeat_eval
read -r JUDGE_API_KEY; export JUDGE_API_KEY
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export T6_BENCH_ROOT=${T6_BENCH_ROOT:-bench}
cd $D/judge_t6
PY=${COTA_PY:-python}
TAG=${T6_BENCH_ROOT#bench}; TAG=${TAG#_}; TAG=${TAG:-lladav}
if [ "$1" = "--detach" ]; then shift
  setsid nohup $PY -u score_math_from_samples.py "$@" > math_scoring_$TAG.log 2> math_scoring_$TAG.err < /dev/null &
  echo "detached pid $! (root $T6_BENCH_ROOT, logs math_scoring_$TAG.log)"
else
  $PY -u score_math_from_samples.py "$@"
fi
