#!/bin/bash
# Rewrite the machine-specific roots inside every script of this repository.
#
#   COTA_ROOT=/path/to/workspace                     the directory holding LLaDA-V/, dLLM-cache/, LaViDa/, MMaDA-official/,
#                                                    hf_cache/, datasets/, coco2014/, information_flow/
#   COTA_EXP=$COTA_ROOT/experiments/repeat_eval      the experiment tree (results/, data/, jobs/, logs/)   [default]
#   COTA_PY=python                                   interpreter used in the Mac-side scripts               [default]
#
# Usage:  COTA_ROOT=/my/workspace bash relocate.sh
# Run once, from the repository root. A backup of every rewritten file is left as <file>.orig (delete when satisfied).
set -euo pipefail
cd "$(dirname "$0")"
: "${COTA_ROOT:?set COTA_ROOT to the workspace that holds the model repositories}"
COTA_EXP="${COTA_EXP:-$COTA_ROOT/experiments/repeat_eval}"
COTA_PY="${COTA_PY:-python}"
OLD_ROOTS=("/data/zhaoqiyan/autodl-tmp" "/root/autodl-tmp" "/home/<user>/zhaoqiyan/autodl-tmp")
n=0
while IFS= read -r -d '' f; do
  if grep -qE "/data/zhaoqiyan|/root/autodl-tmp|/home/<user>|/opt/anaconda3/bin/python" "$f"; then
    cp -p "$f" "$f.orig"
    for r in "${OLD_ROOTS[@]}"; do
      sed -i.tmp -e "s#$r/experiments/repeat_eval#$COTA_EXP#g" -e "s#$r#$COTA_ROOT#g" "$f"
    done
    sed -i.tmp -e "s#/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python#$COTA_PY#g" -e "s#/home/<user>/zhaoqiyan/miniconda3/envs/llada-v/bin/python#$COTA_PY#g" \
               -e "s#/data/zhaoqiyan/miniconda3/envs/mmada/bin/python#$COTA_PY#g" \
               -e "s#/data/zhaoqiyan/venvs/lavida/bin/python#$COTA_PY#g" \
               -e "s#/opt/anaconda3/bin/python#$COTA_PY#g" \
               -e "s#/data/zhaoqiyan#$(dirname "$COTA_ROOT")#g" "$f"
    rm -f "$f.tmp"; n=$((n+1))
  fi
done < <(find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.yaml" \) -not -path "./_build/*" -not -path "./environment/*" -print0)
echo "rewrote $n files (originals kept as *.orig). Remaining machine paths:"
grep -rlE "/data/zhaoqiyan|/root/autodl-tmp|/home/<user>|/opt/anaconda3" --include=*.py --include=*.sh --include=*.txt --include=*.yaml . | grep -v "\.orig$" || echo "  none"
