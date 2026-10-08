#!/bin/bash
# Copy the drop-in files of CoTA++ into a clone of ML-GSAI/LLaDA-V.
#   bash llada_v/apply.sh /path/to/LLaDA-V
# The clone should be at the commit in UPSTREAM_COMMIT (f8b02ce); the script warns if it is not.
# Files that exist upstream are overwritten (a .upstream copy is kept the first time); new files are added.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DST="${1:?usage: apply.sh /path/to/LLaDA-V}"
[ -d "$DST/train/llava" ] || { echo "$DST does not look like an LLaDA-V checkout"; exit 2; }
want=$(cat "$HERE/UPSTREAM_COMMIT"); have=$(git -C "$DST" rev-parse HEAD 2>/dev/null || echo unknown)
[ "$have" = "$want" ] || echo "warning: LLaDA-V is at $have, the files were made for $want"
n=0
while IFS= read -r -d '' src; do
  rel="${src#$HERE/}"
  case "$rel" in apply.sh|README.md|UPSTREAM_COMMIT|pyproject.toml.diff) continue;; esac
  dst="$DST/$rel"; mkdir -p "$(dirname "$dst")"
  if [ -e "$dst" ] && [ ! -e "$dst.upstream" ]; then cp -p "$dst" "$dst.upstream"; fi
  cp -p "$src" "$dst"; echo "$rel"; n=$((n+1))
done < <(find "$HERE" -type f -print0)
echo "copied $n files into $DST"
