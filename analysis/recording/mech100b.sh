#!/bin/bash
# Helper shard for a running mech100.sh job: records images [from,to) of the 100-image list into a
# SEPARATE root (attn_multi100_b) and moves each finished image into attn_multi100/<tag>/, where the
# main job then skips it. The two processes never write the same directory, so nothing can be
# clobbered; at worst an image is recorded twice.
# usage: mech100b.sh <tag> <baseline|dllm_cache> <from> <to> [E_s] [dar_r] [dar_w] [ctae] [lo] [hi] [theta]
set -u
Z=/data/zhaoqiyan/autodl-tmp; IF=$Z/information_flow; PY=/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python
TAG=$1; MODE=$2; FROM=$3; TO=$4
export F1A_TAG=$TAG F1A_MODE=$MODE F1A_OUT=$IF/attn_multi100_b
export F1A_GI=${5:-7} F1A_DAR_R=${6:-0} F1A_DAR_W=${7:-0} F1A_CTAE=${8:-off} F1A_BAND_LO=${9:-24} F1A_BAND_HI=${10:-31} F1A_CTAR_THETA=${11:-0}
export F1A_IMAGES=$($PY -c "import json,os;fs=json.load(open('$IF/coco100_seed0.json'))['files'][$FROM:$TO];print(','.join(f for f in fs if not os.path.exists('$IF/attn_multi100/$TAG/'+os.path.splitext(os.path.basename(f))[0]+'/step_127.npz')))")
mkdir -p $IF/attn_multi100_b/$TAG $IF/attn_multi100/$TAG
move_done() {   # only images whose last step file has been at rest for 90 s
  for d in $IF/attn_multi100_b/$TAG/*/; do
    [ -d "$d" ] || continue; s=$(basename $d); f=$d/step_127.npz
    [ -f "$f" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 90 ] || continue
    if [ -e $IF/attn_multi100/$TAG/$s ]; then echo "[mech100b] $s already in main root, left in place"; mv "$d" "$IF/attn_multi100_b/$TAG.dup_$s" 2>/dev/null
    else mv "$d" $IF/attn_multi100/$TAG/$s && echo "[mech100b] moved $s"; fi
  done
}
( while true; do move_done; sleep 30; done ) &
MOVER=$!
if [ -n "$F1A_IMAGES" ]; then cd $Z/LLaDA-V/train && $PY -u f1_attn_multisample.py; fi
sleep 95; move_done; kill $MOVER 2>/dev/null
date > $IF/attn_multi100_b/${TAG}_${FROM}_${TO}.DONE
