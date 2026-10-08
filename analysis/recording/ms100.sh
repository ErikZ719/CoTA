#!/bin/bash
# usage: ms100.sh <baseline|dllm_cache>    extends information_flow/multisample from 60 to the 100-image sample
set -u
Z=/data/zhaoqiyan/autodl-tmp; IF=$Z/information_flow; MODE=$1
export F3_MODE=$MODE F1F3_OUT=$IF/multisample
export F1F3_IMAGES=$(/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python - <<PY
import json,os
fs=json.load(open("$IF/coco100_seed0.json"))["files"]
have={x.split("_G128_")[1][:-4] for x in os.listdir("$IF/multisample") if x.startswith("${MODE}_G128_")}
print(",".join(f for f in fs if os.path.splitext(os.path.basename(f))[0] not in have))
PY
)
[ -f $IF/multisample/${MODE}_meta.json ] && [ ! -f $IF/multisample/${MODE}_meta_first60.json ] && cp $IF/multisample/${MODE}_meta.json $IF/multisample/${MODE}_meta_first60.json
if [ -n "$F1F3_IMAGES" ]; then cd $Z/LLaDA-V/train && /data/zhaoqiyan/miniconda3/envs/llada-v/bin/python -u f1f3_multisample.py; fi
n=$(/data/zhaoqiyan/miniconda3/envs/llada-v/bin/python - <<PY
import json,os
fs=json.load(open("$IF/coco100_seed0.json"))["files"]
have={x.split("_G128_")[1][:-4] for x in os.listdir("$IF/multisample") if x.startswith("${MODE}_G128_")}
print(sum(os.path.splitext(os.path.basename(f))[0] in have for f in fs))
PY
)
echo "[ms100] $MODE images of the 100 present: $n"
[ "$n" -eq 100 ] && date > $IF/multisample/${MODE}_100.DONE
