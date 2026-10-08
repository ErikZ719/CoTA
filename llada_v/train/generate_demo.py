from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook

# === Optional: per-step attention visualization (Fig. 3) =====================
from llava.model.language_model.utils import (
    attach_attention_recorder, detach_attention_recorder,
    render_attention_maps, render_step_level_attention_maps,
    attach_sar, detach_sar, get_sar_summary,
)
# =============================================================================

from PIL import Image
import os
import requests
import copy
import torch
import time

import sys
import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25
use_fast_dllm = False  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = True  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

# === Visualization toggle (Fig. 3) ===========================================
# When True: save per-step attention NPZs during generation and render heatmaps
# after generation. Does NOT change the generated text output.
# Constraint: use_fast_dllm must be False (Fast-dLLM bypasses our recorder).
# Works with vanilla (use_dllm_cache=False) and cached (=True) paths.
VISUALIZE_ATTENTION = True
VIS_NPZ_ROOT = '/data/zhaoqiyan/autodl-tmp/information_flow/attn_npz'    # base; runtime appends /<mode_tag>/<stem>/
VIS_TOKEN_MAP_ROOT = '/data/zhaoqiyan/autodl-tmp/information_flow/token-level'   # per-step Q x K (existing)
VIS_STEP_MAP_ROOT  = '/data/zhaoqiyan/autodl-tmp/information_flow/step-level'    # per-layer [steps x K] (new)
VIS_LAST_Q = 128                                    # heatmap window: last Q query positions
VIS_LAST_K = 128                                    # heatmap window: last K key positions
VIS_EVERY_LAYER = True                            # True = render every decoder layer; False = last only
# =============================================================================

# === SAR: Staleness-Aware Refresh toggle =====================================
# When True: augment dLLM-Cache's similarity-based refresh with a staleness
# hard-cap: any suffix position with staleness Δ >= SAR_TAU is force-refreshed.
# Non-invasive monkey-patch (matches attention_recorder pattern).
# Requires use_dllm_cache=True to have effect (SAR wraps dLLM-Cache's refresh).
USE_SAR = True
SAR_TAU = 4                     # staleness threshold; V10-V11 calibrated for gen_length=128
SAR_GEN_LEN = 128               # match generation length
# =============================================================================

warnings.filterwarnings('ignore')
pretrained = 'GSAI-ML/LLaDA-V'

model_name = 'llava_llada'
device = 'cuda:0'
device_map = 'cuda:0'
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation='sdpa', device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()
image = Image.open('test.jpg')
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = 'llava_llada'
question = DEFAULT_IMAGE_TOKEN + '\nPlease describe the image in detail.'
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

model.eval()
if use_fast_dllm:
    register_fast_dllm_hook(model)
    print('Testing with Fast dLLM hook enabled')
elif use_dllm_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_LLaDA_V(model, 'model.layers')
    print('Testing with cache enabled')
else:
    print('Testing without cache')

# === Visualization: attach recorder (no-op if VISUALIZE_ATTENTION=False) =====
if VISUALIZE_ATTENTION:
    assert not use_fast_dllm, (
        'VISUALIZE_ATTENTION=True requires use_fast_dllm=False '
        '(Fast-dLLM replaces generate_with_embeds and bypasses our recorder).'
    )
    # Auto-tag the output by acceleration mode so different runs do not
    # overwrite each other (baseline / dllm_cache).
    # Layout:
    #   {VIS_NPZ_ROOT}/{VIS_MODE_TAG}/step_*.npz                (raw NPZs)
    #   {VIS_TOKEN_MAP_ROOT}/{VIS_MODE_TAG}/layer*/step*.jpg    (token-level Q x K)
    #   {VIS_STEP_MAP_ROOT}/{VIS_MODE_TAG}/layer{N}.png         (step-level steps x K)
    if use_dllm_cache:
        VIS_MODE_TAG = 'dllm_cache_sar' if USE_SAR else 'dllm_cache'
    else:
        VIS_MODE_TAG = 'baseline'
    vis_npz_dir = os.path.join(VIS_NPZ_ROOT, VIS_MODE_TAG)
    vis_token_map_dir = os.path.join(VIS_TOKEN_MAP_ROOT, VIS_MODE_TAG)
    vis_step_map_dir  = os.path.join(VIS_STEP_MAP_ROOT,  VIS_MODE_TAG)
    attach_attention_recorder(model, vis_npz_dir, save_last_q=VIS_LAST_Q, save_last_k=VIS_LAST_K)
    print(f'Attention recorder ON  [mode={VIS_MODE_TAG}]  -> NPZ: {vis_npz_dir}')
# =============================================================================

# === SAR: attach (no-op if USE_SAR=False or dLLM-Cache off) =================
if USE_SAR:
    if not use_dllm_cache:
        print('[SAR] WARNING: USE_SAR=True but use_dllm_cache=False; SAR only wraps dLLM-Cache. Skipping.')
    else:
        attach_sar(model, tau=SAR_TAU, gen_len=SAR_GEN_LEN)
        print(f'SAR ON  [tau={SAR_TAU}, gen_len={SAR_GEN_LEN}]')
# =============================================================================

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
image_sizes = [image.size]

start_time = time.time()
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'],
    prefix_refresh_interval=32,
    threshold=1,
)
end_time = time.time()
generation_time = end_time - start_time
print(f'Generation time: {generation_time:.4f} seconds')

print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)

# === Visualization: detach + render =========================================
if VISUALIZE_ATTENTION:
    detach_attention_recorder(model)
    os.makedirs(vis_token_map_dir, exist_ok=True)
    os.makedirs(vis_step_map_dir, exist_ok=True)
    print(f'Rendering token-level heatmaps  [mode={VIS_MODE_TAG}]  -> {vis_token_map_dir}')
    render_attention_maps(
        npz_root=VIS_NPZ_ROOT,
        map_root=VIS_TOKEN_MAP_ROOT,
        last_q=VIS_LAST_Q,
        last_k=VIS_LAST_K,
        every_layer=VIS_EVERY_LAYER,
        only_stems=[VIS_MODE_TAG],
    )
    print(f'Rendering step-level heatmaps   [mode={VIS_MODE_TAG}]  -> {vis_step_map_dir}')
    render_step_level_attention_maps(
        npz_root=VIS_NPZ_ROOT,
        map_root=VIS_STEP_MAP_ROOT,
        only_stems=[VIS_MODE_TAG],
    )
    print(f'Done. NPZ: {vis_npz_dir}/, TOKEN-MAP: {vis_token_map_dir}/, STEP-MAP: {vis_step_map_dir}/')

# === SAR: detach + summary ===================================================
if USE_SAR and use_dllm_cache:
    sar_summary = detach_sar()
    if sar_summary:
        print(f'[SAR] final Δ stats: {sar_summary["current_delta_stats"]}')
        print(f'[SAR] step boundaries={sar_summary["n_step_boundaries"]}, '
              f'refresh calls={sar_summary["n_refresh_calls"]}, '
              f'kicks={sar_summary["total_kicks"]}, saves={sar_summary["total_saves"]}')
# =============================================================================

# =============================================================================
