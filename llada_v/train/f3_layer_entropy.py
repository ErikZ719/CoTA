"""F3 reproduction: cross-layer (logit-lens) entropy of the gen window at every
denoising step, for vanilla LLaDA-V and LLaDA-V + dLLM-Cache.

For each decoding step and each depth d in {0 = embedding, 1..32 = decoder
blocks}, take the hidden state the model actually uses at that step (under
caching this includes stale cached states), apply the final RMSNorm + lm_head,
and record the per-position vocabulary entropy in bits over the gen window.

Output per run: <out>/<mode>_G<gen>_<image_stem>.npz with
    entropy_bits  float16 [n_steps, 33, gen_length]
plus a sidecar json with the decoded text and config.

Env knobs:
    F3_MODE       baseline | dllm_cache          (default baseline)
    F3_IMAGES     comma-separated image paths    (default train/test.jpg)
    F3_GEN_LENGTH / F3_GEN_STEPS / F3_BLOCK_LENGTH   (default 128)
    F3_OUT        output dir (default /data/zhaoqiyan/autodl-tmp/information_flow/f3_entropy)
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', '/data/zhaoqiyan/autodl-tmp/hf_cache')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/data/zhaoqiyan/autodl-tmp/hf_cache/hub')

import copy
import json
import math
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from PIL import Image

warnings.filterwarnings('ignore')

MODE = os.environ.get('F3_MODE', 'baseline')
assert MODE in ('baseline', 'dllm_cache'), MODE
IMAGES = os.environ.get('F3_IMAGES', '/data/zhaoqiyan/autodl-tmp/LLaDA-V/train/test.jpg').split(',')
GEN = int(os.environ.get('F3_GEN_LENGTH', 128))
STEPS = int(os.environ.get('F3_GEN_STEPS', 128))
BLOCK = int(os.environ.get('F3_BLOCK_LENGTH', 128))
OUT = os.environ.get('F3_OUT', '/data/zhaoqiyan/autodl-tmp/information_flow/f3_entropy')
PROMPT_INTERVAL, GEN_INTERVAL, TRANSFER_RATIO = 25, 7, 0.25
QUESTION = 'Please describe the image in detail.'

os.makedirs(OUT, exist_ok=True)
device = 'cuda:0'

tokenizer, model, image_processor, _ = load_pretrained_model(
    'GSAI-ML/LLaDA-V', None, 'llava_llada',
    attn_implementation='sdpa', device_map=device)
model.eval()

if MODE == 'dllm_cache':
    dLLMCache.new_instance(**asdict(dLLMCacheConfig(
        prompt_interval_steps=PROMPT_INTERVAL,
        gen_interval_steps=GEN_INTERVAL,
        transfer_ratio=TRANSFER_RATIO)))
    register_cache_LLaDA_V(model, 'model.layers')
    print(f'[f3] dLLM-Cache ON ({PROMPT_INTERVAL},{GEN_INTERVAL},{TRANSFER_RATIO})')
else:
    print('[f3] baseline (no cache)')

core = model.model                    # LLaDAModel: .layers, .norm
layers = core.layers
final_norm = core.norm
lm_head = model.lm_head
N_LAYERS = len(layers)
LOG2E = 1.0 / math.log(2.0)

step_rows = []          # one [33, GEN] array per decoding step
_current = {}


def entropy_bits(h):
    """h: [1, seq, d] hidden state -> [GEN] entropy in bits (norm+lm_head lens)."""
    with torch.no_grad():
        z = lm_head(final_norm(h[:, -GEN:, :])).float()   # [1, GEN, V]
        logp = torch.log_softmax(z, dim=-1)
        ent = -(logp.exp() * logp).sum(-1) * LOG2E
    return ent[0].to(torch.float16).cpu().numpy()


def pre_hook(module, args, kwargs=None):
    h = args[0] if args else kwargs['hidden_states']
    _current[0] = entropy_bits(h)


def make_hook(depth):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        _current[depth] = entropy_bits(h)
        if depth == N_LAYERS:
            step_rows.append(np.stack([_current[d] for d in range(N_LAYERS + 1)]))
            _current.clear()
    return hook


handles = [layers[0].register_forward_pre_hook(pre_hook)]
for li, blk in enumerate(layers):
    handles.append(blk.register_forward_hook(make_hook(li + 1)))

conv_template = 'llava_llada'
records = []
for img_path in IMAGES:
    img_path = img_path.strip()
    stem = os.path.splitext(os.path.basename(img_path))[0]
    step_rows.clear()
    _current.clear()

    image = Image.open(img_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [t.to(dtype=torch.float16, device=device) for t in image_tensor]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + QUESTION)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors='pt').unsqueeze(0).to(device)

    t0 = time.time()
    cont = model.generate(
        input_ids, images=image_tensor, image_sizes=[image.size],
        steps=STEPS, gen_length=GEN, block_length=BLOCK,
        tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'],
        prefix_refresh_interval=32, threshold=1)
    dt = time.time() - t0

    text_full = tokenizer.batch_decode(cont, skip_special_tokens=False)[0]
    grid = np.stack(step_rows) if step_rows else np.zeros((0, N_LAYERS + 1, GEN))
    gen_ids = cont[0, -GEN:].tolist()
    gen_tokens = [tokenizer.convert_ids_to_tokens(t) for t in gen_ids]

    out_npz = os.path.join(OUT, f'{MODE}_G{GEN}_{stem}.npz')
    np.savez_compressed(out_npz,
                        entropy_bits=grid.astype(np.float16),
                        gen_ids=np.array(gen_ids, dtype=np.int64))
    rec = {'image': img_path, 'mode': MODE,
           'gen_length': GEN, 'steps': STEPS, 'block_length': BLOCK,
           'n_forward_steps': int(grid.shape[0]),
           'generation_time_sec': round(dt, 2),
           'gen_tokens': gen_tokens,
           'text_output': text_full,
           'npz': out_npz}
    records.append(rec)
    with open(os.path.join(OUT, f'{MODE}_G{GEN}_meta.json'), 'w') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'[f3] {stem}: {grid.shape[0]} steps x {grid.shape[1]} depths, '
          f'{dt:.1f}s -> {out_npz}', flush=True)
    print('[f3] text:', text_full[:200].replace('\n', ' '), flush=True)

for h in handles:
    h.remove()
print('[f3] ALL DONE', flush=True)
