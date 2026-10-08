"""
Attention Recorder for LLaDA-V's generate_with_embeds.

PURPOSE
-------
Add per-step attention-matrix saving to LLaDA-V's diffusion generation loop
WITHOUT touching the upstream model code. Used to produce the cross-step
attention visualizations in our paper's Fig. 3.

DESIGN
------
1. attach_attention_recorder(model, npz_dir): replaces model.generate_with_embeds
   with an instrumented copy that mirrors the upstream logic 1:1 and adds save
   I/O at the end of each diffusion step.
2. detach_attention_recorder(model): restores the original method.
3. render_attention_maps(npz_root, map_root): runs vis-scripts/vis_attention.py
   as a subprocess to render Q x K heatmaps from the saved NPZs.

Idempotent: re-attaching just updates npz_dir.

NPZ SCHEMA per step (matches vis-scripts/vis_attention.py)
---------------------------------------------------------
quantized_attentions  int8     [layers, B, H, Q, K]   range -128..127
scale, zero_point     float32                          dequant params
token_ids             int64    [seq_len]   IDs AFTER this step's update
prev_token_ids        int64    [seq_len]   IDs BEFORE this step's update
transfer_index        bool     [seq_len]   True at newly unmasked positions
prompt_length         int64                            prefix length
"""

import os
import subprocess
import sys
import threading
import types

import numpy as np
import torch
import torch.nn.functional as F


# LLaDA's [MASK] token id; constant mirrored from modeling_llada.py.
_MASK_ID = 126336
# Attribute name we attach to a model instance to store recorder state.
_RECORDER_STATE_ATTR = '_attention_recorder_state'


def _quantize_and_save_step(
    npz_dir, step_idx, attentions_tuple,
    token_ids, prev_token_ids, transfer_index, prompt_length,
    save_last_q=None, save_last_k=None,   # kept for API compat; cropping now done in hook
    refresh_q_index=None,                 # 1D int64 gen-region indices recomputed THIS step
):
    """Int8-quantize per-layer attentions and persist as step_{i}.npz.

    attentions_tuple is expected to be a list of ALREADY-CROPPED CPU float32
    tensors (one per decoder layer), populated by _LayerAttentionCollector.
    save_last_q / save_last_k are accepted but unused (preserved for back-compat).
    """
    if len(attentions_tuple) == 0:
        return  # no attentions captured this step
    step_attns = torch.stack(list(attentions_tuple), dim=0)

    q_min, q_max = -128, 127
    a_max = step_attns.max()
    a_min = step_attns.min()
    scale = (a_max - a_min) / (q_max - q_min)
    if torch.is_tensor(scale) and scale.item() == 0:
        scale = torch.tensor(1.0)
    zero_point = q_min - torch.round(a_min / scale)

    quantized = torch.clamp(
        torch.round(step_attns / scale) + zero_point, q_min, q_max,
    ).to(torch.int8)

    path = os.path.join(npz_dir, f'step_{step_idx}.npz')
    np.savez_compressed(
        path,
        quantized_attentions=quantized.numpy(),
        scale=np.float32(scale.item() if torch.is_tensor(scale) else scale),
        zero_point=np.float32(
            zero_point.item() if torch.is_tensor(zero_point) else zero_point
        ),
        token_ids=token_ids[0].detach().cpu().numpy().astype(np.int64),
        prev_token_ids=prev_token_ids[0].detach().cpu().numpy().astype(np.int64),
        transfer_index=transfer_index[0].detach().cpu().numpy().astype(bool),
        prompt_length=np.int64(prompt_length),
        refresh_q_index=(np.asarray(refresh_q_index, dtype=np.int64)
                         if refresh_q_index is not None
                         else np.array([], dtype=np.int64)),
    )


# Thread-local context populated by the per-layer attention wrapper below.
# F.softmax interception reads it to know (layer_idx, q_index) for the current call.
_cache_capture_state = threading.local()

def _free_layer_attn_hook(module, inputs, output):
    """Forward-hook: NULL out the attn_weights slot in a decoder layer’s
    output tuple to free the per-layer attention tensor accumulated in
    all_self_attns when output_attentions=True. The softmax interception has
    already captured a CPU-cropped copy by the time this fires.
    """
    if isinstance(output, tuple) and len(output) >= 2:
        return (output[0], None) + tuple(output[2:])
    return output



def _make_cache_attn_wrapper(original_fn, layer_idx):
    """Wrap llada_attention_hook_for_cache to publish (layer_idx, q_index)
    into thread-local before delegating to the original function."""
    def wrapped(self_attn, q_in_proj, k_in_proj, v_in_proj,
                attention_bias=None, layer_past=None, use_cache=False,
                q_index=None):
        _cache_capture_state.layer_idx = layer_idx
        _cache_capture_state.q_index = q_index
        try:
            return original_fn(q_in_proj, k_in_proj, v_in_proj, attention_bias,
                               layer_past, use_cache, q_index)
        finally:
            _cache_capture_state.layer_idx = None
            _cache_capture_state.q_index = None
    return wrapped


class _AttentionCaptureContext:
    """Context manager that captures attention matrices via F.softmax interception.

    Works uniformly for BOTH:
      (a) vanilla path: LLaDASdpaAttention falls back to eager when
          output_attentions=True; the eager attention calls F.softmax on a
          [B, H, Q, K] tensor.
      (b) dLLM-Cache path: llada_attention_hook_for_cache also calls F.softmax
          on a [B, H, Q, K] tensor; layer-level forward hooks would not catch
          this because the cache hook discards attn_weights before returning.

    On every F.softmax call inside the context, if the input is 4-D (= an
    attention matrix), we crop it on GPU to (last_q x last_k), move to CPU
    float32, and append to self.captures (one tensor per decoder layer).
    """

    def __init__(self, save_last_q, save_last_k, cache_mode=False):
        self.save_last_q = save_last_q
        self.save_last_k = save_last_k
        self.cache_mode = cache_mode
        # Vanilla mode: list of CPU float32 tensors [B, H, save_last_q, save_last_k], one per layer
        # Cache mode:   list of (layer_idx, q_index_cpu_or_None, attn[B, H, q_curr, save_last_k])
        self.captures = []
        self._orig_softmax = None

    def __enter__(self):
        self.captures = []
        self._orig_softmax = F.softmax

        save_last_q = self.save_last_q
        save_last_k = self.save_last_k
        cache_mode = self.cache_mode
        captures = self.captures
        orig = self._orig_softmax

        def capturing_softmax(input, dim=None, _stacklevel=3, dtype=None):
            result = orig(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
            # Filter: 4-D softmax inputs are almost always attention matrices.
            if input.dim() == 4:
                if cache_mode:
                    layer_idx = getattr(_cache_capture_state, 'layer_idx', None)
                    q_index = getattr(_cache_capture_state, 'q_index', None)
                    if layer_idx is not None:
                        # Q has variable length (= q_curr, set of refreshed positions).
                        # Crop only K to last save_last_k; keep all rows so the scatter
                        # logic in the outer loop can place them at the right Q indices.
                        attn_cropped = result[..., :, -save_last_k:].detach().to(torch.float32).cpu()
                        q_index_cpu = q_index.detach().cpu() if q_index is not None else None
                        captures.append((layer_idx, q_index_cpu, attn_cropped))
                    # If layer_idx is None, this softmax call was not inside a cache
                    # attention path (e.g., some auxiliary softmax). Skip.
                else:
                    # Vanilla: full Q always computed; crop both Q and K.
                    cropped_gpu = result[..., -save_last_q:, -save_last_k:].detach()
                    captures.append(cropped_gpu.to(torch.float32).cpu())
            return result

        F.softmax = capturing_softmax
        return self

    def __exit__(self, exc_type, exc, tb):
        F.softmax = self._orig_softmax
        return False


def _instrumented_generate_with_embeds(
    self, inputs_embeds, steps=128, gen_length=128, block_length=128,
    temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=_MASK_ID,
    tokenizer=None, stopping_criteria=None, generation_suffix=None, **kwargs,
):
    """Drop-in replacement for LLaDAModelLM.generate_with_embeds.

    Identical generation logic to the upstream method; only changes:
      - output_attentions=True forced on self.model(...) calls
      - per-step int8 NPZ written to state['npz_dir']
    Marked with '# [RECORDER]' comments at each delta site.
    """
    from llava.cache import dLLMCache  # local import to keep this file decoupled

    state = getattr(self, _RECORDER_STATE_ATTR)
    npz_dir = state['npz_dir']
    save_last_q = state.get('save_last_q') or gen_length
    save_last_k = state.get('save_last_k') or gen_length
    cache_mode = state.get('cache_mode', False)
    step_counter = 0
    prompt_length = int(inputs_embeds.shape[1])
    # Persistent attention buffer for cache mode -- holds last-known attention
    # for every Q position so that un-refreshed rows carry over across steps.
    # Lazy init after the first capture (we learn num_layers from the captures).
    attn_buffer = None  # shape: [num_layers, B, H, save_last_q, save_last_k]
    print(
        f'[attention_recorder] instrumented generate_with_embeds running '
        f'(prompt_len={prompt_length}, gen_length={gen_length}, steps={steps}, '
        f'crop=last_q={save_last_q} x last_k={save_last_k}, '
        f'cache_mode={cache_mode})'
    )

    with torch.cuda.amp.autocast(enabled=True):
        suffix_embeds = None
        suffix_token_ids = None
        suffix_len = 0
        if generation_suffix is not None and tokenizer is not None and len(generation_suffix) > 0:
            suffix_token_ids = tokenizer.encode(generation_suffix, add_special_tokens=False)
            suffix_token_ids = torch.tensor(
                suffix_token_ids, dtype=torch.long, device=inputs_embeds.device,
            ).unsqueeze(0)
            suffix_embeds = self.model.embed_tokens(suffix_token_ids)
            suffix_len = suffix_embeds.shape[1]
        else:
            suffix_len = 0

        total_length = inputs_embeds.shape[1] + gen_length + suffix_len
        masked_embed = self.model.embed_tokens(
            torch.tensor([mask_id]).to(inputs_embeds.device)
        )
        x_embeds = masked_embed.repeat(1, total_length, 1).to(inputs_embeds.device)
        x_embeds[:, :inputs_embeds.shape[1]] = inputs_embeds.clone()
        if suffix_embeds is not None:
            x_embeds[:, -suffix_len:] = suffix_embeds

        x = torch.full(
            (1, total_length), mask_id, dtype=torch.long, device=inputs_embeds.device,
        )
        if suffix_token_ids is not None:
            x[:, -suffix_len:] = suffix_token_ids

        prompt_index = torch.zeros(
            (1, total_length), dtype=torch.bool, device=inputs_embeds.device,
        )
        prompt_index[:, :inputs_embeds.shape[1]] = 1

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps = steps // num_blocks

        stop_position = inputs_embeds.shape[1] + gen_length
        found_stop_seq = False

        stop_tokens = []
        if stopping_criteria is not None:
            assert tokenizer is not None
            for stop_str in stopping_criteria:
                tokens = tokenizer.encode(stop_str, add_special_tokens=False)
                stop_tokens.append(tokens)

        feature_cache = dLLMCache()
        feature_cache.reset_cache(inputs_embeds.shape[1])

        for num_block in range(num_blocks):
            block_start = inputs_embeds.shape[1] + num_block * block_length
            block_end = inputs_embeds.shape[1] + (num_block + 1) * block_length

            if found_stop_seq and stop_position <= block_start:
                break

            block_embeds = x_embeds[:, block_start:block_end]
            block_mask_index = torch.all(
                torch.abs(block_embeds - masked_embed) < 1e-5, dim=2,
            )
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = torch.all(
                    torch.abs(x_embeds - masked_embed) < 1e-5, dim=2,
                )

                if found_stop_seq:
                    pre_stop_masks = mask_index[0, inputs_embeds.shape[1]:stop_position]
                    if not pre_stop_masks.any():
                        break

                current_block_masks = mask_index[0, block_start:block_end]
                if not current_block_masks.any():
                    break

                # [RECORDER] snapshot x before update so we can save prev_token_ids
                prev_x_snapshot = x.clone()
                # [RECORDER] capture attentions of this forward via F.softmax hook
                _cap = _AttentionCaptureContext(save_last_q, save_last_k, cache_mode=cache_mode)

                if cfg_scale > 0.:
                    un_embeds = x_embeds.clone()
                    un_mask = prompt_index.unsqueeze(-1).expand_as(x_embeds)
                    un_embeds[un_mask] = masked_embed.repeat(
                        x_embeds.shape[0], x_embeds.shape[1], 1,
                    )[un_mask]
                    combined_embeds = torch.cat([x_embeds, un_embeds], dim=0)
                    # [RECORDER] output_attentions=True forces SDPA -> eager so F.softmax fires
                    with _cap:
                        outputs = self.model(
                            inputs_embeds=combined_embeds, output_attentions=True,
                        )
                    logits = self.lm_head(outputs[0]).float()
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # [RECORDER] output_attentions=True forces SDPA -> eager so F.softmax fires
                    with _cap:
                        outputs = self.model(
                            inputs_embeds=x_embeds, output_attentions=True,
                        )
                    logits = self.lm_head(outputs[0]).float()

                for token_id in [126081, 126080, 126346, 126347]:
                    logits[:, :, token_id] = torch.where(
                        mask_index, -float('inf'), logits[:, :, token_id],
                    )

                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1,
                    )
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                if found_stop_seq:
                    x0_p[:, stop_position:] = -np.inf
                else:
                    x0_p[:, block_end:] = -np.inf

                if suffix_len > 0:
                    x0_p[:, -suffix_len:] = -np.inf

                x0_embeds = self.model.embed_tokens(x0)
                x0_embeds = torch.where(
                    mask_index.unsqueeze(-1).expand_as(x_embeds), x0_embeds, x_embeds,
                )
                x0 = torch.where(mask_index, x0, x)

                confidence = torch.where(mask_index, x0_p, -np.inf)

                # [COMPONENTS] the recorder owns a private copy of this loop; without
                # these two calls any attention collected under DAR / CTAR-theta would
                # in fact be plain-cache attention.
                if _COMPONENTS['dar']:
                    from llava.hooks import cache_hook_LLaDA_V as _ch_c
                    _g0c = inputs_embeds.shape[1]
                    _ch_c.publish_scores(confidence[0, _g0c:_g0c + gen_length].detach())

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True

                x_embeds[transfer_index] = x0_embeds[transfer_index]
                x[transfer_index] = x0[transfer_index]

                if _COMPONENTS['ctar_theta']:
                    from llava.hooks import cache_hook_LLaDA_V as _ch_c2
                    _g0m = inputs_embeds.shape[1]
                    _ch_c2.publish_mask((x[0, _g0m:_g0m + gen_length] == mask_id).detach())

                # [RECORDER] save this step's attentions + metadata
                try:
                    if cache_mode:
                        # _cap.captures is a list of (layer_idx, q_index, attn[B,H,q_curr,K]) tuples.
                        # Scatter updates into the persistent buffer at the q_index rows.
                        if _cap.captures:
                            # Lazy-init the buffer from the first batch of captures.
                            if attn_buffer is None:
                                _num_layers = max(t[0] for t in _cap.captures) + 1
                                _sample = _cap.captures[0][2]
                                _B, _H, _, _K = _sample.shape
                                attn_buffer = torch.zeros(
                                    _num_layers, _B, _H, save_last_q, _K,
                                    dtype=torch.float32,
                                )
                            _refresh_q_set = set()
                            for layer_idx, q_index, attn in _cap.captures:
                                q_curr = attn.shape[-2]
                                if q_index is not None:
                                    # q_index is [B, Q]; reduce to 1D positions.
                                    q_idx_flat = q_index[0] if q_index.dim() == 2 else q_index
                                    q_index_gen = q_idx_flat - prompt_length
                                    valid_mask = (q_index_gen >= 0) & (q_index_gen < save_last_q)
                                    valid_q = q_index_gen[valid_mask]
                                    if len(valid_q) > 0:
                                        # attn is [B, H, q_curr, K]; index along dim 2.
                                        valid_attn = attn[:, :, valid_mask, :]
                                        attn_buffer[layer_idx][:, :, valid_q, :] = valid_attn
                                        _refresh_q_set.update(int(x) for x in valid_q.tolist())
                                else:
                                    # No q_index given. Assume full-suffix recompute
                                    # if q_curr == save_last_q, else fall back to last-q_curr rows.
                                    if q_curr >= save_last_q:
                                        attn_buffer[layer_idx] = attn[..., -save_last_q:, :].clone()
                                        _refresh_q_set.update(range(save_last_q))
                                    elif q_curr > 0:
                                        attn_buffer[layer_idx][:, :, -q_curr:, :] = attn
                                        _refresh_q_set.update(range(save_last_q - q_curr, save_last_q))
                        # CTAR re-derives the routing of rows the backend never touched;
                        # overwrite those rows with what it actually used.
                        if _COMPONENTS['ctar_A'] and attn_buffer is not None:
                            from llava.hooks import cache_hook_LLaDA_V as _ch_p2
                            for _l in range(attn_buffer.shape[0]):
                                _pa = _ch_p2.take_pub_A(_l)
                                if _pa is None:
                                    continue
                                _A, _idx = _pa
                                _A = _A.to(attn_buffer.dtype)
                                _K = attn_buffer.shape[-1]
                                if _A.shape[-1] != _K:
                                    continue
                                if _idx is None:
                                    if _A.shape[-2] == save_last_q:
                                        attn_buffer[_l][0, :, :, :] = _A
                                        _refresh_q_set.update(range(save_last_q))
                                else:
                                    _v = _idx[(_idx >= 0) & (_idx < save_last_q)]
                                    if _v.numel():
                                        attn_buffer[_l][0][:, _v, :] = _A[:, :_v.numel(), :]
                                        _refresh_q_set.update(int(z) for z in _v.tolist())
                        # Save the FULL buffer (with carried-over cached rows where the
                        # model itself reused cached features) as this step's NPZ.
                        if attn_buffer is not None:
                            full_attentions = [attn_buffer[l].clone()
                                               for l in range(attn_buffer.shape[0])]
                            _quantize_and_save_step(
                                npz_dir=npz_dir,
                                step_idx=step_counter,
                                attentions_tuple=full_attentions,
                                token_ids=x,
                                prev_token_ids=prev_x_snapshot,
                                transfer_index=transfer_index,
                                prompt_length=prompt_length,
                                refresh_q_index=sorted(_refresh_q_set),
                            )
                    else:
                        _quantize_and_save_step(
                            npz_dir=npz_dir,
                            step_idx=step_counter,
                            attentions_tuple=_cap.captures,
                            token_ids=x,
                            prev_token_ids=prev_x_snapshot,
                            transfer_index=transfer_index,
                            prompt_length=prompt_length,
                        )
                except Exception as e:
                    print(
                        f'[attention_recorder] WARN: failed to save step {step_counter}: {e}',
                        file=sys.stderr,
                    )
                step_counter += 1

                if stopping_criteria is not None:
                    generated_part = x[
                        0, inputs_embeds.shape[1]:inputs_embeds.shape[1] + gen_length
                    ]
                    current_stop_position = None
                    for stop_seq in stop_tokens:
                        if not isinstance(stop_seq, list):
                            stop_seq = [stop_seq]
                        for start_idx in range(generated_part.size(0) - len(stop_seq) + 1):
                            if torch.all(
                                generated_part[start_idx:start_idx + len(stop_seq)]
                                == torch.tensor(stop_seq, device=x.device)
                            ):
                                current_position = inputs_embeds.shape[1] + start_idx
                                if not found_stop_seq or current_position < stop_position:
                                    stop_position = current_position
                                    found_stop_seq = True
                                break
                        if found_stop_seq and current_stop_position is None:
                            break

        if found_stop_seq:
            if suffix_len > 0:
                return torch.cat([
                    x[:, inputs_embeds.shape[1]:stop_position],
                    x[:, -suffix_len:],
                ], dim=1)
            else:
                return x[:, inputs_embeds.shape[1]:stop_position]
        else:
            if suffix_len > 0:
                return torch.cat([
                    x[:, inputs_embeds.shape[1]:inputs_embeds.shape[1] + gen_length],
                    x[:, -suffix_len:],
                ], dim=1)
            else:
                return x[:, inputs_embeds.shape[1]:inputs_embeds.shape[1] + gen_length]


_COMPONENTS = {'dar': False, 'ctar_theta': False, 'ctar_A': False}


def set_recorder_components(dar=False, ctar_theta=False, ctar_A=False):
    """Mirror the component calls that modeling_llada.generate makes. Must be set
    whenever attention is collected under a component, or the collection silently
    degrades to the plain cache."""
    _COMPONENTS.update(dar=bool(dar), ctar_theta=bool(ctar_theta), ctar_A=bool(ctar_A))
    if ctar_A:
        from llava.hooks import cache_hook_LLaDA_V as _ch_p
        _ch_p.set_publish_A(True)


def attach_attention_recorder(model, npz_dir, save_last_q=None, save_last_k=None):
    """Begin saving per-step attentions to npz_dir/step_{i}.npz.

    Patches generate_with_embeds at the CLASS level on the class that actually
    defines it (LLaDAModelLM), because LlavaLLaDAModelLM.generate() invokes
    super().generate_with_embeds(...) which bypasses instance attributes and
    routes directly to the class method.

    Idempotent: re-attaching only updates npz_dir.
    """
    os.makedirs(npz_dir, exist_ok=True)
    if hasattr(model, _RECORDER_STATE_ATTR):
        getattr(model, _RECORDER_STATE_ATTR)['npz_dir'] = npz_dir
        return
    # Walk MRO to find the class that actually defines generate_with_embeds.
    target_cls = None
    for cls in type(model).__mro__:
        if 'generate_with_embeds' in cls.__dict__:
            target_cls = cls
            break
    if target_cls is None:
        raise RuntimeError(
            'attach_attention_recorder: generate_with_embeds not found in MRO'
        )

    # Detect dLLM-Cache: if each layer's self_attn already has
    # attention_forward_for_cache (registered by register_cache_LLaDA_V), wrap it
    # so we can publish (layer_idx, q_index) into a thread-local during the call.
    cache_mode = False
    cache_wrap_restore = []   # list of (self_attn, original_bound_method) to restore at detach
    free_hook_handles = []    # list of hook handles installed on decoder layers
    inner = getattr(model, 'model', None)
    if inner is not None and hasattr(inner, 'layers'):
        for layer_idx, tf_block in enumerate(inner.layers):
            self_attn = tf_block.self_attn
            if hasattr(self_attn, 'attention_forward_for_cache'):
                cache_mode = True
                original_method = self_attn.attention_forward_for_cache
                wrapped = _make_cache_attn_wrapper(original_method, layer_idx)
                self_attn.attention_forward_for_cache = types.MethodType(wrapped, self_attn)
                cache_wrap_restore.append((self_attn, original_method))
        if not cache_mode:
            # Vanilla path: each layer returns (hidden, attn_weights, ...). Free attn slot
            # post-forward so all_self_attns does not accumulate ~55GB across 32 layers.
            for tf_block in inner.layers:
                free_hook_handles.append(tf_block.register_forward_hook(_free_layer_attn_hook))

    state = {
        'target_cls': target_cls,
        'original_method': target_cls.generate_with_embeds,
        'npz_dir': npz_dir,
        'save_last_q': save_last_q,
        'save_last_k': save_last_k,
        'cache_mode': cache_mode,
        'cache_wrap_restore': cache_wrap_restore,
        'free_hook_handles': free_hook_handles,
    }
    setattr(model, _RECORDER_STATE_ATTR, state)
    target_cls.generate_with_embeds = _instrumented_generate_with_embeds
    print(
        f'[attention_recorder] patched {target_cls.__name__}.generate_with_embeds '
        f'-> NPZ dir: {npz_dir}  '
        f'(cache_mode={cache_mode}, '
        f'save_last_q={save_last_q}, save_last_k={save_last_k})'
    )


def detach_attention_recorder(model):
    """Restore the original generate_with_embeds and per-layer cache wrappers."""
    if not hasattr(model, _RECORDER_STATE_ATTR):
        return
    state = getattr(model, _RECORDER_STATE_ATTR)
    for self_attn, original_method in state.get('cache_wrap_restore', []):
        self_attn.attention_forward_for_cache = original_method
    for h in state.get('free_hook_handles', []):
        h.remove()
    state['target_cls'].generate_with_embeds = state['original_method']
    delattr(model, _RECORDER_STATE_ATTR)
    print(f'[attention_recorder] restored {state["target_cls"].__name__}.generate_with_embeds')


def render_attention_maps(
    npz_root, map_root,
    vis_script='/data/zhaoqiyan/autodl-tmp/LLaDA-V/vis-scripts/vis_attention.py',
    last_q=64, last_k=64, every_layer=False, max_steps=0,
    tokenizer='GSAI-ML/LLaDA-V',
    only_stems=None,
):
    """Run vis-scripts/vis_attention.py as a subprocess to render heatmaps.

    only_stems: iterable of subdir names under npz_root to restrict rendering to.
                If None, vis_attention.py iterates all subdirs.
    """
    cmd = [
        sys.executable, vis_script,
        '--npz-root', npz_root,
        '--map-root', map_root,
        '--tokenizer', tokenizer,
        '--last-q', str(last_q),
        '--last-k', str(last_k),
        '--max-steps', str(max_steps),
    ]
    if every_layer:
        cmd.append('--every-layer')
    if only_stems:
        cmd.append('--only-stems')
        cmd.extend(list(only_stems))
    print(f'[attention_recorder] rendering: {" ".join(cmd)}')
    return subprocess.run(cmd, check=True)


def render_step_level_attention_maps(
    npz_root, map_root,
    vis_script='/data/zhaoqiyan/autodl-tmp/LLaDA-V/vis-scripts/vis_step_attention.py',
    tokenizer='GSAI-ML/LLaDA-V',
    only_stems=None,
):
    """Step-level companion to render_attention_maps.

    For each subdir of `npz_root`, runs vis_step_attention.py to produce per-layer
    [steps x suffix-position] heatmaps under `map_root/<stem>/layer{N}.png`.
    """
    cmd = [
        sys.executable, vis_script,
        '--npz-root', npz_root,
        '--map-root', map_root,
        '--tokenizer', tokenizer,
    ]
    if only_stems:
        cmd.append('--only-stems')
        cmd.extend(list(only_stems))
    print(f'[attention_recorder] step-level rendering: {" ".join(cmd)}')
    return subprocess.run(cmd, check=True)
