import torch
from typing import Optional, Tuple, List
import torch.nn as nn
import types
from llava.cache import dLLMCache
import math

def _dev(x: torch.Tensor) -> torch.device:
    return x.device


def _align_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.device != ref.device or x.dtype != ref.dtype:
        x = x.to(device=ref.device, dtype=ref.dtype)
    return x


def _to_device(x: Optional[torch.Tensor], device: torch.device, dtype: Optional[torch.dtype] = None) -> Optional[
    torch.Tensor]:
    if x is None:
        return None
    if dtype is None:
        return x.to(device)
    return x.to(device=device, dtype=dtype)


def _ensure_same_device(tensors: List[Optional[torch.Tensor]], device: torch.device,
                        dtype: Optional[torch.dtype] = None):
    out = []
    for t in tensors:
        out.append(_to_device(t, device, dtype) if t is not None else None)
    return out


def _empty_on(shape, device, dtype):
    return torch.empty(shape, device=device, dtype=dtype)


def _zeros_on(shape, device, dtype):
    return torch.zeros(shape, device=device, dtype=dtype)


def _empty_like_on(x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype] = None):
    return torch.empty_like(x, device=device, dtype=(dtype or x.dtype))


def _gather_on(input_: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    index = _to_device(index, _dev(input_))
    return torch.gather(input_, dim=1, index=index)

# Helper functions from the new LLADa model (need to be accessible)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def register_cache_LLaDA_V(model: nn.Module, tf_block_module_key_name: str) -> None:
    """
    Registers cache hooks for a LLaDA-like model.
    tf_block_module_key_name is typically 'model.layers' for LLaMA-style models.
    """
    target_module_path = tf_block_module_key_name.split('.')
    current_module = model
    for part in target_module_path:
        current_module = getattr(current_module, part)

    target_module: Optional[nn.ModuleList] = current_module
    if target_module is None or not isinstance(target_module, nn.ModuleList):
        raise ValueError(f"Could not find nn.ModuleList at {tf_block_module_key_name}")

    for layer_index, tf_block in enumerate(target_module):
        setattr(tf_block, "layer_idx", layer_index)

        setattr(tf_block, "_old_forward", tf_block.forward)
        tf_block.forward = types.MethodType(llada_cache_hook_feature, tf_block)

        setattr(tf_block.self_attn, "_old_forward_main", tf_block.self_attn.forward)
        tf_block.self_attn.attention_forward_for_cache = types.MethodType(
            llada_attention_hook_for_cache, tf_block.self_attn
        )

        setattr(tf_block.self_attn.rotary_emb, "_old_forward", tf_block.self_attn.rotary_emb.forward)
        tf_block.self_attn.rotary_emb.forward = types.MethodType(
            llada_RoPe_forward_hook, tf_block.self_attn.rotary_emb
        )


def llada_attention_hook_for_cache(
        self,  # self is LLaDAAttention instance
        q_in_proj: torch.Tensor,  # Renamed from q to clarify it's post-projection from cache_hook
        k_in_proj: torch.Tensor,  # Renamed from k
        v_in_proj: torch.Tensor,  # Renamed from v
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        q_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    device = _dev(q_in_proj)
    q_in_proj, k_in_proj, v_in_proj, attention_bias, q_index = _ensure_same_device(
        [q_in_proj, k_in_proj, v_in_proj, attention_bias, q_index], device
    )
    # q_in_proj, k_in_proj, v_in_proj are (Batch, SeqLen_specific_to_them, HiddenDim_model)
    B, q_len_current_q = q_in_proj.shape[0], q_in_proj.shape[
        1]  # q_len_current_q is the length of the Q for this specific call

    q_num_heads = self.num_heads
    k_num_heads = self.num_key_value_heads
    v_num_heads = self.num_key_value_heads
    head_dim = self.head_dim

    # Reshape q, k, v to (Batch, NumHeads, SeqLen, HeadDim)
    q = q_in_proj.view(B, q_len_current_q, q_num_heads, head_dim).transpose(1, 2)

    k_seq_len_current_k = k_in_proj.shape[
        1]  # k_seq_len_current_k is the length of K for this call (e.g., full context)
    v_seq_len_current_v = v_in_proj.shape[1]

    k = k_in_proj.view(B, k_seq_len_current_k, k_num_heads, head_dim).transpose(1, 2)
    v = v_in_proj.view(B, v_seq_len_current_v, v_num_heads, head_dim).transpose(1, 2)

    if hasattr(self, 'rotary_emb'):
        q_index = _to_device(q_index, device) if q_index is not None else None
        q, k = self.rotary_emb(q, k, q_index=q_index)

    present = None

    k_repeated = repeat_kv(k, self.num_key_value_groups)
    v_repeated = repeat_kv(v, self.num_key_value_groups)

    # q is (B, q_num_heads, q_len_current_q, head_dim)
    # k_repeated is (B, q_num_heads, k_seq_len_current_k, head_dim)

    attn_weights = torch.matmul(q, k_repeated.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_bias is not None:
        attention_bias = _to_device(attention_bias, device, attn_weights.dtype)
        bias_q_dim = attention_bias.shape[-2]
        bias_k_dim = attention_bias.shape[-1]

        sliced_attention_bias = attention_bias
        if q_len_current_q < bias_q_dim:  # Q is a segment, assume it's the latest tokens
            # This assumes the q segment is the last part of the full query sequence represented by the bias
            sliced_attention_bias = attention_bias[:, :, -q_len_current_q:, :]

        if k_seq_len_current_k < bias_k_dim:  # K is a segment (less likely for full KV cache but possible)
            # This assumes K is the latest part of the full key sequence in the bias
            sliced_attention_bias = sliced_attention_bias[:, :, :, -k_seq_len_current_k:]

        # Final check on dimensions before adding
        if attn_weights.shape[-2] == sliced_attention_bias.shape[-2] and \
                attn_weights.shape[-1] == sliced_attention_bias.shape[-1]:
            attn_weights = attn_weights + sliced_attention_bias
        else:
            if sliced_attention_bias.shape[1] == 1 and attn_weights.shape[1] == self.num_heads:  # Mask has 1 head dim
                attn_weights = attn_weights + sliced_attention_bias  # Broadcast over heads
            elif sliced_attention_bias.shape[1] == self.num_heads:  # Mask has same num heads
                attn_weights = attn_weights + sliced_attention_bias
            else:
                raise RuntimeError(
                    f"Attention bias shape {sliced_attention_bias.shape} incompatible with "
                    f"attn_weights shape {attn_weights.shape} after slicing."
                )

    B, H, S_q, S_k = attn_weights.shape
    # print("S_q", S_q)
    q_sub = 128 if S_q > 3000 else S_q
    k_sub = 128 if S_q > 3000 else S_q

    q_start = S_q - q_sub
    k_start = S_k - k_sub
    attn_block_logits = attn_weights[:, :, q_start:, k_start:]  # [B, H, q_sub, k_sub]

    max_val = attn_block_logits.max().item()
    min_val = attn_block_logits.min().item()
    mean_val = attn_block_logits.mean().item()
    median_val = attn_block_logits.median().item()
    # print(
    #     f"[attn_block_logits stats] "
    #     f"max={max_val:.4f}, min={min_val:.4f}, mean={mean_val:.4f}, median={median_val:.4f}"
    # )


    tau = 2.0
    min_gain = 0.75
    upper_thr = 5.0
    lower_thr = median_val


    q_idx = torch.arange(q_sub, device=attn_block_logits.device, dtype=attn_block_logits.dtype)
    k_idx = torch.arange(k_sub, device=attn_block_logits.device, dtype=attn_block_logits.dtype)
    # |i - j|
    dist = (q_idx[:, None] - k_idx[None, :]).abs()  # [q_sub, k_sub]
    gain = min_gain + (1.0 - min_gain) * torch.exp(- (dist / tau) ** 2)  # [q_sub, k_sub]
    gain = gain.unsqueeze(0).unsqueeze(0)
    # could #
    within = (attn_block_logits >= lower_thr) & (attn_block_logits <= upper_thr)

    scaled_block = attn_block_logits * gain
    attn_block_logits = torch.where(within, scaled_block, attn_block_logits)
    attn_weights[:, :, q_start:, k_start:] = attn_block_logits

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    att_output_heads = torch.matmul(attn_weights, v_repeated)

    # Reshape to (B, q_len_current_q, ModelHiddenDim)
    att_output_heads = att_output_heads.transpose(1, 2).contiguous().view(B, q_len_current_q, q_num_heads * head_dim)

    output = self.o_proj(att_output_heads)
    output = _to_device(output, device)
    return output, present, attn_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def llada_RoPe_forward_hook(
        self_rope,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        q_index: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = _dev(q_in)
    q_in = _to_device(q_in, device)
    k_in = _to_device(k_in, device)
    if q_index is not None:
        q_index = _to_device(q_index, device, torch.long)

    input_dtype = q_in.dtype
    q_, k_ = q_in.float(), k_in.float()

    bs, _, query_len_current, head_dim_calc = q_.shape
    _, _, key_len_current, _ = k_.shape

    max_pos_needed = key_len_current
    if q_index is not None:
        max_pos_needed = max(max_pos_needed, int(q_index.max().item()) + 1 if q_index.numel() > 0 else 0)

    max_pos_needed = max(max_pos_needed, query_len_current)

    if max_pos_needed == 0:
        return q_in, k_in

    dim = self_rope.dim
    inv_freq_to_use = self_rope.inv_freq.to(q_.device)
    t = torch.arange(max_pos_needed, device=q_.device, dtype=torch.float32)
    if hasattr(self_rope, 'scaling_factor'):
        t = t / self_rope.scaling_factor

    freqs = torch.outer(t, inv_freq_to_use.float())
    emb = torch.cat((freqs, freqs), dim=-1)

    pos_cos_table = emb.cos()
    pos_sin_table = emb.sin()

    if q_index is not None:
        actual_q_indices = q_index[:, :query_len_current]
        cos_q = pos_cos_table[actual_q_indices]
        sin_q = pos_sin_table[actual_q_indices]
        q_rotated = (q_ * cos_q.unsqueeze(1)) + (rotate_half(q_) * sin_q.unsqueeze(1))
    else:
        q_indices = torch.arange(query_len_current, device=q_.device)
        cos_q = pos_cos_table[q_indices].unsqueeze(0)
        sin_q = pos_sin_table[q_indices].unsqueeze(0)
        q_rotated = (q_ * cos_q.unsqueeze(1)) + (rotate_half(q_) * sin_q.unsqueeze(1))

    k_indices = torch.arange(key_len_current, device=q_.device)
    cos_k = pos_cos_table[k_indices].unsqueeze(0)
    sin_k = pos_sin_table[k_indices].unsqueeze(0)
    k_rotated = (k_ * cos_k.unsqueeze(1)) + (rotate_half(k_) * sin_k.unsqueeze(1))

    return q_rotated.type_as(q_in), k_rotated.type_as(k_in)


def refresh_index(
        new_features: torch.Tensor,
        cached_features: torch.Tensor = None,
        transfer_ratio: float = 0.5,
        layer_id: int = 0,
) -> torch.Tensor:
    device = _dev(new_features)
    if cached_features is not None:
        cached_features = _to_device(cached_features, device)

    batch_size, gen_len, d_model = new_features.shape
    num_replace = int(gen_len * transfer_ratio)
    if num_replace == 0 or gen_len == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=new_features.device)
    if cached_features is None or cached_features.shape[1] == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=new_features.device)

    cos_sim = torch.nn.functional.cosine_similarity(
        new_features, cached_features, dim=-1
    )
    k_actual = min(num_replace, cos_sim.shape[1])
    if k_actual == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=new_features.device)

    transfer_index = torch.topk(cos_sim, largest=False, k=k_actual).indices
    return transfer_index


def llada_cache_hook_feature(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # This is the original mask for the full layer input
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, ...]]]:
    device = _dev(hidden_states)
    dtype = hidden_states.dtype
    hidden_states = _to_device(hidden_states, device, dtype)
    attention_mask = _to_device(attention_mask, device)
    position_ids = _to_device(position_ids, device, torch.long) if position_ids is not None else None

    current_layer_idx = self.layer_idx
    feature_cache = dLLMCache()
    feature_cache.update_step(current_layer_idx)

    prompt_length = feature_cache.prompt_length
    # x_prompt and x_gen are sub-segments of hidden_states
    x_prompt = hidden_states[:, :prompt_length, :]
    x_gen = hidden_states[:, prompt_length:, :]

    refresh_gen = feature_cache.refresh_gen(layer_id=current_layer_idx)
    refresh_prompt = feature_cache.refresh_prompt(layer_id=current_layer_idx)
    transfer_ratio = feature_cache.transfer_ratio

    bs, seq_len, dim = hidden_states.shape  # seq_len is the length of hidden_states input to this layer
    transfer = transfer_ratio > 0 and transfer_ratio <= 1

    index_from_attn_transfer = None
    index_expanded_from_attn_transfer = None

    def project(x_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_normed = self.input_layernorm(x_input)
        q = self.self_attn.q_proj(x_normed)
        k = self.self_attn.k_proj(x_normed)
        v = self.self_attn.v_proj(x_normed)
        return q, k, v

    # This function needs to be smarter about the attention_bias it passes.
    # q_tensor_origin_slice: A tuple (start_idx, length) indicating where q_tensor comes from
    # relative to the original hidden_states/attention_mask.
    def call_attention_on_qkv(q_tensor, k_tensor, v_tensor,
                              original_full_attention_bias,
                              q_tensor_start_idx: int,  # Start index of q_tensor within the original sequence
                              q_index: Optional[torch.Tensor] = None):

        q_tensor = _to_device(q_tensor, device)
        k_tensor = _to_device(k_tensor, device)
        v_tensor = _to_device(v_tensor, device)
        q_index = _to_device(q_index, device, torch.long) if q_index is not None else None

        q_len_for_this_call = q_tensor.shape[1]
        k_len_for_this_call = k_tensor.shape[1]  # This K is already the full K from dLLM cache

        # Slice the original_full_attention_bias
        # original_full_attention_bias is likely (B, 1 or H, S_full_layer_input, K_full_from_dllm_or_max)
        # We need (B, 1 or H, q_len_for_this_call, k_len_for_this_call)
        sliced_bias = original_full_attention_bias
        if original_full_attention_bias is not None:
            # Slice query dimension based on q_tensor_start_idx and its length
            # Slice key dimension to match k_tensor's length (which is k_full from dLLM)
            original_full_attention_bias = _to_device(original_full_attention_bias, device)
            sliced_bias = original_full_attention_bias[
                          :, :, q_tensor_start_idx: q_tensor_start_idx + q_len_for_this_call, :k_len_for_this_call
                          ]
            # Ensure num_heads dim is compatible (1 for broadcast, or matches self.self_attn.num_heads)
            if sliced_bias.shape[1] != 1 and sliced_bias.shape[1] != self.self_attn.num_heads:
                # This might indicate an issue with mask preparation upstream if it's not 1 or num_heads
                # For now, assume it's (B,1,q,k) and will broadcast if num_heads > 1 in attention_hook
                pass

        att_output, _, attn_weights = self.self_attn.attention_forward_for_cache(
            q_tensor,
            k_tensor,
            v_tensor,
            attention_bias=sliced_bias,
            layer_past=None,
            use_cache=False,
            q_index=q_index,
        )
        return att_output, attn_weights

    def compute_mlp(input_to_mlp: torch.Tensor) -> torch.Tensor:
        if input_to_mlp.shape[1] == 0:
            return torch.empty_like(input_to_mlp)
        x_norm = self.post_attention_layernorm(input_to_mlp)
        gate_proj_out = self.mlp.gate_proj(x_norm)
        up_proj_out = self.mlp.up_proj(x_norm)
        act_out = self.mlp.act_fn(gate_proj_out)
        x = act_out * up_proj_out
        return self.mlp.down_proj(x)

    residual_pre_attn = hidden_states
    all_attentions = []

    if refresh_gen and refresh_prompt:
        q_full, k_full, v_full = project(hidden_states)
        feature_cache.set_cache(
            layer_id=current_layer_idx, feature_name="kv_cache",
            features={"k": k_full[:, :prompt_length, :], "v": v_full[:, :prompt_length, :]}, cache_type="prompt"
        )
        if hidden_states.shape[1] > prompt_length:
            feature_cache.set_cache(
                layer_id=current_layer_idx, feature_name="kv_cache",
                features={"k": k_full[:, prompt_length:, :], "v": v_full[:, prompt_length:, :]}, cache_type="gen"
            )

        # Q is all of hidden_states, K,V also from all of hidden_states
        # q_start_idx is 0 because q_full corresponds to the start of hidden_states
        att, attn_weights = call_attention_on_qkv(q_full, k_full, v_full, attention_mask, q_tensor_start_idx=0,
                                                  q_index=position_ids)
        all_attentions.append(attn_weights)
        feature_cache.set_cache(
            layer_id=current_layer_idx, feature_name="attn",
            features=att[:, :prompt_length, :], cache_type="prompt"
        )
        if hidden_states.shape[1] > prompt_length:
            feature_cache.set_cache(
                layer_id=current_layer_idx, feature_name="attn",
                features=att[:, prompt_length:, :], cache_type="gen"
            )

    elif refresh_gen and not refresh_prompt:
        att_gen_part = torch.empty((bs, 0, dim), device=hidden_states.device)
        if x_gen.shape[1] > 0:
            q_gen, k_gen, v_gen = project(x_gen)
            feature_cache.set_cache(
                layer_id=current_layer_idx, feature_name="kv_cache",
                features={"k": k_gen, "v": v_gen}, cache_type="gen"
            )
            kv_cache_prompt = feature_cache.get_cache(
                layer_id=current_layer_idx, feature_name="kv_cache", cache_type="prompt"
            )

            k_prompt_val = kv_cache_prompt.get("k", _empty_on((bs, 0, dim), device, dtype))
            v_prompt_val = kv_cache_prompt.get("v", _empty_on((bs, 0, dim), device, dtype))
            k_prompt_val = _to_device(k_prompt_val, device, dtype)
            v_prompt_val = _to_device(v_prompt_val, device, dtype)


            k_gen = _to_device(k_gen, device)
            v_gen = _to_device(v_gen, device)
            k_full_ctx = torch.cat([k_prompt_val, k_gen], dim=1)
            v_full_ctx = torch.cat([v_prompt_val, v_gen], dim=1)

            q_gen_pos_ids = position_ids[:, prompt_length:] if position_ids is not None and position_ids.shape[
                1] > prompt_length else None


            att_gen_part, attn_weights = call_attention_on_qkv(q_gen, k_full_ctx, v_full_ctx, attention_mask,
                                                               q_tensor_start_idx=prompt_length, q_index=q_gen_pos_ids)
            all_attentions.append(attn_weights)
            feature_cache.set_cache(
                layer_id=current_layer_idx, feature_name="attn",
                features=att_gen_part, cache_type="gen"
            )

        att_prompt_cache = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="attn", cache_type="prompt"
        )
        att_prompt_cache = _to_device(att_prompt_cache, device, dtype)
        att = torch.cat([att_prompt_cache, att_gen_part], dim=1)

    elif not refresh_gen and refresh_prompt:
        q_prompt, k_prompt, v_prompt = project(x_prompt)
        feature_cache.set_cache(
            layer_id=current_layer_idx, feature_name="kv_cache",
            features={"k": k_prompt, "v": v_prompt}, cache_type="prompt"
        )
        kv_cache_gen = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="kv_cache", cache_type="gen"
        )
        att_gen_cache = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="attn", cache_type="gen"
        )

        k_gen_current = kv_cache_gen.get("k", torch.empty(bs, 0, dim, device=hidden_states.device))
        v_gen_current = kv_cache_gen.get("v", torch.empty(bs, 0, dim, device=hidden_states.device))

        att_gen_cache = _to_device(att_gen_cache, device, dtype)
        k_gen_current = _to_device(k_gen_current, device, dtype)
        v_gen_current = _to_device(v_gen_current, device, dtype)

        q_for_attn_segments = [q_prompt]
        q_idx_for_attn_segments = [position_ids[:, :prompt_length]] if position_ids is not None else [None]
        q_start_indices_segments = [0]  # q_prompt starts at index 0 of hidden_states

        if transfer and x_gen.shape[1] > 0 and k_gen_current.shape[1] > 0:
            _, _, v_gen_for_transfer = project(x_gen)
            index_from_attn_transfer = refresh_index(v_gen_for_transfer, v_gen_current, transfer_ratio,
                                                     current_layer_idx)

            if index_from_attn_transfer.numel() > 0:
                index_from_attn_transfer = _to_device(index_from_attn_transfer, device, torch.long)
                index_expanded_from_attn_transfer = index_from_attn_transfer.unsqueeze(-1).expand(-1, -1, dim)

                index_expanded_from_attn_transfer = _to_device(index_expanded_from_attn_transfer, device)
                x_gen_l = _to_device(self.input_layernorm(x_gen), device)

                x_gen_normed_selected = torch.gather(x_gen_l, dim=1, index=index_expanded_from_attn_transfer)
                q_gen_index = self.self_attn.q_proj(x_gen_normed_selected)
                k_gen_index = self.self_attn.k_proj(x_gen_normed_selected)
                v_gen_index_part = self.self_attn.v_proj(x_gen_normed_selected)
                k_gen_index = _to_device(k_gen_index, device)
                v_gen_index_part = _to_device(v_gen_index_part, device)
                k_gen_current = k_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer, src=k_gen_index)
                v_gen_current = v_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                      src=v_gen_index_part)

                feature_cache.set_cache(
                    layer_id=current_layer_idx, feature_name="kv_cache",
                    features={"k": k_gen_current, "v": v_gen_current}, cache_type="gen"
                )

                q_for_attn_segments.append(q_gen_index)
                if position_ids is not None and position_ids.shape[1] > prompt_length:
                    gen_abs_positions_all = position_ids[:, prompt_length:]
                    gen_abs_positions_all = gen_abs_positions_all.to(index_from_attn_transfer.device)
                    gen_abs_positions_selected = torch.gather(gen_abs_positions_all, 1, index_from_attn_transfer)
                    q_idx_for_attn_segments.append(gen_abs_positions_selected)
                else:
                    q_idx_for_attn_segments.append(None)

        q_combined_for_attn = torch.cat(q_for_attn_segments, dim=1)
        q_idx_combined_for_rope = torch.cat(q_idx_for_attn_segments, dim=1) if all(
            s is not None for s in q_idx_for_attn_segments) else None

        if q_idx_combined_for_rope is not None:
            q_idx_combined_for_rope = _to_device(q_idx_combined_for_rope, device, torch.long)
        k_gen_current = _to_device(k_gen_current, device)
        v_gen_current = _to_device(v_gen_current, device)
        k_prompt = _to_device(k_prompt, device)
        v_prompt = _to_device(v_prompt, device)
        k_full_ctx = torch.cat([k_prompt, k_gen_current], dim=1)
        v_full_ctx = torch.cat([v_prompt, v_gen_current], dim=1)

        # q_combined_for_attn effectively starts at index 0 of a conceptual sequence.
        # Its RoPE is handled by q_idx_combined_for_rope.
        att_for_q_combined, attn_weights = call_attention_on_qkv(q_combined_for_attn, k_full_ctx, v_full_ctx,
                                                                 attention_mask, q_tensor_start_idx=0,
                                                                 # Since Q is combined and starts from effective 0
                                                                 q_index=q_idx_combined_for_rope)
        all_attentions.append(attn_weights)

        att_prompt_new = att_for_q_combined[:, :q_prompt.shape[1], :]
        if transfer and index_from_attn_transfer is not None and index_from_attn_transfer.numel() > 0:
            att_gen_index_new = att_for_q_combined[:, q_prompt.shape[1]:, :]  # Segment for transferred Qs
            if att_gen_cache.shape[1] > 0:
                att_gen_cache = att_gen_cache.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                      src=att_gen_index_new)
                feature_cache.set_cache(
                    layer_id=current_layer_idx, feature_name="attn",
                    features=att_gen_cache, cache_type="gen"
                )

        feature_cache.set_cache(
            layer_id=current_layer_idx, feature_name="attn",
            features=att_prompt_new, cache_type="prompt"
        )
        att = torch.cat([att_prompt_new, att_gen_cache], dim=1)

    else:  # Not refresh gen, not refresh prompt
        att_prompt_cache = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="attn", cache_type="prompt"
        )
        att_gen_cache = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="attn", cache_type="gen"
        )
        kv_cache_gen = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="kv_cache", cache_type="gen"
        )
        kv_cache_prompt = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="kv_cache", cache_type="prompt"
        )

        att_prompt_cache = _to_device(att_prompt_cache, device, dtype)
        att_gen_cache = _to_device(att_gen_cache, device, dtype)

        k_gen_current = kv_cache_gen.get("k", torch.empty(bs, 0, dim, device=hidden_states.device))
        v_gen_current = kv_cache_gen.get("v", torch.empty(bs, 0, dim, device=hidden_states.device))
        k_prompt_val = kv_cache_prompt.get("k", torch.empty(bs, 0, dim, device=hidden_states.device))
        v_prompt_val = kv_cache_prompt.get("v", torch.empty(bs, 0, dim, device=hidden_states.device))

        k_gen_current = _to_device(k_gen_current, device, dtype)
        v_gen_current = _to_device(v_gen_current, device, dtype)
        k_prompt_val = _to_device(k_prompt_val, device, dtype)
        v_prompt_val = _to_device(v_prompt_val, device, dtype)

        if transfer and x_gen.shape[1] > 0 and k_gen_current.shape[1] > 0:
            x_gen_normed = self.input_layernorm(x_gen)
            v_gen_for_transfer = self.self_attn.v_proj(x_gen_normed)
            index_from_attn_transfer = refresh_index(v_gen_for_transfer, v_gen_current, transfer_ratio,
                                                     current_layer_idx)

            if index_from_attn_transfer.numel() > 0:
                index_from_attn_transfer = _to_device(index_from_attn_transfer, device, torch.long)
                index_expanded_from_attn_transfer = index_from_attn_transfer.unsqueeze(-1).expand(-1, -1, dim)
                x_gen_normed = _to_device(x_gen_normed, device)
                x_gen_normed_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded_from_attn_transfer)
                q_gen_index_only = self.self_attn.q_proj(x_gen_normed_selected)  # Q only for transferred items
                k_gen_index = self.self_attn.k_proj(x_gen_normed_selected)
                v_gen_index_part = self.self_attn.v_proj(x_gen_normed_selected)

                k_gen_index = _to_device(k_gen_index, device)
                k_gen_current = k_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer, src=k_gen_index)
                v_gen_index_part = _to_device(v_gen_index_part, device)
                v_gen_current = v_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                      src=v_gen_index_part)

                feature_cache.set_cache(
                    layer_id=current_layer_idx, feature_name="kv_cache",
                    features={"k": k_gen_current, "v": v_gen_current}, cache_type="gen"
                )

                q_idx_for_transferred_rope = None
                if position_ids is not None and position_ids.shape[1] > prompt_length:
                    gen_abs_positions_all = position_ids[:, prompt_length:]
                    q_idx_for_transferred_rope = torch.gather(gen_abs_positions_all, 1, index_from_attn_transfer)

                k_full_ctx = torch.cat([k_prompt_val, k_gen_current], dim=1)
                v_full_ctx = torch.cat([v_prompt_val, v_gen_current], dim=1)
                att_gen_index_new, attn_weights = call_attention_on_qkv(q_gen_index_only, k_full_ctx, v_full_ctx,
                                                                        attention_mask,
                                                                        q_tensor_start_idx=prompt_length,
                                                                        # Approximate for mask slicing
                                                                        q_index=q_idx_for_transferred_rope)
                all_attentions.append(attn_weights)

                if att_gen_cache.shape[1] > 0:  # Make sure att_gen_cache has a gen part
                    att_gen_cache = att_gen_cache.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                          src=att_gen_index_new)
                elif x_gen.shape[1] > 0:  # If original att_gen_cache was for an empty gen part, but x_gen is not empty
                    # Initialize att_gen_cache to be of x_gen's length before scattering
                    att_gen_cache = torch.zeros((bs, x_gen.shape[1], dim), device=hidden_states.device,
                                                dtype=att_gen_index_new.dtype)
                    att_gen_cache = att_gen_cache.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                          src=att_gen_index_new)
                # Else: if x_gen.shape[1] is 0, att_gen_cache remains empty, scatter won't happen.

                feature_cache.set_cache(
                    layer_id=current_layer_idx, feature_name="attn",
                    features=att_gen_cache, cache_type="gen"
                )

        att = torch.cat([att_prompt_cache, att_gen_cache], dim=1)
        att = _to_device(att, device)



    hidden_states_after_attn = residual_pre_attn + att
    residual_pre_mlp = hidden_states_after_attn

    x_prompt_mlp = hidden_states_after_attn[:, :prompt_length, :]
    x_gen_mlp = hidden_states_after_attn[:, prompt_length:, :]

    mlp_out_prompt_part = torch.empty((bs, prompt_length, dim), device=hidden_states.device,
                                      dtype=hidden_states_after_attn.dtype)
    mlp_out_gen_part = torch.empty((bs, x_gen_mlp.shape[1], dim), device=hidden_states.device,
                                   dtype=hidden_states_after_attn.dtype)

    if refresh_gen and refresh_prompt:
        mlp_out_full = compute_mlp(hidden_states_after_attn)
        mlp_out_prompt_part = mlp_out_full[:, :prompt_length, :]
        if x_gen_mlp.shape[1] > 0:
            mlp_out_gen_part = mlp_out_full[:, prompt_length:, :]

        feature_cache.set_cache(
            current_layer_idx, "mlp", mlp_out_prompt_part, cache_type="prompt"
        )
        if x_gen_mlp.shape[1] > 0:
            feature_cache.set_cache(
                current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen"
            )
        if mlp_out_gen_part.shape[1] > 0:
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part


    elif refresh_gen and not refresh_prompt:
        mlp_out_prompt_part = feature_cache.get_cache(
            current_layer_idx, "mlp", cache_type="prompt"
        )
        mlp_out_prompt_part = _to_device(mlp_out_prompt_part, device, dtype)
        if x_gen_mlp.shape[1] > 0:
            mlp_out_gen_part = compute_mlp(x_gen_mlp)
            feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen")

        if mlp_out_gen_part.shape[1] > 0:
            mlp_out_gen_part = _to_device(mlp_out_gen_part, device)
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part


    elif refresh_prompt and not refresh_gen:
        mlp_gen_cache_data = feature_cache.get_cache(current_layer_idx, "mlp", cache_type="gen")
        mlp_gen_cache_data = _to_device(mlp_gen_cache_data, device, dtype)
        if x_gen_mlp.shape[1] > 0:
            mlp_out_gen_part = mlp_gen_cache_data

        mlp_input_for_prompt_path = x_prompt_mlp
        # Use index_expanded_from_attn_transfer which was set in the attention block
        if transfer and index_expanded_from_attn_transfer is not None and index_expanded_from_attn_transfer.numel() > 0 and \
                x_gen_mlp.shape[1] > 0:
            x_gen_mlp_selected = torch.gather(x_gen_mlp, dim=1, index=index_expanded_from_attn_transfer)
            mlp_input_for_prompt_path = torch.cat([x_prompt_mlp, x_gen_mlp_selected], dim=1)

        mlp_out_prompt_path_processed = compute_mlp(mlp_input_for_prompt_path)
        mlp_out_prompt_part = mlp_out_prompt_path_processed[:, :x_prompt_mlp.shape[1], :]

        if transfer and index_expanded_from_attn_transfer is not None and index_expanded_from_attn_transfer.numel() > 0 and \
                x_gen_mlp.shape[1] > 0:
            mlp_gen_index_new = mlp_out_prompt_path_processed[:, x_prompt_mlp.shape[1]:, :]
            if mlp_out_gen_part.shape[1] > 0:  # If gen part exists
                mlp_gen_index_new = _to_device(mlp_gen_index_new, device)
                index_expanded_from_attn_transfer = _to_device(index_expanded_from_attn_transfer, device)
                mlp_out_gen_part = mlp_out_gen_part.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                            src=mlp_gen_index_new)

            feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen")

        feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_prompt_part, cache_type="prompt")
        if mlp_out_gen_part.shape[1] > 0:
            mlp_out_gen_part = _to_device(mlp_out_gen_part, device)
            mlp_out_prompt_part = _to_device(mlp_out_prompt_part, device)
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part


    else:
        mlp_out_prompt_part = feature_cache.get_cache(
            current_layer_idx, "mlp", cache_type="prompt"
        )
        mlp_gen_cache_data = feature_cache.get_cache(current_layer_idx, "mlp", cache_type="gen")
        mlp_out_prompt_part = _to_device(mlp_out_prompt_part, device, dtype)
        mlp_gen_cache_data = _to_device(mlp_gen_cache_data, device, dtype)
        if x_gen_mlp.shape[1] > 0:
            mlp_out_gen_part = mlp_gen_cache_data

        # Use index_expanded_from_attn_transfer
        if transfer and index_expanded_from_attn_transfer is not None and index_expanded_from_attn_transfer.numel() > 0 and \
                x_gen_mlp.shape[1] > 0:
            x_gen_mlp_selected = torch.gather(x_gen_mlp, dim=1, index=index_expanded_from_attn_transfer)
            mlp_out_gen_index = compute_mlp(x_gen_mlp_selected)
            if mlp_out_gen_part.shape[1] > 0:
                mlp_out_gen_index = _to_device(mlp_out_gen_index, device)
                mlp_out_gen_part = mlp_out_gen_part.scatter(dim=1, index=index_expanded_from_attn_transfer,
                                                            src=mlp_out_gen_index)

            feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen")

        if mlp_out_gen_part.shape[1] > 0:
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part

    mlp_out = _to_device(mlp_out, device)
    final_hidden_states = residual_pre_mlp + mlp_out
    returned_outputs = (final_hidden_states,)

    if output_attentions:
        returned_outputs += (all_attentions[0],)



    if use_cache:
        returned_outputs += (None,)
    return returned_outputs
