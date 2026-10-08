"""Self-contained SlowFast stack for MMaDA (composed with SF-evolved cache).

One file, three parts, zero edits to any repo:
1. SFFeatureCache  - evolved dLLM cache (expect_length: tolerates the sampler
   truncated forwards; gen-side set scatters into full buffer, get slices).
2. register_sf_cache_MMaDA / logout - clone of maomaocun cache_hook_MMaDA.py
   (signature-compatible with MMadaModelLM block calls, incl. block_mask),
   with the cache singleton swapped to SFFeatureCache and expect_length set
   per forward.
3. SlowFastSampler - verbatim clone of Slow-Fast-Sampling sampler.py with the
   FeatureCache import redirected to SFFeatureCache.
"""
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import Optional, Tuple
from collections import defaultdict


class _Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SFFeatureCache(metaclass=_Singleton):
    @classmethod
    def new_instance(cls, prompt_interval_steps=1, gen_interval_steps=1,
                     cfg_interval_steps=1, transfer_ratio=0.0):
        ins = cls()
        ins.prompt_interval_steps = prompt_interval_steps
        ins.gen_interval_steps = gen_interval_steps
        ins.cfg_interval_steps = cfg_interval_steps
        ins.transfer_ratio = transfer_ratio
        ins.init()
        return ins

    def init(self):
        self.__cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.__step_counter = defaultdict(lambda: defaultdict(lambda: 0))

    def reset_cache(self, prompt_length=0, gen_length=0):
        self.init()
        torch.cuda.empty_cache()
        self.prompt_length = prompt_length
        self.gen_length = gen_length
        self.expect_length = gen_length
        self.cache_type = "no_cfg"

    def set_cache(self, layer_id, feature_name, features, cache_type):
        if cache_type == "gen" and self.gen_length != self.expect_length:
            tmp = self.__cache[self.cache_type][cache_type][layer_id]["attn"][0]
            b, t, d = tmp.shape
            index = torch.arange(self.expect_length, device=tmp.device).unsqueeze(0).unsqueeze(-1).expand(b, -1, d)
            if feature_name == "kv_cache":
                self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["k"].scatter_(dim=1, index=index, src=features["k"].clone())
                self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["v"].scatter_(dim=1, index=index, src=features["v"].clone())
            else:
                self.__cache[self.cache_type][cache_type][layer_id][feature_name][0].scatter_(dim=1, index=index, src=features.clone())
        else:
            self.__cache[self.cache_type][cache_type][layer_id][feature_name] = {0: features}

    def get_cache(self, layer_id, feature_name, cache_type):
        if cache_type == "gen" and self.gen_length != self.expect_length:
            if feature_name == "kv_cache":
                return {
                    "k": self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["k"].clone()[:, :self.expect_length, :],
                    "v": self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["v"].clone()[:, :self.expect_length, :],
                }
            return self.__cache[self.cache_type][cache_type][layer_id][feature_name][0][:, :self.expect_length, :]
        return self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]

    def update_step(self, layer_id):
        self.__step_counter[self.cache_type][layer_id] += 1

    def refresh_gen(self, layer_id=0):
        return (self.current_step - 1) % self.gen_interval_steps == 0

    def refresh_prompt(self, layer_id=0):
        return (self.current_step - 1) % self.prompt_interval_steps == 0

    def refresh_cfg(self, layer_id=0):
        return (self.current_step - 1) % self.cfg_interval_steps == 0 or self.current_step <= 5

    @property
    def current_step(self):
        return max(list(self.__step_counter[self.cache_type].values()), default=1)


# ============ Part 2: MMaDA block hook (clone of cache_hook_MMaDA.py) ============
def logout_sf_cache_MMaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
    """Restore original functions for transformer blocks, attention, and rotary embeddings."""
    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
            break
    if target_module is None:
        return
    for tf_block in target_module:
        if hasattr(tf_block, "_old_forward"):
            tf_block.forward = tf_block._old_forward
            delattr(tf_block, "_old_forward")
        if hasattr(tf_block, "_old_attention"):
            tf_block.attention = tf_block._old_attention
            delattr(tf_block, "_old_attention")
        if hasattr(tf_block.rotary_emb, "_old_forward"):
            tf_block.rotary_emb.forward = tf_block.rotary_emb._old_forward
            delattr(tf_block.rotary_emb, "_old_forward")
            

def register_sf_cache_MMaDA(model: nn.Module, tf_block_module_key_name: str) -> None:

    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
    for tf_block in target_module:
        setattr(tf_block, "_old_forward", tf_block.forward)
        tf_block.forward = types.MethodType(cache_hook_feature, tf_block)
        setattr(tf_block, "_old_attention", tf_block.attention)
        tf_block.attention = types.MethodType(_attention, tf_block)
        setattr(tf_block.rotary_emb, "_old_forward", tf_block.rotary_emb.forward)
        tf_block.rotary_emb.forward = types.MethodType(
            RoPe_forward, tf_block.rotary_emb
)


def _attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    q_index: torch.Tensor = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

    B, q_len, C = q.size()
    B, k_len, C = k.size()
    B, v_len, C = v.size()
    dtype = k.dtype
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)
    q = q.view(B, q_len, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    k = k.view(
        B, k_len, self.config.effective_n_kv_heads, C // self.config.n_heads
    ).transpose(1, 2)
    v = v.view(
        B, v_len, self.config.effective_n_kv_heads, C // self.config.n_heads
    ).transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)
    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]
    if self.config.rope:
        q, k = self.rotary_emb(q, k, q_index=q_index)
    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
        )
    att = self._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    att = att.transpose(1, 2).contiguous().view(B, q_len, C)
    return self.attn_out(att), present


def RoPe_forward(
    self, q: torch.Tensor, k: torch.Tensor, q_index: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.config.rope_full_precision:
        q_, k_ = q.float(), k.float()
    else:
        q_, k_ = q, k
    with torch.autocast(q.device.type, enabled=False):
        query_len, key_len = q_.shape[-2], k_.shape[-2]
        pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
        pos_sin = pos_sin.type_as(q_)
        pos_cos = pos_cos.type_as(q_)
        if q_index is not None:
            bs, _ = q_index.shape
            q_list = []
            for i in range(bs):
                q_i = self.apply_rotary_pos_emb(
                    pos_sin[:, :, q_index[i], :],
                    pos_cos[:, :, q_index[i], :],
                    q_[i].unsqueeze(0),
                )
                q_list.append(q_i)
            q_ = torch.cat(q_list, dim=0)
        else:
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
        k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
    return q_.type_as(q), k_.type_as(k)


def refresh_index(
    new_features: torch.Tensor,
    cached_features: torch.Tensor = None,
    transfer_ratio: float = 0.5,
    layer_id: int = 0,
) -> torch.Tensor:
    batch_size, gen_len, d_model = new_features.shape
    num_replace = int(gen_len * transfer_ratio)
    cos_sim = torch.nn.functional.cosine_similarity(
        new_features, cached_features, dim=-1
    )
    transfer_index = torch.topk(cos_sim, largest=False, k=num_replace).indices
    return transfer_index


def cache_hook_feature(
    self,
    x: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    block_mask=None
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    feature_cache = SFFeatureCache()
    feature_cache.update_step(self.layer_id)
    prompt_length = feature_cache.prompt_length
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]
    refresh_gen = feature_cache.refresh_gen(layer_id=self.layer_id)
    refresh_prompt = feature_cache.refresh_prompt(layer_id=self.layer_id)
    transfer_ratio = feature_cache.transfer_ratio
    bs, seq_len, dim = x.shape
    feature_cache.expect_length = seq_len - prompt_length  # SF delta: truncated-forward support
    transfer = transfer_ratio > 0 and transfer_ratio <= 1

    def attention(q, k, v, q_index: torch.Tensor = None):
        if self._activation_checkpoint_fn is not None:
            att, _ = self._activation_checkpoint_fn(
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                q_index=q_index,
            )
        else:
            att, _ = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                q_index=q_index,
            )
        return att

    def compute_mlp(input_x):

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, input_x)
        else:
            x = self.ff_norm(input_x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = x * x_up
        return self.ff_out(x)

    def project(x):
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        return q, k, v

    if refresh_gen and refresh_prompt:
        q, k, v = project(x)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k[:, :prompt_length, :], "v": v[:, :prompt_length, :]},
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k[:, prompt_length:, :], "v": v[:, prompt_length:, :]},
            cache_type="gen",
        )
        att = attention(q, k, v)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att[:, :prompt_length, :],
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att[:, prompt_length:, :],
            cache_type="gen",
        )

    elif refresh_gen and not refresh_prompt:
        q, k_gen, v_gen = project(x_gen)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k_gen, "v": v_gen},
            cache_type="gen",
        )
        kv_cache_prompt = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="kv_cache", cache_type="prompt"
        )
        k = torch.cat([kv_cache_prompt["k"], k_gen], dim=1)
        v = torch.cat([kv_cache_prompt["v"], v_gen], dim=1)
        att_gen = attention(q, k, v)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att_gen,
            cache_type="gen",
        )
        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="prompt"
        )
        att = torch.cat([att_prompt_cache, att_gen], dim=1)

    elif not refresh_gen and refresh_prompt:
        q_prompt, k_prompt, v_prompt = project(x_prompt)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k_prompt, "v": v_prompt},
            cache_type="prompt",
        )
        kv_cache_gen = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="kv_cache", cache_type="gen"
        )
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_normed = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_normed)
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.layer_id
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            kv_cache_gen["k"].scatter_(dim=1, index=index_expanded, src=k_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
        k = torch.cat([k_prompt, kv_cache_gen["k"]], dim=1)
        v = torch.cat([v_prompt, kv_cache_gen["v"]], dim=1)
        if transfer:
            q_prompt_gen_index = torch.cat([q_prompt, q_gen_index], dim=1)
            prompt_index = (
                torch.arange(prompt_length)
                .unsqueeze(0)
                .expand(bs, -1)
                .to(q_prompt_gen_index.device)
            )
            gen_index = index + prompt_length
            att_prompt_gen_index = attention(
                q_prompt_gen_index,
                k,
                v,
                q_index=torch.cat([prompt_index, gen_index], dim=1),
            )
            att_prompt = att_prompt_gen_index[:, :prompt_length, :]
            att_gen_index = att_prompt_gen_index[:, prompt_length:, :]
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )
        else:
            att_prompt = attention(
                q_prompt,
                k,
                v,
                q_index=torch.arange(prompt_length).unsqueeze(0).expand(bs, -1),
            )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att_prompt,
            cache_type="prompt",
        )
        att = torch.cat([att_prompt, att_gen_cache], dim=1)
    else:
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_normed = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_normed)
            kv_cache_gen = feature_cache.get_cache(
                layer_id=self.layer_id, feature_name="kv_cache", cache_type="gen"
            )
            kv_cache_prompt = feature_cache.get_cache(
                layer_id=self.layer_id, feature_name="kv_cache", cache_type="prompt"
            )
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.layer_id
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            kv_cache_gen["k"].scatter_(dim=1, index=index_expanded, src=k_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
            k = torch.cat([kv_cache_prompt["k"], kv_cache_gen["k"]], dim=1)
            v = torch.cat([kv_cache_prompt["v"], kv_cache_gen["v"]], dim=1)
            att_gen_index = attention(q_gen_index, k, v, q_index=index + prompt_length)
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )

        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="prompt"
        )
        att = torch.cat([att_prompt_cache, att_gen_cache], dim=1)

    x = x + self.dropout(att)

    og_x = x
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]

    if refresh_gen and refresh_prompt:
        x = compute_mlp(x)
        feature_cache.set_cache(
            self.layer_id, "mlp", x[:, prompt_length:, :], cache_type="gen"
        )
        feature_cache.set_cache(
            self.layer_id, "mlp", x[:, :prompt_length, :], cache_type="prompt"
        )

    elif refresh_gen and not refresh_prompt:
        x_gen = compute_mlp(x_gen)
        feature_cache.set_cache(self.layer_id, "mlp", x_gen, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(
            self.layer_id, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen], dim=1)

    elif refresh_prompt and not refresh_gen:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_prompt_gen_index = torch.cat([x_prompt, x_gen_selected], dim=1)
            x_prompt_gen_index = compute_mlp(x_prompt_gen_index)
            x_prompt = x_prompt_gen_index[:, :prompt_length, :]
            x_gen_index = x_prompt_gen_index[:, prompt_length:, :]
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(self.layer_id, "mlp", x_gen_cache, cache_type="gen")
        else:
            x_prompt = compute_mlp(x_prompt)
        feature_cache.set_cache(self.layer_id, "mlp", x_prompt, cache_type="prompt")
        x = torch.cat([x_prompt, x_gen_cache], dim=1)

    else:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_gen_index = compute_mlp(x_gen_selected)
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(self.layer_id, "mlp", x_gen_cache, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(
            self.layer_id, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen_cache], dim=1)

    x = self.dropout(x)
    x = og_x + x

    return x, None


# ============ Part 3: SlowFast sampler (clone of sampler.py) ============
class SlowFastSampler:
    def __init__(
        self,
        model,
        gen_kwargs,
        mask_id=126336,
        temperature=0.0,
        cfg_scale=0.0
    ):
        self.model = model
        self.mask_id = mask_id
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.k_exploration_steps = gen_kwargs.get("k_exploration_steps", 6)
        self.cycle_len_confidence_threshold = gen_kwargs.get("cycle_len_confidence_threshold", 0.3)
        self.cycle_length_stability_window = gen_kwargs.get("cycle_length_stability_window", 2)
        self.cycle_length_stability_std_dev_threshold = gen_kwargs.get("cycle_length_stability_std_dev_threshold", 1.0)
        self.high_confidence_threshold = gen_kwargs.get("high_confidence_threshold", 0.9)
        self.num_important_low_confidence_tokens = gen_kwargs.get("num_important_low_confidence_tokens", 3)
        self.max_sub_cycles_per_block = gen_kwargs.get("max_sub_cycles_per_block", 256)
        self.gen_length=gen_kwargs.get("gen_length", 128)
        self.block_length=gen_kwargs.get("block_length", 128)
    
    def add_gumbel_noise(self,logits):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if self.temperature == 0:
            return logits.exp()
        noise = torch.rand_like(logits)
        gumbel_noise = (-torch.log(noise)) ** self.temperature
        return logits.exp() / gumbel_noise
    def get_num_tokens_for_phase1_step(self,current_sub_cycle_mask):
        batch_size = current_sub_cycle_mask.shape[0]
        return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device) # try to fill 1 tokens

    def get_num_tokens_for_phase3_step(self,current_sub_cycle_mask):
        batch_size = current_sub_cycle_mask.shape[0]
        return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device) # try to fill 1 tokens

    def slow_phase(self, x, prompt_length, block_idx, last_sub_cycle_length_per_item, actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords, prompt_index_full_x, attention_mask):
        batch_size = x.shape[0]
        block_start_in_gen = block_idx * self.block_length
        block_end_in_gen = (block_idx + 1) * self.block_length
        sub_cycle_determined_per_item = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        history_per_item = [collections.deque(maxlen=self.cycle_length_stability_window) for _ in range(batch_size)]

        for k_step in range(self.k_exploration_steps):
            if self.cfg_scale > 0.0: 
                cfg_x = x.clone()
                cfg_x[prompt_index_full_x] = self.mask_id
                logits_main = self.model(x, attention_mask=attention_mask).logits
                cfg_logits_main = self.model(cfg_x, attention_mask=attention_mask).logits
                logits_full = logits_main + self.cfg_scale * (logits_main - cfg_logits_main)
            else:
                logits_full = self.model(x, attention_mask=attention_mask).logits
            
            logits_gen_part = logits_full[:, prompt_length:]
            x0_gen = torch.argmax(self.add_gumbel_noise(logits_gen_part), dim=-1)
            p_gen = F.softmax(logits_gen_part, dim=-1)
            x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)

            #Estimate sub-cycle length (focus on current block)
            current_global_mask_index_gen_part = (x[:, prompt_length:] == self.mask_id) 
            confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))
            for b_idx in range(batch_size):
                if not sub_cycle_determined_per_item[b_idx]:
                    previous_len_item = last_sub_cycle_length_per_item[b_idx].item()
                    observation_abs_start_in_gen = block_start_in_gen + previous_len_item
                    observation_abs_end_in_gen = block_end_in_gen 
                    increment_len = 0

                    if observation_abs_start_in_gen < observation_abs_end_in_gen:
                        confidence_in_observation_scope = confidence_gen_wide[b_idx, observation_abs_start_in_gen : observation_abs_end_in_gen]
                        if confidence_in_observation_scope.numel() > 0: 
                            above_thresh_indices_in_scope = (confidence_in_observation_scope >= self.cycle_len_confidence_threshold).nonzero(as_tuple=True)[0] 
                            if len(above_thresh_indices_in_scope) > 0:
                                farthest_idx_in_scope = above_thresh_indices_in_scope.max().item()
                                increment_len = farthest_idx_in_scope + 1
                            else:
                                increment_len = 1 
                        else:
                            pass 
                    else:
                        increment_len = 0

                    est_len = previous_len_item + increment_len
                    est_len = max(1, est_len) 
                    est_len = min(est_len, self.block_length) 
                    history_per_item[b_idx].append(est_len)

                    if len(history_per_item[b_idx]) >= self.cycle_length_stability_window:
                        hist_np = np.array(list(history_per_item[b_idx]))
                        if np.std(hist_np) < self.cycle_length_stability_std_dev_threshold:
                            det_len = int(history_per_item[b_idx][-1])
                            actual_sub_cycle_length_per_item[b_idx] = max(1, min(det_len, self.block_length))
                            sub_cycle_determined_per_item[b_idx] = True
                        else:
                            det_len = int(np.mean(hist_np))
                            actual_sub_cycle_length_per_item[b_idx] = max(1, min(det_len, self.block_length))
                            sub_cycle_determined_per_item[b_idx] = False if k_step < self.k_exploration_steps - 1 else True                 

            #Initial fill
            num_to_fill_p1 = self.get_num_tokens_for_phase1_step(mask_in_current_block_abs_coords) 
            transfer_mask_p1 = torch.zeros_like(x0_gen, dtype=torch.bool)
            for b_idx in range(batch_size):
                previous_len_item_fill = last_sub_cycle_length_per_item[b_idx].item()
                fill_op_abs_start_in_gen = block_start_in_gen + previous_len_item_fill
                fill_op_abs_end_in_gen = block_end_in_gen
                increment_len_p1_fill = 0 
                if fill_op_abs_start_in_gen < fill_op_abs_end_in_gen:
                    conf_in_fill_op_scope = confidence_gen_wide[b_idx, fill_op_abs_start_in_gen : fill_op_abs_end_in_gen]
                    mask_in_fill_op_scope = (x[b_idx, prompt_length + fill_op_abs_start_in_gen : prompt_length + fill_op_abs_end_in_gen] == self.mask_id)
                    if conf_in_fill_op_scope.numel() > 0: 
                        eff_conf_in_fill_op_scope = torch.where(mask_in_fill_op_scope, conf_in_fill_op_scope, torch.tensor(-np.inf, device=x.device, dtype=conf_in_fill_op_scope.dtype))
                        num_masked_in_fill_op_scope = mask_in_fill_op_scope.sum().item()
                        if num_to_fill_p1[b_idx] > 0 and num_masked_in_fill_op_scope > 0:
                            k = min(num_to_fill_p1[b_idx].item(), num_masked_in_fill_op_scope)
                            phase1_high_conf_fill_indices = (conf_in_fill_op_scope >= self.high_confidence_threshold) & mask_in_fill_op_scope
                            if phase1_high_conf_fill_indices.any() and phase1_high_conf_fill_indices.sum().item()>1:
                                abs_indices_to_fill = fill_op_abs_start_in_gen + phase1_high_conf_fill_indices.nonzero(as_tuple=True)[0]
                                transfer_mask_p1[b_idx, abs_indices_to_fill] = True
                            else:           
                                if k > 0:
                                    top_k_indices_relative_to_fill_scope = torch.topk(eff_conf_in_fill_op_scope, k=k).indices
                                    abs_indices_to_fill_in_gen = fill_op_abs_start_in_gen + top_k_indices_relative_to_fill_scope
                                    transfer_mask_p1[b_idx, abs_indices_to_fill_in_gen] = True
            x[:, prompt_length:][transfer_mask_p1] = x0_gen[transfer_mask_p1]

        # After k_exploration_steps, if any item's sub-cycle length is not determined, use a fallback.
        for b_idx in range(batch_size):
            if not sub_cycle_determined_per_item[b_idx]:
                if len(history_per_item[b_idx]) > 0: # Use average of what was gathered
                    actual_sub_cycle_length_per_item[b_idx] = max(1, min(int(np.mean(list(history_per_item[b_idx]))), self.block_length))
                else: # Absolute fallback
                    actual_sub_cycle_length_per_item[b_idx] = self.block_length // 2 # Or some other default
                sub_cycle_determined_per_item[b_idx] = True # Mark as determined for next phases
        return x
        
    def fast_phase(self, x, prompt_length, block_idx, last_sub_cycle_length_per_item, actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords, prompt_index_full_x, attention_mask):
        batch_size = x.shape[0]
        phase_2_and_3_calls = 0
        block_start_in_gen = block_idx * self.block_length
        block_end_in_gen = (block_idx + 1) * self.block_length
        # cache list
        cache_out_cycle_logits_list = []
        cache_out_cycle_cfg_logits_list = []
        cache_out_cycle_full_logits_list = []
        # cycle list
        active_region_start_check_list = []
        active_region_end_check_list = []
        while True:
            all_p2_active_regions_filled_for_all_items = True
            for b_idx_check in range(batch_size):
                current_cumulative_len_check = actual_sub_cycle_length_per_item[b_idx_check].item()
                previous_cumulative_len_check = last_sub_cycle_length_per_item[b_idx_check].item()
                active_region_start_check_list.append(block_start_in_gen + previous_cumulative_len_check)
                active_region_end_check_list.append(block_start_in_gen + current_cumulative_len_check)
                if active_region_start_check_list[b_idx_check] < active_region_end_check_list[b_idx_check]:
                    mask_in_ar_check = (x[b_idx_check, prompt_length + active_region_start_check_list[b_idx_check] : prompt_length + active_region_end_check_list[b_idx_check]] == self.mask_id)
                    if mask_in_ar_check.any(): # If any mask exists in this item's active region
                        all_p2_active_regions_filled_for_all_items = False
                        break # No need to check other items, we know P2 still has work
            if all_p2_active_regions_filled_for_all_items:
                break 
            
            phase_2_and_3_calls += 1
            
            # model call
            if self.cfg_scale > 0.0: # Simplified CFG
                if phase_2_and_3_calls == 1:
                    cfg_x = x.clone()
                    cfg_x[prompt_index_full_x] = self.mask_id
                    logits_main = self.model(x, attention_mask=attention_mask).logits 
                    cfg_logits_main = self.model(cfg_x, attention_mask=attention_mask).logits
                    for b_idx_check in range(batch_size):
                        cache_out_cycle_logits_list.append(logits_main[b_idx_check,  prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                        cache_out_cycle_cfg_logits_list.append(cfg_logits_main[b_idx_check,  prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                    logits_full = logits_main + self.cfg_scale * (logits_main - cfg_logits_main)
                else:
                    cfg_x = x.clone()
                    cfg_x[prompt_index_full_x] = self.mask_id
                    logits_main_batch = []
                    cfg_logits_main_batch = []
                    for b_idx_check in range(batch_size):
                        logits_main_part = self.model(x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask).logits
                        cfg_logits_main_part = self.model(cfg_x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask).logits
                        logits_main_batch.append(torch.cat([logits_main_part[b_idx_check].unsqueeze(0), cache_out_cycle_logits_list[b_idx_check]], dim=1))
                        cfg_logits_main_batch.append(torch.cat([cfg_logits_main_part[b_idx_check].unsqueeze(0), cache_out_cycle_cfg_logits_list[b_idx_check]],dim=1))
                    logits_main = torch.cat(logits_main_batch, dim=0)
                    cfg_logits_main = torch.cat(cfg_logits_main_batch, dim=0)
                    logits_full = logits_main + self.cfg_scale * (logits_main - cfg_logits_main)
            else:
                if phase_2_and_3_calls == 1:
                    # logits_full [1, 1149, 126464]
                    logits_full = self.model(x, attention_mask=attention_mask).logits
                    for b_idx_check in range(batch_size):
                        cache_out_cycle_full_logits_list.append(logits_full[b_idx_check, prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                else:
                    logits_full_batch = []
                    for b_idx_check in range(batch_size):
                        logits_full_item = self.model(x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask).logits
                        logits_full_batch.append(torch.cat([logits_full_item[b_idx_check].unsqueeze(0), cache_out_cycle_full_logits_list[b_idx_check]], dim=1))

                    logits_full = torch.cat(logits_full_batch, dim=0)
                
            
            logits_gen_part = logits_full[:, prompt_length:]
            x0_gen = torch.argmax(self.add_gumbel_noise(logits_gen_part), dim=-1)
            p_gen = F.softmax(logits_gen_part, dim=-1)
            x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)
            current_global_mask_index_gen_part = (x[:, prompt_length:] == self.mask_id)
            confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))
            transfer_mask_p2_and_p3 = torch.zeros_like(x0_gen, dtype=torch.bool)
            
            for b_idx in range(batch_size):
                sub_cycle_abs_end_in_gen = block_start_in_gen + actual_sub_cycle_length_per_item[b_idx].item()
                sub_cycle_abs_start_in_gen = block_start_in_gen + last_sub_cycle_length_per_item[b_idx].item() 
                

                conf_in_sub_cycle_scope = confidence_gen_wide[b_idx, sub_cycle_abs_start_in_gen:sub_cycle_abs_end_in_gen]
                mask_in_sub_cycle_scope = (x[b_idx, prompt_length + sub_cycle_abs_start_in_gen : prompt_length + sub_cycle_abs_end_in_gen] == self.mask_id)

                high_conf_fill_indices = (conf_in_sub_cycle_scope >= self.high_confidence_threshold) & mask_in_sub_cycle_scope

                # print(f"high_conf_fill_indices{high_conf_fill_indices}")
                
                if high_conf_fill_indices.any() and high_conf_fill_indices.sum().item()>1:
                    abs_indices_to_fill = sub_cycle_abs_start_in_gen + high_conf_fill_indices.nonzero(as_tuple=True)[0]
                    transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                else:
                    n2_num_transfer_tokens = self.get_num_tokens_for_phase3_step(mask_in_current_block_abs_coords)
                    eff_conf_sub_cycle = torch.where(mask_in_sub_cycle_scope, conf_in_sub_cycle_scope, torch.tensor(-np.inf, device=x.device, dtype=conf_in_sub_cycle_scope.dtype))
            
                    top_k_indices_relative_to_sub_cycle = torch.topk(eff_conf_sub_cycle, k=n2_num_transfer_tokens[b_idx].item()).indices
                    abs_indices_to_fill = sub_cycle_abs_start_in_gen + top_k_indices_relative_to_sub_cycle
                    transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                    
            x[:, prompt_length:][transfer_mask_p2_and_p3] = x0_gen[transfer_mask_p2_and_p3] # Update x
        return x
    def generate(self, input_ids, attention_mask):
        with torch.no_grad():
            batch_size, prompt_length = input_ids.shape
            x = torch.full(
                (batch_size, prompt_length + self.gen_length),
                self.mask_id, dtype=torch.long, device=self.model.device,
            )
            x[:, :prompt_length] = input_ids
            prompt_index_full_x = (x != self.mask_id)

            assert self.gen_length % self.block_length == 0
            num_blocks = self.gen_length // self.block_length

            feature_cache = SFFeatureCache()
            feature_cache.reset_cache(prompt_length,gen_length=self.gen_length)
            
            for block_idx in range(num_blocks):
                block_abs_start_in_x = prompt_length + block_idx * self.block_length
                block_abs_end_in_x = prompt_length + (block_idx + 1) * self.block_length

                current_sub_cycles_in_block = 0
                actual_sub_cycle_length_per_item = torch.full((batch_size,), self.block_length, dtype=torch.long, device=x.device)
                last_sub_cycle_length_per_item = torch.full((batch_size,), 0, dtype=torch.long, device=x.device)
                
                while True:
                    mask_in_current_block_abs_coords = (x[:, block_abs_start_in_x:block_abs_end_in_x] == self.mask_id)
                    if not mask_in_current_block_abs_coords.any():
                        break
                    if current_sub_cycles_in_block >= self.max_sub_cycles_per_block:
                        break
                    
                    current_sub_cycles_in_block += 1
                   
                    x = self.slow_phase(
                        x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                        actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                        prompt_index_full_x, attention_mask)

                    x = self.fast_phase(
                        x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                        actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                        prompt_index_full_x, attention_mask)

                    last_sub_cycle_length_per_item = actual_sub_cycle_length_per_item.clone()

            return x[:, prompt_length:]