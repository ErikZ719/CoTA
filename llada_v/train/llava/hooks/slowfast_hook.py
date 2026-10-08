"""SlowFast Sampling ported to LLaDA-V (attach-style hook).

Faithful port of Slow-Fast-Sampling/slow_fast_sampling/sampler.py
(https://github.com/LiangrunFlora/Slow-Fast-Sampling), adapted in exactly one
way: LLaDA-V's multimodal prefix exists only as inputs_embeds, so every model
call embeds the current suffix token ids, concatenates them with the fixed
prefix embeds, and reads logits through lm_head. The sampler's own
frozen-tail-logits reuse (its Convergence-principle optimization), including
truncated forwards, is preserved verbatim. Standalone mode: no feature cache
is registered.

Restrictions of this port (matching our evaluation protocol):
  batch_size 1, cfg_scale 0.

Usage:
    from llava.hooks.slowfast_hook import register_slowfast_hook, unregister_slowfast_hook
    register_slowfast_hook(model, gen_length=64, block_length=64)
    out = model.generate(input_ids, images=..., image_sizes=..., ...)  # (1, gen_length) suffix ids
    unregister_slowfast_hook(model)

When not registered, no code path in the repo is touched.
"""
import collections
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_HOOK_ATTR = '_slowfast_hook_state'


class SlowFastLLaDAV:
    def __init__(self, model, gen_kwargs):
        self.model = model
        self.mask_id = gen_kwargs.get('mask_id', 126336)
        self.temperature = gen_kwargs.get('temperature', 0.0)
        self.cfg_scale = gen_kwargs.get('cfg_scale', 0.0)
        assert self.cfg_scale == 0.0, 'this port supports cfg_scale=0 only'
        self.k_exploration_steps = gen_kwargs.get('k_exploration_steps', 6)
        self.cycle_len_confidence_threshold = gen_kwargs.get('cycle_len_confidence_threshold', 0.3)
        self.cycle_length_stability_window = gen_kwargs.get('cycle_length_stability_window', 2)
        self.cycle_length_stability_std_dev_threshold = gen_kwargs.get('cycle_length_stability_std_dev_threshold', 1.0)
        self.high_confidence_threshold = gen_kwargs.get('high_confidence_threshold', 0.9)
        self.num_important_low_confidence_tokens = gen_kwargs.get('num_important_low_confidence_tokens', 3)
        self.max_sub_cycles_per_block = gen_kwargs.get('max_sub_cycles_per_block', 256)
        self.gen_length = gen_kwargs.get('gen_length', 128)
        self.block_length = gen_kwargs.get('block_length', 128)
        self.use_cache = gen_kwargs.get('use_cache', True)
        self._prefix_embeds = None  # set per generation

    # ---- faithful helpers -------------------------------------------------
    def add_gumbel_noise(self, logits):
        if self.temperature == 0:
            return logits.exp()
        noise = torch.rand_like(logits)
        gumbel_noise = (-torch.log(noise)) ** self.temperature
        return logits.exp() / gumbel_noise

    def get_num_tokens_for_phase1_step(self, current_sub_cycle_mask):
        batch_size = current_sub_cycle_mask.shape[0]
        return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device)

    def get_num_tokens_for_phase3_step(self, current_sub_cycle_mask):
        batch_size = current_sub_cycle_mask.shape[0]
        return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device)

    # ---- LLaDA-V adaptation: ids -> embeds forward ------------------------
    def _logits(self, x, upto: Optional[int] = None):
        """Forward the multimodal prefix embeds + embedded suffix ids.

        x: (1, P + L) token ids; positions [:P] are placeholders (prefix lives
        in self._prefix_embeds). upto: absolute end index for truncated
        forwards (upto > P always holds in the sampler).
        """
        P = self._prefix_embeds.shape[1]
        end = x.shape[1] if upto is None else upto
        suffix_ids = x[:, P:end]
        emb = self.model.get_model().embed_tokens(suffix_ids)
        full = torch.cat([self._prefix_embeds, emb], dim=1)
        outputs = self.model.model(inputs_embeds=full)
        return self.model.lm_head(outputs[0]).float()

    # ---- sampler phases (verbatim logic, model calls swapped) -------------
    def slow_phase(self, x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                   actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                   prompt_index_full_x, attention_mask):
        batch_size = x.shape[0]
        block_start_in_gen = block_idx * self.block_length
        block_end_in_gen = (block_idx + 1) * self.block_length
        sub_cycle_determined_per_item = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        history_per_item = [collections.deque(maxlen=self.cycle_length_stability_window) for _ in range(batch_size)]

        for k_step in range(self.k_exploration_steps):
            logits_full = self._logits(x)

            logits_gen_part = logits_full[:, prompt_length:]
            x0_gen = torch.argmax(self.add_gumbel_noise(logits_gen_part), dim=-1)
            p_gen = F.softmax(logits_gen_part, dim=-1)
            x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)

            current_global_mask_index_gen_part = (x[:, prompt_length:] == self.mask_id)
            confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen,
                                              torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))
            for b_idx in range(batch_size):
                if not sub_cycle_determined_per_item[b_idx]:
                    previous_len_item = last_sub_cycle_length_per_item[b_idx].item()
                    observation_abs_start_in_gen = block_start_in_gen + previous_len_item
                    observation_abs_end_in_gen = block_end_in_gen
                    increment_len = 0

                    if observation_abs_start_in_gen < observation_abs_end_in_gen:
                        confidence_in_observation_scope = confidence_gen_wide[b_idx, observation_abs_start_in_gen:observation_abs_end_in_gen]
                        if confidence_in_observation_scope.numel() > 0:
                            above_thresh_indices_in_scope = (confidence_in_observation_scope >= self.cycle_len_confidence_threshold).nonzero(as_tuple=True)[0]
                            if len(above_thresh_indices_in_scope) > 0:
                                farthest_idx_in_scope = above_thresh_indices_in_scope.max().item()
                                increment_len = farthest_idx_in_scope + 1
                            else:
                                increment_len = 1
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

            num_to_fill_p1 = self.get_num_tokens_for_phase1_step(mask_in_current_block_abs_coords)
            transfer_mask_p1 = torch.zeros_like(x0_gen, dtype=torch.bool)
            for b_idx in range(batch_size):
                previous_len_item_fill = last_sub_cycle_length_per_item[b_idx].item()
                fill_op_abs_start_in_gen = block_start_in_gen + previous_len_item_fill
                fill_op_abs_end_in_gen = block_end_in_gen
                if fill_op_abs_start_in_gen < fill_op_abs_end_in_gen:
                    conf_in_fill_op_scope = confidence_gen_wide[b_idx, fill_op_abs_start_in_gen:fill_op_abs_end_in_gen]
                    mask_in_fill_op_scope = (x[b_idx, prompt_length + fill_op_abs_start_in_gen: prompt_length + fill_op_abs_end_in_gen] == self.mask_id)
                    if conf_in_fill_op_scope.numel() > 0:
                        eff_conf_in_fill_op_scope = torch.where(mask_in_fill_op_scope, conf_in_fill_op_scope,
                                                                torch.tensor(-np.inf, device=x.device, dtype=conf_in_fill_op_scope.dtype))
                        num_masked_in_fill_op_scope = mask_in_fill_op_scope.sum().item()
                        if num_to_fill_p1[b_idx] > 0 and num_masked_in_fill_op_scope > 0:
                            k = min(num_to_fill_p1[b_idx].item(), num_masked_in_fill_op_scope)
                            phase1_high_conf_fill_indices = (conf_in_fill_op_scope >= self.high_confidence_threshold) & mask_in_fill_op_scope
                            if phase1_high_conf_fill_indices.any() and phase1_high_conf_fill_indices.sum().item() > 1:
                                abs_indices_to_fill = fill_op_abs_start_in_gen + phase1_high_conf_fill_indices.nonzero(as_tuple=True)[0]
                                transfer_mask_p1[b_idx, abs_indices_to_fill] = True
                            else:
                                if k > 0:
                                    top_k_indices_relative_to_fill_scope = torch.topk(eff_conf_in_fill_op_scope, k=k).indices
                                    abs_indices_to_fill_in_gen = fill_op_abs_start_in_gen + top_k_indices_relative_to_fill_scope
                                    transfer_mask_p1[b_idx, abs_indices_to_fill_in_gen] = True
            x[:, prompt_length:][transfer_mask_p1] = x0_gen[transfer_mask_p1]

        for b_idx in range(batch_size):
            if not sub_cycle_determined_per_item[b_idx]:
                if len(history_per_item[b_idx]) > 0:
                    actual_sub_cycle_length_per_item[b_idx] = max(1, min(int(np.mean(list(history_per_item[b_idx]))), self.block_length))
                else:
                    actual_sub_cycle_length_per_item[b_idx] = self.block_length // 2
                sub_cycle_determined_per_item[b_idx] = True
        return x

    def fast_phase(self, x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                   actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                   prompt_index_full_x, attention_mask):
        batch_size = x.shape[0]
        phase_2_and_3_calls = 0
        block_start_in_gen = block_idx * self.block_length
        cache_out_cycle_full_logits_list = []
        active_region_start_check_list = []
        active_region_end_check_list = []
        while True:
            active_region_start_check_list = []
            active_region_end_check_list = []
            all_p2_active_regions_filled_for_all_items = True
            for b_idx_check in range(batch_size):
                current_cumulative_len_check = actual_sub_cycle_length_per_item[b_idx_check].item()
                previous_cumulative_len_check = last_sub_cycle_length_per_item[b_idx_check].item()
                active_region_start_check_list.append(block_start_in_gen + previous_cumulative_len_check)
                active_region_end_check_list.append(block_start_in_gen + current_cumulative_len_check)
                if active_region_start_check_list[b_idx_check] < active_region_end_check_list[b_idx_check]:
                    mask_in_ar_check = (x[b_idx_check, prompt_length + active_region_start_check_list[b_idx_check]: prompt_length + active_region_end_check_list[b_idx_check]] == self.mask_id)
                    if mask_in_ar_check.any():
                        all_p2_active_regions_filled_for_all_items = False
                        break
            if all_p2_active_regions_filled_for_all_items:
                break

            phase_2_and_3_calls += 1

            if phase_2_and_3_calls == 1:
                logits_full = self._logits(x)
                for b_idx_check in range(batch_size):
                    cache_out_cycle_full_logits_list.append(
                        logits_full[b_idx_check, prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
            else:
                logits_full_batch = []
                for b_idx_check in range(batch_size):
                    logits_full_item = self._logits(x, upto=prompt_length + active_region_end_check_list[b_idx_check])
                    logits_full_batch.append(torch.cat(
                        [logits_full_item[b_idx_check].unsqueeze(0), cache_out_cycle_full_logits_list[b_idx_check]], dim=1))
                logits_full = torch.cat(logits_full_batch, dim=0)

            logits_gen_part = logits_full[:, prompt_length:]
            x0_gen = torch.argmax(self.add_gumbel_noise(logits_gen_part), dim=-1)
            p_gen = F.softmax(logits_gen_part, dim=-1)
            x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)
            current_global_mask_index_gen_part = (x[:, prompt_length:] == self.mask_id)
            confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen,
                                              torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))
            transfer_mask_p2_and_p3 = torch.zeros_like(x0_gen, dtype=torch.bool)

            for b_idx in range(batch_size):
                sub_cycle_abs_end_in_gen = block_start_in_gen + actual_sub_cycle_length_per_item[b_idx].item()
                sub_cycle_abs_start_in_gen = block_start_in_gen + last_sub_cycle_length_per_item[b_idx].item()

                conf_in_sub_cycle_scope = confidence_gen_wide[b_idx, sub_cycle_abs_start_in_gen:sub_cycle_abs_end_in_gen]
                mask_in_sub_cycle_scope = (x[b_idx, prompt_length + sub_cycle_abs_start_in_gen: prompt_length + sub_cycle_abs_end_in_gen] == self.mask_id)

                high_conf_fill_indices = (conf_in_sub_cycle_scope >= self.high_confidence_threshold) & mask_in_sub_cycle_scope

                if high_conf_fill_indices.any() and high_conf_fill_indices.sum().item() > 1:
                    abs_indices_to_fill = sub_cycle_abs_start_in_gen + high_conf_fill_indices.nonzero(as_tuple=True)[0]
                    transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                else:
                    n2_num_transfer_tokens = self.get_num_tokens_for_phase3_step(mask_in_current_block_abs_coords)
                    eff_conf_sub_cycle = torch.where(mask_in_sub_cycle_scope, conf_in_sub_cycle_scope,
                                                     torch.tensor(-np.inf, device=x.device, dtype=conf_in_sub_cycle_scope.dtype))
                    top_k_indices_relative_to_sub_cycle = torch.topk(eff_conf_sub_cycle, k=n2_num_transfer_tokens[b_idx].item()).indices
                    abs_indices_to_fill = sub_cycle_abs_start_in_gen + top_k_indices_relative_to_sub_cycle
                    transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True

            x[:, prompt_length:][transfer_mask_p2_and_p3] = x0_gen[transfer_mask_p2_and_p3]
        return x

    # ---- generation entry --------------------------------------------------
    @torch.no_grad()
    def generate_from_embeds(self, inputs_embeds):
        self._prefix_embeds = inputs_embeds
        device = inputs_embeds.device
        batch_size, prompt_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        assert batch_size == 1, 'this port supports batch_size=1 only'

        if self.use_cache:
            from llava.hooks.slowfast_cache import SFFeatureCache
            SFFeatureCache().reset_cache(prompt_length=prompt_length,
                                         gen_length=self.gen_length)

        x = torch.full((batch_size, prompt_length + self.gen_length),
                       self.mask_id, dtype=torch.long, device=device)
        # prefix slots are placeholders; only used for index bookkeeping
        prompt_index_full_x = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full_x[:, :prompt_length] = True

        assert self.gen_length % self.block_length == 0
        num_blocks = self.gen_length // self.block_length

        for block_idx in range(num_blocks):
            block_abs_start_in_x = prompt_length + block_idx * self.block_length
            block_abs_end_in_x = prompt_length + (block_idx + 1) * self.block_length

            current_sub_cycles_in_block = 0
            actual_sub_cycle_length_per_item = torch.full((batch_size,), self.block_length, dtype=torch.long, device=device)
            last_sub_cycle_length_per_item = torch.full((batch_size,), 0, dtype=torch.long, device=device)

            while True:
                mask_in_current_block_abs_coords = (x[:, block_abs_start_in_x:block_abs_end_in_x] == self.mask_id)
                if not mask_in_current_block_abs_coords.any():
                    break
                if current_sub_cycles_in_block >= self.max_sub_cycles_per_block:
                    break

                current_sub_cycles_in_block += 1

                x = self.slow_phase(x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                                    actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                                    prompt_index_full_x, None)
                x = self.fast_phase(x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                                    actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                                    prompt_index_full_x, None)

                last_sub_cycle_length_per_item = actual_sub_cycle_length_per_item.clone()

        self._prefix_embeds = None
        return x[:, prompt_length:]


def register_slowfast_hook(model, **gen_kwargs):
    """gen_kwargs: sampler params + use_cache (default True = composed with the
    SlowFast-compatible dLLM-Cache stack) + prompt_interval_steps /
    gen_interval_steps / transfer_ratio for that cache."""
    assert not hasattr(model, _HOOK_ATTR), 'slowfast hook already registered'
    sampler = SlowFastLLaDAV(model, gen_kwargs)
    original_generate = model.generate

    if sampler.use_cache:
        from llava.hooks.slowfast_cache import SFFeatureCache, register_sf_cache_LLaDA_V
        SFFeatureCache.new_instance(
            prompt_interval_steps=gen_kwargs.get('prompt_interval_steps', 25),
            gen_interval_steps=gen_kwargs.get('gen_interval_steps', 7),
            transfer_ratio=gen_kwargs.get('transfer_ratio', 0.25))
        register_sf_cache_LLaDA_V(model, 'model.layers')

    @torch.no_grad()
    def _slowfast_generate(inputs=None, images=None, image_sizes=None,
                           modalities: Optional[List[str]] = None, **kwargs):
        mods = modalities if modalities is not None else ['image']
        position_ids = kwargs.pop('position_ids', None)
        attention_mask = kwargs.pop('attention_mask', None)
        if images is not None:
            (_, _, _, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, mods,
                image_sizes=image_sizes)
        else:
            inputs_embeds = model.get_model().embed_tokens(inputs)
        return sampler.generate_from_embeds(inputs_embeds)

    setattr(model, _HOOK_ATTR, {'original_generate': original_generate, 'sampler': sampler})
    model.generate = _slowfast_generate
    return sampler


def unregister_slowfast_hook(model):
    state = getattr(model, _HOOK_ATTR, None)
    if state is not None:
        if state['sampler'].use_cache:
            from llava.hooks.slowfast_cache import logout_sf_cache_LLaDA_V
            logout_sf_cache_LLaDA_V(model, 'model.layers')
        model.generate = state['original_generate']
        delattr(model, _HOOK_ATTR)
