"""SlowFast sampling with the CoTA / CoTA++ components on LLaDA-V (2026-09-17).

Cache hook
    Generated at import time from llava/hooks/cache_hook_LLaDA_V.py, the hook behind every
    dLLM-Cache result in the paper, with the SlowFast evolved-cache delta of
    llava/hooks/slowfast_cache.py applied:
      * the feature cache is SFFeatureCache (tolerates truncated forwards),
      * the block hook records expect_length = seq_len - prompt_length,
      * DAR's reserved indices are clipped to the gen span of the current forward.
    The generated module keeps its own component state (_STITCH, _DAR, _CTAE, ...), so the
    dLLM-Cache path is untouched and both paths stay in sync with one source file.

Sampler
    SlowFastLLaDAV with three insertions, each made where the sampler already holds the
    quantity it needs:
      * CTEV: the selection score of every commit rule (high-confidence threshold and top-k)
        is the confidence minus the context-entropy penalty; the slow phase's cycle-length
        estimate keeps reading the raw confidence.
      * DAR: after every commit the scores are published for the next forward
        ('masked': final scores of the positions still masked; 'legacy': raw confidence).
      * CTAR: after every commit the context-anchor monitor is advanced.

With every component off the sampler and hook reproduce slowfast_hook + slowfast_cache.

Two knobs exist only here, for the SlowFast backend on LLaDA-V, and are off by default:
  * sf_thresh='raw' leaves the backend's own high-confidence threshold (which decides whether a
    step commits one token or many) on the raw confidence, and lets the CTEV score re-order only
    the top-k fallback. With the default 'score' the penalty shifts the quantity that the
    threshold is calibrated against, which at L=512 makes the sampler commit in bulk exactly
    where the context has collapsed.
  * sf_repguard adds a penalty proportional to the local density of committed runs.
"""
import inspect
import math
import sys
import textwrap
import types
import collections
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from llava.hooks import slowfast_hook as _sfh
from llava.hooks.slowfast_cache import SFFeatureCache

_HOOK_ATTR = '_slowfast_cotapp_state'


# ---------------------------------------------------------------------------------------
# cache hook: current dLLM-Cache hook + SlowFast delta
# ---------------------------------------------------------------------------------------
def _build_hook_module():
    import llava.hooks.cache_hook_LLaDA_V as base
    src = open(base.__file__).read()
    reps = [
        ("from llava.cache import dLLMCache",
         "from llava.hooks.slowfast_cache import SFFeatureCache as dLLMCache"),
        ("    bs, seq_len, dim = hidden_states.shape # seq_len is the length of hidden_states input to this layer\n",
         "    bs, seq_len, dim = hidden_states.shape # seq_len is the length of hidden_states input to this layer\n"
         "    feature_cache.expect_length = seq_len - prompt_length  # SlowFast delta: truncated forwards\n"),
        ("        cos_sim[0, imminent.to(cos_sim.device)] = -2.0   # below any cosine value",
         "        _imm = imminent.to(cos_sim.device)\n"
         "        _imm = _imm[_imm < cos_sim.shape[1]]            # truncated forward: clip to span\n"
         "        cos_sim[0, _imm] = -2.0   # below any cosine value"),
        # CTAR under truncated forwards: the gen span of this forward is expect_length (<= M)
        ("    if S_k < M or B != 1:\n        return\n    P = S_k - M\n",
         "    if S_k < M or B != 1:\n        return\n"
         "    _g = int(getattr(dLLMCache(), 'expect_length', M) or M)   # SlowFast: truncated span\n"
         "    P = S_k - _g\n"),
        ("        if Qf is not None and Qf.shape == Q.shape:\n"
         "            Q = Qf                       # query recomputed from the current hidden state\n",
         "        if Qf is not None and Qf.shape == Q.shape:\n"
         "            Q = Qf                       # query recomputed from the current hidden state\n"
         "        elif Qf is not None and Qf.shape[0] == Q.shape[0] and Qf.shape[1] < Q.shape[1]:\n"
         "            Q = Q.clone()                # truncated forward: current queries for its span\n"
         "            Q[:, :Qf.shape[1], :] = Qf\n"),
        ("            sidx = sel.nonzero(as_tuple=False).squeeze(-1)\n",
         "            sidx = sel.nonzero(as_tuple=False).squeeze(-1)\n"
         "            sidx = sidx[sidx < S_k - P]      # truncated forward: rows present in this pass\n"),
        ("    if _CTAE['mode'] == 'reroute_q' and _STITCH['on'] and x_gen.shape[1] == (_STITCH['M'] or -1) \\\n",
         "    if _CTAE['mode'] == 'reroute_q' and _STITCH['on'] and 0 < x_gen.shape[1] <= (_STITCH['M'] or -1) \\\n"),
        # A payload is only valid for the forward that built it: SlowFast alternates full and
        # truncated forwards, and the hook applies a payload in some branches only, so stamp the
        # gen span on it and skip it when the span of the current forward differs.
        ("        _STITCH['out'][layer_idx] = out if sidx is None else (out, sidx)\n",
         "        _STITCH['out'][layer_idx] = out if sidx is None else (out, sidx, S_k - P)\n"),
        ("    if isinstance(_st, tuple):\n        _new, _idx = _st\n        _new = _new.to(att_gen.dtype)\n",
         "    if isinstance(_st, tuple):\n        _new, _idx = _st[0], _st[1]\n"
         "        if len(_st) > 2 and _st[2] != att_gen.shape[1]:\n"
         "            return att_gen          # built for a forward with a different gen span\n"
         "        _new = _new.to(att_gen.dtype)\n"),
    ]
    for a, b in reps:
        assert src.count(a) == 1, ('sf_cotapp: anchor not found once in cache_hook_LLaDA_V', a[:60])
        src = src.replace(a, b)
    name = 'llava.hooks._sf_cotapp_hook'
    mod = types.ModuleType(name)
    mod.__file__ = base.__file__ + ' [slowfast delta]'
    mod.__dict__['__generated_source__'] = src      # keep for debugging line numbers
    exec(compile(src, '<sf_cotapp generated hook>', 'exec'), mod.__dict__)
    sys.modules[name] = mod
    return mod


H = _build_hook_module()


def _layers(model, key='model.layers'):
    m = model
    for part in key.split('.'):
        m = getattr(m, part)
    return m


def register_hook(model, key='model.layers'):
    H.register_cache_LLaDA_V(model, key)


def logout_hook(model, key='model.layers'):
    for blk in _layers(model, key):
        if hasattr(blk, '_old_forward'):
            blk.forward = blk._old_forward
        if hasattr(blk.self_attn, '_old_forward_main'):
            blk.self_attn.forward = blk.self_attn._old_forward_main
        if hasattr(blk.self_attn.rotary_emb, '_old_forward'):
            blk.self_attn.rotary_emb.forward = blk.self_attn.rotary_emb._old_forward


# ---------------------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------------------
DEFAULT_COMP = dict(
    ctae_mode='off', ctae_sigma=2.0, ctae_gamma=0.75,          # 'gated' = released CoTA CTAE
    ctar=False, stitch_lo=24, stitch_hi=31, ctar_scope='all', ctar_theta=0, ctar_w=5,
    dar_r=0, dar_mode='masked',
    ctev_mode='off', ctev_lambda=0.25, ctev_window=5, ctev_norm=1,
    ctev_layers=(26, 27, 28, 29, 30),
    # SlowFast-only, default off (see the module docstring)
    sf_thresh='score',      # 'raw': the backend's own threshold keeps reading the raw confidence
    sf_repguard=0.0,        # extra CTEV penalty proportional to the local density of committed runs
)


def _patched_method(fn, edits):
    """Re-compile a SlowFastLLaDAV method with line insertions. Each edit is
    (anchor, text, where): `anchor` must match exactly one stripped source line; `text`
    (possibly several lines) is inserted after/before it at the indentation of the
    statement that line belongs to (for a continuation line, the previous line)."""
    lines = inspect.getsource(fn).split('\n')
    for anchor, text, where in edits:
        hits = [i for i, l in enumerate(lines) if l.strip() == anchor]
        assert len(hits) == 1, ('sf_cotapp: sampler anchor not unique', anchor, len(hits))
        i = hits[0]
        j = i - 1 if (where == 'after_cont') else i
        ind = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
        new = [ind + t for t in text.split('\n')]
        lines[i + 1:i + 1] = new if where in ('after', 'after_cont') else []
        if where == 'before':
            lines[i:i] = new
    src = 'class _Tmp:\n' + '\n'.join(lines)
    ns = dict(_sfh.__dict__)
    exec(compile(src, f'<sf_cotapp {fn.__name__}>', 'exec'), ns)
    return ns['_Tmp'].__dict__[fn.__name__]


_CONF_CONT = "torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))"

_EFF_SLOW = "torch.tensor(-np.inf, device=x.device, dtype=conf_in_fill_op_scope.dtype))"
_EFF_FAST = "torch.tensor(-np.inf, device=x.device, dtype=conf_in_sub_cycle_scope.dtype))"

_slow = _patched_method(_sfh.SlowFastLLaDAV.slow_phase, [
    # DAR legacy reads the raw confidence; the cycle-length estimate keeps reading it too
    (_CONF_CONT, "self._raw_conf = confidence_gen_wide", 'after_cont'),
    ("num_to_fill_p1 = self.get_num_tokens_for_phase1_step(mask_in_current_block_abs_coords)",
     "self._adj = self._score(confidence_gen_wide, x, prompt_length)\n"
     "if self.c['sf_thresh'] != 'raw':\n"
     "    confidence_gen_wide = self._adj", 'before'),
    # 'raw': the high-confidence threshold above stays on the raw confidence and only the
    # top-k fallback is re-ordered by the CTEV score
    (_EFF_SLOW,
     "if self.c['sf_thresh'] == 'raw':\n"
     "    eff_conf_in_fill_op_scope = torch.where(mask_in_fill_op_scope,\n"
     "        self._adj[b_idx, fill_op_abs_start_in_gen:fill_op_abs_end_in_gen],\n"
     "        torch.tensor(-np.inf, device=x.device, dtype=conf_in_fill_op_scope.dtype))", 'after_cont'),
    ("x[:, prompt_length:][transfer_mask_p1] = x0_gen[transfer_mask_p1]",
     "self._after_commit(x, prompt_length, self._adj, transfer_mask_p1)", 'after'),
])

_fast = _patched_method(_sfh.SlowFastLLaDAV.fast_phase, [
    (_CONF_CONT, "self._raw_conf = confidence_gen_wide\n"
                 "self._adj = self._score(confidence_gen_wide, x, prompt_length)\n"
                 "if self.c['sf_thresh'] != 'raw':\n"
                 "    confidence_gen_wide = self._adj", 'after_cont'),
    (_EFF_FAST,
     "if self.c['sf_thresh'] == 'raw':\n"
     "    eff_conf_sub_cycle = torch.where(mask_in_sub_cycle_scope,\n"
     "        self._adj[b_idx, sub_cycle_abs_start_in_gen:sub_cycle_abs_end_in_gen],\n"
     "        torch.tensor(-np.inf, device=x.device, dtype=conf_in_sub_cycle_scope.dtype))", 'after_cont'),
    ("x[:, prompt_length:][transfer_mask_p2_and_p3] = x0_gen[transfer_mask_p2_and_p3]",
     "self._after_commit(x, prompt_length, self._adj, transfer_mask_p2_and_p3)", 'after'),
])


class SlowFastCoTA(_sfh.SlowFastLLaDAV):
    slow_phase = _slow
    fast_phase = _fast

    def __init__(self, model, gen_kwargs, comp):
        super().__init__(model, gen_kwargs)
        self.c = dict(DEFAULT_COMP, **(comp or {}))
        self.log2v = math.log2(model.lm_head.out_features)
        self._hid = None
        self._E = None
        self._raw_conf = None
        self._adj = None
        self.stats = collections.Counter()

    # forward: also keep the deep hidden states when CTEV needs them
    def _logits(self, x, upto: Optional[int] = None):
        P = self._prefix_embeds.shape[1]
        end = x.shape[1] if upto is None else upto
        emb = self.model.get_model().embed_tokens(x[:, P:end])
        full = torch.cat([self._prefix_embeds, emb], dim=1)
        need_h = self.c['ctev_mode'] != 'off'
        out = self.model.model(inputs_embeds=full, output_hidden_states=need_h)
        self._hid = (out.hidden_states, P, end - P) if need_h else None
        self.stats['forwards'] += 1
        return self.model.lm_head(out[0]).float()

    def _deep_entropy(self):
        """Deep-window mean entropy (bits) of the gen positions covered by the last forward,
        merged into a persistent buffer so truncated forwards keep the older values."""
        hs, P, g = self._hid
        acc = None
        for li in self.c['ctev_layers']:
            h = hs[li][:1, P:P + g, :]
            if self.c['ctev_norm']:
                h = self.model.model.norm(h)
            logp = F.log_softmax(self.model.lm_head(h).float(), dim=-1)
            ent = -(logp.exp() * logp).sum(-1)[0]                 # nats
            acc = ent if acc is None else acc + ent
        E_new = acc * (1.4426950408889634 / len(self.c['ctev_layers']))
        if self._E is None or self._E.numel() != self.gen_length:
            self._E = torch.zeros(self.gen_length, device=E_new.device, dtype=E_new.dtype)
        self._E[:g] = E_new
        return self._E

    def _score(self, conf, x, P):
        mode = self.c['ctev_mode']
        if mode == 'off' and self.c['sf_repguard'] <= 0:
            return conf
        if mode != 'off' and self._hid is None:
            return conf
        if mode == 'off':                      # run guard alone, no entropy read
            E = torch.zeros(self.gen_length, device=conf.device, dtype=conf.dtype)
            pen = torch.zeros_like(E)
            return self._repguard(conf, x, P, E, pen)
        E = self._deep_entropy()
        lam = self.c['ctev_lambda']
        if mode == 'self':                                      # conference CoTA release
            pen = lam * (E / self.log2v)
        else:                                                   # CoTA++ CTEV: committed neighbours
            committed = (x[0, P:P + self.gen_length] != self.mask_id).to(E.dtype)
            w = self.c['ctev_window']
            ker = torch.ones(1, 1, 2 * w + 1, device=E.device, dtype=E.dtype)
            sumE = F.conv1d((E * committed).view(1, 1, -1), ker, padding=w).view(-1)
            cnt = F.conv1d(committed.view(1, 1, -1), ker, padding=w).view(-1)
            E_ctx = torch.where(cnt > 0, sumE / cnt.clamp(min=1.0), torch.zeros_like(sumE))
            pen = lam * (E_ctx / self.log2v)
        return self._repguard(conf, x, P, E, pen)

    def _repguard(self, conf, x, P, E, pen):
        if self.c['sf_repguard'] > 0:
            # A neighbourhood that is already a run of one token has low entropy for the wrong
            # reason: it is a degenerate attractor, not a consolidated context. Charge it.
            ids = x[0, P:P + self.gen_length]
            comm = ids != self.mask_id
            same = (ids[1:] == ids[:-1]) & comm[1:] & comm[:-1]
            rep = torch.zeros_like(E)
            rep[1:] += same.to(E.dtype)
            rep[:-1] += same.to(E.dtype)
            w = self.c['ctev_window']
            ker = torch.ones(1, 1, 2 * w + 1, device=E.device, dtype=E.dtype)
            dens = F.conv1d(rep.clamp(max=1.0).view(1, 1, -1), ker, padding=w).view(-1) / (2 * w + 1)
            pen = pen + self.c['sf_repguard'] * dens
            self.stats['repguard_calls'] += 1
        self.stats['ctev_calls'] += 1
        return conf - pen[:conf.shape[1]].unsqueeze(0).to(conf.dtype)

    def _after_commit(self, x, P, score, tmask):
        if self.c['ctar'] and self.c['ctar_theta'] > 0:
            H.publish_mask((x[0, P:P + self.gen_length] == self.mask_id).detach())
        if self.c['dar_r'] > 0:
            if self.c['dar_mode'] in ('masked', 'score'):
                sv = score[0].detach().clone()
                if self.c['dar_mode'] == 'masked':
                    sv[tmask[0]] = -float('inf')
            else:
                sv = self._raw_conf[0].detach()
            H.publish_scores(sv)
        self.stats['commits'] += int(tmask.sum())

    def _setup(self):
        c = self.c
        if c['ctar']:
            H.set_ctae(mode='reroute_q')
            H.set_stitch(True, M=self.gen_length, lo=c['stitch_lo'], hi=c['stitch_hi'])
            H.set_ctarx(scope=c['ctar_scope'], theta=c['ctar_theta'], w=c['ctar_w'])
            H.reset_stitch()
        elif c['ctae_mode'] != 'off':
            H.set_ctae(mode=c['ctae_mode'], sigma=c['ctae_sigma'], gamma=c['ctae_gamma'])
        else:
            H.set_ctae(mode='off')
        if c['dar_r'] > 0:
            H.set_dar(True, r=c['dar_r'])
            H.reset_dar()
        else:
            H.set_dar(False)
        self._E = None
        self._raw_conf = None
        self._adj = None

    @torch.no_grad()
    def generate_from_embeds(self, inputs_embeds):
        self._setup()
        return super().generate_from_embeds(inputs_embeds)


def register_slowfast_cotapp_hook(model, comp=None, **gen_kwargs):
    """Drop-in for slowfast_hook.register_slowfast_hook with the components in `comp`."""
    assert not hasattr(model, _HOOK_ATTR), 'slowfast_cotapp hook already registered'
    sampler = SlowFastCoTA(model, gen_kwargs, comp)
    original_generate = model.generate
    if sampler.use_cache:
        SFFeatureCache.new_instance(
            prompt_interval_steps=gen_kwargs.get('prompt_interval_steps', 25),
            gen_interval_steps=gen_kwargs.get('gen_interval_steps', 7),
            transfer_ratio=gen_kwargs.get('transfer_ratio', 0.25))
        register_hook(model)

    @torch.no_grad()
    def _generate(inputs=None, images=None, image_sizes=None,
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
    model.generate = _generate
    return sampler
