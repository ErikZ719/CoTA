"""CoTA / CoTA++ components for MMaDA-8B on dLLM-Cache and SlowFast (2026-09-17).

The two MMaDA backends used in the paper are
  * dllm_cache/hooks/cache_hook_MMaDA.py      (dLLM-Cache, with mmu_generate_with_cache)
  * scripts/sf_mmada_stack.py                  (SlowFast sampler + its evolved cache)
Both are loaded here from their own source with a few line insertions, so the originals stay
untouched and every component lives in this file:

  DAR   refresh_index -> dar_select(): the reserved positions are forced into the refresh set.
  CTAR  after the backend scatters its recomputed rows into the stored attention output,
        ctar_apply() re-forms the routing of the rows the context-anchor monitor selected
        (query from the current hidden state, keys/values the cache holds, explicit softmax).
  CoTA  ctae_mask(): additive log-gain of the conference Gaussian decay on suffix-suffix
        attention (mode 'bias'); CTEV mode 'self' penalises the candidate's own entropy.
  CTEV  the unmasking score is the confidence minus lambda * context entropy / log2|V|.

With every component off, both backends reproduce the original code token for token.
"""
import collections
import math
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

C = dict(ctar=False, lo=24, hi=31, theta=1, w=5,
         dar_r=0, dar_mode='masked',
         ctae_mode='off', ctae_sigma=5.0, ctae_gamma=0.5,
         ctev_mode='off', ctev_lambda=0.25, ctev_window=5, ctev_norm=1,
         ctev_layers=(26, 27, 28, 29, 30))
S = dict(P=0, M=0, imminent=None, sel=None, late=None, mask_prev=None, hid=None, E=None)
STATS = collections.Counter()
_ME = sys.modules[__name__]


def configure(**kw):
    unknown = set(kw) - set(C)
    assert not unknown, f'mmada_cotapp: unknown component keys {unknown}'
    C.update(kw)


def any_on():
    return C['ctar'] or C['dar_r'] > 0 or C['ctae_mode'] != 'off' or C['ctev_mode'] != 'off'


def reset(P, M):
    S.update(P=int(P), M=int(M), imminent=None, sel=None, late=None, mask_prev=None, hid=None, E=None)


# ------------------------------------------------------------------ DAR
def publish_scores(sv):
    if C['dar_r'] <= 0 or sv is None:
        S['imminent'] = None
        return
    v = torch.nan_to_num(sv.detach().float().reshape(-1), neginf=-1e30)
    S['imminent'] = torch.topk(v, k=min(C['dar_r'], v.numel())).indices


def dar_select(cos_sim, num_replace):
    imm = S['imminent']
    if C['dar_r'] > 0 and imm is not None and num_replace > imm.numel():
        imm = imm.to(cos_sim.device)
        imm = imm[imm < cos_sim.shape[1]]            # truncated forwards (SlowFast)
        cos_sim = cos_sim.clone()
        cos_sim[0, imm] = -2.0
        STATS['dar_steps'] += 1
    return torch.topk(cos_sim, largest=False, k=num_replace).indices


# ------------------------------------------------------------------ CTAR monitor
def publish_mask(m):
    """m: bool [M], True = still masked. Called once per step after the commits."""
    if not C['ctar']:
        S['sel'] = None
        return
    m = m.detach().reshape(-1)
    if C['theta'] <= 0:                                 # plain CTAR: every masked row
        S['sel'] = m.clone()
        return
    M = m.numel()
    late = S['late']
    if late is None or late.numel() != M:
        late = torch.zeros(M, device=m.device, dtype=torch.float32)
        S['late'] = late
        S['mask_prev'] = None
    prev = S['mask_prev']
    if prev is not None and prev.numel() == M:
        newly = (prev & (~m)).float()
        if bool(newly.any()):
            w = C['w']
            ker = torch.ones(1, 1, 2 * w + 1, device=m.device, dtype=torch.float32)
            late += F.conv1d(newly.view(1, 1, -1), ker, padding=w).view(-1)
    S['mask_prev'] = m.clone()
    sel = (late >= float(C['theta'])) & m
    S['sel'] = sel
    late[sel] = 0.0


def explicit_attention(block, q, k, v, q_index):
    """MMaDA _attention with an explicit softmax (the hooked one calls SDPA)."""
    B, ql, Cd = q.shape
    kl = k.shape[1]
    dtype = k.dtype
    if block.q_norm is not None and block.k_norm is not None:
        q = block.q_norm(q).to(dtype=dtype)
        k = block.k_norm(k).to(dtype=dtype)
    nh = block.config.n_heads
    nkv = block.config.effective_n_kv_heads
    hd = Cd // nh
    q = q.view(B, ql, nh, hd).transpose(1, 2)
    k = k.view(B, kl, nkv, hd).transpose(1, 2)
    v = v.view(B, v.shape[1], nkv, hd).transpose(1, 2)
    if block.config.rope:
        q, k = block.rotary_emb(q, k, q_index=q_index)
    if nkv != nh:
        k = k.repeat_interleave(nh // nkv, dim=1)
        v = v.repeat_interleave(nh // nkv, dim=1)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(hd)
    bias = ctae_mask(ql, kl, q_index, torch.float32, q.device)
    if bias is not None:
        logits = logits + bias
    att = torch.matmul(torch.softmax(logits, dim=-1).to(v.dtype), v)
    return block.attn_out(att.transpose(1, 2).contiguous().view(B, ql, Cd))


def ctar_apply(block, x_gen, prompt_length, k_full, v_full, att_gen_cache):
    if not C['ctar'] or not (C['lo'] <= block.layer_id <= C['hi']):
        return att_gen_cache
    sel = S['sel']
    if sel is None:
        return att_gen_cache
    sidx = sel.nonzero(as_tuple=False).squeeze(-1)
    sidx = sidx[sidx < x_gen.shape[1]]
    if sidx.numel() == 0:
        return att_gen_cache
    q = block.q_proj(block.attn_norm(x_gen[:, sidx, :]))
    out = explicit_attention(block, q, k_full, v_full, (sidx + prompt_length).unsqueeze(0))
    att_gen_cache = att_gen_cache.clone()
    att_gen_cache[:, sidx, :] = out.to(att_gen_cache.dtype)
    STATS['ctar_rows'] += int(sidx.numel())
    STATS['ctar_calls'] += 1
    return att_gen_cache


# ------------------------------------------------------------------ CoTA attention decay
def ctae_mask(ql, kl, q_index, dtype, device):
    if C['ctae_mode'] != 'bias':
        return None
    P = S['P']
    if q_index is not None:
        qpos = q_index[0].to(device).long()
    else:
        qpos = torch.arange(kl - ql, kl, device=device)
    kpos = torch.arange(kl, device=device)
    d = (qpos[:, None] - kpos[None, :]).abs().float()
    g = C['ctae_gamma'] + (1.0 - C['ctae_gamma']) * torch.exp(-(d / C['ctae_sigma']) ** 2)
    valid = (qpos[:, None] >= P) & (kpos[None, :] >= P)
    STATS['ctae_calls'] += 1
    return torch.where(valid, torch.log(g), torch.zeros_like(g)).to(dtype)[None, None]


# ------------------------------------------------------------------ CTEV
def _tr_head(model):
    """(transformer ModuleDict, output head) for an MMaDA LM wrapper or a bare LLaDAModel (LaViDa)."""
    tr = model.transformer if hasattr(model, 'transformer') else model.model.transformer
    head = (lambda h: F.linear(h, tr.wte.weight)) if model.config.weight_tying else tr.ff_out
    return tr, head


def deep_entropy(model, hs, P, g):
    """Deep-window mean entropy (bits) of gen positions [0, g); merged into S['E']."""
    tr, head = _tr_head(model)
    acc = None
    for li in C['ctev_layers']:
        h = hs[li][:1, P:P + g, :]
        if C['ctev_norm']:
            h = tr.ln_f(h)
        logp = F.log_softmax(head(h).float(), dim=-1)
        ent = -(logp.exp() * logp).sum(-1)[0]
        acc = ent if acc is None else acc + ent
    E_new = acc * (1.4426950408889634 / len(C['ctev_layers']))
    if S['E'] is None or S['E'].numel() != S['M']:
        S['E'] = torch.zeros(S['M'], device=E_new.device, dtype=E_new.dtype)
    S['E'][:g] = E_new
    return S['E'], logp.shape[-1]


def penalty(model, hs, x_gen_ids, P, g, mask_id):
    E, V = deep_entropy(model, hs, P, g)
    lam = C['ctev_lambda']
    if C['ctev_mode'] == 'self':
        pen = lam * E / math.log2(V)
    else:
        committed = (x_gen_ids != mask_id).to(E.dtype)
        w = C['ctev_window']
        ker = torch.ones(1, 1, 2 * w + 1, device=E.device, dtype=E.dtype)
        sumE = F.conv1d((E * committed).view(1, 1, -1), ker, padding=w).view(-1)
        cnt = F.conv1d(committed.view(1, 1, -1), ker, padding=w).view(-1)
        E_ctx = torch.where(cnt > 0, sumE / cnt.clamp(min=1.0), torch.zeros_like(sumE))
        pen = lam * E_ctx / math.log2(V)
    STATS['ctev_calls'] += 1
    return pen


# ------------------------------------------------------------------ dLLM-Cache generation
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    n = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        n[i, :remainder[i]] += 1
    return n


@torch.no_grad()
def mmu_generate_cotapp(model, input_ids, max_new_tokens=128, steps=128, block_length=128,
                        mask_id=126336, attention_mask=None):
    """mmu_generate_with_cache (temperature 0, low-confidence remasking, no CFG) + components."""
    if attention_mask is not None and 0.0 in attention_mask:
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
    else:
        attention_bias = None
    P, L = input_ids.shape[1], max_new_tokens
    reset(P, L)
    x = torch.full((input_ids.shape[0], P + L), mask_id, dtype=torch.long, device=input_ids.device)
    x[:, :P] = input_ids.clone()
    assert L % block_length == 0
    num_blocks = L // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    need_h = C['ctev_mode'] != 'off'
    for nb in range(num_blocks):
        bmask = (x[:, P + nb * block_length: P + (nb + 1) * block_length] == mask_id)
        ntt = get_num_transfer_tokens(bmask, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if need_h:
                out = model(x, attention_bias=attention_bias, output_hidden_states=True)
            else:
                out = model(x, attention_bias=attention_bias)
            logits = out.logits
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, P + (nb + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            if C['dar_r'] > 0 and C['dar_mode'] == 'legacy':
                publish_scores(confidence[0, P:P + L])
            if need_h:
                pen = penalty(model, out.hidden_states, x[0, P:P + L], P, L, mask_id)
                confidence[:, P:P + L] = confidence[:, P:P + L] - pen.to(confidence.dtype)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=ntt[j, i])
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]
            if C['dar_r'] > 0 and C['dar_mode'] in ('masked', 'score'):
                sv = confidence[0, P:P + L].detach().clone()
                if C['dar_mode'] == 'masked':
                    sv[transfer_index[0, P:P + L]] = -float('inf')
                publish_scores(sv)
            publish_mask(x[0, P:P + L] == mask_id)
            STATS['steps'] += 1
    return x


# ------------------------------------------------------------------ SlowFast glue
class CotaShim:
    """Model wrapper for the SlowFast sampler: absorbs attention_mask and keeps hidden states."""
    def __init__(self, m):
        self._m = m
        self.device = next(m.parameters()).device

    def __call__(self, x, attention_mask=None):
        need = C['ctev_mode'] != 'off'
        out = self._m(x, output_hidden_states=True) if need else self._m(x)
        S['hid'] = (out.hidden_states, x.shape[1]) if need else None
        return out


def sf_score(sampler, conf, x, P):
    if C['ctev_mode'] == 'off' or S['hid'] is None:
        return conf
    hs, xl = S['hid']
    g = xl - P
    pen = penalty(sampler._model_for_head, hs, x[0, P:P + sampler.gen_length], P, g, sampler.mask_id)
    return conf - pen[:conf.shape[1]].unsqueeze(0).to(conf.dtype)


def after_commit(x, P, score, tmask, sampler):
    publish_mask(x[0, P:P + sampler.gen_length] == sampler.mask_id)
    if C['dar_r'] > 0:
        if C['dar_mode'] in ('masked', 'score'):
            sv = score[0].detach().clone()
            if C['dar_mode'] == 'masked':
                sv[tmask[0]] = -float('inf')
        else:
            sv = sampler._raw_conf[0].detach().clone()
        # Block-wise decoding: only the current block can commit, so only it can be imminent.
        # The dLLM-Cache path already masks the confidence beyond the block end before DAR reads
        # it; the sampler's confidence spans the whole response, where the far tail (padding the
        # model is certain of) would otherwise take every reserved slot.
        BL = int(getattr(sampler, 'block_length', 0) or 0)
        if 0 < BL < sv.numel():
            masked = (x[0, P:P + sampler.gen_length] == sampler.mask_id).nonzero(as_tuple=False)
            if masked.numel():
                bend = (int(masked[0]) // BL + 1) * BL
                sv[bend:] = -float('inf')
                STATS['dar_block_clip'] += 1
        publish_scores(sv)
    STATS['commits'] += int(tmask.sum())
    STATS['steps'] += 1


# ------------------------------------------------------------------ source loaders
def _inject(src, edits):
    """edits: (anchor, text, where, occurrence) with occurrence None = every match.
    Lines are matched after stripping indentation and trailing comments."""
    lines = src.split('\n')
    for anchor, text, where, occ in edits:
        hits = [i for i, l in enumerate(lines) if l.split('#')[0].strip() == anchor]
        assert hits, ('mmada_cotapp: anchor not found', anchor)
        if occ is not None:
            hits = [hits[occ]]
        for i in sorted(hits, reverse=True):
            ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            new = [ind + t for t in text.split('\n')]
            if where == 'after':
                lines[i + 1:i + 1] = new
            elif where == 'before':
                lines[i:i] = new
            else:                                  # replace
                lines[i:i + 1] = new
    return '\n'.join(lines)


_HOOK_EDITS = [
    ("transfer_index = torch.topk(cos_sim, largest=False, k=num_replace).indices",
     "transfer_index = _MC.dar_select(cos_sim, num_replace)", 'replace', None),
    ("att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)",
     "att_gen_cache = _MC.ctar_apply(self, x_gen, prompt_length, k, v, att_gen_cache)", 'after', None),
    ("attn_mask=None,",
     "attn_mask=_MC.ctae_mask(query_len, key_len, q_index, q.dtype, q.device),", 'replace', None),
]

_CONF = ("confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, "
         "torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))")
_SAMPLER_EDITS = [
    (_CONF, "self._raw_conf = confidence_gen_wide", 'after', 0),                        # slow
    (_CONF, "self._raw_conf = confidence_gen_wide\n"
            "confidence_gen_wide = _MC.sf_score(self, confidence_gen_wide, x, prompt_length)", 'after', 1),  # fast
    ("num_to_fill_p1 = self.get_num_tokens_for_phase1_step(mask_in_current_block_abs_coords)",
     "confidence_gen_wide = _MC.sf_score(self, confidence_gen_wide, x, prompt_length)", 'before', None),
    ("x[:, prompt_length:][transfer_mask_p1] = x0_gen[transfer_mask_p1]",
     "_MC.after_commit(x, prompt_length, confidence_gen_wide, transfer_mask_p1, self)", 'after', None),
    ("x[:, prompt_length:][transfer_mask_p2_and_p3] = x0_gen[transfer_mask_p2_and_p3]",
     "_MC.after_commit(x, prompt_length, confidence_gen_wide, transfer_mask_p2_and_p3, self)", 'after', None),
    ("feature_cache.reset_cache(prompt_length,gen_length=self.gen_length)",
     "_MC.reset(prompt_length, self.gen_length)", 'after', None),
]


def _load(path, name, edits):
    src = _inject(open(path).read(), edits)
    mod = types.ModuleType(name)
    mod.__file__ = path + ' [cotapp]'
    mod.__dict__['_MC'] = _ME
    exec(compile(src, f'<{name}>', 'exec'), mod.__dict__)
    sys.modules[name] = mod
    return mod


def build_dllm_hook():
    import dllm_cache.hooks.cache_hook_MMaDA as base
    return _load(base.__file__, 'dllm_cache.hooks._cotapp_MMaDA', _HOOK_EDITS)


def build_sf_stack(model):
    import sf_mmada_stack as base
    mod = _load(base.__file__, '_sf_mmada_cotapp', _HOOK_EDITS + _SAMPLER_EDITS)
    mod.SlowFastSampler._model_for_head = model          # the logit-lens head for CTEV
    mod.SlowFastSampler._raw_conf = None
    return mod
