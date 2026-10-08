import torch
from typing import Optional, Tuple, List 
import torch.nn as nn
import types
from llava.cache import dLLMCache 
import math 

# Helper functions from the new LLADa model (need to be accessible)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
# --- End of imports/helpers from new LLaDA model ---


# === CTEVCACHE-PATCH (2026-10-01): rows of the gen span recomputed in the current forward ===================
# Union over layers of the refresh sets; 'all' on the steps with a full gen refresh. Reset when layer 0 is entered.
# Consumed by modeling_llada.generate_with_embeds(ctev_cache=1) to recompute CTEV entropies only where they changed.
_CTEVC = {'refreshed': None, 'all': False}


def ctevc_reset(M, device):
    _CTEVC['refreshed'] = torch.zeros(int(M), dtype=torch.bool, device=device)
    _CTEVC['all'] = False


def ctevc_take():
    return _CTEVC['refreshed'], _CTEVC['all']


def ctevc_off():
    _CTEVC['refreshed'] = None
    _CTEVC['all'] = False
# ============================================================================================================

# === CTAE (TPAMI variant arbitration; default off = clean dLLM-Cache) ========
_CTAE_STATS = {'applied': 0, 'skipped': 0, 'rows': 0}
_CTAE = {'mode': 'off', 'sigma': 2.0, 'gamma': 0.75, 'upper': 5.0, 'block': 128,
         'lo': 0, 'hi': 99}   # inclusive layer band the decay is applied to

def set_ctae(mode='off', sigma=2.0, gamma=0.75, upper=5.0, block=128, lo=0, hi=99):
    """mode: 'off' | 'gated' (faithful released CoTA: pre-softmax, [median,upper]
    value gate, last-128 heuristic incl. partial forwards) | 'bias' (pre-softmax
    additive log-gain == renormalized multiplicative; full forwards only) |
    'mult' (post-softmax multiplicative, no renorm; full forwards only)."""
    _CTAE.update(mode=str(mode), sigma=float(sigma), gamma=float(gamma),
                 upper=float(upper), block=int(block), lo=int(lo), hi=int(hi))

def _ctae_gain(n, device, dtype, sigma, gamma):
    idx = torch.arange(n, device=device, dtype=dtype)
    dist = (idx[:, None] - idx[None, :]).abs()
    return gamma + (1.0 - gamma) * torch.exp(-(dist / sigma) ** 2)
# =============================================================================


def _ctae_suffix_gain(attn_weights, q_index, M, sigma, gamma):
    """Rows/cols of the suffix-suffix sub-block plus its Gaussian gain.

    Works for BOTH full and partial (cache-refresh) forwards: query rows are
    located by their absolute positions in q_index, so the decay always lands on
    the true |i-j| rather than a block-local offset. Keys are [prompt | gen], so
    the gen block is the last M columns and column c has absolute position P+c.
    Returns (rows, P, gain[R, M]) or None when the call carries no suffix query.
    """
    B, H, S_q, S_k = attn_weights.shape
    if S_k < M:
        return None
    P = S_k - M
    dev, dt = attn_weights.device, attn_weights.dtype
    if q_index is not None and q_index.numel() >= S_q:
        qi = q_index.reshape(-1)[:S_q].to(dev).long()
    else:                                   # fall back: Q is the tail of the sequence
        qi = torch.arange(S_k - S_q, S_k, device=dev, dtype=torch.long)
    rel = qi - P
    keep = (rel >= 0) & (rel < M)
    if not bool(keep.any()):
        return None
    rows = keep.nonzero(as_tuple=False).squeeze(-1)
    rq = rel[rows].to(dt)
    cq = torch.arange(M, device=dev, dtype=dt)
    dist = (rq[:, None] - cq[None, :]).abs()
    gain = gamma + (1.0 - gamma) * torch.exp(-(dist / sigma) ** 2)
    return rows, P, gain.to(dt)   # autocast promotes exp() to fp32; cast back


# === F1 anchoring probe: local attention mass w5 =============================
# For every suffix query row that a step actually computes, record the share of
# attention mass landing within +-half_w positions, per layer.  Rows that are
# not recomputed keep their last value -- which is exactly the routing the model
# is still consuming, because the cache reuses the attention OUTPUT produced by
# that row.  Read back as a deep-band mean.
_ANCHOR = {'on': False, 'lo': 26, 'hi': 31, 'half_w': 5, 'W': None, 'M': None}


def set_anchor(on=True, lo=26, hi=31, half_w=5):
    _ANCHOR.update(on=bool(on), lo=int(lo), hi=int(hi), half_w=int(half_w))


def reset_anchor(M, n_layers=64, device='cuda'):
    if _ANCHOR['on']:
        _ANCHOR['M'] = int(M)
        _ANCHOR['W'] = torch.zeros(n_layers, int(M), dtype=torch.float32, device=device)
        if _ANCHOR.get('pair_on'):
            _ANCHOR['pair'] = {k: torch.zeros(int(M), dtype=torch.float32, device=device) for k in ('old', 'new', 'n')}
            _ANCHOR['Wm'] = torch.zeros(n_layers, int(M), dtype=torch.float32, device=device)


def get_anchor_w5():
    """[M] deep-band mean local attention mass, or None."""
    W = _ANCHOR['W']
    if W is None:
        return None
    return W[_ANCHOR['lo']:_ANCHOR['hi'] + 1].mean(dim=0)


def _anchor_probe(attn_weights, q_index, M, layer_idx):
    if not _ANCHOR['on'] or _ANCHOR['W'] is None or layer_idx is None:
        return
    if not (_ANCHOR['lo'] <= layer_idx <= _ANCHOR['hi']):
        return
    B, H, S_q, S_k = attn_weights.shape
    if S_k < M:
        return
    P = S_k - M
    dev = attn_weights.device
    if q_index is not None and q_index.numel() >= S_q:
        qi = q_index.reshape(-1)[:S_q].to(dev).long()
    else:
        qi = torch.arange(S_k - S_q, S_k, device=dev, dtype=torch.long)
    rel = qi - P
    keep = (rel >= 0) & (rel < M)
    if not bool(keep.any()):
        return
    rows = keep.nonzero(as_tuple=False).squeeze(-1)
    rr = rel[rows]
    sub = attn_weights[:, :, rows, P:].float().mean(dim=(0, 1))      # [R, M]
    c = torch.arange(M, device=dev)
    band = ((rr[:, None] - c[None, :]).abs() <= _ANCHOR['half_w']).to(sub.dtype)
    _ANCHOR['W'][layer_idx, rr] = (sub * band).sum(-1)
    if _ANCHOR.get('Wm') is not None:          # analysis only: the routing as the real forwards left it
        _ANCHOR['Wm'][layer_idx, rr] = (sub * band).sum(-1)
# =============================================================================


# === CTAE 'stitch': complete-matrix intervention with output re-projection ===
_STITCH = {'on': False, 'M': None, 'A': {}, 'C': {}, 'Q': {}, 'Vs': {}, 'Qf': {}, 'blend': 1.0, 'out': {}, 'lo': 0, 'hi': 99,
           'stride': 1, 'step': 0, 'seen': set(), 'stats': {'built': 0, 'used': 0}}


_CTARX = {'scope': 'all', 'theta': 0, 'w': 5, 'publish_A': False}


def set_publish_A(on=True):
    """Analysis only: stash the re-anchored rows so the attention recorder can save
    the distribution CTAR actually used instead of the stale stored one."""
    _CTARX['publish_A'] = bool(on)
    _STITCH['pub_A'] = {}


def take_pub_A(layer_idx):
    return _STITCH.get('pub_A', {}).pop(layer_idx, None)


def set_ctarx(scope='all', theta=0, w=5):
    _CTARX.update(scope=str(scope), theta=int(theta), w=int(w))
    _STITCH['sel'] = None
    _STITCH['late'] = None
    _STITCH['mask_prev'] = None


def publish_mask(mask_vec):
    """mask_vec: bool [M], True = position still masked. Called once per decoding step
    after that step's commits. Maintains, per position, how many of its +-w neighbours
    have been committed since the position was last re-anchored, and selects the rows
    whose count has reached theta."""
    if _CTARX['theta'] <= 0 or mask_vec is None:
        _STITCH['sel'] = None
        return
    m = mask_vec.detach().reshape(-1)
    M = m.numel()
    late = _STITCH.get('late')
    if late is None or late.numel() != M:
        late = torch.zeros(M, device=m.device, dtype=torch.float32)
        _STITCH['late'] = late
        _STITCH['mask_prev'] = None
    prev = _STITCH.get('mask_prev')
    if prev is not None and prev.numel() == M:
        newly = (prev & (~m)).float()
        if bool(newly.any()):
            w = _CTARX['w']
            ker = torch.ones(1, 1, 2 * w + 1, device=m.device, dtype=torch.float32)
            late += torch.nn.functional.conv1d(newly.view(1, 1, -1), ker, padding=w).view(-1)
    _STITCH['mask_prev'] = m.clone()
    sel = (late >= float(_CTARX['theta'])) & m       # only positions still to be decided
    _STITCH['sel'] = sel
    late[sel] = 0.0
    st = _STITCH['stats']
    st['rows'] = st.get('rows', 0) + int(sel.sum())
    st['steps'] = st.get('steps', 0) + 1
    st['M'] = M


def set_stitch(on=True, M=128, lo=0, hi=99, stride=1, blend=1.0):
    _STITCH.update(on=bool(on), M=int(M), lo=int(lo), hi=int(hi),
                   stride=max(1, int(stride)), blend=float(blend))


def reset_stitch():
    _STITCH['A'].clear(); _STITCH['C'].clear(); _STITCH['out'].clear()
    _STITCH['Q'].clear(); _STITCH['Vs'].clear(); _STITCH['Qf'].clear()
    _STITCH['step'] = 0; _STITCH['seen'] = set()
    _STITCH['sel'] = None; _STITCH['late'] = None; _STITCH['mask_prev'] = None
    _STITCH['Pm'] = {}
    _STITCH['stats'] = {'built': 0, 'used': 0}


def take_stitch_out(layer_idx):
    return _STITCH['out'].pop(layer_idx, None)


def _apply_stitch(att_gen, layer_idx):
    """Write the re-anchored attention output back. A tuple payload carries only the
    rows CTAR selected this step; everything else keeps whatever the backend produced."""
    _st = take_stitch_out(layer_idx)
    if _st is None:
        return att_gen
    _bl = _STITCH['blend']
    if isinstance(_st, tuple):
        _new, _idx = _st
        _new = _new.to(att_gen.dtype)
        att_gen = att_gen.clone()
        if _bl >= 1.0:
            att_gen[:, _idx, :] = _new
        else:
            att_gen[:, _idx, :] = (1.0 - _bl) * att_gen[:, _idx, :] + _bl * _new
        _STITCH['stats']['used'] += 1
        return att_gen
    if att_gen.shape[1] == _st.shape[1]:
        _new = _st.to(att_gen.dtype)
        att_gen = _new if _bl >= 1.0 else (1.0 - _bl) * att_gen + _bl * _new
        _STITCH['stats']['used'] += 1
    return att_gen


def _stitch_step(self, attn_weights, v_repeated, q_index, layer_idx, in_band,
                 q=None, k_repeated=None):
    """Update the persistent row store, rebuild the full suffix matrix, apply the
    decay, and re-project outputs for all M suffix positions."""
    if not _STITCH['on'] or layer_idx is None:
        return
    # step bookkeeping: a layer revisit marks a new decoding step
    if layer_idx in _STITCH['seen']:
        _STITCH['step'] += 1
        _STITCH['seen'] = set()
    _STITCH['seen'].add(layer_idx)
    if not (_STITCH['lo'] <= layer_idx <= _STITCH['hi']):
        return
    if _STITCH['step'] % _STITCH['stride'] != 0:
        return
    M = _STITCH['M']
    B, H, S_q, S_k = attn_weights.shape
    if S_k < M or B != 1:
        return
    P = S_k - M
    dev = attn_weights.device
    if q_index is not None and q_index.numel() >= S_q:
        qi = q_index.reshape(-1)[:S_q].to(dev).long()
    else:
        qi = torch.arange(S_k - S_q, S_k, device=dev, dtype=torch.long)
    rel = qi - P
    keep = (rel >= 0) & (rel < M)
    if not bool(keep.any()):
        return
    rows = keep.nonzero(as_tuple=False).squeeze(-1)
    rr = rel[rows]

    d_head = v_repeated.shape[-1]
    if _CTAE['mode'].startswith('reroute'):
        A = None
    else:
        A = _STITCH['A'].get(layer_idx)
    if not _CTAE['mode'].startswith('reroute') and (A is None or A.shape != (H, M, M)):
        A = torch.zeros(H, M, M, dtype=attn_weights.dtype, device=dev)
        C = torch.zeros(H, M, d_head, dtype=attn_weights.dtype, device=dev)
        _STITCH['A'][layer_idx] = A
        _STITCH['C'][layer_idx] = C
    C = _STITCH['C'].get(layer_idx)

    mode = _CTAE['mode']
    if mode.startswith('reroute') and q is not None and k_repeated is not None:
        Q = _STITCH['Q'].get(layer_idx)
        if Q is None or Q.shape != (H, M, q.shape[-1]):
            Q = torch.zeros(H, M, q.shape[-1], dtype=q.dtype, device=dev)
            _STITCH['Q'][layer_idx] = Q
        Q[:, rr, :] = q[0][:, rows, :]
        Qf = _STITCH['Qf'].pop(layer_idx, None)
        if Qf is not None and Qf.shape == Q.shape:
            Q = Qf                       # query recomputed from the current hidden state
        scope = _CTARX['scope']
        if scope != 'all':               # keep the stored routing of one block
            A_st = _STITCH['A'].get(layer_idx)
            if A_st is None or A_st.shape != (H, M, M):
                A_st = torch.zeros(H, M, M, dtype=attn_weights.dtype, device=dev)
                _STITCH['A'][layer_idx] = A_st
            Pm = _STITCH.setdefault('Pm', {}).get(layer_idx)
            if Pm is None or Pm.shape != (H, M):
                Pm = torch.zeros(H, M, dtype=attn_weights.dtype, device=dev)
                _STITCH['Pm'][layer_idx] = Pm
            C_st = _STITCH['C'].get(layer_idx)
            if C_st is None or C_st.shape != (H, M, d_head):
                C_st = torch.zeros(H, M, d_head, dtype=attn_weights.dtype, device=dev)
                _STITCH['C'][layer_idx] = C_st
            _wr = attn_weights[0][:, rows, :]
            A_st[:, rr, :] = _wr[:, :, P:].to(A_st.dtype)
            if P > 0:
                Pm[:, rr] = _wr[:, :, :P].sum(-1).to(Pm.dtype)
                # autocast promotes this matmul to fp32; the store stays in the cache dtype
                C_st[:, rr, :] = torch.matmul(_wr[:, :, :P],
                                              v_repeated[0][:, :P, :]).to(C_st.dtype)
        sel = _STITCH.get('sel')
        sidx = None
        if sel is not None and sel.numel() == M:
            sidx = sel.nonzero(as_tuple=False).squeeze(-1)
            if sidx.numel() == 0:
                return                   # no position's anchors moved this step
            Q = Q[:, sidx, :]
        # routing refresh: stored queries against the CURRENT keys
        logits = torch.matmul(Q, k_repeated[0].transpose(1, 2)) / math.sqrt(self.head_dim)
        A_new = torch.softmax(logits.float(), dim=-1).to(attn_weights.dtype)   # [H, m, S_k]
        if scope == 'ctx' and P > 0:
            _A = A_st if sidx is None else A_st[:, sidx, :]
            m_f = A_new[:, :, :P].sum(-1)                        # fresh context mass
            s_st = _A.sum(-1)
            ok = (s_st > 1e-4)
            scale = torch.where(ok, (1.0 - m_f) / s_st.clamp(min=1e-4), torch.ones_like(s_st))
            sfx = torch.where(ok.unsqueeze(-1), _A * scale.unsqueeze(-1), A_new[:, :, P:])
            A_new = torch.cat([A_new[:, :, :P], sfx], dim=-1)
        elif scope == 'sfx' and P > 0:
            # mirror control: fresh intra-suffix routing, stored context contribution
            _Cs = C_st if sidx is None else C_st[:, sidx, :]
            _Pm = Pm if sidx is None else Pm[:, sidx]
            m_f = A_new[:, :, :P].sum(-1)                        # mass the fresh row gives context
            ok = (_Pm > 1e-4)
            _sfx_scale = torch.where(ok, m_f / _Pm.clamp(min=1e-4), torch.zeros_like(_Pm))
            _sfx_ctx = _Cs * _sfx_scale.unsqueeze(-1)
            _sfx_fallback = torch.matmul(A_new[:, :, :P], v_repeated[0][:, :P, :])
            _sfx_ctx = torch.where(ok.unsqueeze(-1), _sfx_ctx, _sfx_fallback)
        if mode == 'reroute_vold':
            Vs = _STITCH['Vs'].get(layer_idx)
            if Vs is None or Vs.shape != v_repeated[0].shape:
                Vs = v_repeated[0].clone(); _STITCH['Vs'][layer_idx] = Vs
            Vuse = Vs
            Vs[:, P:, :][:, rr, :] = v_repeated[0][:, P:, :][:, rr, :]   # only refreshed rows age out
        else:
            Vuse = v_repeated[0]
        if in_band and mode == 'reroute_shape':
            A_sfx = _f1_correct(A_new[:, :, P:], torch.arange(M, device=dev), 'sharpen')
            A_new = torch.cat([A_new[:, :, :P], A_sfx], dim=-1)
        if _ANCHOR['on'] and _ANCHOR['W'] is not None and _ANCHOR['lo'] <= layer_idx <= _ANCHOR['hi']:
            # analysis only: the meter must report the routing CTAR actually serves for these rows
            _rows_m = sidx if sidx is not None else torch.arange(M, device=A_new.device)
            _band_m = ((_rows_m[:, None] - torch.arange(M, device=A_new.device)[None, :]).abs()
                       <= _ANCHOR['half_w']).float()
            _ANCHOR['W'][layer_idx, _rows_m] = (A_new[:, :, P:].float().mean(0) * _band_m).sum(-1)
            if _ANCHOR.get('pair') is not None and _ANCHOR.get('Wm') is not None:
                # same rows, same step: the routing the cache has stored for a row (as the last real
                # forward left it) against the routing CTAR re-forms. Rows recomputed in this very
                # forward are left out, their stored routing is already current.
                _pr = _ANCHOR['pair']
                _fresh = torch.zeros(M, dtype=torch.bool, device=A_new.device); _fresh[rr] = True
                _kp = ~_fresh[_rows_m]
                _newm = (A_new[:, :, P:].float().mean(0) * _band_m).sum(-1)
                _pr['old'][_rows_m[_kp]] += _ANCHOR['Wm'][layer_idx, _rows_m[_kp]]
                _pr['new'][_rows_m[_kp]] += _newm[_kp]
                _pr['n'][_rows_m[_kp]] += 1
        if _CTARX['publish_A']:
            _STITCH.setdefault('pub_A', {})[layer_idx] = (
                A_new[:, :, P:].detach().float().cpu(),
                None if sidx is None else sidx.detach().cpu())
        if scope == 'sfx' and P > 0:
            out_h = _sfx_ctx + torch.matmul(A_new[:, :, P:], Vuse[:, P:, :])
        else:
            out_h = torch.matmul(A_new, Vuse)                   # [H, m, d_head]
        _n = out_h.shape[1]
        out = out_h.transpose(0, 1).contiguous().view(1, _n, H * out_h.shape[-1])
        out = self.o_proj(out)
        _STITCH['out'][layer_idx] = out if sidx is None else (out, sidx)
        _STITCH['stats']['built'] += 1
        return

    w_rows = attn_weights[0][:, rows, :]                       # [H, R, S_k]
    A[:, rr, :] = w_rows[:, :, P:]                             # suffix block
    if P > 0:
        C[:, rr, :] = torch.matmul(w_rows[:, :, :P], v_repeated[0][:, :P, :])
    else:
        C[:, rr, :] = 0

    A_eff = A
    if in_band and _CTAE['mode'].startswith('stitch_') and _CTAE['mode'] != 'stitch_id':
        _rel_all = torch.arange(M, device=dev)
        A_eff = _f1_correct(A, _rel_all, _CTAE['mode'][len('stitch_'):])
    elif in_band and _CTAE['mode'] == 'stitch':
        g = _ctae_stitch_gain(M, dev, A.dtype, _CTAE['sigma'], _CTAE['gamma'])
        A_eff = A * g                                          # broadcast [M, M]
    out_h = C + torch.matmul(A_eff, v_repeated[0][:, P:, :])   # [H, M, d_head]
    out = out_h.transpose(0, 1).contiguous().view(1, M, H * d_head)
    _STITCH['out'][layer_idx] = self.o_proj(out)
    _STITCH['stats']['built'] += 1


_STITCH_GAIN = {}


def _ctae_stitch_gain(M, dev, dt, sigma, gamma):
    key = (M, str(dev), str(dt), sigma, gamma)
    g = _STITCH_GAIN.get(key)
    if g is None:
        idx = torch.arange(M, device=dev, dtype=torch.float32)
        d = (idx[:, None] - idx[None, :]).abs()
        g = (gamma + (1.0 - gamma) * torch.exp(-(d / sigma) ** 2)).to(dt)
        _STITCH_GAIN[key] = g
    return g
# =============================================================================


# === F1-matched corrections: targeted, calibrated, mass-conserving ==========
_F1 = {'ref_q': 0.5, 'cap': 0.10, 'temp': 1.3, 'half_w': 5}


def set_f1(ref_q=0.5, cap=0.10, temp=1.3, half_w=5):
    _F1.update(ref_q=float(ref_q), cap=float(cap), temp=float(temp), half_w=int(half_w))


def _f1_correct(A_sub, rel, mode):
    if mode == 'restore_sharpen':
        return _f1_one(_f1_one(A_sub, rel, 'restore'), rel, 'sharpen')
    return _f1_one(A_sub, rel, mode)


def _f1_one(A_sub, rel, mode):
    """A_sub [H, R, M] post-softmax weights on suffix columns; rel [R] positions."""
    H, R, M = A_sub.shape
    if R == 0:
        return A_sub
    dev, dt = A_sub.device, A_sub.dtype
    A = A_sub.float()
    S = A.sum(-1, keepdim=True).clamp_min(1e-6)
    q = A / S                                            # suffix-normalised row
    if mode == 'restore':
        c = torch.arange(M, device=dev)
        win = ((rel[:, None] - c[None, :]).abs() <= _F1['half_w']).unsqueeze(0)
        rho = (q * win).sum(-1, keepdim=True)            # [H, R, 1] local share
        if R > 1:
            ref = torch.quantile(rho, _F1['ref_q'], dim=1, keepdim=True)
        else:
            ref = rho
        tgt = torch.minimum(torch.maximum(rho, ref), rho + _F1['cap']).clamp(1e-4, 0.98)
        s_in = tgt / rho.clamp_min(1e-6)
        s_out = (1.0 - tgt) / (1.0 - rho).clamp_min(1e-6)
        q = q * torch.where(win, s_in, s_out)
    elif mode == 'sharpen':
        lq = q.clamp_min(1e-9)
        ent = -(lq * lq.log()).sum(-1, keepdim=True)
        if R > 1:
            ref = torch.quantile(ent, _F1['ref_q'], dim=1, keepdim=True)
        else:
            ref = ent
        expo = 1.0 + (_F1['temp'] - 1.0) * (ent > ref).float()
        q = lq.pow(expo)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
    return (q * S).to(dt)
# =============================================================================


def _ctae_full_gain(attn_weights, q_index, M, sigma, gamma):
    """Gain over the FULL key axis for every suffix query row of this call.

    Same absolute-position bookkeeping as _ctae_suffix_gain, but the decay is
    applied across all keys (visual prefix + prompt + generated suffix), i.e.
    over the whole attention matrix the step actually computes, rather than the
    suffix-suffix sub-block only.
    """
    B, H, S_q, S_k = attn_weights.shape
    if S_k < M:
        return None
    P = S_k - M
    dev, dt = attn_weights.device, attn_weights.dtype
    if q_index is not None and q_index.numel() >= S_q:
        qi = q_index.reshape(-1)[:S_q].to(dev).long()
    else:
        qi = torch.arange(S_k - S_q, S_k, device=dev, dtype=torch.long)
    rel = qi - P
    keep = (rel >= 0) & (rel < M)
    if not bool(keep.any()):
        return None
    rows = keep.nonzero(as_tuple=False).squeeze(-1)
    rq = qi[rows].to(dt)                                   # absolute q positions
    ck = torch.arange(S_k, device=dev, dtype=dt)           # absolute k positions
    dist = (rq[:, None] - ck[None, :]).abs()
    gain = gamma + (1.0 - gamma) * torch.exp(-(dist / sigma) ** 2)
    return rows, gain.to(dt)


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

        setattr(tf_block.self_attn, "layer_idx", layer_index)
        setattr(tf_block.self_attn, "_old_forward_main", tf_block.self_attn.forward) 
        tf_block.self_attn.attention_forward_for_cache = types.MethodType(
            llada_attention_hook_for_cache, tf_block.self_attn
        )

        setattr(tf_block.self_attn.rotary_emb, "_old_forward", tf_block.self_attn.rotary_emb.forward)
        tf_block.self_attn.rotary_emb.forward = types.MethodType(
            llada_RoPe_forward_hook, tf_block.self_attn.rotary_emb
        )


def llada_attention_hook_for_cache(
    self, # self is LLaDAAttention instance
    q_in_proj: torch.Tensor, # Renamed from q to clarify it's post-projection from cache_hook
    k_in_proj: torch.Tensor, # Renamed from k
    v_in_proj: torch.Tensor, # Renamed from v
    attention_bias: Optional[torch.Tensor] = None, 
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
    use_cache: bool = False, 
    q_index: Optional[torch.Tensor] = None, 
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 
    
    # q_in_proj, k_in_proj, v_in_proj are (Batch, SeqLen_specific_to_them, HiddenDim_model)
    B, q_len_current_q = q_in_proj.shape[0], q_in_proj.shape[1] # q_len_current_q is the length of the Q for this specific call
    
    q_num_heads = self.num_heads
    k_num_heads = self.num_key_value_heads
    v_num_heads = self.num_key_value_heads
    head_dim = self.head_dim
    
    # Reshape q, k, v to (Batch, NumHeads, SeqLen, HeadDim)
    q = q_in_proj.view(B, q_len_current_q, q_num_heads, head_dim).transpose(1, 2)
    
    k_seq_len_current_k = k_in_proj.shape[1] # k_seq_len_current_k is the length of K for this call (e.g., full context)
    v_seq_len_current_v = v_in_proj.shape[1]

    k = k_in_proj.view(B, k_seq_len_current_k, k_num_heads, head_dim).transpose(1, 2)
    v = v_in_proj.view(B, v_seq_len_current_v, v_num_heads, head_dim).transpose(1, 2)
        
    if hasattr(self, 'rotary_emb'): 
        # q_index passed here should be for q_in_proj
        q, k = self.rotary_emb(q, k, q_index=q_index) 

    present = None 
    
    k_repeated = repeat_kv(k, self.num_key_value_groups) 
    v_repeated = repeat_kv(v, self.num_key_value_groups)

    # q is (B, q_num_heads, q_len_current_q, head_dim)
    # k_repeated is (B, q_num_heads, k_seq_len_current_k, head_dim)
    
    attn_weights = torch.matmul(q, k_repeated.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_bias is not None:
        bias_q_dim = attention_bias.shape[-2]
        bias_k_dim = attention_bias.shape[-1]

        sliced_attention_bias = attention_bias
        if q_len_current_q < bias_q_dim : # Q is a segment, assume it's the latest tokens
            # This assumes the q segment is the last part of the full query sequence represented by the bias
            sliced_attention_bias = attention_bias[:, :, -q_len_current_q:, :]
        
        if k_seq_len_current_k < bias_k_dim: # K is a segment (less likely for full KV cache but possible)
            # This assumes K is the latest part of the full key sequence in the bias
            sliced_attention_bias = sliced_attention_bias[:, :, :, -k_seq_len_current_k:]


        # Final check on dimensions before adding
        if attn_weights.shape[-2] == sliced_attention_bias.shape[-2] and \
           attn_weights.shape[-1] == sliced_attention_bias.shape[-1]:
            attn_weights = attn_weights + sliced_attention_bias
        else:
            if sliced_attention_bias.shape[1] == 1 and attn_weights.shape[1] == self.num_heads: # Mask has 1 head dim
                 attn_weights = attn_weights + sliced_attention_bias # Broadcast over heads
            elif sliced_attention_bias.shape[1] == self.num_heads: # Mask has same num heads
                 attn_weights = attn_weights + sliced_attention_bias
            else:
                raise RuntimeError(
                    f"Attention bias shape {sliced_attention_bias.shape} incompatible with "
                    f"attn_weights shape {attn_weights.shape} after slicing."
                )
    
    # === CTAE pre-softmax variants ===========================================
    _li = getattr(self, 'layer_idx', None)
    _in_band = (_li is None) or (_CTAE['lo'] <= _li <= _CTAE['hi'])
    if _in_band and _CTAE['mode'] == 'gated':
        _B, _H, _Sq, _Sk = attn_weights.shape
        _qs = 128 if _Sq > 3000 else _Sq
        _ks = 128 if _Sq > 3000 else _Sq
        _blk = attn_weights[:, :, _Sq - _qs:, _Sk - _ks:]
        _gain = _ctae_gain(max(_qs, _ks), _blk.device, _blk.dtype,
                           _CTAE['sigma'], _CTAE['gamma'])[:_qs, :_ks]
        _within = (_blk >= _blk.median()) & (_blk <= _CTAE['upper'])
        attn_weights[:, :, _Sq - _qs:, _Sk - _ks:] = torch.where(
            _within, _blk * _gain.unsqueeze(0).unsqueeze(0), _blk)
    elif _in_band and _CTAE['mode'] == 'aligned_bias':
        _g = _ctae_suffix_gain(attn_weights, q_index, _CTAE['block'],
                               _CTAE['sigma'], _CTAE['gamma'])
        if _g is not None:
            _rows, _P, _gain = _g
            attn_weights[:, :, _rows, _P:] = attn_weights[:, :, _rows, _P:] + torch.log(_gain).to(attn_weights.dtype)
    elif _in_band and _CTAE['mode'] == 'bias':
        _B, _H, _Sq, _Sk = attn_weights.shape
        _n = _CTAE['block']
        if _Sq == _Sk and _Sq > 3000:
            _gain = _ctae_gain(_n, attn_weights.device, attn_weights.dtype,
                               _CTAE['sigma'], _CTAE['gamma'])
            attn_weights[:, :, -_n:, -_n:] = attn_weights[:, :, -_n:, -_n:] + torch.log(_gain).unsqueeze(0).unsqueeze(0)
    # =========================================================================

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # === CTAE post-softmax variant ===========================================
    _anchor_probe(attn_weights, q_index, _ANCHOR['M'] or _CTAE['block'], _li)
    _stitch_step(self, attn_weights, v_repeated, q_index, _li, _in_band,
                 q=q, k_repeated=k_repeated)

    if _in_band and _CTAE['mode'] in ('restore', 'sharpen', 'restore_sharpen'):
        _g = _ctae_suffix_gain(attn_weights, q_index, _CTAE['block'], 1.0, 1.0)
        if _g is not None:
            _rows, _P, _ = _g
            _rel = (q_index.reshape(-1)[:attn_weights.shape[2]].to(attn_weights.device).long()[_rows] - _P
                    if q_index is not None and q_index.numel() >= attn_weights.shape[2]
                    else torch.arange(_rows.numel(), device=attn_weights.device))
            _sub = attn_weights[0][:, _rows, _P:]
            attn_weights[0, :, _rows, _P:] = _f1_correct(_sub, _rel, _CTAE['mode'])
            _CTAE_STATS['applied'] += 1
            _CTAE_STATS['rows'] += int(_rows.numel())

    if _in_band and _CTAE['mode'] == 'aligned_full':
        _g = _ctae_full_gain(attn_weights, q_index, _CTAE['block'],
                             _CTAE['sigma'], _CTAE['gamma'])
        if _g is not None:
            _rows, _gain = _g
            attn_weights[:, :, _rows, :] = attn_weights[:, :, _rows, :] * _gain
            _CTAE_STATS['applied'] += 1
            _CTAE_STATS['rows'] += int(_rows.numel())
        else:
            _CTAE_STATS['skipped'] += 1
    if _in_band and _CTAE['mode'] == 'aligned':
        _g = _ctae_suffix_gain(attn_weights, q_index, _CTAE['block'],
                               _CTAE['sigma'], _CTAE['gamma'])
        if _g is not None:
            _rows, _P, _gain = _g
            attn_weights[:, :, _rows, _P:] = attn_weights[:, :, _rows, _P:] * _gain.to(attn_weights.dtype)
            _CTAE_STATS['applied'] += 1
            _CTAE_STATS['rows'] += int(_rows.numel())
        else:
            _CTAE_STATS['skipped'] += 1
    if _in_band and _CTAE['mode'] == 'mult':
        _B, _H, _Sq, _Sk = attn_weights.shape
        _n = _CTAE['block']
        if _Sq == _Sk and _Sq > 3000:
            _gain = _ctae_gain(_n, attn_weights.device, attn_weights.dtype,
                               _CTAE['sigma'], _CTAE['gamma'])
            attn_weights[:, :, -_n:, -_n:] = attn_weights[:, :, -_n:, -_n:] * _gain.unsqueeze(0).unsqueeze(0)
    # =========================================================================

    att_output_heads = torch.matmul(attn_weights, v_repeated)

    # Reshape to (B, q_len_current_q, ModelHiddenDim)
    att_output_heads = att_output_heads.transpose(1, 2).contiguous().view(B, q_len_current_q, q_num_heads * head_dim) 
    
    output = self.o_proj(att_output_heads)
    
    return output, present


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


# === DAR: decode-aware refresh (F2) =========================================
# `score` is the decoder's own per-position selection score from the previous step
# (-inf at already-committed positions), published by generate_with_embeds.
_DAR = {'on': False, 'r': 4, 'w': 0, 'score': None, 'imminent': None,
        'r_anchor': 0, 'r_entropy': 0, 'entropy': None, 'reserved': None}


def set_dar(on=True, r=4, w=0, r_anchor=0, r_entropy=0):
    _DAR.update(on=bool(on), r=int(r), w=int(w),
                r_anchor=int(r_anchor), r_entropy=int(r_entropy))


def publish_entropy(ent_vec):
    """ent_vec: [M] local-context deep-layer entropy, higher = less consolidated."""
    _DAR['entropy'] = ent_vec


def publish_scores(score_vec):
    """score_vec: [M] tensor, the previous step's decoding scores.

    The top-r selection is resolved here, once per step, rather than inside the
    per-layer refresh path where it would be repeated for every layer.
    """
    _DAR['score'] = score_vec
    _DAR['imminent'] = None
    if _DAR['on'] and score_vec is not None and torch.isfinite(score_vec).any():
        r = min(_DAR['r'], score_vec.numel())
        idx = (torch.topk(torch.nan_to_num(score_vec.float(), neginf=-1e30), k=r).indices
               if r > 0 else torch.empty(0, dtype=torch.long, device=score_vec.device))
        w = _DAR['w']
        if w > 0:
            off = torch.arange(-w, w + 1, device=idx.device)
            idx = (idx[:, None] + off[None, :]).reshape(-1)
            idx = idx[(idx >= 0) & (idx < score_vec.numel())].unique()
        chans = [idx]
        # F1 channel: positions whose local anchoring has degraded most
        if _DAR['r_anchor'] > 0:
            w5 = get_anchor_w5()
            if w5 is not None and w5.numel() >= score_vec.numel():
                v = w5[:score_vec.numel()].to(idx.device).float().clone()
                v[~torch.isfinite(score_vec.to(v.device))] = float('inf')   # committed: skip
                chans.append(torch.topk(-v, k=_DAR['r_anchor']).indices)
        # F3 channel: positions whose local context has not consolidated
        if _DAR['r_entropy'] > 0 and _DAR['entropy'] is not None:
            e = _DAR['entropy']
            if e.numel() >= score_vec.numel():
                v = e[:score_vec.numel()].to(idx.device).float().clone()
                v[~torch.isfinite(score_vec.to(v.device))] = -float('inf')
                chans.append(torch.topk(v, k=_DAR['r_entropy']).indices)
        chans = [c for c in chans if c.numel() > 0]
        _DAR['imminent'] = torch.cat(chans).unique() if chans else None


def reset_dar():
    _DAR['score'] = None; _DAR['imminent'] = None


def refresh_index(
    new_features: torch.Tensor,
    cached_features: torch.Tensor = None,
    transfer_ratio: float = 0.5,
    layer_id: int = 0, 
) -> torch.Tensor:
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

    # --- DAR: reserve slots for the positions about to be committed -----------
    imminent = _DAR['imminent']
    if _DAR['on'] and imminent is not None and k_actual > imminent.numel():
        cos_sim = cos_sim.clone()
        cos_sim[0, imminent.to(cos_sim.device)] = -2.0   # below any cosine value
    # --------------------------------------------------------------------------
    transfer_index = torch.topk(cos_sim, largest=False, k=k_actual).indices
    if _CTEVC['refreshed'] is not None and transfer_index.numel():          # CTEVCACHE-PATCH
        _idx = transfer_index[0]
        _idx = _idx[_idx < _CTEVC['refreshed'].numel()]
        _CTEVC['refreshed'][_idx] = True
    return transfer_index


def llada_cache_hook_feature(
    self, 
    hidden_states: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None, # This is the original mask for the full layer input
    position_ids: Optional[torch.LongTensor] = None, 
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False, 
    cache_position: Optional[torch.LongTensor] = None, 
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, ...]]]:

    current_layer_idx = self.layer_idx
    feature_cache = dLLMCache()
    feature_cache.update_step(current_layer_idx)
    if current_layer_idx == 0 and _CTEVC['refreshed'] is not None:          # CTEVCACHE-PATCH: new forward
        _CTEVC['refreshed'].zero_(); _CTEVC['all'] = False
    
    prompt_length = feature_cache.prompt_length
    # x_prompt and x_gen are sub-segments of hidden_states
    x_prompt = hidden_states[:, :prompt_length, :]
    x_gen = hidden_states[:, prompt_length:, :] 
    
    if _CTAE['mode'] == 'reroute_q' and _STITCH['on'] and x_gen.shape[1] == (_STITCH['M'] or -1) \
            and _STITCH['lo'] <= current_layer_idx <= _STITCH['hi']:
        with torch.no_grad():
            _att = self.self_attn
            _qa = _att.q_proj(self.input_layernorm(x_gen))
            _B, _M, _ = _qa.shape
            _qa = _qa.view(_B, _M, _att.num_heads, _att.head_dim).transpose(1, 2)
            _kd = torch.zeros(_B, _att.num_key_value_heads, 1, _att.head_dim,
                              device=_qa.device, dtype=_qa.dtype)
            _pos = torch.arange(prompt_length, prompt_length + _M,
                                device=_qa.device).unsqueeze(0)
            _qa, _ = _att.rotary_emb(_qa, _kd, q_index=_pos)
            _STITCH['Qf'][current_layer_idx] = _qa[0]

    refresh_gen = feature_cache.refresh_gen(layer_id=current_layer_idx)
    refresh_prompt = feature_cache.refresh_prompt(layer_id=current_layer_idx)
    transfer_ratio = feature_cache.transfer_ratio
    
    bs, seq_len, dim = hidden_states.shape # seq_len is the length of hidden_states input to this layer
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
                              q_tensor_start_idx: int, # Start index of q_tensor within the original sequence
                              q_index: Optional[torch.Tensor] = None):
        
        q_len_for_this_call = q_tensor.shape[1]
        k_len_for_this_call = k_tensor.shape[1] # This K is already the full K from dLLM cache
        
        # Slice the original_full_attention_bias
        # original_full_attention_bias is likely (B, 1 or H, S_full_layer_input, K_full_from_dllm_or_max)
        # We need (B, 1 or H, q_len_for_this_call, k_len_for_this_call)
        sliced_bias = original_full_attention_bias
        if original_full_attention_bias is not None:
            # Slice query dimension based on q_tensor_start_idx and its length
            # Slice key dimension to match k_tensor's length (which is k_full from dLLM)
            sliced_bias = original_full_attention_bias[
                :, :, q_tensor_start_idx : q_tensor_start_idx + q_len_for_this_call, :k_len_for_this_call
            ]
            # Ensure num_heads dim is compatible (1 for broadcast, or matches self.self_attn.num_heads)
            if sliced_bias.shape[1] != 1 and sliced_bias.shape[1] != self.self_attn.num_heads:
                 # This might indicate an issue with mask preparation upstream if it's not 1 or num_heads
                 # For now, assume it's (B,1,q,k) and will broadcast if num_heads > 1 in attention_hook
                 pass


        att_output, _ = self.self_attn.attention_forward_for_cache(
            q_tensor,
            k_tensor,
            v_tensor,
            attention_bias=sliced_bias, 
            layer_past=None, 
            use_cache=False, 
            q_index=q_index,
        )
        return att_output
        
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
    if refresh_gen and _CTEVC['refreshed'] is not None:                       # CTEVCACHE-PATCH: full gen refresh
        _CTEVC['all'] = True

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
        att = call_attention_on_qkv(q_full, k_full, v_full, attention_mask, q_tensor_start_idx=0, q_index=position_ids)
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
            k_prompt_val = kv_cache_prompt.get("k", torch.empty(bs,0,dim,device=hidden_states.device))
            v_prompt_val = kv_cache_prompt.get("v", torch.empty(bs,0,dim,device=hidden_states.device))

            k_full_ctx = torch.cat([k_prompt_val, k_gen], dim=1)
            v_full_ctx = torch.cat([v_prompt_val, v_gen], dim=1)
            
            q_gen_pos_ids = position_ids[:, prompt_length:] if position_ids is not None and position_ids.shape[1] > prompt_length else None
            
            # q_gen starts at prompt_length in the original hidden_states sequence
            att_gen_part = call_attention_on_qkv(q_gen, k_full_ctx, v_full_ctx, attention_mask, 
                                                 q_tensor_start_idx=prompt_length, q_index=q_gen_pos_ids)
            
            att_gen_part = _apply_stitch(att_gen_part, current_layer_idx)
            feature_cache.set_cache(
                layer_id=current_layer_idx, feature_name="attn",
                features=att_gen_part, cache_type="gen"
            )
        
        att_prompt_cache = feature_cache.get_cache(
            layer_id=current_layer_idx, feature_name="attn", cache_type="prompt"
        )
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
        
        k_gen_current = kv_cache_gen.get("k", torch.empty(bs,0,dim, device=hidden_states.device))
        v_gen_current = kv_cache_gen.get("v", torch.empty(bs,0,dim, device=hidden_states.device))

        q_for_attn_segments = [q_prompt]
        q_idx_for_attn_segments = [position_ids[:, :prompt_length]] if position_ids is not None else [None]
        q_start_indices_segments = [0] # q_prompt starts at index 0 of hidden_states

        if transfer and x_gen.shape[1] > 0 and k_gen_current.shape[1] > 0:
            _, _, v_gen_for_transfer = project(x_gen) 
            index_from_attn_transfer = refresh_index(v_gen_for_transfer, v_gen_current, transfer_ratio, current_layer_idx)
            
            if index_from_attn_transfer.numel() > 0: 
                index_expanded_from_attn_transfer = index_from_attn_transfer.unsqueeze(-1).expand(-1, -1, dim)
                
                x_gen_normed_selected = torch.gather(self.input_layernorm(x_gen), dim=1, index=index_expanded_from_attn_transfer)
                q_gen_index = self.self_attn.q_proj(x_gen_normed_selected)
                k_gen_index = self.self_attn.k_proj(x_gen_normed_selected)
                v_gen_index_part = self.self_attn.v_proj(x_gen_normed_selected) 

                k_gen_current = k_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer, src=k_gen_index)
                v_gen_current = v_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer, src=v_gen_index_part) 

                feature_cache.set_cache( 
                    layer_id=current_layer_idx, feature_name="kv_cache",
                    features={"k": k_gen_current, "v": v_gen_current}, cache_type="gen"
                )
                
                q_for_attn_segments.append(q_gen_index)
                if position_ids is not None and position_ids.shape[1] > prompt_length:
                    gen_abs_positions_all = position_ids[:, prompt_length:]
                    gen_abs_positions_selected = torch.gather(gen_abs_positions_all, 1, index_from_attn_transfer)
                    q_idx_for_attn_segments.append(gen_abs_positions_selected)
                else:
                    q_idx_for_attn_segments.append(None)

        q_combined_for_attn = torch.cat(q_for_attn_segments, dim=1)
        q_idx_combined_for_rope = torch.cat(q_idx_for_attn_segments, dim=1) if all(s is not None for s in q_idx_for_attn_segments) else None


        k_full_ctx = torch.cat([k_prompt, k_gen_current], dim=1)
        v_full_ctx = torch.cat([v_prompt, v_gen_current], dim=1)
        
        # q_combined_for_attn effectively starts at index 0 of a conceptual sequence.
        # Its RoPE is handled by q_idx_combined_for_rope.
        att_for_q_combined = call_attention_on_qkv(q_combined_for_attn, k_full_ctx, v_full_ctx, 
                                                   attention_mask, q_tensor_start_idx=0, # Since Q is combined and starts from effective 0
                                                   q_index=q_idx_combined_for_rope)
        
        att_prompt_new = att_for_q_combined[:, :q_prompt.shape[1], :] 
        if transfer and index_from_attn_transfer is not None and index_from_attn_transfer.numel() > 0:
            att_gen_index_new = att_for_q_combined[:, q_prompt.shape[1]:, :] # Segment for transferred Qs
            if att_gen_cache.shape[1] > 0: 
                att_gen_cache = att_gen_cache.scatter(dim=1, index=index_expanded_from_attn_transfer, src=att_gen_index_new)
                feature_cache.set_cache(
                    layer_id=current_layer_idx, feature_name="attn",
                    features=att_gen_cache, cache_type="gen"
                )
        
        feature_cache.set_cache(
            layer_id=current_layer_idx, feature_name="attn",
            features=att_prompt_new, cache_type="prompt"
        )
        att = torch.cat([att_prompt_new, att_gen_cache], dim=1)

    else: # Not refresh gen, not refresh prompt
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
        
        k_gen_current = kv_cache_gen.get("k", torch.empty(bs,0,dim, device=hidden_states.device))
        v_gen_current = kv_cache_gen.get("v", torch.empty(bs,0,dim, device=hidden_states.device))
        k_prompt_val = kv_cache_prompt.get("k", torch.empty(bs,0,dim,device=hidden_states.device))
        v_prompt_val = kv_cache_prompt.get("v", torch.empty(bs,0,dim,device=hidden_states.device))
        
        if transfer and x_gen.shape[1] > 0 and k_gen_current.shape[1] > 0:
            x_gen_normed = self.input_layernorm(x_gen) 
            v_gen_for_transfer = self.self_attn.v_proj(x_gen_normed)
            index_from_attn_transfer = refresh_index(v_gen_for_transfer, v_gen_current, transfer_ratio, current_layer_idx)

            if index_from_attn_transfer.numel() > 0:
                index_expanded_from_attn_transfer = index_from_attn_transfer.unsqueeze(-1).expand(-1, -1, dim)
                
                x_gen_normed_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded_from_attn_transfer)
                q_gen_index_only = self.self_attn.q_proj(x_gen_normed_selected) # Q only for transferred items
                k_gen_index = self.self_attn.k_proj(x_gen_normed_selected)
                v_gen_index_part = self.self_attn.v_proj(x_gen_normed_selected)

                k_gen_current = k_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer, src=k_gen_index)
                v_gen_current = v_gen_current.scatter(dim=1, index=index_expanded_from_attn_transfer, src=v_gen_index_part)
                
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

                att_gen_index_new = call_attention_on_qkv(q_gen_index_only, k_full_ctx, v_full_ctx, attention_mask,
                                                          q_tensor_start_idx=prompt_length, # Approximate for mask slicing
                                                          q_index=q_idx_for_transferred_rope)
                
                # A full-M stitched output already covers the rows the backend just
                # recomputed, so the scatter is skipped only in that case; a row-selective
                # payload is written on top of the scattered result.
                _peek = _STITCH['out'].get(current_layer_idx)
                _full = (_peek is not None and not isinstance(_peek, tuple)
                         and att_gen_cache.shape[1] == _peek.shape[1])
                if not _full:
                    if att_gen_cache.shape[1] > 0: # Make sure att_gen_cache has a gen part
                        att_gen_cache = att_gen_cache.scatter(dim=1, index=index_expanded_from_attn_transfer, src=att_gen_index_new)
                    elif x_gen.shape[1] > 0: # If original att_gen_cache was for an empty gen part, but x_gen is not empty
                        att_gen_cache = torch.zeros((bs, x_gen.shape[1], dim), device=hidden_states.device, dtype=att_gen_index_new.dtype)
                        att_gen_cache = att_gen_cache.scatter(dim=1, index=index_expanded_from_attn_transfer, src=att_gen_index_new)
                    # Else: if x_gen.shape[1] is 0, att_gen_cache remains empty.
                att_gen_cache = _apply_stitch(att_gen_cache, current_layer_idx)

                feature_cache.set_cache(
                    layer_id=current_layer_idx, feature_name="attn",
                    features=att_gen_cache, cache_type="gen"
                )
        
        att = torch.cat([att_prompt_cache, att_gen_cache], dim=1)


    # ... rest of the llada_cache_hook_feature (MLP part) remains the same ...
    hidden_states_after_attn = residual_pre_attn + att 
    residual_pre_mlp = hidden_states_after_attn 
    
    x_prompt_mlp = hidden_states_after_attn[:, :prompt_length, :]
    x_gen_mlp = hidden_states_after_attn[:, prompt_length:, :] 

    mlp_out_prompt_part = torch.empty((bs, prompt_length, dim), device=hidden_states.device, dtype=hidden_states_after_attn.dtype)
    mlp_out_gen_part = torch.empty((bs, x_gen_mlp.shape[1], dim), device=hidden_states.device, dtype=hidden_states_after_attn.dtype)


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
        if x_gen_mlp.shape[1] > 0:
            mlp_out_gen_part = compute_mlp(x_gen_mlp)
            feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen")
        
        if mlp_out_gen_part.shape[1] > 0:
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part


    elif refresh_prompt and not refresh_gen:
        mlp_gen_cache_data = feature_cache.get_cache(current_layer_idx, "mlp", cache_type="gen")
        if x_gen_mlp.shape[1] > 0: 
            mlp_out_gen_part = mlp_gen_cache_data
        
        mlp_input_for_prompt_path = x_prompt_mlp
        # Use index_expanded_from_attn_transfer which was set in the attention block
        if transfer and index_expanded_from_attn_transfer is not None and index_expanded_from_attn_transfer.numel() > 0 and x_gen_mlp.shape[1] > 0 :
            x_gen_mlp_selected = torch.gather(x_gen_mlp, dim=1, index=index_expanded_from_attn_transfer) 
            mlp_input_for_prompt_path = torch.cat([x_prompt_mlp, x_gen_mlp_selected], dim=1)
        
        mlp_out_prompt_path_processed = compute_mlp(mlp_input_for_prompt_path)
        mlp_out_prompt_part = mlp_out_prompt_path_processed[:, :x_prompt_mlp.shape[1], :]

        if transfer and index_expanded_from_attn_transfer is not None and index_expanded_from_attn_transfer.numel() > 0 and x_gen_mlp.shape[1] > 0:
            mlp_gen_index_new = mlp_out_prompt_path_processed[:, x_prompt_mlp.shape[1]:, :]
            if mlp_out_gen_part.shape[1] > 0: # If gen part exists
                 mlp_out_gen_part = mlp_out_gen_part.scatter(dim=1, index=index_expanded_from_attn_transfer, src=mlp_gen_index_new)


            feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen")
        
        feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_prompt_part, cache_type="prompt")
        if mlp_out_gen_part.shape[1] > 0:
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part

        
    else: 
        mlp_out_prompt_part = feature_cache.get_cache(
            current_layer_idx, "mlp", cache_type="prompt"
        )
        mlp_gen_cache_data = feature_cache.get_cache(current_layer_idx, "mlp", cache_type="gen")
        if x_gen_mlp.shape[1] > 0:
            mlp_out_gen_part = mlp_gen_cache_data

        # Use index_expanded_from_attn_transfer
        if transfer and index_expanded_from_attn_transfer is not None and index_expanded_from_attn_transfer.numel() > 0 and x_gen_mlp.shape[1] > 0:
            x_gen_mlp_selected = torch.gather(x_gen_mlp, dim=1, index=index_expanded_from_attn_transfer)
            mlp_out_gen_index = compute_mlp(x_gen_mlp_selected)
            if mlp_out_gen_part.shape[1] > 0:
                 mlp_out_gen_part = mlp_out_gen_part.scatter(dim=1, index=index_expanded_from_attn_transfer, src=mlp_out_gen_index)

            feature_cache.set_cache(current_layer_idx, "mlp", mlp_out_gen_part, cache_type="gen")
        
        if mlp_out_gen_part.shape[1] > 0:
            mlp_out = torch.cat([mlp_out_prompt_part, mlp_out_gen_part], dim=1)
        else:
            mlp_out = mlp_out_prompt_part


    final_hidden_states = residual_pre_mlp + mlp_out
    
    returned_outputs = (final_hidden_states,)
    if output_attentions: 
        returned_outputs += (None,) 
    if use_cache: 
        returned_outputs += (None,) 

    return returned_outputs

