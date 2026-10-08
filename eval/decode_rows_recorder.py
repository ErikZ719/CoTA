"""Decode-moment attention rows, sampler-agnostic (2026-09-30, for the Fig. 12 pairs on other model/backend pairs).

For every suffix position p, the attention row of p over the suffix keys (head mean, mean over the deep band of
layers) as it stood at the last forward pass in which p was still masked, i.e. the attention the model decoded p
against. Rows that a cached forward did not recompute keep their last computed values, so the matrix is the
effective attention, as in attention_recorder.py; the re-anchored rows of CTAR replace the stored ones, again as
in attention_recorder.py (set_publish_A / take_pub_A of the LLaDA-V hook; recomputed from ctar_apply's inputs
for the MMaDA/LaViDa port).

Two attach strategies:
  DecodeRowsLLaDAV   LLaDA-V under the dLLM-Cache hook, the SlowFast cache or the SlowFast CoTA++ port: the
                     per-layer `attention_forward_for_cache` is wrapped to publish (layer, q_index) and the
                     nn.functional.softmax it ends in is intercepted.
  DecodeRowsOLMo     LaViDa (OLMo-style LLaDA blocks) under dllm_cache.hooks.cache_hook_MMaDA, plain or through
                     mmada_cotapp: the hooked `block.attention(q, k, v, ...)` is wrapped and the probabilities
                     are recomputed explicitly (same math as mmada_cotapp.explicit_attention; SDPA in the hook).

    rec = DecodeRowsLLaDAV(model, core=model.model, layers_module=model.model.layers, G=512, band=(24, 31),
                           mask_id=126336, hook_mod=<hook module or None>)
    rec.attach();  out = model.generate(...);  rec.detach();  rec.save(path, final_ids)

Saved keys: rows [G, G] float16 (decode-moment rows, zero where the position never got a row), fwd [G] int32
(index of the forward the row comes from, -1 if none), decided_after_last, P, G, n_forwards, n_captures,
n_pub, band, ids (the final suffix ids the caller passes).
"""
import math
import threading
import types
import numpy as np
import torch
import torch.nn.functional as F

_ctx = threading.local()


class _DecodeRowsBase:
    def __init__(self, model, core, G, band=(24, 31), mask_id=126336, embeds_kw='inputs_embeds', embed_fn=None):
        self.model, self.core, self.G = model, core, int(G)
        self.lo, self.hi = int(band[0]), int(band[1])
        self.mask_id = int(mask_id)
        self.embeds_kw = embeds_kw
        self.embed_fn = embed_fn or (lambda ids: core.embed_tokens(ids))
        self.nband = self.hi - self.lo + 1
        self._handles = []
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        dev = next(self.model.parameters()).device
        self.buf = torch.zeros(self.nband, self.G, self.G, dtype=torch.float32, device=dev)
        self.rows = torch.zeros(self.G, self.G, dtype=torch.float32, device=dev)
        self.fwd = np.full(self.G, -1, dtype=np.int32)
        self.snapshot = None
        self.prev_masked = np.ones(self.G, dtype=bool)
        self.P = None
        self.n_forwards = self.n_captures = self.n_pub = 0
        self.mask_emb = None

    def _attach_state(self):
        self.reset()
        self._handles.append(self.core.register_forward_pre_hook(self._pre, with_kwargs=True))
        self._handles.append(self.core.register_forward_hook(self._post))

    def _detach_state(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ masked-state tracking
    def _masked_vec(self, emb):
        """bool [G]: True where the suffix position is masked or absent from this forward."""
        if self.mask_emb is None:
            self.mask_emb = self.embed_fn(torch.tensor([self.mask_id], device=emb.device)).to(emb.dtype).reshape(-1)
        S = emb.shape[1]
        if self.P is None:
            ism = (emb[0] - self.mask_emb).abs().max(-1).values < 1e-5      # [S]
            idx = ism.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                raise RuntimeError('DecodeRows: first forward has no masked position, cannot locate the prompt length')
            self.P = int(idx[0])
        m = np.ones(self.G, dtype=bool)
        present = min(S - self.P, self.G)
        if present > 0:
            ism = (emb[0, self.P:self.P + present] - self.mask_emb).abs().max(-1).values < 1e-5
            m[:present] = ism.cpu().numpy()
        return m

    def _pre(self, module, args, kwargs):
        emb = kwargs.get(self.embeds_kw)
        if emb is None:
            emb = args[0] if args and torch.is_tensor(args[0]) and args[0].dim() == 3 else None
        if emb is None or emb.dim() != 3:
            return
        masked = self._masked_vec(emb)
        new = self.prev_masked & ~masked
        if new.any() and self.snapshot is not None:
            idx = torch.from_numpy(np.nonzero(new)[0]).to(self.rows.device)
            self.rows[idx] = self.snapshot[idx]
            self.fwd[new] = self.n_forwards - 1
        self.prev_masked = masked

    def _post(self, module, args, output):
        self.snapshot = self.buf.mean(0)
        self.n_forwards += 1

    # ------------------------------------------------------------------ capture
    def _capture(self, li, A, q_index, pub=False):
        """A: [B, H, Q, K] probabilities over the keys [prompt | gen]; q_index: absolute query positions or None."""
        B, H, Q, K = A.shape
        P = self.P
        if P is None or K <= P or B != 1:
            return
        if q_index is not None and q_index.numel() >= Q:
            qi = q_index.reshape(-1)[:Q].to(A.device).long()
        elif Q == K:
            qi = torch.arange(K, device=A.device)
        else:
            qi = torch.arange(K - Q, K, device=A.device)
        rel = qi - P
        keep = (rel >= 0) & (rel < self.G)
        if not bool(keep.any()):
            return
        r = keep.nonzero(as_tuple=False).view(-1)
        cols = min(K - P, self.G)
        blk = A[0][:, r, P:P + cols].float().mean(0)                         # [R, cols]
        L = li - self.lo
        self.buf[L, rel[r]] = 0.0
        self.buf[L, rel[r], :cols] = blk
        if pub:
            self.n_pub += 1
        else:
            self.n_captures += 1

    # ------------------------------------------------------------------ output
    def save(self, path, final_ids):
        """final_ids: the suffix ids the sampler returned; a response cut at the stop token is shorter than G and
        the positions after the cut count as masked (no row)."""
        ids = np.asarray(final_ids, dtype=np.int64)
        if len(ids) >= self.G:
            ids = ids[-self.G:]
        else:
            ids = np.concatenate([ids, np.full(self.G - len(ids), self.mask_id, dtype=np.int64)])
        late = self.prev_masked & (ids != self.mask_id)
        if late.any() and self.snapshot is not None:
            idx = torch.from_numpy(np.nonzero(late)[0]).to(self.rows.device)
            self.rows[idx] = self.snapshot[idx]
            self.fwd[late] = self.n_forwards - 1
        np.savez_compressed(path, rows=self.rows.cpu().numpy().astype(np.float16), fwd=self.fwd,
                            decided_after_last=late, P=self.P if self.P is not None else -1, G=self.G,
                            n_forwards=self.n_forwards, n_captures=self.n_captures, n_pub=self.n_pub,
                            band=np.array([self.lo, self.hi]), ids=ids)
        return dict(P=self.P, n_forwards=self.n_forwards, n_captures=self.n_captures, n_pub=self.n_pub,
                    rows_filled=int((self.fwd >= 0).sum()))


class DecodeRowsLLaDAV(_DecodeRowsBase):
    """LLaDA-V: per-layer attention_forward_for_cache + softmax interception; CTAR rows via the hook's pub_A."""

    def __init__(self, model, core, layers_module, G, band=(24, 31), mask_id=126336, hook_mod=None):
        super().__init__(model, core, G, band, mask_id, embeds_kw='inputs_embeds')
        self.layers = layers_module
        self.hook_mod = hook_mod
        self._wrapped = []
        self._orig_softmax = None

    def attach(self):
        self._attach_state()
        rec = self
        for li, blk in enumerate(self.layers):
            sa = blk.self_attn
            fn = getattr(sa, 'attention_forward_for_cache', None)
            if fn is None:
                continue

            def make(li, fn):
                def wrapped(self_attn, *a, **kw):
                    qi = kw.get('q_index', a[6] if len(a) > 6 else None)
                    _ctx.layer_idx, _ctx.q_index = li, qi
                    try:
                        return fn(*a, **kw)
                    finally:
                        _ctx.layer_idx, _ctx.q_index = None, None
                        rec._pull_pub(li)
                return wrapped
            self._wrapped.append((sa, fn))
            sa.attention_forward_for_cache = types.MethodType(make(li, fn), sa)
        if not self._wrapped:
            raise RuntimeError('DecodeRows: no layer has attention_forward_for_cache (is a cache hook registered?)')
        self._orig_softmax = F.softmax
        orig = self._orig_softmax

        def capturing_softmax(input, dim=None, _stacklevel=3, dtype=None):
            out = orig(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
            li = getattr(_ctx, 'layer_idx', None)
            if li is not None and rec.lo <= li <= rec.hi and out.dim() == 4:
                rec._capture(li, out, getattr(_ctx, 'q_index', None))
            return out
        F.softmax = capturing_softmax
        torch.nn.functional.softmax = capturing_softmax
        if self.hook_mod is not None and hasattr(self.hook_mod, 'set_publish_A'):
            self.hook_mod.set_publish_A(True)

    def detach(self):
        for sa, fn in self._wrapped:
            sa.attention_forward_for_cache = fn
        self._wrapped = []
        if self._orig_softmax is not None:
            F.softmax = self._orig_softmax
            torch.nn.functional.softmax = self._orig_softmax
            self._orig_softmax = None
        if self.hook_mod is not None and hasattr(self.hook_mod, 'set_publish_A'):
            self.hook_mod.set_publish_A(False)
        self._detach_state()

    def _pull_pub(self, li):
        if self.hook_mod is None or not hasattr(self.hook_mod, 'take_pub_A') or not (self.lo <= li <= self.hi):
            return
        pa = self.hook_mod.take_pub_A(li)
        if pa is None or self.P is None:
            return
        A, sidx = pa                                                          # [H, m, M] cpu float, rows
        A = A.to(self.buf.device)
        m, M = A.shape[1], A.shape[2]
        rows = torch.arange(m, device=A.device) if sidx is None else sidx.to(A.device).long()
        ok = rows < self.G
        rows = rows[ok]
        if rows.numel() == 0:
            return
        cols = min(M, self.G)
        L = li - self.lo
        self.buf[L, rows] = 0.0
        self.buf[L, rows, :cols] = A[:, ok, :cols].mean(0)
        self.n_pub += 1


class DecodeRowsOLMo(_DecodeRowsBase):
    """LaViDa / MMaDA-style blocks under cache_hook_MMaDA: wrap block.attention and recompute the probabilities
    explicitly (q_norm/k_norm, RoPE with q_index, GQA repeat, softmax). With mmada_cotapp (mc), the CTAR rows are
    recomputed from ctar_apply's inputs (stored queries of the selected rows against the current keys)."""

    def __init__(self, model, core, blocks, G, band=(24, 31), mask_id=126336, mc=None):
        super().__init__(model, core, G, band, mask_id, embeds_kw='input_embeddings',
                         embed_fn=lambda ids: core.transformer.wte(ids))
        self.blocks, self.mc = blocks, mc
        self._wrapped = []
        self._orig_ctar = None

    def _probs(self, block, q, k, q_index):
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
        if block.config.rope:
            q, k = block.rotary_emb(q, k, q_index=q_index)
        if nkv != nh:
            k = k.repeat_interleave(nh // nkv, dim=1)
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(hd)
        if self.mc is not None:
            bias = self.mc.ctae_mask(ql, kl, q_index, torch.float32, q.device)
            if bias is not None:
                logits = logits + bias
        return torch.softmax(logits, dim=-1)                                  # [B, H, Q, K]

    def attach(self):
        self._attach_state()
        rec = self
        for blk in self.blocks:
            li = int(getattr(blk, 'layer_id'))
            if not (self.lo <= li <= self.hi):
                continue
            fn = blk.attention

            def make(li, blk, fn):
                def wrapped(self_blk, q, k, v, *a, **kw):
                    qi = kw.get('q_index', a[3] if len(a) > 3 else None)
                    with torch.no_grad():
                        rec._capture(li, rec._probs(blk, q, k, qi), qi)
                    return fn(q, k, v, *a, **kw)
                return wrapped
            self._wrapped.append((blk, fn))
            blk.attention = types.MethodType(make(li, blk, fn), blk)
        if not self._wrapped:
            raise RuntimeError('DecodeRows: no block in the band (is the cache hook registered, layer_id set?)')
        if self.mc is not None and hasattr(self.mc, 'ctar_apply'):
            self._orig_ctar = self.mc.ctar_apply
            orig = self._orig_ctar

            def ctar_apply(block, x_gen, prompt_length, k_full, v_full, att_gen_cache):
                li = int(block.layer_id)
                C, S = rec.mc.C, rec.mc.S
                if C['ctar'] and C['lo'] <= li <= C['hi'] and rec.lo <= li <= rec.hi and S.get('sel') is not None:
                    sidx = S['sel'].nonzero(as_tuple=False).squeeze(-1)
                    sidx = sidx[sidx < x_gen.shape[1]]
                    if sidx.numel():
                        with torch.no_grad():
                            q = block.q_proj(block.attn_norm(x_gen[:, sidx, :]))
                            qi = (sidx + prompt_length).unsqueeze(0)
                            rec._capture(li, rec._probs(block, q, k_full, qi), qi, pub=True)
                return orig(block, x_gen, prompt_length, k_full, v_full, att_gen_cache)
            self.mc.ctar_apply = ctar_apply

    def detach(self):
        for blk, fn in self._wrapped:
            blk.attention = fn
        self._wrapped = []
        if self._orig_ctar is not None:
            self.mc.ctar_apply = self._orig_ctar
            self._orig_ctar = None
        self._detach_state()
