#!/usr/bin/env python
"""CTEV entropy cache (2026-10-01): space for time, exact under dLLM-Cache.

CTEV reads the deep-band entropy of the committed neighbours of every candidate. The released implementation
re-projects all gen positions through the lm_head at five layers at every step. Under dLLM-Cache the hidden state
of a position changes only when (i) the position is recomputed in this forward (the per-layer refresh set, or a
full refresh), or (ii) its token changed since the last forward (it was committed). Everything else is bit-for-bit
the cached state, so its entropy is unchanged. The patch keeps a per-position entropy store and recomputes only the
rows that are dirty and needed (committed rows for the context modes), which cuts the lm_head work per step from
5 x M rows to about 5 x (alpha x committed + newly committed) rows.

  generate(..., ctev_cache=1)              opt-in; default 0 = the released computation
  run_repeat_eval.py --ctev_cache 1        forwards it; summary.json gains 'peak_mem_gib'

Hook side: cache_hook_LLaDA_V accumulates the union over layers of the rows refreshed in the current forward
(ctevc_reset / ctevc_take). Anchored, refuses to run twice, backs up to patches/ctevcache_2026-10-01/.
"""
import io, os, shutil, sys, time

MODEL = "/data/zhaoqiyan/autodl-tmp/LLaDA-V/train/llava/model/language_model/modeling_llada.py"
HOOK = "/data/zhaoqiyan/autodl-tmp/LLaDA-V/train/llava/hooks/cache_hook_LLaDA_V.py"
RUNNER = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts/run_repeat_eval.py"
BAK = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/patches/ctevcache_2026-10-01"
TAG = "CTEVCACHE-PATCH"

# ------------------------------------------------------------------ hook
H1_OLD = """# === CTAE (TPAMI variant arbitration; default off = clean dLLM-Cache) ========
_CTAE_STATS = {'applied': 0, 'skipped': 0, 'rows': 0}
"""
H1_NEW = """# === CTEVCACHE-PATCH (2026-10-01): rows of the gen span recomputed in the current forward ===================
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
"""
H2_OLD = """    transfer_index = torch.topk(cos_sim, largest=False, k=k_actual).indices
    return transfer_index
"""
H2_NEW = """    transfer_index = torch.topk(cos_sim, largest=False, k=k_actual).indices
    if _CTEVC['refreshed'] is not None and transfer_index.numel():          # CTEVCACHE-PATCH
        _idx = transfer_index[0]
        _idx = _idx[_idx < _CTEVC['refreshed'].numel()]
        _CTEVC['refreshed'][_idx] = True
    return transfer_index
"""
H3_OLD = """    current_layer_idx = self.layer_idx
    feature_cache = dLLMCache()
    feature_cache.update_step(current_layer_idx)
"""
H3_NEW = """    current_layer_idx = self.layer_idx
    feature_cache = dLLMCache()
    feature_cache.update_step(current_layer_idx)
    if current_layer_idx == 0 and _CTEVC['refreshed'] is not None:          # CTEVCACHE-PATCH: new forward
        _CTEVC['refreshed'].zero_(); _CTEVC['all'] = False
"""
H4_OLD = """    residual_pre_attn = hidden_states

    if refresh_gen and refresh_prompt:
        q_full, k_full, v_full = project(hidden_states)
"""
H4_NEW = """    residual_pre_attn = hidden_states
    if refresh_gen and _CTEVC['refreshed'] is not None:                       # CTEVCACHE-PATCH: full gen refresh
        _CTEVC['all'] = True

    if refresh_gen and refresh_prompt:
        q_full, k_full, v_full = project(hidden_states)
"""

# ------------------------------------------------------------------ model
M1_OLD = """            return acc * (_CTEV_LOG2E / float(len(ctev_layers)))    # mean, bits
        # ======================================================================
"""
M1_NEW = """            return acc * (_CTEV_LOG2E / float(len(ctev_layers)))    # mean, bits

        # === CTEVCACHE-PATCH (2026-10-01): per-position entropy store, recomputed only where the state changed
        ctev_cache_on = ctev_on and int(kwargs.get('ctev_cache', 0)) > 0
        _ctevc = None
        if ctev_cache_on:
            try:
                from llava.hooks import cache_hook_LLaDA_V as _ctevc
                _ctevc.ctevc_reset(gen_length, inputs_embeds.device)
            except Exception:
                _ctevc = None
        _ctev_E = torch.zeros(gen_length, dtype=torch.float32, device=inputs_embeds.device)
        _ctev_valid = torch.zeros(gen_length, dtype=torch.bool, device=inputs_embeds.device)
        _ctev_prev_committed = torch.zeros(gen_length, dtype=torch.bool, device=inputs_embeds.device)
        _ctev_stats = {'rows_full': 0, 'rows_done': 0, 'steps': 0}

        def _ctev_entropy_rows(hidden_states_tuple, g0, rows):
            \"\"\"Entropy (bits) of the given gen rows (absolute positions g0 + rows), same readout as above.\"\"\"
            acc = None
            for li in ctev_layers:
                h = hidden_states_tuple[li][:1, g0 + rows, :]
                if ctev_norm:
                    h = self.model.norm(h)
                logp = F.log_softmax(self.lm_head(h).float(), dim=-1)
                ent = -(logp.exp() * logp).sum(dim=-1)
                acc = ent if acc is None else acc + ent
            return (acc * (_CTEV_LOG2E / float(len(ctev_layers))))[0]

        def _ctev_deep_entropy_cached(hidden_states_tuple, g0, g1, committed_bool):
            \"\"\"(1, gen_length) entropies; rows that are needed and dirty are recomputed, the rest come from the store.\"\"\"
            if _ctevc is not None:
                refreshed, allref = _ctevc.ctevc_take()
            else:
                refreshed, allref = None, True
            need = committed_bool if ctev_mode != 'self' else torch.ones_like(committed_bool)
            dirty = (~_ctev_valid) | (committed_bool & ~_ctev_prev_committed)
            if allref or refreshed is None:
                dirty = torch.ones_like(dirty)
            else:
                dirty = dirty | refreshed
            rows = (dirty & need).nonzero(as_tuple=False).view(-1)
            _ctev_stats['steps'] += 1; _ctev_stats['rows_full'] += int(need.sum()); _ctev_stats['rows_done'] += int(rows.numel())
            if rows.numel():
                _ctev_E[rows] = _ctev_entropy_rows(hidden_states_tuple, g0, rows).to(_ctev_E.dtype)
                _ctev_valid[rows] = True
            _ctev_prev_committed.copy_(committed_bool)
            return _ctev_E.unsqueeze(0)
        # ======================================================================
"""
M2_OLD = """                    if ctev_on:
                        g0 = inputs_embeds.shape[1]
                        g1 = g0 + gen_length
                        E_deep = _ctev_deep_entropy_bits(ctev_hidden, g0, g1)   # (1, gen_length)
"""
M2_NEW = """                    if ctev_on:
                        g0 = inputs_embeds.shape[1]
                        g1 = g0 + gen_length
                        if ctev_cache_on:                                        # CTEVCACHE-PATCH
                            E_deep = _ctev_deep_entropy_cached(ctev_hidden, g0, g1, ~mask_index[0, g0:g1])
                        else:
                            E_deep = _ctev_deep_entropy_bits(ctev_hidden, g0, g1)   # (1, gen_length)
"""
# expose the statistics on the model for the runner
M3_OLD = """            # Return the generated result, up to stop_position, and append the suffix
            if found_stop_seq:
"""
M3_NEW = """            if ctev_cache_on:                                                     # CTEVCACHE-PATCH
                self._ctev_cache_stats = dict(_ctev_stats)
                if _ctevc is not None:
                    _ctevc.ctevc_off()
            # Return the generated result, up to stop_position, and append the suffix
            if found_stop_seq:
"""

# ------------------------------------------------------------------ runner
R1_OLD = """    ap.add_argument('--decode_rows', default='', help='ROWS-PATCH: save decode-moment attention rows (layers 25-32) per image to this dir')
"""
R1_NEW = """    ap.add_argument('--decode_rows', default='', help='ROWS-PATCH: save decode-moment attention rows (layers 25-32) per image to this dir')
    ap.add_argument('--ctev_cache', type=int, default=0, help='CTEVCACHE-PATCH: 1 = recompute CTEV entropies only where the state changed')
"""
R2_OLD = """    if args.ngram > 0:
        _exp3_kw['ngram'] = args.ngram
"""
R2_NEW = """    if args.ngram > 0:
        _exp3_kw['ngram'] = args.ngram
    if args.ctev_cache > 0:
        _exp3_kw['ctev_cache'] = args.ctev_cache
"""
R3_OLD = """        'ctev': {'mode': args.ctev_mode, 'lambda': args.ctev_lambda,
                 'theta_bits': args.ctev_theta, 'window': args.ctev_window, 'norm_lens': args.ctev_norm, 'empty': args.ctev_empty,
"""
R3_NEW = """        'ctev': {'mode': args.ctev_mode, 'lambda': args.ctev_lambda, 'cache': args.ctev_cache,
                 'theta_bits': args.ctev_theta, 'window': args.ctev_window, 'norm_lens': args.ctev_norm, 'empty': args.ctev_empty,
"""
R4_OLD = """    summary = {'config': meta, 'aggregate': RM.aggregate(rows),
               'wall_seconds': round(time.time() - t0, 1),
               'end_time': time.strftime('%Y-%m-%d %H:%M:%S')}
"""
R4_NEW = """    summary = {'config': meta, 'aggregate': RM.aggregate(rows),
               'wall_seconds': round(time.time() - t0, 1),
               'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
               'peak_mem_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2),          # CTEVCACHE-PATCH
               'gen_seconds_mean': round(float(sum(r_['gen_seconds'] for r_ in _gen_rows) / max(len(_gen_rows), 1)), 2)}
    if getattr(model, '_ctev_cache_stats', None):
        summary['ctev_cache'] = dict(model._ctev_cache_stats)
"""
R5_OLD = """    out_jsonl = os.path.join(args.out, 'outputs.jsonl')
    rows = []
"""
R5_NEW = """    out_jsonl = os.path.join(args.out, 'outputs.jsonl')
    rows = []
    _gen_rows = []                                                                         # CTEVCACHE-PATCH
"""
R6_OLD = """            row = {'idx': k, 'image': fname, 'gen_seconds': round(gen_s, 2),
                   'text': text, 'ids': ids, **m}
            rows.append(m)
"""
R6_NEW = """            row = {'idx': k, 'image': fname, 'gen_seconds': round(gen_s, 2),
                   'text': text, 'ids': ids, **m}
            rows.append(m); _gen_rows.append(row)                                         # CTEVCACHE-PATCH
"""


import re


def _find(src, old):
    """Match the anchor with trailing whitespace allowed at the end of every line (the hook has such lines)."""
    pat = "".join(re.escape(l) + "[ \\t]*\\n" for l in old.rstrip("\n").split("\n"))
    return list(re.finditer(pat, src))


def patch(path, edits, check):
    src = io.open(path, encoding="utf-8").read()
    if TAG in src:
        sys.exit("%s already patched" % path)
    for old, new in edits:
        ms = _find(src, old)
        assert len(ms) == 1, "anchor not unique (%d) in %s:\n%s" % (len(ms), path, old)
    if check:
        print("anchors ok:", path); return
    os.makedirs(BAK, exist_ok=True)
    shutil.copy(path, os.path.join(BAK, os.path.basename(path) + ".orig"))
    for old, new in edits:
        m = _find(src, old)[0]
        src = src[:m.start()] + new + src[m.end():]
    io.open(path, "w", encoding="utf-8").write(src)
    shutil.copy(path, os.path.join(BAK, os.path.basename(path) + ".patched"))
    print("patched:", path)


if __name__ == "__main__":
    check = "--check" in sys.argv
    patch(HOOK, [(H1_OLD, H1_NEW), (H2_OLD, H2_NEW), (H3_OLD, H3_NEW), (H4_OLD, H4_NEW)], check)
    patch(MODEL, [(M1_OLD, M1_NEW), (M2_OLD, M2_NEW), (M3_OLD, M3_NEW)], check)
    patch(RUNNER, [(R1_OLD, R1_NEW), (R2_OLD, R2_NEW), (R3_OLD, R3_NEW), (R4_OLD, R4_NEW), (R5_OLD, R5_NEW), (R6_OLD, R6_NEW)], check)
    if not check:
        io.open(os.path.join(BAK, "APPLIED.txt"), "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " applied by apply_ctevcache_patch.py\n")
