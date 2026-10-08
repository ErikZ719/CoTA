"""Fidelity probe: does CTAR's re-derived routing reproduce the model's own attention
distribution on rows the backend actually recomputes this step?

For rows in the refresh set we have ground truth (`attn_weights`) and CTAR's
reconstruction (`A_new`) side by side inside _stitch_step. If the re-derivation
(q_proj . LN(h) + RoPE) is exact, the two must agree to numerical precision.
A systematic gap would mean CTAR has been running on an approximate query all along.
"""
import os, sys, json, math, types
import torch
sys.path.insert(0, '/data/zhaoqiyan/autodl-tmp/LLaDA-V/train')
sys.path.insert(0, '/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts')
import llava.hooks.cache_hook_LLaDA_V as HK

STATS = []
_orig = HK._stitch_step

def patched(self, attn_weights, v_repeated, q_index, layer_idx, in_band, q=None, k_repeated=None):
    if (HK._STITCH['on'] and layer_idx is not None
            and HK._STITCH['lo'] <= layer_idx <= HK._STITCH['hi']
            and HK._CTAE['mode'].startswith('reroute')
            and q is not None and k_repeated is not None):
        M = HK._STITCH['M']; B, H, S_q, S_k = attn_weights.shape
        if S_k >= M and B == 1:
            P = S_k - M
            dev = attn_weights.device
            if q_index is not None and q_index.numel() >= S_q:
                qi = q_index.reshape(-1)[:S_q].to(dev).long()
            else:
                qi = torch.arange(S_k - S_q, S_k, device=dev, dtype=torch.long)
            rel = qi - P
            keep = (rel >= 0) & (rel < M)
            if bool(keep.any()):
                rows = keep.nonzero(as_tuple=False).squeeze(-1)
                rr = rel[rows]
                Qf = HK._STITCH['Qf'].get(layer_idx)
                if Qf is not None:
                    logits = torch.matmul(Qf[:, rr, :], k_repeated[0].transpose(1, 2)) / math.sqrt(self.head_dim)
                    A_ctar = torch.softmax(logits.float(), dim=-1)
                    A_true = attn_weights[0][:, rows, :].float()
                    cos = torch.nn.functional.cosine_similarity(A_ctar, A_true, dim=-1)
                    l1 = (A_ctar - A_true).abs().sum(-1)
                    # how much mass sits on the prefix (context) block, ctar vs true
                    mp_c = A_ctar[:, :, :P].sum(-1); mp_t = A_true[:, :, :P].sum(-1)
                    STATS.append((layer_idx, float(cos.mean()), float(l1.mean()),
                                  float(mp_c.mean()), float(mp_t.mean()), int(rows.numel())))
    return _orig(self, attn_weights, v_repeated, q_index, layer_idx, in_band, q=q, k_repeated=k_repeated)

HK._stitch_step = patched

import run_repeat_eval as RR
sys.argv = ['probe', '--out', '/tmp/probe_out', '--length', '128', '--limit', '3', '--images', '/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/data/coco500_final.json',
            '--mode', 'dllm_cache', '--ctae_mode', 'reroute_q',
            '--stitch_lo', '24', '--stitch_hi', '31', '--device', 'cuda:0']
try:
    RR.main()
except SystemExit:
    pass

import statistics as st
print('\n===== FIDELITY PROBE =====')
print(f'records: {len(STATS)}')
if STATS:
    cos = [s[1] for s in STATS]; l1 = [s[2] for s in STATS]
    print(f'cos(A_ctar, A_true): mean={st.mean(cos):.6f}  min={min(cos):.6f}  p05={sorted(cos)[len(cos)//20]:.6f}')
    print(f'L1 distance        : mean={st.mean(l1):.6f}  max={max(l1):.6f}')
    print(f'prefix mass  ctar={st.mean([s[3] for s in STATS]):.4f}  true={st.mean([s[4] for s in STATS]):.4f}')
    bylayer = {}
    for s in STATS: bylayer.setdefault(s[0], []).append(s[1])
    for L in sorted(bylayer):
        print(f'  layer {L}: cos={st.mean(bylayer[L]):.6f}  n={len(bylayer[L])}')
