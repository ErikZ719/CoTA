#!/usr/bin/env python
"""Three opt-in options for the CoTA++ main-text experiments (2026-09-30), replacing appendix items 10-12:

  (a) n-gram penalty baseline   generate(..., ngram=n): at every step, a masked position may not be filled with a
                                token that would complete an n-gram already present in the committed response
                                (the diffusion analogue of no_repeat_ngram_size; every window through the position
                                whose other n-1 tokens are committed is checked). Off unless ngram > 0.
  (b) block-length sweep        run_repeat_eval.py --block B  -> generate(block_length=B); steps stay L (one token
                                per step, B steps per block). Default 0 = single block as before.
  (c) CTEV layer window         run_repeat_eval.py --ctev_layers lo-hi -> generate(ctev_layers=range(lo, hi+1)),
                                i.e. hidden_states[lo..hi] = outputs of layers lo..hi (1-indexed). Default = the
                                existing (26..30).

Applies line-level, anchored replacements; refuses to run twice; keeps a backup next to the patch.
  python apply_exp3_patch.py            # apply
  python apply_exp3_patch.py --check    # only verify the anchors
"""
import io, os, shutil, sys, time

MODEL = "/data/zhaoqiyan/autodl-tmp/LLaDA-V/train/llava/model/language_model/modeling_llada.py"
RUNNER = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts/run_repeat_eval.py"
BAK = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/patches/exp3_2026-09-30"


def patch(path, edits, check):
    src = io.open(path, encoding="utf-8").read()
    if "EXP3-PATCH" in src:
        sys.exit("%s already patched" % path)
    for old, new in edits:
        assert src.count(old) == 1, "anchor not unique in %s:\n%s" % (path, old)
    if check:
        print("anchors ok:", path); return
    os.makedirs(BAK, exist_ok=True)
    shutil.copy(path, os.path.join(BAK, os.path.basename(path) + ".orig"))
    for old, new in edits:
        src = src.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(src)
    shutil.copy(path, os.path.join(BAK, os.path.basename(path) + ".patched"))
    print("patched:", path)


# ---------------------------------------------------------------- modeling_llada.py
M_OLD_1 = """                    for token_id in [126081, 126080, 126346, 126347]:
                        logits[:, :, token_id] = torch.where(mask_index, -float('inf'), logits[:, :, token_id])
"""
M_NEW_1 = """                    for token_id in [126081, 126080, 126346, 126347]:
                        logits[:, :, token_id] = torch.where(mask_index, -float('inf'), logits[:, :, token_id])

                    # === EXP3-PATCH (a): n-gram penalty baseline (opt-in, ngram > 0) ==============
                    # A masked position may not take a token that completes an n-gram already present in
                    # the committed response: for every n-window through the position whose other n-1
                    # tokens are committed, the tokens that would close a seen n-gram are set to -inf.
                    _ngram_n = int(kwargs.get('ngram', 0))
                    if _ngram_n > 0:
                        _g0n = inputs_embeds.shape[1]
                        _seqn = x[0, _g0n:_g0n + gen_length].tolist()
                        _comn = (~mask_index[0, _g0n:_g0n + gen_length]).tolist()
                        _ban = self._no_repeat_ngram_bans(_seqn, _comn, _ngram_n)
                        for _pn, _toks in _ban.items():
                            logits[0, _g0n + _pn, list(_toks)] = -float('inf')
                    # ==============================================================================
"""
M_OLD_2 = """    def generate_with_embeds(self, inputs_embeds, steps=128, gen_length=128, block_length=128, temperature=0.,
        cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=None, stopping_criteria=None, generation_suffix=None, **kwargs):
"""
M_NEW_2 = """    @staticmethod
    def _no_repeat_ngram_bans(seq, committed, n):
        \"\"\"EXP3-PATCH (a). seq: token ids of the response (any value at masked positions), committed: bool per
        position. Returns {masked position: set of banned token ids}. A token t is banned at p if some window
        [s, s+n) containing p, all of whose other positions are committed, would read as an n-gram that already
        occurs among the fully committed windows of seq.\"\"\"
        G = len(seq)
        seen = {}                                                   # (hole, prefix, suffix) -> {token}
        for s in range(G - n + 1):
            if all(committed[s:s + n]):
                g = tuple(seq[s:s + n])
                for h in range(n):
                    seen.setdefault((h, g[:h], g[h + 1:]), set()).add(g[h])
        bans = {}
        if not seen:
            return bans
        for p in range(G):
            if committed[p]:
                continue
            for s in range(max(0, p - n + 1), min(p, G - n) + 1):
                if all(committed[s:p]) and all(committed[p + 1:s + n]):
                    key = (p - s, tuple(seq[s:p]), tuple(seq[p + 1:s + n]))
                    if key in seen:
                        bans.setdefault(p, set()).update(seen[key])
        return bans

    def generate_with_embeds(self, inputs_embeds, steps=128, gen_length=128, block_length=128, temperature=0.,
        cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=None, stopping_criteria=None, generation_suffix=None, **kwargs):
"""

# ---------------------------------------------------------------- run_repeat_eval.py
R_OLD_1 = """    ap.add_argument('--steps', type=int, default=0, help='denoising steps; 0 = one per token')
"""
R_NEW_1 = """    ap.add_argument('--steps', type=int, default=0, help='denoising steps; 0 = one per token')
    # EXP3-PATCH (2026-09-30): block-length sweep, CTEV layer window, n-gram baseline (all opt-in)
    ap.add_argument('--block', type=int, default=0, help='block length for semi-AR decoding; 0 = single block (= L)')
    ap.add_argument('--ctev_layers', default='', help='CTEV layer window lo-hi (1-indexed, inclusive); empty = 26-30')
    ap.add_argument('--ngram', type=int, default=0, help='n-gram penalty baseline: forbid completing a seen n-gram; 0 = off')
"""
R_OLD_2 = """    meta = {
        'protocol': {'steps': args.steps or args.length, 'dar_mode': args.dar_mode, 'gen_length': args.length,
                     'block_length': args.length, 'prompt': args.prompt,
"""
R_NEW_2 = """    # EXP3-PATCH: block length and CTEV layer window
    block_length = args.block or args.length
    assert args.length % block_length == 0, 'L must be a multiple of the block length'
    ctev_layers = None
    if args.ctev_layers:
        _lo, _hi = [int(v) for v in args.ctev_layers.split('-')]
        assert 1 <= _lo <= _hi <= 32, 'ctev_layers must be lo-hi within 1-32'
        ctev_layers = tuple(range(_lo, _hi + 1))
    _exp3_kw = {}
    if ctev_layers is not None:
        _exp3_kw['ctev_layers'] = ctev_layers
    if args.ngram > 0:
        _exp3_kw['ngram'] = args.ngram

    meta = {
        'protocol': {'steps': args.steps or args.length, 'dar_mode': args.dar_mode, 'gen_length': args.length,
                     'block_length': block_length, 'prompt': args.prompt,
                     'ngram': args.ngram,
"""
R_OLD_3 = """        'ctev': {'mode': args.ctev_mode, 'lambda': args.ctev_lambda,
                 'theta_bits': args.ctev_theta, 'window': args.ctev_window, 'norm_lens': args.ctev_norm, 'empty': args.ctev_empty},
"""
R_NEW_3 = """        'ctev': {'mode': args.ctev_mode, 'lambda': args.ctev_lambda,
                 'theta_bits': args.ctev_theta, 'window': args.ctev_window, 'norm_lens': args.ctev_norm, 'empty': args.ctev_empty,
                 'layers': list(ctev_layers) if ctev_layers is not None else [26, 27, 28, 29, 30]},
"""
R_OLD_4 = """                cont = model.generate(
                    input_ids, images=image_tensor, image_sizes=[img.size],
                    steps=args.steps or args.length, gen_length=args.length,
                    block_length=args.length, tokenizer=tokenizer,
"""
R_NEW_4 = """                cont = model.generate(
                    input_ids, images=image_tensor, image_sizes=[img.size],
                    steps=args.steps or args.length, gen_length=args.length,
                    block_length=block_length, tokenizer=tokenizer,
                    **_exp3_kw,
"""

if __name__ == "__main__":
    check = "--check" in sys.argv
    patch(MODEL, [(M_OLD_2, M_NEW_2), (M_OLD_1, M_NEW_1)], check)
    patch(RUNNER, [(R_OLD_1, R_NEW_1), (R_OLD_2, R_NEW_2), (R_OLD_3, R_NEW_3), (R_OLD_4, R_NEW_4)], check)
    if not check:
        io.open(os.path.join(BAK, "APPLIED.txt"), "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " applied by apply_exp3_patch.py\n")
