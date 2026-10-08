#!/usr/bin/env python
"""Phase 0 driver: Repeat-Curse evaluation of LLaDA-V over a fixed COCO image list.

Protocol (user-confirmed 2026-09-04):
  steps = gen_length = block_length = L  (single block, one token per step)
  prompt: "Please describe the image in detail."  (llava_llada template)
  batch size 1; greedy confidence-based remasking (deterministic)

Modes:
  baseline    no hook registered -> pristine upstream code path
  dllm_cache  dLLM-Cache with prompt_interval=25, gen_interval=7, ratio=0.25

Minimal-invasion: this driver only *calls* the repo, registers no hook in
baseline mode, and never edits repo files.

Outputs under --out:
  outputs.jsonl   one line per image: ids, text, per-sample metrics
  summary.json    aggregate metrics + full config echo
  run_meta.json   environment fingerprint (written at start)
"""
import argparse, copy, json, os, sys, time, warnings

warnings.filterwarnings('ignore')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/data/coco500_seed42.json')
    ap.add_argument('--length', type=int, required=True, help='L = steps = gen_length = block_length')
    ap.add_argument('--mode', choices=['baseline', 'dllm_cache', 'slowfast', 'slowfast_nocache', 'fast_dllm'], required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='debug: stop after N images')
    ap.add_argument('--offset', type=int, default=0, help='shard: start at image N')
    ap.add_argument('--prompt', default='Please describe the image in detail.')
    ap.add_argument('--cache_pi', type=int, default=25, help='prompt_interval_steps')
    ap.add_argument('--cache_gi', type=int, default=7, help='gen_interval_steps (E_s)')
    ap.add_argument('--cache_tr', type=float, default=0.25, help='transfer_ratio (alpha)')
    ap.add_argument('--ctev_mode', default='off', choices=['off', 'self', 'ctx', 'ctx_gated'])
    ap.add_argument('--ctev_lambda', type=float, default=0.25)
    ap.add_argument('--ctev_theta', type=float, default=12.5, help='gate threshold, bits')
    ap.add_argument('--ctev_window', type=int, default=5)
    ap.add_argument('--ctev_norm', type=int, default=1, choices=[0, 1])
    ap.add_argument('--ctev_empty', default='zero', choices=['zero', 'mean', 'max'],
                    help='context entropy of a position without committed neighbours (zero = released behaviour)')
    ap.add_argument('--ctae_mode', default='off', choices=['off', 'gated', 'bias', 'mult', 'aligned', 'aligned_bias', 'aligned_full', 'stitch', 'stitch_id', 'restore', 'sharpen', 'restore_sharpen',
                             'stitch_restore', 'stitch_sharpen', 'stitch_restore_sharpen',
                             'reroute', 'reroute_vold', 'reroute_shape', 'reroute_q'])
    ap.add_argument('--ctae_sigma', type=float, default=2.0)
    ap.add_argument('--ctae_gamma', type=float, default=0.75)
    ap.add_argument('--ctae_lo', type=int, default=0, help='first layer the decay applies to')
    ap.add_argument('--ctae_hi', type=int, default=99, help='last layer (inclusive)')
    ap.add_argument('--f1_ref_q', type=float, default=0.5, help='reference quantile for the healthy row')
    ap.add_argument('--f1_cap', type=float, default=0.10, help='max local-share repair per row')
    ap.add_argument('--f1_temp', type=float, default=1.3, help='sharpening exponent for flagged rows')
    ap.add_argument('--f1_w', type=int, default=5, help='half-width of the local window')
    ap.add_argument('--stitch_lo', type=int, default=0, help='first layer that re-projects')
    ap.add_argument('--stitch_hi', type=int, default=99, help='last layer that re-projects')
    ap.add_argument('--stitch_stride', type=int, default=1, help='re-project every k-th step')
    ap.add_argument('--stitch_blend', type=float, default=1.0, help='mix with the cached output')
    ap.add_argument('--ctar_scope', default='all', choices=['all', 'ctx', 'sfx'],
                    help="which block of the row is re-anchored: whole row / context only / suffix only")
    ap.add_argument('--ctar_theta', type=int, default=0,
                    help='0 = re-anchor every position every step; k = only once k of its '
                         '+-w neighbours have been committed since it was last re-anchored')
    ap.add_argument('--ctar_w', type=int, default=5, help='half-width of the neighbourhood')
    ap.add_argument('--dar_r', type=int, default=0, help='0 = DAR off; else reserved slots')
    ap.add_argument('--dar_mode', default='legacy', choices=['legacy', 'masked', 'score'])
    ap.add_argument('--case_trace', type=int, default=0,
                    help='analysis only: log per commit the anchoring, the age of the cached state and the context entropy')
    ap.add_argument('--sf_thresh', choices=['score', 'raw'], default='score',
                    help="slowfast only: 'raw' keeps the backend's high-confidence threshold on the raw confidence")
    ap.add_argument('--sf_repguard', type=float, default=0.0,
                    help='slowfast only: extra CTEV penalty proportional to the local density of committed runs')
    ap.add_argument('--sf_port', type=int, default=0, help='slowfast: use the CoTA++ port even with no component (identity test)')
    ap.add_argument('--steps', type=int, default=0, help='denoising steps; 0 = one per token')
    # EXP3-PATCH (2026-09-30): block-length sweep, CTEV layer window, n-gram baseline (all opt-in)
    ap.add_argument('--block', type=int, default=0, help='block length for semi-AR decoding; 0 = single block (= L)')
    ap.add_argument('--ctev_layers', default='', help='CTEV layer window lo-hi (1-indexed, inclusive); empty = 26-30')
    ap.add_argument('--ngram', type=int, default=0, help='n-gram penalty baseline: forbid completing a seen n-gram; 0 = off')
    ap.add_argument('--decode_rows', default='', help='ROWS-PATCH: save decode-moment attention rows (layers 25-32) per image to this dir')
    ap.add_argument('--ctev_cache', type=int, default=0, help='CTEVCACHE-PATCH: 1 = recompute CTEV entropies only where the state changed')
    ap.add_argument('--dar_w', type=int, default=0, help='also reserve the +-w anchors of each imminent position')
    ap.add_argument('--dar_anchor', type=int, default=0, help='F1 channel: slots for the most dispersed positions')
    ap.add_argument('--dar_entropy', type=int, default=0, help='F3 channel: slots for the least consolidated positions')
    ap.add_argument('--sar_tau', type=int, default=0, help='0 = SAR off; else staleness cap')
    ap.add_argument('--sar_anchor_priority', type=int, default=0, choices=[0, 1],
                    help='fill the SAR budget by lowest anchoring instead of similarity order')
    ap.add_argument('--arg_lambda', type=float, default=0.0, help='0 = ARG off')
    ap.add_argument('--arg_lo', type=int, default=26)
    ap.add_argument('--arg_hi', type=int, default=31)
    ap.add_argument('--arg_w', type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, '/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts')
    import repeat_metrics as RM

    import torch
    from PIL import Image
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    os.makedirs(args.out, exist_ok=True)
    spec = json.load(open(args.images))
    files = spec['files']
    if args.offset:
        files = files[args.offset:]
    if args.limit:
        files = files[:args.limit]

    # EXP3-PATCH: block length and CTEV layer window
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
    if args.ctev_cache > 0:
        _exp3_kw['ctev_cache'] = args.ctev_cache

    meta = {
        'protocol': {'steps': args.steps or args.length, 'dar_mode': args.dar_mode, 'gen_length': args.length,
                     'block_length': block_length, 'prompt': args.prompt,
                     'ngram': args.ngram,
                     'stopping_criteria': ['<|eot_id|>'],
                     'prefix_refresh_interval': 32, 'threshold': 1,
                     'conv_template': 'llava_llada', 'batch_size': 1},
        'mode': args.mode,
        'ctev': {'mode': args.ctev_mode, 'lambda': args.ctev_lambda, 'cache': args.ctev_cache,
                 'theta_bits': args.ctev_theta, 'window': args.ctev_window, 'norm_lens': args.ctev_norm, 'empty': args.ctev_empty,
                 'layers': list(ctev_layers) if ctev_layers is not None else [26, 27, 28, 29, 30]},
        'ctarx': {'scope': args.ctar_scope, 'theta': args.ctar_theta, 'w': args.ctar_w},
        'ctae': {'mode': args.ctae_mode, 'sigma': args.ctae_sigma,
                 'gamma': args.ctae_gamma, 'band': [args.ctae_lo, args.ctae_hi],
                 'f1': {'ref_q': args.f1_ref_q, 'cap': args.f1_cap, 'temp': args.f1_temp, 'w': args.f1_w},
                 'stitch': {'lo': args.stitch_lo, 'hi': args.stitch_hi, 'stride': args.stitch_stride},
                 'dar': {'r': args.dar_r} if args.dar_r > 0 else None},
        'sar': {'tau': args.sar_tau, 'anchor_priority': bool(args.sar_anchor_priority)} if args.sar_tau > 0 else None,
        'arg': {'lambda': args.arg_lambda, 'band': [args.arg_lo, args.arg_hi],
                'half_w': args.arg_w} if args.arg_lambda != 0 else None,
        'cache_config': ({'prompt_interval_steps': args.cache_pi,
                          'gen_interval_steps': args.cache_gi,
                          'transfer_ratio': args.cache_tr} if args.mode == 'dllm_cache'
                         else ({'sampler': 'slowfast_port',
                                'k_exploration_steps': 6,
                                'cycle_len_confidence_threshold': 0.3,
                                'high_confidence_threshold': 0.9,
                                'feature_cache': ({'prompt_interval_steps': 25,
                                                   'gen_interval_steps': 7,
                                                   'transfer_ratio': 0.25,
                                                   'note': 'SF evolved cache (expect_length), ICLR composed semantics'}
                                                  if args.mode == 'slowfast' else None)}
                               if args.mode.startswith('slowfast') else None)),
        'images_spec': {'path': args.images, 'seed': spec['seed'], 'n': len(files),
                        'offset': args.offset, 'limit': args.limit},
        'model': 'GSAI-ML/LLaDA-V',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': args.device, 'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metric_note': 'metrics on token ids trimmed at first <|eot_id|>; see repeat_metrics.py',
    }
    with open(os.path.join(args.out, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        'GSAI-ML/LLaDA-V', None, 'llava_llada',
        attn_implementation='sdpa', device_map=args.device)
    model.eval()

    eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    assert isinstance(eot_id, int) and eot_id >= 0, f'bad eot id: {eot_id}'

    if args.mode == 'dllm_cache':
        from dataclasses import asdict
        from llava.cache import dLLMCache, dLLMCacheConfig
        from llava.hooks import register_cache_LLaDA_V
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
            prompt_interval_steps=args.cache_pi, gen_interval_steps=args.cache_gi,
            transfer_ratio=args.cache_tr)))
        register_cache_LLaDA_V(model, 'model.layers')
        from llava.hooks import cache_hook_LLaDA_V as _chook
        _chook.set_ctae(mode=args.ctae_mode, sigma=args.ctae_sigma, gamma=args.ctae_gamma,
                        lo=args.ctae_lo, hi=args.ctae_hi)
        _chook.set_f1(ref_q=args.f1_ref_q, cap=args.f1_cap, temp=args.f1_temp, half_w=args.f1_w)
        if args.ctae_mode != 'off':
            print(f'[CTAE] mode={args.ctae_mode} sigma={args.ctae_sigma} gamma={args.ctae_gamma}', flush=True)
        if args.sar_tau > 0:
            from llava.model.language_model.utils.sar_hook import attach_sar
            attach_sar(model, tau=args.sar_tau, gen_len=args.length,
                       anchor_priority=bool(args.sar_anchor_priority))
        print(f'[mode] dLLM-Cache registered ({args.cache_pi}/{args.cache_gi}/{args.cache_tr})', flush=True)
    elif args.mode == 'fast_dllm':
        from llava.hooks.fast_dllm_hook import register_fast_dllm_hook
        register_fast_dllm_hook(model)
        print('[mode] Fast-dLLM hook registered (block-level cache, no per-step refresh set)', flush=True)
    elif args.mode.startswith('slowfast') and (args.sf_port or args.ctae_mode != 'off'
                                                or args.dar_r > 0 or args.ctev_mode != 'off'):
        from llava.hooks.sf_cotapp import register_slowfast_cotapp_hook
        _rr = args.ctae_mode.startswith('reroute')
        _sf_comp = dict(ctae_mode='off' if _rr else args.ctae_mode,
                        ctae_sigma=args.ctae_sigma, ctae_gamma=args.ctae_gamma,
                        ctar=_rr, stitch_lo=args.stitch_lo, stitch_hi=args.stitch_hi,
                        ctar_scope=args.ctar_scope, ctar_theta=args.ctar_theta, ctar_w=args.ctar_w,
                        dar_r=args.dar_r, dar_mode=args.dar_mode,
                        ctev_mode=args.ctev_mode, ctev_lambda=args.ctev_lambda,
                        ctev_window=args.ctev_window, ctev_norm=args.ctev_norm,
                        sf_thresh=args.sf_thresh, sf_repguard=args.sf_repguard)
        _sf_sampler = register_slowfast_cotapp_hook(model, comp=_sf_comp, gen_length=args.length,
                                                    block_length=args.length,
                                                    use_cache=(args.mode == 'slowfast'))
        meta['sf_components'] = _sf_comp
        print(f'[mode] SlowFast + CoTA/CoTA++ port {_sf_comp}', flush=True)
    elif args.mode.startswith('slowfast'):
        from llava.hooks.slowfast_hook import register_slowfast_hook
        register_slowfast_hook(model, gen_length=args.length, block_length=args.length,
                               use_cache=(args.mode == 'slowfast'))
        print(f'[mode] SlowFast sampler registered '
              f'({"composed with SF dLLM-Cache 25/7/0.25" if args.mode == "slowfast" else "standalone, no cache"})',
              flush=True)
    else:
        print('[mode] baseline (no hook registered)', flush=True)

    # ROWS-PATCH: decode-moment attention rows (analysis only)
    _rows_rec = None
    if args.decode_rows:
        from decode_rows_recorder import DecodeRowsLLaDAV
        _rows_hook = None
        if args.mode == 'dllm_cache':
            from llava.hooks import cache_hook_LLaDA_V as _rows_hook
        elif '_sf_sampler' in locals():
            from llava.hooks import sf_cotapp as _sfc_r
            _rows_hook = _sfc_r.H
        _rows_rec = DecodeRowsLLaDAV(model, core=model.model, layers_module=model.model.layers, G=args.length,
                               band=(24, 31), mask_id=126336, hook_mod=_rows_hook)
        os.makedirs(args.decode_rows, exist_ok=True)
        print(f'[rows] decode-moment recorder on, hook={getattr(_rows_hook, "__name__", None)} -> {args.decode_rows}', flush=True)

    conv_template = 'llava_llada'
    question = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt

    out_jsonl = os.path.join(args.out, 'outputs.jsonl')
    rows = []
    _gen_rows = []                                                                         # CTEVCACHE-PATCH
    t0 = time.time()
    with open(out_jsonl, 'w') as fout:
        for k, fname in enumerate(files):
            img = Image.open(os.path.join(spec['src'], fname)).convert('RGB')
            image_tensor = process_images([img], image_processor, model.config)
            image_tensor = [t.to(dtype=torch.float16, device=args.device) for t in image_tensor]

            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer,
                                              IMAGE_TOKEN_INDEX, return_tensors='pt'
                                              ).unsqueeze(0).to(args.device)
            _trace = None
            if args.case_trace:
                _trace = {'step': 0, 'rows': [], 'fwd': {'n': 0}, 'refresh': {}}
                if args.mode == 'dllm_cache':
                    from llava.hooks import cache_hook_LLaDA_V as _cht
                    _cht.set_anchor(True, lo=24, hi=31, half_w=5)
                    _cht._ANCHOR['pair_on'] = True
                    _cht.reset_anchor(args.length, device=args.device)
                    if not hasattr(_cht, '_refresh_index_orig'):
                        _cht._refresh_index_orig = _cht.refresh_index
                    def _ri(new_features, cached_features=None, transfer_ratio=0.5, layer_id=0, _t=_trace, _o=_cht._refresh_index_orig):
                        idx = _o(new_features, cached_features, transfer_ratio, layer_id)
                        _t['refresh'].setdefault(_t['fwd']['n'], {})[int(layer_id)] = idx[0].tolist() if idx.numel() else []
                        return idx
                    _cht.refresh_index = _ri
                if not hasattr(model, '_ct_hook'):
                    model._ct_box = {}
                    model._ct_hook = model.model.register_forward_pre_hook(
                        lambda m, a: model._ct_box['t']['fwd'].__setitem__('n', model._ct_box['t']['fwd']['n'] + 1) if model._ct_box.get('t') else None)
                model._ct_box['t'] = _trace
            # entropy is logged with a zero penalty when CTEV is off, which leaves the decoding untouched
            _cm_mode = args.ctev_mode if not (args.case_trace and args.ctev_mode == 'off') else 'ctx'
            _cm_lam = args.ctev_lambda if args.ctev_mode != 'off' else 0.0
            t_gen = time.time()
            if _rows_rec is not None:
                _rows_rec.attach()
            with torch.no_grad():
                cont = model.generate(
                    input_ids, images=image_tensor, image_sizes=[img.size],
                    steps=args.steps or args.length, gen_length=args.length,
                    block_length=block_length, tokenizer=tokenizer,
                    **_exp3_kw,
                    stopping_criteria=['<|eot_id|>'],
                    prefix_refresh_interval=32, threshold=1,
                    ctev_mode=(_cm_mode if args.case_trace else args.ctev_mode),
                    ctev_lambda=(_cm_lam if args.case_trace else args.ctev_lambda),
                    case_trace=_trace,
                    ctev_theta=args.ctev_theta, ctev_window=args.ctev_window,
                    ctev_norm=args.ctev_norm,
                    ctev_empty=args.ctev_empty,
                    ctae_stitch='1' if (args.ctae_mode.startswith('stitch') or args.ctae_mode.startswith('reroute')) else '0',
                    stitch_lo=args.stitch_lo, stitch_hi=args.stitch_hi,
                    stitch_stride=args.stitch_stride,
                    stitch_blend=args.stitch_blend,
                    ctar_scope=args.ctar_scope, ctar_theta=args.ctar_theta, ctar_w=args.ctar_w,
                    dar_r=args.dar_r,
                    dar_w=args.dar_w, dar_mode=args.dar_mode,
                    dar_anchor=args.dar_anchor, dar_entropy=args.dar_entropy,
                    arg_lambda=args.arg_lambda, arg_lo=args.arg_lo,
                    arg_hi=args.arg_hi, arg_w=args.arg_w)
            gen_s = time.time() - t_gen
            if _rows_rec is not None:
                _rows_rec.detach()
                _rows_info = _rows_rec.save(os.path.join(args.decode_rows, 'rows_%s.npz' % os.path.splitext(fname)[0]),
                                            cont[0].tolist())
                print('[rows]', fname, _rows_info, flush=True)

            if _trace is not None:
                _pair = None
                if args.mode == 'dllm_cache':
                    from llava.hooks import cache_hook_LLaDA_V as _chp
                    _pp = _chp._ANCHOR.get('pair')
                    if _pp is not None:
                        _pair = {k2: v2.cpu().tolist() for k2, v2 in _pp.items()}
                json.dump({'image': fname, 'rows': _trace['rows'], 'ctar_pair': _pair,
                           'refresh': {str(k2): v2 for k2, v2 in _trace['refresh'].items()},
                           'n_forwards': _trace['fwd']['n']},
                          open(os.path.join(args.out, 'trace_%s.json' % os.path.splitext(fname)[0]), 'w'))
            ids = cont[0].tolist()
            if len(ids) > args.length:          # generate returned prompt+suffix
                ids = ids[-args.length:]
            m = RM.sample_metrics(ids, eot_id)
            text = tokenizer.decode(RM.trim_at_eot(ids, eot_id), skip_special_tokens=True)
            row = {'idx': k, 'image': fname, 'gen_seconds': round(gen_s, 2),
                   'text': text, 'ids': ids, **m}
            rows.append(m); _gen_rows.append(row)                                         # CTEVCACHE-PATCH
            fout.write(json.dumps(row) + '\n')
            fout.flush()
            if (k + 1) % 10 == 0 or k == 0:
                el = time.time() - t0
                print(f'[{k+1}/{len(files)}] arr={m["arr"]:.3f} mrl={m["mrl"]:.0f} '
                      f'len={m["len_trimmed"]} {el/ (k+1):.1f}s/img '
                      f'eta={(len(files)-k-1)*el/(k+1)/60:.0f}min', flush=True)

    summary = {'config': meta, 'aggregate': RM.aggregate(rows),
               'wall_seconds': round(time.time() - t0, 1),
               'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
               'peak_mem_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2),          # CTEVCACHE-PATCH
               'gen_seconds_mean': round(float(sum(r_['gen_seconds'] for r_ in _gen_rows) / max(len(_gen_rows), 1)), 2)}
    if getattr(model, '_ctev_cache_stats', None):
        summary['ctev_cache'] = dict(model._ctev_cache_stats)
    if args.ctae_mode != 'off':
        try:
            from llava.hooks import cache_hook_LLaDA_V as _ch
            summary['ctae_coverage'] = dict(_ch._CTAE_STATS)
            print('[CTAE] coverage', _ch._CTAE_STATS, flush=True)
        except Exception as _e:
            print('[CTAE] coverage unavailable:', _e, flush=True)
    if '_sf_sampler' in locals():
        from llava.hooks import sf_cotapp as _sfc
        _cov = {'sampler': dict(_sf_sampler.stats), 'stitch': dict(_sfc.H._STITCH.get('stats', {}))}
        summary['sf_coverage'] = _cov
        print('[SF-port] coverage', _cov, flush=True)
        _c = _sf_sampler.c
        _bad = ((_c['ctar'] and _cov['stitch'].get('used', 0) == 0) or
                (_c['ctev_mode'] != 'off' and _cov['sampler'].get('ctev_calls', 0) == 0))
        if _bad:
            print('[SF-port] a requested component never fired; refusing to write summary', flush=True)
            sys.exit(3)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary['aggregate'], indent=1), flush=True)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
