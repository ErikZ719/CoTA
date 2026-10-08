#!/usr/bin/env python
"""Repeat-Curse evaluation for LaViDa-LLaDa (jacklishufan/lavida-llada-v1.0-instruct), 2026-09-17.

Same images, prompt and metrics as run_repeat_eval.py (LLaDA-V). Decoding reproduces LaViDa's own
generate() without its prefix cache (temperature 0, low-confidence remasking, one token per step by
default), so 'baseline' is the uncached model. 'dllm_cache' registers the dLLM-Cache hook of
MMaDA (the same OLMo-style LLaDA block) on LaViDa's language model; the CoTA/CoTA++ components come
from mmada_cotapp.py. Run with /data/zhaoqiyan/venvs/lavida/bin/python.
"""
import argparse, copy, json, os, sys, time, glob, warnings
warnings.filterwarnings('ignore')
REPO = '/data/zhaoqiyan/autodl-tmp/LaViDa'
DLLM = '/data/zhaoqiyan/autodl-tmp/dLLM-cache'
SCRIPTS = '/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts'
for pth in (SCRIPTS, DLLM, REPO):
    if pth not in sys.path:
        sys.path.insert(0, pth)
CKPT = glob.glob('/data/zhaoqiyan/autodl-tmp/hf_cache/hub/models--jacklishufan--lavida-llada-v1.0-instruct/snapshots/*')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/data/coco500_final.json')
    ap.add_argument('--length', type=int, required=True)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--block', type=int, default=0)
    ap.add_argument('--mode', choices=['baseline', 'dllm_cache', 'slowfast'], required=True)
    ap.add_argument('--cache_pi', type=int, default=25)
    ap.add_argument('--cache_gi', type=int, default=7)
    ap.add_argument('--cache_tr', type=float, default=0.25)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--prompt', default='Please describe the image in detail.')
    ap.add_argument('--port', type=int, default=0)
    ap.add_argument('--ctae_mode', default='off', choices=['off', 'bias', 'reroute_q'])
    ap.add_argument('--ctae_sigma', type=float, default=5.0)
    ap.add_argument('--ctae_gamma', type=float, default=0.5)
    ap.add_argument('--stitch_lo', type=int, default=24)
    ap.add_argument('--stitch_hi', type=int, default=31)
    ap.add_argument('--ctar_theta', type=int, default=1)
    ap.add_argument('--ctar_w', type=int, default=5)
    ap.add_argument('--dar_r', type=int, default=0)
    ap.add_argument('--dar_mode', default='masked', choices=['legacy', 'masked', 'score'])
    ap.add_argument('--ctev_mode', default='off', choices=['off', 'self', 'ctx'])
    ap.add_argument('--ctev_lambda', type=float, default=0.25)
    ap.add_argument('--ctev_window', type=int, default=5)
    ap.add_argument('--ctev_norm', type=int, default=1)
    ap.add_argument('--decode_rows', default='', help='ROWS-PATCH: save decode-moment attention rows (layers 25-32) per image to this dir')
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import repeat_metrics as RM
    import mmada_cotapp as MC
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    L = args.length
    steps = args.steps or L
    block = args.block or L
    os.makedirs(args.out, exist_ok=True)
    spec = json.load(open(args.images))
    files = spec['files'][args.offset:]
    if args.limit:
        files = files[:args.limit]
    comp = dict(ctar=args.ctae_mode == 'reroute_q', lo=args.stitch_lo, hi=args.stitch_hi,
                theta=args.ctar_theta, w=args.ctar_w, dar_r=args.dar_r, dar_mode=args.dar_mode,
                ctae_mode='bias' if args.ctae_mode == 'bias' else 'off',
                ctae_sigma=args.ctae_sigma, ctae_gamma=args.ctae_gamma,
                ctev_mode=args.ctev_mode, ctev_lambda=args.ctev_lambda,
                ctev_window=args.ctev_window, ctev_norm=args.ctev_norm)
    MC.configure(**comp)
    use_port = MC.any_on() or bool(args.port)
    meta = {'model': 'jacklishufan/lavida-llada-v1.0-instruct',
            'protocol': {'gen_length': L, 'steps': steps, 'block_length': block, 'prompt': args.prompt,
                         'temperature': 0, 'remasking': 'low_confidence', 'conv_template': 'llada',
                         'prefix_lm': False, 'batch_size': 1},
            'mode': args.mode,
            'cache_config': ({'prompt_interval_steps': args.cache_pi, 'gen_interval_steps': args.cache_gi,
                              'transfer_ratio': args.cache_tr} if args.mode == 'dllm_cache' else None),
            'components': comp if use_port else None,
            'images_spec': {'path': args.images, 'seed': spec['seed'], 'n': len(files),
                            'offset': args.offset, 'limit': args.limit},
            'torch': torch.__version__, 'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(meta, open(os.path.join(args.out, 'run_meta.json'), 'w'), indent=1)

    vision_kwargs = dict(mm_vision_tower='google/siglip-so400m-patch14-384', mm_resampler_type=None,
                         mm_projector_type='mlp2x_gelu', mm_hidden_size=1152, use_mm_proj=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        CKPT, None, 'llava_llada', device_map='cuda:0', vision_kwargs=vision_kwargs, torch_dtype='bfloat16')
    model.eval(); model.tie_weights(); model.to(torch.bfloat16)
    llm = model.get_model()
    llm.set_activation_checkpointing(None)          # inference: no checkpoint wrappers
    device = 'cuda'
    mask_id = 126336
    terms = {i for i in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>'),
                         tokenizer.convert_tokens_to_ids('<|endoftext|>')) if isinstance(i, int) and i >= 0}
    meta['terminator_ids'] = sorted(terms)

    if args.mode == 'dllm_cache':
        from dataclasses import asdict
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(prompt_interval_steps=args.cache_pi,
                                                        gen_interval_steps=args.cache_gi,
                                                        transfer_ratio=args.cache_tr)))
        H = MC.build_dllm_hook()                    # MMaDA hook source; with no component it is the original
        H.register_cache_MMaDA(llm, 'transformer.blocks')
        print(f'[mode] dLLM-Cache on LaViDa ({args.cache_pi}/{args.cache_gi}/{args.cache_tr}) components={comp if use_port else None}', flush=True)
    elif args.mode == 'slowfast':
        # SlowFast on LaViDa: the sampler and the evolved cache of the MMaDA stack (both models are
        # OLMo-style LLaDA blocks), with a shim that feeds the image prefix as embeddings.
        _SF = MC.build_sf_stack(llm) if use_port else __import__('sf_mmada_stack')
        _SF.SFFeatureCache.new_instance(prompt_interval_steps=args.cache_pi,
                                        gen_interval_steps=args.cache_gi,
                                        cfg_interval_steps=1,
                                        transfer_ratio=args.cache_tr)
        _SF.register_sf_cache_MMaDA(llm, 'transformer.blocks')

        class _LaViDaShim:
            """The sampler calls model(x, attention_mask=...); LaViDa needs the prefix as embeddings."""
            def __init__(self, inner):
                self._llm = inner
                self.device = next(inner.parameters()).device
                self.prefix = None

            def __call__(self, x, attention_mask=None):
                need = comp['ctev_mode'] != 'off'
                emb = self._llm.transformer.wte(x)
                emb[:, :self.prefix.shape[1]] = self.prefix
                out = self._llm(None, input_embeddings=emb, output_hidden_states=need)
                if use_port:
                    MC.S['hid'] = (out.hidden_states, x.shape[1]) if need else None
                return out

        sf_shim = _LaViDaShim(llm)
        sf_sampler = _SF.SlowFastSampler(sf_shim, {'gen_length': L,
                                                   'block_length': block}, mask_id=mask_id)
        print(f'[mode] SlowFast on LaViDa ({args.cache_pi}/{args.cache_gi}/{args.cache_tr}) '
              f'components={comp if use_port else None}', flush=True)
    else:
        assert not use_port or args.port, 'components need a cache backend'
        print('[mode] baseline (no cache)', flush=True)

    @torch.no_grad()
    def generate(inputs_embeds):
        P = inputs_embeds.shape[1]
        MC.reset(P, L)
        x = torch.full((1, P + L), mask_id, dtype=torch.long, device=device)
        x[:, :P] = 0                                 # placeholder ids; the prefix lives in inputs_embeds
        assert L % block == 0
        nblocks = L // block
        assert steps % nblocks == 0
        spb = steps // nblocks
        need_h = comp['ctev_mode'] != 'off'
        for nb in range(nblocks):
            bm = (x[:, P + nb * block: P + (nb + 1) * block] == mask_id)
            ntt = MC.get_num_transfer_tokens(bm, spb)
            for i in range(spb):
                mask_index = (x == mask_id)
                if not mask_index[:, P + nb * block: P + (nb + 1) * block].any():
                    continue
                emb = llm.transformer.wte(x)
                emb[:, :P] = inputs_embeds
                out = llm(None, input_embeddings=emb, output_hidden_states=need_h)
                logits = out.logits
                x0 = torch.argmax(logits, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, P + (nb + 1) * block:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device, dtype=x0_p.dtype))
                if comp['dar_r'] > 0 and comp['dar_mode'] == 'legacy':
                    MC.publish_scores(conf[0, P:P + L])
                if need_h:
                    pen = MC.penalty(llm, out.hidden_states, x[0, P:P + L], P, L, mask_id)
                    conf[:, P:P + L] = conf[:, P:P + L] - pen.to(conf.dtype)
                ti = torch.zeros_like(x0, dtype=torch.bool)
                _, sel = torch.topk(conf[0], k=int(ntt[0, i]))
                ti[0, sel] = True
                x[ti] = x0[ti]
                if comp['dar_r'] > 0 and comp['dar_mode'] in ('masked', 'score'):
                    sv = conf[0, P:P + L].detach().clone()
                    if comp['dar_mode'] == 'masked':
                        sv[ti[0, P:P + L]] = -float('inf')
                    MC.publish_scores(sv)
                MC.publish_mask(x[0, P:P + L] == mask_id)
                MC.STATS['steps'] += 1
        return x[:, P:]

    # ROWS-PATCH: decode-moment attention rows (analysis only, dllm_cache mode)
    _rows_rec = None
    if args.decode_rows:
        assert args.mode == 'dllm_cache', 'decode_rows: dllm_cache mode only'
        from decode_rows_recorder import DecodeRowsOLMo
        _rows_rec = DecodeRowsOLMo(llm, core=llm, blocks=llm.transformer.blocks, G=L, band=(24, 31),
                                   mask_id=mask_id, mc=(MC if use_port else None))
        os.makedirs(args.decode_rows, exist_ok=True)
        print(f'[rows] decode-moment recorder on, port={use_port} -> {args.decode_rows}', flush=True)

    conv_template = 'llada'
    question = DEFAULT_IMAGE_TOKEN + '\n' + args.prompt
    rows, t0 = [], time.time()
    with open(os.path.join(args.out, 'outputs.jsonl'), 'w') as fout:
        for k, fname in enumerate(files):
            img = Image.open(os.path.join(spec['src'], fname)).convert('RGB')
            image_tensor = process_images([img], image_processor, model.config)
            image_tensor = [t.to(dtype=torch.bfloat16, device=device) for t in image_tensor]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).to(device)
            with torch.no_grad():
                (_, _, _, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, image_tensor, ['image'], image_sizes=[img.size])
                if args.mode == 'dllm_cache':
                    from dllm_cache.cache import dLLMCache
                    dLLMCache().reset_cache(prompt_length=inputs_embeds.shape[1])
                t_gen = time.time()
                if args.mode == 'slowfast':
                    sf_shim.prefix = inputs_embeds.to(torch.bfloat16)
                    MC.reset(inputs_embeds.shape[1], L)
                    gen = sf_sampler.generate(
                        torch.zeros((1, inputs_embeds.shape[1]), dtype=torch.long, device=device), None)
                else:
                    if _rows_rec is not None:
                        _rows_rec.attach()
                    try:
                        gen = generate(inputs_embeds.to(torch.bfloat16))
                    finally:
                        if _rows_rec is not None:
                            _rows_rec.detach()
                gen_s = time.time() - t_gen
                if _rows_rec is not None:
                    _rows_info = _rows_rec.save(os.path.join(args.decode_rows, 'rows_%s.npz' % os.path.splitext(fname)[0]),
                                                gen[0].tolist())
                    print('[rows]', fname, _rows_info, flush=True)
            ids = gen[0].tolist()
            m = RM.sample_metrics(ids, terms)
            text = tokenizer.decode(RM.trim_at_eot(ids, terms), skip_special_tokens=True)
            fout.write(json.dumps({'idx': k, 'image': fname, 'gen_seconds': round(gen_s, 2),
                                   'text': text, 'ids': ids, **m}) + '\n')
            fout.flush()
            rows.append(m)
            if (k + 1) % 10 == 0 or k < 2:
                el = time.time() - t0
                print(f'[{k+1}/{len(files)}] arr={m["arr"]:.3f} mrl={m["mrl"]:.0f} len={m["len_trimmed"]} '
                      f'{el/(k+1):.1f}s/img | {text[:120]!r}', flush=True)
    summary = {'config': meta, 'aggregate': RM.aggregate(rows), 'wall_seconds': round(time.time() - t0, 1),
               'end_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if use_port:
        summary['port_coverage'] = dict(MC.STATS)
        print('[port] coverage', dict(MC.STATS), flush=True)
        bad = ((comp['ctar'] and MC.STATS['ctar_rows'] == 0) or
               (comp['dar_r'] > 0 and MC.STATS['dar_steps'] == 0) or
               (comp['ctae_mode'] != 'off' and MC.STATS['ctae_calls'] == 0) or
               (comp['ctev_mode'] != 'off' and MC.STATS['ctev_calls'] == 0))
        if bad:
            print('[port] a requested component never fired; refusing to write summary', flush=True)
            sys.exit(3)
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary['aggregate'], indent=1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
