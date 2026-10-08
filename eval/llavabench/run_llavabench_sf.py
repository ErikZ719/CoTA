#!/usr/bin/env python
"""LLaVA-Bench (in the wild) generation under SlowFast, LLaDA-V (2026-09-30). A copy of run_llavabench.py in which
the backend is the SlowFast sampler of run_repeat_eval.py --mode slowfast (llava.hooks.slowfast_hook, or the
component port llava.hooks.sf_cotapp as soon as a component is on). Everything else is unchanged.

LLaVA-Bench (in the wild) generation, standalone.

This lmms-eval build has no llava-bench task, and the benchmark is small enough that a
direct runner is simpler than adding one. Generation only: answers are written to disk
and scored separately, so a judge that is unreachable from this host is not a blocker.
Component flags mirror run_repeat_eval.py.
"""
import argparse, copy, json, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--parquet', default='/data/zhaoqiyan/autodl-tmp/datasets/llava-bench-in-the-wild/data/train-00000-of-00001.parquet')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--mode', default='slowfast', choices=['slowfast'])
    ap.add_argument('--length', type=int, default=512)
    ap.add_argument('--cache_tr', type=float, default=0.25)
    ap.add_argument('--cache_pi', type=int, default=25)
    ap.add_argument('--cache_gi', type=int, default=7)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ctae_mode', default='off')
    ap.add_argument('--stitch_lo', type=int, default=24)
    ap.add_argument('--stitch_hi', type=int, default=31)
    ap.add_argument('--stitch_stride', type=int, default=1)
    ap.add_argument('--stitch_blend', type=float, default=1.0)
    ap.add_argument('--ctar_scope', default='all', choices=['all', 'ctx', 'sfx'])
    ap.add_argument('--ctar_theta', type=int, default=0)
    ap.add_argument('--ctar_w', type=int, default=5)
    ap.add_argument('--dar_r', type=int, default=0)
    ap.add_argument('--ctev_lambda', type=float, default=0.0)
    ap.add_argument('--dar_mode', default='legacy')
    ap.add_argument('--sf_thresh', choices=['score', 'raw'], default='score')
    ap.add_argument('--sf_repguard', type=float, default=0.0)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    os.makedirs(args.out, exist_ok=True)
    ds = load_dataset('parquet', data_files=args.parquet)['train']
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    tokenizer, model, image_processor, _ = load_pretrained_model(
        'GSAI-ML/LLaDA-V', None, 'llava_llada', attn_implementation='sdpa', device_map=args.device)
    model.eval()

    _rr = args.ctae_mode.startswith('reroute')
    comp = dict(ctae_mode='off' if _rr else args.ctae_mode, ctae_sigma=2.0, ctae_gamma=0.75, ctar=_rr,
                stitch_lo=args.stitch_lo, stitch_hi=args.stitch_hi, ctar_scope=args.ctar_scope,
                ctar_theta=args.ctar_theta, ctar_w=args.ctar_w, dar_r=args.dar_r, dar_mode=args.dar_mode,
                ctev_mode='ctx' if args.ctev_lambda != 0 else 'off',
                ctev_lambda=args.ctev_lambda if args.ctev_lambda != 0 else 0.25, ctev_window=5, ctev_norm=1,
                sf_thresh=args.sf_thresh, sf_repguard=args.sf_repguard)
    use_port = args.ctae_mode != 'off' or args.dar_r > 0 or args.ctev_lambda != 0
    if use_port:
        from llava.hooks.sf_cotapp import register_slowfast_cotapp_hook
        sampler = register_slowfast_cotapp_hook(model, comp=comp, gen_length=args.length, block_length=args.length,
                                                use_cache=True)
    else:
        from llava.hooks.slowfast_hook import register_slowfast_hook
        sampler = register_slowfast_hook(model, gen_length=args.length, block_length=args.length, use_cache=True)
    gen_extra = comp if use_port else {}

    meta = dict(mode=args.mode, length=args.length, cache_tr=args.cache_tr,
                cache_pi=args.cache_pi, cache_gi=args.cache_gi,
                components=gen_extra, n=len(ds))
    json.dump(meta, open(f'{args.out}/run_meta.json', 'w'), indent=1)

    eot = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    t0 = time.time()
    with open(f'{args.out}/answers.jsonl', 'w') as fout:
        for k, row in enumerate(ds):
            img = row['image'].convert('RGB')
            image_tensor = [t.to(dtype=torch.float16, device=args.device)
                            for t in process_images([img], image_processor, model.config)]
            conv = copy.deepcopy(conv_templates['llava_llada'])
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + row['question'])
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).to(args.device)
            with torch.no_grad():
                cont = model.generate(input_ids, images=image_tensor, image_sizes=[img.size],
                                      steps=args.length, gen_length=args.length,
                                      block_length=args.length, tokenizer=tokenizer,
                                      stopping_criteria=['<|eot_id|>'],
                                      prefix_refresh_interval=32, threshold=1)
            ids = cont[0].tolist()
            if len(ids) > args.length:
                ids = ids[-args.length:]
            if eot in ids:
                ids = ids[:ids.index(eot)]
            fout.write(json.dumps({
                'question_id': int(row['question_id']), 'category': row['category'],
                'question': row['question'], 'reference': row['gpt_answer'],
                'caption': row.get('caption', ''),
                'answer': tokenizer.decode(ids, skip_special_tokens=True).strip()}) + '\n')
            fout.flush()
            if (k + 1) % 10 == 0:
                print(f'[{k+1}/{len(ds)}] {(time.time()-t0)/(k+1):.1f}s/q', flush=True)
    if use_port:
        from llava.hooks import sf_cotapp as _sfc
        cov = {'sampler': dict(sampler.stats), 'stitch': dict(_sfc.H._STITCH.get('stats', {}))}
        print(f'[SF-port] coverage {cov}', flush=True)
        if (comp['ctar'] and cov['stitch'].get('used', 0) == 0) or \
           (comp['ctev_mode'] != 'off' and cov['sampler'].get('ctev_calls', 0) == 0):
            sys.exit('[SF-port] FATAL: a requested component never fired. Aborting so this cannot pass silently.')
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
