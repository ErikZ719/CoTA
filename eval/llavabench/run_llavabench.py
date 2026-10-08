#!/usr/bin/env python
"""LLaVA-Bench (in the wild) generation, standalone.

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
    ap.add_argument('--mode', default='dllm_cache', choices=['baseline', 'dllm_cache'])
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

    if args.mode == 'dllm_cache':
        from dataclasses import asdict
        from llava.cache import dLLMCache, dLLMCacheConfig
        from llava.hooks import register_cache_LLaDA_V
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
            prompt_interval_steps=args.cache_pi, gen_interval_steps=args.cache_gi,
            transfer_ratio=args.cache_tr)))
        register_cache_LLaDA_V(model, 'model.layers')
        from llava.hooks import cache_hook_LLaDA_V as _chook
        _chook.set_ctae(mode=args.ctae_mode)

    gen_extra = {}
    if args.ctae_mode != 'off':
        gen_extra.update(ctae_mode=args.ctae_mode, stitch_lo=args.stitch_lo, stitch_hi=args.stitch_hi,
                         stitch_stride=args.stitch_stride, stitch_blend=args.stitch_blend,
                         ctar_scope=args.ctar_scope, ctar_theta=args.ctar_theta, ctar_w=args.ctar_w,
                         ctae_stitch='1' if args.ctae_mode.startswith(('stitch', 'reroute')) else '0')
    if args.dar_r > 0:
        gen_extra.update(dar_r=args.dar_r)
    if args.ctev_lambda != 0:
        gen_extra.update(ctev_mode='ctx', ctev_lambda=args.ctev_lambda)

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
                                      prefix_refresh_interval=32, threshold=1, **gen_extra)
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
    if args.ctae_mode != 'off':
        from llava.hooks import cache_hook_LLaDA_V as _ch
        st = dict(_ch._STITCH['stats'])
        print(f'[CTAR] stitch stats {st}', flush=True)
        if st.get('used', 0) == 0:
            sys.exit('[CTAR] FATAL: component requested but never applied -- results would be '
                     'identical to the plain cache. Aborting so this cannot pass silently.')
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
