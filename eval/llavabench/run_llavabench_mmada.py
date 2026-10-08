#!/usr/bin/env python
"""LLaVA-Bench (in the wild), mmada. Derived from run_repeat_eval_mmada.py by make_lb_runners.py:
the 60 questions of the released parquet replace the COCO list and the single fixed prompt;
model loading, cache/component wiring and the decoding loop are unchanged. Answers go to
answers.jsonl in the format code-final/eval/score_llavabench.py expects.
ORIGINAL HEADER FOLLOWS.
Phase 0 driver (MMaDA): Repeat-Curse evaluation over the same 500-COCO list.

Protocol (user-confirmed 2026-09-04):
  max_new_tokens = steps = block_length = L  (single block)
  native prompt: "Please describe this image in detail."
  cache mode uses the ICLR-era intervals: prompt 20 / gen 5 / ratio 0.25
  batch size 1; temperature 0 (deterministic low-confidence remasking)

Pipeline follows dLLM-cache/demo_MMada_mmu_cache.py exactly (MAGVITv2 image
tokens + UniversalPrompting 'mmu'); the only additions are batching over the
image list, per-image cache reset, and metric computation via repeat_metrics.

Run with cwd (or sys.path) at /data/zhaoqiyan/autodl-tmp/dLLM-cache so repo-local
packages (mmada_models, mmada_training, dllm_cache) resolve.
"""
import argparse, json, os, sys, time, warnings

warnings.filterwarnings('ignore')

REPO = '/data/zhaoqiyan/autodl-tmp/dLLM-cache'
SCRIPTS = '/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/data/coco500_seed42.json')
    ap.add_argument('--length', type=int, required=True)
    ap.add_argument('--steps', type=int, default=0, help='override; default = length')
    ap.add_argument('--block', type=int, default=0, help='override; default = length')
    ap.add_argument('--mode', choices=['baseline', 'dllm_cache', 'slowfast'], required=True)
    ap.add_argument('--cache_pi', type=int, default=20, help='prompt_interval_steps')
    ap.add_argument('--cache_gi', type=int, default=5, help='gen_interval_steps')
    ap.add_argument('--cache_tr', type=float, default=0.25, help='transfer_ratio')
    ap.add_argument('--ckpt', default='Gen-Verse/MMaDA-8B-MixCoT')
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--parquet', default='/data/zhaoqiyan/autodl-tmp/datasets/llava-bench-in-the-wild/data/train-00000-of-00001.parquet')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    # CoTA / CoTA++ components (scripts/mmada_cotapp.py), 2026-09-17
    ap.add_argument('--port', type=int, default=0, help='use the component port with no component (identity test)')
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
    ap.add_argument('--prompt', default='Please describe this image in detail.')
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    sys.path.insert(0, SCRIPTS)
    os.chdir(REPO)
    import repeat_metrics as RM

    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoTokenizer
    from mmada_models import MMadaModelLM, MAGVITv2
    from mmada_training.prompting_utils import UniversalPrompting
    from demo_MMada_mmu_cache import mmu_generate_with_cache

    device = 'cuda'
    os.makedirs(args.out, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset('parquet', data_files=args.parquet)['train']
    files = [ds[i] for i in range(len(ds))][args.offset:]
    if args.limit:
        files = files[:args.limit]
    comp = dict(ctar=args.ctae_mode == 'reroute_q', lo=args.stitch_lo, hi=args.stitch_hi,
                theta=args.ctar_theta, w=args.ctar_w, dar_r=args.dar_r, dar_mode=args.dar_mode,
                ctae_mode='bias' if args.ctae_mode == 'bias' else 'off',
                ctae_sigma=args.ctae_sigma, ctae_gamma=args.ctae_gamma,
                ctev_mode=args.ctev_mode, ctev_lambda=args.ctev_lambda,
                ctev_window=args.ctev_window, ctev_norm=args.ctev_norm)
    use_port = bool(args.port) or comp['ctar'] or comp['dar_r'] > 0 or comp['ctae_mode'] != 'off' \
        or comp['ctev_mode'] != 'off'
    gen_fn = None

    meta = {
        'model': args.ckpt, 'vq': 'showlab/magvitv2',
        'protocol': {'max_new_tokens': args.length,
                     'steps': args.steps or args.length,
                     'block_length': args.block or args.length,
                     'prompt': args.prompt,
                     'temperature': 0, 'remasking': 'low_confidence',
                     'resolution': args.resolution, 'batch_size': 1},
        'mode': args.mode,
        'cache_config': ({'prompt_interval_steps': args.cache_pi,
                          'gen_interval_steps': args.cache_gi,
                          'transfer_ratio': args.cache_tr,
                          **({'sampler': 'slowfast_original+SF_evolved_cache',
                              'note': 'steps knob inapplicable; sampler self-schedules'}
                             if args.mode == 'slowfast' else {})}
                         if args.mode in ('dllm_cache', 'slowfast') else None),
        'images_spec': {'benchmark': 'llava-bench-in-the-wild', 'path': args.parquet,
                        'n': len(files), 'offset': args.offset, 'limit': args.limit},
        'components': comp if use_port else None,
        'torch': torch.__version__,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metric_note': 'metrics on suffix token ids trimmed at first eos; see repeat_metrics.py',
    }
    with open(os.path.join(args.out, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    model = MMadaModelLM.from_pretrained(args.ckpt,
                                         trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    vq_model = MAGVITv2().from_pretrained('showlab/magvitv2').to(device)

    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=512,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                        "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=0.1, use_reserved_token=True)

    # Official MMaDA inference chat template (Gen-Verse/MMaDA generate.py)
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'"
        "+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}{% endfor %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}")

    eos_id = tokenizer.eos_token_id            # <|eot|> 126081
    eot_chat_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')  # chat turn terminator
    terminators = {i for i in (eos_id, eot_chat_id) if isinstance(i, int) and i >= 0}
    assert terminators, 'no terminator ids resolved'
    meta['terminator_ids'] = sorted(terminators)
    with open(os.path.join(args.out, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    if args.mode == 'dllm_cache':
        from dataclasses import asdict
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache import register_cache_MMaDA
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(
            prompt_interval_steps=args.cache_pi, gen_interval_steps=args.cache_gi,
            transfer_ratio=args.cache_tr)))
        if use_port:
            import mmada_cotapp as MC
            MC.configure(**comp)
            _H = MC.build_dllm_hook()
            _H.register_cache_MMaDA(model, 'model.transformer.blocks')
            gen_fn = MC.mmu_generate_cotapp
            print(f'[mode] dLLM-Cache + CoTA/CoTA++ port {comp}', flush=True)
        else:
            register_cache_MMaDA(model, 'model.transformer.blocks')
        print(f'[mode] dLLM-Cache registered ({args.cache_pi}/{args.cache_gi}/{args.cache_tr})', flush=True)
    elif args.mode == 'slowfast' and use_port:
        import mmada_cotapp as MC
        MC.configure(**comp)
        _SF = MC.build_sf_stack(model)
        _SF.SFFeatureCache.new_instance(prompt_interval_steps=args.cache_pi,
                                        gen_interval_steps=args.cache_gi,
                                        cfg_interval_steps=1,
                                        transfer_ratio=args.cache_tr)
        _SF.register_sf_cache_MMaDA(model, 'model.transformer.blocks')
        sf_sampler = _SF.SlowFastSampler(MC.CotaShim(model),
                                         {'gen_length': args.length,
                                          'block_length': args.block or args.length})
        print(f'[mode] SlowFast + CoTA/CoTA++ port {comp}', flush=True)
    elif args.mode == 'slowfast':
        from sf_mmada_stack import SFFeatureCache, register_sf_cache_MMaDA, SlowFastSampler
        SFFeatureCache.new_instance(prompt_interval_steps=args.cache_pi,
                                    gen_interval_steps=args.cache_gi,
                                    cfg_interval_steps=1,
                                    transfer_ratio=args.cache_tr)
        register_sf_cache_MMaDA(model, 'model.transformer.blocks')

        class _ModelShim:
            """MMaDA forward 无 attention_mask 形参;采样器传该 kwarg,垫片吸收之。"""
            def __init__(self, m):
                self._m = m
                self.device = next(m.parameters()).device
            def __call__(self, x, attention_mask=None):
                return self._m(x)

        sf_sampler = SlowFastSampler(_ModelShim(model),
                                     {'gen_length': args.length,
                                      'block_length': args.block or args.length})
        print(f'[mode] SlowFast sampler + SF cache registered '
              f'({args.cache_pi}/{args.cache_gi}/{args.cache_tr})', flush=True)
    else:
        print('[mode] baseline (no hook registered)', flush=True)

    # demo's process_image, plus RGB conversion for grayscale COCO images
    tfm = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    out_jsonl = os.path.join(args.out, 'answers.jsonl')
    rows, t0 = [], time.time()
    with open(out_jsonl, 'w') as fout:
        for k, row in enumerate(files):
            img = row['image'].convert('RGB')
            pixel = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                # Official MMaDA MMU inference format (inference_mmu.py):
                # [<|mmu|>][<|soi|>][image codes + text-vocab offset][<|eoi|>][chat-templated text]
                image_tokens = vq_model.get_code(pixel) + len(tokenizer)
                text_ids = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': row['question']}],
                    tokenize=True, add_generation_prompt=True,
                    return_tensors='pt').to(device)
                sp = uni_prompting.sptids_dict
                input_ids = torch.cat([
                    sp['<|mmu|>'].to(device).unsqueeze(0),
                    sp['<|soi|>'].to(device).unsqueeze(0),
                    image_tokens,
                    sp['<|eoi|>'].to(device).unsqueeze(0),
                    text_ids], dim=1).long()
                attention_mask = torch.ones_like(input_ids)

                if args.mode == 'dllm_cache':
                    from dllm_cache.cache import dLLMCache
                    dLLMCache().reset_cache(prompt_length=input_ids.shape[1])

                t_gen = time.time()
                if args.mode == 'slowfast':
                    gen_ids = sf_sampler.generate(input_ids, attention_mask)
                elif gen_fn is not None:
                    output = gen_fn(model, input_ids=input_ids,
                                    max_new_tokens=args.length,
                                    steps=args.steps or args.length,
                                    block_length=args.block or args.length,
                                    mask_id=126336, attention_mask=attention_mask)
                    gen_ids = output[:, input_ids.shape[1]:]
                else:
                    output = mmu_generate_with_cache(
                        model, input_ids=input_ids,
                        max_new_tokens=args.length,
                        steps=args.steps or args.length,
                        block_length=args.block or args.length,
                        temperature=0,
                        remasking='low_confidence', mask_id=126336,
                        attention_mask=attention_mask)
                    gen_ids = output[:, input_ids.shape[1]:]
                gen_s = time.time() - t_gen

            ids = gen_ids[0].tolist()
            m = RM.sample_metrics(ids, terminators)
            text = tokenizer.decode(RM.trim_at_eot(ids, terminators), skip_special_tokens=True)
            rec = {'question_id': int(row['question_id']), 'category': row['category'],
                   'question': row['question'], 'reference': row['gpt_answer'],
                   'caption': row.get('caption', ''), 'answer': text,
                   'gen_seconds': round(gen_s, 2), **m}
            rows.append(m)
            fout.write(json.dumps(rec) + '\n')
            fout.flush()
            if (k + 1) % 10 == 0 or k == 0:
                el = time.time() - t0
                print(f'[{k+1}/{len(files)}] arr={m["arr"]:.3f} mrl={m["mrl"]:.0f} '
                      f'len={m["len_trimmed"]} {el/(k+1):.1f}s/q '
                      f'eta={(len(files)-k-1)*el/(k+1)/60:.0f}min', flush=True)

    summary = {'config': meta, 'aggregate': RM.aggregate(rows),
               'wall_seconds': round(time.time() - t0, 1),
               'end_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    if use_port:
        import mmada_cotapp as MC
        summary['port_coverage'] = dict(MC.STATS)
        print('[port] coverage', dict(MC.STATS), flush=True)
        bad = ((comp['ctar'] and MC.STATS['ctar_rows'] == 0) or
               (comp['dar_r'] > 0 and MC.STATS['dar_steps'] == 0) or
               (comp['ctae_mode'] != 'off' and MC.STATS['ctae_calls'] == 0) or
               (comp['ctev_mode'] != 'off' and MC.STATS['ctev_calls'] == 0))
        if bad:
            print('[port] a requested component never fired; refusing to write summary', flush=True)
            sys.exit(3)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary['aggregate'], indent=1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
