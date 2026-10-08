#!/usr/bin/env python
"""Deterministically sample N images from COCO val2014 for the repeat-curse eval.

Output JSON records seed, source dir, and the exact file list, so every later
run (baseline / cached / method) evaluates the identical image set.
"""
import argparse, json, os, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/data/zhaoqiyan/autodl-tmp/coco2014/val2014')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/data/coco500_seed42.json')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.src) if f.endswith('.jpg'))
    assert len(files) >= args.n, f'only {len(files)} images in {args.src}'
    picked = random.Random(args.seed).sample(files, args.n)
    picked.sort()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'seed': args.seed, 'n': args.n, 'src': args.src,
                   'total_pool': len(files), 'files': picked}, f, indent=1)
    print(f'wrote {args.n} files (seed={args.seed}, pool={len(files)}) -> {args.out}')

if __name__ == '__main__':
    main()
