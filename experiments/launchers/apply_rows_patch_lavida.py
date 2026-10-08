#!/usr/bin/env python
"""Adds --decode_rows DIR to run_repeat_eval_lavida.py (2026-09-30): decode-moment attention rows (layers 25-32)
per image via decode_rows_recorder.DecodeRowsOLMo, dllm_cache mode only. Off by default. Refuses to run twice."""
import io, os, shutil, sys, time

RUNNER = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts/run_repeat_eval_lavida.py"
BAK = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/patches/exp3_2026-09-30"

E1_OLD = """    ap.add_argument('--ctev_norm', type=int, default=1)
    args = ap.parse_args()
"""
E1_NEW = """    ap.add_argument('--ctev_norm', type=int, default=1)
    ap.add_argument('--decode_rows', default='', help='ROWS-PATCH: save decode-moment attention rows (layers 25-32) per image to this dir')
    args = ap.parse_args()
"""
E2_OLD = """    conv_template = 'llada'
    question = DEFAULT_IMAGE_TOKEN + '\\n' + args.prompt
"""
E2_NEW = """    # ROWS-PATCH: decode-moment attention rows (analysis only, dllm_cache mode)
    _rows_rec = None
    if args.decode_rows:
        assert args.mode == 'dllm_cache', 'decode_rows: dllm_cache mode only'
        from decode_rows_recorder import DecodeRowsOLMo
        _rows_rec = DecodeRowsOLMo(llm, core=llm, blocks=llm.transformer.blocks, G=L, band=(24, 31),
                                   mask_id=mask_id, mc=(MC if use_port else None))
        os.makedirs(args.decode_rows, exist_ok=True)
        print(f'[rows] decode-moment recorder on, port={use_port} -> {args.decode_rows}', flush=True)

    conv_template = 'llada'
    question = DEFAULT_IMAGE_TOKEN + '\\n' + args.prompt
"""
E3_OLD = """                else:
                    gen = generate(inputs_embeds.to(torch.bfloat16))
                gen_s = time.time() - t_gen
"""
E3_NEW = """                else:
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
"""

if __name__ == "__main__":
    src = io.open(RUNNER, encoding="utf-8").read()
    if "ROWS-PATCH" in src:
        sys.exit("already patched")
    for old, new in [(E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (E3_OLD, E3_NEW)]:
        assert src.count(old) == 1, "anchor not unique:\n" + old
    if "--check" in sys.argv:
        print("anchors ok"); sys.exit(0)
    os.makedirs(BAK, exist_ok=True)
    shutil.copy(RUNNER, os.path.join(BAK, "run_repeat_eval_lavida.py.pre_rows"))
    for old, new in [(E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (E3_OLD, E3_NEW)]:
        src = src.replace(old, new, 1)
    io.open(RUNNER, "w", encoding="utf-8").write(src)
    shutil.copy(RUNNER, os.path.join(BAK, "run_repeat_eval_lavida.py.patched_rows"))
    io.open(os.path.join(BAK, "APPLIED.txt"), "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " applied by apply_rows_patch_lavida.py\n")
    print("patched", RUNNER)
