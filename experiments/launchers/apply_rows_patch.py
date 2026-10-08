#!/usr/bin/env python
"""Adds --decode_rows DIR to run_repeat_eval.py (2026-09-30): with it, every image's decode-moment attention rows
(decode_rows_recorder.DecodeRows, deep band 25-32 = indices 24..31) are saved to DIR/rows_<stem>.npz. Off by
default; the generation itself is untouched. Anchored line-level edits, refuses to run twice."""
import io, os, shutil, sys, time

RUNNER = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts/run_repeat_eval.py"
BAK = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/patches/exp3_2026-09-30"

E1_OLD = """    ap.add_argument('--ngram', type=int, default=0, help='n-gram penalty baseline: forbid completing a seen n-gram; 0 = off')
"""
E1_NEW = """    ap.add_argument('--ngram', type=int, default=0, help='n-gram penalty baseline: forbid completing a seen n-gram; 0 = off')
    ap.add_argument('--decode_rows', default='', help='ROWS-PATCH: save decode-moment attention rows (layers 25-32) per image to this dir')
"""
E2_OLD = """    conv_template = 'llava_llada'
    question = DEFAULT_IMAGE_TOKEN + '\\n' + args.prompt
"""
E2_NEW = """    # ROWS-PATCH: decode-moment attention rows (analysis only)
    _rows_rec = None
    if args.decode_rows:
        from decode_rows_recorder import DecodeRows
        _rows_hook = None
        if args.mode == 'dllm_cache':
            from llava.hooks import cache_hook_LLaDA_V as _rows_hook
        elif '_sf_sampler' in locals():
            from llava.hooks import sf_cotapp as _sfc_r
            _rows_hook = _sfc_r.H
        _rows_rec = DecodeRows(model, core=model.model, layers_module=model.model.layers, G=args.length,
                               band=(24, 31), mask_id=126336, hook_mod=_rows_hook)
        os.makedirs(args.decode_rows, exist_ok=True)
        print(f'[rows] decode-moment recorder on, hook={getattr(_rows_hook, "__name__", None)} -> {args.decode_rows}', flush=True)

    conv_template = 'llava_llada'
    question = DEFAULT_IMAGE_TOKEN + '\\n' + args.prompt
"""
E3_OLD = """            t_gen = time.time()
            with torch.no_grad():
                cont = model.generate(
"""
E3_NEW = """            t_gen = time.time()
            if _rows_rec is not None:
                _rows_rec.attach()
            with torch.no_grad():
                cont = model.generate(
"""
E4_OLD = """            gen_s = time.time() - t_gen
"""
E4_NEW = """            gen_s = time.time() - t_gen
            if _rows_rec is not None:
                _rows_rec.detach()
                _rows_info = _rows_rec.save(os.path.join(args.decode_rows, 'rows_%s.npz' % os.path.splitext(fname)[0]),
                                            cont[0].tolist())
                print('[rows]', fname, _rows_info, flush=True)
"""

if __name__ == "__main__":
    src = io.open(RUNNER, encoding="utf-8").read()
    if "ROWS-PATCH" in src:
        sys.exit("already patched")
    for old, new in [(E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (E3_OLD, E3_NEW), (E4_OLD, E4_NEW)]:
        assert src.count(old) == 1, "anchor not unique:\n" + old
    if "--check" in sys.argv:
        print("anchors ok"); sys.exit(0)
    shutil.copy(RUNNER, os.path.join(BAK, "run_repeat_eval.py.pre_rows"))
    for old, new in [(E1_OLD, E1_NEW), (E2_OLD, E2_NEW), (E3_OLD, E3_NEW), (E4_OLD, E4_NEW)]:
        src = src.replace(old, new, 1)
    io.open(RUNNER, "w", encoding="utf-8").write(src)
    shutil.copy(RUNNER, os.path.join(BAK, "run_repeat_eval.py.patched_rows"))
    io.open(os.path.join(BAK, "APPLIED.txt"), "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " applied by apply_rows_patch.py\n")
    print("patched", RUNNER)
