import re, os
T = "/data/zhaoqiyan/autodl-tmp/LLaDA-V/eval/lmms-eval/lmms_eval/tasks"; D = "/data/zhaoqiyan/autodl-tmp/datasets"
def make(src, dst, old, new, split, path):
    s = open(f"{T}/{src}").read()
    s = re.sub(r"^dataset_path:.*$", "dataset_path: parquet", s, flags=re.M)
    s = re.sub(r"^dataset_name:.*\n", "", s, flags=re.M)
    s = re.sub(r"^dataset_kwargs:\n(  .*\n)+", f"dataset_kwargs:\n  data_files:\n    {split}: {path}\n", s, flags=re.M)
    s = re.sub(rf'^task:\s*"?{old}"?\s*$', f'task: "{new}"', s, flags=re.M)
    open(f"{T}/{dst}", "w").write(s); print("task", new)
make("mathvista/mathvista_testmini_cot.yaml", "mathvista/mathvista_testmini_cot_local.yaml",
     "mathvista_testmini_cot", "mathvista_testmini_cot_local", "testmini", f"{D}/MathVista/testmini.parquet")
for v in ("vision_dominant", "vision_intensive", "vision_only"):
    make(f"mathverse/mathverse_testmini_{v}.yaml", f"mathverse/mathverse_testmini_{v}_local.yaml",
         f"mathverse_testmini_{v}", f"mathverse_testmini_{v}_local", v, f"{D}/MathVerse/testmini_{v}.parquet")
# plain parquet delivers images as bytes or {'bytes': ...}: make both visual accessors tolerant
for mod, fn in (("mathverse/utils.py", "mathverse_doc_to_visual"), ("mathvista/utils.py", "mathvista_doc_to_visual")):
    p = f"{T}/{mod}"; s = open(p).read()
    if "_as_pil(" in s: continue
    helper = ('\n\ndef _as_pil(img):\n    """plain-parquet loading (the *_local tasks) yields raw bytes instead of a decoded image"""\n'
              '    import io\n    from PIL import Image\n    if isinstance(img, dict) and img.get("bytes") is not None:\n        img = img["bytes"]\n'
              '    if isinstance(img, (bytes, bytearray)):\n        img = Image.open(io.BytesIO(img))\n    return img\n')
    m = re.search(rf"def {fn}\(doc\):\n((?:    .*\n)+)", s)
    body = m.group(1)
    newbody = re.sub(r'doc\["(decoded_image|image)"\]', r'_as_pil(doc["\1"])', body)
    s = s.replace(m.group(0), f"def {fn}(doc):\n{newbody}") ; s = s.replace(f"def {fn}(doc):", helper.strip("\n") + f"\n\n\ndef {fn}(doc):", 1)
    open(p, "w").write(s); print("patched", mod, "->", newbody.strip().replace("\n", " | ")[:120])
# the launcher: scoring APIs are unreachable from the server, make them fail at once and keep the responses
SH = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval/scripts/launchers/bench_eval.sh"; s = open(SH).read()
if "OPENAI_API_URL" not in s:
    s = s.replace('OUT=$E/results/', '# GPT-based scorers (MathVista, MathVerse, MMBench fallback) cannot reach a gateway from here: fail fast,\n'
                  '# keep the logged responses, and score them later on the machine that has the judge key.\n'
                  'export OPENAI_API_URL=http://127.0.0.1:9/v1/chat/completions OPENAI_API_KEY=none\nOUT=$E/results/', 1)
    open(SH, "w").write(s); print("launcher: scorer API set to fail fast")
