"""Local variants of the lmms-eval tasks whose data came from ModelScope: same task definition, parquet files
read from /data/zhaoqiyan/autodl-tmp/datasets instead of the Hugging Face hub (same pattern as mme_local)."""
import re, os, glob
T = "/data/zhaoqiyan/autodl-tmp/LLaDA-V/eval/lmms-eval/lmms_eval/tasks"; D = "/data/zhaoqiyan/autodl-tmp/datasets"
def localize(text, split, pattern, drop_name=True):
    text = re.sub(r"^dataset_path:.*$", "dataset_path: parquet", text, flags=re.M)
    text = re.sub(r"^dataset_kwargs:\n(  .*\n)+", f"dataset_kwargs:\n  data_files:\n    {split}: {pattern}\n", text, flags=re.M)
    if drop_name: text = re.sub(r"^dataset_name:.*\n", "", text, flags=re.M)
    return text
jobs = [  # (source yaml, new yaml, old task name, new task name, split, files, template (src, dst) or None)
 ("chartqa/chartqa.yaml", "chartqa/chartqa_local.yaml", "chartqa", "chartqa_local", "test", f"{D}/ChartQA/data/test-*.parquet", None),
 ("seedbench/seedbench.yaml", "seedbench/seedbench_local.yaml", "seedbench", "seedbench_local", "test", f"{D}/SEED-Bench/data/test-*.parquet", None),
 ("docvqa/docvqa_val.yaml", "docvqa/docvqa_val_local.yaml", "docvqa_val", "docvqa_val_local", "validation", f"{D}/DocVQA/DocVQA/validation-*.parquet",
  ("docvqa/_default_template_docvqa_yaml", "docvqa/_default_template_docvqa_local_yaml")),
 ("mmbench/mmbench_en_dev.yaml", "mmbench/mmbench_en_dev_local.yaml", "mmbench_en_dev", "mmbench_en_dev_local", "dev", f"{D}/MMBench/en/dev-*.parquet",
  ("mmbench/_default_template_mmbench_en_yaml", "mmbench/_default_template_mmbench_en_local_yaml")),
]
for src, dst, old, new, split, files, tpl in jobs:
    s = open(f"{T}/{src}").read()
    s = re.sub(rf'^task:\s*"?{old}"?\s*$', f'task: "{new}"', s, flags=re.M)
    if tpl:
        t = localize(open(f"{T}/{tpl[0]}").read(), split, files)
        open(f"{T}/{tpl[1]}", "w").write(t)
        s = s.replace(f"include: {os.path.basename(tpl[0])}", f"include: {os.path.basename(tpl[1])}")
    else:
        s = localize(s, split, files)
    open(f"{T}/{dst}", "w").write(s)
    n = len(glob.glob(files)); print(f"{new:24s} <- {n} parquet file(s)")
