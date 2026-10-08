#!/usr/bin/env python
"""Fetch the large benchmark datasets from ModelScope (28 MB/s here, against 0.2 MB/s through hf-mirror).
Files go to /data/zhaoqiyan/autodl-tmp/datasets/<name>/<path in repo>; sizes are verified against the listing.
Usage: python ms_download.py            (resumable: complete files are skipped)"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
ROOT = "/data/zhaoqiyan/autodl-tmp/datasets"
WANT = [("lmms-lab/ChartQA", "ChartQA", lambda p: p.startswith("data/")),
        ("lmms-lab/MMBench", "MMBench", lambda p: p.startswith("en/")),
        ("lmms-lab/DocVQA", "DocVQA", lambda p: p.startswith("DocVQA/validation")),
        ("lmms-lab/SEED-Bench", "SEED-Bench", lambda p: p.startswith("data/"))]
def tree(repo):
    url = f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision=master&Root=%2F&Recursive=True&PageSize=2000"
    d = json.loads(subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True, text=True).stdout)
    return [(f["Path"], f["Size"]) for f in d["Data"]["Files"] if f["Type"] != "tree"]
def fetch(job):
    repo, name, path, size = job; dst = os.path.join(ROOT, name, path)
    if os.path.exists(dst) and os.path.getsize(dst) == size: return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"
    for _ in range(6):
        subprocess.run(["curl", "-L", "-s", "-C", "-", "--max-time", "3600", "-o", dst + ".part", url])
        if os.path.exists(dst + ".part") and os.path.getsize(dst + ".part") == size:
            os.replace(dst + ".part", dst); return "ok"
        time.sleep(5)
    return "FAILED " + path
jobs = []
for repo, name, keep in WANT:
    fs = [(p, s) for p, s in tree(repo) if keep(p)]
    print(f"{repo}: {len(fs)} files, {sum(s for _, s in fs)/1e9:.2f} GB", flush=True)
    jobs += [(repo, name, p, s) for p, s in fs]
t0 = time.time(); done = 0
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(fetch, jobs):
        done += 1
        if r.startswith("FAILED") or done % 20 == 0 or done == len(jobs):
            print(f"[{time.strftime('%H:%M')}] {done}/{len(jobs)} {r}", flush=True)
bad = [j for j in jobs if not (os.path.exists(os.path.join(ROOT, j[1], j[2])) and os.path.getsize(os.path.join(ROOT, j[1], j[2])) == j[3])]
print("MS-DOWNLOAD-FINISHED", "all files verified by size" if not bad else f"{len(bad)} files incomplete", f"{(time.time()-t0)/60:.0f} min", flush=True)
