#!/usr/bin/env python
"""LLM-judge score for runs outside Table VII (2026-09-30: the n-gram baseline of the main text), with exactly the
Table VII pipeline: texts decoded as tab5_dump_texts.py does (cut at the first end-of-text token, MMaDA tokenizer,
special tokens skipped), judged by score_coco_desc.judge (gpt-5.6-luna, temperature 0, 1-10 against the five human
captions), cached in the shared judge cache (sha1 of image + text, so identical texts share their score).

  JUDGE_API_KEY=... python judge_extra.py --label ng2 --runs 'lladav/exp3/ng2_128_*' [--label ... --runs ...]
  python judge_extra.py --summarize                     # no calls; rebuild extra_cells.json from the cache
Output: judge_t6/extra_cells.json  {label: {n, missing, mean, ci95}}. The key is read from the environment only.
"""
import argparse, glob, hashlib, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HOME", "/data/zhaoqiyan/autodl-tmp/hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
E = "/data/zhaoqiyan/autodl-tmp/experiments/repeat_eval"
sys.path.insert(0, E + "/scripts"); sys.path.insert(0, E + "/judge_t6")
import repeat_metrics as RM
import score_coco_desc as SC
CACHE, OUT = E + "/judge_t6/cache.jsonl", E + "/judge_t6/extra_cells.json"
CAPS, FILES = E + "/judge_t6/coco500_captions.json", E + "/data/coco500_final.json"
MODEL, BASE = "gpt-5.6-luna", "https://www.duckcoding.ai/v1"
EOT = {126081, 126348}


def sha(image, text):
    return hashlib.sha1((image + "\x00" + text).encode()).hexdigest()


def boot(xs, seed=0, n=2000):
    rnd = random.Random(seed); k = len(xs); ms = []
    for _ in range(n):
        ms.append(sum(xs[rnd.randrange(k)] for _ in range(k)) / k)
    ms.sort(); return [ms[int(0.025 * n)], ms[int(0.975 * n)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--runs", action="append", default=[], help="glob under results/, one per --label")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    assert len(a.label) == len(a.runs)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-Base", trust_remote_code=True)
    keep = set(json.load(open(FILES))["files"])
    cells = {}
    for lab, pat in zip(a.label, a.runs):
        d = {}
        for p in sorted(glob.glob("%s/results/%s/outputs.jsonl" % (E, pat))):
            for l in open(p):
                r = json.loads(l)
                if r["image"] in keep:
                    ids = RM.trim_at_eot(r["ids"], EOT)
                    d[r["image"]] = tk.decode(ids, skip_special_tokens=True).strip()
        cells[lab] = d
        print(lab, pat, "responses:", len(d), flush=True)
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l); cache[r["h"]] = r["score"]
    if not a.summarize:
        key = os.environ.get("JUDGE_API_KEY") or sys.exit("set JUDGE_API_KEY")
        caps = json.load(open(CAPS))
        todo = {}
        for lab, d in cells.items():
            for im, txt in d.items():
                h = sha(im, txt)
                if h not in cache:
                    todo[h] = (im, txt)
        print("items to judge:", len(todo), "| cached:", len(cache), flush=True)
        lock = threading.Lock(); done = [0, 0]; t0 = time.time(); fh = open(CACHE, "a")

        def work(item):
            h, (im, txt) = item
            s = SC.judge(BASE, key, MODEL, caps[im], txt)
            with lock:
                if s is None:
                    done[1] += 1
                else:
                    cache[h] = s; fh.write(json.dumps(dict(h=h, image=im, score=s)) + "\n"); fh.flush()
                done[0] += 1
                if done[0] % 100 == 0:
                    el = time.time() - t0
                    print("  %d/%d failed %d  %.1f items/min" % (done[0], len(todo), done[1], 60 * done[0] / el), flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, todo.items()))
        fh.close()
        print("judged", done[0] - done[1], "| failed", done[1], flush=True)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for lab, d in cells.items():
        xs = [cache[sha(im, txt)] for im, txt in d.items() if sha(im, txt) in cache]
        miss = len(d) - len(xs)
        if xs and miss <= 0.05 * len(d):
            out[lab] = dict(n=len(d), missing=miss, mean=round(sum(xs) / len(xs), 3), ci95=[round(v, 3) for v in boot(xs)])
        print(lab, "n", len(d), "missing", miss, "mean", out.get(lab, {}).get("mean"), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
