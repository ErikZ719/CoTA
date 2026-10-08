#!/usr/bin/env python
"""LLM-judge score for every Table V cell (the Judge column of Table VI). Runs on the local Mac only:
the gateway is reachable through the system proxy. Since 2026-09-22 the server reaches it too: see README.md.

Input   ../server_mirror/results/table5_texts/<cell>.jsonl   the exact texts the repetition metrics were
        computed on (truncated at the first end-of-text token), written on the server by
        scripts/analysis/tab5_dump_texts.py from table5_provenance.json (exact run directories, no globs).
        ../coco_desc/coco500_captions.json                    five human captions per image (the judge is text-only).
Judge   the prompt and the request are imported from code-final/eval/score_coco_desc.py, so the scale is the one
        already used for the absolute scores of the description study: 1-10 for accuracy against the captions,
        level of detail and coherence, gpt-5.6-luna, temperature 0.
Output  judge/cache.jsonl      append-only, one line per distinct (image, text): identical texts are judged once
                               and share their score across cells. Safe to interrupt and rerun.
        judge_cells.json       per cell: n, mean, bootstrap 95% interval. A cell is written only if at most 5% of
                               its items failed. make_table5.py reads this file.

The key is read from the environment variable JUDGE_API_KEY and is never written anywhere.

  JUDGE_API_KEY=... /opt/anaconda3/bin/python judge_table6.py --limit 20 --cells 'LLaDA-V\\|.*\\|128'   # pilot
  JUDGE_API_KEY=... /opt/anaconda3/bin/python judge_table6.py                                          # everything
  /opt/anaconda3/bin/python judge_table6.py --summarize                                                # no calls
"""
import argparse, hashlib, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# Paths default to the local layout. On the server the same script runs from experiments/repeat_eval/judge_t6/
# with T6_* set by run_on_server.sh (the cache file is portable between the two).
E = os.environ.get
sys.path.insert(0, E("T6_SCORER", os.path.join(ROOT, "code-final", "eval")))
import score_coco_desc as SC                                  # judge(base, key, model, caps, desc)

TEXTS = E("T6_TEXTS", os.path.join(HERE, "..", "server_mirror", "results", "table5_texts"))
CAPS = E("T6_CAPS", os.path.join(HERE, "..", "coco_desc", "coco500_captions.json"))
EVAL = E("T6_EVAL", os.path.join(HERE, "..", "server_mirror", "data", "coco500_final.json"))
CACHE = E("T6_CACHE", os.path.join(HERE, "judge", "cache.jsonl"))
OUT = E("T6_OUT", os.path.join(HERE, "judge_cells.json"))
MODEL, BASE = "gpt-5.6-luna", "https://www.duckcoding.ai/v1"


def sha(image, text):
    return hashlib.sha1((image + "\x00" + text).encode()).hexdigest()


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l)
            c[r["h"]] = r["score"]
    return c


def boot(xs, n=2000, seed=0):
    rnd = random.Random(seed); k = len(xs)
    ms = sorted(sum(xs[rnd.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=".*", help="regex on the cell key 'Model|Method|L'")
    ap.add_argument("--limit", type=int, default=0, help="first N images of the evaluation list (0 = all 500)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summarize", action="store_true", help="write judge_cells.json from the cache, no calls")
    a = ap.parse_args()

    INDEX = json.load(open(os.path.join(TEXTS, "INDEX.json")))
    if "cells" not in INDEX:
        sys.exit("table5_texts/INDEX.json has no provenance stamp: re-run scripts/analysis/tab5_dump_texts.py")
    index = INDEX["cells"]
    PROV = os.environ.get("T6_PROV", os.path.join(TEXTS, "..", "table5_provenance.json"))
    prov = json.load(open(PROV))
    if prov["built"] != INDEX["provenance_built"]:
        sys.exit("texts were dumped from provenance built %s but the current provenance is %s: re-run tab5_dump_texts.py"
                 % (INDEX["provenance_built"], prov["built"]))
    bad = [k for k, c in prov["cells"].items() if [r["dir"] for r in c["runs"]] != INDEX["runs"].get(k)]
    if bad:
        sys.exit("texts of %d cells come from other runs than the current provenance (e.g. %s): re-run tab5_dump_texts.py" % (len(bad), bad[0]))
    caps = json.load(open(CAPS))
    order = json.load(open(EVAL))["files"]
    keep = set(order[:a.limit] if a.limit else order)
    cells = {}
    for key, v in index.items():
        rows = [json.loads(l) for l in open(os.path.join(TEXTS, v["file"]))]
        cells[key] = [r for r in rows if r["image"] in keep]
    cache = load_cache()

    if not a.summarize:
        key = os.environ.get("JUDGE_API_KEY") or sys.exit("set JUDGE_API_KEY in the environment")
        todo = {}
        for k, rows in cells.items():
            if re.search(a.cells, k):
                for r in rows:
                    h = sha(r["image"], r["text"])
                    if h not in cache:
                        todo[h] = r
        print("cells selected:", sum(1 for k in cells if re.search(a.cells, k)), "| items to judge:", len(todo),
              "| already cached:", len(cache), flush=True)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        lock = threading.Lock(); done = [0, 0]; t0 = time.time()
        fh = open(CACHE, "a")

        def work(item):
            h, r = item
            s = SC.judge(BASE, key, MODEL, caps[r["image"]], r["text"])
            with lock:
                if s is None:
                    done[1] += 1
                else:
                    cache[h] = s
                    fh.write(json.dumps(dict(h=h, image=r["image"], score=s)) + "\n"); fh.flush()
                done[0] += 1
                if done[0] % 200 == 0:
                    el = time.time() - t0
                    print("  %d/%d  failed %d  %.1f items/min  eta %.1f h" % (
                        done[0], len(todo), done[1], 60 * done[0] / el, (len(todo) - done[0]) * el / done[0] / 3600),
                        flush=True)

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, todo.items()))
        fh.close()
        print("judged", done[0] - done[1], "| failed", done[1])

    out = {}
    for k, rows in cells.items():
        sc = [cache.get(sha(r["image"], r["text"])) for r in rows]
        got = [s for s in sc if s is not None]
        miss = len(sc) - len(got)
        if not got or miss > 0.05 * len(sc):
            continue                                            # incomplete cell: not reported
        lo, hi = boot(got)
        out[k] = dict(n=len(got), missing=miss, mean=sum(got) / len(got), ci95=[lo, hi])
    json.dump(dict(judge=MODEL, temperature=0, scale="1-10", images=len(keep), provenance_built=prov["built"], cells=out), open(OUT, "w"), indent=1)
    print("cells with a score: %d / %d  ->  %s" % (len(out), len(cells), OUT))


if __name__ == "__main__":
    main()
