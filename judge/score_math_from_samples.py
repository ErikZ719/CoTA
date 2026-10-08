#!/usr/bin/env python
"""Score the saved MathVista / MathVerse responses with the official lmms-eval scorers, offline.

The benchmark runs were generated with the GPT-based scorer pointed at an unreachable address (bench_eval.sh), so
their results json carries no score. This script replays exactly the scoring code of the LLaDA-V lmms-eval fork
(`lmms_eval/tasks/mathvista/mathvista_evals.py`, `.../mathverse/mathverse_evals.py`) on the logged samples, with
our gateway and judge model instead of GPT-4o:
  MathVista : extract_answer (few-shot GPT extraction, quick_extract=False) -> normalize_extracted_answer -> safe_equal
  MathVerse : last 30 words of the response (trunk_response=30) -> extract_answer -> score_answer (GPT judgement 0/1,
              quick_match=False); the official score_answer loops forever on a non-0/1 reply, so it is re-implemented
              here with the same prompts and a bounded number of attempts.
Per-item results are cached in <out>/cache.jsonl (config|task|doc_id), so the script can be interrupted and re-run.
A (config, task) cell is written only if at most 5 % of its items failed.

Results root: T6_BENCH_ROOT (bench | bench_lavida | bench_mmada); the output goes to results/bench_scores[_<model>]/.
Runs on dgx056 from the home tree. The key comes from stdin through run_math_scoring.sh, never from a file:
  printf '%s\n' "$KEY" | ssh <loaner-machine> 'bash .../judge_t6/run_math_scoring.sh --detach'
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

E = os.environ.get("T6_E", "/home/<user>/zhaoqiyan/autodl-tmp/experiments/repeat_eval")
LMMS = os.environ.get("T6_LMMS", "/home/<user>/zhaoqiyan/autodl-tmp/LLaDA-V/eval/lmms-eval")
MODEL, BASE = "gpt-5.6-luna", "https://www.duckcoding.ai/v1/chat/completions"
CONFIGS = ["van", "cache", "cota", "cpp"]
TASKS = {"mathvista": ["mathvista_testmini_cot_local"],
         "mathverse": ["mathverse_testmini_vision_dominant_local", "mathverse_testmini_vision_intensive_local",
                       "mathverse_testmini_vision_only_local"]}


BENCH_ROOT = os.environ.get("T6_BENCH_ROOT", "bench")   # bench | bench_lavida | bench_mmada


def latest_samples(cfg, task):
    d = os.path.join(E, "results", BENCH_ROOT, cfg, task)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".jsonl") and "_samples_" in f) if os.path.isdir(d) else []
    return os.path.join(d, fs[-1]) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(E, "results", "bench_scores" if BENCH_ROOT == "bench" else "bench_scores_" + BENCH_ROOT.replace("bench_", "")))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N items of every cell (pilot)")
    ap.add_argument("--only", default="", help="regex on 'cfg|task'")
    ap.add_argument("--summarize", action="store_true", help="no API calls, rebuild the summary from the cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    key = os.environ.get("JUDGE_API_KEY")
    if not a.summarize and not key:
        sys.exit("set JUDGE_API_KEY (stdin through run_math_scoring.sh)")
    os.environ["OPENAI_API_URL"] = BASE                       # read at import by both evaluator classes
    os.environ["OPENAI_API_KEY"] = key or "none"
    sys.path.insert(0, LMMS)
    from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator
    from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator, DEMO_PROMPT_SCORE
    mv = MathVistaEvaluator(api_key=key or "none", gpt_model=MODEL)
    me = MathVerseEvaluator(api_key=key or "none", gpt_model=MODEL)

    cache_p = os.path.join(a.out, "cache.jsonl")
    cache = {}
    if os.path.exists(cache_p):
        for l in open(cache_p):
            r = json.loads(l); cache[r["k"]] = r
    lock = threading.Lock(); fh = open(cache_p, "a")

    def mathvista_item(r):
        doc = r["doc"]; pred = r["filtered_resps"][0].strip()
        problem = dict(question_type=doc["question_type"], answer_type=doc["answer_type"], query=doc["query"],
                       choices=doc["choices"], answer=doc.get("answer"), precision=doc.get("precision", 0))
        ext = mv.extract_answer(pred, problem, False)
        if ext is None or ext == "":
            raise RuntimeError("empty extraction")
        norm = mv.normalize_extracted_answer(ext, problem["choices"], problem["question_type"], problem["answer_type"], problem["precision"])
        ok = mv.safe_equal(norm, problem["answer"]) if problem["answer"] is not None else False
        return dict(extraction=ext, prediction=str(norm), correct=bool(ok))

    def mathverse_item(r):
        doc = r["doc"]; full = r["filtered_resps"][0].strip()
        pred = " ".join(full.split(" ")[-30:])                 # trunk_response = 30 as in the released yaml
        ext = me.extract_answer(pred)
        if ext is None or ext == "":
            raise RuntimeError("empty extraction")
        prompt = me.create_match_prompt(DEMO_PROMPT_SCORE, doc.get("question_for_eval", ""), doc["answer"], ext)
        for _ in range(5):                                     # official score_answer: while True on the same prompt
            j = me.get_chat_response(prompt, temperature=0, max_tokens=8, n=1)
            j = (j or "").replace("Judgement:", "").strip()
            m = re.match(r"^\s*([01])\b", j)
            if m:
                return dict(extraction=ext, judgement=int(m.group(1)), correct=m.group(1) == "1")
        raise RuntimeError("no 0/1 judgement: %r" % j[:40])

    todo = []
    for cfg in CONFIGS:
        for fam, tasks in TASKS.items():
            for task in tasks:
                if a.only and not re.search(a.only, f"{cfg}|{task}"):
                    continue
                p = latest_samples(cfg, task)
                if not p:
                    print("no samples for", cfg, task); continue
                rows = [json.loads(l) for l in open(p)]
                if a.limit: rows = rows[:a.limit]
                for r in rows:
                    k = f"{cfg}|{task}|{r['doc_id']}"
                    if k not in cache:
                        todo.append((k, fam, r))
    print("items to score:", len(todo), "| cached:", len(cache), flush=True)

    done = [0, 0]; t0 = time.time()

    def work(item):
        k, fam, r = item
        try:
            res = mathvista_item(r) if fam == "mathvista" else mathverse_item(r)
            rec = dict(k=k, ok=True, **res)
        except Exception as e:
            rec = dict(k=k, ok=False, err="%s: %s" % (type(e).__name__, str(e)[:80]))
        with lock:
            if rec["ok"]:
                cache[k] = rec; fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            else:
                done[1] += 1
            done[0] += 1
            if done[0] % 100 == 0:
                el = time.time() - t0
                print("  %d/%d failed %d  %.1f items/min" % (done[0], len(todo), done[1], 60 * done[0] / el), flush=True)

    if not a.summarize and todo:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(work, todo))
    fh.close()

    summary = {"judge": MODEL, "cells": {}}
    for cfg in CONFIGS:
        for fam, tasks in TASKS.items():
            accs = []
            for task in tasks:
                p = latest_samples(cfg, task)
                if not p: continue
                rows = [json.loads(l) for l in open(p)]
                if a.limit: rows = rows[:a.limit]
                got = [cache.get(f"{cfg}|{task}|{r['doc_id']}") for r in rows]
                ok = [g for g in got if g]
                miss = len(got) - len(ok)
                if not ok or miss > 0.05 * len(got):
                    summary["cells"][f"{cfg}|{task}"] = dict(n=len(got), missing=miss, accuracy=None); continue
                acc = 100.0 * sum(1 for g in ok if g["correct"]) / len(ok)
                summary["cells"][f"{cfg}|{task}"] = dict(n=len(got), scored=len(ok), missing=miss, accuracy=acc)
                accs.append(acc)
            if fam == "mathverse" and len(accs) == 3:
                summary["cells"][f"{cfg}|mathverse_testmini_vision"] = dict(accuracy=sum(accs) / 3, note="mean of the three vision splits (lmms-eval group)")
    json.dump(summary, open(os.path.join(a.out, "math_scores.json"), "w"), indent=1)
    for k, v in summary["cells"].items():
        print("%-48s %s" % (k, "%.2f" % v["accuracy"] if v.get("accuracy") is not None else "-- (missing %d)" % v.get("missing", 0)))


if __name__ == "__main__":
    main()
