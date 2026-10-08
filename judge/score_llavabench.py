#!/usr/bin/env python
"""Score LLaVA-Bench answers with an LLM judge (runs locally; the experiment host has no
route to the API).

Protocol follows LLaVA-Bench: the judge sees the image context (caption), the question,
the reference answer and the model answer, and rates both 1-10; the reported figure is
the model's score relative to the reference. The judge is configurable because the
gateway available here serves no GPT-4 class model -- scores are therefore comparable
across our own configurations, not against published LLaVA-Bench numbers.
"""
import argparse, json, os, re, ssl, sys, time
import urllib.request

try:                                    # some interpreters ship without a usable CA store
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None

SYS = ("You are an impartial evaluator of multimodal assistants. You are given a "
       "description of an image, a question about it, a reference answer and an "
       "assistant's answer. Rate the helpfulness, relevance, accuracy and level of "
       "detail of each answer on a scale of 1 to 10. Reply with exactly two numbers "
       "on one line: the reference score, then the assistant score, space separated. "
       "No other text.")

def judge(base, key, model, ctx, q, ref, ans, retries=4):
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": 32,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": (f"[Image description]\n{ctx}\n\n[Question]\n{q}\n\n"
                                         f"[Reference answer]\n{ref}\n\n[Assistant answer]\n{ans}")}]
    }).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
            nums = re.findall(r"\d+(?:\.\d+)?", out)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
        except Exception as e:
            if a == retries - 1:
                print(f"    judge failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            time.sleep(2 * (a + 1))
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', required=True, help='directory holding answers.jsonl')
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--base', default='https://www.duckcoding.ai/v1')
    args = ap.parse_args()
    key = os.environ.get('JUDGE_API_KEY')
    if not key:
        sys.exit('set JUDGE_API_KEY')

    rows = [json.loads(l) for l in open(f'{args.answers}/answers.jsonl')]
    scored, by_cat = [], {}
    for k, r in enumerate(rows):
        ctx = r.get('caption') or '(no description available)'
        sref, sans = judge(args.base, key, args.model, ctx, r['question'], r['reference'], r['answer'])
        if sref is None or not sref:
            continue
        rel = 100.0 * sans / sref
        scored.append(dict(question_id=r['question_id'], category=r['category'],
                           ref=sref, ans=sans, relative=rel))
        by_cat.setdefault(r['category'], []).append(rel)
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(rows)}', flush=True)
    failed = len(rows) - len(scored)
    if failed > max(2, int(0.05 * len(rows))):
        # Never let a mostly-failed run masquerade as a result: no summary, non-zero exit.
        sys.exit(f'judge failed on {failed}/{len(rows)} items -- summary NOT written; rerun')
    overall = sum(s['relative'] for s in scored) / max(len(scored), 1)
    res = dict(model_judge=args.model, n=len(scored), overall_relative=overall,
               by_category={c: sum(v) / len(v) for c, v in by_cat.items()})
    json.dump({'summary': res, 'per_item': scored},
              open(f'{args.answers}/judge_{args.model}.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
