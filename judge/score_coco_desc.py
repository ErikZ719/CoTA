#!/usr/bin/env python
"""Score long-form COCO descriptions with an LLM judge (runs locally).

LLaVA-Bench has only 60 questions, too few to resolve differences of ~5 points. The
repeat-eval runs already hold hundreds of 512-token descriptions, so they are scored
here with the same judge. The judge is text-only, so the five human COCO captions stand
in for the image; each description is rated 1-10 for accuracy against them, level of
detail, and coherence. Scores are absolute (no reference answer), comparable across our
own configurations only.
"""
import argparse, json, os, re, ssl, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None

SYS = ("You are an impartial evaluator of image descriptions. You are given five short "
       "human captions of an image and a long description written by an assistant. Using "
       "the captions as the only evidence of what the image shows, rate the description "
       "from 1 to 10 for accuracy (no invented objects or attributes), level of detail, "
       "and coherence (penalize repetition and loops). Reply with a single number only.")

def judge(base, key, model, caps, desc, retries=5):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 8, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "[Captions]\n" + "\n".join(f"- {c}" for c in caps)
                                    + f"\n\n[Description]\n{desc}"}]}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
            m = re.findall(r"\d+(?:\.\d+)?", out)
            if m:
                return float(m[0])
        except Exception as e:
            if a == retries - 1:
                print(f"    judge failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        time.sleep(3 * (a + 1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', required=True, help='outputs.jsonl files (comma-separated) of one config')
    ap.add_argument('--captions', required=True, help='json {image_filename: [captions]}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--base', default='https://www.duckcoding.ai/v1')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    key = os.environ.get('JUDGE_API_KEY') or sys.exit('set JUDGE_API_KEY')
    caps = json.load(open(args.captions))
    rows = []
    for p in args.outputs.split(','):
        rows += [json.loads(l) for l in open(p)]
    rows = [r for r in rows if r['image'] in caps]
    with ThreadPoolExecutor(args.workers) as ex:
        scores = list(ex.map(lambda r: judge(args.base, key, args.model, caps[r['image']], r['text']), rows))
    per = [dict(image=r['image'], score=s) for r, s in zip(rows, scores) if s is not None]
    failed = len(rows) - len(per)
    if failed > max(2, int(0.05 * len(rows))):
        sys.exit(f'judge failed on {failed}/{len(rows)} items -- summary NOT written; rerun')
    summ = dict(model_judge=args.model, n=len(per), mean=sum(x['score'] for x in per) / max(len(per), 1))
    json.dump({'summary': summ, 'per_item': per}, open(args.out, 'w'), indent=1)
    print(json.dumps(summ))

if __name__ == '__main__':
    main()
