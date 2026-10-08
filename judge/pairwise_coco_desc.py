#!/usr/bin/env python
"""Pairwise judge for two configurations' COCO descriptions of the same image.

Absolute 1-10 scores tie on ~40% of items when two configurations are close, so small
differences are unresolvable. Here the judge sees both descriptions and picks the better
one. Each pair is judged twice with A/B swapped; a configuration wins an item only if it
wins both orders, otherwise the item is a tie. This cancels position bias and keeps
the protocol conservative.
"""
import argparse, json, os, random, re, ssl, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = None

SYS = ("You compare two descriptions of the same image. Five short human captions are the "
       "only evidence of what the image shows. Prefer the description that is more accurate "
       "(no invented objects or attributes), then more detailed, then more coherent (no "
       "repetition or loops). Reply with exactly one letter: A or B.")

def ask(base, key, model, caps, a, b, retries=6):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 4, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": "[Captions]\n" + "\n".join(f"- {c}" for c in caps)
                                    + f"\n\n[Description A]\n{a}\n\n[Description B]\n{b}"}]}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as r:
                out = json.load(r)["choices"][0]["message"]["content"].strip().upper()
            m = re.search(r"\b([AB])\b", out) or re.search(r"([AB])", out)
            if m:
                return m.group(1)
        except Exception as e:
            if k == retries - 1:
                print(f"    judge failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        time.sleep(10 * (k + 1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', required=True, help='outputs.jsonl files of config X (comma-separated)')
    ap.add_argument('--y', required=True, help='outputs.jsonl files of config Y (comma-separated)')
    ap.add_argument('--captions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='gpt-5.6-luna')
    ap.add_argument('--base', default='https://www.duckcoding.ai/v1')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    key = os.environ.get('JUDGE_API_KEY') or sys.exit('set JUDGE_API_KEY')
    caps = json.load(open(args.captions))
    load = lambda ps: {json.loads(l)['image']: json.loads(l)['text'] for p in ps.split(',') for l in open(p)}
    X, Y = load(args.x), load(args.y)
    ids = sorted(i for i in set(X) & set(Y) if i in caps)
    part = args.out + '.partial.jsonl'                   # verdicts survive a flaky gateway
    done = {}
    if os.path.exists(part):
        for l in open(part):
            r = json.loads(l); done[r['image']] = r
    plog = open(part, 'a')

    def one(i):
        if i in done:
            return done[i]
        if X[i] == Y[i]:
            r = dict(image=i, outcome='tie', identical=True)
            plog.write(json.dumps(r) + '\n'); plog.flush(); return r
        r1 = ask(args.base, key, args.model, caps[i], X[i], Y[i])     # X is A
        r2 = ask(args.base, key, args.model, caps[i], Y[i], X[i])     # Y is A
        if r1 is None or r2 is None:
            return dict(image=i, outcome=None)
        x_wins = (r1 == 'A') + (r2 == 'B')
        outcome = 'x' if x_wins == 2 else 'y' if x_wins == 0 else 'tie'
        r = dict(image=i, outcome=outcome, first=r1, swapped=r2)
        plog.write(json.dumps(r) + '\n'); plog.flush(); return r

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, ids))
    ok = [r for r in res if r['outcome'] is not None]
    failed = len(res) - len(ok)
    if failed > max(2, int(0.05 * len(res))):
        sys.exit(f'judge failed on {failed}/{len(res)} items -- summary NOT written; rerun')
    wx = sum(r['outcome'] == 'x' for r in ok); wy = sum(r['outcome'] == 'y' for r in ok)
    tie = len(ok) - wx - wy
    # position bias: how often the judge picked slot A regardless of content
    firsts = [r.get('first') for r in ok if 'first' in r] + [r.get('swapped') for r in ok if 'swapped' in r]
    pa = sum(f == 'A' for f in firsts) / max(len(firsts), 1)
    from math import comb
    n = wx + wy
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(wx, wy) + 1)) / 2 ** n) if n else 1.0
    summ = dict(model_judge=args.model, n=len(ok), x_wins=wx, y_wins=wy, ties=tie,
                sign_test_p=p, slot_A_rate=pa)
    json.dump({'summary': summ, 'per_item': ok}, open(args.out, 'w'), indent=1)
    print(json.dumps(summ))

if __name__ == '__main__':
    main()
