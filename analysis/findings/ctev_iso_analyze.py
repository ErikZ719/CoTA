#!/usr/bin/env python
'''Do CTEV runs commit more positions that have no committed neighbour within +-5?
Reads the decoding traces written by run_repeat_eval.py --case_trace 1 (one JSON per image).
For every commit the set of committed positions is rebuilt from the trace itself, so the count of
committed neighbours does not depend on how the context entropy was filled in.
usage: ctev_iso_analyze.py <results dir with tr_*/> [w]'''
import json, os, sys, glob, numpy as np
from scipy import stats
root = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EOT = {126081, 126348}; LAYOUT = {198, 220, 197, 256, 262144}
def repeats(g):
    cut = next((k for k, x in enumerate(g) if int(x) in EOT), len(g))
    seq = [(k, int(x)) for k, x in enumerate(g[:cut]) if int(x) not in LAYOUT]
    return set(seq[k][0] for k in range(1, len(seq)) if seq[k][1] == seq[k - 1][1])
def load(tag):
    out = {}
    ids = {}
    p = os.path.join(root, tag, 'outputs.jsonl')
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); ids[os.path.splitext(r['image'])[0]] = r['ids']
    for f in sorted(glob.glob(os.path.join(root, tag, 'trace_*.json'))):
        t = json.load(open(f)); stem = os.path.splitext(t['image'])[0]
        rows = sorted(t['rows'], key=lambda r: (r['step'], r['pos']))
        done = set(); rec = []
        for r in rows:
            i = r['pos']; nb = sum(1 for j in done if 0 < abs(j - i) <= W)
            rec.append(dict(step=r['step'], pos=i, n=nb, conf=r['conf'], score=r['score'], E=r['E_ctx'], alt=r.get('alt_pos')))
            done.add(i)
        out[stem] = (rec, repeats(ids[stem]) if stem in ids else None)
    return out
T = {t: load(t) for t in ('tr_van', 'tr_cache', 'tr_ctev', 'tr_full') if os.path.isdir(os.path.join(root, t))}
common = sorted(set.intersection(*[set(v) for v in T.values()])) if T else []
print('images per condition:', {k: len(v) for k, v in T.items()}, '| common:', len(common))
per = {}
for tag, d in T.items():
    iso = []; tot = 0; isoN = 0; rep_iso = [0, 0]; rep_non = [0, 0]; late = 0; conf_iso = []; conf_non = []; redir = [0, 0, 0]
    for s in common:
        rec, reps = d[s]; k = 0
        for r in rec[1:]:                                   # the first commit has no neighbour by construction
            tot += 1
            if r['n'] == 0:
                k += 1; isoN += 1; conf_iso.append(r['conf'])
                if r['step'] >= len(rec) / 2: late += 1
                if reps is not None: rep_iso[0] += (r['pos'] in reps); rep_iso[1] += 1
            else:
                conf_non.append(r['conf'])
                if reps is not None: rep_non[0] += (r['pos'] in reps); rep_non[1] += 1
            if r['alt'] is not None and r['alt'] != r['pos']:
                redir[0] += 1; redir[1] += (r['n'] == 0)
            redir[2] += 1
        iso.append(k)
    per[tag] = np.array(iso)
    print('%-9s commits without a committed neighbour: %5d of %6d (%.2f%%) | per response mean %.2f median %.0f max %d | in the second half %d'
          % (tag, isoN, tot, 100 * isoN / max(1, tot), np.mean(iso), np.median(iso), max(iso) if iso else 0, late))
    print('          confidence at such commits: mean %.3f ; at the others %.3f' % (np.mean(conf_iso) if conf_iso else float('nan'), np.mean(conf_non)))
    if rep_iso[1]:
        print('          repeat tokens among them: %d of %d (%.2f%%) ; among the others %d of %d (%.2f%%)' % (rep_iso[0], rep_iso[1], 100 * rep_iso[0] / rep_iso[1], rep_non[0], rep_non[1], 100 * rep_non[0] / rep_non[1]))
    if redir[0]:
        print('          commits where the score overruled the confidence: %d of %d (%.1f%%), of which to a position without neighbour: %d (%.1f%%)' % (redir[0], redir[2], 100 * redir[0] / redir[2], redir[1], 100 * redir[1] / redir[0]))
for a, b in (('tr_cache', 'tr_ctev'), ('tr_cache', 'tr_full'), ('tr_van', 'tr_ctev'), ('tr_van', 'tr_full')):
    if a in per and b in per and len(common) >= 6:
        d = per[b] - per[a]
        try: p = stats.wilcoxon(per[b], per[a]).pvalue
        except ValueError: p = float('nan')
        print('%s against %s: per response %+.2f ; more in %d, fewer in %d, equal in %d of %d responses ; Wilcoxon p = %.2g' % (b, a, d.mean(), (d > 0).sum(), (d < 0).sum(), (d == 0).sum(), len(d), p))
