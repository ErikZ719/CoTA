"""Repeat-Curse metrics, computed on token-id sequences.

Definitions follow the paper (TPAMI CoTA++, Sec. IV-A):
  ARR      mean_i 1(y_i == y_{i+1}) over i = 1..L-1   (adjacent repetition rate)
  repeat position: the latter position of an adjacent identical pair
  runs     lengths r_k >= 2 of maximal runs of identical tokens
  MRL      max r_k          (0 if no runs)
  ARL      mean r_k         (0 if no runs)
  p95RL    95th percentile of r_k (0 if no runs)
  seq-rep-n   1 - unique_ngrams/total_ngrams   (0 if fewer than n tokens)
  distinct-n  unique_ngrams/total_ngrams       (1.0 if fewer than n tokens)

EOS handling: the decoded suffix has fixed length; the response is trimmed at
the FIRST eot token (exclusive) before computing metrics. Raw (untrimmed) ARR
is also reported for sanity checking.
"""
from typing import List, Dict


# Layout tokens for the LLaDA-V vocabulary: newline, space, tab and the
# common leading-space/indent pieces. Repetition of these is formatting, not
# degeneration, so the adjacent-run family is computed with them removed.
LAYOUT_IDS = {198, 220, 197, 256, 262144}


def strip_layout(ids: List[int]) -> List[int]:
    return [t for t in ids if t not in LAYOUT_IDS]


def trim_at_eot(ids: List[int], eot_id) -> List[int]:
    """Trim at the first occurrence of any terminator id (int or iterable of ints)."""
    eot_ids = {eot_id} if isinstance(eot_id, int) else set(eot_id)
    for i, t in enumerate(ids):
        if t in eot_ids:
            return list(ids[:i])
    return list(ids)


def arr(ids: List[int]) -> float:
    if len(ids) < 2:
        return 0.0
    return sum(ids[i] == ids[i + 1] for i in range(len(ids) - 1)) / (len(ids) - 1)


def run_lengths(ids: List[int]) -> List[int]:
    runs, i = [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[i]:
            j += 1
        if j > i:
            runs.append(j - i + 1)
        i = j + 1
    return runs


def _quantile(sorted_vals: List[float], q: float) -> float:
    # linear interpolation, matches numpy default
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def seq_rep_n(ids: List[int], n: int) -> float:
    total = len(ids) - n + 1
    if total <= 0:
        return 0.0
    grams = set()
    for i in range(total):
        grams.add(tuple(ids[i:i + n]))
    return 1.0 - len(grams) / total


def sample_metrics(raw_ids: List[int], eot_id) -> Dict:
    ids_full = trim_at_eot(raw_ids, eot_id)
    ids = strip_layout(ids_full)          # adjacent-run family: content tokens only
    runs = sorted(run_lengths(ids))
    runs_layout = sorted(run_lengths(ids_full))
    return {
        'arr_with_layout': arr(ids_full),
        'mrl_with_layout': float(runs_layout[-1]) if runs_layout else 0.0,
        'len_content': len(ids),
        'len_raw': len(raw_ids),
        'len_trimmed': len(ids_full),
        'arr': arr(ids),
        'arr_raw': arr(raw_ids),
        'n_runs': len(runs),
        'mrl': float(runs[-1]) if runs else 0.0,
        'arl': (sum(runs) / len(runs)) if runs else 0.0,
        'p95rl': _quantile(runs, 0.95),
        'seq_rep_2': seq_rep_n(ids_full, 2),
        'seq_rep_4': seq_rep_n(ids_full, 4),
        'distinct_1': (len(set(ids_full)) / len(ids_full)) if ids_full else 1.0,
        'distinct_2': 1.0 - seq_rep_n(ids_full, 2) if len(ids_full) >= 2 else 1.0,
    }


def aggregate(rows: List[Dict]) -> Dict:
    n = len(rows)
    if n == 0:
        return {}
    def mean(key):
        return sum(r[key] for r in rows) / n
    rep_rows = [r for r in rows if r['arr'] > 0]
    return {
        'n_samples': n,
        'SRR': len(rep_rows) / n,
        'SRR_with_layout': sum(1 for r in rows if r.get('arr_with_layout', 0) > 0) / n,
        'ARR_with_layout_pct': 100.0 * (sum(r.get('arr_with_layout', 0) for r in rows) / n),
        'MRL_max_with_layout': max((r.get('mrl_with_layout', 0) for r in rows), default=0.0),
        'ARR_mean': mean('arr'),
        'ARR_mean_pct': 100.0 * mean('arr'),
        'MRL_max': max(r['mrl'] for r in rows),
        'MRL_mean_over_repeat_samples':
            (sum(r['mrl'] for r in rep_rows) / len(rep_rows)) if rep_rows else 0.0,
        'ARL_mean_over_repeat_samples':
            (sum(r['arl'] for r in rep_rows) / len(rep_rows)) if rep_rows else 0.0,
        'p95RL_mean_over_repeat_samples':
            (sum(r['p95rl'] for r in rep_rows) / len(rep_rows)) if rep_rows else 0.0,
        'seq_rep_2_mean': mean('seq_rep_2'),
        'seq_rep_4_mean': mean('seq_rep_4'),
        'distinct_1_mean': mean('distinct_1'),
        'distinct_2_mean': mean('distinct_2'),
        'len_trimmed_mean': mean('len_trimmed'),
        'arr_raw_mean': mean('arr_raw'),
    }


if __name__ == '__main__':
    # self-test
    eot = 99
    assert trim_at_eot([1, 2, eot, 5, 5], eot) == [1, 2]
    assert trim_at_eot([1, 2, 98, 5, eot], {98, eot}) == [1, 2]
    assert arr([1, 1, 2]) == 0.5
    assert run_lengths([1, 1, 1, 2, 3, 3]) == [3, 2]
    assert seq_rep_n([1, 2, 1, 2], 2) == 1.0 - 2 / 3
    m = sample_metrics([7, 7, 7, 2, eot, 7, 7], eot)
    assert m['len_trimmed'] == 4 and m['mrl'] == 3.0 and m['n_runs'] == 1
    assert abs(m['arr'] - 2 / 3) < 1e-9
    m2 = sample_metrics([1, 2, 3, 4], eot)
    assert m2['arr'] == 0.0 and m2['mrl'] == 0.0 and m2['p95rl'] == 0.0
    agg = aggregate([m, m2])
    assert agg['SRR'] == 0.5
    print('repeat_metrics self-test OK')
