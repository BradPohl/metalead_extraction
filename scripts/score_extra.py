"""Extra views of a run, using the same edges and one-to-one matching as score.py.

    python scripts/score_extra.py --pred runs/x --require-value
    python scripts/score_extra.py --pred runs/x --papers 1812.09471 1812.10235

Prints, for the chosen papers:
  - pooled P/R/F1 (what score.py prints, restricted to --papers)
  - macro P/R/F1: the mean of per-paper values, as in the MetaLead paper
  - ET-Accuracy: share of matched pairs whose experiment_type equals the gold Experiment-Type,
    pooled and averaged per paper (records without an experiment_type count as wrong)
  - pooled P/R/F1 per group, when the run folder has run_meta.json with "groups"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

sys.path.insert(0, str(Path(__file__).parent))
import score  # noqa: E402


def matched_pairs(tuples, gold):
    """(gold index, record index) pairs of a maximum one-to-one matching."""
    rows, cols = [], []
    for gi, g in enumerate(gold):
        for tj, t in enumerate(tuples):
            if score._edge(t, g):
                rows.append(gi); cols.append(tj)
    if not rows:
        return []
    m = maximum_bipartite_matching(csr_matrix(([1] * len(rows), (rows, cols)), shape=(len(gold), len(tuples))),
                                   perm_type='column')
    return [(gi, int(tj)) for gi, tj in enumerate(m) if tj >= 0]


def _et(t):
    return str(t.get('experiment_type') or '').strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--papers', nargs='*')
    ap.add_argument('--require-value', action='store_true')
    a = ap.parse_args()
    score.REQUIRE_VALUE = a.require_value
    papers = a.papers or score.PAPERS

    per = {}
    for pid in papers:
        tuples, gold = score.load(a.pred, a.prefix, pid), score.gold_for(pid)
        pairs = matched_pairs(tuples, gold)
        et_ok = sum(_et(tuples[tj]) == gold[gi]['Experiment-Type'].lower() for gi, tj in pairs)
        per[pid] = {'tp': len(pairs), 'fp': len(tuples) - len(pairs), 'fn': len(gold) - len(pairs), 'et_ok': et_ok}

    tp, fp, fn = (sum(v[k] for v in per.values()) for k in ('tp', 'fp', 'fn'))
    p, r, f = score.prf(tp, fp, fn)
    macro = [score.prf(v['tp'], v['fp'], v['fn']) for v in per.values()]
    mp, mr, mf = (sum(x[i] for x in macro) / len(macro) for i in range(3))
    et_papers = [v['et_ok'] / v['tp'] for v in per.values() if v['tp']]
    print(f'run: {a.pred}   papers: {len(papers)}' + ('   (value must match)' if a.require_value else ''))
    print(f'pooled   precision {p:.4f}   recall {r:.4f}   F1 {f:.4f}   (matched {tp}  extra {fp}  missed {fn})')
    print(f'macro    precision {mp:.4f}   recall {mr:.4f}   F1 {mf:.4f}   (mean over papers)')
    if tp:
        et_pool = sum(v['et_ok'] for v in per.values()) / tp
        et_macro = sum(et_papers) / len(et_papers)
        print(f'ET-Accuracy   pooled {et_pool:.4f}   per-paper mean {et_macro:.4f}   (over matched pairs)')

    meta = Path(a.pred) / 'run_meta.json'
    if meta.exists():
        groups = json.loads(meta.read_text(encoding='utf-8')).get('groups') or []
        print('by group:')
        for gi, g in enumerate(groups):
            g = [x for x in g if x in per]
            if not g:
                continue
            gtp, gfp, gfn = (sum(per[x][k] for x in g) for k in ('tp', 'fp', 'fn'))
            gp, gr, gf = score.prf(gtp, gfp, gfn)
            print(f'  {gi:2d}  P {gp:.3f}  R {gr:.3f}  F1 {gf:.3f}  matched {gtp:4d} of {gtp + gfn:4d}  {",".join(g)}')


if __name__ == '__main__':
    main()
