"""Cross-paper consolidation inside one group.

(a) Names: records whose dataset / metric / method share a name key get the group's most
    common surface form.
(b) Corroboration: a (dataset, metric, value) reported by other papers of the group raises a
    record's confidence. Base confidence comes from the chunk's node weight.
(c) Recall probe: a value that other papers report, and that is printed in one of this paper's
    chunks but missing from its records, triggers one hinted re-extraction of that chunk.
(d) Dedupe within each paper on (method, dataset, metric, value).
"""
from __future__ import annotations
from collections import Counter, defaultdict

from gog.chunk import Chunk, numbers_in
from gog.names import name_key


def _keys(r: dict) -> tuple[str, str, str, float]:
    return (name_key(r['method']['name']), name_key(r['datasets'][0]['name']),
            name_key(r['metrics'][0]['name']), round(r['metrics'][0]['value'], 4))


def harmonize_names(records_by_paper: dict[str, list[dict]]) -> None:
    fields = {'method': lambda r: r['method'], 'dataset': lambda r: r['datasets'][0],
              'metric': lambda r: r['metrics'][0]}
    for get in fields.values():
        forms = defaultdict(Counter)
        for recs in records_by_paper.values():
            for r in recs:
                forms[name_key(get(r)['name'])][get(r)['name']] += 1
        for recs in records_by_paper.values():
            for r in recs:
                k = name_key(get(r)['name'])
                if k:
                    get(r)['name'] = forms[k].most_common(1)[0][0]


def dedupe(records: list[dict]) -> list[dict]:
    best = {}
    for r in records:
        k = _keys(r)
        if k not in best or r.get('confidence', 0) > best[k].get('confidence', 0):
            best[k] = r
    return list(best.values())


def corroborate(records_by_paper: dict[str, list[dict]], node_weight: dict[str, float]) -> None:
    seen = defaultdict(set)                          # (dataset, metric, value) -> papers
    for pid, recs in records_by_paper.items():
        for r in recs:
            _, d, m, v = _keys(r)
            seen[(d, m, v)].add(pid)
    for pid, recs in records_by_paper.items():
        for r in recs:
            _, d, m, v = _keys(r)
            others = sorted(seen[(d, m, v)] - {pid})
            base = 0.5 + 0.5 * node_weight.get(r['source']['chunk_id'], 0.0)
            r['confidence'] = round(min(1.0, base + 0.25 * min(len(others), 2)), 4)
            r['corroborated_by'] = others


def _distinctive(v: float) -> bool:
    """Values specific enough that a match across papers is unlikely to be chance (e.g. 91.21, 28.4)."""
    s = f'{v:g}'
    return '.' in s and len(s.replace('.', '').lstrip('0')) >= 3


def probe_hints(records_by_paper: dict[str, list[dict]], chunks_by_paper: dict[str, list[Chunk]],
                max_chunks_per_paper: int) -> dict[str, list[dict]]:
    """chunk_id -> records from other papers whose value is printed in that chunk but not yet extracted."""
    out = {}
    for pid, chunks in chunks_by_paper.items():
        have = {r['metrics'][0]['value'] for r in records_by_paper.get(pid, [])}
        wanted = {}
        for q, recs in records_by_paper.items():
            if q == pid:
                continue
            for r in recs:
                v = r['metrics'][0]['value']
                if v not in have and _distinctive(v):
                    wanted.setdefault(v, r)
        if not wanted:
            continue
        hits = []
        for c in chunks:
            if not c.has_numbers:
                continue
            found = [wanted[v] for v in sorted(numbers_in(c.text) & wanted.keys())]
            if found:
                hits.append((len(found), c.chunk_id, found))
        hits.sort(key=lambda h: (-h[0], h[1]))
        for _, cid, found in hits[:max_chunks_per_paper]:
            out[cid] = found[:8]
    return out


def hint_text(found: list[dict]) -> str:
    lines = [f'- {r["method"]["name"] or "?"} | {r["datasets"][0]["name"]} | {r["metrics"][0]["name"]} | '
             f'{r["metrics"][0]["value"]:g}' for r in found]
    return ('Related papers report these results, and the same numbers appear in the target excerpt. '
            'If the target excerpt reports them, extract them with the target paper\'s own names, '
            'together with every other result in the excerpt:\n' + '\n'.join(lines))
