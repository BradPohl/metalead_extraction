"""Run one extraction condition and write scorer-format outputs.

    python -m gog.run --condition A3 --model gpt-4.1-mini --out runs/gpt41mini_A3 --dry-run
    python -m gog.run --condition A4 --model qwen2.5:14b --base-url http://localhost:11434/v1 --out runs/qwen14_A4

Conditions (each adds one thing):
    A1  chunked, per paper, no cross-paper information
    A2  A1 + group vocabulary
    A3  A2 + neighbor chunks from other papers in the group
    A4  A3 + corroboration + recall probe (full graph of graphs)
    C3  A3 with random groups and random neighbors (control)

The graphs are always built over all 31 papers; --papers limits which papers are extracted.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gog import corroborate as cor
from gog.chunk import chunk_paper
from gog.config import ROOT, load_config, paper_ids
from gog.embed import embed_chunks
from gog.extract import extract_chunk
from gog.graphs import chunk_graph, group_papers, paper_embeddings, paper_graph, random_groups
from gog.latex import load_paper
from gog.llm import LLM, BudgetExceeded
from gog.vocab import merge_vocab, paper_vocab

CONDITIONS = {
    'A1': dict(vocab=False, neighbors=False, corroborate=False, random=False),
    'A2': dict(vocab=True, neighbors=False, corroborate=False, random=False),
    'A3': dict(vocab=True, neighbors=True, corroborate=False, random=False),
    'A4': dict(vocab=True, neighbors=True, corroborate=True, random=False),
    'C3': dict(vocab=True, neighbors=True, corroborate=False, random=True),
}


def _parallel(fn, items, workers):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', required=True, choices=sorted(CONDITIONS))
    ap.add_argument('--model', required=True)
    ap.add_argument('--base-url', default=None, help='e.g. http://localhost:11434/v1 for Ollama')
    ap.add_argument('--out', required=True)
    ap.add_argument('--papers', nargs='*', help='extract only these paper ids')
    ap.add_argument('--dry-run', action='store_true', help='estimate tokens and cost, make no calls')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--reasoning-effort', default=None,
                    help="sent as reasoning_effort; default 'none' for --base-url (local) models, unset for OpenAI. "
                         "Pass 'default' to leave thinking on for a local model")
    ap.add_argument('--budget', type=float, default=None)
    ap.add_argument('--alpha', type=float, help='override paper_graph.alpha and chunk_graph.alpha')
    ap.add_argument('--k', type=int, help='override chunk_graph.k')
    ap.add_argument('--tau', type=float, help='override chunk_graph.tau')
    ap.add_argument('--max-probe-chunks', type=int, default=10, help='recall-probe chunks per paper (A4)')
    a = ap.parse_args()

    cond = CONDITIONS[a.condition]
    cfg = copy.deepcopy(load_config())
    if a.alpha is not None:
        cfg['paper_graph']['alpha'] = cfg['chunk_graph']['alpha'] = a.alpha
    if a.k is not None:
        cfg['chunk_graph']['k'] = a.k
    if a.tau is not None:
        cfg['chunk_graph']['tau'] = a.tau
    out = Path(a.out)
    (out / 'graphs').mkdir(parents=True, exist_ok=True)
    workers = a.workers or (1 if a.base_url else 4)
    effort = a.reasoning_effort or ('none' if a.base_url else None)
    llm = LLM(a.model, a.base_url, dry_run=a.dry_run, budget_usd=a.budget,
              reasoning_effort=None if effort == 'default' else effort)
    t0 = time.time()

    all_pids = paper_ids()
    targets_pids = a.papers or all_pids
    papers = {p: load_paper(p) for p in all_pids}
    chunks = {p: chunk_paper(papers[p]) for p in all_pids}
    by_id = {c.chunk_id: c for cs in chunks.values() for c in cs}
    emb = embed_chunks(list(by_id.values()))

    # Top graph: papers.
    vocabs = {}
    if cond['vocab']:
        vocabs = _parallel(lambda p: paper_vocab(llm, papers[p], chunks[p]), all_pids, workers)
        pg = cfg['paper_graph']
        G = paper_graph(all_pids, paper_embeddings(chunks, emb), vocabs, pg['alpha'], pg['min_edge'], pg['knn'])
        groups = group_papers(G, pg['max_group_size'], pg['seed'])
        if cond['random']:
            groups = random_groups(groups, pg['seed'] + 1)
        (out / 'graphs' / 'paper_graph.json').write_text(json.dumps(
            [{'a': u, 'b': v, **{k: round(x, 4) for k, x in d.items()}} for u, v, d in G.edges(data=True)],
            indent=1), encoding='utf-8')
    else:
        groups = [[p] for p in all_pids]
    group_of = {p: gi for gi, g in enumerate(groups) for p in g}

    # Inner graphs: chunks of each group.
    cgs, gvocab = {}, {}
    for gi, g in enumerate(groups):
        gvocab[gi] = merge_vocab([vocabs[p] for p in g]) if cond['vocab'] else {}
        cgs[gi] = chunk_graph(gi, g, [c for p in g for c in chunks[p]], emb, gvocab[gi], cfg['chunk_graph'],
                              randomize_seed=(cfg['paper_graph']['seed'] + 2 + gi) if cond['random'] else None)
        (out / 'graphs' / f'chunk_graph_{gi:02d}.json').write_text(json.dumps(cgs[gi].to_dict()), encoding='utf-8')
    node_weight = {cid: w for cg in cgs.values() for cid, w in cg.node_weight.items()}
    (out / 'graphs' / 'groups.json').write_text(json.dumps(
        {'groups': groups, 'vocab': gvocab, 'paper_vocab': vocabs}, indent=1, ensure_ascii=False), encoding='utf-8')

    # Extraction.
    min_w = cfg['chunk_graph']['min_node_weight']
    jobs = [c.chunk_id for p in targets_pids for c in chunks[p] if c.has_numbers and node_weight[c.chunk_id] >= min_w]

    def run_job(cid, hint=''):
        c = by_id[cid]
        gi = group_of[c.paper_id]
        nbrs = [(by_id[n], papers[by_id[n].paper_id].title) for n, _ in cgs[gi].neighbors.get(cid, [])] \
            if cond['neighbors'] else []
        return extract_chunk(llm, c, papers[c.paper_id].title, gvocab[gi] or None, nbrs, cfg, hint)

    records = defaultdict(list)
    stats = defaultdict(int)
    stopped = None

    def collect(results):
        for cid, (recs, st) in results.items():
            for k, v in st.items():
                stats[k] += v
            for r in recs:
                r['confidence'] = round(0.5 + 0.5 * node_weight[cid], 4)
                records[by_id[cid].paper_id].append(r)

    try:
        collect(_parallel(run_job, jobs, workers))
        if cond['corroborate'] and not a.dry_run:
            for gi, g in enumerate(groups):
                grp = {p: records[p] for p in g}
                cor.harmonize_names(grp)
                hints = cor.probe_hints(grp, {p: chunks[p] for p in g if p in targets_pids}, a.max_probe_chunks)
                stats['probe_chunks'] += len(hints)
                collect(_parallel(lambda cid: run_job(cid, cor.hint_text(hints[cid])), list(hints), workers))
                grp = {p: records[p] for p in g}
                cor.harmonize_names(grp)
                cor.corroborate(grp, node_weight)
    except BudgetExceeded as e:
        stopped = str(e)
        print(f'STOPPED: {e}')

    for p in targets_pids:
        recs = cor.dedupe(records[p])
        meta = {'condition': a.condition, 'model': a.model, 'base_url': a.base_url, 'group': group_of[p],
                'group_papers': groups[group_of[p]], 'n_chunks': len(chunks[p]),
                'n_target_chunks': sum(1 for j in jobs if by_id[j].paper_id == p)}
        (out / f'{p}.json').write_text(json.dumps({'paper_id': p, 'tuples': recs, '_meta': meta},
                                                  indent=1, ensure_ascii=False), encoding='utf-8')

    run_meta = {'condition': a.condition, **cond, 'model': a.model, 'base_url': a.base_url, 'dry_run': a.dry_run,
                'reasoning_effort': llm.reasoning_effort,
                'papers': targets_pids, 'n_jobs': len(jobs), 'workers': workers, 'stopped': stopped,
                'extract_stats': dict(stats), 'llm': llm.stats, 'wall_seconds': round(time.time() - t0, 1),
                'config': cfg, 'groups': groups}
    (out / 'run_meta.json').write_text(json.dumps(run_meta, indent=1), encoding='utf-8')

    s = llm.stats
    print(f'condition {a.condition}  model {a.model}  groups {len(groups)}  target chunks {len(jobs)}')
    print('groups: ' + ' | '.join(','.join(g) for g in groups))
    if a.dry_run:
        print(f'estimate (uncached calls): in {s["est_tokens_in"]:,} tok, out {s["est_tokens_out"]:,} tok, '
              f'cost ${s["est_cost_usd"]:.2f} at {llm.price}')
        if cond['vocab']:
            print('  (vocabulary block not in the estimate: add about '
                  f'{cfg["extract"]["vocab_tokens"] * len(jobs):,} input tokens)')
        if cond['corroborate']:
            print(f'  (recall probes not in the estimate: at most {a.max_probe_chunks} chunks per paper)')
    else:
        print(f'calls {s["calls"]} (cached {s["cached"]}, failed {s["failed"]})  in {s["tokens_in"]:,}  '
              f'out {s["tokens_out"]:,}  cost ${s["cost_usd"]:.2f}  wall {run_meta["wall_seconds"]}s')
        print('records: ' + str(sum(len(cor.dedupe(records[p])) for p in targets_pids)) + '  ' + json.dumps(dict(stats)))


if __name__ == '__main__':
    main()
