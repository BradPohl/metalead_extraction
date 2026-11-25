"""The two graph levels.

Paper graph: w(p, q) = alpha * cos(mean chunk embedding) + (1 - alpha) * jaccard(vocabulary keys).
Edges below paper_graph.min_edge are dropped; Louvain groups the rest, and groups larger than
max_group_size are split by re-running Louvain on the subgraph at higher resolution.

Chunk graph (one per group): every chunk of the group's papers is a node.
w(c, d) = alpha * cos(embedding) + (1 - alpha) * jaccard(names each chunk mentions).
Node weight = PageRank (normalized to max 1) * results prior (table / results section / numbers).
Neighbors of a chunk are its top-k chunks from *other* papers with w >= tau, ranked by
w * neighbor node weight.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from gog.chunk import Chunk
from gog.names import jaccard, key_set, text_mentions
from gog.vocab import FIELDS


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def paper_embeddings(chunks_by_paper: dict[str, list[Chunk]], emb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for pid, cs in chunks_by_paper.items():
        v = np.mean([emb[c.chunk_id] for c in cs], axis=0)
        out[pid] = v / (np.linalg.norm(v) + 1e-12)
    return out


def vocab_keys(v: dict) -> set[str]:
    return set().union(*(key_set(v.get(f, [])) for f in FIELDS)) if v else set()


def paper_graph(pids: list[str], pemb: dict[str, np.ndarray], vocabs: dict[str, dict],
                alpha: float, min_edge: float, knn: int) -> nx.Graph:
    """kNN graph: an edge is kept when it is among the top `knn` of either endpoint.
    All NLP papers are fairly similar (median cosine ~0.7), so a dense graph has no community structure."""
    keys = {p: vocab_keys(vocabs.get(p, {})) for p in pids}
    W = {}
    for i, p in enumerate(pids):
        for q in pids[i + 1:]:
            cos, jac = _cos(pemb[p], pemb[q]), jaccard(keys[p], keys[q])
            W[(p, q)] = W[(q, p)] = (alpha * cos + (1 - alpha) * jac, cos, jac)
    G = nx.Graph()
    G.add_nodes_from(pids)
    for p in pids:
        top = sorted((q for q in pids if q != p), key=lambda q: -W[(p, q)][0])[:knn]
        for q in top:
            w, cos, jac = W[(p, q)]
            if w >= min_edge:
                G.add_edge(p, q, weight=w, cos=cos, jaccard=jac)
    return G


def group_papers(G: nx.Graph, max_size: int, seed: int) -> list[list[str]]:
    def split(nodes):
        if len(nodes) <= max_size:
            return [sorted(nodes)]
        sub = G.subgraph(nodes)
        for res in (1.5, 2, 3, 5, 8, 13):
            parts = nx.community.louvain_communities(sub, weight='weight', resolution=res, seed=seed)
            if len(parts) > 1:
                return [g for p in parts for g in split(p)]
        ordered = sorted(nodes)
        return [ordered[i:i + max_size] for i in range(0, len(ordered), max_size)]

    groups = [g for comm in nx.community.louvain_communities(G, weight='weight', seed=seed) for g in split(comm)]
    return sorted(groups, key=lambda g: (-len(g), g))


def random_groups(groups: list[list[str]], seed: int) -> list[list[str]]:
    """Control: the same group sizes, papers assigned at random."""
    pids = [p for g in groups for p in g]
    random.Random(seed).shuffle(pids)
    out, i = [], 0
    for g in groups:
        out.append(sorted(pids[i:i + len(g)])); i += len(g)
    return out


@dataclass
class ChunkGraph:
    group_id: int
    papers: list[str]
    node_weight: dict[str, float] = field(default_factory=dict)
    neighbors: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    edges: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self):
        return {'group_id': self.group_id, 'papers': self.papers, 'node_weight': self.node_weight,
                'neighbors': self.neighbors, 'edges': self.edges}


def results_prior(c: Chunk, prior: dict) -> float:
    if c.kind == 'table':
        return prior['table']
    if c.results_section and c.has_numbers:
        return prior['results_section']
    return prior['other_with_numbers'] if c.has_numbers else prior['other']


def chunk_graph(group_id: int, papers: list[str], chunks: list[Chunk], emb: dict[str, np.ndarray],
                group_vocab: dict, cfg: dict, randomize_seed: int | None = None) -> ChunkGraph:
    alpha, k, tau = cfg['alpha'], cfg['k'], cfg['tau']
    names = [n for f in FIELDS for n in group_vocab.get(f, [])]
    mentions = {c.chunk_id: text_mentions(c.caption + ' ' + c.text, names) for c in chunks}
    ids = [c.chunk_id for c in chunks]
    paper_of = {c.chunk_id: c.paper_id for c in chunks}
    M = np.stack([emb[i] for i in ids]) if ids else np.zeros((0, 1))
    cos = M @ M.T if len(ids) else M
    G = nx.Graph()
    G.add_nodes_from(ids)
    cg = ChunkGraph(group_id=group_id, papers=papers)
    W = {}
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            w = alpha * float(cos[i, j]) + (1 - alpha) * jaccard(mentions[a], mentions[b])
            if w >= 0.2:
                G.add_edge(a, b, weight=w)
            W[(a, b)] = W[(b, a)] = w
            if paper_of[a] != paper_of[b] and w >= tau:
                cg.edges.append((a, b, round(w, 4)))
    pr = nx.pagerank(G, weight='weight') if G.number_of_edges() else {i: 1.0 for i in ids}
    top = max(pr.values()) if pr else 1.0
    by_id = {c.chunk_id: c for c in chunks}
    cg.node_weight = {i: round(pr[i] / top * results_prior(by_id[i], cfg['results_prior']), 4) for i in ids}

    rng = random.Random(randomize_seed) if randomize_seed is not None else None
    for a in ids:
        others = [b for b in ids if paper_of[b] != paper_of[a]]
        if rng is not None:
            # Control: k random chunks from other papers in the group, same budget as the real neighbors.
            pick = rng.sample(others, min(k, len(others)))
            cg.neighbors[a] = [(b, round(W[(a, b)], 4)) for b in pick]
            continue
        cand = [(b, W[(a, b)]) for b in others if W[(a, b)] >= tau]
        cand.sort(key=lambda x: -x[1] * cg.node_weight[x[0]])
        cg.neighbors[a] = [(b, round(w, 4)) for b, w in cand[:k]]
    return cg
