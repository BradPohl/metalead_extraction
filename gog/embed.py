"""Local sentence-transformers embeddings for chunks, cached on disk by text hash."""
from __future__ import annotations
import hashlib
import re
from functools import lru_cache

import numpy as np

from gog.chunk import Chunk
from gog.config import ROOT, load_config

_CMD_RE = re.compile(r'\\[A-Za-z]+\*?(\[[^\]]*\])?')


def embed_text(c: Chunk) -> str:
    """What the embedding sees: location, caption, then the chunk with LaTeX commands removed."""
    body = _CMD_RE.sub(' ', c.text)
    body = re.sub(r'[{}&\\$~^_]+', ' ', body)
    body = re.sub(r'\s+', ' ', body)
    return f'{c.section_path}. {c.caption}. {body}'[:2000]


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(load_config()['embed']['model'], device='cpu')


def embed_chunks(chunks: list[Chunk]) -> dict[str, np.ndarray]:
    name = load_config()['embed']['model']
    path = ROOT / load_config()['llm']['cache_dir'] / f'emb_{re.sub(r"[^A-Za-z0-9]+", "_", name)}.npz'
    cache = dict(np.load(path)) if path.exists() else {}
    texts = {c.chunk_id: embed_text(c) for c in chunks}
    keys = {cid: hashlib.sha1(t.encode('utf-8')).hexdigest() for cid, t in texts.items()}
    todo = [cid for cid in texts if keys[cid] not in cache]
    if todo:
        vecs = _model().encode([texts[c] for c in todo], batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False)
        for cid, v in zip(todo, vecs):
            cache[keys[cid]] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cache)
    return {cid: cache[keys[cid]] for cid in texts}
