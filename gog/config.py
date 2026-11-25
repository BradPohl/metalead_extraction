"""Paths and the config/gog.yaml loader."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LATEX_DIR = ROOT / 'data' / 'papers_latex'
CFG_PATH = ROOT / 'config' / 'gog.yaml'


@lru_cache(maxsize=1)
def load_config() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding='utf-8'))


def paper_ids() -> list[str]:
    return json.loads((ROOT / 'splits' / 'metalead_31_papers.json').read_text(encoding='utf-8'))


def approx_tokens(text: str) -> int:
    # LaTeX is symbol-dense; 3.5 chars per token errs on the side of overestimating.
    return int(len(text) / 3.5) + 1
