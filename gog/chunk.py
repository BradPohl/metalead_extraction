"""Split a loaded paper into table chunks and section chunks.

Table chunk: a table environment (or a tabular outside one, with its enclosing figure),
its caption and label, plus the paragraphs that \\ref the label. Long tables are split
into row blocks that repeat the header rows.
Section chunk: the text under one \\section/\\subsection path, split on blank lines
when it exceeds chunk.max_tokens. Tables are cut out of the section text so no
number appears in two chunks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict

from gog.config import approx_tokens, load_config
from gog.latex import Paper, balanced

_TABLE_ENVS = r'(?:table\*?|sidewaystable\*?|longtable\*?|wraptable)'
_TABULAR_ENVS = r'(?:tabular[xy*]?|tabu|array)'
_SECTION_RE = re.compile(r'\\(section|subsection|subsubsection)\*?\s*(\[[^\]]*\])?\s*\{')
_RESULTS_RE = re.compile(r'result|experiment|evaluat|ablation|analys|comparison|performance|benchmark', re.I)
_NUM_RE = re.compile(r'(?<![\w.])\d+(?:\.\d+)?')
_STRIP_FOR_NUMS = re.compile(r'\\(?:cite[tp]?|ref|label|eqref|autoref|cref|Cref)\*?\s*\{[^}]*\}')


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    kind: str                # 'table' | 'section'
    section_path: str
    text: str
    caption: str = ''
    label: str = ''
    in_appendix: bool = False
    has_numbers: bool = False
    results_section: bool = False

    def to_dict(self):
        return asdict(self)


def numbers_in(text: str) -> set[float]:
    """Numbers printed in the text, ignoring those inside \\cite, \\ref and \\label."""
    return {float(x) for x in _NUM_RE.findall(_STRIP_FOR_NUMS.sub(' ', text))}


def has_numbers(text: str) -> bool:
    return bool(_NUM_RE.search(_STRIP_FOR_NUMS.sub(' ', text)))


def _env_spans(body: str, env_pat: str) -> list[tuple[int, int, str]]:
    """Non-nested (start, end, env_name) spans of \\begin{env}...\\end{env}."""
    spans = []
    for m in re.finditer(r'\\begin\{(' + env_pat + r')\}', body):
        name = m.group(1)
        if spans and m.start() < spans[-1][1]:
            continue
        end = body.find('\\end{' + name + '}', m.end())
        if end < 0:
            continue
        spans.append((m.start(), end + len('\\end{' + name + '}'), name))
    return spans


def _arg(text: str, cmd: str) -> str:
    m = re.search(r'\\' + cmd + r'\s*(\[[^\]]*\])?\s*\{', text)
    return balanced(text, m.end() - 1)[0].strip() if m else ''


def _table_spans(body: str) -> list[tuple[int, int]]:
    spans = [(s, e) for s, e, _ in _env_spans(body, _TABLE_ENVS)]
    inside = lambda a, b: any(s <= a and b <= e for s, e in spans)
    figures = _env_spans(body, r'figure\*?')
    for s, e, _ in _env_spans(body, _TABULAR_ENVS):
        if inside(s, e):
            continue
        fig = next(((fs, fe) for fs, fe, _ in figures if fs <= s and e <= fe), None)
        span = fig or (s, e)
        if not inside(*span) and span not in spans:
            spans.append(span)
    return sorted(spans)


def _section_paths(body: str) -> list[tuple[int, str]]:
    """(offset, 'Section > Subsection') for each heading, in order."""
    out, stack = [], []
    level = {'section': 0, 'subsection': 1, 'subsubsection': 2}
    for m in _SECTION_RE.finditer(body):
        title, _ = balanced(body, m.end() - 1)
        lv = level[m.group(1)]
        stack = stack[:lv] + [re.sub(r'\s+', ' ', title).strip()]
        out.append((m.start(), ' > '.join(stack)))
    return out


def _path_at(paths: list[tuple[int, str]], pos: int) -> str:
    cur = ''
    for off, p in paths:
        if off > pos:
            break
        cur = p
    return cur


def _ref_context(body: str, label: str, budget_tokens: int) -> str:
    if not label:
        return ''
    paras = re.split(r'\n\s*\n', body)
    hits = [p.strip() for p in paras if re.search(r'\\[a-zA-Z]*ref\*?\s*\{' + re.escape(label) + r'\}', p)]
    out, used = [], 0
    for p in hits:
        t = approx_tokens(p)
        if used + t > budget_tokens:
            p = p[:int((budget_tokens - used) * 3.5)]
            if p:
                out.append(p + ' ...')
            break
        out.append(p); used += t
    return '\n\n'.join(out)


def _split_table(text: str, max_tokens: int) -> list[str]:
    """Split a long table into row blocks, repeating rows up to the first rule as a header."""
    if approx_tokens(text) <= max_tokens:
        return [text]
    rows = re.split(r'(?<=\\\\)', text)
    head_n = next((i + 1 for i, r in enumerate(rows[:8]) if re.search(r'\\(midrule|hline)', r) and i > 0), 2)
    header, rest = ''.join(rows[:head_n]), rows[head_n:]
    blocks, cur = [], ''
    for r in rest:
        if cur and approx_tokens(header + cur + r) > max_tokens:
            blocks.append(header + cur + '\n% [table continues]'); cur = ''
        cur += r
    if cur:
        blocks.append(header + cur)
    return blocks


def _split_paragraphs(text: str, max_tokens: int) -> list[str]:
    paras = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    out, cur = [], ''
    for p in paras:
        if cur and approx_tokens(cur + p) > max_tokens:
            out.append(cur.strip()); cur = ''
        cur += p + '\n\n'
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_paper(paper: Paper) -> list[Chunk]:
    cfg = load_config()['chunk']
    body = paper.body
    paths = _section_paths(body)
    in_app = lambda pos: paper.appendix_at >= 0 and pos >= paper.appendix_at
    chunks: list[Chunk] = []

    def add(kind, path, text, pos, caption='', label=''):
        chunks.append(Chunk(chunk_id=f'{paper.paper_id}#{len(chunks):03d}', paper_id=paper.paper_id,
                            kind=kind, section_path=path, text=text, caption=caption, label=label,
                            in_appendix=in_app(pos), has_numbers=has_numbers(text),
                            results_section=bool(_RESULTS_RE.search(path + ' ' + caption))))

    tables = _table_spans(body)
    for s, e in tables:
        t = body[s:e]
        caption = re.sub(r'\\label\s*\{[^}]*\}', '', _arg(t, 'caption')).strip()
        label = _arg(t, 'label')
        ctx = _ref_context(body, label, cfg['ref_context_tokens'])
        for block in _split_table(t, cfg['table_max_tokens']):
            text = block + (f'\n\n% Text that refers to this table:\n{ctx}' if ctx else '')
            add('table', _path_at(paths, s), text, s, caption, label)

    # Section text with tables removed, split at headings.
    cuts, last = [], 0
    for s, e in tables:
        cuts.append((last, s)); last = e
    cuts.append((last, len(body)))
    pieces = [(a, body[a:b]) for a, b in cuts]
    heads = [0] + [off for off, _ in paths] + [len(body)]
    for (a, piece) in pieces:
        # Split each table-free piece at section headings that fall inside it.
        inner = [h for h in heads if a < h < a + len(piece)]
        bounds = [a] + inner + [a + len(piece)]
        for x, y in zip(bounds, bounds[1:]):
            seg = piece[x - a:y - a]
            if len(seg.strip()) < 40:
                continue
            for part in _split_paragraphs(seg, cfg['max_tokens']):
                add('section', _path_at(paths, x) or 'Front matter', part, x)
    return chunks
