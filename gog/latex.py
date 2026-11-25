"""Load one paper's LaTeX source as a single document body.

Steps: pick the main file (has \\begin{document}; on ties the largest after inlining),
inline \\input / \\include / \\subfile recursively, strip comments, expand short zero-argument
macros, cut the preamble and the bibliography, and record where \\appendix starts.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from gog.config import LATEX_DIR

_INPUT_RE = re.compile(r'\\(?:input|include|subfile)\s*\{([^}]+)\}')
_COMMENT_RE = re.compile(r'(?<!\\)%.*')


@dataclass
class Paper:
    paper_id: str
    main_file: str
    body: str
    title: str = ''
    abstract: str = ''
    appendix_at: int = -1          # offset of \appendix in body, -1 if none
    missing_inputs: list[str] = field(default_factory=list)


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def strip_comments(text: str) -> str:
    return '\n'.join(_COMMENT_RE.sub('', line) for line in text.split('\n'))


def _resolve(root: Path, name: str) -> Path | None:
    name = name.strip()
    for cand in (root / name, root / f'{name}.tex'):
        if cand.is_file():
            return cand
    return None


def inline_inputs(text: str, root: Path, missing: list[str], depth: int = 0) -> str:
    if depth > 10:
        return text

    def sub(m):
        f = _resolve(root, m.group(1))
        if f is None:
            missing.append(m.group(1))
            return ''
        return inline_inputs(strip_comments(_read(f)), root, missing, depth + 1)

    return _INPUT_RE.sub(sub, text)


def balanced(text: str, start: int) -> tuple[str, int]:
    """Return (content, end) of the brace group opening at text[start] == '{'."""
    if start >= len(text) or text[start] != '{':
        return '', start
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{' and (i == 0 or text[i - 1] != '\\'):
            depth += 1
        elif c == '}' and (i == 0 or text[i - 1] != '\\'):
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)


_MACRO_RE = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\\([A-Za-z]+)\}?\s*(\[\d\])?\s*'
                       r'|\\def\s*\\([A-Za-z]+)\s*(#)?')


def collect_macros(text: str) -> dict[str, str]:
    """Zero-argument macros with short bodies, e.g. \\newcommand{\\pegasuslarge}{PEGASUS_LARGE}."""
    out = {}
    for m in _MACRO_RE.finditer(text):
        name = m.group(1) or m.group(3)
        if m.group(2) or m.group(4):
            continue
        body, _ = balanced(text, m.end())
        if body and len(body) <= 80 and '#' not in body:
            out[name] = body.replace('\\xspace', '').strip()
    return out


def expand_macros(text: str, macros: dict[str, str]) -> str:
    if not macros:
        return text
    pat = re.compile(r'\\(' + '|'.join(sorted(map(re.escape, macros), key=len, reverse=True)) + r')(?![A-Za-z])(\{\})?')
    for _ in range(2):
        text = pat.sub(lambda m: macros[m.group(1)], text)
    return text


def _command_arg(text: str, cmd: str) -> str:
    m = re.search(r'\\' + cmd + r'\s*(\[[^\]]*\])?\s*\{', text)
    return balanced(text, m.end() - 1)[0].strip() if m else ''


def _env(text: str, name: str) -> str:
    m = re.search(r'\\begin\{' + name + r'\}(.*?)\\end\{' + name + r'\}', text, re.S)
    return m.group(1).strip() if m else ''


def load_paper(paper_id: str, latex_dir: Path = LATEX_DIR) -> Paper:
    root = latex_dir / paper_id
    best = None
    for f in sorted(root.rglob('*.tex')):
        raw = strip_comments(_read(f))
        if '\\begin{document}' not in raw:
            continue
        missing: list[str] = []
        full = inline_inputs(raw, f.parent, missing)
        if best is None or len(full) > len(best[1]):
            best = (f, full, missing)
    if best is None:
        raise FileNotFoundError(f'no main .tex with \\begin{{document}} in {root}')
    main, full, missing = best
    pre, _, body = full.partition('\\begin{document}')
    body = body.split('\\end{document}')[0]
    macros = collect_macros(pre + body)
    body = expand_macros(body, macros)
    body = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', body, flags=re.S)
    body = re.sub(r'\\bibliography(style)?\s*\{[^}]*\}', '', body)
    title = next((t for t in (_command_arg(src, cmd) for src in (expand_macros(pre, macros), body)
                              for cmd in ('title', 'icmltitle')) if t), '')
    m = re.search(r'\\appendix\b', body)
    return Paper(paper_id=paper_id, main_file=str(main.relative_to(root)), body=body,
                 title=title, abstract=_env(body, 'abstract'),
                 appendix_at=m.start() if m else -1, missing_inputs=missing)
