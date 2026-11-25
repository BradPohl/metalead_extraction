"""One OpenAI-compatible chat client for OpenAI and Ollama (set base_url for Ollama).

- Responses are cached on disk under llm.cache_dir, keyed by hash(model, messages, max_tokens).
- A thread-safe meter adds up tokens and cost from llm.prices (USD per million tokens).
- The budget guard refuses a call when spent + estimate would exceed llm.budget_usd.
- dry_run=True never calls the API; it adds the estimated tokens and returns '{}'.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from gog.config import ROOT, approx_tokens, load_config


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 dry_run: bool = False, budget_usd: float | None = None, expected_output_tokens: int = 800,
                 reasoning_effort: str | None = None):
        cfg = load_config()['llm']
        self.model, self.base_url, self.dry_run = model, base_url, dry_run
        # 'none' turns off hidden thinking on Ollama models such as Qwen3 (sent as reasoning_effort).
        self.reasoning_effort = reasoning_effort
        self.temperature = cfg['temperature']
        self.budget = cfg['budget_usd'] if budget_usd is None else budget_usd
        prices = cfg.get('prices', {})
        self.price = prices.get(model) or prices.get('default') or {'input': 0.0, 'output': 0.0}
        self.expected_out = expected_output_tokens
        self.cache_dir = ROOT / cfg['cache_dir'] / re.sub(r'[^A-Za-z0-9._-]+', '_', model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.is_openai = base_url is None or 'api.openai.com' in base_url
        self._client = None
        self._api_key = api_key
        self._lock = threading.Lock()
        self._send_temperature = True
        self.stats = {'calls': 0, 'cached': 0, 'failed': 0, 'tokens_in': 0, 'tokens_out': 0,
                      'cost_usd': 0.0, 'est_tokens_in': 0, 'est_tokens_out': 0, 'est_cost_usd': 0.0,
                      'wall_seconds': 0.0}

    def _cost(self, tin: int, tout: int) -> float:
        return (tin * self.price['input'] + tout * self.price['output']) / 1e6

    def _key(self, messages, max_tokens) -> Path:
        parts = [self.model, messages, max_tokens]
        if self.reasoning_effort:        # left out when unset, so earlier cached replies stay valid
            parts.append(self.reasoning_effort)
        h = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode('utf-8'))
        return self.cache_dir / f'{h.hexdigest()}.json'

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            key = self._api_key or os.environ.get('OPENAI_API_KEY') or ('ollama' if not self.is_openai else None)
            self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=600)
        return self._client

    def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        path = self._key(messages, max_tokens)
        if path.exists():
            with self._lock:
                self.stats['cached'] += 1
            return json.loads(path.read_text(encoding='utf-8'))['text']
        est_in = sum(approx_tokens(m['content']) for m in messages)
        est = self._cost(est_in, self.expected_out)
        with self._lock:
            self.stats['est_tokens_in'] += est_in
            self.stats['est_tokens_out'] += self.expected_out
            self.stats['est_cost_usd'] += est
            if self.dry_run:
                return '{}'
            if self.stats['cost_usd'] + est > self.budget:
                raise BudgetExceeded(f'spent ${self.stats["cost_usd"]:.2f}, next call ~${est:.3f}, budget ${self.budget:.2f}')
        t0 = time.time()
        text, usage = self._call(messages, max_tokens)
        tin = getattr(usage, 'prompt_tokens', None) or est_in
        tout = getattr(usage, 'completion_tokens', None) or approx_tokens(text)
        with self._lock:
            self.stats['calls'] += 1
            self.stats['tokens_in'] += tin
            self.stats['tokens_out'] += tout
            self.stats['cost_usd'] += self._cost(tin, tout)
            self.stats['wall_seconds'] += time.time() - t0
        path.write_text(json.dumps({'text': text, 'tokens_in': tin, 'tokens_out': tout}, ensure_ascii=False),
                        encoding='utf-8')
        return text

    def _call(self, messages, max_tokens):
        from openai import BadRequestError
        kw = {'model': self.model, 'messages': messages, 'response_format': {'type': 'json_object'}}
        for attempt in range(4):
            # Sent as raw body fields because the installed SDK (1.12) predates these keywords.
            body = {}
            if self.is_openai:
                body['max_completion_tokens'] = max_tokens
            else:
                kw['max_tokens'] = max_tokens
            if self.reasoning_effort:
                body['reasoning_effort'] = self.reasoning_effort
            kw['extra_body'] = body
            if self._send_temperature:
                kw['temperature'] = self.temperature
            else:
                kw.pop('temperature', None)
            try:
                r = self._get_client().chat.completions.create(**kw)
                return r.choices[0].message.content or '', r.usage
            except BadRequestError as e:
                # Some reasoning models accept only the default temperature.
                if 'temperature' in str(e) and self._send_temperature:
                    self._send_temperature = False
                    continue
                # Models without a thinking mode may reject the setting; run them without it.
                if self.reasoning_effort and ('reasoning' in str(e) or 'think' in str(e)):
                    self.reasoning_effort = None
                    continue
                raise
            except Exception:
                if attempt == 3:
                    with self._lock:
                        self.stats['failed'] += 1
                    raise
                time.sleep(2 ** attempt * 5)
        raise RuntimeError('unreachable')


def parse_json(text: str):
    """Parse a model reply that should be JSON; tolerate code fences and leading prose."""
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'[\[{].*[\]}]', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
