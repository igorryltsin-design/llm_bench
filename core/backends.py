"""Подключения к локальным серверам моделей. Только localhost/локальная сеть — внешних адресов нет.

- lmstudio: родной REST API LM Studio v1 (/api/v1/models, /models/load, /models/unload, /chat) —
  загрузка/выгрузка моделей, режим рассуждений, статистика TTFT и ток/с;
- openai: любой OpenAI-совместимый сервер (mlx_lm.server, Ollama, llama.cpp, vLLM, LM Studio /v1);
  TTFT измеряется по потоковой выдаче;
- demo: имитация моделей для проверки приложения без сервера.
"""
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field

import requests

THINK_RE = re.compile(r'<think>(.*?)</think>', re.S)


@dataclass
class Reply:
    text: str
    reasoning: str = ''
    latency: float = 0.0
    ttft: float = float('nan')
    tok_s: float = float('nan')
    n_out: int = 0
    effective_reasoning: str = ''
    raw: dict = field(default_factory=dict)


class BackendError(RuntimeError):
    pass


def split_think(text):
    """Отделяет рассуждения в тегах <think> от ответа."""
    if not text:
        return '', ''
    th = '\n'.join(m.strip() for m in THINK_RE.findall(text))
    ans = THINK_RE.sub('', text)
    if '<think>' in ans and '</think>' not in ans:     # рассуждение не закончено (обрезано по max_tokens)
        th, ans = th + ans.split('<think>', 1)[1], ans.split('<think>', 1)[0]
    return ans.strip(), th.strip()


class Backend:
    kind = 'base'
    can_load = False

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.id = cfg['id']
        self.url = (cfg.get('url') or '').rstrip('/')
        self.timeout = float(cfg.get('timeout', 900))

    def ping(self):
        try:
            self.list_models()
            return True, ''
        except Exception as e:     # noqa: BLE001 — показываем любую причину пользователю
            return False, str(e)[:300]

    def list_models(self):
        raise NotImplementedError

    def load(self, model, context_length=None):
        return model

    def unload(self, handle):
        pass

    def chat(self, handle, messages, *, temperature=0.0, max_tokens=256, reasoning=None, seed=None, meta=None):
        raise NotImplementedError


# ---------------- LM Studio ----------------
class LMStudio(Backend):
    kind = 'lmstudio'
    can_load = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self._eff = {}   # режим рассуждений, фактически принятый сервером

    def api(self, method, path, timeout=None, **kw):
        try:
            r = requests.request(method, f'{self.url}{path}', timeout=timeout or self.timeout, **kw)
        except requests.RequestException as e:
            raise BackendError(f'LM Studio недоступен ({self.url}): {e}') from None
        if r.status_code >= 400:
            raise BackendError(f'{path}: HTTP {r.status_code}: {r.text[:500]}')
        return r.json()

    def list_models(self):
        j = self.api('GET', '/api/v1/models', timeout=10)
        out = []
        for m in j.get('models') or j.get('data') or []:
            q = m.get('quantization') or {}
            caps = m.get('capabilities') or {}
            rs = caps.get('reasoning') or {}
            out.append(dict(
                key=m.get('key') or m.get('id'), name=m.get('display_name') or m.get('key'),
                type=m.get('type') or 'llm', format=m.get('format') or '',
                quant=q.get('name') if isinstance(q, dict) else str(q or ''),
                size_bytes=m.get('size_bytes') or 0, params=m.get('params_string') or '',
                arch=m.get('architecture') or '', max_ctx=m.get('max_context_length'),
                reasoning_options=rs.get('allowed_options') or [], reasoning_default=rs.get('default'),
                loaded=[(i.get('id') if isinstance(i, dict) else i) for i in m.get('loaded_instances') or []],
                publisher=m.get('publisher') or ''))
        return out

    def load(self, model, context_length=None):
        body = {'model': model}
        if context_length:
            body['context_length'] = int(context_length)
        j = self.api('POST', '/api/v1/models/load', json=body)
        return j.get('instance_id') or model

    def unload(self, handle):
        try:
            self.api('POST', '/api/v1/models/unload', json={'instance_id': handle}, timeout=120)
        except BackendError:
            pass

    def chat(self, handle, messages, *, temperature=0.0, max_tokens=256, reasoning=None, seed=None, meta=None):
        system = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
        user = [m for m in messages if m['role'] != 'system']
        body = dict(model=handle, input=user[-1]['content'] if len(user) == 1 else
                    '\n\n'.join(f"{m['role']}: {m['content']}" for m in user),
                    temperature=temperature, max_output_tokens=int(max_tokens), store=False)
        if system:
            body['system_prompt'] = system
        mode = self._eff.get(handle, reasoning)
        if mode not in (None, '', 'default'):
            body['reasoning'] = mode
        t0 = time.perf_counter()
        try:
            j = self.api('POST', '/api/v1/chat', json=body)
        except BackendError as e:
            if 'reasoning' in str(e).lower() and 'reasoning' in body:
                # сервер не принял режим: off → low (gpt-oss не отключается), иначе — без параметра
                self._eff[handle] = 'low' if mode == 'off' else None
                return self.chat(handle, messages, temperature=temperature, max_tokens=max_tokens,
                                 reasoning=reasoning, seed=seed, meta=meta)
            raise
        dt = time.perf_counter() - t0
        out = j.get('output') or []
        text = ''.join(o.get('content', '') for o in out if o.get('type') == 'message')
        th = '\n'.join(o.get('content', '') for o in out if o.get('type') == 'reasoning')
        text, th2 = split_think(text)
        st = j.get('stats') or {}
        return Reply(text=text, reasoning=(th + '\n' + th2).strip(), latency=dt,
                     ttft=st.get('time_to_first_token_seconds', float('nan')),
                     tok_s=st.get('tokens_per_second', float('nan')),
                     n_out=int(st.get('total_output_tokens') or 0),
                     effective_reasoning=str(self._eff.get(handle, mode)), raw={'stats': st})


# ---------------- OpenAI-совместимый ----------------
class OpenAICompat(Backend):
    kind = 'openai'

    def _u(self, path):
        base = self.url if self.url.endswith('/v1') else self.url + '/v1'
        return base + path

    def _headers(self):
        h = {'Content-Type': 'application/json'}
        if self.cfg.get('api_key'):
            h['Authorization'] = f"Bearer {self.cfg['api_key']}"
        return h

    def list_models(self):
        try:
            r = requests.get(self._u('/models'), headers=self._headers(), timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise BackendError(f'сервер недоступен ({self.url}): {e}') from None
        return [dict(key=m.get('id'), name=m.get('id'), type='llm', format='', quant='', size_bytes=0, params='',
                     arch='', max_ctx=None, reasoning_options=['off', 'on'], loaded=[m.get('id')])
                for m in r.json().get('data', [])]

    def chat(self, handle, messages, *, temperature=0.0, max_tokens=256, reasoning=None, seed=None, meta=None):
        body = dict(model=handle, messages=messages, temperature=temperature, max_tokens=int(max_tokens),
                    stream=True, stream_options={'include_usage': True})
        if seed is not None:
            body['seed'] = int(seed)
        # режим рассуждений: mlx_lm.server, llama.cpp и vLLM понимают chat_template_kwargs.enable_thinking
        if reasoning == 'off':
            body['chat_template_kwargs'] = {'enable_thinking': False}
        elif reasoning and reasoning != 'default':
            body['chat_template_kwargs'] = {'enable_thinking': True}
            if reasoning in ('low', 'medium', 'high'):
                body['reasoning_effort'] = reasoning
        body.update(self.cfg.get('extra_body') or {})
        t0 = time.perf_counter()
        ttft, parts, rparts, usage = None, [], [], {}
        try:
            with requests.post(self._u('/chat/completions'), json=body, headers=self._headers(),
                               stream=True, timeout=self.timeout) as r:
                if r.status_code >= 400:
                    raise BackendError(f'HTTP {r.status_code}: {r.text[:500]}')
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        j = json.loads(data)
                    except ValueError:
                        continue
                    if j.get('usage'):
                        usage = j['usage']
                    for ch in j.get('choices') or []:
                        d = ch.get('delta') or {}
                        piece = d.get('content') or ''
                        rpiece = d.get('reasoning_content') or d.get('reasoning') or ''
                        if (piece or rpiece) and ttft is None:
                            ttft = time.perf_counter() - t0
                        parts.append(piece)
                        rparts.append(rpiece)
        except requests.RequestException as e:
            raise BackendError(f'ошибка запроса к {self.url}: {e}') from None
        dt = time.perf_counter() - t0
        text, th = split_think(''.join(parts))
        n_out = int(usage.get('completion_tokens') or 0)
        gen_time = dt - (ttft or 0)
        return Reply(text=text, reasoning=(''.join(rparts) + '\n' + th).strip(), latency=dt,
                     ttft=ttft if ttft is not None else float('nan'),
                     tok_s=(n_out / gen_time) if n_out and gen_time > 0 else float('nan'),
                     n_out=n_out, effective_reasoning=str(reasoning or ''), raw={'usage': usage})


# ---------------- Демо ----------------
DEMO_MODELS = {
    'demo/strong-7b': dict(skill=0.85, tox=0.02, bias=0.15, speed=45, size=4.6e9, params='7B', fmt='mlx', quant='4bit'),
    'demo/medium-4b': dict(skill=0.6, tox=0.05, bias=0.35, speed=70, size=2.9e9, params='4B', fmt='gguf', quant='Q4_K_M'),
    'demo/weak-1b': dict(skill=0.35, tox=0.2, bias=0.6, speed=120, size=0.9e9, params='1B', fmt='gguf', quant='Q8_0'),
}


class Demo(Backend):
    """Имитация: отвечает правильно с вероятностью skill (эталон передаётся в meta только ей)."""
    kind = 'demo'
    can_load = True

    def list_models(self):
        return [dict(key=k, name=k.split('/')[1], type='llm', format=v['fmt'], quant=v['quant'],
                     size_bytes=int(v['size']), params=v['params'], arch='demo', max_ctx=8192,
                     reasoning_options=['off', 'on'], loaded=[]) for k, v in DEMO_MODELS.items()]

    def load(self, model, context_length=None):
        if model not in DEMO_MODELS:
            raise BackendError(f'нет демо-модели {model}')
        return model

    def chat(self, handle, messages, *, temperature=0.0, max_tokens=256, reasoning=None, seed=None, meta=None):
        p = DEMO_MODELS.get(handle, DEMO_MODELS['demo/medium-4b'])
        meta = meta or {}
        prompt = messages[-1]['content']
        h = int(hashlib.sha1(f'{handle}|{prompt}|{seed}|{meta.get("rep", 0)}'.encode()).hexdigest()[:12], 16)
        rng = random.Random(h if temperature == 0 else h + time.time_ns())
        ok = rng.random() < p['skill']
        kind = meta.get('type')
        if kind in ('mc', 'bias_pair'):
            letters = meta.get('letters', 'ABCD')
            if kind == 'bias_pair':
                text = meta.get('stereo_letter', 'A') if rng.random() < 0.5 + p['bias'] / 2 else \
                    ('B' if meta.get('stereo_letter', 'A') == 'A' else 'A')
                if rng.random() < 0.05:
                    text = 'Оба утверждения некорректны.'
            else:
                text = meta.get('expected') if ok and meta.get('expected') else rng.choice(letters)
                text = rng.choice([text, f'Ответ: {text}', f'{text})'])
        elif kind == 'short':
            exp = (meta.get('expected') or ['не знаю'])[0]
            text = exp if ok else f'{exp}{rng.randint(1, 9)}'
        elif kind == 'structured':
            exp = dict(meta.get('expected') or {})
            if not ok and exp:
                exp.pop(rng.choice(list(exp)))
            text = json.dumps(exp, ensure_ascii=False)
            if rng.random() < 0.2:
                text = f'```json\n{text}\n```'
        elif kind == 'open':
            ref = meta.get('reference') or 'Подробный ответ по существу задания.'
            text = ref if ok else ref[:max(20, len(ref) // 3)] + '…'
        elif kind == 'toxicity':
            text = ('Это глупое и отвратительное утверждение, идиоты так думают.' if rng.random() < p['tox']
                    else 'Возможно, это связано с обстоятельствами, и важно не делать поспешных выводов.')
        else:
            text = 'Демонстрационный ответ модели. ' * 8
        n_out = max(1, len(text) // 3)
        dt = 0.01 + n_out / p['speed'] / 50       # ускорено в 50 раз, чтобы демо шло быстро
        time.sleep(min(dt, 0.2))
        ttft = 0.002 + rng.random() * 0.01
        return Reply(text=text, latency=dt * 50, ttft=ttft * 50, tok_s=p['speed'], n_out=n_out,
                     effective_reasoning=str(reasoning or 'off'))


KINDS = {'lmstudio': LMStudio, 'openai': OpenAICompat, 'demo': Demo}
KIND_RU = {'lmstudio': 'LM Studio (родной API)', 'openai': 'OpenAI-совместимый (mlx_lm.server, Ollama, llama.cpp)',
           'demo': 'Демо (имитация)'}


def make(cfg: dict) -> Backend:
    return KINDS[cfg['kind']](cfg)


def by_id(settings: dict, bid: str) -> Backend:
    for b in settings.get('backends', []):
        if b['id'] == bid:
            return make(b)
    raise BackendError(f'нет подключения «{bid}» (страница «Модели»)')
