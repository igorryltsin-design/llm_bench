"""Дообучение (LoRA/QLoRA через mlx-lm на Apple Silicon): наборы данных, проверка утечек, запуск,
подключение результата для сравнения с исходной моделью."""
import json
import random
import re
import sys
import time
from pathlib import Path

from core import bench as B
from core import config as C
from core import db
from core import jobs
from core import tasks as T

EXAMPLES = C.TRAIN_DIR / 'examples.jsonl'     # примеры, сохранённые из «Диалога»


# ---------------- базовые модели MLX ----------------
def mlx_models(models_dir=None):
    """Папки моделей в формате MLX (config.json + *.safetensors) из каталога LM Studio и workspace/training/models."""
    roots = [Path(models_dir or C.load_settings()['lmstudio_models_dir']).expanduser(), C.TRAIN_DIR / 'models']
    out = []
    for root in roots:
        if not root.exists():
            continue
        for cfg in root.glob('*/*/config.json'):
            d = cfg.parent
            if not any(d.glob('*.safetensors')):
                continue
            try:
                conf = json.loads(cfg.read_text(encoding='utf-8'))
            except ValueError:
                conf = {}
            q = conf.get('quantization') or (conf.get('text_config') or {}).get('quantization') or {}
            size = sum(f.stat().st_size for f in d.glob('*.safetensors'))
            out.append(dict(path=str(d), name=f'{d.parent.name}/{d.name}', arch=(conf.get('architectures') or ['?'])[0],
                            bits=q.get('bits'), size_gb=size / 2 ** 30, fused=root == C.TRAIN_DIR / 'models'
                            or 'local' in d.parts))
    return sorted(out, key=lambda m: m['name'])


# ---------------- обучающие примеры ----------------
def _item_example(b: B.Benchmark, it: dict, letter_only=True):
    rqs = T.build_requests(b, it, perturb=False)
    if not rqs:
        return None
    msgs = [dict(m) for m in rqs[0]['messages']]
    t = b.type
    if t == 'mc':
        ans = str(it['answer']).upper()
        a = ans if letter_only else f"{ans}) {it['options'][B.LETTERS.index(ans)]}"
    elif t == 'short':
        a = (it.get('answers') or [''])[0]
    elif t == 'structured':
        if not it.get('expected'):
            return None
        a = json.dumps(it['expected'], ensure_ascii=False)
    elif t == 'open':
        a = it.get('reference') or ''
    else:
        return None
    if not str(a).strip():
        return None
    return dict(messages=msgs + [{'role': 'assistant', 'content': str(a)}],
                source=f'{b.id}/{it["id"]}', split=it.get('split', 'eval'))


def build_examples(bench_ids, splits=('train',), letter_only=True, graded_min=None, include_saved=True,
                   system_prompt=''):
    """Примеры из выбранных выборок своих наборов, одобренных ответов моделей и сохранённых из «Диалога»."""
    ex = []
    for bid in bench_ids:
        b = B.load(bid)
        for it in b.select(splits):
            e = _item_example(b, it, letter_only)
            if e:
                ex.append(e)
        if graded_min is not None and b.type == 'open':
            items = {str(i['id']): i for i in b.items if i.get('split', 'eval') in splits}
            resp = {r['id']: r for r in db.rows(
                'SELECT id, item_id, prompt, output FROM responses WHERE bench_id=? AND error IS NULL', (bid,))}
            for g in db.grades(list(resp)):
                r = resp[g['response_id']]
                if g['grader'].startswith('expert') and (g['score'] or 0) >= graded_min and r['item_id'] in items:
                    ex.append(dict(messages=json.loads(r['prompt']) + [{'role': 'assistant', 'content': r['output']}],
                                   source=f'{bid}/{r["item_id"]} (одобренный ответ)', split='train'))
    if include_saved and EXAMPLES.exists():
        for line in EXAMPLES.read_text(encoding='utf-8').splitlines():
            if line.strip():
                e = json.loads(line)
                ex.append(dict(messages=e['messages'], source=e.get('source', 'диалог'), split='train'))
    if system_prompt:
        for e in ex:
            if e['messages'][0]['role'] != 'system':
                e['messages'] = [{'role': 'system', 'content': system_prompt}] + e['messages']
    return ex


def save_example(messages, source='диалог'):
    EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    with EXAMPLES.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(messages=messages, source=source, ts=time.strftime('%Y-%m-%d %H:%M')),
                           ensure_ascii=False) + '\n')


def saved_examples():
    if not EXAMPLES.exists():
        return []
    return [json.loads(x) for x in EXAMPLES.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_saved_examples(examples):
    EXAMPLES.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in examples), encoding='utf-8')


# ---------------- проверка утечек ----------------
def _shingles(s, n=8):
    s = B.norm_text(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))} if s else set()


def _user_text(e):
    return '\n'.join(m['content'] for m in e['messages'] if m['role'] == 'user')


def leakage(examples, threshold=0.6):
    """Сравнение обучающих примеров со ВСЕМИ заданиями выборки «оценка» всех наборов:
    точные совпадения и доля общих 8-символьных фрагментов (≥ threshold)."""
    evals = []
    for m in B.list_benchmarks():
        b = B.load(m['id'])
        for it in b.select(('eval',), enabled_only=False):
            key = {'mc': 'question', 'short': 'question', 'structured': 'prompt', 'open': 'prompt',
                   'bias_pair': 'stereo', 'toxicity': 'text', 'perf': 'prompt'}.get(b.type)
            txt = str(it.get(key, ''))
            if txt:
                evals.append((f'{b.id}/{it["id"]}', B.norm_text(txt), _shingles(txt)))
    found = []
    for e in examples:
        u = _user_text(e)
        nu, su = B.norm_text(u), _shingles(u)
        for eid, nt, st in evals:
            if not st:
                continue
            if nt and nt in nu:
                found.append(dict(example=e['source'], eval_item=eid, overlap=1.0, kind='совпадение'))
                continue
            inter = len(su & st)
            if inter and inter / len(st) >= threshold:
                found.append(dict(example=e['source'], eval_item=eid, overlap=round(inter / len(st), 2),
                                  kind='пересечение'))
    return found


# ---------------- наборы в формате mlx-lm ----------------
def write_dataset(examples, out_dir: Path, valid_frac=0.1, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = [dict(messages=e['messages']) for e in examples]
    rng = random.Random(seed)
    rng.shuffle(ex)
    n_val = max(1, int(round(len(ex) * valid_frac))) if len(ex) > 1 else 0
    val, train = ex[:n_val], ex[n_val:]
    if not val and train:
        val = train[:1]
    for name, part in (('train', train), ('valid', val)):
        (out_dir / f'{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in part),
                                               encoding='utf-8')
    return len(train), len(val)


# ---------------- задания на обучение ----------------
def job_dir(name):
    return C.TRAIN_DIR / 'jobs' / name


def list_jobs():
    root = C.TRAIN_DIR / 'jobs'
    out = []
    for d in sorted(root.glob('*'), reverse=True) if root.exists() else []:
        j = C.read_json(d / 'job.json')
        if j:
            j['alive'] = jobs.alive(j.get('pid'))
            if j.get('state') == 'running' and not j['alive']:
                log = (d / 'train.log').read_text(encoding='utf-8', errors='replace') if (d / 'train.log').exists() else ''
                j['state'] = 'done' if (d / 'adapter' / 'adapters.safetensors').exists() and 'Saved final' in log \
                    else 'error'
                C.write_json(d / 'job.json', {k: v for k, v in j.items() if k != 'alive'})
            out.append(j)
    return out


def start_job(name, base_path, examples, params: dict, base_candidate=None):
    d = job_dir(name)
    if d.exists():
        raise ValueError(f'задание «{name}» уже есть')
    n_train, n_val = write_dataset(examples, d / 'data', params.get('valid_frac', 0.1), params.get('seed', 42))
    cfg = d / 'lora.yaml'
    d.mkdir(parents=True, exist_ok=True)
    cfg.write_text('lora_parameters:\n'
                   f"  rank: {int(params.get('rank', 8))}\n"
                   f"  scale: {float(params.get('scale', 20.0))}\n"
                   f"  dropout: {float(params.get('dropout', 0.0))}\n", encoding='utf-8')
    args = [sys.executable, '-m', 'tools.mlx_run', 'lora', '--model', base_path, '--train', '--data', str(d / 'data'),
            '--adapter-path', str(d / 'adapter'), '--fine-tune-type', params.get('type', 'lora'),
            '--iters', str(int(params.get('iters', 200))), '--batch-size', str(int(params.get('batch_size', 1))),
            '--num-layers', str(int(params.get('num_layers', 8))),
            '--learning-rate', str(float(params.get('lr', 1e-5))),
            '--max-seq-length', str(int(params.get('max_seq', 1024))),
            '--steps-per-report', str(int(params.get('report', 10))),
            '--steps-per-eval', str(int(params.get('eval_every', 50))),
            '--save-every', str(int(params.get('save_every', 100))), '--seed', str(int(params.get('seed', 42))),
            '-c', str(cfg)]
    if params.get('mask_prompt', True):
        args.append('--mask-prompt')
    if params.get('grad_checkpoint', True):
        args.append('--grad-checkpoint')
    p = jobs.spawn(args, d / 'train.log')
    job = dict(name=name, base_path=base_path, base_candidate=base_candidate, params=params, n_train=n_train,
               n_val=n_val, pid=p.pid, state='running', started=time.strftime('%Y-%m-%d %H:%M'), cmd=args)
    C.write_json(d / 'job.json', job)
    return job


_ITER = re.compile(r'Iter (\d+): Train loss ([\d.]+)')
_VAL = re.compile(r'Iter (\d+): Val loss ([\d.]+)')


def parse_log(text):
    train = [(int(a), float(b)) for a, b in _ITER.findall(text)]
    val = [(int(a), float(b)) for a, b in _VAL.findall(text)]
    return train, val


def fuse(job, out_name):
    """Объединяет адаптер с базовой моделью в папку LM Studio (models/local/<имя>) — модель видна в LM Studio."""
    target = Path(C.load_settings()['lmstudio_models_dir']).expanduser() / 'local' / out_name
    args = [sys.executable, '-m', 'tools.mlx_run', 'fuse', '--model', job['base_path'],
            '--adapter-path', str(job_dir(job['name']) / 'adapter'), '--save-path', str(target)]
    p = jobs.spawn(args, job_dir(job['name']) / 'fuse.log')
    job.update(fuse_pid=p.pid, fused_path=str(target))
    C.write_json(job_dir(job['name']) / 'job.json', {k: v for k, v in job.items() if k != 'alive'})
    return target


def serve(job, port=8090, with_adapter=True):
    """OpenAI-совместимый сервер mlx_lm: с адаптером — дообученная модель; без адаптера — исходная модель
    на том же движке (для честного сравнения «до/после»)."""
    port = jobs.free_port(port)
    args = [sys.executable, '-m', 'tools.mlx_run', 'server', '--model', job['base_path'],
            '--host', '127.0.0.1', '--port', str(port)]
    if with_adapter:
        args[5:5] = ['--adapter-path', str(job_dir(job['name']) / 'adapter')]
    p = jobs.spawn(args, job_dir(job['name']) / ('server.log' if with_adapter else 'server_base.log'))
    key = 'server' if with_adapter else 'base_server'
    job.update({f'{key}_pid': p.pid, f'{key}_port': port})
    C.write_json(job_dir(job['name']) / 'job.json', {k: v for k, v in job.items() if k != 'alive'})
    return port


def _set_backend(bid, name, port):
    s = C.load_settings()
    s['backends'] = [b for b in s['backends'] if b['id'] != bid] + [
        dict(id=bid, kind='openai', name=name, url=f'http://127.0.0.1:{port}')]
    C.save_settings(s)


def register_base_candidate(job, port):
    """Исходная модель через mlx_lm.server (без адаптера) — связывается с дообученной как «исходная»."""
    bid = f"mlx-base-{job['name']}"
    _set_backend(bid, f"Исходная {Path(job['base_path']).name} (mlx_lm.server :{port})", port)
    base = db.candidate(job.get('base_candidate')) if job.get('base_candidate') else None
    rs = (base or {}).get('reasoning') or 'off'
    cid = db.candidate_id(bid, job['base_path'], rs, 'исходная')
    db.upsert_candidate(dict(id=cid, backend=bid, model=job['base_path'], reasoning=rs,
                             label=f"{Path(job['base_path']).name} (исходная, mlx_lm)",
                             info=dict((base or {}).get('info') or {}), note='исходная модель на движке mlx_lm'))
    for c in db.candidates(True):     # дообученная модель этого задания сравнивается с ней
        if (c.get('info') or {}).get('adapter') == job['name']:
            c['base_id'] = cid
            db.upsert_candidate(c)
    return cid


def register_adapter_candidate(job, port):
    """Подключение «mlx_lm.server с адаптером» и кандидат, связанный с исходной моделью."""
    bid = f"mlx-{job['name']}"
    _set_backend(bid, f"Адаптер «{job['name']}» (mlx_lm.server :{port})", port)
    base = db.candidate(job.get('base_candidate')) if job.get('base_candidate') else None
    rs = (base or {}).get('reasoning') or 'off'          # тот же режим рассуждений, что у исходной модели
    cid = db.candidate_id(bid, job['base_path'], rs, job['name'])
    db.upsert_candidate(dict(id=cid, backend=bid, model=job['base_path'], reasoning=rs,
                             label=f"{Path(job['base_path']).name} + адаптер «{job['name']}»",
                             base_id=job.get('base_candidate'),
                             info=dict((base or {}).get('info') or {}, adapter=job['name']),
                             note=f"дообучение: {job['n_train']} примеров, {job['params'].get('iters')} итераций"))
    return cid
