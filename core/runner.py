"""Фоновый прогон: python -m core.runner <run_id>

Каждый ответ сразу пишется в базу, поэтому прогон можно остановить и продолжить:
уже полученные ответы (тот же запрос и те же настройки генерации) повторно не запрашиваются.
Остановка — сигнал SIGTERM: текущий запрос дорабатывает, модель выгружается."""
import argparse
import json
import math
import os
import re
import signal
import sys
import time
import traceback

from core import backends as BK
from core import bench as B
from core import config as C
from core import db
from core import metrics as M
from core import sysinfo
from core import tasks as T
from core import toxicity

STOP = False
LEAK_RE = re.compile(r'(the user|thinking|let me|let\'s|пользовател|сначала|рассужд|analy[sz]e)', re.I)


class Stopped(Exception):
    pass


def _on_signal(signum, frame):
    global STOP
    STOP = True
    log('получен сигнал остановки — завершаю текущий запрос')


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


class Progress:
    def __init__(self, run_id):
        self.run_id = run_id
        self.state = dict(db.run(run_id).get('progress') or {})
        self._last = 0.0

    def set(self, force=False, **kw):
        self.state.update(kw)
        now = time.time()
        if force or now - self._last > 0.7:
            db.update_run(self.run_id, progress=self.state, heartbeat=now)
            self._last = now


def check_stop():
    if STOP:
        raise Stopped()


def load_candidate(backend, cand, ctx, prog):
    """→ (handle, загружена_нами). Уже загруженную в LM Studio модель используем как есть."""
    if not backend.can_load:
        return cand['model'], False
    info = next((m for m in backend.list_models() if m['key'] == cand['model']), None)
    if info is None:
        raise BK.BackendError(f"модель {cand['model']} не найдена на сервере «{backend.id}»")
    meta = {k: info.get(k) for k in ('format', 'quant', 'size_bytes', 'params', 'arch', 'max_ctx')}
    if info.get('loaded'):
        log(f"  модель уже загружена ({info['loaded'][0]}) — использую её")
        _merge_info(cand, meta)
        return info['loaded'][0], False
    prog.set(force=True, stage='загрузка модели')
    m0, t0 = sysinfo.used_memory_gb(), time.time()
    handle = backend.load(cand['model'], ctx)
    load_s = time.time() - t0
    time.sleep(1.0)
    mem = sysinfo.used_memory_gb() - m0
    meta.update(load_s=round(load_s, 2), mem_gb=round(mem, 2) if not math.isnan(mem) else None)
    log(f'  загружена за {load_s:.1f} с, прирост занятой памяти ≈ {mem:.1f} ГБ')
    _merge_info(cand, meta)
    return handle, True


def _merge_info(cand, meta):
    c = db.candidate(cand['id'])
    info = dict(c.get('info') or {})
    info.update({k: v for k, v in meta.items() if v is not None})
    c['info'] = info
    db.upsert_candidate(c)


def probe(backend, handle, cand):
    """Пробный запрос: выявляет фактический режим рассуждений (у gpt-oss он не отключается)."""
    mt = 16 if cand.get('reasoning') in ('off', None, '', 'default') else 512
    r = backend.chat(handle, [{'role': 'user', 'content': 'Ответьте одним словом: готово.'}],
                     temperature=0, max_tokens=mt, reasoning=cand.get('reasoning') or None,
                     meta={'type': 'probe'})
    eff = r.effective_reasoning or cand.get('reasoning') or 'default'
    if eff in ('None', ''):
        eff = 'default'
    if eff != (cand.get('reasoning') or 'default'):
        log(f"  сервер не принял режим рассуждений «{cand.get('reasoning')}», фактически: {eff}")
    leak = eff == 'off' and len(r.text) > 30 and bool(LEAK_RE.search(r.text[:200]))
    c = db.candidate(cand['id'])
    if c and bool((c.get('info') or {}).get('reasoning_leak')) != leak:
        c['info'] = dict(c.get('info') or {}, reasoning_leak=leak)
        db.upsert_candidate(c)
    if leak:
        log('  ВНИМАНИЕ: режим рассуждений отключён, но модель пишет рассуждения в тексте ответа '
            f'(«{r.text[:70]}…»). Короткие ответы будут обрезаны — см. показатель «Соблюдение формата». '
            'Варианты: режим рассуждений «on» (рассуждения отделяются) или запуск модели через mlx_lm.server.')
    return eff


def run_bench(run_id, s, prog):
    app = C.load_settings()
    temperature, seed = float(s.get('temperature', 0)), int(s.get('seed', 0))
    splits, limit = tuple(s.get('splits') or ['eval']), int(s.get('limit') or 0)
    cands = s['candidates']
    benches = [B.load(bid) for bid in s['benches']]
    total = 0
    plan = {}
    for b in benches:
        items = b.select(splits, limit)
        reqs = [(it, rq) for it in items for rq in T.build_requests(
            b, it, perturb=s.get('perturb', True), repeats=1 if b.type == 'perf' else int(s.get('repeats', 1)), seed=seed)]
        plan[b.id] = reqs
        total += len(reqs)
    prog.set(force=True, candidates=cands, done_candidates=[], total_per_candidate=total, stage='старт')
    for ci, cid in enumerate(cands):
        check_stop()
        cand = db.candidate(cid)
        if not cand:
            log(f'нет кандидата {cid} — пропуск')
            continue
        log(f'=== {db.display_name(cand)}  [{cand["backend"]}: {cand["model"]}], рассуждения: {cand.get("reasoning") or "—"}')
        backend = BK.by_id(app, cand['backend'])
        prog.set(force=True, candidate=cid, candidate_index=ci, i=0, n=total, bench=None)
        handle, ours = load_candidate(backend, cand, int(s.get('context_length') or app['context_length']), prog)
        t_cand = time.time()
        try:
            eff = probe(backend, handle, cand)
            done_n = 0
            for b in benches:
                check_stop()
                reqs = plan[b.id]
                done = db.done_keys(cid, b.id)
                mt = T.max_tokens_for(b.meta, eff)
                todo = [(it, rq) for it, rq in reqs
                        if (str(it['id']), rq['variant'], rq['rep'],
                            T.req_hash(rq['messages'], mt, temperature, eff)) not in done]
                skipped = len(reqs) - len(todo)
                done_n += skipped
                log(f'  {b.name}: запросов {len(reqs)}' + (f', уже есть {skipped}' if skipped else ''))
                prog.set(force=True, bench=b.id, bench_name=b.name, stage=b.name, i=done_n, n=total)
                if b.type == 'perf' and todo and int(b.meta.get('warmup', 1)):
                    rq0 = todo[0][1]
                    backend.chat(handle, rq0['messages'], temperature=temperature, max_tokens=mt,
                                 reasoning=cand.get('reasoning') or None, seed=seed, meta=rq0['meta'])   # прогрев
                consecutive_err = 0
                t_b = time.time()
                for k, (it, rq) in enumerate(todo, 1):
                    check_stop()
                    h = T.req_hash(rq['messages'], mt, temperature, eff)
                    rec = dict(run_id=run_id, candidate_id=cid, bench_id=b.id, item_id=str(it['id']),
                               variant=rq['variant'], rep=rq['rep'], req_hash=h, prompt=T.prompt_key(rq['messages']))
                    try:
                        rep = backend.chat(handle, rq['messages'], temperature=temperature, max_tokens=mt,
                                           reasoning=cand.get('reasoning') or None, seed=seed,
                                           meta=rq['meta'] if backend.kind == 'demo' else None)
                        parsed, sc, detail = T.score(b, it, rq['meta'], rep.text)
                        for key in ('perm', 'stereo_letter'):
                            if key in rq['meta']:
                                detail[key] = rq['meta'][key]
                        if rep.raw.get('stats', {}).get('reasoning_output_tokens'):
                            detail['reasoning_tokens'] = rep.raw['stats']['reasoning_output_tokens']
                        rec.update(output=rep.text, reasoning=rep.reasoning or None, parsed=parsed, score=sc,
                                   detail=detail, latency=rep.latency,
                                   ttft=None if math.isnan(rep.ttft) else rep.ttft,
                                   tok_s=None if math.isnan(rep.tok_s) else rep.tok_s, n_out=rep.n_out)
                        consecutive_err = 0
                    except BK.BackendError as e:
                        rec.update(error=str(e)[:500], detail={})
                        consecutive_err += 1
                        log(f'  ошибка запроса: {str(e)[:200]}')
                        if consecutive_err >= 5:
                            raise BK.BackendError('5 ошибок подряд — прогон прерван: ' + str(e)[:300])
                    db.save_response(rec)
                    done_n += 1
                    el = time.time() - t_b
                    prog.set(i=done_n, n=total, eta=el / k * (len(todo) - k) + _eta_rest(plan, benches, b, el / k))
                if b.type == 'toxicity':
                    classify_toxicity(cid, b, prog)
                res = M.aggregate(b, cid, splits=splits)
                shown = ', '.join(f'{M.mname(m)} {res["values"][m]:.2f}' for m in M.TYPE_METRICS[b.type][:3]
                                  if m in res['values'])
                log(f'    {shown or "нет данных"}')
        finally:
            if ours and s.get('unload_after', True):
                prog.set(force=True, stage='выгрузка модели')
                backend.unload(handle)
        log(f'  готово за {(time.time() - t_cand) / 60:.1f} мин')
        prog.set(force=True, done_candidates=prog.state.get('done_candidates', []) + [cid])


def _eta_rest(plan, benches, current_b, per_req):
    after = False
    rest = 0
    for b in benches:
        if after:
            rest += len(plan[b.id])
        if b.id == current_b.id:
            after = True
    return rest * per_req


def classify_toxicity(cid, b, prog):
    rs = [r for r in db.responses(cid, b.id) if r['score'] is None and not r.get('error')]
    if not rs:
        return
    ok, where = toxicity.available()
    if not ok:
        log(f'  классификатор токсичности недоступен ({where}) — ответы сохранены без оценки')
        return
    prog.set(force=True, stage='классификация токсичности')
    log(f'  классификация токсичности: {len(rs)} ответов ({where})')
    probs = toxicity.classify([r['output'] or ' ' for r in rs])
    for r, p in zip(rs, probs):
        d = dict(r.get('detail') or {}, p_toxic=round(float(p), 4), method='классификатор')
        db.update_response(r['id'], score=float(p), detail=d)


def run_judge(run_id, s, prog):
    """Оценка развёрнутых ответов локальной моделью-судьёй."""
    app = C.load_settings()
    judge = db.candidate(s['judge'])
    backend = BK.by_id(app, judge['backend'])
    grader = f"judge:{judge['id']}"
    work = []
    for bid in s['benches']:
        b = B.load(bid)
        items = {str(it['id']): it for it in b.select(tuple(s.get('splits') or ['eval']))}
        for cid in s['candidates']:
            cur = M.current(b, list(items.values()), db.responses(cid, bid))
            ids = [r['id'] for v in cur.values() for r in v if not r.get('error')]
            graded = {g['response_id'] for g in db.grades(ids) if g['grader'] == grader}
            for (iid, var), rs in cur.items():
                for r in rs:
                    if not r.get('error') and (s.get('overwrite') or r['id'] not in graded):
                        work.append((b, items[iid], r))
    log(f'Модель-судья: {db.display_name(judge)}; ответов к оценке: {len(work)}')
    prog.set(force=True, candidate=judge['id'], stage='оценка моделью-судьёй', i=0, n=len(work))
    if not work:
        return
    handle, ours = load_candidate(backend, judge, int(app['context_length']) * 2, prog)
    try:
        eff = probe(backend, handle, judge)
        for k, (b, it, r) in enumerate(work, 1):
            check_stop()
            user_prompt = [m['content'] for m in json.loads(r['prompt']) if m['role'] == 'user'][-1]
            msgs = T.judge_messages(b, it, user_prompt, r['output'])
            try:
                rep = backend.chat(handle, msgs, temperature=0, max_tokens=T.max_tokens_for({'max_tokens': 400,
                                   'max_tokens_reasoning': 2048}, eff), reasoning=judge.get('reasoning') or None,
                                   meta={'type': 'judge'})
                frac, crit, comment = T.parse_judge(it, rep.text)
                if frac is None:
                    log(f'  {r["item_id"]}: {comment}')
                else:
                    db.save_grade(r['id'], grader, frac, crit, comment)
            except BK.BackendError as e:
                log(f'  ошибка: {str(e)[:200]}')
            prog.set(i=k, n=len(work))
    finally:
        if ours:
            backend.unload(handle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_id', type=int)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    r = db.run(a.run_id)
    if r is None:
        sys.exit(f'нет прогона {a.run_id}')
    db.update_run(a.run_id, state='running', started=time.time(), pid=os.getpid(), error=None, finished=None)
    prog = Progress(a.run_id)
    log(f'Прогон №{a.run_id} ({r["kind"]}): {r["name"]}')
    try:
        if r['kind'] == 'judge':
            run_judge(a.run_id, r['settings'], prog)
        else:
            run_bench(a.run_id, r['settings'], prog)
        prog.set(force=True, stage='завершено')
        db.update_run(a.run_id, state='done', finished=time.time())
        log('Прогон завершён.')
    except Stopped:
        prog.set(force=True, stage='остановлено пользователем')
        db.update_run(a.run_id, state='stopped', finished=time.time())
        log('Прогон остановлен. Полученные ответы сохранены — его можно продолжить.')
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        db.update_run(a.run_id, state='error', error=str(e)[:1000], finished=time.time())
        log(f'Ошибка: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
