"""Хранилище результатов: SQLite в режиме WAL (одновременно пишет прогон и читает интерфейс)."""
import json
import sqlite3
import time
from contextlib import contextmanager

from core import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates(
  id TEXT PRIMARY KEY,
  backend TEXT NOT NULL,
  model TEXT NOT NULL,
  label TEXT,
  reasoning TEXT,
  params TEXT DEFAULT '{}',
  base_id TEXT,
  info TEXT DEFAULT '{}',
  note TEXT DEFAULT '',
  hidden INTEGER DEFAULT 0,
  created REAL
);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT DEFAULT 'bench',
  name TEXT,
  created REAL,
  started REAL,
  finished REAL,
  state TEXT,
  pid INTEGER,
  settings TEXT,
  progress TEXT DEFAULT '{}',
  error TEXT,
  app_version TEXT,
  heartbeat REAL
);
CREATE TABLE IF NOT EXISTS responses(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  candidate_id TEXT NOT NULL,
  bench_id TEXT NOT NULL,
  item_id TEXT NOT NULL,
  variant TEXT NOT NULL DEFAULT 'base',
  rep INTEGER NOT NULL DEFAULT 0,
  req_hash TEXT NOT NULL,
  prompt TEXT,
  output TEXT,
  reasoning TEXT,
  parsed TEXT,
  score REAL,
  detail TEXT DEFAULT '{}',
  latency REAL,
  ttft REAL,
  tok_s REAL,
  n_out INTEGER,
  error TEXT,
  ts REAL,
  UNIQUE(candidate_id, bench_id, item_id, variant, rep, req_hash)
);
CREATE INDEX IF NOT EXISTS ix_resp_cb ON responses(candidate_id, bench_id);
CREATE TABLE IF NOT EXISTS grades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  response_id INTEGER NOT NULL,
  grader TEXT NOT NULL,
  score REAL,
  criteria TEXT DEFAULT '{}',
  comment TEXT DEFAULT '',
  ts REAL,
  UNIQUE(response_id, grader)
);
CREATE TABLE IF NOT EXISTS rankings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  expert TEXT NOT NULL,
  scope TEXT NOT NULL,
  ranking TEXT NOT NULL,
  ts REAL,
  UNIQUE(expert, scope)
);
CREATE TABLE IF NOT EXISTS manual_metrics(
  candidate_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL,
  source TEXT NOT NULL DEFAULT 'manual',
  ts REAL,
  PRIMARY KEY(candidate_id, metric, source)
);
"""


def connect():
    con = sqlite3.connect(C.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    return con


@contextmanager
def tx():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with tx() as con:
        con.executescript(SCHEMA)


def rows(sql, params=()):
    with tx() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def one(sql, params=()):
    r = rows(sql, params)
    return r[0] if r else None


def execute(sql, params=()):
    with tx() as con:
        cur = con.execute(sql, params)
        return cur.lastrowid


def _j(x):
    return json.dumps(x, ensure_ascii=False)


def _load(r, *fields):
    if r is None:
        return None
    for f in fields:
        if f in r and isinstance(r[f], str):
            try:
                r[f] = json.loads(r[f])
            except ValueError:
                pass
    return r


# ---------------- кандидаты ----------------
def candidate_id(backend, model, reasoning, label=''):
    cid = f'{backend}:{model}@{reasoning or "default"}'
    return f'{cid}#{label}' if label else cid


def upsert_candidate(c: dict):
    now = time.time()
    with tx() as con:
        con.execute("""INSERT INTO candidates(id, backend, model, label, reasoning, params, base_id, info, note, hidden, created)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET label=excluded.label, params=excluded.params,
                         base_id=excluded.base_id, info=excluded.info, note=excluded.note, hidden=excluded.hidden""",
                    (c['id'], c['backend'], c['model'], c.get('label') or '', c.get('reasoning') or '',
                     _j(c.get('params') or {}), c.get('base_id'), _j(c.get('info') or {}), c.get('note') or '',
                     int(c.get('hidden') or 0), c.get('created') or now))


def candidates(include_hidden=False):
    q = 'SELECT * FROM candidates' + ('' if include_hidden else ' WHERE hidden=0') + ' ORDER BY created'
    return [_load(r, 'params', 'info') for r in rows(q)]


def candidate(cid):
    return _load(one('SELECT * FROM candidates WHERE id=?', (cid,)), 'params', 'info')


def delete_candidate(cid, with_responses=False):
    with tx() as con:
        if with_responses:
            con.execute('DELETE FROM grades WHERE response_id IN (SELECT id FROM responses WHERE candidate_id=?)', (cid,))
            con.execute('DELETE FROM responses WHERE candidate_id=?', (cid,))
            con.execute('DELETE FROM manual_metrics WHERE candidate_id=?', (cid,))
        con.execute('DELETE FROM candidates WHERE id=?', (cid,))


def display_name(c):
    if not c:
        return '—'
    name = c.get('label') or c['model']
    rs = c.get('reasoning')
    return f'{name} ({rs})' if rs and rs not in ('default', 'off', '') and not c.get('label') else name


# ---------------- прогоны ----------------
def create_run(kind, name, settings):
    now = time.time()
    return execute('INSERT INTO runs(kind, name, created, state, settings, app_version) VALUES(?,?,?,?,?,?)',
                   (kind, name, now, 'queued', _j(settings), C.APP_VERSION))


def update_run(run_id, **kw):
    if not kw:
        return
    for k in ('settings', 'progress'):
        if k in kw and not isinstance(kw[k], str):
            kw[k] = _j(kw[k])
    cols = ', '.join(f'{k}=?' for k in kw)
    execute(f'UPDATE runs SET {cols} WHERE id=?', (*kw.values(), run_id))


def run(run_id):
    return _load(one('SELECT * FROM runs WHERE id=?', (run_id,)), 'settings', 'progress')


def runs(limit=50):
    return [_load(r, 'settings', 'progress') for r in rows('SELECT * FROM runs ORDER BY id DESC LIMIT ?', (limit,))]


# ---------------- ответы ----------------
RESP_FIELDS = ('run_id', 'candidate_id', 'bench_id', 'item_id', 'variant', 'rep', 'req_hash', 'prompt', 'output',
               'reasoning', 'parsed', 'score', 'detail', 'latency', 'ttft', 'tok_s', 'n_out', 'error', 'ts')


def save_response(r: dict):
    r = dict(r, ts=time.time())
    if not isinstance(r.get('detail'), str):
        r['detail'] = _j(r.get('detail') or {})
    vals = [r.get(f) for f in RESP_FIELDS]
    with tx() as con:
        cur = con.execute(f"""INSERT INTO responses({','.join(RESP_FIELDS)}) VALUES({','.join('?' * len(RESP_FIELDS))})
                              ON CONFLICT(candidate_id, bench_id, item_id, variant, rep, req_hash) DO UPDATE SET
                              run_id=excluded.run_id, output=excluded.output, reasoning=excluded.reasoning,
                              parsed=excluded.parsed, score=excluded.score, detail=excluded.detail,
                              latency=excluded.latency, ttft=excluded.ttft, tok_s=excluded.tok_s,
                              n_out=excluded.n_out, error=excluded.error, ts=excluded.ts""", vals)
        return cur.lastrowid


def update_response(resp_id, **kw):
    if 'detail' in kw and not isinstance(kw['detail'], str):
        kw['detail'] = _j(kw['detail'])
    cols = ', '.join(f'{k}=?' for k in kw)
    execute(f'UPDATE responses SET {cols} WHERE id=?', (*kw.values(), resp_id))


def done_keys(candidate_id, bench_id):
    """Ключи уже полученных (без ошибки) ответов — для продолжения прогона."""
    return {(r['item_id'], r['variant'], r['rep'], r['req_hash']) for r in rows(
        'SELECT item_id, variant, rep, req_hash FROM responses WHERE candidate_id=? AND bench_id=? AND error IS NULL',
        (candidate_id, bench_id))}


def responses(candidate_id=None, bench_id=None):
    q, p = 'SELECT * FROM responses WHERE 1=1', []
    if candidate_id:
        q += ' AND candidate_id=?'
        p.append(candidate_id)
    if bench_id:
        q += ' AND bench_id=?'
        p.append(bench_id)
    return [_load(r, 'detail') for r in rows(q + ' ORDER BY id', p)]


def delete_responses(candidate_id, bench_id=None):
    q, p = 'candidate_id=?', [candidate_id]
    if bench_id:
        q += ' AND bench_id=?'
        p.append(bench_id)
    with tx() as con:
        con.execute(f'DELETE FROM grades WHERE response_id IN (SELECT id FROM responses WHERE {q})', p)
        con.execute(f'DELETE FROM responses WHERE {q}', p)


# ---------------- оценки ответов ----------------
def save_grade(response_id, grader, score, criteria=None, comment=''):
    with tx() as con:
        con.execute("""INSERT INTO grades(response_id, grader, score, criteria, comment, ts) VALUES(?,?,?,?,?,?)
                       ON CONFLICT(response_id, grader) DO UPDATE SET score=excluded.score,
                       criteria=excluded.criteria, comment=excluded.comment, ts=excluded.ts""",
                    (response_id, grader, score, _j(criteria or {}), comment, time.time()))


def grades(response_ids=None):
    if response_ids is None:
        return [_load(r, 'criteria') for r in rows('SELECT * FROM grades')]
    out = []
    ids = list(response_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        out += rows(f'SELECT * FROM grades WHERE response_id IN ({",".join("?" * len(chunk))})', chunk)
    return [_load(r, 'criteria') for r in out]


def delete_grade(response_id, grader):
    execute('DELETE FROM grades WHERE response_id=? AND grader=?', (response_id, grader))


# ---------------- ранжировки экспертов ----------------
def save_ranking(expert, scope, ranking):
    execute("""INSERT INTO rankings(expert, scope, ranking, ts) VALUES(?,?,?,?)
               ON CONFLICT(expert, scope) DO UPDATE SET ranking=excluded.ranking, ts=excluded.ts""",
            (expert, scope, _j(ranking), time.time()))


def rankings(scope):
    return [_load(r, 'ranking') for r in rows('SELECT * FROM rankings WHERE scope=? ORDER BY expert', (scope,))]


def delete_ranking(expert, scope):
    execute('DELETE FROM rankings WHERE expert=? AND scope=?', (expert, scope))


# ---------------- показатели, введённые вручную или импортированные ----------------
def set_manual_metric(candidate_id, metric, value, source='manual'):
    execute("""INSERT INTO manual_metrics(candidate_id, metric, value, source, ts) VALUES(?,?,?,?,?)
               ON CONFLICT(candidate_id, metric, source) DO UPDATE SET value=excluded.value, ts=excluded.ts""",
            (candidate_id, metric, value, source, time.time()))


def manual_metrics():
    return rows('SELECT * FROM manual_metrics')


init()
