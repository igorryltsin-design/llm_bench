"""Проверки ядра: ./.venv/bin/python -m pytest -q"""
import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import bench as B      # noqa: E402
from core import config as C     # noqa: E402
from core import fuzzy as F      # noqa: E402
from core import metrics as M    # noqa: E402
from core import tasks as T      # noqa: E402

ARTICLE = {'A': dict(acc=85, lat=5, tox=2, rob=80, bias=5, res=80),
           'B': dict(acc=78, lat=1, tox=1, rob=70, bias=10, res=40),
           'C': dict(acc=90, lat=1, tox=30, rob=85, bias=10, res=40)}
TABLE5 = {'A': (6.25, 5.83, 5.75, 5.72), 'B': (7.17, 7.39, 7.48, 7.29), 'C': (5.28, 4.17, 2.50, 2.50)}


# ---------------- нечёткая модель ----------------
def test_article_table5():
    m = F.model_from_dict(F.ARTICLE_V5)
    for name, raw in ARTICLE.items():
        g, mu, res = m.infer(raw)
        mact = sum(r['w'] > 0 for r in res)
        got = (F.q_centroid(res)[0], F.q_wmsm(res, 1), F.q_wmsm(res, 2), F.q_wmsm(res, mact))
        assert [round(x, 2) for x in got] == list(TABLE5[name]), name


def test_article_completeness():
    res = F.completeness(F.model_from_dict(F.ARTICLE_V5))
    assert res['exhaustive'] and res['min_max_w'] >= 0.5 and not res['unused_vars']


def test_incomplete_base_detected():
    d = dict(F.ARTICLE_V5, rules=F.ARTICLE_V5['rules'][:5])      # исходные R1–R5 без правил полноты
    assert F.completeness(F.model_from_dict(d))['min_max_w'] == 0


@pytest.mark.parametrize('line', ['acc=H -> В', 'acc=В И -> В', 'foo=В -> В', 'acc=В -> X', '(acc=В -> В',
                                  'acc=В -> В [1.5]', 'acc=В'])
def test_rule_errors(line):
    with pytest.raises(F.RuleError):
        F.parse_rule(line, ['acc', 'lat'])


def test_rule_parse_and_eval():
    r = F.parse_rule('R7: (acc=В ИЛИ lat=С) И НЕ lat=Н -> В [0.5]', ['acc', 'lat'])
    assert r.name == 'R7' and r.out == 'В' and r.weight == 0.5
    mu = {'acc': {'Н': 0, 'С': 0.2, 'В': 0.8}, 'lat': {'Н': 0.3, 'С': 0.7, 'В': 0}}
    assert F.eval_expr(r.expr, mu) == pytest.approx(0.7)       # min(max(0.8, 0.7), 1 − 0.3)


def test_wmsm_properties():
    q, w = [2.0, 5.0, 9.0], [0.3, 1.0, 0.6]
    for k in (1, 2, 3):
        v = F.wmsm(q, w, k)
        assert min(q) <= v <= max(q)                            # ограниченность
        assert F.wmsm([4.0] * 3, w, k) == pytest.approx(4.0)    # идемпотентность
    assert F.wmsm(q, w, 1) >= F.wmsm(q, w, 2) >= F.wmsm(q, w, 3)


# ---------------- разбор и проверка ответов ----------------
@pytest.mark.parametrize('text,exp', [('B', 'B'), ('Ответ: C', 'C'), ('В', 'B'), ('A) Париж', 'A'),
                                      ('не знаю', None), ('<think>думаю A</think>\nD', 'D'), ('вариант А', 'A')])
def test_parse_letter(text, exp):
    assert T.parse_letter(text) == exp


def _bench(btype, items, **meta):
    m = B.default_meta('t', 'тест', btype)
    m.update(meta)
    return B.Benchmark(m, items)


def test_short_number_and_regex():
    b = _bench('short', [], scoring={'mode': 'number', 'tolerance': 0.01, 'rel_tolerance': 0})
    it = {'id': '1', 'question': '?', 'answers': ['4.5']}
    assert T.score(b, it, {}, 'Ответ: 4,5 часа')[1] == 1.0
    assert T.score(b, it, {}, '4,52')[1] == 0.0
    b.meta['scoring'] = {'mode': 'regex'}
    assert T.score(b, {'id': '1', 'answers': [r'\b1945\b']}, {}, 'в 1945 году')[1] == 1.0


def test_structured_scoring():
    it = {'id': '1', 'prompt': 'x', 'schema': {'type': 'object', 'required': ['a', 'b'],
                                                 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'string'}}},
          'expected': {'a': 3, 'b': 'Тверь'}}
    b = _bench('structured', [it])
    parsed, sc, d = T.score(b, it, {}, 'Вот ответ:\n```json\n{"a": 3, "b": "тверь"}\n```')
    assert d['json_ok'] and d['schema_ok'] and sc == 1.0
    parsed, sc, d = T.score(b, it, {}, '{"a": "три"}')
    assert d['json_ok'] and not d['schema_ok'] and sc < 0.5


def test_mc_requests_and_shuffle_mapping():
    it = {'id': 'q1', 'question': 'Сколько будет 2+2?', 'options': ['3', '4', '5', '6'], 'answer': 'B'}
    b = _bench('mc', [it])
    reqs = {r['variant']: r for r in T.build_requests(b, it)}
    assert set(reqs) == {'base', 'typo', 'shuffle', 'instr'}
    assert reqs['base']['messages'][-1]['content'].startswith('Выберите один правильный ответ. Ответьте только буквой (A, B, C или D).')
    sh = reqs['shuffle']['meta']
    parsed, sc, d = T.score(b, it, sh, sh['expected'])
    assert sc == 1.0 and d['mapped'] == 'B'


# ---------------- статистика ----------------
def test_statistics():
    lo, hi = M.wilson(80, 100)
    assert 0.70 < lo < 0.72 and 0.86 < hi < 0.88
    assert M.mcnemar(0, 10) < 0.01 and M.mcnemar(5, 5) == 1.0
    assert M.kendall_w([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) == pytest.approx(1.0)
    assert M.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert M.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert M.pareto_front([[1, 1], [2, 2], [3, 0]], [1, 1]) == [1, 2]


# ---------------- прогон на имитации: продолжение без повторных запросов ----------------
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    from core import db
    monkeypatch.setattr(C, 'DB_PATH', tmp_path / 'bench.db')
    monkeypatch.setattr(C, 'BENCH_DIR', tmp_path / 'benchmarks')
    monkeypatch.setattr(C, 'SETTINGS_PATH', tmp_path / 'settings.json')
    C.BENCH_DIR.mkdir()
    db.init()
    return db


def test_demo_run_and_resume(sandbox, monkeypatch):
    db = sandbox
    from core import backends as BK
    from core import runner
    items = [{'id': f'q{i}', 'question': f'Вопрос {i}', 'options': ['а', 'б', 'в', 'г'], 'answer': 'ABCD'[i % 4]}
             for i in range(6)]
    B.save(_bench('mc', items, id='mc'), 'тест')
    db.upsert_candidate(dict(id='demo:demo/strong-7b@off', backend='demo', model='demo/strong-7b', reasoning='off'))
    calls = {'n': 0}
    real = BK.Demo.chat

    def counting(self, *a, **kw):
        calls['n'] += 1
        return real(self, *a, **kw)

    monkeypatch.setattr(BK.Demo, 'chat', counting)
    settings = dict(candidates=['demo:demo/strong-7b@off'], benches=['mc'], splits=['eval'], limit=0, temperature=0,
                    seed=0, repeats=1, perturb=True)
    rid = db.create_run('bench', 'т', settings)
    runner.run_bench(rid, settings, runner.Progress(rid))
    first = calls['n']
    assert db.one('SELECT COUNT(*) n FROM responses')['n'] == 24          # 6 заданий × 4 варианта
    runner.run_bench(rid, settings, runner.Progress(rid))
    assert calls['n'] - first == 1                                        # только пробный запрос, ответы из базы
    db.execute("DELETE FROM responses WHERE item_id='q0'")
    runner.run_bench(rid, settings, runner.Progress(rid))
    assert calls['n'] - first == 1 + 1 + 4                                 # пробный + пробный + 4 варианта q0
    res = M.aggregate(B.load('mc'), 'demo:demo/strong-7b@off')
    assert res['n'] == 6 and 0 <= res['values']['acc'] <= 100 and 'rob' in res['values']


# ---------------- дообучение: утечки ----------------
def test_leakage(sandbox):
    from core import training as TR
    items = [{'id': 'e1', 'question': 'Какой метод ранжирует по близости к идеальному решению?',
              'options': ['а', 'б'], 'answer': 'A', 'split': 'eval'},
             {'id': 't1', 'question': 'Совсем другой вопрос про логистику и склады', 'options': ['а', 'б'],
              'answer': 'B', 'split': 'train'}]
    B.save(_bench('mc', items, id='own'), 'тест')
    clean = TR.build_examples(['own'], ('train',), include_saved=False)
    assert len(clean) == 1 and not TR.leakage(clean)
    dirty = TR.build_examples(['own'], ('train', 'eval'), include_saved=False)
    assert TR.leakage(dirty)


# ---------------- рисунки ----------------
def test_figure_size_for_word():
    from PIL import Image
    from core import charts as CH
    st = CH.Style(caption='Проверка', number='1', width_cm=16)
    png = CH.to_bytes(CH.bars(st, ['а', 'б'], {'x': [1.5, 2.5]}), 'png', 300)
    im = Image.open(io.BytesIO(png))
    assert abs(im.size[0] / im.info['dpi'][0] * 2.54 - 16) < 0.05
    svg = CH.to_bytes(CH.bars(st, ['а'], {'x': [1.0]}), 'svg').decode()
    assert 'Times New Roman' in svg and 'Рис. 1. Проверка' in svg


# ---------------- офлайн: в коде приложения нет внешних адресов ----------------
def test_no_external_urls():
    bad = []
    for p in list((ROOT / 'core').glob('*.py')) + list((ROOT / 'ui').glob('*.py')) + \
            list((ROOT / 'app_pages').glob('*.py')) + [ROOT / 'app.py']:
        for url in re.findall(r'https?://[^\s\'"]+', p.read_text(encoding='utf-8')):
            if not re.match(r'https?://(127\.0\.0\.1|localhost)', url):
                bad.append(f'{p.name}: {url}')
    assert not bad, bad
