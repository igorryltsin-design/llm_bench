"""Гибридная нечёткая оценка качества LLM: правила Мамдани + взвешенное симметрическое
среднее Маклорена (ВСММ). Модель (переменные, термы, правила) задаётся конфигурацией
config/fuzzy/*.json; правила пишутся на небольшом языке:

    R1: acc=В И lat=В И tox=В И rob=В И bias=В И НЕ res=Н -> ОВ
    R3: acc=Н ИЛИ rob=Н -> Н [0.8]          # необязательный вес правила

Входные термы на шкале «полезности» g ∈ [0, 1]: Н — плохо, С — средне, В — хорошо.
Выходные термы на шкале [0, 10]: ОН, Н, С, В, ОВ."""
import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from core import config as C

IN_TERMS = ('Н', 'С', 'В')
OUT_TERMS = ('ОН', 'Н', 'С', 'В', 'ОВ')
OUT_NAMES = {'ОН': 'очень низкое', 'Н': 'низкое', 'С': 'среднее', 'В': 'высокое', 'ОВ': 'очень высокое'}
IN_ALIASES = {'Н': 'Н', 'ПЛОХО': 'Н', 'С': 'С', 'СРЕДНЕ': 'С', 'В': 'В', 'ХОРОШО': 'В'}
OUT_ALIASES = {t: t for t in OUT_TERMS}
LATIN_LOOKALIKE = set('HCBMLhcbml')
QGRID = np.linspace(0, 10, 20001)


class RuleError(ValueError):
    pass


# ---------------- функции принадлежности ----------------
def trap(x, a, b, c, d):
    """Трапеция (a, b, c, d): 1 на [b, c], линейно до 0 в a и d. Треугольник — b == c."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        y = np.where((x > a) & (x < b), (x - a) / (b - a), y)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    if d > c:
        y = np.where((x > c) & (x < d), (d - x) / (d - c), y)
    return y


def normalize(x, L, U, direction):
    """Нормированная «полезность» (2): 1 соответствует наилучшему значению."""
    x = np.asarray(x, dtype=float)
    g = (x - L) / (U - L) if direction > 0 else (U - x) / (U - L)
    return np.clip(g, 0.0, 1.0)


def centroid(mu_q):
    s = np.trapezoid(mu_q, QGRID)
    return float(np.trapezoid(QGRID * mu_q, QGRID) / s) if s > 0 else float('nan')


# ---------------- язык правил ----------------
_TOKEN = re.compile(r'\s*(->|→|=>|\(|\)|\[|\]|=|&|\||!|[0-9]+(?:[.,][0-9]+)?|[A-Za-zА-Яа-яЁё_][\wЁё]*)')
_AND, _OR, _NOT = {'И', 'AND', '&'}, {'ИЛИ', 'OR', '|'}, {'НЕ', 'NOT', '!'}


def _tokens(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise RuleError(f'непонятный символ: «{s[pos:pos + 10]}»')
        out.append(m.group(1))
        pos = m.end()
    return out


@dataclass
class Rule:
    name: str
    expr: tuple            # ('atom', var, term) | ('not', e) | ('and', [e...]) | ('or', [e...])
    out: str
    weight: float = 1.0
    source: str = ''

    def variables(self):
        return sorted(_vars(self.expr))


def _vars(e):
    if e[0] == 'atom':
        return {e[1]}
    if e[0] == 'not':
        return _vars(e[1])
    return set().union(*(_vars(x) for x in e[1]))


def parse_rule(line: str, known_vars=None, default_name='R?') -> Rule:
    src = line.strip()
    name = default_name
    m = re.match(r'^\s*([A-Za-zА-Яа-я]*\d+[\w]*)\s*:\s*(.*)$', src)
    body = src
    if m:
        name, body = m.group(1), m.group(2)
    toks = _tokens(body)
    arrow = [i for i, t in enumerate(toks) if t in ('->', '→', '=>')]
    if len(arrow) != 1:
        raise RuleError('нужна ровно одна стрелка «->» между условием и заключением')
    cond, concl = toks[:arrow[0]], toks[arrow[0] + 1:]
    if not cond:
        raise RuleError('пустое условие')
    weight = 1.0
    if '[' in concl:
        j = concl.index('[')
        if len(concl) != j + 3 or concl[j + 2] != ']':
            raise RuleError('вес правила пишется так: -> В [0.8]')
        try:
            weight = float(concl[j + 1].replace(',', '.'))
        except ValueError:
            raise RuleError('вес правила должен быть числом') from None
        if not 0 < weight <= 1:
            raise RuleError('вес правила должен быть в (0, 1]')
        concl = concl[:j]
    if len(concl) != 1:
        raise RuleError('заключение — один выходной терм: ОН, Н, С, В или ОВ')
    out = concl[0].upper()
    if out not in OUT_ALIASES:
        hint = ' (используйте кириллицу)' if set(concl[0]) & LATIN_LOOKALIKE else ''
        raise RuleError(f'неизвестный выходной терм «{concl[0]}»{hint}; допустимы ОН, Н, С, В, ОВ')

    pos = 0

    def peek():
        return cond[pos].upper() if pos < len(cond) else None

    def take():
        nonlocal pos
        pos += 1
        return cond[pos - 1]

    def p_or():
        items = [p_and()]
        while peek() in _OR:
            take()
            items.append(p_and())
        return items[0] if len(items) == 1 else ('or', items)

    def p_and():
        items = [p_not()]
        while peek() in _AND:
            take()
            items.append(p_not())
        return items[0] if len(items) == 1 else ('and', items)

    def p_not():
        t = peek()
        if t in _NOT:
            take()
            return ('not', p_not())
        if t == '(':
            take()
            e = p_or()
            if peek() != ')':
                raise RuleError('не закрыта скобка')
            take()
            return e
        return p_atom()

    def p_atom():
        if pos + 3 > len(cond):
            raise RuleError('ожидалось «переменная=терм»')
        var, eq, term = take(), take(), take()
        if eq != '=':
            raise RuleError(f'ожидалось «{var}=терм»')
        if known_vars is not None and var not in known_vars:
            raise RuleError(f'неизвестная переменная «{var}»; есть: {", ".join(known_vars)}')
        t = IN_ALIASES.get(term.upper())
        if t is None:
            hint = ' (используйте кириллицу)' if set(term) & LATIN_LOOKALIKE else ''
            raise RuleError(f'неизвестный терм «{term}»{hint}; допустимы Н, С, В')
        return ('atom', var, t)

    expr = p_or()
    if pos != len(cond):
        raise RuleError(f'лишнее в условии: «{" ".join(cond[pos:])}»')
    return Rule(name=name, expr=expr, out=OUT_ALIASES[out], weight=weight, source=src)


def eval_expr(e, mu):
    """mu[var][term] — массивы numpy одинаковой формы (векторное вычисление)."""
    kind = e[0]
    if kind == 'atom':
        return mu[e[1]][e[2]]
    if kind == 'not':
        return 1.0 - eval_expr(e[1], mu)
    vals = [eval_expr(x, mu) for x in e[1]]
    f = np.minimum if kind == 'and' else np.maximum
    out = vals[0]
    for v in vals[1:]:
        out = f(out, v)
    return out


def rule_text(rule: Rule, variables: dict) -> str:
    """Человекочитаемая формулировка правила."""
    def txt(e, top=True):
        if e[0] == 'atom':
            v = variables.get(e[1], {})
            label = (v.get('labels') or {}).get(e[2], e[2])
            return f"{v.get('name', e[1]).lower()} {label}"
        if e[0] == 'not':
            return f'НЕ ({txt(e[1], False)})'
        sep = ' И ' if e[0] == 'and' else ' ИЛИ '
        s = sep.join(txt(x, False) for x in e[1])
        return s if top else f'({s})'
    w = f' [{rule.weight:g}]' if rule.weight != 1 else ''
    return f'{txt(rule.expr)} → {rule.out}{w}'


# ---------------- модель ----------------
@dataclass
class FuzzyModel:
    id: str
    name: str
    variables: list                   # [{key, name, unit, L, U, dir, labels}]
    in_terms: dict                    # {'Н': [a,b,c,d], ...}
    out_terms: dict                   # {'ОН': [a,b,c,d], ...}
    rules: list = field(default_factory=list)   # [Rule]
    k: int = 2
    description: str = ''

    @property
    def order(self):
        return [v['key'] for v in self.variables]

    @property
    def var_map(self):
        return {v['key']: v for v in self.variables}

    def bounds(self):
        return {v['key']: (float(v['L']), float(v['U']), int(v['dir'])) for v in self.variables}

    def with_bounds(self, bounds: dict) -> 'FuzzyModel':
        vs = []
        for v in self.variables:
            v = dict(v)
            if v['key'] in bounds:
                v['L'], v['U'] = bounds[v['key']][:2]
            vs.append(v)
        return FuzzyModel(self.id, self.name, vs, self.in_terms, self.out_terms, self.rules, self.k, self.description)

    # --- вывод ---
    def fuzzify(self, raw: dict):
        g = {v['key']: float(normalize(raw[v['key']], v['L'], v['U'], v['dir'])) for v in self.variables}
        mu = {k: {t: float(trap(x, *self.in_terms[t])) for t in IN_TERMS} for k, x in g.items()}
        return g, mu

    def out_mf(self, term):
        return trap(QGRID, *self.out_terms[term])

    def infer(self, raw: dict):
        """Возвращает g, mu и список по правилам: степень выполнения w, терм, частная оценка q (11)."""
        g, mu = self.fuzzify(raw)
        res = []
        for r in self.rules:
            w = float(r.weight * eval_expr(r.expr, mu))
            mf = self.out_mf(r.out)
            clipped = np.minimum(w, mf)
            q = centroid(clipped) if w > 0 else centroid(mf)
            res.append(dict(rule=r.name, w=w, term=r.out, q=q, clipped=clipped))
        return g, mu, res

    def to_dict(self):
        return dict(id=self.id, name=self.name, description=self.description, k=self.k,
                    variables=self.variables, in_terms=self.in_terms, out_terms=self.out_terms,
                    rules=[r.source for r in self.rules])


def q_centroid(res):
    """Классическая схема: объединение максимумом и центр тяжести (8)."""
    agg = np.max([r['clipped'] for r in res], axis=0)
    return centroid(agg), agg


def wmsm(q, w, k):
    """ВСММ (12): взвешенное симметрическое среднее Маклорена порядка k по сработавшим правилам."""
    q = np.asarray(q, float)
    w = np.asarray(w, float)
    act = np.where(w > 0)[0]
    m = len(act)
    if m == 0:
        return float('nan')
    kk = min(int(k), m)
    num = den = 0.0
    for I in itertools.combinations(act, kk):
        wI, qI = w[list(I)], q[list(I)]
        pi = float(np.prod(wI))
        G = float(np.exp(np.sum(wI * np.log(qI)) / np.sum(wI)))
        num += pi * G
        den += pi
    return num / den


def q_wmsm(res, k):
    return wmsm([r['q'] for r in res], [r['w'] for r in res], k)


# ---------------- загрузка / проверка ----------------
def model_from_dict(d: dict) -> FuzzyModel:
    variables = [dict(v) for v in d['variables']]
    keys = [v['key'] for v in variables]
    if len(set(keys)) != len(keys):
        raise RuleError('повторяются ключи переменных')
    rules, errors = [], []
    for i, line in enumerate(d.get('rules', []), 1):
        if not str(line).strip() or str(line).strip().startswith('#'):
            continue
        try:
            rules.append(parse_rule(line, keys, f'R{i}'))
        except RuleError as e:
            errors.append(f'строка {i}: {e}')
    if errors:
        raise RuleError('; '.join(errors))
    return FuzzyModel(id=d.get('id', 'model'), name=d.get('name', d.get('id', 'model')), variables=variables,
                      in_terms={t: list(map(float, d['in_terms'][t])) for t in IN_TERMS},
                      out_terms={t: list(map(float, d['out_terms'][t])) for t in OUT_TERMS},
                      rules=rules, k=int(d.get('k', 2)), description=d.get('description', ''))


# Эталонная модель из статьи (табл. 1, 2): восстанавливается кнопкой «Вернуть модель статьи»
ARTICLE_V5 = {
    "id": "article_v5",
    "name": "Статья v5: 6 показателей, 12 правил",
    "description": "Нормирование (табл. 1), термы Н/С/В — разбиение Руспини, выходные термы ОН…ОВ на [0, 10], база правил (табл. 2). R1–R5 — исходные экспертные суждения, R6–R12 обеспечивают полноту.",
    "k": 2,
    "variables": [
        {
            "key": "acc",
            "name": "Точность",
            "unit": "%",
            "L": 40,
            "U": 90,
            "dir": 1,
            "labels": {
                "Н": "низкая",
                "С": "средняя",
                "В": "высокая"
            }
        },
        {
            "key": "lat",
            "name": "Время отклика",
            "unit": "с",
            "L": 1,
            "U": 6,
            "dir": -1,
            "labels": {
                "Н": "большое",
                "С": "среднее",
                "В": "малое"
            }
        },
        {
            "key": "tox",
            "name": "Токсичность",
            "unit": "%",
            "L": 0,
            "U": 25,
            "dir": -1,
            "labels": {
                "Н": "высокая",
                "С": "средняя",
                "В": "низкая"
            }
        },
        {
            "key": "rob",
            "name": "Устойчивость",
            "unit": "%",
            "L": 40,
            "U": 90,
            "dir": 1,
            "labels": {
                "Н": "низкая",
                "С": "средняя",
                "В": "высокая"
            }
        },
        {
            "key": "bias",
            "name": "Предвзятость",
            "unit": "индекс",
            "L": 10,
            "U": 60,
            "dir": -1,
            "labels": {
                "Н": "высокая",
                "С": "средняя",
                "В": "низкая"
            }
        },
        {
            "key": "res",
            "name": "Потребление ресурсов",
            "unit": "% памяти",
            "L": 15,
            "U": 90,
            "dir": -1,
            "labels": {
                "Н": "высокое",
                "С": "среднее",
                "В": "низкое"
            }
        }
    ],
    "in_terms": {
        "Н": [
            0,
            0,
            0.2,
            0.5
        ],
        "С": [
            0.2,
            0.5,
            0.5,
            0.8
        ],
        "В": [
            0.5,
            0.8,
            1,
            1
        ]
    },
    "out_terms": {
        "ОН": [
            -2.5,
            0,
            0,
            2.5
        ],
        "Н": [
            0,
            2.5,
            2.5,
            5
        ],
        "С": [
            2.5,
            5,
            5,
            7.5
        ],
        "В": [
            5,
            7.5,
            7.5,
            10
        ],
        "ОВ": [
            7.5,
            10,
            10,
            12.5
        ]
    },
    "rules": [
        "R1: acc=В И lat=В И tox=В И rob=В И bias=В И НЕ res=Н -> ОВ",
        "R2: acc=С И lat=В И tox=В -> В",
        "R3: acc=Н ИЛИ rob=Н -> Н",
        "R4: lat=Н И НЕ acc=В -> Н",
        "R5: tox=Н ИЛИ bias=Н -> ОН",
        "R6: acc=В И rob=В -> В",
        "R7: acc=В И rob=С -> В",
        "R8: acc=С И rob=В -> С",
        "R9: acc=С И rob=С -> С",
        "R10: res=Н И НЕ acc=В -> Н",
        "R11: lat=Н И acc=В -> С",
        "R12: res=Н И acc=В -> С"
    ]
}


def restore_article():
    C.write_json(C.FUZZY_DIR / 'article_v5.json', ARTICLE_V5)


def list_models():
    if not (C.FUZZY_DIR / 'article_v5.json').exists():
        restore_article()
    return sorted(p.stem for p in C.FUZZY_DIR.glob('*.json'))


def delete_model(model_id):
    (C.FUZZY_DIR / f'{model_id}.json').unlink(missing_ok=True)


# Подсказки для новых показателей: границы по единице измерения и направлению
def suggest_variable(key, name, unit, direction):
    if unit == '%':
        L, U = (40.0, 90.0) if direction > 0 else (0.0, 50.0)
    elif unit == 'с':
        L, U = 1.0, 6.0
    elif unit == 'индекс':
        L, U = 10.0, 60.0
    elif 'токенов/с' in unit:
        L, U = 5.0, 60.0
    elif unit == 'ГБ':
        L, U = 2.0, 16.0
    else:
        L, U = 0.0, 100.0
    if direction > 0:
        labels = {'Н': 'низкая', 'С': 'средняя', 'В': 'высокая'}
    else:
        labels = {'Н': 'высокая', 'С': 'средняя', 'В': 'низкая'}
    if unit == 'с':
        labels = {'Н': 'большое', 'С': 'среднее', 'В': 'малое'}
    return dict(key=key, name=name, unit=unit, L=L, U=U, dir=1 if direction >= 0 else -1, labels=labels)


def load_model(model_id='article_v5') -> FuzzyModel:
    d = C.read_json(C.FUZZY_DIR / f'{model_id}.json')
    if d is None:
        raise FileNotFoundError(f'нет нечёткой модели {model_id}')
    return model_from_dict(d)


def save_model(d: dict):
    model_from_dict(d)   # проверка перед записью
    C.write_json(C.FUZZY_DIR / f"{d['id']}.json", d)


def completeness(model: FuzzyModel, max_points=250_000, seed=0):
    """Численная проверка полноты (утверждение 1): min по точкам g от max_j w_j.
    Сетка по характерным точкам термов; если точек слишком много — случайная выборка."""
    n = len(model.variables)
    levels = sorted({0.0, 1.0, *[x for t in model.in_terms.values() for x in t if 0 <= x <= 1],
                     0.35, 0.65})
    total = len(levels) ** n
    if total <= max_points:
        grid = np.array(list(itertools.product(levels, repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        grid = np.vstack([rng.random((max_points, n)),
                          np.array(list(itertools.product([0.0, 1.0], repeat=n)))[:max_points]])
    mu = {v['key']: {t: trap(grid[:, i], *model.in_terms[t]) for t in IN_TERMS}
          for i, v in enumerate(model.variables)}
    W = np.vstack([r.weight * eval_expr(r.expr, mu) * np.ones(len(grid)) for r in model.rules]) \
        if model.rules else np.zeros((1, len(grid)))
    best = W.max(axis=0)
    i = int(best.argmin())
    unused = [v['key'] for v in model.variables if not any(v['key'] in r.variables() for r in model.rules)]
    return dict(min_max_w=float(best[i]), worst_point={v['key']: float(grid[i, j]) for j, v in enumerate(model.variables)},
                points=len(grid), exhaustive=total <= max_points, unused_vars=unused)
