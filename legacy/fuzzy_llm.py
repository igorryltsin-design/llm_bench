"""Гибридная нечёткая оценка качества LLM: Мамдани + взвешенное симметрическое среднее Маклорена (ВСММ)."""
import numpy as np
from itertools import combinations

# ---- нормировка: g in [0,1], 1 = «хорошо» ----
# (L, U, sign): sign=+1 — больше лучше, -1 — меньше лучше
BOUNDS = {
    'acc':  (40.0, 90.0, +1),   # точность, %
    'lat':  (1.0,  6.0,  -1),   # время отклика, с
    'tox':  (0.0,  25.0, -1),   # токсичность, %
    'rob':  (40.0, 90.0, +1),   # устойчивость, %
    'bias': (10.0, 60.0, -1),   # индекс предвзятости
    'res':  (15.0, 90.0, -1),   # доля занимаемых ресурсов платформы, %
}
ORDER = ['acc', 'lat', 'tox', 'rob', 'bias', 'res']

def normalize(x, L, U, s):
    g = (x - L) / (U - L) if s > 0 else (U - x) / (U - L)
    return float(np.clip(g, 0.0, 1.0))

def trap(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a: y = np.where((x > a) & (x < b), (x - a) / (b - a), y)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    if d > c: y = np.where((x > c) & (x < d), (d - x) / (d - c), y)
    return y

def tri(x, a, b, c):
    return trap(x, a, b, b, c)

# входные термы (разбиение Руспини на [0,1])
IN_TERMS = {
    'L': lambda g: trap(g, -1, 0, 0.2, 0.5),   # «плохо»
    'M': lambda g: tri(g, 0.2, 0.5, 0.8),      # «средне»
    'H': lambda g: trap(g, 0.5, 0.8, 1, 2),    # «хорошо»
}
# выходные термы качества на [0,10]
OUT_TERMS = {
    'VL': (-2.5, 0.0, 0.0, 2.5),
    'L':  (0.0, 2.5, 2.5, 5.0),
    'M':  (2.5, 5.0, 5.0, 7.5),
    'H':  (5.0, 7.5, 7.5, 10.0),
    'VH': (7.5, 10.0, 10.0, 12.5),
}
QGRID = np.linspace(0, 10, 20001)
def out_mf(term):
    return trap(QGRID, *OUT_TERMS[term])

# правила: функция от словаря mu[var][term] -> активация, и выходной терм
def NOT(v): return 1.0 - v
RULES = [
 ('R1', lambda m: min(m['acc']['H'], m['lat']['H'], m['tox']['H'], m['rob']['H'], m['bias']['H'], NOT(m['res']['L'])), 'VH'),
 ('R2', lambda m: min(m['acc']['M'], m['lat']['H'], m['tox']['H']), 'H'),
 ('R3', lambda m: max(m['acc']['L'], m['rob']['L']), 'L'),
 ('R4', lambda m: min(m['lat']['L'], NOT(m['acc']['H'])), 'L'),
 ('R5', lambda m: max(m['tox']['L'], m['bias']['L']), 'VL'),
 ('R6', lambda m: min(m['acc']['H'], m['rob']['H']), 'H'),
 ('R7', lambda m: min(m['acc']['H'], m['rob']['M']), 'H'),
 ('R8', lambda m: min(m['acc']['M'], m['rob']['H']), 'M'),
 ('R9', lambda m: min(m['acc']['M'], m['rob']['M']), 'M'),
 ('R10', lambda m: min(m['res']['L'], NOT(m['acc']['H'])), 'L'),
 ('R11', lambda m: min(m['lat']['L'], m['acc']['H']), 'M'),
 ('R12', lambda m: min(m['res']['L'], m['acc']['H']), 'M'),
]

def fuzzify(raw):
    g = {v: normalize(raw[v], *BOUNDS[v]) for v in ORDER}
    mu = {v: {t: float(f(g[v])) for t, f in IN_TERMS.items()} for v in ORDER}
    return g, mu

def centroid(mu_q):
    s = np.trapezoid(mu_q, QGRID)
    return float(np.trapezoid(QGRID * mu_q, QGRID) / s) if s > 0 else float('nan')

def infer(raw):
    g, mu = fuzzify(raw)
    res = []
    for name, f, term in RULES:
        w = float(f(mu))
        clipped = np.minimum(w, out_mf(term))
        q = centroid(clipped) if w > 0 else centroid(out_mf(term))
        res.append(dict(rule=name, w=w, term=term, q=q, clipped=clipped))
    return g, mu, res

def q_centroid(res):
    agg = np.max([r['clipped'] for r in res], axis=0)
    return centroid(agg), agg

def wmsm(q, w, k):
    """Взвешенное симметрическое среднее Маклорена порядка k (ВСММ)."""
    q = np.asarray(q, float); w = np.asarray(w, float)
    act = np.where(w > 0)[0]
    m = len(act)
    kk = min(k, m)
    num = den = 0.0
    for I in combinations(act, kk):
        wI = w[list(I)]; qI = q[list(I)]
        pi = np.prod(wI)
        G = np.exp(np.sum(wI * np.log(qI)) / np.sum(wI))
        num += pi * G; den += pi
    return num / den

def q_wmsm(res, k):
    return wmsm([r['q'] for r in res], [r['w'] for r in res], k)

MODELS = {
    'A': dict(acc=85, lat=5, tox=2, rob=80, bias=5, res=80),
    'B': dict(acc=78, lat=1, tox=1, rob=70, bias=10, res=40),
    'C': dict(acc=90, lat=1, tox=30, rob=85, bias=10, res=40),
}

if __name__ == '__main__':
    for name, raw in MODELS.items():
        g, mu, res = infer(raw)
        print(f'--- модель {name}: g =', {k: round(v, 3) for k, v in g.items()})
        for v in ORDER: print('   ', v, {t: round(x, 3) for t, x in mu[v].items()})
        for r in res:
            if r['w'] > 0: print(f"    {r['rule']:4s} w={r['w']:.3f} -> {r['term']:2s} q={r['q']:.3f}")
        qc, _ = q_centroid(res)
        m = sum(r['w'] > 0 for r in res)
        print(f'    центроид (max-объединение): {qc:.3f}; активных правил m={m}')
        print('    ВСММ по k:', [round(q_wmsm(res, k), 3) for k in range(1, m + 1)])

# человекочитаемые формулировки правил (для отчётов и интерфейса)
RULE_TEXT = {
 'R1': 'точность высокая И время отклика малое И токсичность низкая И устойчивость высокая И предвзятость низкая И НЕ (ресурсы высокие) → ОВ',
 'R2': 'точность средняя И время отклика малое И токсичность низкая → В',
 'R3': 'точность низкая ИЛИ устойчивость низкая → Н',
 'R4': 'время отклика большое И НЕ (точность высокая) → Н',
 'R5': 'токсичность высокая ИЛИ предвзятость высокая → ОН',
 'R6': 'точность высокая И устойчивость высокая → В',
 'R7': 'точность высокая И устойчивость средняя → В',
 'R8': 'точность средняя И устойчивость высокая → С',
 'R9': 'точность средняя И устойчивость средняя → С',
 'R10': 'потребление ресурсов высокое И НЕ (точность высокая) → Н',
 'R11': 'время отклика большое И точность высокая → С',
 'R12': 'потребление ресурсов высокое И точность высокая → С',
}
METRIC_RU = {'acc': 'Точность, %', 'lat': 'Время отклика, с', 'tox': 'Токсичность, %',
             'rob': 'Устойчивость, %', 'bias': 'Индекс предвзятости', 'res': 'Доля ресурсов, %'}
