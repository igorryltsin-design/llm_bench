#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Обобщённая оценка моделей по results/raw_metrics.csv: центроид Мамдани, ВСММ (k=1,2,m),
взвешенная сумма и TOPSIS. Результат: results/scores.csv.

    python evaluate.py                    # границы нормирования — как в статье (табл. 1)
    python evaluate.py --lat 5 20         # переопределить L и U для времени отклика, с
"""
import argparse, csv, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import fuzzy_llm as F

DEFAULT_BOUNDS = dict(F.BOUNDS)

def topsis(X, w, sign):
    X = np.asarray(X, float); nr = np.linalg.norm(X, axis=0); nr[nr == 0] = 1; v = X / nr * w
    best = np.where(sign > 0, v.max(0), v.min(0)); worst = np.where(sign > 0, v.min(0), v.max(0))
    dp = np.linalg.norm(v - best, axis=1); dm = np.linalg.norm(v - worst, axis=1)
    den = dp + dm; den[den == 0] = 1
    return 10 * dm / den

def score(rows, bounds=None, k=2, ks=range(1, 7)):
    """rows: список словарей с полями model + F.ORDER. Возвращает (оценки, пропущенные)."""
    F.BOUNDS.clear(); F.BOUNDS.update(bounds or DEFAULT_BOUNDS)
    out, X, skipped = [], [], []
    for r in rows:
        try: raw = {v: float(r[v]) for v in F.ORDER}
        except (KeyError, TypeError, ValueError): skipped.append((r.get('model'), 'нет данных')); continue
        if any(math.isnan(x) for x in raw.values()):
            skipped.append((r['model'], 'есть незаполненные показатели')); continue
        g, mu, res = F.infer(raw); act = [x for x in res if x['w'] > 0]; m = len(act)
        out.append(dict(model=r['model'], wmsm_k=F.q_wmsm(res, k), centroid=F.q_centroid(res)[0],
                        wmsm1=F.q_wmsm(res, 1), wmsm_m=F.q_wmsm(res, m), m=m,
                        wsum=10 * float(np.mean([g[v] for v in F.ORDER])),
                        by_k={kk: F.q_wmsm(res, kk) for kk in ks},
                        g=g, rules=[dict(rule=x['rule'], w=x['w'], term=x['term'], q=x['q']) for x in act]))
        X.append([raw[v] for v in F.ORDER])
    if out:
        sign = np.array([F.BOUNDS[v][2] for v in F.ORDER])
        for o, t in zip(out, topsis(X, np.full(6, 1 / 6), sign)): o['topsis'] = float(t)
    out.sort(key=lambda o: -o['wmsm_k'])
    return out, skipped

def main():
    ap = argparse.ArgumentParser()
    for v in F.ORDER: ap.add_argument(f'--{v}', nargs=2, type=float, metavar=('L', 'U'))
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--file', default='raw_metrics.csv')
    a = ap.parse_args()
    b = dict(DEFAULT_BOUNDS)
    for v in F.ORDER:
        if getattr(a, v): L, U = getattr(a, v); b[v] = (L, U, b[v][2])
    rows = list(csv.DictReader((HERE / 'results' / a.file).open(encoding='utf-8')))
    out, skipped = score(rows, b, a.k)
    for m, why in skipped: print('пропуск:', m, '—', why)
    keys = ['model', 'wmsm_k', 'centroid', 'wmsm1', 'wmsm_m', 'm', 'wsum', 'topsis']
    with (HERE / 'results' / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(keys + ['rules'])
        for o in out:
            w.writerow([o['model']] + [round(o[x], 3) if isinstance(o[x], float) else o[x] for x in keys[1:]]
                       + [' '.join(f"{r['rule']}:{r['w']:.2f}" for r in o['rules'])])
    for o in out:
        print(f"{o['model']:50s} ВСММ(k={a.k})={o['wmsm_k']:.2f}  центроид={o['centroid']:.2f}  "
              f"сумма={o['wsum']:.2f}  TOPSIS={o['topsis']:.2f}")

if __name__ == '__main__':
    main()
