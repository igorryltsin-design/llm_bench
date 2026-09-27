"""Сводная оценка: профиль (показатели ← тестовые наборы) → нечёткий вывод (ВСММ, центроид)
и базовые свёртки (взвешенная сумма, взвешенное геометрическое среднее, TOPSIS), парето-фронт,
чувствительность, согласованность с экспертами (Кендалл)."""
import math

import numpy as np

from core import bench as B
from core import config as C
from core import db
from core import fuzzy as F
from core import metrics as M

METHODS = [('wmsm_k', 'ВСММ'), ('centroid', 'Центроид Мамдани'), ('wsum', 'Взвешенная сумма'),
           ('wgeo', 'Взвешенное геом. среднее'), ('topsis', 'TOPSIS')]
METHOD_RU = dict(METHODS)


# ---------------- профили ----------------
def list_profiles():
    return sorted(p.stem for p in C.PROFILE_DIR.glob('*.json'))


def load_profile(pid='article'):
    p = C.read_json(C.PROFILE_DIR / f'{pid}.json')
    if p is None:
        raise FileNotFoundError(f'нет профиля {pid}')
    return p


def save_profile(p):
    C.write_json(C.PROFILE_DIR / f"{p['id']}.json", p)


def indicator_table(profile, candidates, platform_gb, splits=('eval',)):
    """→ {cid: {показатель: значение}}, {cid: {показатель: источник}}, {(cid, bench): aggregate}."""
    benches = {}
    for srcs in profile['indicators'].values():
        for s in srcs:
            if s.get('bench') and s['bench'] not in benches and B.exists(s['bench']):
                benches[s['bench']] = B.load(s['bench'])
    manual = {}
    for r in db.manual_metrics():
        manual.setdefault(r['candidate_id'], {})[r['metric']] = (r['value'], r['source'])
    values, sources, aggs = {}, {}, {}
    for c in candidates:
        cid = c['id']
        mm = M.model_metrics(c, platform_gb)
        vals, src = {}, {}
        for ind, srcs in profile['indicators'].items():
            acc, wsum, used = 0.0, 0.0, []
            for s in srcs:
                w = float(s.get('weight', 1))
                v = None
                if s.get('bench') in benches:
                    key = (cid, s['bench'])
                    if key not in aggs:
                        aggs[key] = M.aggregate(benches[s['bench']], cid, splits=splits)
                    v = aggs[key]['values'].get(s.get('metric', ind))
                    lbl = benches[s['bench']].name
                elif s.get('model'):
                    v = mm.get(s['model'])
                    lbl = 'сведения о модели'
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    acc += w * v
                    wsum += w
                    used.append(lbl)
            if wsum > 0:
                vals[ind], src[ind] = acc / wsum, ', '.join(used)
            elif profile.get('use_manual', True) and ind in manual.get(cid, {}):
                vals[ind], src[ind] = manual[cid][ind][0], f'импорт: {manual[cid][ind][1]}'
        values[cid], sources[cid] = vals, src
    return values, sources, aggs


# ---------------- свёртки ----------------
def topsis(X, w, sign):
    X = np.asarray(X, float)
    nr = np.linalg.norm(X, axis=0)
    nr[nr == 0] = 1
    v = X / nr * w
    best = np.where(sign > 0, v.max(0), v.min(0))
    worst = np.where(sign > 0, v.min(0), v.max(0))
    dp = np.linalg.norm(v - best, axis=1)
    dm = np.linalg.norm(v - worst, axis=1)
    den = dp + dm
    den[den == 0] = 1
    return 10 * dm / den


def score(model: F.FuzzyModel, rows: dict, k=None, weights=None, ks=range(1, 7)):
    """rows: {cid: {показатель: значение}}. → (оценки по убыванию ВСММ, пропущенные [(cid, причина)])."""
    k = int(k or model.k)
    order = model.order
    wv = np.array([float((weights or {}).get(v, 1)) for v in order])
    wv = wv / wv.sum() if wv.sum() > 0 else np.full(len(order), 1 / len(order))
    out, X, skipped = [], [], []
    for cid, raw in rows.items():
        miss = [v for v in order if raw.get(v) is None or (isinstance(raw.get(v), float) and math.isnan(raw[v]))]
        if miss:
            skipped.append((cid, 'нет показателей: ' + ', '.join(model.var_map[v]['name'] for v in miss)))
            continue
        g, mu, res = model.infer(raw)
        act = [x for x in res if x['w'] > 0]
        m = len(act)
        gv = np.array([g[v] for v in order])
        out.append(dict(
            cid=cid, raw={v: float(raw[v]) for v in order}, g=g, mu=mu, m=m,
            wmsm_k=F.q_wmsm(res, k), centroid=F.q_centroid(res)[0], wmsm1=F.q_wmsm(res, 1), wmsm_m=F.q_wmsm(res, m),
            wsum=10 * float((wv * gv).sum()),
            wgeo=10 * float(np.exp((wv * np.log(np.maximum(gv, 1e-3))).sum())),
            by_k={kk: F.q_wmsm(res, kk) for kk in ks},
            rules=[dict(rule=x['rule'], w=x['w'], term=x['term'], q=x['q']) for x in act],
            clipped={x['rule']: x['clipped'] for x in act}))
        X.append([raw[v] for v in order])
    if out:
        sign = np.array([model.var_map[v]['dir'] for v in order])
        for o, t in zip(out, topsis(X, wv, sign)):
            o['topsis'] = float(t)
        front = M.pareto_front(X, sign)
        for i, o in enumerate(out):
            o['pareto'] = i in front
    out.sort(key=lambda o: -o['wmsm_k'])
    for i, o in enumerate(out, 1):
        o['rank'] = i
    return out, skipped


def ranks_by_method(scores):
    """{метод: ранги (1 — лучший) в порядке scores}."""
    return {key: M.ranks_from_scores([o[key] for o in scores]) for key, _ in METHODS}


def sensitivity(model: F.FuzzyModel, rows: dict, k, weights=None, n=300, spread=0.2, seed=0):
    """Варьирование границ L, U в пределах ±spread (Монте-Карло): частоты рангов и τ с исходным ранжированием."""
    base, _ = score(model, rows, k, weights)
    if len(base) < 2:
        return None
    cids = [o['cid'] for o in base]
    rows = {c: rows[c] for c in cids}
    base_r = M.ranks_from_scores([o['wmsm_k'] for o in base])
    rng = np.random.default_rng(seed)
    freq = np.zeros((len(cids), len(cids)))
    taus, same = [], 0
    for _ in range(n):
        bounds = {}
        for v in model.variables:
            L, U = float(v['L']), float(v['U'])
            span = U - L
            L2 = L * (1 + rng.uniform(-spread, spread)) if L != 0 else L + span * rng.uniform(-spread, spread) / 2
            U2 = U * (1 + rng.uniform(-spread, spread)) if U != 0 else U + span * rng.uniform(-spread, spread) / 2
            if U2 <= L2:
                L2, U2 = L, U
            bounds[v['key']] = (L2, U2)
        sc, _ = score(model.with_bounds(bounds), rows, k, weights)
        pos = {o['cid']: o['wmsm_k'] for o in sc}
        r = M.ranks_from_scores([pos[c] for c in cids])
        for i, rr in enumerate(r):
            freq[i, int(round(rr)) - 1] += 1
        taus.append(M.kendall_tau(base_r, r))
        same += int(np.array_equal(np.round(r), np.round(base_r)))
    return dict(cids=cids, base_ranks=base_r, freq=freq / n, tau_mean=float(np.nanmean(taus)),
                tau_min=float(np.nanmin(taus)), same_share=same / n, n=n, spread=spread)


def k_sensitivity(scores):
    """Ранги по ВСММ для k = 1…6."""
    ks = sorted(scores[0]['by_k']) if scores else []
    return {kk: M.ranks_from_scores([o['by_k'][kk] for o in scores]) for kk in ks}


def expert_agreement(scores, scope):
    """Согласованность экспертов (W Кендалла) и соответствие методов их консенсусу (τ)."""
    rk = db.rankings(scope)
    cids = [o['cid'] for o in scores]
    mats = []
    for r in rk:
        order = [c for c in r['ranking'] if c in cids]
        if len(order) == len(cids):
            pos = {c: i + 1 for i, c in enumerate(order)}
            mats.append([pos[c] for c in cids])
    if not mats:
        return None
    R = np.array(mats, float)
    out = dict(n_experts=len(mats), consensus=R.mean(axis=0))
    if len(mats) >= 2:
        w = M.kendall_w(R)
        chi2, p = M.kendall_w_chi2(w, len(mats), len(cids))
        out.update(W=w, chi2=chi2, p=p)
    cons = R.mean(axis=0)
    out['tau'] = {key: M.kendall_tau(-cons, np.array([o[key] for o in scores])) for key, _ in METHODS}
    return out
