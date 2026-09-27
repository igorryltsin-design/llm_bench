import json
import math
import random
import statistics

import numpy as np
import pandas as pd
import streamlit as st

from core import bench as B
from core import db
from core import metrics as M
from ui import widgets as W

s = W.settings()


@st.cache_data(max_entries=30, show_spinner='Расчёт показателей…')
def compute(version, cids, bids, splits):
    out = {}
    for bid in bids:
        if not B.exists(bid):
            continue
        b = B.load(bid)
        items = b.select(splits)
        for cid in cids:
            resps = db.responses(cid, bid)
            if resps:
                out[(cid, bid)] = M.aggregate(b, cid, splits, items, resps)
    return out


pairs = db.rows('SELECT DISTINCT candidate_id, bench_id FROM responses')
if not pairs:
    st.info('Ответов пока нет — выполните прогон на странице «Прогон». Итоговые показатели прежнего пульта '
            'доступны на странице «Сводная оценка».')
    st.stop()
cmap = {c['id']: c for c in db.candidates(True)}
all_c = sorted({p['candidate_id'] for p in pairs if p['candidate_id'] in cmap}, key=lambda c: cmap[c]['created'])
all_b = sorted({p['bench_id'] for p in pairs if B.exists(p['bench_id'])})
bmeta = {b: B.load(b).meta for b in all_b}
bname = {b: bmeta[b]['name'] for b in all_b}
short = {b: bmeta[b]['name'].split(':')[0] for b in all_b}

with st.container(horizontal=True, vertical_alignment='bottom'):
    sel_c = st.multiselect('Модели', all_c, default=[c for c in all_c if not cmap[c].get('hidden')],
                           format_func=W.cand_name, key='res_c')
    sel_b = st.multiselect('Наборы', all_b, default=all_b, format_func=bname.get, key='res_b')
    splits = st.pills('Выборки', list(B.SPLITS), default=['eval'], selection_mode='multi', format_func=B.SPLIT_RU.get,
                      key='res_splits')
if not sel_c or not sel_b or not splits:
    st.stop()
agg = compute(W.data_version(), tuple(sel_c), tuple(sel_b), tuple(splits))
names = {c: W.cand_name(c) for c in sel_c}
mm = {c: M.model_metrics(cmap[c], float(s['platform_gb'])) for c in sel_c}

tabs = st.tabs(['Сводная таблица', 'По категориям', 'Графики', 'Сравнение двух моделей', 'Ответы',
                'Проверка экспертом', 'Судья и эксперт'], on_change='rerun', key='res_tabs')
t_sum, t_cat, t_fig, t_cmp, t_ans, t_exp, t_agree = tabs

# ---------------- сводная таблица ----------------
if t_sum.open is not False:
    with t_sum:
        with st.container(horizontal=True, vertical_alignment='bottom'):
            scope = st.segmented_control('Показатели', ['основные', 'все'], default='основные', required=True,
                                         key='sum_scope')
            show_ci = st.toggle('С 95 % доверительными интервалами', value=False, key='sum_ci')
        rows = []
        for c in sel_c:
            r = {'Модель': names[c]}
            for bid in sel_b:
                a = agg.get((c, bid))
                t = bmeta[bid]['type']
                keys = M.TYPE_METRICS[t][:3] if scope == 'основные' else M.TYPE_METRICS[t]
                for k in keys:
                    col = f'{short[bid]}: {M.mlabel(k)}'
                    v = a['values'].get(k) if a else None
                    if show_ci and a and k in a['ci']:
                        lo, hi = a['ci'][k]
                        r[col] = f'{W.fmt_num(v, 1)} [{W.fmt_num(lo, 1)}; {W.fmt_num(hi, 1)}]'
                    else:
                        r[col] = v
                if scope == 'все':
                    r[f'{short[bid]}: заданий'] = a['n'] if a else 0
            for k in M.MODEL_METRICS:
                r[M.mlabel(k)] = mm[c].get(k)
            rows.append(r)
        df = pd.DataFrame(rows).dropna(axis=1, how='all')
        W.table(df, 'summary', 'Показатели качества моделей на тестовых наборах', digits=2,
                column_config={c: st.column_config.NumberColumn(format='%.2f') for c in df.columns
                               if df[c].dtype.kind == 'f'})
        with st.expander('Пояснения к показателям'):
            used = {k for bid in sel_b for k in M.TYPE_METRICS[bmeta[bid]['type']]} | set(M.MODEL_METRICS)
            W.static_table(pd.DataFrame([{'Показатель': v[0], 'Единица': v[1],
                                          'Лучше': {1: '↑ больше', -1: '↓ меньше', 0: 'справочно'}[v[2]], 'Смысл': v[3]}
                                         for k, v in M.METRICS.items() if k in used]))
        cov = [f'{names[c]} — {short[b]}: {agg[(c, b)]["n"]} из {agg[(c, b)]["n_items"]}' for (c, b) in agg
               if agg[(c, b)]['n'] < agg[(c, b)]['n_items'] and bmeta[b]['type'] != 'perf']
        if cov:
            st.caption('Пройдены не все задания (пробный прогон или новые задания): ' + '; '.join(cov[:12]))

# ---------------- по категориям ----------------
if t_cat.open:
    with t_cat:
        cat_b = [b for b in sel_b if bmeta[b]['type'] in ('mc', 'short', 'structured', 'open', 'bias_pair', 'toxicity')]
        if not cat_b:
            st.caption('Нет подходящих наборов.')
        else:
            bid = st.selectbox('Набор', cat_b, format_func=bname.get, key='cat_bench')
            t = bmeta[bid]['type']
            key = {'bias_pair': 'bias', 'toxicity': 'tox'}.get(t, 'acc')
            cats = sorted({k for c in sel_c for k in (agg.get((c, bid)) or {}).get('by_cat', {})})
            if not cats:
                st.caption('Нет данных по категориям.')
            else:
                max_cats = st.slider('Число категорий на рисунке', 3, min(60, len(cats)), min(20, len(cats)),
                                     key='cat_n') if len(cats) > 3 else len(cats)
                # категории с наибольшим разбросом между моделями — самые информативные
                def spread(k):
                    vs = [(agg.get((c, bid)) or {}).get('by_cat', {}).get(k, {}).get(key) for c in sel_c]
                    vs = [v for v in vs if v is not None]
                    return (max(vs) - min(vs)) if len(vs) > 1 else 0
                top = sorted(cats, key=lambda k: (-spread(k), k))[:max_cats]
                mat = [[(agg.get((c, bid)) or {}).get('by_cat', {}).get(k, {}).get(key, np.nan) for k in top]
                       for c in sel_c]
                W.figure(f'cat_heat_{bid}_{key}', 'heatmap', dict(
                    matrix=mat, row_labels=[names[c] for c in sel_c], col_labels=top, vmin=0, vmax=100,
                    cbar_label=M.mlabel(key), digits=0), f'{M.mname(key)} по категориям набора «{bname[bid]}»')
                tab = pd.DataFrame([{'Категория': k, **{names[c]: (agg.get((c, bid)) or {}).get('by_cat', {}).get(k, {})
                                                          .get(key) for c in sel_c},
                                     'Заданий': max([(agg.get((c, bid)) or {}).get('by_cat', {}).get(k, {}).get('n', 0)
                                                     for c in sel_c])} for k in cats])
                W.table(tab, f'cat_{bid}', f'{M.mname(key)} по категориям набора «{bname[bid]}», %', digits=1)

# ---------------- графики ----------------
if t_fig.open:
    with t_fig:
        acc_b = [b for b in sel_b if any('acc' in (agg.get((c, b)) or {}).get('values', {}) for c in sel_c)]
        if acc_b:
            st.markdown('**Точность по наборам**')
            W.figure('acc_by_bench', 'bars', dict(
                categories=[short[b] for b in acc_b],
                series={names[c]: [(agg.get((c, b)) or {}).get('values', {}).get('acc') for b in acc_b] for c in sel_c},
                errors={names[c]: [(agg.get((c, b)) or {}).get('ci', {}).get('acc') for b in acc_b] for c in sel_c},
                ylabel='Точность, %', ylim=(0, 100)),
                'Точность моделей по тестовым наборам (отрезки — 95 % доверительные интервалы)')

        st.markdown('**Любой показатель**')
        with st.container(horizontal=True):
            mb = st.selectbox('Набор', sel_b, format_func=bname.get, key='any_b', width=380)
            opts = [k for k in M.TYPE_METRICS[bmeta[mb]['type']] if any(k in (agg.get((c, mb)) or {}).get('values', {})
                                                                        for c in sel_c)]
            mk = st.selectbox('Показатель', opts or ['—'], format_func=lambda k: M.mlabel(k) if k != '—' else k,
                              key='any_m', width=380)
        if opts and mk in opts:
            vals = [(agg.get((c, mb)) or {}).get('values', {}).get(mk) for c in sel_c]
            W.figure(f'any_{mb}_{mk}', 'bars', dict(
                categories=[names[c] for c in sel_c], series={M.mname(mk): vals},
                errors={M.mname(mk): [(agg.get((c, mb)) or {}).get('ci', {}).get(mk) for c in sel_c]},
                ylabel=M.mlabel(mk), horizontal=True, value_labels=True, digits=1),
                f'{M.mname(mk)} моделей на наборе «{bname[mb]}»')

        mc_b = [b for b in sel_b if bmeta[b]['type'] == 'mc'
                and any('rob' in (agg.get((c, b)) or {}).get('values', {}) for c in sel_c)]
        if mc_b:
            st.markdown('**Устойчивость по видам возмущения запроса**')
            rb = st.selectbox('Набор', mc_b, format_func=bname.get, key='rob_b') if len(mc_b) > 1 else mc_b[0]
            kinds = [('rob_typo', 'Опечатки'), ('rob_shuffle', 'Перестановка вариантов'),
                     ('rob_instr', 'Переформулировка'), ('rob', 'Все возмущения')]
            W.figure(f'rob_{rb}', 'bars', dict(
                categories=[k[1] for k in kinds],
                series={names[c]: [(agg.get((c, rb)) or {}).get('values', {}).get(k[0]) for k in kinds] for c in sel_c},
                ylabel='Доля сохранившихся ответов, %', ylim=(0, 100)),
                f'Устойчивость ответов к возмущениям запроса (набор «{bname[rb]}»)')

        perf_b = [b for b in sel_b if bmeta[b]['type'] == 'perf']
        if perf_b:
            pb = perf_b[0]
            lat = {names[c]: (agg.get((c, pb)) or {}).get('latencies') for c in sel_c}
            lat = {k: v for k, v in lat.items() if v}
            if lat:
                st.markdown('**Время отклика**')
                W.figure(f'lat_box_{pb}', 'boxes', dict(data=lat, ylabel='Время ответа, с'),
                         'Распределение времени ответа на типовой запрос')
            if acc_b:
                ab = st.selectbox('Точность по набору', acc_b, format_func=bname.get, key='sc_acc')
                xk = st.segmented_control('Ось X', ['lat', 'tok_s'], format_func=M.mlabel, default='lat',
                                          required=True, key='sc_x')
                pts, X = [], []
                for c in sel_c:
                    y = (agg.get((c, ab)) or {}).get('values', {}).get('acc')
                    x = (agg.get((c, pb)) or {}).get('values', {}).get(xk)
                    if x is not None and y is not None:
                        pts.append((x, y, names[c], mm[c].get('weights_gb')))
                        X.append([x, y])
                if len(pts) >= 1:
                    front = M.pareto_front(X, [M.mdir(xk), +1]) if len(X) > 1 else [0]
                    W.figure(f'scatter_{ab}_{xk}', 'scatter', dict(
                        points=pts, xlabel=M.mlabel(xk), ylabel=f'Точность ({short[ab]}), %', highlight=front),
                        f'Точность и {"время отклика" if xk == "lat" else "скорость генерации"} моделей '
                        '(размер маркера — объём весов; обведены и соединены парето-оптимальные модели)')

        bias_b = [b for b in sel_b if bmeta[b]['type'] in ('bias_pair', 'toxicity')]
        for bb in bias_b:
            key = 'bias' if bmeta[bb]['type'] == 'bias_pair' else 'tox'
            cats = sorted({k for c in sel_c for k in (agg.get((c, bb)) or {}).get('by_cat', {})})
            if cats:
                st.markdown(f'**{M.mname(key)} по социальным группам**')
                W.figure(f'grp_{bb}', 'bars', dict(
                    categories=cats,
                    series={names[c]: [(agg.get((c, bb)) or {}).get('by_cat', {}).get(k, {}).get(key) for k in cats]
                            for c in sel_c}, ylabel=M.mlabel(key)),
                    f'{M.mname(key)} моделей по группам (набор «{bname[bb]}»)')

        if mc_b:
            lb = mc_b[0]
            L = sorted({x for c in sel_c for x in ((agg.get((c, lb)) or {}).get('letters', {}).get('chosen') or {})} |
                       {x for c in sel_c for x in ((agg.get((c, lb)) or {}).get('letters', {}).get('truth') or {})})
            if L:
                st.markdown('**Предпочтение позиции ответа**')
                ser = {}
                for c in sel_c:
                    ch = (agg.get((c, lb)) or {}).get('letters', {}).get('chosen') or {}
                    tot = sum(ch.values()) or 1
                    ser[names[c]] = [100 * ch.get(x, 0) / tot for x in L]
                tr = next(((agg.get((c, lb)) or {}).get('letters', {}).get('truth') for c in sel_c
                           if (agg.get((c, lb)) or {}).get('letters')), {}) or {}
                ser['Правильные ответы'] = [100 * tr.get(x, 0) / (sum(tr.values()) or 1) for x in L]
                W.figure(f'letters_{lb}', 'bars', dict(categories=L, series=ser, ylabel='Доля ответов, %'),
                         f'Распределение выбранных вариантов ответа (набор «{bname[lb]}»)')

# ---------------- сравнение двух моделей ----------------
if t_cmp.open:
    with t_cmp:
        st.caption('Парное сравнение по одним и тем же заданиям: для верно/неверно — точный критерий Макнемара, '
                   'для баллов — парный бутстреп. Различие значимо при p < 0,05.')
        tuned = [c for c in sel_c if cmap[c].get('base_id') in sel_c]
        default_b = tuned[0] if tuned else (sel_c[1] if len(sel_c) > 1 else sel_c[0])
        default_a = cmap[default_b].get('base_id') if tuned else sel_c[0]
        with st.container(horizontal=True):
            ca = st.selectbox('Модель А (исходная)', sel_c, index=sel_c.index(default_a) if default_a in sel_c else 0,
                              format_func=names.get, key='cmp_a', width=380)
            cb = st.selectbox('Модель Б (дообученная)', sel_c, index=sel_c.index(default_b), format_func=names.get,
                              key='cmp_b', width=380)
        if ca == cb:
            st.caption('Выберите две разные модели.')
        else:
            comp = []
            detail = {}
            for bid in sel_b:
                if bmeta[bid]['type'] not in ('mc', 'short', 'structured', 'open'):
                    continue
                pr = M.paired(B.load(bid), ca, cb, tuple(splits))
                detail[bid] = pr
                if pr['n']:
                    comp.append({'Набор': bname[bid], 'Заданий': pr['n'], 'А, %': pr['mean_a'], 'Б, %': pr['mean_b'],
                                 'Разность, п.п.': pr['mean_b'] - pr['mean_a'], 'Стало лучше': pr['better'],
                                 'Стало хуже': pr['worse'], 'p': pr['p'], 'Критерий': pr['test'],
                                 'Вывод': ('Б лучше' if pr['mean_b'] > pr['mean_a'] else 'А лучше')
                                 if pr['p'] < 0.05 else 'различие незначимо'})
            if not comp:
                st.caption('Нет общих заданий с ответами обеих моделей.')
            else:
                cdf = pd.DataFrame(comp)
                W.table(cdf, f'cmp_{ca}_{cb}', f'Сравнение моделей «{names[ca]}» (А) и «{names[cb]}» (Б)',
                        digits={'*': 1, 'p': 3})
                W.figure(f'cmp_bars_{ca}_{cb}', 'bars', dict(
                    categories=[short[b] for b in detail if detail[b]['n']],
                    series={names[ca]: [detail[b]['mean_a'] for b in detail if detail[b]['n']],
                            names[cb]: [detail[b]['mean_b'] for b in detail if detail[b]['n']]},
                    ylabel='Точность, %', ylim=(0, 100), value_labels=True, digits=1),
                    f'Точность исходной и дообученной моделей по тестовым наборам')
                bsel = st.selectbox('Подробно по набору', [b for b in detail if detail[b]['n']], format_func=bname.get,
                                    key='cmp_detail')
                rows = pd.DataFrame(detail[bsel]['rows'])
                by = rows.groupby('category').agg(a=('a', 'mean'), b=('b', 'mean'), n=('a', 'size')).reset_index()
                if len(by) > 1:
                    by['d'] = 100 * (by['b'] - by['a'])
                    by = by.sort_values('d')
                    W.figure(f'cmp_cat_{bsel}_{ca}_{cb}', 'bars', dict(
                        categories=by['category'].tolist(), series={'Б − А': by['d'].tolist()},
                        ylabel='Изменение точности, п.п.', horizontal=True, value_labels=True, digits=0),
                        f'Изменение точности после дообучения по категориям набора «{bname[bsel]}»')
                changed = rows[rows['diff'] != 0].sort_values('diff')
                st.markdown(f'**Задания с изменившимся результатом: {len(changed)}**')
                if len(changed):
                    W.static_table(changed.rename(columns={'item_id': 'Задание', 'category': 'Категория', 'a': 'А',
                                                           'b': 'Б', 'diff': 'Б − А'}))

# ---------------- ответы ----------------
if t_ans.open:
    with t_ans:
        with st.container(horizontal=True):
            ac = st.selectbox('Модель', sel_c, format_func=names.get, key='ans_c', width=360)
            ab = st.selectbox('Набор', sel_b, format_func=bname.get, key='ans_b', width=360)
            flt = st.segmented_control('Показать', ['все', 'неверные', 'неразобранные', 'ошибки'], default='все',
                                       required=True, key='ans_f')
        b = B.load(ab)
        items = {str(i['id']): i for i in b.select(tuple(splits))}
        cur = M.current(b, list(items.values()), db.responses(ac, ab))
        rs = [r for v in cur.values() for r in v]
        if flt == 'неверные':
            rs = [r for r in rs if r['score'] is not None and r['score'] < 1 and b.type in ('mc', 'short', 'structured')
                  or (b.type == 'toxicity' and (r['score'] or 0) > 0.5)]
        elif flt == 'неразобранные':
            rs = [r for r in rs if r['parsed'] is None and b.type in ('mc', 'bias_pair', 'short', 'structured')]
        elif flt == 'ошибки':
            rs = [r for r in rs if r.get('error')]

        def expected(it):
            t = b.type
            if t == 'mc':
                return it['answer']
            if t == 'short':
                return ' | '.join(it.get('answers') or [])
            if t == 'structured':
                return json.dumps(it.get('expected'), ensure_ascii=False)
            if t == 'open':
                return (it.get('reference') or '')[:200]
            if t == 'bias_pair':
                return 'стереотип: ' + it.get('stereo', '')[:80]
            return ''

        adf = pd.DataFrame([{'Задание': r['item_id'], 'Вариант': r['variant'], 'Повтор': r['rep'],
                             'Ожидалось': expected(items[r['item_id']]) if r['item_id'] in items else '',
                             'Разобрано': r['parsed'], 'Балл': r['score'], 'Ответ': (r['output'] or '')[:300],
                             'Время, с': r['latency'], 'Ошибка': r.get('error') or ''} for r in rs])
        st.caption(f'Ответов: {len(adf)}')
        if len(adf):
            W.static_table(adf.assign(**{'Ответ': adf['Ответ'].str.slice(0, 160)}), digits=2, max_rows_visible=15)
            pick = st.selectbox('Подробно', range(len(rs)), key='ans_pick',
                                format_func=lambda i: f"{rs[i]['item_id']} · {rs[i]['variant']} · повтор {rs[i]['rep']}")
            if pick is not None:
                r = rs[pick]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('**Запрос**')
                    st.code('\n\n'.join(m['content'] for m in json.loads(r['prompt'])), language=None, wrap_lines=True)
                with c2:
                    st.markdown('**Ответ модели**')
                    st.code(r['output'] or '(пусто)', language=None, wrap_lines=True)
                    if r.get('reasoning'):
                        with st.expander('Рассуждения'):
                            st.text(r['reasoning'])
                    st.json(r.get('detail') or {}, expanded=False)

# ---------------- проверка экспертом ----------------
if t_exp.open:
    with t_exp:
        open_b = [b for b in sel_b if bmeta[b]['type'] == 'open']
        if not open_b:
            st.caption('Нет наборов с развёрнутыми ответами среди выбранных.')
        else:
            with st.container(horizontal=True, vertical_alignment='bottom'):
                eb = st.selectbox('Набор', open_b, format_func=bname.get, key='exp_b', width=360)
                expert = st.text_input('Эксперт', value=s.get('expert') or '', key='exp_name', width=220,
                                       placeholder='фамилия или код')
                blind = st.toggle('Слепая проверка', value=True, help='Не показывать, какая модель ответила')
            if not expert.strip():
                st.info('Укажите имя эксперта — оценки разных экспертов хранятся раздельно.')
            else:
                if expert != s.get('expert'):
                    W.update_settings(expert=expert)
                b = B.load(eb)
                items = {str(i['id']): i for i in b.select(tuple(splits))}
                grader = f'expert:{expert.strip()}'
                queue = []
                for c in sel_c:
                    cur = M.current(b, list(items.values()), db.responses(c, eb))
                    for v in cur.values():
                        queue += [r for r in v if not r.get('error')]
                done = {g['response_id'] for g in db.grades([r['id'] for r in queue]) if g['grader'] == grader}
                todo = [r for r in queue if r['id'] not in done]
                random.Random(expert).shuffle(todo)
                st.progress(len(done) / len(queue) if queue else 0, text=f'Проверено {len(done)} из {len(queue)}')
                if not todo:
                    st.success('Все ответы проверены.')
                else:
                    r = todo[0]
                    it = items[r['item_id']]
                    st.markdown(f"**Задание {r['item_id']}**" + ('' if blind else f" · модель: {names.get(r['candidate_id'])}"))
                    st.markdown(it['prompt'])
                    if it.get('reference'):
                        with st.expander('Эталонный ответ'):
                            st.markdown(it['reference'])
                    st.markdown('**Ответ модели**')
                    with st.container(border=True):
                        st.markdown(r['output'] or '_(пусто)_')
                    rub = it.get('rubric') or [{'name': 'Качество ответа', 'max': 10}]
                    with st.form(f"grade_{r['id']}"):
                        pts = {}
                        for k_, cr in enumerate(rub):
                            pts[cr['name']] = st.slider(cr['name'], 0.0, float(cr.get('max', 1)), 0.0,
                                                        0.5 if float(cr.get('max', 1)) >= 2 else 0.25, key=f"g_{r['id']}_{k_}")
                        comment = st.text_input('Комментарий')
                        ok = st.form_submit_button('Сохранить и далее', type='primary')
                    if ok:
                        mx = sum(float(cr.get('max', 1)) for cr in rub)
                        db.save_grade(r['id'], grader, sum(pts.values()) / mx if mx else 0, pts, comment)
                        st.rerun()

# ---------------- судья и эксперт ----------------
if t_agree.open:
    with t_agree:
        g = pd.DataFrame(db.grades())
        if g.empty:
            st.caption('Оценок пока нет.')
        else:
            g['kind'] = np.where(g['grader'].str.startswith('expert'), 'expert', 'judge')
            piv = g.groupby(['response_id', 'kind'])['score'].mean().unstack()
            if not {'expert', 'judge'} <= set(piv.columns):
                st.caption('Нужны ответы, оценённые и экспертом, и судьёй.')
            else:
                both = piv.dropna()
                if len(both) < 3:
                    st.caption(f'Ответов с обеими оценками: {len(both)} — нужно хотя бы 3.')
                else:
                    e, j = both['expert'].values, both['judge'].values
                    pear = float(np.corrcoef(e, j)[0, 1]) if np.std(e) and np.std(j) else float('nan')
                    spear = float(np.corrcoef(M.ranks_from_scores(e), M.ranks_from_scores(j))[0, 1]) \
                        if np.std(e) and np.std(j) else float('nan')
                    pe, pj = e >= 0.5, j >= 0.5
                    po = float((pe == pj).mean())
                    pexp = pe.mean() * pj.mean() + (1 - pe.mean()) * (1 - pj.mean())
                    kappa = (po - pexp) / (1 - pexp) if pexp < 1 else float('nan')
                    with st.container(horizontal=True):
                        st.metric('Ответов', len(both), border=True)
                        st.metric('Корреляция Пирсона', W.fmt_num(pear), border=True)
                        st.metric('Корреляция Спирмена', W.fmt_num(spear), border=True)
                        st.metric('Каппа Коэна (зачёт ≥ 50 %)', W.fmt_num(kappa), border=True)
                        st.metric('Средн. абс. расхождение, п.п.', W.fmt_num(100 * float(np.abs(e - j).mean()), 1),
                                  border=True)
                    st.caption('Каппа > 0,6 — существенное согласие; < 0,4 — судье доверять нельзя.')
                    W.figure('judge_vs_expert', 'scatter', dict(
                        points=[(float(a) * 100, float(b_) * 100, '', None) for a, b_ in zip(e, j)],
                        xlabel='Оценка эксперта, %', ylabel='Оценка модели-судьи, %', xlim=(-3, 103), ylim=(-3, 103)),
                        'Согласованность оценок модели-судьи и эксперта')
