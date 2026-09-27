import json
import math

import numpy as np
import pandas as pd
import streamlit as st

from core import bench as B
from core import charts as CH
from core import db
from core import evaluate as E
from core import fuzzy as F
from core import metrics as M
from ui import widgets as W

s = W.settings()
TERM_NAMES = F.OUT_NAMES

profiles = E.list_profiles()
if not profiles:
    st.error('Нет профилей оценки в config/profiles.')
    st.stop()
with st.container(horizontal=True, vertical_alignment='bottom'):
    pid = st.selectbox('Профиль оценки', profiles, format_func=lambda p: E.load_profile(p)['name'], key='ev_prof',
                       width=380)
    prof = E.load_profile(pid)
    fm_ids = F.list_models()
    fid = st.selectbox('Нечёткая модель', fm_ids, index=fm_ids.index(prof.get('fuzzy')) if prof.get('fuzzy') in fm_ids else 0,
                       format_func=lambda f: F.load_model(f).name, key='ev_fuzzy', width=380)
model0 = F.load_model(fid)
st.caption(prof.get('description', ''))


@st.cache_data(max_entries=20, show_spinner='Расчёт показателей…')
def indicators(version, pid_, prof_json, cids, platform_gb):
    p = json.loads(prof_json)
    cands = [db.candidate(c) for c in cids]
    return E.indicator_table(p, [c for c in cands if c], platform_gb)[:2]


# ---------------- профиль: откуда берутся показатели ----------------
with st.expander('Источники показателей (профиль)'):
    benches = {b['id']: b for b in B.list_benchmarks()}
    rows = []
    for ind in model0.order:
        for src in prof['indicators'].get(ind, []) or [{}]:
            rows.append({'Показатель': ind, 'Источник': src.get('bench') or ('модель' if src.get('model') else ''),
                         'Метрика': src.get('metric') or src.get('model') or '', 'Вес': float(src.get('weight', 1))})
    pdf = pd.DataFrame(rows)
    src_opts = [''] + list(benches) + ['модель']
    metric_opts = sorted(set(M.METRICS))
    ped = st.data_editor(pdf, hide_index=True, num_rows='dynamic', key=f'prof_ed_{pid}', column_config={
        'Показатель': st.column_config.SelectboxColumn(options=model0.order, required=True,
                                                      help='переменная нечёткой модели'),
        'Источник': st.column_config.SelectboxColumn(options=src_opts, help='тестовый набор или «модель» (сведения о модели)'),
        'Метрика': st.column_config.SelectboxColumn(options=metric_opts, help='acc, rob, tox, bias, lat, res, …'),
        'Вес': st.column_config.NumberColumn(min_value=0.0, help='вес источника при усреднении')})
    st.caption('Если у показателя несколько источников — берётся взвешенное среднее по доступным. Например, точность = '
               '0,5·MMLU-ru + 0,5·свой набор задач. Показатели моделей из прежнего пульта берутся из импорта, если '
               'по наборам данных нет.')
    use_manual = st.toggle('Использовать импортированные показатели (прежний пульт), если нет данных прогонов',
                           value=prof.get('use_manual', True))
    with st.container(horizontal=True, vertical_alignment='bottom'):
        new_name = st.text_input('Название профиля', value=prof['name'], width=380)
        new_id = st.text_input('Код профиля', value=pid, width=200)
        if st.button('Сохранить профиль', icon=':material/save:'):
            ind = {}
            for r in ped.to_dict('records'):
                if not r.get('Показатель') or not r.get('Источник'):
                    continue
                src = {'weight': float(r.get('Вес') or 1)}
                if r['Источник'] == 'модель':
                    src['model'] = r.get('Метрика') or 'res'
                else:
                    src['bench'], src['metric'] = r['Источник'], r.get('Метрика') or r['Показатель']
                ind.setdefault(r['Показатель'], []).append(src)
            E.save_profile(dict(prof, id=B.slugify(new_id) or pid, name=new_name, fuzzy=fid, indicators=ind,
                                use_manual=use_manual))
            st.toast('Профиль сохранён', icon=':material/check:')
            st.rerun()
prof = dict(prof, use_manual=use_manual)

# ---------------- модели ----------------
allc = [c for c in db.candidates()]
vals_all, srcs_all = indicators(W.data_version(), pid, json.dumps(prof, ensure_ascii=False),
                                tuple(c['id'] for c in allc), float(s['platform_gb']))
complete = [c['id'] for c in allc if all(k in vals_all.get(c['id'], {}) for k in model0.order)]
sel = st.multiselect('Модели', [c['id'] for c in allc], default=complete, format_func=W.cand_name, key='ev_c')
if not sel:
    st.info('Нет моделей с полным набором показателей. Выполните прогон по наборам профиля или выберите модели вручную.')
    st.stop()
names = {c: W.cand_name(c) for c in sel}

# ---------------- показатели ----------------
st.subheader('Показатели моделей')
raw_rows = []
for c in sel:
    r = {'Модель': names[c]}
    for v in model0.variables:
        r[f"{v['name']}, {v['unit']}"] = vals_all.get(c, {}).get(v['key'])
    raw_rows.append(r)
raw_df = pd.DataFrame(raw_rows)
W.table(raw_df, f'raw_{pid}', 'Показатели моделей', digits=2)
with st.expander('Откуда взяты значения'):
    W.static_table(pd.DataFrame([{'Модель': names[c], **{model0.var_map[k]['name']: v for k, v in srcs_all.get(c, {}).items()
                                                        if k in model0.var_map}} for c in sel]))

# ---------------- параметры метода ----------------
with st.expander('Параметры метода: границы нормирования и порядок k', expanded=False):
    bk = f'bounds_{fid}'
    if bk not in st.session_state:
        st.session_state[bk] = pd.DataFrame([{'Показатель': v['name'], 'key': v['key'], 'L': float(v['L']),
                                              'U': float(v['U']), 'Лучше': '↑ больше' if v['dir'] > 0 else '↓ меньше'}
                                             for v in model0.variables])
    with st.container(horizontal=True):
        if st.button('Сбросить к значениям модели', key='b_reset'):
            del st.session_state[bk]
            st.rerun()
        if st.button('Подобрать по диапазону данных', key='b_fit',
                     help='L и U — минимум и максимум среди выбранных моделей с запасом 10 %. Оценка становится '
                          'относительной; для статьи укажите выбранные границы.'):
            b_ = st.session_state[bk].copy()
            for i, r in b_.iterrows():
                col = [vals_all[c][r['key']] for c in sel if r['key'] in vals_all.get(c, {})]
                if len(col) >= 2 and max(col) > min(col):
                    pad = 0.1 * (max(col) - min(col))
                    b_.at[i, 'L'], b_.at[i, 'U'] = round(min(col) - pad, 3), round(max(col) + pad, 3)
            st.session_state[bk] = b_
            st.rerun()
    bdf = st.data_editor(st.session_state[bk], hide_index=True, disabled=['Показатель', 'key', 'Лучше'],
                         column_config={'key': None}, key=f'bed_{fid}')
    bdf = bdf.assign(**{'«Плохо» (g = 0,2)': [r['L'] + 0.2 * (r['U'] - r['L']) if r['Лучше'].startswith('↑')
                                             else r['U'] - 0.2 * (r['U'] - r['L']) for _, r in bdf.iterrows()],
                        '«Хорошо» (g = 0,8)': [r['L'] + 0.8 * (r['U'] - r['L']) if r['Лучше'].startswith('↑')
                                              else r['U'] - 0.8 * (r['U'] - r['L']) for _, r in bdf.iterrows()]})
    st.caption('Пороговые значения: ' + '; '.join(f"{r['Показатель']}: плохо — {W.fmt_num(r['«Плохо» (g = 0,2)'], 1)}, "
                                                    f"хорошо — {W.fmt_num(r['«Хорошо» (g = 0,8)'], 1)}"
                                                    for _, r in bdf.iterrows()))
    k = st.slider('Порядок k ВСММ', 1, 6, int(model0.k), key='ev_k',
                  help='1 — компенсирующее усреднение; ≥ 2 — критические недостатки не компенсируются')
    wts = {v['key']: st.session_state.get(f'w_{v["key"]}', float(prof.get('weights', {}).get(v['key'], 1)))
           for v in model0.variables}
    st.caption('Веса показателей для взвешенной суммы, геометрического среднего и TOPSIS:')
    wc = st.columns(len(model0.variables))
    for i, v in enumerate(model0.variables):
        wts[v['key']] = wc[i].number_input(v['name'], 0.0, 10.0, wts[v['key']], 0.1, key=f'w_{v["key"]}')
k = st.session_state.get('ev_k', int(model0.k))
bad = [r['Показатель'] for _, r in bdf.iterrows() if not r['U'] > r['L']]
if bad:
    st.error('Нужно L < U: ' + ', '.join(bad))
    st.stop()
model = model0.with_bounds({r['key']: (float(r['L']), float(r['U'])) for _, r in bdf.iterrows()})
rows_in = {c: vals_all.get(c, {}) for c in sel}
scores, skipped = E.score(model, rows_in, k, wts)
for c, why in skipped:
    st.warning(f'{names[c]}: {why}')
if not scores:
    st.stop()

# ---------------- обобщённые оценки ----------------
st.subheader(f'Обобщённые оценки (шкала 0–10), k = {k}')
sc_df = pd.DataFrame([{'Ранг': o['rank'], 'Модель': names[o['cid']], 'ВСММ': o['wmsm_k'], 'Центроид': o['centroid'],
                       'Взв. сумма': o['wsum'], 'Взв. геом.': o['wgeo'], 'TOPSIS': o['topsis'],
                       'Сработало правил': o['m'], 'Парето': 'да' if o['pareto'] else ''} for o in scores])
W.table(sc_df, f'scores_{pid}', 'Обобщённые оценки качества', digits=2)
st.caption('Парето — модель не уступает всем остальным одновременно по всем показателям. '
           'Ранги по разным методам совпадают не всегда — см. «Согласованность методов» ниже.')

order = [o['cid'] for o in scores]
tabs = st.tabs(['Сравнение методов', 'Зависимость от k', 'Профили моделей', 'Правила по модели', 'Чувствительность',
                'Экспертные ранжировки', 'Отчёт в Word'], on_change='rerun', key='ev_tabs')
t1, t2, t3, t4, t5, t6, t7 = tabs
methods = [m for m in E.METHODS]

if t1.open is not False:
    with t1:
        W.figure(f'methods_{pid}', 'bars', dict(
            categories=[names[c] for c in order],
            series={name: [next(o[key] for o in scores if o['cid'] == c) for c in order] for key, name in methods},
            ylabel='Оценка', ylim=(0, 10), horizontal=True), 'Обобщённые оценки моделей разными методами')
        rk = E.ranks_by_method(scores)
        tau = pd.DataFrame([[M.kendall_tau(rk[a], rk[b_]) for b_, _ in methods] for a, _ in methods],
                           columns=[n for _, n in methods], index=[n for _, n in methods])
        st.markdown('**Согласованность методов** (τ Кендалла между ранжированиями)')
        W.figure(f'tau_methods_{pid}', 'heatmap', dict(
            matrix=tau.values, row_labels=list(tau.index), col_labels=list(tau.columns), vmin=-1, vmax=1,
            cbar_label='τ Кендалла', digits=2), 'Согласованность ранжирований, полученных разными методами')

if t2.open:
    with t2:
        W.figure(f'k_sens_{pid}', 'lines', dict(
            x=list(range(1, 7)), series={names[o['cid']]: [o['by_k'][kk] for kk in range(1, 7)] for o in scores},
            dashed={names[o['cid']]: o['centroid'] for o in scores}, xlabel='Порядок k', ylabel='Обобщённая оценка',
            ylim=(0, 10), xticks=list(range(1, 7))),
            'Зависимость обобщённой оценки от порядка k (штриховые линии — центроидная дефаззификация)')
        ks = E.k_sensitivity(scores)
        st.caption('Места моделей при разных k:')
        W.static_table(pd.DataFrame({'Модель': [names[c] for c in order],
                                     **{f'k = {kk}': ks[kk] for kk in ks}}), digits=1)

if t3.open:
    with t3:
        W.figure(f'radar_{pid}', 'radar', dict(
            labels=[v['name'] for v in model.variables],
            series={names[o['cid']]: [o['g'][v['key']] for v in model.variables] for o in scores}),
            'Нормированные показатели моделей g (1 — наилучшее значение)')
        gdf = pd.DataFrame([{'Модель': names[o['cid']], **{v['name']: o['g'][v['key']] for v in model.variables}}
                            for o in scores])
        W.table(gdf, f'g_{pid}', 'Нормированные значения показателей g', digits=2)

if t4.open:
    with t4:
        cx = st.selectbox('Модель', order, format_func=names.get, key='ev_rule_c')
        o = next(o for o in scores if o['cid'] == cx)
        W.figure(f'rule_{pid}_{cx}', 'rule', dict(model=model.to_dict(), raw=o['raw'], k=k),
                 f'Усечённые заключения правил и итоговые оценки для модели «{names[cx]}»')
        rdf = pd.DataFrame([{'Правило': r['rule'], 'Формулировка': F.rule_text(
            next(x for x in model.rules if x.name == r['rule']), model.var_map), 'ω': r['w'], 'Терм': r['term'],
            'q': r['q']} for r in o['rules']])
        W.table(rdf, f'rules_{pid}_{cx}', f'Сработавшие правила для модели «{names[cx]}»', digits=2)
        centers = {'ОН': 0, 'Н': 2.5, 'С': 5, 'В': 7.5, 'ОВ': 10}
        near = min(centers, key=lambda t: abs(centers[t] - o['wmsm_k']))
        top3 = sorted(o['rules'], key=lambda r: -r['w'])[:3]
        st.markdown(f"**Объяснение.** ВСММ = {W.fmt_num(o['wmsm_k'])} — ближе всего к терму «{TERM_NAMES[near]}». "
                    "Наибольший вклад: " + ', '.join(f"{r['rule']} (ω = {W.fmt_num(r['w'])}, {TERM_NAMES[r['term']]})"
                                                    for r in top3) + '.')

if t5.open:
    with t5:
        st.caption('Границы нормирования L и U варьируются случайно в пределах ±20 % (методика статьи); '
                   'смотрим, как часто модели сохраняют свои места.')
        with st.container(horizontal=True, vertical_alignment='bottom'):
            spread = st.slider('Разброс границ, %', 5, 50, 20, 5, key='sens_spread', width=300)
            n_mc = st.select_slider('Число испытаний', [100, 200, 500, 1000], value=200, key='sens_n', width=300)
        if len(scores) < 2:
            st.caption('Нужно хотя бы две модели.')
        else:
            sens = E.sensitivity(model, rows_in, k, wts, n=n_mc, spread=spread / 100)
            with st.container(horizontal=True):
                st.metric('Ранжирование не изменилось', f"{W.fmt_num(100 * sens['same_share'], 0)} %", border=True)
                st.metric('Средний τ с исходным', W.fmt_num(sens['tau_mean']), border=True)
                st.metric('Наименьший τ', W.fmt_num(sens['tau_min']), border=True)
            W.figure(f'sens_{pid}', 'heatmap', dict(
                matrix=100 * sens['freq'], row_labels=[names[c] for c in sens['cids']],
                col_labels=[str(i + 1) for i in range(len(sens['cids']))], vmin=0, vmax=100,
                cbar_label='Частота, %', digits=0, xlabel='Место'),
                f'Частота мест моделей при варьировании границ нормирования (±{spread} %)')

if t6.open:
    with t6:
        scope = f'{pid}'
        st.caption('Каждый эксперт упорядочивает модели от лучшей к худшей. Согласованность экспертов — коэффициент '
                   'конкордации W Кендалла; соответствие методов консенсусу — τ Кендалла.')
        with st.form('rank_form'):
            expert = st.text_input('Эксперт', value=s.get('expert') or '')
            rank_df = pd.DataFrame({'Модель': [names[c] for c in order], 'Место': list(range(1, len(order) + 1))})
            red = st.data_editor(rank_df, hide_index=True, disabled=['Модель'], key='rank_ed',
                                 column_config={'Место': st.column_config.NumberColumn(min_value=1, max_value=len(order),
                                                                                       step=1)})
            ok = st.form_submit_button('Сохранить ранжировку', type='primary')
        if ok:
            if not expert.strip():
                st.error('Укажите эксперта.')
            elif sorted(red['Место']) != list(range(1, len(order) + 1)):
                st.error('Места должны быть 1…n без повторов.')
            else:
                by = dict(zip(red['Модель'], red['Место']))
                ranking = sorted(order, key=lambda c: by[names[c]])
                db.save_ranking(expert.strip(), scope, ranking)
                st.rerun()
        rks = db.rankings(scope)
        if rks:
            W.static_table(pd.DataFrame([{'Эксперт': r['expert'], **{names.get(c, c): i + 1 for i, c in enumerate(r['ranking'])
                                                                     if c in names}} for r in rks]))
            with st.popover('Удалить ранжировку'):
                ex = st.selectbox('Эксперт', [r['expert'] for r in rks])
                if st.button('Удалить', key='del_rank'):
                    db.delete_ranking(ex, scope)
                    st.rerun()
            ag = E.expert_agreement(scores, scope)
            if ag:
                with st.container(horizontal=True):
                    st.metric('Экспертов', ag['n_experts'], border=True)
                    if 'W' in ag:
                        st.metric('W Кендалла', W.fmt_num(ag['W']), border=True)
                        st.metric('χ²', W.fmt_num(ag['chi2']), border=True)
                        st.metric('p', W.fmt_num(ag['p'], 3), border=True,
                                  help='p < 0,05 — согласованность экспертов неслучайна')
                tdf = pd.DataFrame([{'Метод': name, 'τ с консенсусом экспертов': ag['tau'][key]} for key, name in methods])
                W.table(tdf, f'tau_experts_{pid}', 'Соответствие методов консенсусу экспертов', digits=2)
            else:
                st.caption('Ранжировки не охватывают все выбранные модели.')

if t7.open:
    with t7:
        st.caption('Отчёт с таблицами и рисунками в оформлении для статьи (Times New Roman 12 пт). '
                   'Нумерация берётся из подписей, заданных у рисунков и таблиц.')
        if st.button('Сформировать отчёт', type='primary', icon=':material/description:'):
            style = W.chart_style()

            def cap(key, default, kind='fig'):
                c_ = W.caption_for(key, default, kind)
                return (f"Рис. {c_['number']}. {c_['caption']}" if kind == 'fig' else
                        W.table_caption(c_['number'], c_['caption']))

            parts = [('h', f"Сводная оценка: {prof['name']}"),
                     ('p', f"Нечёткая модель: {model.name}; порядок ВСММ k = {k}; память стенда {s['platform_gb']} ГБ."),
                     ('table', raw_df, cap(f't_raw_{pid}', 'Показатели моделей', 'tab'), 2),
                     ('table', bdf.drop(columns=['key']), 'Таблица – Параметры нормирования показателей', 2),
                     ('table', sc_df, cap(f't_scores_{pid}', 'Обобщённые оценки качества', 'tab'), 2)]
            for o in scores:
                rdf = pd.DataFrame([{'Правило': r['rule'], 'ω': r['w'], 'Терм': r['term'], 'q': r['q']} for r in o['rules']])
                parts.append(('table', rdf, f"Таблица – Сработавшие правила: {names[o['cid']]}", 2))
            figs = [('bars', dict(categories=[names[c] for c in order],
                                  series={name: [next(o[key] for o in scores if o['cid'] == c) for c in order]
                                          for key, name in methods}, ylabel='Оценка', ylim=(0, 10), horizontal=True),
                     cap(f'methods_{pid}', 'Обобщённые оценки моделей разными методами')),
                    ('lines', dict(x=list(range(1, 7)), series={names[o['cid']]: [o['by_k'][kk] for kk in range(1, 7)]
                                                                for o in scores},
                                   dashed={names[o['cid']]: o['centroid'] for o in scores}, xlabel='Порядок k',
                                   ylabel='Обобщённая оценка', ylim=(0, 10), xticks=list(range(1, 7))),
                     cap(f'k_sens_{pid}', 'Зависимость обобщённой оценки от порядка k (штриховые линии — '
                                          'центроидная дефаззификация)')),
                    ('radar', dict(labels=[v['name'] for v in model.variables],
                                   series={names[o['cid']]: [o['g'][v['key']] for v in model.variables] for o in scores}),
                     cap(f'radar_{pid}', 'Нормированные показатели моделей g (1 — наилучшее значение)'))]
            with st.spinner('Формирование отчёта…'):
                for kind, kw, capt in figs:
                    parts.append(('fig', W.figure_png(kind, kw, style), style.width_cm, capt))
                data = CH.report_docx(parts)
            st.download_button('Скачать отчёт (.docx)', data, f'сводная_оценка_{pid}.docx', on_click='ignore',
                               icon=':material/download:', type='primary',
                               mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

with st.expander('Самопроверка: пример из статьи (модели A, B, C, табл. 5)'):
    ex = {'A': dict(acc=85, lat=5, tox=2, rob=80, bias=5, res=80), 'B': dict(acc=78, lat=1, tox=1, rob=70, bias=10, res=40),
          'C': dict(acc=90, lat=1, tox=30, rob=85, bias=10, res=40)}
    try:
        art = F.load_model('article_v5')
        sc_ex, _ = E.score(art, ex, 2)
        W.static_table(pd.DataFrame([{'Модель': o['cid'], 'Центроид': o['centroid'], 'ВСММ, k = 1': o['wmsm1'],
                                      'ВСММ, k = 2': o['by_k'][2], 'ВСММ, k = m': o['wmsm_m'],
                                      'm': o['m']} for o in sorted(sc_ex, key=lambda o: o['cid'])]))
        st.caption('Ожидается: A — 6,25 / 5,83 / 5,75 / 5,72; B — 7,17 / 7,39 / 7,48 / 7,29; C — 5,28 / 4,17 / 2,50 / 2,50.')
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
