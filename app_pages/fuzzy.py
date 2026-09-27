import copy

import pandas as pd
import streamlit as st

from core import bench as B
from core import config as C
from core import fuzzy as F
from core import metrics as M
from ui import widgets as W

st.caption('Конструктор нечёткой модели сводной оценки: показатели (входные переменные) и их нормирование, термы, '
           'база правил «ЕСЛИ … ТО …», проверка полноты и пробный расчёт. Изменения копятся в черновике и '
           'записываются кнопкой «Сохранить модель».')

ids = F.list_models()
drafts = st.session_state.setdefault('fz_drafts', {})


def file_dict(mid):
    return C.read_json(C.FUZZY_DIR / f'{mid}.json')


def bump(mid):
    st.session_state[f'fz_ver_{mid}'] = st.session_state.get(f'fz_ver_{mid}', 0) + 1


@st.dialog('Новая нечёткая модель')
def new_model_dialog(current):
    name = st.text_input('Название', placeholder='например: Оценка для задач планирования')
    code = st.text_input('Код (латиница)', value=B.slugify(name) if name else '')
    base = st.radio('Основа', ['копия текущей модели', 'показатели статьи без правил', 'пустая модель'])
    if st.button('Создать', type='primary', disabled=not name.strip()):
        mid = B.slugify(code or name)
        if mid in ids:
            st.error('Такой код уже есть.')
            return
        if base == 'копия текущей модели':
            d = copy.deepcopy(current)
        else:
            d = copy.deepcopy(F.ARTICLE_V5)
            d['rules'] = []
            if base == 'пустая модель':
                d['variables'] = []
        d.update(id=mid, name=name.strip(), description='')
        C.write_json(C.FUZZY_DIR / f'{mid}.json', d)
        drafts.pop(mid, None)
        st.session_state['fz_sel'] = mid
        st.rerun()


with st.container(horizontal=True, vertical_alignment='bottom'):
    fid = st.selectbox('Модель', ids, format_func=lambda f: (file_dict(f) or {}).get('name', f), key='fz_sel', width=420)
    if fid not in drafts:
        drafts[fid] = copy.deepcopy(file_dict(fid))
    d = drafts[fid]
    if st.button('Новая модель', icon=':material/add:'):
        new_model_dialog(d)
    if fid == 'article_v5' and st.button('Вернуть модель статьи', icon=':material/restart_alt:',
                                         help='Восстановить переменные, термы и 12 правил из статьи'):
        F.restore_article()
        drafts[fid] = copy.deepcopy(F.ARTICLE_V5)
        bump(fid)
        st.rerun()
    if len(ids) > 1:
        with st.popover('Удалить', icon=':material/delete:'):
            st.warning('Файл модели будет удалён.')
            if st.button('Удалить модель', type='primary'):
                F.delete_model(fid)
                drafts.pop(fid, None)
                st.session_state.pop('fz_sel', None)
                st.rerun()

ver = st.session_state.get(f'fz_ver_{fid}', 0)
with st.container(horizontal=True, vertical_alignment='bottom'):
    d['name'] = st.text_input('Название модели', value=d.get('name', fid), key=f'fz_name_{fid}_{ver}', width=420)
    d['k'] = st.number_input('Порядок k ВСММ по умолчанию', 1, 12, int(d.get('k', 2)), key=f'fz_k_{fid}_{ver}', width=240)
d['description'] = st.text_area('Описание', value=d.get('description', ''), key=f'fz_desc_{fid}_{ver}', height=68)

t_vars, t_terms, t_rules, t_check = st.tabs(['Показатели', 'Термы', 'Правила', 'Проверка и пробный расчёт'])

# ---------------- показатели ----------------
with t_vars:
    vdf = pd.DataFrame([{'Код': v['key'], 'Название': v['name'], 'Единица': v.get('unit', ''), 'L': float(v['L']),
                         'U': float(v['U']), 'Лучше': '↑ больше' if v['dir'] > 0 else '↓ меньше',
                         'Н (плохо)': v['labels'].get('Н', ''), 'С': v['labels'].get('С', ''),
                         'В (хорошо)': v['labels'].get('В', '')} for v in d['variables']],
                       columns=['Код', 'Название', 'Единица', 'L', 'U', 'Лучше', 'Н (плохо)', 'С', 'В (хорошо)'])
    ved = st.data_editor(vdf, hide_index=True, num_rows='dynamic', key=f'fz_vars_{fid}_{ver}', column_config={
        'Лучше': st.column_config.SelectboxColumn(options=['↑ больше', '↓ меньше'], required=True),
        'Код': st.column_config.TextColumn(help='латинский код; совпадает с показателем профиля оценки (acc, lat, …)'),
        'L': st.column_config.NumberColumn(help='нижняя граница нормирования'),
        'U': st.column_config.NumberColumn(help='верхняя граница нормирования')})
    d['variables'] = [dict(key=str(r['Код']).strip(), name=r['Название'] or str(r['Код']), unit=r['Единица'] or '',
                           L=float(r['L'] if r['L'] == r['L'] and r['L'] is not None else 0),
                           U=float(r['U'] if r['U'] == r['U'] and r['U'] is not None else 1),
                           dir=1 if str(r['Лучше']).startswith('↑') else -1,
                           labels={'Н': r['Н (плохо)'] or 'плохо', 'С': r['С'] or 'средне', 'В': r['В (хорошо)'] or 'хорошо'})
                      for r in ved.to_dict('records') if r.get('Код') and str(r['Код']).strip()]
    keys = [v['key'] for v in d['variables']]
    st.caption('Нормирование (2): g = (x − L)/(U − L) для «больше — лучше» и g = (U − x)/(U − L) для «меньше — лучше», '
               'с ограничением отрезком [0, 1]. Точки g = 0,2 и g = 0,8 — пороги «плохо» и «хорошо».')
    with st.container(horizontal=True, vertical_alignment='bottom'):
        free = [k for k in M.METRICS if k not in keys]
        mk = st.selectbox('Добавить показатель из каталога', free, key=f'fz_cat_{fid}', width=420,
                          format_func=lambda k: f'{M.METRICS[k][0]}, {M.METRICS[k][1]} ({k})')
        if st.button('Добавить', icon=':material/add:', key=f'fz_add_var_{fid}'):
            name, unit, direction, _ = M.METRICS[mk]
            d['variables'].append(F.suggest_variable(mk, name, unit, direction or 1))
            bump(fid)
            st.rerun()
    st.caption('Код показателя должен совпадать с показателем в профиле оценки (страница «Сводная оценка» → '
               '«Источники показателей»): там указывается, из какого тестового набора он берётся.')

# ---------------- термы ----------------
with t_terms:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Входные термы** на шкале g ∈ [0, 1] (трапеция a, b, c, d; треугольник — b = c)')
        idf = pd.DataFrame([{'Терм': t, 'Смысл': {'Н': 'плохо', 'С': 'средне', 'В': 'хорошо'}[t], 'a': p[0], 'b': p[1],
                             'c': p[2], 'd': p[3]} for t, p in d['in_terms'].items()])
        ied = st.data_editor(idf, hide_index=True, disabled=['Терм', 'Смысл'], key=f'fz_in_{fid}_{ver}')
    with c2:
        st.markdown('**Выходные термы «Качество»** на шкале [0, 10]')
        odf = pd.DataFrame([{'Терм': t, 'Смысл': F.OUT_NAMES[t], 'a': p[0], 'b': p[1], 'c': p[2], 'd': p[3]}
                            for t, p in d['out_terms'].items()])
        oed = st.data_editor(odf, hide_index=True, disabled=['Терм', 'Смысл'], key=f'fz_out_{fid}_{ver}')
    d['in_terms'] = {r['Терм']: [float(r[x]) for x in 'abcd'] for r in ied.to_dict('records')}
    d['out_terms'] = {r['Терм']: [float(r[x]) for x in 'abcd'] for r in oed.to_dict('records')}
    W.figure(f'terms_{fid}', 'terms', dict(in_terms=d['in_terms'], out_terms=d['out_terms']),
             'Термы: а — входных переменных на нормированной шкале g; б — выходной переменной «Качество»')

# ---------------- правила ----------------
vmap = {v['key']: v for v in d['variables']}


def next_name():
    used = set()
    for line in d['rules']:
        head = line.split(':', 1)[0].strip()
        if head[:1] in 'RР' and head[1:].isdigit():
            used.add(int(head[1:]))
    n = 1
    while n in used:
        n += 1
    return f'R{n}'


def label(var, t):
    return f"{t} — {vmap.get(var, {}).get('labels', {}).get(t, '')}"


with t_rules:
    if not keys:
        st.info('Сначала добавьте показатели на вкладке «Показатели».')
    else:
        with st.container(border=True):
            st.markdown('**Новое правило**')
            with st.container(horizontal=True, vertical_alignment='bottom'):
                n_cond = st.number_input('Условий', 1, 8, 2, key=f'fz_nc_{fid}', width=110)
                conn = st.segmented_control('Связка условий', ['И', 'ИЛИ'], default='И', required=True, key=f'fz_conn_{fid}')
            conds = []
            for i in range(int(n_cond)):
                with st.container(horizontal=True, vertical_alignment='bottom'):
                    neg = st.checkbox('НЕ', key=f'fz_neg_{fid}_{i}')
                    var = st.selectbox(f'Показатель {i + 1}', keys, key=f'fz_v_{fid}_{i}', width=300,
                                       index=min(i, len(keys) - 1), format_func=lambda k: vmap[k]['name'])
                    term = st.selectbox('Терм', list(F.IN_TERMS), key=f'fz_t_{fid}_{i}', width=260,
                                        format_func=lambda t, v=var: label(v, t), index=2)
                conds.append(('НЕ ' if neg else '') + f'{var}={term}')
            with st.container(horizontal=True, vertical_alignment='bottom'):
                out = st.selectbox('ТО качество', list(F.OUT_TERMS), index=3, key=f'fz_out_t_{fid}', width=260,
                                   format_func=lambda t: f'{t} — {F.OUT_NAMES[t]}')
                w = st.number_input('Вес правила', 0.05, 1.0, 1.0, 0.05, key=f'fz_w_{fid}', width=140)
            line = f'{next_name()}: ' + f' {conn} '.join(conds) + f' -> {out}' + (f' [{w:g}]' if w < 1 else '')
            try:
                preview = F.rule_text(F.parse_rule(line, keys), vmap)
                st.markdown(f'Будет добавлено: **{line.split(":", 1)[0]}** — ЕСЛИ {preview}')
                if st.button('Добавить правило', type='primary', icon=':material/add:', key=f'fz_add_rule_{fid}'):
                    d['rules'].append(line)
                    bump(fid)
                    st.rerun()
            except F.RuleError as e:
                st.error(str(e))

        with st.expander('Таблица решений: правила для всех сочетаний двух показателей'):
            st.caption('Для каждой клетки выберите вывод о качестве; пустые клетки пропускаются. Так строятся правила '
                       'вида R6–R9 статьи («точность высокая И устойчивость средняя → В»), обеспечивающие полноту базы.')
            with st.container(horizontal=True):
                xv = st.selectbox('Строки', keys, key=f'fz_dx_{fid}', format_func=lambda k: vmap[k]['name'], width=300)
                yv = st.selectbox('Столбцы', [k for k in keys if k != xv] or keys, key=f'fz_dy_{fid}',
                                  format_func=lambda k: vmap[k]['name'], width=300)
            cols = [label(yv, t) for t in F.IN_TERMS]
            grid = pd.DataFrame([[''] * 3 for _ in F.IN_TERMS], columns=cols,
                                index=[f"{vmap[xv]['name']}: {label(xv, t)}" for t in F.IN_TERMS])
            ged = st.data_editor(grid, key=f'fz_grid_{fid}_{xv}_{yv}_{ver}', column_config={
                c: st.column_config.SelectboxColumn(options=[''] + list(F.OUT_TERMS)) for c in cols})
            if st.button('Добавить правила из таблицы', icon=':material/grid_on:', key=f'fz_grid_add_{fid}'):
                added = 0
                for i, xt in enumerate(F.IN_TERMS):
                    for j, yt in enumerate(F.IN_TERMS):
                        val = ged.iloc[i, j]
                        if val:
                            d['rules'].append(f'{next_name()}: {xv}={xt} И {yv}={yt} -> {val}')
                            added += 1
                if added:
                    bump(fid)
                    st.rerun()

        st.markdown(f"**База правил** ({len(d['rules'])})")
        rows, errs = [], []
        for i, line in enumerate(d['rules']):
            try:
                r = F.parse_rule(line, keys, f'R{i + 1}')
                text = F.rule_text(r, vmap)
            except F.RuleError as e:
                text = f'ошибка: {e}'
                errs.append(f'{line.split(":", 1)[0]}: {e}')
            rows.append({'Правило': line, 'Формулировка': text, 'Удалить': False})
        red = st.data_editor(pd.DataFrame(rows, columns=['Правило', 'Формулировка', 'Удалить']), hide_index=True,
                             num_rows='fixed', key=f'fz_rules_{fid}_{ver}', disabled=['Формулировка'],
                             column_config={'Правило': st.column_config.TextColumn(width='large',
                                                                                  help='можно править прямо в ячейке'),
                                            'Формулировка': st.column_config.TextColumn(width='large')})
        d['rules'] = [r['Правило'] for r in red.to_dict('records') if str(r['Правило'] or '').strip()]
        for e in errs:
            st.error(e)
        with st.container(horizontal=True):
            if st.button('Удалить отмеченные', icon=':material/delete:', key=f'fz_del_{fid}'):
                d['rules'] = [r['Правило'] for r in red.to_dict('records') if not r['Удалить'] and r['Правило']]
                bump(fid)
                st.rerun()
            if st.button('Перенумеровать R1…Rn', icon=':material/format_list_numbered:', key=f'fz_renum_{fid}'):
                d['rules'] = [f"R{i + 1}: {line.split(':', 1)[1].strip() if ':' in line else line}"
                              for i, line in enumerate(d['rules'])]
                bump(fid)
                st.rerun()
        with st.expander('Правила текстом'):
            st.caption('Одно правило в строке: «R1: acc=В И lat=В И НЕ res=Н -> ОВ». Связки И, ИЛИ, НЕ и скобки; '
                       'термы входа Н/С/В, выхода ОН/Н/С/В/ОВ; вес правила — «-> С [0.8]».')
            txt = st.text_area('Правила', value='\n'.join(d['rules']), height=260, key=f'fz_txt_{fid}_{ver}',
                               label_visibility='collapsed')
            if st.button('Применить текст', key=f'fz_txt_apply_{fid}'):
                d['rules'] = [x for x in txt.splitlines() if x.strip()]
                bump(fid)
                st.rerun()
        ok_rules = []
        for line in d['rules']:
            try:
                ok_rules.append(F.parse_rule(line, keys))
            except F.RuleError:
                pass
        if ok_rules:
            W.table(pd.DataFrame([{'№': r.name, 'ЕСЛИ': F.rule_text(r, vmap).rsplit(' → ', 1)[0],
                                   'ТО качество': r.out + (f' [{r.weight:g}]' if r.weight != 1 else '')}
                                  for r in ok_rules]),
                    f'rules_{fid}', 'База нечётких правил обобщённой оценки качества LLM')

# ---------------- проверка и пробный расчёт ----------------
with t_check:
    try:
        mm = F.model_from_dict(d)
        model_err = None
    except (F.RuleError, KeyError, ValueError) as e:
        mm, model_err = None, str(e)
    if model_err:
        st.error(f'Модель не собирается: {model_err}')
    elif not mm.rules:
        st.info('Добавьте правила.')
    else:
        st.markdown('**Полнота базы правил**')
        st.caption('Утверждение 1 статьи: при любых значениях показателей хотя бы одно правило выполняется со степенью '
                   'не менее 1/2. Проверка — перебор характерных точек термов по всем показателям.')
        if st.button('Проверить полноту', icon=':material/fact_check:', key=f'fz_compl_{fid}'):
            with st.spinner('Перебор точек…'):
                res = F.completeness(mm)
            pts = f"{res['points']:,}".replace(',', ' ')
            if res['min_max_w'] >= 0.5:
                st.success(f"База полна: min по точкам max ω = {W.fmt_num(res['min_max_w'])} (≥ 0,5). Проверено точек: "
                           f"{pts}" + ('' if res['exhaustive'] else ' (случайная выборка)'))
            elif res['min_max_w'] > 0:
                st.warning(f"Есть области со слабым срабатыванием правил: min max ω = {W.fmt_num(res['min_max_w'])}.")
            else:
                st.error('Есть значения показателей, при которых не срабатывает ни одно правило — оценка не определена. '
                         'Добавьте правила (например, через таблицу решений).')
            if res['min_max_w'] < 0.5:
                st.write('Наихудшая точка (значения g):', {mm.var_map[k]['name']: round(v, 2)
                                                           for k, v in res['worst_point'].items()})
            if res['unused_vars']:
                st.warning('Показатели не участвуют ни в одном правиле: ' +
                           ', '.join(mm.var_map[k]['name'] for k in res['unused_vars']))

        st.markdown('**Пробный расчёт**')
        st.caption('Введите значения показателей условной модели — увидите нормированные значения, сработавшие правила '
                   'и итоговые оценки.')
        cols = st.columns(min(6, len(mm.variables)))
        raw = {}
        for i, v in enumerate(mm.variables):
            raw[v['key']] = cols[i % len(cols)].number_input(
                f"{v['name']}, {v['unit']}", value=float((float(v['L']) + float(v['U'])) / 2), key=f'fz_try_{fid}_{v["key"]}')
        k_try = st.slider('Порядок k', 1, 6, int(mm.k), key=f'fz_try_k_{fid}')
        g, mu, res = mm.infer(raw)
        act = [r for r in res if r['w'] > 0]
        with st.container(horizontal=True):
            st.metric(f'ВСММ (k = {k_try})', W.fmt_num(F.q_wmsm(res, k_try)), border=True)
            st.metric('Центроид', W.fmt_num(F.q_centroid(res)[0]), border=True)
            st.metric('Сработало правил', len(act), border=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            W.static_table(pd.DataFrame([{'Показатель': mm.var_map[kk]['name'], 'g': g[kk],
                                          'μН': mu[kk]['Н'], 'μС': mu[kk]['С'], 'μВ': mu[kk]['В']} for kk in mm.order]))
        with c2:
            W.static_table(pd.DataFrame([{'Правило': r['rule'], 'ω': r['w'], 'Терм': r['term'], 'q': r['q']} for r in act]))
        if act:
            W.figure(f'try_rule_{fid}', 'rule', dict(model=d, raw=raw, k=k_try),
                     'Усечённые заключения правил и итоговые оценки для заданных значений показателей')

# ---------------- сохранение ----------------
st.divider()
saved = file_dict(fid)
dirty = saved != d
with st.container(horizontal=True, vertical_alignment='center'):
    if st.button('Сохранить модель', type='primary', icon=':material/save:', disabled=not dirty):
        try:
            F.save_model(d)
            st.toast('Модель сохранена', icon=':material/check:')
            st.rerun()
        except (F.RuleError, KeyError, ValueError) as e:
            st.error(f'Не сохранено: {e}')
    if st.button('Отменить изменения', disabled=not dirty):
        drafts[fid] = copy.deepcopy(saved)
        bump(fid)
        st.rerun()
    st.caption('Есть несохранённые изменения.' if dirty else 'Изменений нет.')
