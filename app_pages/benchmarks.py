import json

import pandas as pd
import streamlit as st

from core import bench as B
from core import db
from core import metrics as M
from core import tasks as T
from ui import widgets as W

COLS_RU = {'id': 'Код', 'split': 'Выборка', 'category': 'Категория', 'question': 'Вопрос', 'answer': 'Ответ',
           'answers': 'Допустимые ответы', 'prompt': 'Задание', 'schema': 'Схема (JSON)', 'expected': 'Эталон (JSON)',
           'reference': 'Эталонный ответ', 'rubric': 'Критерии (название | баллы)', 'stereo': 'Стереотип',
           'anti': 'Антистереотип', 'text': 'Утверждение', 'weight': 'Вес', 'tags': 'Метки', 'note': 'Заметка',
           'disabled': 'Отключено'}
REV = {v: k for k, v in COLS_RU.items()}
SPLIT_RU, SPLIT_EN = B.SPLIT_RU, {v: k for k, v in B.SPLIT_RU.items()}
INDICATORS = {'acc': 'точность', 'rob': 'устойчивость', 'tox': 'токсичность', 'bias': 'предвзятость',
              'lat': 'время отклика'}

st.caption('Тестовый набор — папка benchmarks/<код>/ с паспортом (benchmark.json) и заданиями (items.jsonl). '
           'Внешние наборы — готовые (MMLU-ru, RuBia); свои — ваши задания предметной области. '
           'Выборка «оценка» используется в прогонах, «обучение» — для дообучения, «отладка» — для проверки.')


# ---------------- диалоги ----------------
@st.dialog('Новый тестовый набор', width='medium')
def create_dialog():
    name = st.text_input('Название', placeholder='например: Задачи планирования перевозок')
    btype = st.selectbox('Тип заданий', list(B.TYPES), format_func=B.type_name)
    st.caption(B.TYPES[btype]['help'])
    kind = st.segmented_control('Вид', ['own', 'external'], format_func=B.KIND_RU.get, default='own', required=True)
    code = st.text_input('Код (латиница, для папки)', value=B.slugify(name) if name else '')
    if st.button('Создать', type='primary', disabled=not name.strip()):
        bid = B.unique_id(B.slugify(code or name))
        B.save(B.Benchmark(B.default_meta(bid, name.strip(), btype, kind), []), 'создан')
        st.session_state['_bench_open'] = bid
        st.rerun()


@st.dialog('Копия набора')
def copy_dialog(src):
    b = B.load(src)
    name = st.text_input('Название копии', value=f'{b.name} (копия)')
    kind = st.segmented_control('Вид', ['own', 'external'], format_func=B.KIND_RU.get, default='own', required=True)
    st.caption('Копия внешнего набора удобна как основа своего: её задания можно править, не трогая исходник.')
    if st.button('Копировать', type='primary'):
        nb = B.copy_benchmark(src, B.unique_id(B.slugify(name)), name, kind)
        st.session_state['_bench_open'] = nb.id
        st.rerun()


# ---------------- библиотека ----------------
benches = B.list_benchmarks()
with st.container(horizontal=True, vertical_alignment='bottom'):
    flt = st.segmented_control('Показать', ['все', 'внешние', 'свои'], default='все', required=True, key='bflt')
    if st.button('Создать набор', icon=':material/add:', type='primary'):
        create_dialog()
shown = [b for b in benches if flt == 'все' or (flt == 'свои') == (b.get('kind') == 'own')]
lib = pd.DataFrame([{'Название': b['name'], 'Код': b['id'], 'Вид': B.KIND_RU.get(b.get('kind'), ''),
                     'Тип заданий': B.type_name(b['type']), 'Заданий': b['n_items'],
                     'оценка': b['split_counts'].get('eval', 0), 'обучение': b['split_counts'].get('train', 0),
                     'отладка': b['split_counts'].get('dev', 0),
                     'Показатель': INDICATORS.get(b.get('indicator'), b.get('indicator') or ''),
                     'Изменён': b.get('updated', '')} for b in shown])
if not lib.empty:
    W.static_table(lib.rename(columns={'оценка': 'Оценка', 'обучение': 'Обучение', 'отладка': 'Отладка'}))

if not benches:
    st.info('Наборов пока нет. Создайте свой набор или выполните: python -m tools.import_external legacy')
    st.stop()

ids = [b['id'] for b in benches]
names = {b['id']: b['name'] for b in benches}
if st.session_state.get('_bench_open') in ids:
    st.session_state['bench_sel'] = st.session_state.pop('_bench_open')
with st.container(horizontal=True, vertical_alignment='bottom'):
    bid = st.selectbox('Открыть набор', ids, format_func=lambda x: f'{names[x]} ({x})', key='bench_sel', width=520)
    if st.button('Копировать', icon=':material/content_copy:'):
        copy_dialog(bid)
    with st.popover('Удалить', icon=':material/delete:'):
        st.warning('Папка набора будет удалена. Ответы моделей в базе останутся.')
        if st.button('Удалить набор', type='primary', key='del_bench'):
            B.delete(bid)
            st.session_state.pop('bench_sel', None)
            st.rerun()

b = B.load(bid)
ver_key = f'_ver_{bid}'
st.session_state.setdefault(ver_key, 0)
st.subheader(b.name)
st.caption(f"{B.KIND_RU.get(b.meta.get('kind'), '')} набор · {B.type_name(b.type)} · заданий: {len(b.items)} · "
           f"версия содержимого {b.content_hash()}" + (f" · источник: {b.meta['source']}" if b.meta.get('source') else ''))

t_items, t_card, t_meta, t_check, t_split, t_io, t_hist = st.tabs(
    ['Задания', 'Карточка задания', 'Паспорт и запрос', 'Проверка', 'Выборки', 'Импорт и выгрузка', 'История'])

# ---------------- задания (таблица) ----------------
with t_items:
    rows = B.to_rows(b)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['id', 'split', 'category'])
    with st.container(horizontal=True, vertical_alignment='bottom'):
        q = st.text_input('Поиск по тексту', key=f'q_{bid}', placeholder='слово или фраза', width=300)
        cats = sorted({r.get('category', '') for r in rows if r.get('category')})
        cat = st.selectbox('Категория', ['все'] + cats, key=f'cat_{bid}', width=260)
        spl = st.pills('Выборка', list(B.SPLITS), format_func=SPLIT_RU.get, selection_mode='multi',
                       default=list(B.SPLITS), key=f'spl_{bid}')
    mask = pd.Series(True, index=df.index)
    if q:
        mask &= df.astype(str).apply(lambda r: q.lower() in ' '.join(r).lower(), axis=1)
    if cat != 'все':
        mask &= df['category'] == cat
    if 'split' in df:
        mask &= df['split'].isin(spl or [])
    sub = df[mask].copy()
    if 'split' in sub:
        sub['split'] = sub['split'].map(SPLIT_RU).fillna('оценка')
    sub = sub.rename(columns=COLS_RU)
    cc = {'Выборка': st.column_config.SelectboxColumn(options=list(SPLIT_EN), required=True, width='small'),
          'Код': st.column_config.TextColumn(width='small'), 'Вес': st.column_config.NumberColumn(min_value=0.01, width='small'),
          'Отключено': st.column_config.CheckboxColumn(width='small', help='Не использовать в прогонах')}
    if b.type == 'mc':
        n_opt = B.n_option_columns(b.items)
        cc['Ответ'] = st.column_config.SelectboxColumn(options=list(B.LETTERS[:n_opt]), width='small')
        cc['Вопрос'] = st.column_config.TextColumn(width='large')
    for c in ('Задание', 'Эталонный ответ', 'Утверждение', 'Стереотип', 'Антистереотип'):
        cc[c] = st.column_config.TextColumn(width='large')
    st.caption(f'Показано {len(sub)} из {len(df)}. Длинные тексты удобнее править во вкладке «Карточка задания». '
               'Новые строки добавляются внизу таблицы; для удаления выделите строки и нажмите «Удалить».')
    edited = st.data_editor(sub, hide_index=True, num_rows='dynamic', column_config=cc,
                            key=f'items_{bid}_{st.session_state[ver_key]}', height=460)
    with st.container(horizontal=True):
        if st.button('Сохранить задания', type='primary', icon=':material/save:', key=f'save_items_{bid}'):
            ed = edited.rename(columns=REV)
            if 'split' in ed:
                ed['split'] = ed['split'].map(SPLIT_EN).fillna('eval')
            ed_rows = ed.to_dict('records')
            sub_ids = set(df[mask]['id']) if 'id' in df else set()
            by_id = {}
            fresh = []
            for r in ed_rows:
                rid = str(r.get('id') or '').strip()
                if rid and rid in sub_ids and rid not in by_id:
                    by_id[rid] = r
                else:
                    fresh.append(r)
            new_rows = []
            for r in rows:
                if r['id'] in sub_ids:
                    if r['id'] in by_id:
                        new_rows.append(by_id[r['id']])
                else:
                    new_rows.append(r)
            new_rows += fresh
            items, errs = B.from_rows(b, new_rows)
            if errs:
                st.error('Не сохранено: ' + '; '.join(errs[:5]))
            else:
                n_del = len(rows) - (len(new_rows) - len(fresh))
                b.items = items
                B.save(b, f'правка заданий: изменено/добавлено {len(ed_rows)}, удалено {n_del}')
                st.session_state[ver_key] += 1
                st.toast('Задания сохранены', icon=':material/check:')
                st.rerun()
        if st.button('Отменить правки', key=f'undo_{bid}'):
            st.session_state[ver_key] += 1
            st.rerun()

# ---------------- карточка задания ----------------
with t_card:
    if not b.items:
        st.info('В наборе нет заданий — добавьте первое.')
    ids_items = [str(i['id']) for i in b.items]
    with st.container(horizontal=True, vertical_alignment='bottom'):
        iid = st.selectbox('Задание', ids_items + ['— новое —'], key=f'card_{bid}', width=320)
    it = next((dict(i) for i in b.items if str(i['id']) == iid), None)
    is_new = it is None
    if is_new:
        it = {'id': B.new_item_id(b.items), 'split': 'eval'}
    with st.form(f'card_form_{bid}_{iid}'):
        with st.container(horizontal=True):
            new_id = st.text_input('Код', value=str(it['id']), width=200)
            split = st.selectbox('Выборка', list(B.SPLITS), index=list(B.SPLITS).index(it.get('split', 'eval')),
                                 format_func=SPLIT_RU.get, width=180)
            category = st.text_input('Категория', value=it.get('category', ''), width=260)
            weight = st.number_input('Вес', 0.01, 100.0, float(it.get('weight', 1) or 1), width=120)
        t = b.type
        if t == 'mc':
            question = st.text_area('Вопрос', value=it.get('question', ''), height=120)
            opts = list(it.get('options') or ['', '', '', ''])
            n = st.number_input('Число вариантов', 2, 10, max(2, len(opts)))
            opts = (opts + [''] * 10)[:n]
            new_opts = [st.text_input(f'Вариант {B.LETTERS[i]}', value=opts[i], key=f'opt_{bid}_{iid}_{i}')
                        for i in range(n)]
            ans_cur = str(it.get('answer', 'A')).upper()
            answer = st.radio('Правильный ответ', list(B.LETTERS[:n]), horizontal=True,
                              index=B.LETTERS.index(ans_cur) if ans_cur in B.LETTERS[:n] else 0)
        elif t == 'short':
            question = st.text_area('Вопрос', value=it.get('question', ''), height=120)
            answers = st.text_area('Допустимые ответы (по одному в строке)', value='\n'.join(it.get('answers') or []))
        elif t == 'structured':
            prompt = st.text_area('Задание', value=it.get('prompt', ''), height=160)
            schema_s = st.text_area('Схема ответа (JSON Schema: type, required, properties, enum, pattern…)',
                                    value=json.dumps(it.get('schema') or {'type': 'object', 'required': [],
                                                                          'properties': {}}, ensure_ascii=False, indent=2),
                                    height=200)
            expected_s = st.text_area('Эталонные значения полей (JSON)',
                                      value=json.dumps(it.get('expected') or {}, ensure_ascii=False, indent=2), height=150)
        elif t == 'open':
            prompt = st.text_area('Задание', value=it.get('prompt', ''), height=160)
            reference = st.text_area('Эталонный ответ', value=it.get('reference', ''), height=160)
            rubric_s = st.text_area('Критерии оценки (строка: название | максимальный балл)',
                                    value=B.rubric_to_text(it.get('rubric') or []), height=110)
        elif t == 'bias_pair':
            stereo = st.text_area('Стереотипное утверждение', value=it.get('stereo', ''))
            anti = st.text_area('Антистереотипное утверждение', value=it.get('anti', ''))
        elif t == 'toxicity':
            text = st.text_area('Утверждение для продолжения', value=it.get('text', ''))
        elif t == 'perf':
            prompt = st.text_area('Типовой запрос', value=it.get('prompt', ''), height=120)
        note = st.text_input('Заметка', value=it.get('note', ''))
        disabled = st.checkbox('Отключено (не использовать в прогонах)', value=bool(it.get('disabled')))
        saved = st.form_submit_button('Сохранить задание', type='primary', icon=':material/save:')
    if saved:
        new = dict(it, id=new_id.strip() or it['id'], split=split, category=category, weight=weight, note=note)
        new.pop('disabled', None)
        if disabled:
            new['disabled'] = True
        err = None
        try:
            if t == 'mc':
                new.update(question=question, options=[o for o in new_opts], answer=answer)
            elif t == 'short':
                new.update(question=question, answers=[a.strip() for a in answers.splitlines() if a.strip()])
            elif t == 'structured':
                new.update(prompt=prompt, schema=json.loads(schema_s or '{}'), expected=json.loads(expected_s or '{}'))
            elif t == 'open':
                new.update(prompt=prompt, reference=reference, rubric=B.rubric_from_text(rubric_s))
            elif t == 'bias_pair':
                new.update(stereo=stereo, anti=anti)
            elif t == 'toxicity':
                new.update(text=text)
            elif t == 'perf':
                new.update(prompt=prompt)
        except ValueError as e:
            err = f'неверный JSON: {e}'
        if not err and new['id'] != it['id'] and new['id'] in ids_items:
            err = f"код {new['id']} уже занят"
        if err:
            st.error(err)
        else:
            new = B._clean(new)
            if is_new:
                b.items.append(new)
            else:
                b.items = [new if str(x['id']) == iid else x for x in b.items]
            B.save(b, f"{'добавлено' if is_new else 'изменено'} задание {new['id']}")
            st.session_state[ver_key] += 1
            st.session_state[f'card_{bid}'] = new['id'] if not is_new else '— новое —'
            st.toast('Сохранено', icon=':material/check:')
            st.rerun()
    if not is_new:
        probs = [p for p in B.validate(B.Benchmark(b.meta, [it])) if p[0]]
        for _, lvl, msg in probs:
            (st.error if lvl == 'ошибка' else st.warning)(msg)
        st.markdown('**Запросы к модели**')
        try:
            reqs = T.build_requests(b, it, perturb=True)
            for rq in reqs[:4]:
                st.caption({'base': 'Основной запрос', 'typo': 'С опечатками', 'shuffle': 'С переставленными вариантами',
                            'instr': 'С другой инструкцией'}.get(rq['variant'], rq['variant']))
                st.code('\n\n'.join(f"[{m['role']}]\n{m['content']}" if m['role'] == 'system' else m['content']
                                    for m in rq['messages']), language=None, wrap_lines=True)
        except (KeyError, ValueError, IndexError) as e:
            st.warning(f'Запрос не строится: {e}')
            reqs = []
        cands = [c for c in W.candidates_cached() if c['backend'] != 'legacy']
        if cands and reqs:
            with st.container(horizontal=True, vertical_alignment='bottom'):
                cid = st.selectbox('Проверить на модели', [c['id'] for c in cands], format_func=W.cand_name,
                                   key=f'try_{bid}', width=380)
                go = st.button('Отправить', icon=':material/send:', key=f'try_btn_{bid}')
            if go:
                rq = reqs[0]
                try:
                    c = db.candidate(cid)
                    rs = c.get('reasoning')
                    rep = W.ask(cid, rq['messages'], T.max_tokens_for(b.meta, rs), 0.0, meta=rq['meta'])
                    parsed, sc, detail = T.score(b, it, rq['meta'], rep.text)
                    st.markdown('**Ответ модели**')
                    st.code(rep.text or '(пусто)', language=None, wrap_lines=True)
                    if rep.reasoning:
                        with st.expander('Рассуждения модели'):
                            st.text(rep.reasoning)
                    st.caption(f"Время {W.fmt_num(rep.latency)} с · разобрано: {parsed if parsed is not None else '—'} · "
                               f"балл: {W.fmt_num(sc) if sc is not None else 'оценивается отдельно'}")
                except Exception as e:  # noqa: BLE001
                    st.error(f'Ошибка: {e}')

# ---------------- паспорт и запрос ----------------
with t_meta:
    m = b.meta
    with st.form(f'meta_{bid}'):
        name = st.text_input('Название', value=m.get('name', ''))
        desc = st.text_area('Описание', value=m.get('description', ''), height=80)
        with st.container(horizontal=True):
            kind = st.selectbox('Вид', ['own', 'external'], index=0 if m.get('kind') == 'own' else 1,
                                format_func=B.KIND_RU.get, width=160)
            indicator = st.text_input('Показатель сводной оценки', value=m.get('indicator', ''), width=200,
                                      help='acc — точность, rob — устойчивость, tox — токсичность, bias — предвзятость, '
                                           'lat — время отклика, либо свой код (например, sppr)')
            lic = st.text_input('Лицензия', value=m.get('license', ''), width=200)
        source = st.text_input('Источник', value=m.get('source', ''))
        citation = st.text_area('Библиографическая ссылка', value=m.get('citation', ''), height=70)
        st.markdown('**Запрос к модели**')
        system = st.text_area('Системная инструкция (необязательно)', value=(m.get('prompt') or {}).get('system', ''),
                              height=70)
        template = st.text_area('Шаблон запроса', value=(m.get('prompt') or {}).get('template', ''), height=120,
                                help={'mc': 'Подстановки: {instruction}, {question}, {options}, {letters}',
                                      'short': 'Подстановка: {question}', 'structured': 'Подстановки: {prompt}, {schema}',
                                      'open': 'Подстановка: {prompt}', 'bias_pair': 'Подстановки: {a}, {b}',
                                      'toxicity': 'Подстановка: {text}', 'perf': 'Подстановка: {prompt}'}[b.type])
        instr = None
        if b.type == 'mc':
            instr = st.text_area('Инструкции (первая — основная, вторая — для проверки устойчивости к переформулировке)',
                                 value='\n'.join((m.get('prompt') or {}).get('instructions') or []), height=80)
            pt = m.get('perturbations') or {}
            st.markdown('**Возмущения для оценки устойчивости**')
            with st.container(horizontal=True):
                p_typos = st.checkbox('Опечатки', value=pt.get('typos', True))
                p_rate = st.number_input('Доля позиций', 0.01, 0.3, float(pt.get('typo_rate', 0.05)), 0.01, width=140)
                p_shuffle = st.checkbox('Перестановка вариантов', value=pt.get('shuffle', True))
                p_instr = st.checkbox('Другая инструкция', value=pt.get('instruction', True))
        with st.container(horizontal=True):
            mt = st.number_input('Ограничение ответа, токенов', 1, 32768, int(m.get('max_tokens', 256)), width=220)
            mtr = st.number_input('…при включённых рассуждениях', 1, 65536, int(m.get('max_tokens_reasoning', 1024)),
                                  width=220)
        sc = dict(m.get('scoring') or {})
        if b.type == 'short':
            modes = {'normalized': 'точное совпадение (после нормализации)', 'contains': 'ответ содержит эталон',
                     'number': 'число с допуском', 'regex': 'регулярное выражение'}
            sc['mode'] = st.selectbox('Способ проверки', list(modes), format_func=modes.get,
                                      index=list(modes).index(sc.get('mode', 'normalized')))
            with st.container(horizontal=True):
                sc['tolerance'] = st.number_input('Допуск (абсолютный)', 0.0, 1e9, float(sc.get('tolerance', 0)),
                                                  format='%.4f', width=200)
                sc['rel_tolerance'] = st.number_input('Допуск (относительный)', 0.0, 1.0,
                                                      float(sc.get('rel_tolerance', 0)), format='%.4f', width=200)
        elif b.type == 'structured':
            sc['check_expected'] = st.checkbox('Сверять значения полей с эталоном', value=sc.get('check_expected', True))
            sc['numeric_tolerance'] = st.number_input('Относительный допуск для чисел', 0.0, 1.0,
                                                      float(sc.get('numeric_tolerance', 1e-6)), format='%.6f')
        elif b.type == 'open':
            sc['prefer'] = st.selectbox('Итоговый балл', ['expert', 'judge'], index=0 if sc.get('prefer', 'expert') == 'expert' else 1,
                                        format_func={'expert': 'оценка эксперта (если есть), иначе судьи',
                                                     'judge': 'оценка судьи (если есть), иначе эксперта'}.get)
            sc['judge_template'] = st.text_area('Инструкция для модели-судьи', value=sc.get('judge_template', B.JUDGE_TEMPLATE),
                                                height=260, help='Подстановки: {prompt}, {reference}, {rubric}, {answer}')
        elif b.type == 'toxicity':
            sc['threshold'] = st.slider('Порог вероятности токсичности', 0.05, 0.95, float(sc.get('threshold', 0.5)), 0.05)
        if b.type == 'perf':
            with st.container(horizontal=True):
                reps = st.number_input('Повторов', 1, 200, int(m.get('repeats', 20)), width=160)
                warm = st.number_input('Прогревочных запросов', 0, 5, int(m.get('warmup', 1)), width=200)
        msg = st.text_input('Комментарий к изменению (для истории)', value='')
        ok = st.form_submit_button('Сохранить паспорт', type='primary', icon=':material/save:')
    if ok:
        m.update(name=name, description=desc, kind=kind, indicator=indicator.strip(), license=lic, source=source,
                 citation=citation, max_tokens=int(mt), max_tokens_reasoning=int(mtr))
        m.setdefault('prompt', {}).update(system=system, template=template)
        if instr is not None:
            m['prompt']['instructions'] = [x for x in instr.splitlines() if x.strip()]
            m['perturbations'] = dict(typos=p_typos, typo_rate=p_rate, shuffle=p_shuffle, instruction=p_instr)
        if sc:
            m['scoring'] = sc
        if b.type == 'perf':
            m['repeats'], m['warmup'] = int(reps), int(warm)
        B.save(b, msg or 'изменён паспорт')
        st.toast('Паспорт сохранён', icon=':material/check:')
        st.rerun()
    if b.items:
        st.markdown('**Пример итогового запроса (первое задание)**')
        try:
            rq = T.build_requests(b, b.items[0], perturb=False)[0]
            st.code('\n\n'.join(x['content'] for x in rq['messages']), language=None, wrap_lines=True)
        except (KeyError, ValueError, IndexError) as e:
            st.warning(f'Запрос не строится: {e}')

# ---------------- проверка ----------------
with t_check:
    probs = B.validate(b)
    n_err = sum(1 for p in probs if p[1] == 'ошибка')
    if not probs:
        st.success('Замечаний нет.')
    else:
        (st.error if n_err else st.warning)(f'Ошибок: {n_err}, предупреждений: {len(probs) - n_err}')
        W.static_table(pd.DataFrame(probs, columns=['Задание', 'Уровень', 'Замечание']))
    if b.type == 'mc' and b.items:
        cnt = pd.Series([str(i.get('answer')) for i in b.items]).value_counts().sort_index()
        st.caption('Распределение правильных ответов по буквам (сильный перекос облегчает угадывание): ' +
                   ', '.join(f'{k} — {v}' for k, v in cnt.items()))

# ---------------- выборки ----------------
with t_split:
    counts = {s_: sum(1 for i in b.items if i.get('split', 'eval') == s_) for s_ in B.SPLITS}
    with st.container(horizontal=True):
        for s_ in B.SPLITS:
            st.metric(SPLIT_RU[s_].capitalize(), counts[s_], border=True)
    st.markdown('**Случайное разбиение**')
    st.caption('Задания, попавшие в «обучение», не должны использоваться при оценке — иначе сравнение до и после '
               'дообучения будет некорректным. Перед обучением проверяйте утечки (страница «Дообучение»).')
    with st.form(f'split_{bid}'):
        with st.container(horizontal=True):
            f_eval = st.number_input('Оценка, %', 0, 100, 80, 5, width=130)
            f_dev = st.number_input('Отладка, %', 0, 100, 0, 5, width=130)
            f_train = st.number_input('Обучение, %', 0, 100, 20, 5, width=140)
            seed = st.number_input('Начальное число', 0, 10 ** 6, 42, width=160)
        strat = st.checkbox('Стратифицировать по категориям', value=True)
        go = st.form_submit_button('Разбить', type='primary')
    if go:
        if f_eval + f_dev + f_train == 0:
            st.error('Все доли нулевые.')
        else:
            B.assign_splits(b.items, {'eval': f_eval, 'dev': f_dev, 'train': f_train}, int(seed), strat)
            B.save(b, f'разбиение {f_eval}/{f_dev}/{f_train}, seed {seed}')
            st.session_state[ver_key] += 1
            st.rerun()

# ---------------- импорт и выгрузка ----------------
with t_io:
    st.markdown('**Выгрузка**')
    with st.container(horizontal=True):
        st.download_button('JSONL', B.export_jsonl(b), f'{bid}.jsonl', on_click='ignore', icon=':material/download:')
        st.download_button('CSV', B.export_csv(b), f'{bid}.csv', 'text/csv', on_click='ignore')
        st.download_button('Excel', lambda: W.to_xlsx(pd.DataFrame(B.to_rows(b)).rename(columns=COLS_RU)),
                           f'{bid}.xlsx', on_click='ignore')
        st.download_button('Паспорт (JSON)', json.dumps(b.meta, ensure_ascii=False, indent=2), f'{bid}.benchmark.json',
                           'application/json', on_click='ignore')
    st.markdown('**Импорт заданий из файла**')
    st.caption('Форматы: JSONL, JSON, CSV, TSV, Excel, Parquet. После загрузки сопоставьте колонки файла с полями задания.')
    up = st.file_uploader('Файл', type=['jsonl', 'json', 'csv', 'tsv', 'txt', 'xlsx', 'parquet'], key=f'up_{bid}')
    if up is not None:
        try:
            src = B.read_table(up.name, up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f'Не удалось прочитать файл: {e}')
            src = None
        if src is not None:
            W.static_table(src.head(5).astype(str))
            cols = ['—'] + [str(c) for c in src.columns]

            def guess(*names):
                for n in names:
                    for c in src.columns:
                        if str(c).lower() == n:
                            return cols.index(str(c))
                for n in names:
                    for c in src.columns:
                        if n in str(c).lower():
                            return cols.index(str(c))
                return 0

            fields = {'id': ('id', 'код', 'idx'), 'category': ('category', 'subject', 'domain', 'категория', 'тема'),
                      'split': ('split', 'выборка')}
            fields.update({
                'mc': {'question': ('question', 'вопрос', 'question_ru'), 'options': ('options', 'choices', 'choices_ru',
                                                                                    'варианты'),
                       'answer': ('answer', 'ответ', 'label', 'target')},
                'short': {'question': ('question', 'вопрос'), 'answers': ('answers', 'answer', 'ответ')},
                'structured': {'prompt': ('prompt', 'задание', 'input'), 'schema': ('schema', 'схема'),
                               'expected': ('expected', 'эталон', 'output')},
                'open': {'prompt': ('prompt', 'задание', 'question', 'вопрос'), 'reference': ('reference', 'эталон', 'answer'),
                         'rubric': ('rubric', 'критерии')},
                'bias_pair': {'stereo': ('stereo', 'pro-trope', 'стереотип'), 'anti': ('anti', 'anti-trope', 'антистереотип')},
                'toxicity': {'text': ('text', 'prompt', 'утверждение')},
                'perf': {'prompt': ('prompt', 'запрос')}}[b.type])
            mapping = {}
            grid = st.columns(3)
            for i, (f, names_) in enumerate(fields.items()):
                sel = grid[i % 3].selectbox(COLS_RU.get(f, f) if f != 'options' else 'Варианты (списком)', cols,
                                            index=guess(*names_), key=f'map_{bid}_{f}')
                mapping[f] = None if sel == '—' else sel
            if b.type == 'mc' and not mapping.get('options'):
                st.caption('Либо варианты отдельными колонками:')
                grid = st.columns(4)
                for i, L in enumerate(B.LETTERS[:6]):
                    sel = grid[i % 4].selectbox(f'Вариант {L}', cols, index=guess(L.lower(), f'option_{L.lower()}'),
                                                key=f'map_{bid}_opt{L}')
                    mapping[f'opt_{L}'] = None if sel == '—' else sel
            if b.type == 'mc':
                mapping['answer_index_base'] = st.radio('Если ответ — номер, нумерация с', [0, 1], horizontal=True)
            mode = st.segmented_control('Режим', ['добавить', 'заменить все задания'], default='добавить', required=True)
            if st.button('Импортировать', type='primary', key=f'imp_{bid}'):
                items, errs = B.items_from_table(src, b.type, mapping)
                if errs:
                    st.warning(f'Пропущено строк: {len(errs)}. ' + '; '.join(errs[:5]))
                if items:
                    if mode == 'добавить':
                        used = {str(i['id']) for i in b.items}
                        for it_ in items:
                            if str(it_['id']) in used:
                                it_['id'] = B.new_item_id(b.items + items, 'imp')
                            used.add(str(it_['id']))
                        b.items += items
                    else:
                        b.items = items
                    B.save(b, f'импорт из {up.name}: {len(items)} заданий ({mode})')
                    st.session_state[ver_key] += 1
                    st.success(f'Импортировано заданий: {len(items)}')

# ---------------- история ----------------
with t_hist:
    log = b.meta.get('changelog') or []
    if log:
        W.static_table(pd.DataFrame(list(reversed(log))).rename(columns={'ts': 'Время', 'msg': 'Изменение'}))
    else:
        st.caption('Изменений пока не было.')
    n_resp = db.one('SELECT COUNT(*) n FROM responses WHERE bench_id=?', (bid,))['n']
    st.caption(f'Ответов моделей на этот набор в базе: {n_resp}. При изменении текста задания старые ответы на него '
               'перестают учитываться (задание считается не пройденным), остальные сохраняются.')
