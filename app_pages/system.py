import os

import pandas as pd
import streamlit as st

from core import charts as CH
from core import config as C
from core import db, jobs, sysinfo
from ui import widgets as W

s = W.settings()
t_fig, t_off, t_set, t_logs, t_data = st.tabs(['Оформление рисунков', 'Работа без сети', 'Настройки', 'Журналы',
                                               'Хранилище'])

with t_fig:
    st.caption('Параметры по умолчанию для всех рисунков. У каждого рисунка их можно поменять в меню «Сохранить рисунок».')
    cs = W.chart_settings()
    fonts = CH.font_names()
    with st.form('fig_form'):
        with st.container(horizontal=True):
            font = st.selectbox('Шрифт', fonts, index=fonts.index(cs['font']) if cs['font'] in fonts else 0, width=300)
            size = st.number_input('Размер, пт', 6.0, 20.0, float(cs['size']), 0.5, width=140)
            width = st.number_input('Ширина рисунка, см', 6.0, 30.0, float(cs['width_cm']), 0.5, width=180,
                                    help='Ширина текста на листе A4 с полями 2 и 1,5 см — 17,5 см')
            start = st.number_input('Нумерация рисунков и таблиц с', 1, 999, int(cs.get('start', 1)), width=220,
                                    help='Для новых рисунков и таблиц; у уже созданных номер хранится в их подписи')
        bw = st.toggle('Чёрно-белые рисунки (штриховка, типы линий)', value=bool(cs['bw']))
        comma = st.toggle('Десятичная запятая', value=bool(cs['decimal_comma']))
        if st.form_submit_button('Сохранить', type='primary'):
            W.update_settings(charts=dict(font=font, size=size, width_cm=width, bw=bw, decimal_comma=comma, start=start))
            st.rerun()
    if 'Times New Roman' not in fonts:
        st.warning('Шрифт Times New Roman не найден в системе — рисунки будут построены другим шрифтом с засечками.')
    W.figure('style_demo', 'bars', dict(categories=['Набор 1', 'Набор 2', 'Набор 3'],
                                        series={'Модель А': [72.5, 64.0, 81.2], 'Модель Б': [68.1, 70.3, 77.9]},
                                        errors={'Модель А': [(66, 78), (57, 70), (75, 86)],
                                                'Модель Б': [(61, 74), (63, 76), (71, 83)]},
                                        ylabel='Точность, %', ylim=(0, 100)),
             'Пример оформления рисунка')
    if st.button('Сбросить нумерацию и подписи всех рисунков и таблиц', type='tertiary'):
        C.write_json(W.CAPTIONS, {})
        for k in [k for k in st.session_state if str(k).startswith(('fig_', 'tab_'))]:
            del st.session_state[k]
        st.rerun()

with t_off:
    rows = sysinfo.offline_checks()
    W.static_table(pd.DataFrame([{'Проверка': r[0], 'Результат': 'да' if r[1] else 'нет', 'Подробно': r[2]} for r in rows]))
    st.markdown(
        '- Пакеты Python лежат в `vendor/wheels`; при первом запуске `start.command` ставит их без сети.\n'
        '- Классификатор токсичности — `vendor/models/russian_toxicity_classifier`.\n'
        '- Исходные файлы внешних наборов для пересборки — `vendor/datasets` (MMLU-ru, RuBia).\n'
        '- Приложение слушает только 127.0.0.1; обращения к внешним адресам в коде отсутствуют.\n'
        '- Обновить комплект на машине с сетью: `bash tools/make_offline_bundle.sh`.')
    m = sysinfo.machine()
    st.table({'Система': m['os'], 'Процессор': m['chip'], 'Память, ГБ': str(m['memory_gb']), 'Python': m['python'],
              'Версия приложения': C.APP_VERSION}, border='horizontal', width='content')

with t_set:
    with st.form('set_form'):
        ctx = st.number_input('Длина контекста при загрузке модели', 512, 262144, int(s['context_length']), 512)
        lms_dir = st.text_input('Каталог моделей LM Studio', value=s['lmstudio_models_dir'])
        unload = st.toggle('Выгружать модель после прогона', value=bool(s.get('unload_after', True)))
        expert = st.text_input('Эксперт по умолчанию', value=s.get('expert') or '')
        if st.form_submit_button('Сохранить', type='primary'):
            W.update_settings(context_length=int(ctx), lmstudio_models_dir=lms_dir, unload_after=unload, expert=expert)
            st.toast('Сохранено', icon=':material/check:')

with t_logs:
    logs = sorted(C.LOG_DIR.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        lp = st.selectbox('Журнал', logs, format_func=lambda p: f'{p.name} ({W.ago(p.stat().st_mtime)})')
        st.code(jobs.tail(lp, 300) or '(пусто)', language=None)
    else:
        st.caption('Журналов пока нет.')

with t_data:
    st.metric('Размер базы', f"{W.fmt_num(C.DB_PATH.stat().st_size / 2 ** 20, 1)} МБ" if C.DB_PATH.exists() else '—',
              border=True, width=200)
    cnt = db.one('SELECT (SELECT COUNT(*) FROM responses) r, (SELECT COUNT(*) FROM grades) g, '
                 '(SELECT COUNT(*) FROM runs) n, (SELECT COUNT(*) FROM candidates) c')
    st.caption(f"Ответов: {cnt['r']}, оценок: {cnt['g']}, прогонов: {cnt['n']}, моделей: {cnt['c']}. "
               f'База: {C.DB_PATH}')
    st.download_button('Резервная копия базы', lambda: C.DB_PATH.read_bytes(), 'bench.db', on_click='ignore',
                       icon=':material/download:')
    with st.popover('Удалить ответы модели', icon=':material/delete:'):
        cands = db.candidates(True)
        if cands:
            cid = st.selectbox('Модель', [c['id'] for c in cands], format_func=W.cand_name)
            if st.button('Удалить все её ответы и оценки', type='primary'):
                db.delete_responses(cid)
                st.toast('Удалено')
