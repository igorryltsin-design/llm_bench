import json

import pandas as pd
import streamlit as st

from core import bench as B
from core import db, jobs, sysinfo
from ui import widgets as W

st.caption('Оценка и сравнение локальных языковых моделей без доступа к интернету: внешние и собственные '
           'тестовые наборы, сводная нечёткая оценка (Мамдани + ВСММ), дообучение и сравнение с исходной моделью.')

s = W.settings()
benches = B.list_benchmarks()
cands = [c for c in db.candidates() if c['backend'] != 'legacy']
n_resp = db.one('SELECT COUNT(*) n FROM responses')['n']
with st.container(horizontal=True):
    ok_backends = sum(W.ping(json.dumps(b))[0] for b in s['backends'])
    st.metric('Серверы моделей', f"{ok_backends} из {len(s['backends'])}", border=True,
              help='Доступные подключения (страница «Модели»)')
    st.metric('Модели для оценки', len(cands), border=True)
    st.metric('Тестовые наборы', len(benches), border=True,
              delta=f"своих {sum(1 for b in benches if b.get('kind') == 'own')}", delta_color='off', delta_arrow='off')
    st.metric('Ответов в базе', f'{n_resp:,}'.replace(',', ' '), border=True)

r = jobs.active_run()
if r:
    st.info(f"Идёт прогон №{r['id']} «{r['name']}» — страница «Прогон».", icon=':material/play_circle:')

left, right = st.columns([3, 2])
with left:
    st.subheader('Порядок работы')
    st.markdown(
        '1. **Модели** — подключите сервер (LM Studio, mlx_lm.server, Ollama) и добавьте модели для оценки.\n'
        '2. **Тестовые наборы** — внешние (MMLU-ru, RuBia) уже подготовлены; создайте свои: выбор варианта, '
        'короткий ответ, структурированный вывод (JSON), развёрнутый ответ.\n'
        '3. **Прогон** — сначала пробный (N заданий), затем полный. Прогон можно остановить и продолжить.\n'
        '4. **Результаты** — показатели с доверительными интервалами, разбивка по категориям, графики, '
        'проверка развёрнутых ответов экспертом или моделью-судьёй, сравнение двух моделей.\n'
        '5. **Сводная оценка** — ВСММ, центроид, взвешенная сумма, TOPSIS, чувствительность, согласованность '
        'с экспертами; выгрузка таблиц и рисунков в Word.\n'
        '6. **Дообучение** — обучающий набор из своих заданий, LoRA на MLX, сравнение с исходной моделью.\n'
        '7. **Диалог** — разговор с одной или двумя моделями (например, исходной и дообученной) бок о бок.')
with right:
    st.subheader('Подключения')
    for b in s['backends']:
        ok, err = W.ping(json.dumps(b))
        st.markdown(f"{':material/check_circle:' if ok else ':material/error:'} **{b['name']}** "
                    f"{'— доступен' if ok else '— недоступен'}")
        if not ok and b['kind'] == 'lmstudio':
            st.caption('В LM Studio: вкладка Developer → Start Server (порт 1234).')
    st.subheader('Офлайн-режим')
    checks = sysinfo.offline_checks()
    bad = [c for c in checks if not c[1]]
    if bad:
        st.warning('Есть замечания: ' + '; '.join(c[0] for c in bad) + '. Подробнее — страница «Система».')
    else:
        st.success('Всё необходимое для работы без сети на месте.')

runs = db.runs(8)
if runs:
    st.subheader('Последние прогоны')
    W.static_table(pd.DataFrame([{'№': x['id'], 'Вид': 'оценка судьёй' if x['kind'] == 'judge' else 'прогон',
                                'Название': x['name'], 'Состояние': W.STATE_RU.get(x['state'], x['state']),
                                'Начат': W.ago(x['started']), 'Длительность': W.duration(x['started'], x['finished'])}
                               for x in runs]))
