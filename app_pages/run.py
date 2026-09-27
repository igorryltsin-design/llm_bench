import time

import pandas as pd
import streamlit as st

from core import bench as B
from core import db, jobs
from core import tasks as T
from ui import widgets as W

s = W.settings()


@st.fragment(run_every=2)
def live():
    r = jobs.active_run()
    if not r:
        if st.session_state.get('_was_running'):
            st.session_state['_was_running'] = False
            st.rerun()
        return
    st.session_state['_was_running'] = True
    p = r.get('progress') or {}
    with st.container(border=True):
        st.markdown(f"**Прогон №{r['id']}** — {r['name']}")
        cands = p.get('candidates') or []
        done = p.get('done_candidates') or []
        if cands:
            st.progress(len(done) / len(cands),
                        text=f"Модели: {len(done)} из {len(cands)}" +
                             (f" · сейчас: **{W.cand_name(p['candidate'])}**" if p.get('candidate') else ''))
        n, i = p.get('n') or 0, p.get('i') or 0
        if n:
            eta = p.get('eta')
            st.progress(min(1.0, i / n), text=f"{p.get('stage', '')}: {i} из {n}" +
                        (f" · осталось ≈ {int(eta // 60)} мин {int(eta % 60)} с" if eta else ''))
        else:
            st.caption(p.get('stage', ''))
        st.caption(f"Идёт {W.duration(r['started'], None)}")
        log = jobs.tail(jobs.run_log(r['id']), 14)
        if log:
            st.code(log, language=None)
        with st.container(horizontal=True):
            if st.button('Остановить', icon=':material/stop:', help='Текущий запрос завершится, модель выгрузится; '
                                                                    'полученные ответы сохранятся'):
                jobs.stop(r['pid'])
                db.update_run(r['id'], state='stopping')
                st.toast('Останавливаю…')
            if st.button('Прервать немедленно', icon=':material/cancel:', type='tertiary'):
                jobs.stop(r['pid'], hard=True)
                time.sleep(0.5)
                st.rerun()


live()
active = jobs.active_run()

tab_new, tab_judge, tab_hist = st.tabs(['Новый прогон', 'Оценка моделью-судьёй', 'История прогонов'])

cands = [c for c in W.candidates_cached() if c['backend'] != 'legacy']
benches = B.list_benchmarks()
bnames = {b['id']: f"{b['name']} ({B.KIND_RU.get(b.get('kind'), '')})" for b in benches}

with tab_new:
    if not cands:
        st.info('Сначала добавьте модели на странице «Модели».')
    else:
        with st.container(border=True):
            sel_c = st.multiselect('Модели', [c['id'] for c in cands], format_func=W.cand_name, key='run_cands')
            ext = [b['id'] for b in benches if b.get('kind') == 'external']
            sel_b = st.multiselect('Тестовые наборы', [b['id'] for b in benches], format_func=bnames.get,
                                   default=[x for x in ext if x in bnames], key='run_benches')
            with st.container(horizontal=True, vertical_alignment='bottom'):
                mode = st.segmented_control('Режим', ['пробный', 'полный'], default='пробный', required=True,
                                            help='Пробный — первые N заданий каждого набора: проверить, что всё работает.')
                limit = st.number_input('N заданий', 1, 1000, 5, width=120, disabled=mode == 'полный')
                splits = st.pills('Выборки', list(B.SPLITS), default=['eval'], selection_mode='multi',
                                  format_func=B.SPLIT_RU.get)
            with st.expander('Параметры генерации'):
                with st.container(horizontal=True):
                    temperature = st.number_input('Температура', 0.0, 2.0, 0.0, 0.1, width=140,
                                                  help='0 — детерминированная генерация (как в методике)')
                    seed = st.number_input('Начальное число', 0, 10 ** 6, 0, width=160,
                                           help='Для случайных возмущений и порядка утверждений')
                    repeats = st.number_input('Повторов каждого задания', 1, 20, 1, width=200,
                                              help='Больше 1 имеет смысл при температуре > 0 (оценка разброса)')
                    ctx = st.number_input('Длина контекста', 512, 262144, int(s['context_length']), 512, width=160)
                perturb = st.toggle('Возмущения для оценки устойчивости (задания с выбором варианта)', value=True)
                unload = st.toggle('Выгружать модель после прогона', value=bool(s.get('unload_after', True)))
            # оценка числа запросов
            n_req = 0
            for bid in sel_b:
                b = B.load(bid)
                items = b.select(tuple(splits or ['eval']), 0 if mode == 'полный' else int(limit))
                per = len(T.build_requests(b, items[0], perturb=perturb, repeats=1 if b.type == 'perf' else repeats)) if items else 0
                n_req += per * len(items)
            st.caption(f'Запросов на одну модель: {n_req}; всего: {n_req * len(sel_c)}. Уже полученные ответы '
                       'с теми же настройками повторно не запрашиваются.')
            disabled = bool(active) or not sel_c or not sel_b or not splits
            if st.button('Запустить', type='primary', icon=':material/play_arrow:', disabled=disabled):
                name = f"{'Пробный' if mode == 'пробный' else 'Полный'}: {len(sel_c)} мод. × {len(sel_b)} наб."
                rid = db.create_run('bench', name, dict(
                    candidates=sel_c, benches=sel_b, splits=list(splits), limit=0 if mode == 'полный' else int(limit),
                    temperature=float(temperature), seed=int(seed), repeats=int(repeats), perturb=perturb,
                    context_length=int(ctx), unload_after=unload, platform_gb=float(s['platform_gb'])))
                jobs.launch_run(rid)
                st.session_state['_was_running'] = True
                time.sleep(0.8)
                st.rerun()
            if active:
                st.caption('Дождитесь окончания текущего прогона или остановите его.')

with tab_judge:
    st.caption('Модель-судья оценивает развёрнутые ответы по критериям из заданий и эталонному ответу. '
               'Перед тем как доверять судье, сравните его оценки с экспертными (страница «Результаты» → «Судья и эксперт»).')
    open_b = [b['id'] for b in benches if b['type'] == 'open']
    if not open_b:
        st.info('Нет наборов с развёрнутыми ответами.')
    elif cands:
        judge = st.selectbox('Модель-судья', [c['id'] for c in cands], format_func=W.cand_name, key='judge_c')
        jb = st.multiselect('Наборы', open_b, default=open_b, format_func=bnames.get, key='judge_b')
        jc = st.multiselect('Чьи ответы оценивать', [c['id'] for c in cands], default=[c['id'] for c in cands],
                            format_func=W.cand_name, key='judge_targets')
        jsplits = st.pills('Выборки', list(B.SPLITS), default=['eval'], selection_mode='multi',
                           format_func=B.SPLIT_RU.get, key='judge_splits')
        over = st.toggle('Переоценить уже оценённые этим судьёй', value=False)
        if st.button('Запустить оценку', type='primary', icon=':material/gavel:',
                     disabled=bool(active) or not jb or not jc):
            rid = db.create_run('judge', f'Судья: {W.cand_name(judge)}', dict(
                judge=judge, benches=jb, candidates=jc, splits=list(jsplits or ['eval']), overwrite=over))
            jobs.launch_run(rid)
            st.session_state['_was_running'] = True
            time.sleep(0.8)
            st.rerun()

with tab_hist:
    runs = db.runs(100)
    if not runs:
        st.caption('Прогонов ещё не было.')
    else:
        hdf = pd.DataFrame([{'№': r['id'], 'Вид': 'судья' if r['kind'] == 'judge' else 'прогон', 'Название': r['name'],
                             'Состояние': W.STATE_RU.get(r['state'], r['state']), 'Начат': W.ago(r['started']),
                             'Длительность': W.duration(r['started'], r['finished']),
                             'Ошибка': (r.get('error') or '')[:120]} for r in runs])
        W.static_table(hdf, max_rows_visible=10)
        pick = st.selectbox('Подробно о прогоне', range(len(runs)), key='runs_pick',
                            format_func=lambda i: f"№{runs[i]['id']} — {runs[i]['name']} ({W.STATE_RU.get(runs[i]['state'], '')})")
        if pick is not None:
            r = runs[pick]
            st.markdown(f"**Прогон №{r['id']}** — {r['name']}")
            st_ = r.get('settings') or {}
            if r['kind'] == 'bench':
                st.caption('Модели: ' + ', '.join(W.cand_name(c) for c in st_.get('candidates', [])) + ' · наборы: ' +
                           ', '.join(bnames.get(b, b) for b in st_.get('benches', [])) +
                           f" · N = {st_.get('limit') or 'все'} · температура {st_.get('temperature')}")
            with st.container(horizontal=True):
                if r['state'] in ('stopped', 'error') and not active:
                    if st.button('Продолжить прогон', icon=':material/resume:', type='primary'):
                        jobs.launch_run(r['id'])
                        st.session_state['_was_running'] = True
                        time.sleep(0.8)
                        st.rerun()
                st.download_button('Журнал', jobs.tail(jobs.run_log(r['id']), 100000), f"run_{r['id']}.log",
                                   on_click='ignore', icon=':material/download:')
            st.code(jobs.tail(jobs.run_log(r['id']), 60) or '(журнал пуст)', language=None)
