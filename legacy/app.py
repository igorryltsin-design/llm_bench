# -*- coding: utf-8 -*-
"""Пульт тестирования LLM (LM Studio) и нечёткой оценки качества.
Запуск:  streamlit run app.py   (или start.command)"""
import csv, json, os, signal, subprocess, sys, time
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import altair as alt

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import fuzzy_llm as F
import evaluate as E

DATA, OUT = HERE / 'data', HERE / 'results'
STATUS, LOG = OUT / 'status.json', OUT / 'run.log'
OUT.mkdir(exist_ok=True)
METHODS = [('wmsm_k', 'ВСММ'), ('centroid', 'Центроид Мамдани'), ('wsum', 'Взвешенная сумма'), ('topsis', 'TOPSIS')]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # фиксированный порядок категорий
TERM_RU = {'VL': 'ОН', 'L': 'Н', 'M': 'С', 'H': 'В', 'VH': 'ОВ'}

st.set_page_config(page_title='Оценка LLM', page_icon='📊', layout='wide')

# ---------------- служебное ----------------
def read_status():
    try: return json.loads(STATUS.read_text(encoding='utf-8'))
    except Exception: return {}

def alive(pid):
    if not pid: return False
    try: os.kill(pid, 0)
    except OSError: return False
    try:   # зомби-процесс считаем завершённым
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True

def running():
    s = read_status()
    return s.get('state') in ('running', 'preparing') and alive(st.session_state.get('pid') or s.get('pid'))

def launch(args):
    LOG.write_text('', encoding='utf-8')
    STATUS.write_text(json.dumps({'state': 'running', 'stage': 'запуск', 'updated': time.time()}), encoding='utf-8')
    env = dict(os.environ, LMSTUDIO_HOST=st.session_state.host, PYTHONUNBUFFERED='1')
    p = subprocess.Popen([sys.executable, '-u', str(HERE / 'run_bench.py'), *args], cwd=HERE, env=env,
                         stdout=LOG.open('a', encoding='utf-8'), stderr=subprocess.STDOUT, start_new_session=True)
    st.session_state.pid = p.pid
    s = read_status(); s['pid'] = p.pid; STATUS.write_text(json.dumps(s, ensure_ascii=False), encoding='utf-8')

def lms(path, method='GET', **kw):
    r = requests.request(method, st.session_state.host + path, timeout=kw.pop('timeout', 10), **kw)
    r.raise_for_status(); return r.json()

def lms_models():
    j = lms('/api/v1/models')
    return j.get('models') or j.get('data') or []

def unload_all():
    for m in lms_models():
        for inst in m.get('loaded_instances') or []:
            iid = inst.get('id') or inst.get('instance_id') if isinstance(inst, dict) else inst
            try: lms('/api/v1/models/unload', 'POST', json={'instance_id': iid})
            except Exception: pass

def stop():
    pid = st.session_state.get('pid') or read_status().get('pid')
    if pid:
        try: os.killpg(pid, signal.SIGTERM)
        except OSError: pass
    s = read_status(); s.update(state='stopped', stage='остановлено пользователем')
    STATUS.write_text(json.dumps(s, ensure_ascii=False), encoding='utf-8')
    try: unload_all()
    except Exception: pass

def measured(fname):
    p = OUT / fname
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def drop_rows(fname, keys):
    df = measured(fname)
    if not df.empty: df[~df['model'].isin(keys)].to_csv(OUT / fname, index=False)

def jl_count(name):
    p = DATA / name
    return sum(1 for l in p.read_text(encoding='utf-8').splitlines() if l.strip()) if p.exists() else 0

# ---------------- боковая панель ----------------
with st.sidebar:
    st.header('Настройки')
    st.text_input('Адрес сервера LM Studio', 'http://127.0.0.1:1234', key='host')
    st.number_input('Память стенда, ГБ', 1.0, 512.0, 24.0, 1.0, key='platform_gb')
    try:
        n_models = len(lms_models()); st.success(f'LM Studio доступен: моделей {n_models}')
        lms_ok = True
    except Exception as e:
        st.error('LM Studio недоступен. Включите сервер: Developer → Start Server.'); lms_ok = False
    st.divider()
    st.caption('Наборы данных')
    c = {n: jl_count(f'{n}.jsonl') for n in ('qa', 'toxic', 'bias')}
    st.write(f"Вопросы: **{c['qa']}** · провокации: **{c['toxic']}** · пары: **{c['bias']}**")
    with st.expander('Подготовить наборы' if not all(c.values()) else 'Пересобрать наборы', expanded=not all(c.values())):
        nq = st.number_input('Вопросов (MMLU-ru)', 20, 2000, 200, 20)
        nt = st.number_input('Провоцирующих запросов', 10, 1000, 100, 10)
        nb = st.number_input('Пар для предвзятости', 20, 2000, 200, 20)
        if st.button('Скачать и подготовить', disabled=running()):
            launch(['--prepare', '--n-qa', str(nq), '--n-tox', str(nt), '--n-bias', str(nb)])
            st.session_state.prep = True; st.rerun()

tab_run, tab_res, tab_log = st.tabs(['▶ Прогон', '📊 Результаты и оценка', '📄 Журнал'])

# ---------------- вкладка «Прогон» ----------------
with tab_run:
    mode = st.radio('Режим', ['Пробный', 'Полный'], horizontal=True,
                    help='Пробный — первые N элементов каждого набора, результаты в отдельный файл.')
    limit = st.number_input('N для пробного прогона', 1, 100, 5) if mode == 'Пробный' else 0
    fname = 'raw_metrics_trial.csv' if mode == 'Пробный' else 'raw_metrics.csv'
    done_df = measured(fname); done = set(done_df['model']) if 'model' in done_df else set()

    if lms_ok:
        rows = []
        for m in lms_models():
            if (m.get('type') or 'llm') != 'llm': continue
            key = m.get('key') or m.get('id') or m.get('model_key')
            q = m.get('quantization') or {}
            rows.append({'Выбрать': key not in done, 'Модель': key,
                         'Параметры': m.get('params_string') or '', 'Формат': m.get('format') or '',
                         'Квантование': q.get('name') if isinstance(q, dict) else str(q),
                         'Размер, ГБ': round((m.get('size_bytes') or 0) / 2**30, 1),
                         'Рассуждения': 'low' if 'gpt-oss' in str(key).lower() else 'off',
                         'Статус': '✅ измерено' if key in done else '—'})
        if not rows: st.info('В LM Studio нет загруженных моделей.')
        edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width='stretch',
            disabled=['Модель', 'Параметры', 'Формат', 'Квантование', 'Размер, ГБ', 'Статус'],
            column_config={'Выбрать': st.column_config.CheckboxColumn(width='small'),
                           'Рассуждения': st.column_config.SelectboxColumn(options=['off', 'low', 'medium', 'high'],
                               help='off — отключить; для gpt-oss полностью отключить нельзя')},
            key=f'models_{mode}') if rows else pd.DataFrame()
        sel = edited[edited['Выбрать']] if len(edited) else edited
        remeasure = st.checkbox('Перемерить уже измеренные из выбранных', value=False)

        c1, c2, _ = st.columns([1, 1, 4])
        is_run = running()
        if c1.button('▶ Запустить', type='primary', disabled=is_run or not all(c.values()) or sel.empty):
            keys = list(sel['Модель'])
            if remeasure: drop_rows(fname, keys)
            args = ['--models', *keys, '--reasoning', *[f"{r['Модель']}={r['Рассуждения']}" for _, r in sel.iterrows()],
                    '--platform-gb', str(st.session_state.platform_gb)]
            if limit: args += ['--limit', str(limit)]
            launch(args); st.rerun()
        if c2.button('■ Остановить', disabled=not is_run):
            stop(); st.rerun()
        if not all(c.values()): st.warning('Сначала подготовьте наборы данных (боковая панель).')

    @st.fragment(run_every=2)
    def live():
        s = read_status(); state = s.get('state', 'idle'); on = running()
        if state in ('running', 'preparing') and not on:
            state = 'error' if not s.get('error') else state
        label = {'running': '⏳ идёт прогон', 'preparing': '⏳ подготовка наборов', 'done': '✅ завершено',
                 'stopped': '⏹ остановлено', 'error': '⚠️ ошибка / процесс завершился', 'idle': 'ожидание'}.get(state, state)
        st.subheader(label)
        if s.get('error'): st.error(s['error'])
        models, dm = s.get('models') or [], s.get('done_models') or []
        if models:
            st.progress(len(dm) / len(models), text=f'Модели: {len(dm)} из {len(models)}'
                        + (f" · сейчас: **{s.get('model')}**" if s.get('model') and on else ''))
        if on and s.get('n'):
            st.progress(min(1.0, s['i'] / s['n']), text=f"{s.get('stage')}: {s['i']} из {s['n']}")
        elif on: st.caption(s.get('stage', ''))
        if s.get('started') and on:
            st.caption(f"Прошло {int((time.time() - s['started']) // 60)} мин")
        if LOG.exists():
            tail = LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-12:]
            if tail: st.code('\n'.join(tail), language=None)
        if st.session_state.get('was_on') and not on:
            st.session_state.was_on = False; st.rerun()
        st.session_state.was_on = on
    live()

# ---------------- вкладка «Результаты» ----------------
with tab_res:
    files = [f for f in ('raw_metrics.csv', 'raw_metrics_trial.csv') if not measured(f).empty]
    if not files:
        st.info('Результатов пока нет. Запустите прогон на вкладке «Прогон».')
    else:
        f_sel = st.radio('Файл результатов', files, horizontal=True,
                         format_func=lambda x: 'полный прогон' if x == 'raw_metrics.csv' else 'пробный прогон')
        raw = measured(f_sel)
        st.markdown('**Измеренные показатели**')
        show = raw.rename(columns={**F.METRIC_RU, 'model': 'Модель', 'format': 'Формат', 'quant': 'Квант.',
                                   'reasoning': 'Рассужд.', 'weights_gb': 'Веса, ГБ', 'ttft': 'TTFT, с',
                                   'tok_s': 'Ток/с', 'unparsed_pct': 'Неразобр. ответов, %',
                                   'bias_refusal_pct': 'Отказы (предвз.), %'})
        st.dataframe(show, hide_index=True, width='stretch')

        with st.expander('Параметры метода: порядок k и границы нормирования', expanded=False):
            k = st.slider('Порядок k ВСММ', 1, 6, 2, help='1 — компенсирующее усреднение, ≥2 — недостатки не компенсируются')
            if 'bounds' not in st.session_state or st.button('Сбросить к значениям статьи'):
                st.session_state.bounds = pd.DataFrame(
                    [{'Показатель': F.METRIC_RU[v], 'key': v, 'L': E.DEFAULT_BOUNDS[v][0], 'U': E.DEFAULT_BOUNDS[v][1],
                      'Лучше': '↑ больше' if E.DEFAULT_BOUNDS[v][2] > 0 else '↓ меньше'} for v in F.ORDER])
            if st.button('Подобрать L и U по диапазону данных'):
                b = st.session_state.bounds.copy()
                for i, r in b.iterrows():
                    col = raw[r['key']].dropna()
                    if len(col) >= 2 and col.max() > col.min():
                        pad = 0.1 * (col.max() - col.min())
                        b.at[i, 'L'], b.at[i, 'U'] = round(col.min() - pad, 3), round(col.max() + pad, 3)
                st.session_state.bounds = b
            bdf = st.data_editor(st.session_state.bounds, hide_index=True, width='stretch',
                                 disabled=['Показатель', 'key', 'Лучше'], column_config={'key': None})
            st.caption('Точки g = 0,2 и g = 0,8 соответствуют порогам «плохо» и «хорошо». '
                       'Подбор по данным делает оценку относительной — для статьи укажите выбранные границы.')
        bounds = {r['key']: (float(r['L']), float(r['U']), E.DEFAULT_BOUNDS[r['key']][2]) for _, r in bdf.iterrows()}
        bad = [F.METRIC_RU[v] for v, (L, U, _) in bounds.items() if not U > L]
        if bad:
            st.error('Нужно L < U: ' + ', '.join(bad)); st.stop()
        scores, skipped = E.score(raw.to_dict('records'), bounds, k)
        for m, why in skipped: st.warning(f'{m}: {why}')

        if scores:
            st.markdown(f'**Обобщённые оценки (шкала 0–10), порядок k = {k}**')
            tbl = pd.DataFrame([{'Ранг': i + 1, 'Модель': o['model'], 'ВСММ': round(o['wmsm_k'], 2),
                                 'Центроид Мамдани': round(o['centroid'], 2), 'Взвешенная сумма': round(o['wsum'], 2),
                                 'TOPSIS': round(o['topsis'], 2), 'Сработало правил': o['m']}
                                for i, o in enumerate(scores)])
            st.dataframe(tbl, hide_index=True, width='stretch')

            long = pd.DataFrame([{'Модель': o['model'], 'Метод': name, 'Оценка': round(o[key], 2)}
                                 for o in scores for key, name in METHODS])
            order = [o['model'] for o in scores]
            chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y('Модель:N', sort=order, title=None, axis=alt.Axis(labelLimit=320)),
                yOffset=alt.YOffset('Метод:N', sort=[n for _, n in METHODS]),
                x=alt.X('Оценка:Q', scale=alt.Scale(domain=[0, 10]), title='Оценка'),
                color=alt.Color('Метод:N', sort=[n for _, n in METHODS],
                                scale=alt.Scale(domain=[n for _, n in METHODS], range=COLORS),
                                legend=alt.Legend(orient='top', title=None)),
                tooltip=['Модель', 'Метод', 'Оценка']).properties(height=alt.Step(18))
            st.altair_chart(chart, width='stretch')

            kdf = pd.DataFrame([{'Модель': o['model'], 'k': kk, 'ВСММ': round(v, 3)}
                                for o in scores for kk, v in o['by_k'].items()])
            st.markdown('**Чувствительность к порядку k**')
            st.altair_chart(alt.Chart(kdf).mark_line(point=alt.OverlayMarkDef(size=64), strokeWidth=2).encode(
                x=alt.X('k:O', title='Порядок k', axis=alt.Axis(labelAngle=0)), y=alt.Y('ВСММ:Q', scale=alt.Scale(domain=[0, 10])),
                color=alt.Color('Модель:N', sort=order, scale=alt.Scale(range=COLORS + ['#e87ba4', '#008300', '#4a3aa7', '#e34948']),
                                legend=alt.Legend(orient='top', title=None)),
                tooltip=['Модель', 'k', 'ВСММ']).properties(height=260), width='stretch')

            st.markdown('**Объяснение оценок**')
            for o in scores:
                with st.expander(f"{o['model']} — ВСММ {o['wmsm_k']:.2f}"):
                    st.dataframe(pd.DataFrame([{'Показатель': F.METRIC_RU[v], 'Нормированное g': round(o['g'][v], 3)}
                                               for v in F.ORDER]), hide_index=True)
                    st.dataframe(pd.DataFrame([{'Правило': r['rule'], 'Формулировка': F.RULE_TEXT[r['rule']],
                                                'ω': round(r['w'], 3), 'Терм': TERM_RU[r['term']], 'q': round(r['q'], 2)}
                                               for r in o['rules']]), hide_index=True, width='stretch')

            out_csv = tbl.to_csv(index=False).encode('utf-8-sig')
            st.download_button('Скачать оценки (CSV)', out_csv, 'scores.csv', 'text/csv')
            st.download_button('Скачать показатели (CSV)', raw.to_csv(index=False).encode('utf-8-sig'),
                               f_sel, 'text/csv')

# ---------------- вкладка «Журнал» ----------------
with tab_log:
    if LOG.exists():
        st.code(LOG.read_text(encoding='utf-8', errors='replace')[-20000:] or '(пусто)', language=None)
    else:
        st.info('Журнал появится после первого запуска.')
    ans = sorted(OUT.glob('toxic_answers_*.jsonl'))
    if ans:
        st.markdown('**Ответы на провоцирующие запросы** (для ручной проверки классификатора)')
        pick = st.selectbox('Модель', ans, format_func=lambda p: p.stem.replace('toxic_answers_', ''))
        df = pd.read_json(pick, lines=True).sort_values('p_toxic', ascending=False)
        st.dataframe(df, hide_index=True, width='stretch')
