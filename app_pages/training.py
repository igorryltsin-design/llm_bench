import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core import bench as B
from core import db, jobs, sysinfo
from core import training as TR
from ui import widgets as W

st.caption('Дообучение методом LoRA (QLoRA для 4-битных моделей) через mlx-lm на Apple Silicon. Базовая модель — '
           'папка MLX из каталога LM Studio. Результат подключается как отдельная модель, связанная с исходной, '
           'и сравнивается с ней на тех же тестовых наборах (страница «Результаты» → «Сравнение двух моделей»).')
if not sysinfo.has('mlx_lm'):
    st.error('Не установлен пакет mlx-lm (установите из vendor/wheels).')
    st.stop()

t_data, t_train, t_result, t_saved = st.tabs(['Данные', 'Обучение', 'Подключение результата', 'Примеры из диалога'],
                                            on_change='rerun', key='tr_tabs')
own = [b for b in B.list_benchmarks() if b['type'] in ('mc', 'short', 'structured', 'open')]

# ---------------- данные ----------------
if t_data.open is not False:
    with t_data:
        bsel = st.multiselect('Наборы', [b['id'] for b in own], default=[b['id'] for b in own if b.get('kind') == 'own'],
                              format_func=lambda x: next(b['name'] for b in own if b['id'] == x), key='tr_b')
        with st.container(horizontal=True, vertical_alignment='bottom'):
            splits = st.pills('Выборки', list(B.SPLITS), default=['train'], selection_mode='multi',
                              format_func=B.SPLIT_RU.get, key='tr_splits')
            letter_only = st.toggle('Выбор варианта: ответ только буквой', value=True, key='tr_letter',
                                    help='Как требует инструкция в тестах; иначе «B) текст варианта»')
        with st.container(horizontal=True, vertical_alignment='bottom'):
            use_graded = st.toggle('Добавить ответы моделей, одобренные экспертом', value=False, key='tr_graded')
            gmin = st.slider('Минимальная оценка эксперта', 0.5, 1.0, 0.8, 0.05, key='tr_gmin', disabled=not use_graded,
                             width=300)
            use_saved = st.toggle('Добавить примеры из диалога', value=True, key='tr_saved')
        system = st.text_input('Системная инструкция для всех примеров (необязательно)', key='tr_sys')
        ex = TR.build_examples(bsel, tuple(splits or ['train']), letter_only, gmin if use_graded else None, use_saved,
                               system)
        st.session_state['tr_examples'] = ex
        st.metric('Обучающих примеров', len(ex), border=True, width=220)
        if ex:
            W.static_table(pd.DataFrame([{'Источник': e['source'],
                                          'Запрос': next(m['content'] for m in e['messages'] if m['role'] == 'user')[:200],
                                          'Ответ': e['messages'][-1]['content'][:200]} for e in ex]), max_rows_visible=8)
            leaks = TR.leakage(ex)
            if leaks:
                st.error(f'Найдены пересечения с выборкой «оценка»: {len(leaks)}. Такие задания нельзя использовать '
                         'для обучения — иначе сравнение до и после дообучения будет завышено.')
                W.static_table(pd.DataFrame(leaks).rename(columns={'example': 'Обучающий пример',
                                                                   'eval_item': 'Задание оценки', 'overlap': 'Доля общего',
                                                                   'kind': 'Вид'}))
            else:
                st.success('Утечек в выборку «оценка» не найдено.')
            st.download_button('Скачать набор (JSONL, формат chat)',
                               ''.join(json.dumps({'messages': e['messages']}, ensure_ascii=False) + '\n' for e in ex),
                               'train.jsonl', on_click='ignore', icon=':material/download:')
        else:
            st.info('Нет примеров: разметьте задания выборкой «обучение» (страница «Тестовые наборы» → «Выборки»), '
                    'сохраните ответы из «Диалога» или включите одобренные экспертом ответы.')

# ---------------- обучение ----------------
if t_train.open:
    with t_train:
        ex = st.session_state.get('tr_examples') or []
        models = TR.mlx_models()
        if not models:
            st.warning('Не найдено моделей MLX в каталоге LM Studio (' + W.settings()['lmstudio_models_dir'] + ').')
        else:
            with st.form('train_form'):
                mi = st.selectbox('Базовая модель (MLX)', range(len(models)),
                                  format_func=lambda i: f"{models[i]['name']} · {models[i]['arch']} · "
                                                        f"{models[i]['bits'] or 16} бит · {W.fmt_num(models[i]['size_gb'], 1)} ГБ")
                cands = [c for c in W.candidates_cached() if c['backend'] != 'legacy']
                base_c = st.selectbox('С какой моделью сравнивать (исходная модель в списке «Модели»)',
                                      [''] + [c['id'] for c in cands], format_func=lambda x: W.cand_name(x) if x else '—')
                name = st.text_input('Название задания', value=time.strftime('lora_%Y%m%d_%H%M'))
                with st.container(horizontal=True):
                    iters = st.number_input('Итераций', 10, 100000, 300, 10, width=130)
                    lr = st.number_input('Скорость обучения', 1e-7, 1e-2, 1e-5, format='%.1e', width=160)
                    batch = st.number_input('Размер пакета', 1, 64, 1, width=130)
                    layers = st.number_input('Слоёв (−1 — все)', -1, 200, 8, width=150)
                with st.container(horizontal=True):
                    rank = st.number_input('Ранг LoRA', 1, 256, 8, width=130)
                    scale = st.number_input('Масштаб LoRA', 0.1, 100.0, 20.0, width=150)
                    max_seq = st.number_input('Макс. длина примера, токенов', 128, 32768, 1024, 128, width=220)
                    valid = st.number_input('Доля на проверку', 0.0, 0.5, 0.1, 0.05, width=150)
                with st.container(horizontal=True):
                    ftype = st.segmented_control('Метод', ['lora', 'dora'], default='lora', required=True)
                    mask = st.toggle('Учить только на ответах (маска запроса)', value=True)
                    ckpt = st.toggle('Экономия памяти (grad checkpoint)', value=True)
                go = st.form_submit_button('Начать обучение', type='primary', icon=':material/model_training:',
                                           disabled=not ex)
            if not ex:
                st.caption('Сначала соберите обучающие примеры на вкладке «Данные».')
            m = models[mi] if models else None
            if m:
                st.caption(f"Оценка памяти: веса ≈ {W.fmt_num(m['size_gb'], 1)} ГБ + активации и адаптер; "
                           f"при нехватке уменьшите длину примера, число слоёв или размер пакета. "
                           f"Свободно сейчас ≈ {W.fmt_num(sysinfo.total_memory_gb() - sysinfo.used_memory_gb(), 1)} ГБ.")
            if go and m:
                leaks = TR.leakage(ex)
                if leaks:
                    st.error('Обучение не запущено: есть пересечения с выборкой «оценка» (вкладка «Данные»).')
                else:
                    try:
                        TR.start_job(B.slugify(name), m['path'], ex, dict(
                            iters=iters, lr=lr, batch_size=batch, num_layers=layers, rank=rank, scale=scale,
                            max_seq=max_seq, valid_frac=valid, type=ftype, mask_prompt=mask, grad_checkpoint=ckpt,
                            eval_every=max(10, iters // 10), save_every=max(50, iters // 4)), base_c or None)
                        st.toast('Обучение запущено', icon=':material/check:')
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))

        jl = TR.list_jobs()
        if jl:
            st.subheader('Задания на обучение')
            state_ru = {'running': 'идёт', 'done': 'завершено', 'error': 'ошибка', 'stopped': 'остановлено'}
            jsel = st.selectbox('Задание', [j['name'] for j in jl],
                                format_func=lambda n: f"{n} — {state_ru.get(next(j for j in jl if j['name'] == n)['state'], '')}",
                                key='tr_job')

            @st.fragment(run_every=3)
            def job_view():
                j = next((x for x in TR.list_jobs() if x['name'] == jsel), None)
                if not j:
                    return
                log_path = TR.job_dir(j['name']) / 'train.log'
                text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
                tr, va = TR.parse_log(text)
                with st.container(horizontal=True):
                    st.metric('Состояние', state_ru.get(j['state'], j['state']), border=True)
                    st.metric('Итерация', f"{tr[-1][0] if tr else 0} из {j['params'].get('iters')}", border=True)
                    st.metric('Потери (обуч.)', W.fmt_num(tr[-1][1], 3) if tr else '—', border=True)
                    st.metric('Потери (пров.)', W.fmt_num(va[-1][1], 3) if va else '—', border=True)
                    st.metric('Примеров', f"{j['n_train']} + {j['n_val']}", border=True)
                if j['state'] == 'running' and st.button('Остановить обучение', icon=':material/stop:'):
                    jobs.stop(j['pid'])
                if tr or va:
                    W.figure(f"loss_{j['name']}", 'training', dict(train=tr, val=va),
                             f"Функция потерь при дообучении ({Path(j['base_path']).name}, {j['name']})")
                st.code('\n'.join(text.splitlines()[-12:]) or '(журнал пуст)', language=None)

            job_view()

# ---------------- подключение результата ----------------
if t_result.open:
    with t_result:
        done = [j for j in TR.list_jobs() if (TR.job_dir(j['name']) / 'adapter' / 'adapters.safetensors').exists()]
        if not done:
            st.info('Нет завершённых заданий с сохранённым адаптером.')
        else:
            jn = st.selectbox('Задание', [j['name'] for j in done], key='res_job')
            j = next(x for x in done if x['name'] == jn)
            st.caption(f"База: {j['base_path']} · примеров {j['n_train']} · итераций {j['params'].get('iters')}")
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown('**Вариант 1. Сервер с адаптером** (быстро, без объединения весов)')
                    st.caption('Запускается mlx_lm.server с базовой моделью и адаптером (OpenAI-совместимый). '
                               'Появится подключение и модель для оценки, связанная с исходной.')
                    port = st.number_input('Порт', 1024, 65535, int(j.get('server_port') or 8090), key='srv_port')
                    running = jobs.alive(j.get('server_pid'))
                    if running:
                        st.success(f'Сервер работает: http://127.0.0.1:{j.get("server_port")}')
                        if st.button('Остановить сервер', icon=':material/stop:'):
                            jobs.stop(j['server_pid'])
                            st.rerun()
                    elif st.button('Запустить сервер и добавить модель', type='primary', icon=':material/play_arrow:'):
                        used = TR.serve(j, int(port))
                        cid = TR.register_adapter_candidate(j, used)
                        W.candidates_cached.clear()
                        W.ping.clear()
                        st.success(f'Добавлена модель «{W.cand_name(cid)}». Сервер запускается (загрузка модели '
                                   'занимает до минуты) — затем её можно включать в прогон и диалог.')
                with st.container(border=True):
                    st.markdown('**Исходная модель на том же движке**')
                    st.caption('Для честного сравнения «до/после» обе модели должны работать на одном движке: '
                               'LM Studio и mlx_lm по-разному применяют шаблон диалога и режим рассуждений. '
                               'Кнопка запускает исходную модель через mlx_lm.server (без адаптера) и назначает её '
                               'исходной для дообученной.')
                    bport = st.number_input('Порт', 1024, 65535, int(j.get('base_server_port') or port + 1), key='base_port')
                    if jobs.alive(j.get('base_server_pid')):
                        st.success(f'Работает: http://127.0.0.1:{j.get("base_server_port")}')
                        if st.button('Остановить', icon=':material/stop:', key='stop_base_srv'):
                            jobs.stop(j['base_server_pid'])
                            st.rerun()
                    elif st.button('Запустить исходную модель', icon=':material/play_arrow:', key='start_base_srv'):
                        used = TR.serve(j, int(bport), with_adapter=False)
                        cid = TR.register_base_candidate(j, used)
                        W.candidates_cached.clear()
                        W.ping.clear()
                        st.success(f'Добавлена модель «{W.cand_name(cid)}».')
            with c2:
                with st.container(border=True):
                    st.markdown('**Вариант 2. Объединить в модель LM Studio**')
                    st.caption('Адаптер вливается в веса; модель сохраняется в каталог LM Studio (models/local/…) и '
                               'после обновления списка появляется в LM Studio и на странице «Модели».')
                    out = st.text_input('Название модели', value=f"{Path(j['base_path']).name}-{jn}")
                    if st.button('Объединить', icon=':material/merge:'):
                        target = TR.fuse(j, B.slugify(out).replace('_', '-'))
                        st.info(f'Объединение запущено → {target}. Журнал: {TR.job_dir(jn) / "fuse.log"}')
                    fl = TR.job_dir(jn) / 'fuse.log'
                    if fl.exists():
                        st.code(jobs.tail(fl, 8) or '…', language=None)

# ---------------- примеры из диалога ----------------
if t_saved.open:
    with t_saved:
        saved = TR.saved_examples()
        if not saved:
            st.caption('Примеров пока нет. На странице «Диалог» нажмите «В обучение» под удачным ответом.')
        else:
            sdf = pd.DataFrame([{'Удалить': False, 'Источник': e.get('source', ''), 'Время': e.get('ts', ''),
                                 'Запрос': next((m['content'] for m in e['messages'][::-1] if m['role'] == 'user'), ''),
                                 'Ответ': e['messages'][-1]['content']} for e in saved])
            sed = st.data_editor(sdf, hide_index=True, disabled=['Источник', 'Время'], key='saved_ed',
                                 column_config={'Ответ': st.column_config.TextColumn(width='large')})
            if st.button('Сохранить изменения', type='primary', icon=':material/save:'):
                keep = []
                for e, r in zip(saved, sed.to_dict('records')):
                    if r['Удалить']:
                        continue
                    e['messages'][-1]['content'] = r['Ответ']
                    keep.append(e)
                TR.write_saved_examples(keep)
                st.rerun()
