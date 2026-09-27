import json

import pandas as pd
import streamlit as st

from core import backends as BK
from core import db
from ui import widgets as W

s = W.settings()
tab_c, tab_b = st.tabs(['Модели для оценки', 'Подключения к серверам'])

# ---------------- подключения ----------------
with tab_b:
    st.caption('Все адреса — локальные. LM Studio: родной API с загрузкой и выгрузкой моделей и режимом рассуждений. '
               'OpenAI-совместимый: mlx_lm.server (в т. ч. с адаптером дообучения), Ollama (http://127.0.0.1:11434), '
               'llama.cpp server, vLLM. «Демо» — имитация моделей для проверки работы программы.')
    df = pd.DataFrame([{'id': b['id'], 'Название': b['name'], 'Тип': b['kind'], 'Адрес': b.get('url', ''),
                        'Состояние': 'доступен' if W.ping(json.dumps(b))[0] else 'недоступен'} for b in s['backends']])
    ed = st.data_editor(df, hide_index=True, num_rows='dynamic', key='backends_editor', column_config={
        'id': st.column_config.TextColumn('Код', help='короткий латинский код подключения', required=True),
        'Тип': st.column_config.SelectboxColumn(options=list(BK.KINDS), required=True,
                                                help='lmstudio / openai (совместимый) / demo'),
        'Состояние': st.column_config.TextColumn(disabled=True)})
    with st.container(horizontal=True):
        if st.button('Сохранить подключения', type='primary', icon=':material/save:'):
            rows = [r for r in ed.to_dict('records') if r.get('id')]
            ids = [r['id'] for r in rows]
            if len(set(ids)) != len(ids):
                st.error('Коды подключений должны быть уникальными.')
            else:
                old = {b['id']: b for b in s['backends']}
                s['backends'] = [dict(old.get(r['id'], {}), id=r['id'], name=r['Название'] or r['id'],
                                      kind=r['Тип'] or 'openai', url=(r['Адрес'] or '').strip()) for r in rows]
                W.update_settings(backends=s['backends'])
                W.ping.clear()
                W.list_models.clear()
                st.rerun()
        if st.button('Проверить', icon=':material/refresh:'):
            W.ping.clear()
            W.list_models.clear()
            st.rerun()
    for b in s['backends']:
        ok, err = W.ping(json.dumps(b))
        if not ok:
            st.caption(f"{b['name']}: {err}")

# ---------------- модели для оценки ----------------
with tab_c:
    st.number_input('Память стенда, ГБ', 1.0, 1024.0, float(s['platform_gb']), 1.0, key='platform_gb',
                    help='Для показателя «Доля памяти стенда» = объём весов / память стенда.',
                    on_change=lambda: W.update_settings(platform_gb=st.session_state.platform_gb), width=220)
    st.subheader('Добавить из сервера')
    bmap = {b['id']: b for b in s['backends']}
    bid = st.selectbox('Сервер', list(bmap), format_func=lambda x: bmap[x]['name'], key='add_backend')
    cfg = bmap[bid]
    try:
        models = [m for m in W.list_models(json.dumps(cfg)) if m.get('type', 'llm') == 'llm']
    except Exception as e:  # noqa: BLE001
        models = []
        st.error(f'Сервер недоступен: {e}')
    existing = {(c['backend'], c['model']) for c in db.candidates(True)}
    if models:
        mdf = pd.DataFrame([{'Модель': m['key'], 'Параметры': m.get('params', ''), 'Формат': m.get('format', ''),
                             'Квантование': m.get('quant', ''), 'Объём, ГБ': round((m.get('size_bytes') or 0) / 2 ** 30, 2),
                             'Контекст': m.get('max_ctx'), 'Режимы рассуждений': ', '.join(m.get('reasoning_options') or []),
                             'Загружена': 'да' if m.get('loaded') else '',
                             'Уже добавлена': 'да' if (bid, m['key']) in existing else ''} for m in models])
        W.static_table(mdf)
        keys_sel = st.multiselect('Выберите модели для добавления', [m['key'] for m in models], key='srv_pick',
                                  format_func=lambda k: k + ('  (уже добавлена)' if (bid, k) in existing else ''))
        sel = [m for m in models if m['key'] in keys_sel]
        opts = sorted({o for m in sel for o in (m.get('reasoning_options') or [])}) if sel else []
        with st.container(horizontal=True, vertical_alignment='bottom'):
            reasoning = st.selectbox('Режим рассуждений', ['по умолчанию'] + (opts or ['off', 'low', 'medium', 'high', 'on']),
                                     index=1 if 'off' in opts else 0, width=220,
                                     help='В методике статьи рассуждения отключаются (off); у gpt-oss минимальный — low.')
            label = st.text_input('Метка (необязательно)', width=260,
                                  help='Чтобы добавить ту же модель с другими настройками как отдельного кандидата.')
            if st.button('Добавить выбранные', type='primary', disabled=not sel, icon=':material/add:'):
                for m in sel:
                    rs = 'default' if reasoning == 'по умолчанию' else reasoning
                    if 'gpt-oss' in m['key'] and rs == 'off':
                        rs = 'low'
                    cid = db.candidate_id(bid, m['key'], rs, label.strip())
                    db.upsert_candidate(dict(id=cid, backend=bid, model=m['key'], reasoning=rs, label=label.strip(),
                                             info={k: m.get(k) for k in ('format', 'quant', 'size_bytes', 'params',
                                                                         'arch', 'max_ctx')}))
                W.candidates_cached.clear()
                st.toast(f'Добавлено: {len(sel)}', icon=':material/check:')
                st.rerun()

    st.subheader('Модели для оценки')
    cands = db.candidates(True)
    if not cands:
        st.info('Пока нет моделей. Добавьте их из списка сервера выше.')
    else:
        names = {c['id']: db.display_name(c) for c in cands}
        cdf = pd.DataFrame([{'Код': c['id'], 'Название': c.get('label') or '', 'Сервер': c['backend'],
                             'Модель': c['model'], 'Рассуждения': c.get('reasoning') or '',
                             'Исходная модель': names.get(c.get('base_id'), '') if c.get('base_id') else '',
                             'Объём, ГБ': round(((c.get('info') or {}).get('size_bytes') or 0) / 2 ** 30, 2),
                             'Скрыта': bool(c.get('hidden')), 'Заметка': c.get('note') or '',
                             'Предупреждение': 'рассуждает в тексте ответа' if (c.get('info') or {}).get('reasoning_leak')
                             else ''} for c in cands])
        ced = st.data_editor(cdf, hide_index=True, key='cand_editor',
                             disabled=['Код', 'Сервер', 'Модель', 'Рассуждения', 'Объём, ГБ', 'Предупреждение'],
                             column_config={
                                 'Название': st.column_config.TextColumn(help='Отображаемое имя (метка)'),
                                 'Исходная модель': st.column_config.SelectboxColumn(
                                     options=[''] + list(names.values()),
                                     help='Для дообученной модели — с какой моделью её сравнивать'),
                                 'Скрыта': st.column_config.CheckboxColumn(help='Не показывать в списках')})
        with st.container(horizontal=True):
            if st.button('Сохранить изменения', type='primary', icon=':material/save:', key='save_cands'):
                by_name = {v: k for k, v in names.items()}
                for r, c in zip(ced.to_dict('records'), cands):
                    c.update(label=r['Название'] or '', base_id=by_name.get(r['Исходная модель']) or None,
                             hidden=int(bool(r['Скрыта'])), note=r['Заметка'] or '')
                    db.upsert_candidate(c)
                W.candidates_cached.clear()
                st.toast('Сохранено', icon=':material/check:')
                st.rerun()
            with st.popover('Удалить модель', icon=':material/delete:'):
                victim = st.selectbox('Модель', list(names), format_func=names.get, key='del_cand')
                with_resp = st.checkbox('Удалить и все её ответы и оценки', value=False)
                if st.button('Удалить', type='primary', key='del_cand_btn'):
                    db.delete_candidate(victim, with_resp)
                    W.candidates_cached.clear()
                    st.rerun()
        st.caption('Модели с сервером «legacy» перенесены из прежнего пульта: у них есть только итоговые показатели '
                   '(для сводной оценки), без отдельных ответов.')
