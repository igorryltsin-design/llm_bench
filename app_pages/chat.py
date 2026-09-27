import time

import streamlit as st

from core import bench as B
from core import db
from core import training as TR
from ui import widgets as W

cands = [c for c in W.candidates_cached() if c['backend'] != 'legacy']
if not cands:
    st.info('Добавьте модели на странице «Модели».')
    st.stop()
ids = [c['id'] for c in cands]
cmap = {c['id']: c for c in cands}

# по умолчанию — дообученная модель и её исходная, если есть
tuned = [c for c in cands if c.get('base_id') in cmap]
default = [tuned[0]['base_id'], tuned[0]['id']] if tuned else ids[:1]

with st.sidebar:
    st.subheader('Параметры диалога')
    sel = st.multiselect('Модели (одна или две)', ids, default=default, max_selections=2, format_func=W.cand_name,
                         key='chat_models')
    system = st.text_area('Системная инструкция', value='', height=90, key='chat_sys',
                          placeholder='например: Отвечай кратко и по существу.')
    temperature = st.slider('Температура', 0.0, 1.5, 0.3, 0.05, key='chat_temp')
    max_tokens = st.number_input('Ограничение ответа, токенов', 16, 32768, 1024, 64, key='chat_mt')
    if st.button('Очистить диалог', icon=':material/clear_all:'):
        st.session_state['chat'] = []
        st.rerun()
    if st.button('Выгрузить модели, загруженные здесь', icon=':material/eject:'):
        W.unload_ui_models()
        st.toast('Выгружено')

if not sel:
    st.info('Выберите одну или две модели в боковой панели.')
    st.stop()

hist = st.session_state.setdefault('chat', [])   # [{'user': str, 'answers': {cid: {...}}}]


def messages_for(cid, upto=None):
    msgs = [{'role': 'system', 'content': system}] if system.strip() else []
    for turn in hist[:upto]:
        msgs.append({'role': 'user', 'content': turn['user']})
        a = turn['answers'].get(cid)
        if a and a.get('text'):
            msgs.append({'role': 'assistant', 'content': a['text']})
    return msgs


@st.dialog('Сохранить как пример для дообучения', width='large')
def save_example_dialog(ti, cid):
    turn = hist[ti]
    msgs = messages_for(cid, ti) + [{'role': 'user', 'content': turn['user']}]
    st.caption('Отредактируйте ответ до эталонного вида — модель будет учиться именно на нём.')
    ans = st.text_area('Эталонный ответ', value=turn['answers'][cid]['text'], height=260)
    if st.button('Сохранить пример', type='primary'):
        TR.save_example(msgs + [{'role': 'assistant', 'content': ans}], source=f'диалог: {W.cand_name(cid)}')
        st.toast('Пример сохранён (страница «Дообучение» → «Примеры из диалога»)', icon=':material/check:')
        st.rerun()


@st.dialog('Добавить в тестовый набор', width='large')
def save_task_dialog(ti, cid):
    turn = hist[ti]
    own = [b for b in B.list_benchmarks() if b['type'] in ('open', 'short')]
    if not own:
        st.info('Нет наборов с развёрнутым или коротким ответом — создайте набор на странице «Тестовые наборы».')
        return
    bid = st.selectbox('Набор', [b['id'] for b in own], format_func=lambda x: next(b['name'] for b in own if b['id'] == x))
    b = B.load(bid)
    text = st.text_area('Задание', value=turn['user'], height=120)
    ref = st.text_area('Эталонный ответ' if b.type == 'open' else 'Допустимые ответы (по строке)',
                       value=turn['answers'][cid]['text'], height=200)
    split = st.segmented_control('Выборка', list(B.SPLITS), format_func=B.SPLIT_RU.get, default='eval', required=True)
    if st.button('Добавить', type='primary'):
        it = {'id': B.new_item_id(b.items, 'd'), 'split': split, 'category': 'из диалога'}
        if b.type == 'open':
            it.update(prompt=text, reference=ref, rubric=[{'name': 'Качество ответа', 'max': 10}])
        else:
            it.update(question=text, answers=[x.strip() for x in ref.splitlines() if x.strip()])
        b.items.append(it)
        B.save(b, f"добавлено задание {it['id']} из диалога")
        st.toast(f'Задание добавлено в «{b.name}»', icon=':material/check:')
        st.rerun()


def show_answer(ti, cid, a):
    with st.chat_message('assistant', avatar=':material/smart_toy:'):
        if a.get('error'):
            st.error(a['error'])
            return
        st.markdown(a['text'] or '_(пустой ответ)_')
        if a.get('reasoning'):
            with st.expander('Рассуждения', expanded=False):
                st.text(a['reasoning'])
        speed = f" · {W.fmt_num(a['tok_s'], 1)} ток/с" if a.get('tok_s') == a.get('tok_s') and a.get('tok_s') else ''
        with st.container(horizontal=True, vertical_alignment='center'):
            st.caption(f"{W.fmt_num(a['latency'], 1)} с · {a.get('n_out') or '—'} токенов{speed}")
            if st.button('В обучение', icon=':material/school:', key=f'ex_{ti}_{cid}',
                         help='Сохранить как пример для дообучения', type='tertiary'):
                save_example_dialog(ti, cid)
            if st.button('В набор', icon=':material/playlist_add:', key=f'task_{ti}_{cid}',
                         help='Добавить в тестовый набор', type='tertiary'):
                save_task_dialog(ti, cid)


cols = st.columns(len(sel))
for j, cid in enumerate(sel):
    with cols[j]:
        c = cmap[cid]
        base = f" · исходная: {W.cand_name(c['base_id'])}" if c.get('base_id') else ''
        st.markdown(f'**{W.cand_name(cid)}**')
        st.caption(f"{c['backend']}: {c['model']}{base}")
for ti, turn in enumerate(hist):
    cols = st.columns(len(sel))
    for j, cid in enumerate(sel):
        with cols[j]:
            with st.chat_message('user', avatar=':material/person:'):
                st.markdown(turn['user'])
            if cid in turn['answers']:
                show_answer(ti, cid, turn['answers'][cid])

prompt = st.chat_input('Сообщение моделям' if len(sel) > 1 else 'Сообщение модели')
if prompt:
    hist.append({'user': prompt, 'answers': {}})
    ti = len(hist) - 1
    cols = st.columns(len(sel))
    for j, cid in enumerate(sel):
        with cols[j]:
            with st.chat_message('user', avatar=':material/person:'):
                st.markdown(prompt)
            with st.spinner('Модель отвечает…'):
                try:
                    rep = W.ask(cid, messages_for(cid, ti) + [{'role': 'user', 'content': prompt}],
                                int(max_tokens), float(temperature))
                    a = dict(text=rep.text, reasoning=rep.reasoning, latency=rep.latency, tok_s=rep.tok_s,
                             n_out=rep.n_out, ts=time.time())
                except Exception as e:  # noqa: BLE001
                    a = dict(error=f'Ошибка: {e}', text='', latency=0)
            hist[ti]['answers'][cid] = a
    st.rerun()
if not hist:
    st.caption('Один и тот же вопрос отправляется всем выбранным моделям; история диалога у каждой своя. '
               'Удачный ответ можно сохранить как пример для дообучения («В обучение») или как задание '
               'тестового набора («В набор»).')
