"""Общие элементы интерфейса: рисунки и таблицы с сохранением для Word, выбор моделей и наборов."""
import io
import json
import math
import time

import numpy as np
import pandas as pd
import streamlit as st

from core import bench as B
from core import charts as CH
from core import config as C
from core import db
from core import fuzzy as F

CAPTIONS = C.WORK / 'captions.json'


# ---------------- настройки ----------------
def settings():
    return C.load_settings()


def update_settings(**kw):
    s = C.load_settings()
    s.update(kw)
    C.save_settings(s)


def chart_settings():
    d = dict(font='Times New Roman', size=12.0, width_cm=16.0, bw=False, decimal_comma=True, start=1)
    d.update(settings().get('charts') or {})
    return d


def chart_style(**kw):
    s = chart_settings()
    st_ = CH.Style(font=s['font'], size=float(s['size']), width_cm=float(s['width_cm']), bw=bool(s['bw']),
                   decimal_comma=bool(s['decimal_comma']))
    return CH.with_style(st_, **kw)


def fmt_num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    s = f'{x:.{digits}f}'
    return s.replace('.', ',') if chart_settings()['decimal_comma'] else s


# ---------------- подписи рисунков и таблиц (сохраняются между запусками) ----------------
def _captions():
    return C.read_json(CAPTIONS, {}) or {}


def caption_for(key, default_caption, kind='fig'):
    caps = _captions()
    if key not in caps:
        seq = caps.get(f'__next_{kind}', int(chart_settings().get('start', 1)))
        caps[key] = dict(caption=default_caption, number=str(seq))
        caps[f'__next_{kind}'] = seq + 1
        C.write_json(CAPTIONS, caps)
    return caps[key]


def save_caption(key, caption, number):
    caps = _captions()
    caps[key] = dict(caption=caption, number=number)
    C.write_json(CAPTIONS, caps)


# ---------------- рисунки ----------------
def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def _build(kind, kwargs, style):
    if kind == 'terms':
        return CH.fuzzy_terms(style, **kwargs)
    if kind == 'rule':
        m = F.model_from_dict(kwargs['model'])
        return CH.rule_clipped(style, m, kwargs['raw'], kwargs.get('k', 2))
    return {'bars': CH.bars, 'heatmap': CH.heatmap, 'lines': CH.lines, 'scatter': CH.scatter, 'boxes': CH.boxes,
            'radar': CH.radar, 'training': CH.training_curve}[kind](style, **kwargs)


def _style_json(style):
    return json.dumps(style.__dict__, ensure_ascii=False, sort_keys=True)


@st.cache_data(max_entries=400, show_spinner=False)
def render(kind, kwargs_json, style_json, fmt='png', dpi=120):
    style = CH.Style(**json.loads(style_json))
    fig = _build(kind, json.loads(kwargs_json), style)
    return CH.to_bytes(fig, fmt, dpi)


def figure(key, kind, kwargs, default_caption, filename=None):
    """Рисунок в оформлении для статьи с меню сохранения (PNG 300 dpi, SVG, PDF, Word)."""
    cap = caption_for(key, default_caption)
    ks = f'fig_{key}'
    st.session_state.setdefault(f'{ks}_cap', cap['caption'])
    st.session_state.setdefault(f'{ks}_num', cap['number'])
    base = chart_settings()
    st.session_state.setdefault(f'{ks}_w', float(base['width_cm']))
    st.session_state.setdefault(f'{ks}_bw', bool(base['bw']))
    st.session_state.setdefault(f'{ks}_wc', True)
    style = chart_style(caption=st.session_state[f'{ks}_cap'], number=st.session_state[f'{ks}_num'],
                        width_cm=float(st.session_state[f'{ks}_w']), bw=st.session_state[f'{ks}_bw'],
                        with_caption=st.session_state[f'{ks}_wc'])
    kj = json.dumps(kwargs, default=_jsonable, ensure_ascii=False)
    fname = filename or f'рис_{st.session_state[f"{ks}_num"]}_{key}'
    with st.container(border=True):
        try:
            png = render(kind, kj, _style_json(style), 'png', 110)
        except Exception as e:  # noqa: BLE001
            st.warning(f'Не удалось построить рисунок: {e}')
            return
        st.image(png, width='content')
        with st.popover('Сохранить рисунок', icon=':material/download:'):
            st.text_area('Подпись', key=f'{ks}_cap', height=80,
                         on_change=lambda: save_caption(key, st.session_state[f'{ks}_cap'], st.session_state[f'{ks}_num']))
            with st.container(horizontal=True):
                st.text_input('Номер', key=f'{ks}_num', width=110,
                              on_change=lambda: save_caption(key, st.session_state[f'{ks}_cap'],
                                                             st.session_state[f'{ks}_num']))
                st.number_input('Ширина, см', 6.0, 30.0, step=0.5, key=f'{ks}_w', width=140)
            with st.container(horizontal=True):
                st.toggle('Чёрно-белый', key=f'{ks}_bw')
                st.toggle('Подпись на рисунке', key=f'{ks}_wc',
                          help='Для Word удобнее без неё: в файле .docx подпись добавляется отдельным абзацем.')
            st.caption(f'Шрифт {style.font}, {style.size:g} пт. PNG вставляется в Word в натуральную величину.')
            sj = _style_json(style)
            sj_nocap = _style_json(CH.with_style(style, with_caption=False))
            with st.container(horizontal=True):
                st.download_button('PNG 300 dpi', lambda: render(kind, kj, sj, 'png', 300), f'{fname}.png',
                                   'image/png', on_click='ignore', key=f'{ks}_png', icon=':material/image:')
                st.download_button('SVG', lambda: render(kind, kj, sj, 'svg', 72), f'{fname}.svg', 'image/svg+xml',
                                   on_click='ignore', key=f'{ks}_svg')
                st.download_button('PDF', lambda: render(kind, kj, sj, 'pdf', 72), f'{fname}.pdf', 'application/pdf',
                                   on_click='ignore', key=f'{ks}_pdf')
                st.download_button('Word', lambda: CH.figure_docx(render(kind, kj, sj_nocap, 'png', 300),
                                                                  style.width_cm, style.caption_text()),
                                   f'{fname}.docx', on_click='ignore', key=f'{ks}_docx', icon=':material/description:',
                                   mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')


def figure_png(kind, kwargs, style):
    """PNG 300 dpi без подписи — для отчёта в Word."""
    kj = json.dumps(kwargs, default=_jsonable, ensure_ascii=False)
    return render(kind, kj, _style_json(CH.with_style(style, with_caption=False)), 'png', 300)


# ---------------- таблицы ----------------
def formatted(df: pd.DataFrame, digits=2):
    """Значения как в Word: десятичная запятая, «—» вместо пропусков."""
    out = df.copy()
    for c in out.columns:
        col = out[c]
        dg = digits.get(c, digits.get('*', 2)) if isinstance(digits, dict) else digits
        if col.dtype.kind in 'fc':
            out[c] = [fmt_num(v, dg) for v in col]
        elif col.dtype.kind == 'b':
            out[c] = ['да' if v else '' for v in col]
        else:
            out[c] = ['—' if v is None or (isinstance(v, float) and math.isnan(v)) else
                      (fmt_num(v, dg) if isinstance(v, float) else str(v)) for v in col]
    return out


def static_table(df: pd.DataFrame, digits=2, max_rows_visible=25):
    """Чёткая HTML-таблица (без размытия при масштабировании страницы)."""
    if df is None or len(df) == 0:
        st.caption('Нет данных.')
        return
    h = 'content' if len(df) <= max_rows_visible else 36 * max_rows_visible + 40
    st.table(formatted(df, digits), hide_index=True, border='horizontal', height=h)


def table(df: pd.DataFrame, key, default_caption, digits=2, column_config=None, **kw):
    """Таблица с сохранением в Word (подпись «Таблица N – …»), Excel и CSV."""
    static_table(df, digits)
    cap = caption_for(f't_{key}', default_caption, 'tab')
    ks = f'tab_{key}'
    st.session_state.setdefault(f'{ks}_cap', cap['caption'])
    st.session_state.setdefault(f'{ks}_num', cap['number'])
    with st.popover('Сохранить таблицу', icon=':material/download:'):
        st.text_area('Название', key=f'{ks}_cap', height=70,
                     on_change=lambda: save_caption(f't_{key}', st.session_state[f'{ks}_cap'], st.session_state[f'{ks}_num']))
        st.text_input('Номер', key=f'{ks}_num', width=110,
                      on_change=lambda: save_caption(f't_{key}', st.session_state[f'{ks}_cap'], st.session_state[f'{ks}_num']))
        caption = table_caption(st.session_state[f'{ks}_num'], st.session_state[f'{ks}_cap'])
        with st.container(horizontal=True):
            st.download_button('Word', lambda: CH.table_docx(df, caption, digits, chart_settings()['decimal_comma']),
                               f'таблица_{st.session_state[f"{ks}_num"]}_{key}.docx', on_click='ignore',
                               key=f'{ks}_docx', icon=':material/description:',
                               mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            st.download_button('Excel', lambda: to_xlsx(df), f'{key}.xlsx', on_click='ignore', key=f'{ks}_xlsx',
                               icon=':material/table_view:',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            st.download_button('CSV', lambda: df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'),
                               f'{key}.csv', 'text/csv', on_click='ignore', key=f'{ks}_csv')


def table_caption(number, caption):
    return f'Таблица {number} – {caption}' if str(number).strip() else caption


def to_xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ---------------- выбор моделей и наборов ----------------
def cand_name(cid):
    c = db.candidate(cid)
    return db.display_name(c) if c else cid


@st.cache_data(ttl=3, show_spinner=False)
def candidates_cached(include_hidden=False):
    return db.candidates(include_hidden)


def cand_map():
    return {c['id']: c for c in candidates_cached()}


def benches_cached():
    return B.list_benchmarks()


def bench_names():
    return {b['id']: b['name'] for b in benches_cached()}


@st.cache_data(ttl=10, show_spinner=False)
def ping(backend_cfg_json):
    from core import backends as BK
    ok, err = BK.make(json.loads(backend_cfg_json)).ping()
    return ok, err


@st.cache_data(ttl=10, show_spinner=False)
def list_models(backend_cfg_json):
    from core import backends as BK
    return BK.make(json.loads(backend_cfg_json)).list_models()


def data_version():
    """Меняется при появлении новых ответов/оценок — ключ для кэша расчётов."""
    r = db.one('SELECT (SELECT COUNT(*) FROM responses) n, (SELECT MAX(ts) FROM responses) t, '
               '(SELECT COUNT(*) FROM grades) g, (SELECT MAX(ts) FROM grades) gt, '
               '(SELECT COUNT(*) FROM manual_metrics) mm')
    bench_mtime = max([p.stat().st_mtime for p in C.BENCH_DIR.glob('*/items.jsonl')] +
                      [p.stat().st_mtime for p in C.BENCH_DIR.glob('*/benchmark.json')] + [0])
    return f"{r['n']}|{r['t']}|{r['g']}|{r['gt']}|{r['mm']}|{bench_mtime}"


def ago(ts):
    if not ts:
        return '—'
    return time.strftime('%d.%m %H:%M', time.localtime(ts))


def duration(a, b):
    if not a:
        return '—'
    s = int((b or time.time()) - a)
    return f'{s // 3600} ч {s % 3600 // 60} мин' if s >= 3600 else f'{s // 60} мин {s % 60} с'


STATE_RU = {'queued': 'в очереди', 'running': 'идёт', 'stopping': 'останавливается', 'done': 'завершён',
            'stopped': 'остановлен', 'error': 'ошибка'}


# ---------------- разовые запросы к модели (песочница, диалог) ----------------
def backend_for(cand):
    from core import backends as BK
    return BK.by_id(settings(), cand['backend'])


def ensure_loaded(cand):
    """Загружает модель в LM Studio при необходимости; возвращает handle для chat()."""
    be = backend_for(cand)
    handles = st.session_state.setdefault('_handles', {})
    if cand['id'] in handles:
        return be, handles[cand['id']]
    handle = cand['model']
    if be.can_load:
        info = next((m for m in be.list_models() if m['key'] == cand['model']), None)
        if info is None:
            raise RuntimeError(f"модель {cand['model']} не найдена на сервере")
        if info.get('loaded'):
            handle = info['loaded'][0]
        else:
            with st.spinner(f"Загрузка модели {cand['model']}…"):
                handle = be.load(cand['model'], settings()['context_length'])
            st.session_state.setdefault('_loaded_by_ui', {})[cand['id']] = handle
    handles[cand['id']] = handle
    return be, handle


def unload_ui_models():
    loaded = st.session_state.get('_loaded_by_ui', {})
    for cid, h in list(loaded.items()):
        c = db.candidate(cid)
        if c:
            try:
                backend_for(c).unload(h)
            except Exception:  # noqa: BLE001
                pass
    st.session_state['_loaded_by_ui'] = {}
    st.session_state['_handles'] = {}


def ask(cid, messages, max_tokens=512, temperature=0.0, meta=None):
    c = db.candidate(cid)
    be, h = ensure_loaded(c)
    rs = c.get('reasoning')
    return be.chat(h, messages, temperature=temperature, max_tokens=max_tokens,
                   reasoning=None if rs in (None, '', 'default') else rs, meta=meta)
