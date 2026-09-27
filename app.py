"""Стенд оценки языковых моделей — запуск: ./start.command (или streamlit run app.py)."""
import streamlit as st

from core import config as C  # noqa: F401 — выставляет офлайн-переменные до импорта остальных модулей
from core import db, jobs
from ui import widgets as W

st.set_page_config(page_title='Стенд оценки языковых моделей', page_icon=':material/analytics:', layout='wide')

P = lambda path, title, icon: st.Page(path, title=title, icon=f':material/{icon}:')  # noqa: E731
page = st.navigation({
    '': [P('app_pages/home.py', 'Главная', 'home')],
    'Подготовка': [P('app_pages/models.py', 'Модели', 'memory'),
                   P('app_pages/benchmarks.py', 'Тестовые наборы', 'checklist'),
                   P('app_pages/fuzzy.py', 'Нечёткая модель', 'account_tree')],
    'Испытания': [P('app_pages/run.py', 'Прогон', 'play_circle'),
                  P('app_pages/results.py', 'Результаты', 'bar_chart'),
                  P('app_pages/evaluate.py', 'Сводная оценка', 'leaderboard'),
                  P('app_pages/chat.py', 'Диалог', 'forum')],
    'Дообучение': [P('app_pages/training.py', 'Дообучение', 'model_training')],
    'Сервис': [P('app_pages/system.py', 'Система', 'settings')],
})


@st.fragment(run_every=3)
def run_status():
    r = jobs.active_run()
    if r:
        p = r.get('progress') or {}
        n, i = p.get('n') or 0, p.get('i') or 0
        st.caption(f"Идёт прогон №{r['id']}: {W.cand_name(p['candidate']) if p.get('candidate') else ''}")
        st.progress(min(1.0, i / n) if n else 0.0, text=p.get('stage') or '')


with st.sidebar:
    run_status()

st.title(page.title)
page.run()
