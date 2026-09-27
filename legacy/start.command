#!/bin/bash
# Запуск: bash start.command   (или двойной щелчок после chmod +x start.command)
cd "$(dirname "$0")"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Нужен Python 3.10+. Установите: brew install python@3.12"; read -n 1; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю окружение и ставлю зависимости (несколько минут)…"
  "$PY" -m venv .venv && ./.venv/bin/pip install -q --upgrade pip && ./.venv/bin/pip install -q -r requirements.txt || { echo "Ошибка установки"; read -n 1; exit 1; }
fi
exec ./.venv/bin/python -m streamlit run app.py --browser.gatherUsageStats false
