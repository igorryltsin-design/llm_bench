#!/bin/bash
# Стенд оценки языковых моделей. Запуск: двойной щелчок или  bash start.command
# Работает без интернета: пакеты ставятся из vendor/wheels, модели — только локальные.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
PORT="${PORT:-8501}"

PY=""
for c in python3.13 python3.12 python3.11 python3 /opt/anaconda3/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY="$c"; break; fi
done
if [ ! -x .venv/bin/python ]; then
  if [ -z "$PY" ]; then echo "Нужен Python 3.10+ (для vendor/wheels — 3.13)."; read -n 1; exit 1; fi
  echo "Первый запуск: создаю окружение и ставлю пакеты из vendor/wheels (без сети)…"
  "$PY" -m venv .venv || { echo "Не удалось создать окружение"; read -n 1; exit 1; }
  if ls vendor/wheels/*.whl >/dev/null 2>&1; then
    ./.venv/bin/pip install -q --no-index --find-links vendor/wheels -r requirements.txt || { echo "Ошибка установки из vendor/wheels"; read -n 1; exit 1; }
  else
    echo "Папка vendor/wheels пуста — ставлю из интернета."
    ./.venv/bin/pip install -q -r requirements.txt || { echo "Ошибка установки"; read -n 1; exit 1; }
  fi
fi
# первая сборка тестовых наборов из данных прежнего пульта
if [ ! -f benchmarks/mmlu_ru/benchmark.json ] && [ -f legacy/data/qa.jsonl ]; then
  ./.venv/bin/python -m tools.import_external legacy && ./.venv/bin/python -m tools.import_external samples
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Порт $PORT занят (возможно, приложение уже запущено): http://127.0.0.1:$PORT"
  open "http://127.0.0.1:$PORT" 2>/dev/null; exit 0
fi
( sleep 4; open "http://127.0.0.1:$PORT" 2>/dev/null ) &
echo "Приложение: http://127.0.0.1:$PORT  (остановка — Ctrl+C)"
exec ./.venv/bin/python -m streamlit run app.py --server.port "$PORT" --server.address 127.0.0.1 \
     --server.headless true --browser.gatherUsageStats false
