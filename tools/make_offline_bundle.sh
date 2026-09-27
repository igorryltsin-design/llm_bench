#!/bin/bash
# Сборка офлайн-комплекта (запускать ОДИН РАЗ на машине с интернетом, той же ОС и архитектуры — macOS arm64).
#   bash tools/make_offline_bundle.sh
# Результат: vendor/wheels (пакеты Python), vendor/models (классификатор токсичности),
#            vendor/datasets (исходные файлы MMLU-ru и RuBia). После этого папку можно переносить без сети.
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
[ -x "$PY" ] || PY=python3
echo "1/4 Пакеты Python → vendor/wheels"
"$PY" -m pip download -q -d vendor/wheels -r requirements.txt
echo "2/4 Классификатор токсичности → vendor/models/russian_toxicity_classifier"
HF_HUB_OFFLINE=0 "$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('s-nlp/russian_toxicity_classifier', local_dir='vendor/models/russian_toxicity_classifier',
                  allow_patterns=['*.json', '*.txt', '*.safetensors'])
PY
echo "3/4 Исходные файлы наборов → vendor/datasets"
HF_HUB_OFFLINE=0 "$PY" -m tools.import_external fetch
echo "4/4 Список версий пакетов → requirements.lock"
"$PY" -m pip freeze > requirements.lock 2>/dev/null || true
du -sh vendor/* 2>/dev/null
echo "Готово. Комплект можно переносить на машину без интернета."
