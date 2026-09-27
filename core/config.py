"""Пути, настройки и офлайн-режим. Импортируется первым: выставляет переменные окружения,
запрещающие библиотекам HF ходить в сеть."""
import json
import os
from pathlib import Path

APP_VERSION = '2.0.0'

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / 'benchmarks'
CONFIG_DIR = ROOT / 'config'
FUZZY_DIR = CONFIG_DIR / 'fuzzy'
PROFILE_DIR = CONFIG_DIR / 'profiles'
WORK = ROOT / 'workspace'
DB_PATH = WORK / 'bench.db'
LOG_DIR = WORK / 'logs'
EXPORT_DIR = WORK / 'exports'
TRAIN_DIR = WORK / 'training'
SETTINGS_PATH = WORK / 'settings.json'
VENDOR = ROOT / 'vendor'
VENDOR_MODELS = VENDOR / 'models'
VENDOR_DATASETS = VENDOR / 'datasets'
WHEELS = VENDOR / 'wheels'
LEGACY = ROOT / 'legacy'

TOX_MODEL_NAME = 'russian_toxicity_classifier'   # s-nlp/russian_toxicity_classifier

OFFLINE_ENV = {
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1', 'TOKENIZERS_PARALLELISM': 'false',
}
for _k, _v in OFFLINE_ENV.items():
    os.environ.setdefault(_k, _v)

for _d in (WORK, LOG_DIR, EXPORT_DIR, TRAIN_DIR, BENCH_DIR, FUZZY_DIR, PROFILE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {
    'backends': [
        {'id': 'lmstudio', 'kind': 'lmstudio', 'name': 'LM Studio', 'url': 'http://127.0.0.1:1234'},
        {'id': 'demo', 'kind': 'demo', 'name': 'Демо (имитация моделей)', 'url': ''},
    ],
    'platform_gb': 24.0,          # память стенда для показателя «доля ресурсов»
    'context_length': 4096,
    'unload_after': True,          # выгружать модель, загруженную прогоном
    'lmstudio_models_dir': str(Path.home() / '.lmstudio' / 'models'),
    'expert': '',                  # имя эксперта по умолчанию для ручных оценок
}


def load_settings() -> dict:
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        s.update(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        pass
    return s


def save_settings(s: dict) -> None:
    tmp = SETTINGS_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(SETTINGS_PATH)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
