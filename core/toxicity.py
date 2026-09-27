"""Локальный классификатор токсичности s-nlp/russian_toxicity_classifier — только из локальных файлов."""
from pathlib import Path

from core import config as C

_pipe = None


def local_dir():
    """vendor/models/russian_toxicity_classifier, иначе снимок в кэше HF (без обращения к сети)."""
    p = C.VENDOR_MODELS / C.TOX_MODEL_NAME
    if (p / 'config.json').exists():
        return p
    snap = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--s-nlp--russian_toxicity_classifier' / 'snapshots'
    if snap.exists():
        for d in sorted(snap.iterdir(), reverse=True):
            if (d / 'config.json').exists() and (any(d.glob('*.safetensors')) or (d / 'pytorch_model.bin').exists()):
                return d
    return None


def available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False, 'не установлены torch/transformers'
    d = local_dir()
    if d is None:
        return False, f'нет файлов модели в {C.VENDOR_MODELS / C.TOX_MODEL_NAME}'
    return True, str(d)


def classify(texts, batch_size=8):
    """Вероятности токсичности для списка текстов."""
    global _pipe
    if _pipe is None:
        ok, where = available()
        if not ok:
            raise RuntimeError(f'классификатор токсичности недоступен: {where}')
        from transformers import pipeline
        _pipe = pipeline('text-classification', model=where, tokenizer=where, top_k=None, truncation=True)
    res = _pipe([t or ' ' for t in texts], batch_size=batch_size)
    return [{d['label']: d['score'] for d in r}.get('toxic', 0.0) for r in res]
