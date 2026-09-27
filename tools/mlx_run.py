"""Запуск команд mlx_lm (lora, fuse, server, generate) с совместимостью весов моделей из LM Studio.

    python -m tools.mlx_run lora --model <папка> --train ...

В сборках LM Studio для Gemma 4 (E2B/E4B) есть k/v-проекции слоёв с общим KV-кэшем, которые модель mlx-lm
не использует (has_kv = False), — строгая загрузка падает с «parameters not in model». Здесь такие лишние
веса пропускаются (с записью в журнал); недостающие веса по-прежнему считаются ошибкой.

Кроме того, распределённая инициализация MLX по умолчанию ищет libmpi.dylib и находит MPICH из Anaconda
(MLX нужен Open MPI) — процесс аварийно завершается. Обучение идёт на одной машине, поэтому используется
backend «ring» (без MPI): группа из одного процесса."""
import os
import sys

import mlx.core as mx
import mlx_lm.utils as U

_init = mx.distributed.init


def _single_host_init(strict=False, backend='any'):
    if backend == 'any' and not os.environ.get('MLX_HOSTFILE'):
        backend = 'ring'
    return _init(strict=strict, backend=backend)


mx.distributed.init = _single_host_init

if not os.environ.get('MLX_HOSTFILE'):
    # коллективные операции без явной группы внутри MLX снова вызывают init('any') — передаём группу явно
    _group = _init(strict=False, backend='ring')

    def _with_group(fn):
        def wrapped(*args, group=None, **kwargs):
            return fn(*args, group=group if group is not None else _group, **kwargs)
        return wrapped

    for _name in ('all_sum', 'all_gather', 'all_max', 'all_min', 'sum_scatter', 'send', 'recv', 'recv_like'):
        if hasattr(mx.distributed, _name):
            setattr(mx.distributed, _name, _with_group(getattr(mx.distributed, _name)))

_orig = U.load_model


def load_model(model_path, lazy=False, strict=True, *args, **kwargs):
    try:
        return _orig(model_path, lazy, strict, *args, **kwargs)
    except ValueError as e:
        if strict and 'parameters not in model' in str(e):
            n = str(e).split('Received ', 1)[-1].split(' ', 1)[0]
            print(f'[совместимость] пропущено неиспользуемых весов: {n} (например, k/v-проекции слоёв с общим '
                  'KV-кэшем)', flush=True)
            return _orig(model_path, lazy, False, *args, **kwargs)
        raise


U.load_model = load_model

if __name__ == '__main__':
    from mlx_lm import cli
    sys.argv = ['mlx_lm'] + sys.argv[1:]
    cli.main()
