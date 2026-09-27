"""Сведения о системе и самопроверка офлайн-режима."""
import importlib.util
import os
import platform
import re
import shutil
import subprocess

from core import config as C


def used_memory_gb():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 2 ** 30
    except ImportError:
        pass
    try:   # macOS без psutil
        out = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r'page size of (\d+)', out).group(1))
        get = lambda k: int(re.search(rf'{k}:\s+(\d+)', out).group(1))  # noqa: E731
        return (get('Pages active') + get('Pages wired down') + get('Pages occupied by compressor')) * page / 2 ** 30
    except Exception:  # noqa: BLE001
        return float('nan')


def total_memory_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 2 ** 30
    except ImportError:
        try:
            return int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout) / 2 ** 30
        except Exception:  # noqa: BLE001
            return float('nan')


def machine():
    chip = platform.processor()
    try:
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True,
                              timeout=3).stdout.strip() or chip
    except Exception:  # noqa: BLE001
        pass
    return dict(os=f'{platform.system()} {platform.mac_ver()[0] or platform.release()}', chip=chip,
                python=platform.python_version(), memory_gb=round(total_memory_gb(), 1))


def has(mod):
    return importlib.util.find_spec(mod) is not None


def offline_checks():
    """Список (проверка, ok, пояснение)."""
    from core import toxicity
    out = []
    for k, v in C.OFFLINE_ENV.items():
        if k.endswith('OFFLINE'):
            out.append((f'Запрет сетевых обращений библиотек ({k})', os.environ.get(k) == '1', os.environ.get(k, 'не задано')))
    wheels = list(C.WHEELS.glob('*.whl')) + list(C.WHEELS.glob('*.tar.gz'))
    out.append(('Пакеты для установки без сети (vendor/wheels)', len(wheels) > 20, f'{len(wheels)} файлов'))
    ok, where = toxicity.available()
    vend = (C.VENDOR_MODELS / C.TOX_MODEL_NAME / 'config.json').exists()
    out.append(('Классификатор токсичности в vendor/models', vend,
                where if vend else ('найден только в кэше HF: ' + where if ok else where)))
    for mod, name in (('matplotlib', 'Построение графиков'), ('docx', 'Выгрузка в Word'),
                      ('torch', 'Классификатор токсичности (torch)'), ('mlx_lm', 'Дообучение (mlx-lm)')):
        out.append((name, has(mod), 'установлено' if has(mod) else 'не установлено'))
    free = shutil.disk_usage(C.ROOT).free / 2 ** 30
    out.append(('Свободное место на диске', free > 20, f'{free:.0f} ГБ'))
    return out
