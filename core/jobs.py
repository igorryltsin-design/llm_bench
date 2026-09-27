"""Запуск и контроль фоновых процессов (прогоны, дообучение, сервер адаптера)."""
import os
import signal
import subprocess
import sys
import time

from core import config as C
from core import db

_procs = {}   # pid → Popen: держим объекты, чтобы прибирать завершившиеся процессы (без «зомби»)


def spawn(args, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, PYTHONUNBUFFERED='1', **C.OFFLINE_ENV)
    e.update(env or {})
    p = subprocess.Popen(args, cwd=C.ROOT, env=e, stdout=open(log_path, 'a', encoding='utf-8'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    _procs[p.pid] = p
    return p


def alive(pid):
    if not pid:
        return False
    p = _procs.get(pid)
    if p is not None:
        return p.poll() is None
    try:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False          # наш завершившийся потомок — прибрали
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def free_port(start, tries=100):
    """Первый свободный порт на 127.0.0.1, начиная со start."""
    import socket
    for port in range(int(start), int(start) + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            try:
                sk.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f'нет свободных портов начиная с {start}')


def run_log(run_id):
    return C.LOG_DIR / f'run_{run_id}.log'


def launch_run(run_id):
    p = spawn([sys.executable, '-u', '-m', 'core.runner', str(run_id)], run_log(run_id))
    db.update_run(run_id, pid=p.pid, state='running', started=time.time())
    return p.pid


def stop(pid, hard=False):
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGKILL if hard else signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL if hard else signal.SIGTERM)
        except OSError:
            pass


def active_run():
    """Идущий прогон (с живым процессом). Зависшие записи помечаются как прерванные."""
    for r in db.runs(20):
        if r['state'] in ('running', 'queued', 'stopping'):
            if alive(r['pid']):
                return r
            if r['state'] != 'queued' or time.time() - (r['created'] or 0) > 30:
                db.update_run(r['id'], state='error' if r['state'] == 'running' else 'stopped',
                              error=r.get('error') or 'процесс прогона завершился аварийно (см. журнал)',
                              finished=time.time())
    return None


def tail(path, n=40):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return ''.join(f.readlines()[-n:])
    except OSError:
        return ''
