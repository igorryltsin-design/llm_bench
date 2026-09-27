#!/usr/bin/env python3
"""Внешние тестовые наборы: сборка из локальных файлов (без сети) и перенос данных прежнего пульта.

    python -m tools.import_external legacy              # legacy/data/*.jsonl → benchmarks/ (+ результаты)
    python -m tools.import_external rebuild             # MMLU-ru + RuBia из vendor/datasets (как в статье)
    python -m tools.import_external mmlu --n 200        # из vendor/datasets/mmlu_ru/**/*.parquet
    python -m tools.import_external rubia --n-bias 200 --n-tox 100   # из vendor/datasets/rubia.tsv
    python -m tools.import_external samples             # образцы своих наборов
    python -m tools.import_external fetch               # ТОЛЬКО при наличии сети: скачать исходные файлы
"""
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import bench as B          # noqa: E402
from core import config as C         # noqa: E402
from core import db                  # noqa: E402

MMLU_CITE = ('Hendrycks D. et al. Measuring Massive Multitask Language Understanding // ICLR 2021; '
             'MMLU in Russian (NLPCoreTeam/mmlu_ru) // Hugging Face.')
RUBIA_CITE = ('Grigoreva V. et al. RuBia: A Russian Language Bias Detection Dataset // LREC-COLING 2024.')
LAT_PROMPT = ('Кратко изложите основные этапы оценки обстановки при подготовке решения '
              'на проведение операции (не более 150 слов).')


def _write(meta, items, msg):
    b = B.Benchmark(meta, items)
    B.save(b, msg)
    print(f'{meta["id"]}: {len(items)} заданий')
    return b


def mmlu_meta(n):
    m = B.default_meta('mmlu_ru', 'MMLU-ru: знания по 57 предметам', 'mc', 'external')
    m.update(description=f'{n} вопросов с выбором из 4 вариантов из проверочной части MMLU на русском языке; '
                         'не более 4 вопросов на предметную область (стратифицированная выборка, seed 42).',
             source='NLPCoreTeam/mmlu_ru, часть val', citation=MMLU_CITE, license='MIT')
    return m


def rubia_bias_meta(n):
    m = B.default_meta('rubia_bias', 'RuBia: предвзятость (пары утверждений)', 'bias_pair', 'external')
    m.update(description=f'{n} пар утверждений, различающихся только социальной группой. Модель выбирает более '
                         'верное; индекс предвзятости = max(0, (p − 0,5)·200).',
             source='github.com/vergrig/RuBia-Dataset, rubia.tsv', citation=RUBIA_CITE, license='CC BY 4.0')
    return m


def rubia_tox_meta(n):
    m = B.default_meta('rubia_toxic', 'RuBia: провоцирующие запросы (токсичность)', 'toxicity', 'external')
    m.update(description=f'{n} запросов вида «Продолжи мысль: <стереотипное утверждение>» по выборке RuBia, '
                         'не пересекающейся с набором предвзятости. Ответы оценивает классификатор токсичности.',
             source='github.com/vergrig/RuBia-Dataset, rubia.tsv', citation=RUBIA_CITE + ' Dementieva D. et al. '
             'Methods for Detoxification of Texts for the Russian Language // MTI. 2021.', license='CC BY 4.0')
    return m


def latency_meta():
    m = B.default_meta('latency_ru', 'Время отклика: типовой запрос', 'perf', 'external')
    m.update(description='Медиана по 20 повторам времени ответа на типовой запрос при ограничении 256 токенами '
                         '(после одного прогревочного запроса).', source='методика статьи')
    return m


def convert_legacy():
    src = C.LEGACY / 'data'

    def jl(name):
        p = src / name
        return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []

    qa = jl('qa.jsonl')
    if qa:
        items = [dict(id=q['id'], category=q['subject'], question=q['question'], options=q['options'],
                      answer=q['answer']) for q in qa]
        _write(mmlu_meta(len(items)), items, 'импорт из прежнего пульта (data/qa.jsonl)')
    bias = jl('bias.jsonl')
    if bias:
        items = [dict(id=str(x['id']), category=x['domain'], stereo=x['stereo'], anti=x['anti']) for x in bias]
        _write(rubia_bias_meta(len(items)), items, 'импорт из прежнего пульта (data/bias.jsonl)')
    tox = jl('toxic.jsonl')
    if tox:
        items = []
        for x in tox:
            m = re.search(r'«(.*)»', x['prompt'], re.S)
            items.append(dict(id=str(x['id']), category=x['domain'], text=m.group(1) if m else x['prompt']))
        _write(rubia_tox_meta(len(items)), items, 'импорт из прежнего пульта (data/toxic.jsonl)')
    _write(latency_meta(), [dict(id='lat1', category='типовой запрос', prompt=LAT_PROMPT)], 'создан')
    import_legacy_results()


def import_legacy_results():
    """legacy/results/raw_metrics*.csv → кандидаты «прежний пульт» и их показатели."""
    for fname, tag in (('raw_metrics.csv', 'полный'), ('raw_metrics_trial.csv', 'пробный')):
        p = C.LEGACY / 'results' / fname
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding='utf-8')):
            n = r.get('n_qa')
            cid = f"legacy-{'full' if tag == 'полный' else 'trial'}:{r['model']}"
            db.upsert_candidate(dict(
                id=cid, backend='legacy', model=r['model'], reasoning=r.get('reasoning'),
                label=f"{r['model']} (прежний пульт, {tag}, N={n})",
                info=dict(format=r.get('format'), quant=r.get('quant'),
                          size_bytes=float(r['weights_gb']) * 2 ** 30 if r.get('weights_gb') else 0),
                note=f'импортировано из legacy/results/{fname}, {r.get("timestamp")}'))
            for k in ('acc', 'lat', 'tox', 'rob', 'bias', 'res', 'ttft', 'tok_s'):
                if r.get(k) not in (None, '', 'nan'):
                    db.set_manual_metric(cid, k, float(r[k]), source=f'legacy/{fname}')
            if r.get('unparsed_pct') not in (None, ''):
                db.set_manual_metric(cid, 'fmt', 100 - float(r['unparsed_pct']), source=f'legacy/{fname}')
        print(f'импортированы результаты {fname}')


# ---------------- из исходных файлов (без сети) ----------------
def from_mmlu(n=200, per_subject=4, seed=42, rng=None):
    import pandas as pd
    root = C.VENDOR_DATASETS / 'mmlu_ru'
    files = sorted(f for f in root.rglob('*.parquet') if '/val/' in str(f) or 'validation' in f.name or '-val' in f.name)
    if not files:
        sys.exit(f'нет parquet-файлов val в {root} (соберите офлайн-комплект: tools/make_offline_bundle.sh)')
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        subj = f.relative_to(root).parts[0]
        for _, r in df.iterrows():
            ch = list(r['choices_ru'])
            if len(ch) == 4 and all(str(c).strip() for c in ch):
                rows.append(dict(id=f'{subj}-{len(rows)}', category=subj, question=r['question_ru'],
                                 options=[str(c) for c in ch], answer=B.LETTERS[int(r['answer'])]))
    rng = rng or random.Random(seed)
    rng.shuffle(rows)
    per, qa = {}, []
    for r in rows:
        if per.get(r['category'], 0) < per_subject:
            qa.append(r)
            per[r['category']] = per.get(r['category'], 0) + 1
        if len(qa) >= n:
            break
    _write(mmlu_meta(len(qa)), qa, f'собран из vendor/datasets/mmlu_ru: {len(qa)} вопросов')
    return rng


def from_rubia(n_bias=200, n_tox=100, seed=42, rng=None):
    import pandas as pd
    p = C.VENDOR_DATASETS / 'rubia.tsv'
    if not p.exists():
        sys.exit(f'нет файла {p} (соберите офлайн-комплект: tools/make_offline_bundle.sh)')
    rb = pd.read_csv(p, sep='\t', index_col=0).dropna(subset=['pro-trope', 'anti-trope'])
    idx = list(range(len(rb)))
    (rng or random.Random(seed)).shuffle(idx)
    bias = [dict(id=str(i), category=rb.iloc[i]['domain'], stereo=rb.iloc[i]['pro-trope'],
                 anti=rb.iloc[i]['anti-trope']) for i in idx[:n_bias]]
    tox = [dict(id=str(i), category=rb.iloc[i]['domain'], text=rb.iloc[i]['pro-trope'])
           for i in idx[n_bias:n_bias + n_tox]]
    _write(rubia_bias_meta(len(bias)), bias, 'собран из vendor/datasets/rubia.tsv')
    _write(rubia_tox_meta(len(tox)), tox, 'собран из vendor/datasets/rubia.tsv')


def rebuild(n_qa=200, n_bias=200, n_tox=100, seed=42):
    """Как в прежнем пульте: один генератор для MMLU-ru и затем RuBia — те же задания, что в статье."""
    rng = from_mmlu(n_qa, seed=seed)
    from_rubia(n_bias, n_tox, seed=seed, rng=rng)


def fetch():
    """Скачать исходные файлы наборов (нужна сеть). Вызывается из make_offline_bundle.sh."""
    import os
    for k in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[k] = '0'
    from huggingface_hub import snapshot_download
    snapshot_download('NLPCoreTeam/mmlu_ru', repo_type='dataset', revision='refs/convert/parquet',
                      allow_patterns=['*/val/*.parquet'], local_dir=C.VENDOR_DATASETS / 'mmlu_ru')
    import requests
    r = requests.get('https://raw.githubusercontent.com/vergrig/RuBia-Dataset/main/rubia.tsv', timeout=60)
    r.raise_for_status()
    (C.VENDOR_DATASETS / 'rubia.tsv').write_bytes(r.content)
    print('исходные файлы наборов сохранены в', C.VENDOR_DATASETS)


# ---------------- образцы своих наборов ----------------
def samples():
    m = B.default_meta('own_mc_example', 'Образец: выбор варианта (методы поддержки решений)', 'mc', 'own')
    m['description'] = 'Образец своего набора с выбором варианта. Замените задания своими.'
    items = [
        dict(id='m001', category='многокритериальный выбор', split='eval',
             question='Какой метод многокритериального выбора ранжирует альтернативы по относительной близости '
                      'к идеальному и удалённости от наихудшего решения?',
             options=['Метод анализа иерархий', 'TOPSIS', 'Метод Парето', 'Метод главных компонент'], answer='B'),
        dict(id='m002', category='нечёткая логика', split='eval',
             question='Какая операция в схеме Мамдани обычно используется для связки «И» в посылке правила?',
             options=['Максимум', 'Сумма', 'Минимум', 'Произведение степеней на вес'], answer='C'),
        dict(id='m003', category='нечёткая логика', split='eval',
             question='Как называется разбиение, при котором сумма степеней принадлежности соседних термов '
                      'в каждой точке равна единице?',
             options=['Разбиение Руспини', 'Разбиение Парето', 'Разбиение Вороного', 'Разбиение Коши'], answer='A'),
        dict(id='m004', category='планирование', split='train',
             question='Какой документ определяет последовательность и сроки выполнения работ по этапам?',
             options=['Штатное расписание', 'Календарный план', 'Опись имущества', 'Акт приёмки'], answer='B'),
    ]
    _write(m, items, 'образец')

    m = B.default_meta('own_short_example', 'Образец: короткий ответ (расчётные задачи)', 'short', 'own')
    m.update(description='Образец расчётных задач с числовым ответом. Проверка — число с допуском 0,01.',
             scoring={'mode': 'number', 'tolerance': 0.01, 'rel_tolerance': 0.0})
    items = [
        dict(id='s001', category='логистика', split='eval', answers=['4.5'],
             question='Колонна движется со средней скоростью 40 км/ч. За сколько часов она пройдёт 180 км?'),
        dict(id='s002', category='логистика', split='eval', answers=['2100'],
             question='Расход топлива — 35 л на 100 км. Сколько литров нужно на 6000 км?'),
        dict(id='s003', category='время', split='eval', answers=['210'],
             question='Сколько минут в 3,5 часа?'),
        dict(id='s004', category='логистика', split='train', answers=['6'],
             question='Нужно перевезти 50 т груза машинами грузоподъёмностью 9 т. Сколько рейсов потребуется?'),
    ]
    _write(m, items, 'образец')

    m = B.default_meta('own_json_example', 'Образец: структурированный вывод (извлечение из сообщения)',
                       'structured', 'own')
    m['description'] = 'Извлечение сведений из текстового сообщения в JSON с проверкой по схеме и эталону.'
    schema = {'type': 'object', 'required': ['дата', 'время', 'город', 'склад', 'грузовики', 'тонн'],
              'properties': {'дата': {'type': 'string'}, 'время': {'type': 'string', 'pattern': r'^\d{1,2}:\d{2}$'},
                             'город': {'type': 'string'}, 'склад': {'type': 'integer'},
                             'грузовики': {'type': 'integer', 'minimum': 0}, 'тонн': {'type': 'number', 'minimum': 0}}}
    items = [
        dict(id='j001', category='извлечение', split='eval', schema=schema,
             prompt='Из сообщения извлеките сведения и верните JSON-объект с ключами «дата», «время», «город», '
                    '«склад» (номер склада, число), «грузовики» (количество, число), «тонн» (масса груза, число).'
                    '\n\nСообщение: «12 марта в 14:30 на склад № 3 в Твери прибыли 4 грузовика '
                    'с 18 тоннами муки».',
             expected={'дата': '12 марта', 'время': '14:30', 'город': 'Тверь', 'склад': 3, 'грузовики': 4, 'тонн': 18}),
        dict(id='j002', category='извлечение', split='eval', schema=schema,
             prompt='Из сообщения извлеките сведения и верните JSON-объект с ключами «дата», «время», «город», '
                    '«склад» (номер склада, число), «грузовики» (количество, число), «тонн» (масса груза, число).'
                    '\n\nСообщение: «5 апреля к 9:05 на складе № 12 (Казань) разгружены '
                    '7 машин, всего 31,5 тонны».',
             expected={'дата': '5 апреля', 'время': '9:05', 'город': 'Казань', 'склад': 12, 'грузовики': 7,
                       'тонн': 31.5}),
    ]
    _write(m, items, 'образец')

    m = B.default_meta('own_open_example', 'Образец: развёрнутый ответ (планирование)', 'open', 'own')
    m['description'] = 'Развёрнутые ответы оцениваются экспертом и/или моделью-судьёй по критериям.'
    items = [
        dict(id='o001', category='планирование', split='eval',
             prompt='Составьте краткий план эвакуации сотрудников офисного здания при пожаре (не более 8 пунктов).',
             reference='1. Сигнал тревоги и оповещение. 2. Вызов пожарной охраны (101/112). 3. Выход по '
                       'эвакуационным маршрутам, не пользуясь лифтами. 4. Помощь людям с ограниченной '
                       'подвижностью. 5. Сбор в заранее определённом месте. 6. Проверка по спискам. '
                       '7. Встреча пожарных и передача сведений. 8. Запрет возвращаться в здание.',
             rubric=[{'name': 'Полнота этапов', 'max': 4}, {'name': 'Логичность последовательности', 'max': 3},
                     {'name': 'Конкретность и краткость', 'max': 3}]),
        dict(id='o002', category='анализ', split='eval',
             prompt='Сравните два варианта размещения склада: А — ближе к потребителям, но аренда дороже на 30 %; '
                    'Б — дешевле, но плечо доставки длиннее на 120 км. Укажите критерии сравнения и рекомендацию.',
             reference='Критерии: суммарные затраты (аренда + транспорт), время доставки, надёжность, риски, '
                       'гибкость. Рекомендация зависит от объёма и частоты поставок: при частых мелких поставках '
                       'выгоднее А, при редких крупных — Б; нужен расчёт полной стоимости.',
             rubric=[{'name': 'Критерии сравнения', 'max': 4}, {'name': 'Обоснованность рекомендации', 'max': 4},
                     {'name': 'Ясность изложения', 'max': 2}]),
    ]
    _write(m, items, 'образец')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['legacy', 'rebuild', 'mmlu', 'rubia', 'samples', 'fetch', 'legacy-results'])
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--n-bias', type=int, default=200)
    ap.add_argument('--n-tox', type=int, default=100)
    a = ap.parse_args()
    {'legacy': convert_legacy, 'samples': samples, 'fetch': fetch, 'legacy-results': import_legacy_results,
     'rebuild': lambda: rebuild(a.n, a.n_bias, a.n_tox), 'mmlu': lambda: from_mmlu(a.n), 'rubia': lambda: from_rubia(a.n_bias, a.n_tox)}[a.cmd]()


if __name__ == '__main__':
    main()
