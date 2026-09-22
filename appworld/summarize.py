"""Summarize matched tasks; incomplete scenario groups are never SGC successes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(values, official_ids):
    groups = {}
    for task_id in official_ids:
        groups.setdefault(task_id.rsplit('_', 1)[0], []).append(task_id)
    complete = [ids for ids in groups.values() if all(t in values for t in ids)]
    success = sum(all(values[t] for t in ids) for ids in complete)
    return {'task_success': sum(values.values()), 'tasks': len(values),
            'scenario_success': success, 'scenarios': len(complete),
            'TGC': 100 * sum(values.values()) / len(values) if values else None,
            'SGC': 100 * success / len(complete) if complete else None}


def collect(root, split, official_ids):
    directory = root / split
    if not (directory / 'experiment.json').exists():
        return None
    config = read(directory / 'experiment.json')
    results = {'Vanilla': {}, 'Faithful v3': {}, 'OAgents parallel Best-of-4\u2020': {}}
    for task_id in config['task_ids']:
        a = directory / 'faithful_v3' / task_id / 'candidate_a.json'
        if not a.exists():
            a = directory / 'oagents_candidates' / task_id / 'candidate_a.json'
        if a.exists():
            results['Vanilla'][task_id] = bool(read(a)['evaluation_success'])
        st = directory / 'faithful_v3' / task_id / 'final.json'
        oa = directory / 'oagents_selector' / f'{task_id}.json'
        if st.exists():
            results['Faithful v3'][task_id] = bool(read(st)['selected_success'])
        if oa.exists():
            results['OAgents parallel Best-of-4\u2020'][task_id] = bool(read(oa)['selected_success'])
    active = {k: v for k, v in results.items() if v}
    common = set.intersection(*(set(v) for v in active.values())) if active else set()
    matched = {
        k: metrics({t: v[t] for t in common}, official_ids)
        for k, v in active.items()
    }
    return {'model': config['model_name'], 'requested': len(config['task_ids']),
            'official_tasks': len(official_ids), 'common_tasks': len(common),
            'complete': len(common) == len(official_ids),
            'comparison_methods': list(active),
            'available': {k: metrics(v, official_ids) for k, v in results.items()}, 'matched': matched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['APPWORLD_ROOT'] = str(ROOT / 'runtime')
    from appworld import load_task_ids
    report = {s: collect(args.output, s, load_task_ids(s)) for s in ['test_normal', 'test_challenge']}
    names = ['Vanilla', 'Faithful v3', 'OAgents parallel Best-of-4\u2020']
    rows = ['| Method | Test-N TGC | Test-N SGC | Test-C TGC | Test-C SGC |',
            '|---|---:|---:|---:|---:|']
    for name in names:
        cells = []
        for split in ['test_normal', 'test_challenge']:
            result = (report[split] or {}).get('matched', {}).get(name, {})
            for metric in ['TGC', 'SGC']:
                value = result.get(metric)
                cells.append('--' if value is None else f'{value:.1f}')
        rows.append('| ' + ' | '.join([name, *cells]) + ' |')
    for left, right in [
        ('Faithful v3', 'Vanilla'),
        ('OAgents parallel Best-of-4\u2020', 'Vanilla'),
        ('Faithful v3', 'OAgents parallel Best-of-4\u2020'),
    ]:
        deltas = []
        for split in ['test_normal', 'test_challenge']:
            matched = (report[split] or {}).get('matched', {})
            for metric in ['TGC', 'SGC']:
                a = matched.get(left, {}).get(metric)
                b = matched.get(right, {}).get(metric)
                deltas.append('--' if a is None or b is None else f'{a-b:+.1f}pp')
        rows.append('| ' + ' | '.join([f'{left} - {right}', *deltas]) + ' |')
    rows += ['', 'Percentages; same-task intersection per split. Only complete official scenario groups count for SGC.']
    for split, info in report.items():
        if info:
            rows.append(f"{split}: {info['common_tasks']}/{info['official_tasks']} matched tasks; model={info['model']}; Faithful/Vanilla complete={info['complete']}")
    text = '\n'.join(rows) + '\n'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'metrics.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
