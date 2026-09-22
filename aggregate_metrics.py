"""Aggregate one method across repeated runs, never across different methods."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from phase0_runner import DOMAINS, bundled_benchmark_root, read_split_ids, write_json

REQUIRED = ('task_completion_pass', 'state_requirements_met', 'task_requirements_met', 'ux_score')


def expected_keys(split='test'):
    return {f'statebench::{d}::{t}' for d in DOMAINS
            for t in read_split_ids(bundled_benchmark_root(), d, split)}


def load_run(spec):
    label, raw_dir = spec.split('=', 1)
    rows = {}
    for path in sorted(Path(raw_dir).glob('*.json')):
        if path.name.endswith(('.error.json', 'summary.json')) or path.name in {'metrics.json', 'report.json'}:
            continue
        row = json.loads(path.read_text(encoding='utf-8'))
        if not row.get('task_id') and not row.get('task_key'):
            raise ValueError(f'{path}: not a scored task record')
        key = row.get('task_key') or f"statebench::{row['domain']}::{row['task_id']}"
        if key in rows:
            raise ValueError(f'{path}: duplicate task key')
        for name in REQUIRED[:-1]:
            value = row.get(name)
            if not isinstance(value, (int, bool)) or value not in (0, 1):
                raise ValueError(f'{path}: invalid {name}')
        ux = row.get('ux_score')
        if isinstance(ux, bool) or not isinstance(ux, (int, float)) or not math.isfinite(ux) or not 1 <= ux <= 5:
            raise ValueError(f'{path}: invalid UX')
        rows[key] = row
    if not rows:
        raise ValueError(f'{raw_dir}: no scored tasks')
    return label, rows


def summarize(rows):
    values = list(rows.values())
    n = len(values)
    return {'num_tasks': n,
            'pass_count': sum(r['task_completion_pass'] for r in values),
            'pass_at_1': sum(r['task_completion_pass'] for r in values) / n,
            'ux_mean': sum(r['ux_score'] for r in values) / n,
            'state_requirements_mean': sum(r['state_requirements_met'] for r in values) / n,
            'task_requirements_mean': sum(r['task_requirements_met'] for r in values) / n}


def aggregate(runs, method, split='test', allow_partial=False):
    expected = expected_keys(split)
    partial = any(set(rows) != expected for rows in runs.values())
    if partial and not allow_partial:
        raise ValueError(f'Incomplete/mismatched {split} coverage: need exactly {len(expected)} task keys in EVERY run')
    if any(set(rows) - expected for rows in runs.values()):
        raise ValueError('Unexpected task outside requested split')
    all_rows = {f'{seed}/{key}': row for seed, rows in runs.items() for key, row in rows.items()}
    report = {'method': method, 'num_runs': len(runs), 'split': split, 'partial': partial,
              'expected_tasks_per_run': len(expected), **summarize(all_rows),
              'runs': {seed: summarize(rows) for seed, rows in runs.items()},
              'pass_power_5': None, 'pass_power_5_count': None}
    report['num_scored_rows'] = report.pop('num_tasks')
    if len(runs) == 5 and not partial:
        count = sum(all(rows[key]['task_completion_pass'] == 1 for rows in runs.values()) for key in expected)
        report.update(pass_power_5=count / len(expected), pass_power_5_count=count)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['vanilla', 'eds_eca', 'best4'])
    parser.add_argument('--run', action='append', required=True, metavar='SEED=DIR')
    parser.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--allow-partial', action='store_true', help='Smoke/progress only; suppress pass^5')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = {}
    directories = set()
    for spec in args.run:
        label, rows = load_run(spec)
        directory = Path(spec.split('=', 1)[1]).resolve()
        if label in runs or directory in directories:
            raise ValueError('Duplicate run label or directory; five copies are not five runs')
        directories.add(directory)
        runs[label] = rows
    result = aggregate(runs, args.method, args.split, args.allow_partial)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
