"""Portable pipeline: shared Vanilla A, EDS-ECA, parallel Best-of-4, judges, metrics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
from oagents_official_orm import PROTOCOL as BEST4_PROTOCOL


def run(script, *args):
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / script), *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=[53403])
    p.add_argument('--selector-seed', type=int, default=77113)
    p.add_argument('--judge-seed', type=int, default=88002)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--split', choices=['train', 'dev', 'test'], default='test')
    p.add_argument('--tasks-per-domain', type=int, default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default=None, help='Chat Completions model ID; default MIMO_MODEL')
    p.add_argument('--base-url', default=None)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Seeds must be distinct')
    if a.model:
        os.environ['MIMO_MODEL'] = a.model
    if a.base_url:
        os.environ['MIMO_BASE_URL'] = a.base_url
    if not os.environ.get('MIMO_API_KEY') and os.environ.get('OPENAI_API_KEY'):
        os.environ['MIMO_API_KEY'] = os.environ['OPENAI_API_KEY']
    if not os.environ.get('MIMO_MODEL'):
        p.error('Specify --model or MIMO_MODEL')
    os.environ['PYTHONUTF8'] = '1'
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = {'model': os.environ['MIMO_MODEL'], 'base_url': os.environ.get('MIMO_BASE_URL', 'https://api.openai.com/v1'),
              'best4_protocol': BEST4_PROTOCOL,
              'seeds': a.seeds, 'selector_seed': a.selector_seed, 'judge_seed': a.judge_seed,
              'split': a.split, 'tasks_per_domain': a.tasks_per_domain,
              'temperature': os.environ.get('MIMO_TEMPERATURE', '0'),
              'max_tokens': os.environ.get('MIMO_MAX_TOKENS', '4096')}
    config['provider_options'] = {k: os.environ[k] for k in (
        'MIMO_SEND_SEED', 'MIMO_SEND_TEMPERATURE', 'MIMO_TOKEN_PARAMETER',
        'MIMO_REASONING_EFFORT', 'MIMO_NEMOTRON_THINKING', 'MIMO_CHAT_TEMPLATE_THINKING',
        'MIMO_THINKING', 'MIMO_REASONING_BUDGET', 'MIMO_SELECTOR_MAX_TOKENS') if k in os.environ}
    config_path = a.output_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        p.error('Output directory belongs to another configuration; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    common = ['--split', a.split, '--workers', a.workers, '--tasks-per-domain', a.tasks_per_domain]
    score_dirs = {m: [] for m in ('vanilla', 'eds_eca', 'best4')}
    for seed in a.seeds:
        folder = a.output_dir / f'seed_{seed}'
        anchor, eds, best = [folder / m for m in ('vanilla', 'eds_eca', 'best4')]
        run('phase0_runner.py', *common, '--seed', seed, '--output-dir', anchor)
        run('run_statetrace_eds_eca_paper.py', *common, '--seed', seed,
            '--selector-seed', a.selector_seed, '--anchor-dir', anchor, '--output-dir', eds)
        run('run_oagents_best_of4.py', *common, '--seed', seed, '--anchor-dir', anchor,
            '--selector-seed', a.selector_seed, '--selector-workers', a.workers, '--output-dir', best)
        for method, source in [('vanilla', anchor), ('eds_eca', eds), ('best4', best / 'final')]:
            out = folder / 'scores' / method
            run('phase0_score.py', '--input-dir', source, '--output-dir', out,
                '--split', a.split, '--workers', a.workers, '--judge-seed', a.judge_seed, '--with-ux')
            score_dirs[method].extend(['--run', f'{seed}={out}'])
    reports = {}
    for method, dirs in score_dirs.items():
        report = a.output_dir / f'{method}_metrics.json'
        run('aggregate_metrics.py', '--method', method, '--split', a.split, *dirs,
            *(['--allow-partial'] if a.tasks_per_domain else []), '--output', report)
        reports[method] = json.loads(report.read_text(encoding='utf-8'))
    (a.output_dir / 'comparison.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    lines = ['| Method | pass@1 | pass^5 | UX |', '| --- | ---: | ---: | ---: |']
    for method, r in reports.items():
        p5 = 'N/A' if r['pass_power_5'] is None else f"{r['pass_power_5']:.2%}"
        lines.append(f"| {method} | {r['pass_at_1']:.2%} | {p5} | {r['ux_mean']:.4f} |")
    (a.output_dir / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
