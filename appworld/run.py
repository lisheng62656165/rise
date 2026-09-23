"""Portable Faithful-v3 and shared-A OAgents APPWorld experiment driver."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def execute(script, arguments):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)], check=True, cwd=ROOT)


def share_a(source_root, dest_root, task_ids):
    for task_id in task_ids:
        source = source_root / task_id / 'candidate_a.json'
        dest = dest_root / task_id / 'candidate_a.json'
        if not source.exists():
            continue
        if dest.exists():
            if source.read_bytes() != dest.read_bytes():
                raise RuntimeError(f'Different Vanilla A artifacts for {task_id}; use a new output directory.')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['faithful-v3', 'both', 'oagents'], default='faithful-v3')
    parser.add_argument('--split', choices=['both', 'train', 'dev', 'test_normal', 'test_challenge'], default='both')
    parser.add_argument('--config', type=Path, help='Optional JSON preset; explicit CLI options override it')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', help='OpenAI-compatible API model ID; never a web ChatGPT subscription')
    parser.add_argument('--base-url')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Smoke/subset only; omit for the entire official split')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--selector-tokens', type=int, default=2048)
    parser.add_argument('--seed-a', type=int, default=53403,
                        help='Vanilla anchor seed shared by Faithful and OAgents')
    parser.add_argument('--proposal-seeds', default='64639,64640,64641',
                        help='Three comma-separated Faithful proposal seeds')
    parser.add_argument('--faithful-selector-seed', type=int, default=77113,
                        help='Base seed; per-task selector seed adds official task ordinal')
    parser.add_argument('--oagents-selector-seed', type=int, default=53403,
                        help='Base seed; per-task selector seed adds official task ordinal')
    args = parser.parse_args()
    config = {}
    if args.config:
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        config = json.loads(config_path.resolve().read_text(encoding='utf-8'))
        args.split = args.split if args.split != 'both' else config.get('split', args.split)
        args.model = args.model or config.get('model')
        args.base_url = args.base_url or config.get('base_url')
        args.max_steps = args.max_steps if args.max_steps != 50 else config.get('max_steps', args.max_steps)
        args.selector_tokens = (args.selector_tokens if args.selector_tokens != 2048
                                else config.get('selector_max_completion_tokens', args.selector_tokens))
        seeds = config.get('seeds', {})
        args.seed_a = args.seed_a if args.seed_a != 53403 else seeds.get('vanilla_anchor_and_oagents_A', args.seed_a)
        args.proposal_seeds = (
            args.proposal_seeds if args.proposal_seeds != '64639,64640,64641'
            else ','.join(map(str, seeds.get('faithful_proposals_BCD', [64639, 64640, 64641])))
        )
        args.faithful_selector_seed = (
            args.faithful_selector_seed if args.faithful_selector_seed != 77113
            else seeds.get('faithful_selector_base', args.faithful_selector_seed)
        )
        args.oagents_selector_seed = (
            args.oagents_selector_seed if args.oagents_selector_seed != 53403
            else seeds.get('oagents_selector_base', args.oagents_selector_seed)
        )
    if args.workers < 1 or args.max_steps < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers, max-steps and limit must be positive')
    # Load the local .env first; explicit CLI configuration wins in all children.
    from src.config import get_settings
    if args.model:
        os.environ['MODEL_NAME'] = args.model
    if args.base_url:
        os.environ['OPENAI_BASE_URL'] = args.base_url
    os.environ.pop('AGENT_EVAL_ENV_FILE', None)
    settings = get_settings()
    proposal_seeds = [int(value) for value in args.proposal_seeds.split(',')]
    if len(proposal_seeds) != 3:
        parser.error('--proposal-seeds requires exactly three comma-separated integers')
    os.environ['OPENAI_BASE_URL'] = settings.base_url
    os.environ['OPENAI_API_KEY'] = settings.api_key
    appworld_root = Path(os.environ.get('APPWORLD_ROOT', str(ROOT / 'runtime'))).resolve()
    os.environ['APPWORLD_ROOT'] = str(appworld_root)
    from appworld import load_task_ids
    splits = ['test_normal', 'test_challenge'] if args.split == 'both' else [args.split]
    output = args.output.resolve()
    incomplete = []
    for split in splits:
        all_ids = load_task_ids(split)
        task_ids = all_ids[:args.limit] if args.limit else all_ids
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / 'experiment.json'
        record = {key: value for key, value in vars(settings).items() if key != 'api_key'}
        record.update(task_ids=task_ids, split=split, max_steps=args.max_steps,
                      selector_tokens=args.selector_tokens, method=args.method,
                      seed_a=args.seed_a, proposal_seeds=proposal_seeds,
                      faithful_selector_seed=args.faithful_selector_seed,
                      oagents_selector_seed=args.oagents_selector_seed,
                      config_name=config.get('name'),
                      provider_options={k: os.getenv(k) for k in ['OPENAI_REASONING', 'SEND_SEED', 'REASONING_EFFORT', 'NVIDIA_ENABLE_THINKING', 'NVIDIA_CONTEXT_COMPACT']})
        if record_path.exists():
            old = json.loads(record_path.read_text())
            run_id = old.pop('run_id')
            # Existing candidate files are keyed by task, not model. Detect
            # accidental reuse across models instead of silently mixing results.
            if old != record:
                raise SystemExit(f'Experiment settings differ in {directory}; use a new --output.')
        else:
            run_id = 'release_' + uuid.uuid4().hex[:12]
            record_path.write_text(json.dumps({**record, 'run_id': run_id}, indent=2) + '\n')
        common = ['--appworld-root', appworld_root, '--dataset', split, '--workers', args.workers, '--max-steps', args.max_steps]
        if args.limit:
            common += ['--limit', args.limit]
        st = directory / 'faithful_v3'
        pool = directory / 'oagents_candidates'
        if args.method == 'oagents':
            # Produce the shared Vanilla anchor with the same seed if no
            # Faithful run already populated it in this output directory.
            share_a(st, pool, task_ids)
            missing_a = [task_id for task_id in task_ids
                         if not (pool / task_id / 'candidate_a.json').exists()]
            if missing_a:
                execute('run_appworld_best_of4.py', common + [
                    '--candidate-seeds', ','.join(map(str, [args.seed_a, *proposal_seeds])),
                    '--selector-seed', args.oagents_selector_seed,
                    '--output-dir', pool, '--run-name', 'shared_vanilla_anchor', '--candidates-only',
                    '--task-ids', ','.join(missing_a)])
        if args.method in ['both', 'faithful-v3']:
            share_a(pool, st, task_ids)
            execute('run_appworld_state_trace_eds_eca.py', common + [
                '--seed-a', args.seed_a, '--proposal-seeds', ','.join(map(str, proposal_seeds)),
                '--selector-seed', args.faithful_selector_seed,
                '--output-dir', st, '--run-name', run_id + '_st',
                '--selector-max-completion-tokens', args.selector_tokens])
        if args.method in ['both', 'oagents']:
            # Share Vanilla A exactly; B/C/D are independent fresh ReAct rollouts,
            # never the feedback-conditioned StateTrace proposals.
            share_a(st, pool, task_ids)
            execute('run_appworld_best_of4.py', common + [
                '--candidate-seeds', ','.join(map(str, [args.seed_a, *proposal_seeds])),
                '--selector-seed', args.oagents_selector_seed,
                '--output-dir', pool, '--run-name', run_id + '_oa', '--candidates-only'])
            original_seed = os.environ.get('LLM_SEED')
            # The selector derives a per-task seed from the supplied base.
            os.environ['LLM_SEED'] = str(args.oagents_selector_seed)
            execute('select_appworld_oagents_bon4.py', [
                '--candidate-dir', pool, '--output-dir', directory / 'oagents_selector',
                '--workers', args.workers, '--max-tokens', args.selector_tokens])
            if original_seed is None:
                os.environ.pop('LLM_SEED', None)
            else:
                os.environ['LLM_SEED'] = original_seed
        for task_id in task_ids:
            if args.method in ['both', 'faithful-v3'] and not (st / task_id / 'final.json').exists():
                incomplete.append(f'{split}/faithful_v3/{task_id}')
            if args.method in ['both', 'oagents'] and not (directory / 'oagents_selector' / f'{task_id}.json').exists():
                incomplete.append(f'{split}/oagents/{task_id}')
    subprocess.run([sys.executable, str(ROOT / 'summarize.py'), '--output', str(output)], check=True)
    if incomplete:
        print(f'{len(incomplete)} method/task outputs remain incomplete. Inspect error.json files and repeat the same command.', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
