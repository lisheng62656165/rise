"""Score a fixed completed cohort, pairing Vanilla A with accepted final."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from score_occubench_stages_nvidia import load_keys, load_tasks, read_jsonl


def candidates(result):
    anchor = next(s for s in result['stages'] if s['stage'] == 'A')
    final = next(s for s in result['stages']
                 if s['stage'] == result['final_stage']
                 and s['trajectory'] == result['final_trajectory'])
    return {'A': anchor, 'FINAL': final}


def summarize(ids, labels):
    paired = [i for i in ids if (i, 'A') in labels and (i, 'FINAL') in labels]
    a = sum(labels[i, 'A']['is_correct'] for i in paired)
    f = sum(labels[i, 'FINAL']['is_correct'] for i in paired)
    wins = sum(not labels[i, 'A']['is_correct'] and labels[i, 'FINAL']['is_correct'] for i in paired)
    losses = sum(labels[i, 'A']['is_correct'] and not labels[i, 'FINAL']['is_correct'] for i in paired)
    n = len(paired)
    return dict(cohort=len(ids), paired_valid=n, pending_pairs=len(ids)-n,
                A_valid=sum((i, 'A') in labels for i in ids),
                FINAL_valid=sum((i, 'FINAL') in labels for i in ids),
                A_pass=a, FINAL_pass=f, wins=wins, losses=losses, ties=n-wins-losses,
                A_pass_at_1=a/n if n else None, FINAL_pass_at_1=f/n if n else None,
                delta_pp=100*(f-a)/n if n else None,
                complete=n == len(ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--key-file', type=Path)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--count', type=int, default=204)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--non-stream', action='store_true')
    p.add_argument('--key-back', type=int, default=4)
    p.add_argument('--verifier-max-tokens', type=int, default=4096)
    p.add_argument('--model', default='nvidia/nemotron-3.5-lightning-30b-a3b')
    p.add_argument('--base-url', default='https://api.openai.com/v1')
    ns = p.parse_args()
    out = ns.run_dir / f'paired_{ns.count}'
    out.mkdir(exist_ok=True)
    snapshot = out / 'cohort.jsonl'
    if not snapshot.exists():
        rows = {}
        with (ns.run_dir / 'results.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                result = json.loads(line)
                tid = int(result['task_id'])
                if tid in rows:
                    continue
                selected = candidates(result)
                rows[tid] = {'task_id': tid, 'final_stage': result['final_stage'],
                             'candidates': selected}
                if len(rows) == ns.count:
                    break
        if len(rows) != ns.count:
            raise ValueError(f'Expected {ns.count} completed tasks, found {len(rows)}')
        with snapshot.open('w', encoding='utf-8') as handle:
            for row in rows.values():
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    rows = read_jsonl(snapshot)
    ids = [int(r['task_id']) for r in rows]
    score_path = out / 'scores.jsonl'
    labels = {(int(r['task_id']), r['stage']): r for r in read_jsonl(score_path)
              if r.get('verification_valid') is True and r.get('verifier_model') == ns.model
              and type(r.get('is_correct')) is bool}

    def record(tid, stages, verdict):
        with score_path.open('a', encoding='utf-8') as handle:
            for stage in stages:
                item = dict(verdict, task_id=tid, stage=stage, verifier_model=ns.model)
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                if item['verification_valid']:
                    labels[tid, stage] = item

    def report():
        summary = summarize(ids, labels)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)

    jobs = []
    for result in rows:
        tid = int(result['task_id'])
        groups = {}
        for stage, row in result['candidates'].items():
            groups.setdefault(row['trajectory'], []).append(stage)
        for trajectory, stages in groups.items():
            pending = [s for s in stages if (tid, s) not in labels]
            if not pending:
                continue
            known = next((labels[tid, s] for s in stages if (tid, s) in labels), None)
            if known is None:
                for s in stages:
                    row = result['candidates'][s]
                    if (row.get('verification_valid') is True
                            and row.get('verifier_model') == ns.model
                            and row.get('verification_valid_votes') == 1
                            and type(row.get('is_correct')) is bool):
                        known = dict(is_correct=row['is_correct'], verification_valid=True,
                                     feedback=row.get('feedback', ''), source='stored_matching_trajectory')
                        break
            if known:
                record(tid, pending, known)
            else:
                jobs.append((tid, pending, trajectory))
    report()
    print(json.dumps({'unique_verifier_jobs': len(jobs), 'workers': ns.workers}), flush=True)
    if not jobs:
        return
    sys.path.insert(0, str(ns.dataset_root))
    from occubench.lwm import WorldModelRegistry, create_client
    from occubench.verifier import Verifier
    os.environ['OCCUBENCH_REQUEST_TIMEOUT'] = '180'
    os.environ['OCCUBENCH_VERIFIER_MAX_TOKENS'] = str(ns.verifier_max_tokens)
    os.environ['OCCUBENCH_STREAM'] = '0' if ns.non_stream else '1'
    os.environ['OCCUBENCH_RESPONSE_LOG'] = str(out / 'verifier_responses.jsonl')
    os.environ['OCCUBENCH_VERIFIER_EXTRA_BODY_JSON'] = '{"chat_template_kwargs":{"enable_thinking":false}}'
    tasks = load_tasks(ns.dataset_root)
    registry = WorldModelRegistry(str(ns.dataset_root / 'data' / 'world_model_configs'))
    if ns.key_file:
        keys = load_keys(ns.key_file, ns.key_back, 10)
    else:
        api_key = os.environ.get(ns.api_key_env, '')
        if not api_key:
            raise ValueError(f'Missing API key environment variable: {ns.api_key_env}')
        keys = [(0, api_key)]

    def score(index, tid, trajectory):
        task = tasks[tid]
        config = registry.get(task['env_name'])
        initial = config.get('task_initial_state', '{}')
        if isinstance(initial, dict):
            initial = json.dumps(initial, ensure_ascii=False)
        error = 'Verification error'
        # Rotate keys and retry transient endpoint/stream failures. Invalid
        # verifier responses remain unscored rather than becoming negatives.
        for attempt in range(8):
            try:
                key_back, key = keys[(index * 3 + attempt) % len(keys)]
                print(json.dumps(dict(task_id=tid, attempt=attempt+1,
                                      key_back=key_back, status='request_started',
                                      stream=not ns.non_stream)), flush=True)
                client = create_client(key, ns.base_url)
                try:
                    verdict = Verifier(ns.model, client=client, num_votes=1).check(
                        task_scenario_name=task['task_scenario_name'], task_initial_state=initial,
                        state_description=config.get('state_description', ''),
                        agent_instruction=task['agent_instruction'],
                        verification_plan=task['verification_plan'], trajectory=trajectory)
                finally:
                    client.close()
                if verdict.get('feedback') != 'Verification error' and type(verdict.get('is_correct')) is bool:
                    return dict(is_correct=verdict['is_correct'], feedback=verdict.get('feedback', ''),
                                verification_valid=True, source='paired_verifier', attempt=attempt+1,
                                verifier_max_tokens=ns.verifier_max_tokens, stream=not ns.non_stream)
            except Exception as exc:
                error = type(exc).__name__
            time.sleep(min(30, 2 ** min(attempt, 4)))
        return dict(is_correct=None, verification_valid=False, error=error)

    with ThreadPoolExecutor(max_workers=ns.workers) as pool:
        futures = {pool.submit(score, i, tid, trajectory): (tid, stages)
                   for i, (tid, stages, trajectory) in enumerate(jobs)}
        for future in as_completed(futures):
            tid, stages = futures[future]
            verdict = future.result()
            record(tid, stages, verdict)
            print(json.dumps(dict(task_id=tid, stages=stages, valid=verdict['verification_valid'])), flush=True)
            report()


if __name__ == '__main__':
    main()
