"""Independent Vanilla anchor + three fresh rollouts, then public listwise selection."""
import argparse
import copy
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_occubench_base as base
from score_occubench_stages_nvidia import read_jsonl, load_keys
from occubench_request_monitor import RequestMonitor

MODEL = 'nvidia/nemotron-3.5-lightning-30b-a3b'
URL = 'https://integrate.api.nvidia.com/v1'
LOCK = threading.Lock()


def generation_settings(source_config=None):
    if source_config is not None:
        return {
            'agent_extra': json.loads(source_config.get('agent_extra_body_json') or '{}'),
            'world_extra': json.loads(source_config.get('world_extra_body_json') or '{}'),
            'max_steps': source_config['max_steps'],
            'max_tokens': source_config['max_tokens'],
        }
    return {
        'agent_extra': {'temperature': 1, 'top_p': 0.95,
                        'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_budget': 16384},
        'world_extra': {'chat_template_kwargs': {'enable_thinking': False}},
        'max_steps': 200, 'max_tokens': 16384,
    }


INSTRUCTION = '''Compare four independent trajectories using only the visible task and
public actions, observations, and responses. Choose the one that best satisfies the
request, with correct objects, grounded parameters, valid action order, and verified
effects. A rejection can be the correct outcome. Length, number of calls, confident
claims and absence of error words do not prove success. Consider recovery handling,
unsupported claims, contradictory results and missing obligations. Call select_trajectory
exactly once. Do not infer hidden requirements or evaluator labels.'''
TOOL = {'type': 'function', 'function': {'name': 'select_trajectory',
        'description': 'Choose one public trajectory.', 'parameters': {'type': 'object',
        'properties': {'candidate_index': {'type': 'integer', 'enum': [0, 1, 2, 3]},
                       'reason': {'type': 'string'}},
        'required': ['candidate_index', 'reason'], 'additionalProperties': False}}}


def packet(task, rows, seed):
    if len(rows) != 4:
        raise ValueError('Best-of-4 requires four candidates')
    order = list(range(4))
    random.Random(seed).shuffle(order)
    return {'visible_instruction': task['agent_instruction'],
            'task_scenario_name': task['task_scenario_name'],
            'candidates': [{'candidate_index': i, 'trajectory': rows[j]['trajectory']}
                           for i, j in enumerate(order)]}, order


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_four_way_selection(response, order):
    """Parse exactly one valid four-way selector tool call."""
    choices = getattr(response, 'choices', None) or []
    if not choices:
        raise ValueError('selector_empty_response')
    message = getattr(choices[0], 'message', None)
    calls = getattr(message, 'tool_calls', None) or []
    matching = []
    for call in calls:
        function = getattr(call, 'function', None)
        if function is None or getattr(function, 'name', None) != 'select_trajectory':
            continue
        raw = getattr(function, 'arguments', None)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError('selector_invalid_json') from exc
        if not isinstance(parsed, dict):
            raise ValueError('selector_arguments_not_object')
        selected = parsed.get('candidate_index')
        if type(selected) is not int or selected not in range(4):
            raise ValueError('selector_invalid_candidate_index')
        if not str(parsed.get('reason') or '').strip():
            raise ValueError('selector_missing_reason')
        matching.append((selected, parsed))
    if len(matching) != 1:
        raise ValueError('selector_missing_or_ambiguous_tool_call')
    selected, parsed = matching[0]
    return dict(selected=order[selected], order=order, response=parsed)


def extend_ids(available, retained, count, seed):
    if len(set(retained)) != len(retained) or not set(retained) <= set(available):
        raise ValueError('Invalid retained task IDs')
    if not len(retained) <= count <= len(set(available)):
        raise ValueError('Invalid cohort size')
    additions = random.Random(seed).sample(sorted(set(available)-set(retained)), count-len(retained))
    return sorted(retained + additions)


def summarize_results(results, total):
    n = len(results)
    def correct(row, name):
        verdict = row.get(name)
        return bool(verdict and type(verdict.get('is_correct')) is bool and verdict['is_correct'])

    successes = {name: sum(correct(r, name) for r in results)
                 for name in ('vanilla', 'bestof4', 'eds_eca')}
    paired = [r for r in results
              if type((r.get('eds_eca') or {}).get('is_correct')) is bool
              and type((r.get('bestof4') or {}).get('is_correct')) is bool]
    wins = sum(correct(r, 'eds_eca') and not correct(r, 'bestof4') for r in paired)
    losses = sum(correct(r, 'bestof4') and not correct(r, 'eds_eca') for r in paired)
    return dict(completed=n, total=total, successes=successes,
                pass_at_1={k: v/n if n else None for k, v in successes.items()},
                eds_vs_bestof4=dict(wins=wins, losses=losses, ties=n-wins-losses,
                                   paired_valid=len(paired),
                                   delta_pp=100*(wins-losses)/len(paired) if paired else None))


def main():
    global MODEL
    global URL
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--key-file', type=Path)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY',
                   help='Environment variable used when --key-file is omitted.')
    p.add_argument('--base-url', default=URL)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--active-requests', type=int, default=8)
    p.add_argument('--adaptive-requests', action='store_true')
    p.add_argument('--seed', type=int, default=53403)
    p.add_argument('--count', type=int, default=30)
    p.add_argument('--extend-from', type=Path)
    p.add_argument('--match-source-config', action='store_true')
    p.add_argument('--short-initial-context', type=int, choices=[0, 1], default=1)
    p.add_argument('--scores-dir', type=Path)
    p.add_argument('--key-back', type=int, default=3)
    p.add_argument('--selector-policy', choices=['custom', 'oagents'], default='custom')
    p.add_argument('--candidate-source', type=Path)
    p.add_argument('--task-ids-file', type=Path)
    p.add_argument('--allow-missing-scores', action='store_true')
    p.add_argument('--defer-verification', action='store_true')
    ns = p.parse_args()
    random.seed(ns.seed)
    URL = ns.base_url
    if ns.candidate_source and ns.candidate_source.resolve() == ns.out.resolve():
        raise ValueError('Use a separate output directory for re-selection')
    source_config = {}
    if ns.match_source_config:
        source_config = json.loads((ns.source / 'config.json').read_text(encoding='utf-8'))
        MODEL = source_config['agent_model']
        if any(source_config[role + '_model'] != MODEL for role in ('world', 'selector', 'verifier')):
            raise ValueError('This runner requires matching role models')
    # An empty source extra-body is meaningful: preserve the source runtime
    # defaults so the reused anchor and fresh candidates have the same API
    # sampling configuration. Selector settings remain independent below.
    generation = generation_settings(source_config if ns.match_source_config else None)
    agent_extra = generation['agent_extra']
    world_extra = generation['world_extra']
    selector_extra = json.loads(source_config.get('selector_extra_body_json') or '{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":16384}')
    ns.out.mkdir(parents=True, exist_ok=True)
    source_ids = []
    with (ns.source / 'results.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            source_ids.append(int(json.loads(line)['task_id']))
    manifest = ns.out / 'manifest.json'
    retained = []
    if ns.extend_from:
        previous = json.loads((ns.extend_from / 'manifest.json').read_text(encoding='utf-8'))
        if previous['model'] != MODEL or previous['seed'] != ns.seed:
            raise ValueError('Extension model/seed mismatch')
        retained = previous['task_ids']
    if manifest.exists():
        existing_manifest = json.loads(manifest.read_text(encoding='utf-8'))
        if existing_manifest.get('selector_policy', 'custom') != ns.selector_policy:
            raise ValueError('Output directory belongs to a different selector')
        ids = existing_manifest['task_ids']
        if len(ids) != ns.count or not set(retained) <= set(ids):
            raise ValueError('Existing cohort does not match requested extension')
    else:
        if ns.task_ids_file:
            ids = [int(x) for x in ns.task_ids_file.read_text(encoding='utf-8').split()]
        else:
            ids = (json.loads((ns.candidate_source / 'manifest.json').read_text(encoding='utf-8'))['task_ids']
                   if ns.candidate_source else extend_ids(source_ids, retained, ns.count, ns.seed))
        if len(ids) != ns.count or not set(ids) <= set(source_ids):
            raise ValueError('Candidate cohort mismatch')
        save(manifest, dict(task_ids=ids, sampling='random sample of sorted task IDs, no scores',
                            seed=ns.seed, model=MODEL, candidate_count=4,
                            anchor='existing Vanilla A, not regenerated',
                            new_agent=dict(extra_body=agent_extra, max_tokens=generation['max_tokens']),
                            world_extra=world_extra, selector_extra=selector_extra,
                            max_steps=generation['max_steps'], selector_policy=ns.selector_policy,
                            short_initial_context=bool(ns.short_initial_context),
                            candidate_source=str(ns.candidate_source) if ns.candidate_source else None,
                            selector=('OAgents ORM-list-wise; temperature=0; max_tokens=2048; text JSON; no thinking'
                                      if ns.selector_policy == 'oagents' else 'public anonymous listwise four-way'),
                            verifier_votes=1, source=str(ns.source.resolve()),
                            retained_task_ids=retained, count=ns.count,
                            extend_from=str(ns.extend_from) if ns.extend_from else None))
    for tid in retained:
        destination = ns.out / str(tid)
        destination.mkdir(exist_ok=True)
        for original in (ns.extend_from / str(tid)).glob('*.json'):
            if not (destination / original.name).exists():
                shutil.copy2(original, destination / original.name)
    anchors = {}
    with (ns.source / 'results.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            result = json.loads(line)
            if int(result['task_id']) in ids:
                anchors[int(result['task_id'])] = next(s for s in result['stages'] if s['stage'] == 'A')
    del result
    sys.path.insert(0, str(ns.dataset_root.resolve()))
    from occubench.lwm import WorldModelRegistry, LWMEnvironment, create_client
    from occubench.verifier import Verifier
    from occubench.fault_injection import build_fault_prompt
    monitor = RequestMonitor(ns.active_requests, ns.out / 'requests.jsonl')
    if ns.adaptive_requests:
        from adaptive_request_monitor import AdaptiveRequestMonitor
        monitor = AdaptiveRequestMonitor(ns.active_requests, ns.active_requests, ns.out / 'requests.jsonl')
    original_make_client = base.make_client
    original_create_client = create_client
    thread_clients = threading.local()

    def monitored_client(*args, **kwargs):
        client = original_make_client(*args, **kwargs)
        if hasattr(thread_clients, 'clients'):
            thread_clients.clients.append(client._client)
        return monitor.wrap(client._client) if not args[2] else type(client)(monitor.wrap(client._client), args[2])

    base.make_client = monitored_client
    create_client = lambda *a, **kw: monitor.wrap(original_create_client(*a, **kw))
    registry = WorldModelRegistry(str(ns.dataset_root / 'data/world_model_configs'))
    modules = dict(LWMEnvironment=LWMEnvironment, Verifier=Verifier, build_fault_prompt=build_fault_prompt)
    tasks = {int(t['task_id']): t for t in base.load_tasks(ns.dataset_root, ids, None, None)}
    if ns.key_file:
        keys = load_keys(ns.key_file, ns.key_back, 10)
    else:
        api_key = os.environ.get(ns.api_key_env, '')
        if not api_key:
            raise ValueError(f'Missing API key environment variable: {ns.api_key_env}')
        keys = [(0, api_key)]
    # Allow recovery runs to shorten a provider hang without changing the
    # default used by existing experiments.
    os.environ.setdefault('OCCUBENCH_REQUEST_TIMEOUT', '180')
    os.environ.setdefault('OCCUBENCH_STREAM', '1')
    os.environ.update(OCCUBENCH_LLM_RETRY_CAP='3',
                      OCCUBENCH_SHORT_INITIAL_CONTEXT=str(ns.short_initial_context), OCCUBENCH_LENGTH_CONTINUATIONS='1',
                      OCCUBENCH_VERIFIER_MAX_TOKENS='4096')
    os.environ['OCCUBENCH_VERIFIER_EXTRA_BODY_JSON'] = '{"chat_template_kwargs":{"enable_thinking":false}}'
    historical = {(r['task_id'], r['stage']): r for r in read_jsonl((ns.scores_dir or ns.source / 'paired_382') / 'scores.jsonl')
                  if r.get('verification_valid') is True and r.get('verifier_model') == MODEL
                  and type(r.get('is_correct')) is bool}
    for tid in ids:
        if not ns.allow_missing_scores and any((tid, stage) not in historical for stage in ('A', 'FINAL')):
            raise ValueError(f'Missing paired labels for {tid}')
        if anchors[tid]['agent_model'] != MODEL:
            raise ValueError(f'Anchor model mismatch for {tid}')

    def emit(**data):
        with LOCK:
            print(json.dumps(data), flush=True)

    def run(index, tid):
        out = ns.out / str(tid)
        out.mkdir(exist_ok=True)
        if (out / 'result.json').exists():
            return
        if ns.candidate_source and any(not (ns.candidate_source / str(tid) / f'candidate_{c}.json').exists()
                                       for c in range(1, 4)):
            emit(task_id=tid, status='waiting_for_candidates')
            return
        task = tasks[tid]
        anchor = anchors[tid]
        rows = [anchor]
        for c in range(1, 4):
            path = out / f'candidate_{c}.json'
            if ns.candidate_source:
                path = ns.candidate_source / str(tid) / f'candidate_{c}.json'
            if path.exists():
                rows.append(json.loads(path.read_text(encoding='utf-8')))
                continue
            for attempt in range(3):
                back, key = keys[(index + attempt * ns.workers) % len(keys)]
                args = argparse.Namespace(env_mode='E0', fault_count=0, fault_duration=0,
                    extra_body_json='', agent_extra_body_json=json.dumps(agent_extra),
                    world_extra_body_json=json.dumps(world_extra),
                    verifier_extra_body_json='', agent_model=MODEL, world_model=MODEL, verifier_model=MODEL,
                    agent_api_key=key, world_api_key=key, verifier_api_key=key,
                    agent_base_url=URL, world_base_url=URL, verifier_base_url=URL,
                    agent_reasoning_effort=None, agent_request_retries=source_config.get('agent_request_retries', 2), agent_retry_delay=source_config.get('agent_retry_delay', 5),
                    persist_max_interventions=0, persist_min_steps_before_stop=0, persist_policy='strict_v1',
                    persist_repeat_threshold=3, persist_max_length_continuations=0,
                    max_steps=generation['max_steps'], max_tokens=generation['max_tokens'], verifier_votes=1, verifier_retries=2)
                emit(task_id=tid, candidate=c, status='generating', key_back=back)
                try:
                    thread_clients.clients = []
                    row = base.evaluate_one(args, copy.deepcopy(task), 'raw', registry, modules, verify=False)
                    save(path, row)
                    rows.append(row)
                    emit(task_id=tid, candidate=c, status='saved', steps=row['step_count'])
                    break
                except Exception as exc:
                    emit(task_id=tid, candidate=c, error=type(exc).__name__)
                    if attempt == 2:
                        raise
                finally:
                    for transport in thread_clients.clients:
                        transport.close()
                    del thread_clients.clients
        choice_path = out / 'selection.json'
        if choice_path.exists():
            decision = json.loads(choice_path.read_text(encoding='utf-8'))
        elif ns.selector_policy == 'oagents':
            import oagents_bon_selector as official
            selector_messages = official.messages(task, rows)
            request_batch = time.time_ns()
            for attempt in range(3):
                _, key = keys[(index + attempt * ns.workers) % len(keys)]
                client = base.make_client(key, URL, {'chat_template_kwargs': {'enable_thinking': False}})
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=(official.format_retry_messages(selector_messages) if attempt else selector_messages),
                        temperature=0, max_tokens=2048, stream=False,
                        **({'response_format': {'type': 'json_object'}} if attempt else {}))
                    save(out / f'selector_attempt_{request_batch}_{attempt + 1}.json', response.model_dump())
                    decision = official.parse(response)
                    decision['format_retry'] = bool(attempt)
                    decision['raw_response_file'] = f'selector_attempt_{request_batch}_{attempt + 1}.json'
                    save(choice_path, decision)
                    break
                except Exception as exc:
                    emit(task_id=tid, selector_error=type(exc).__name__, selector_attempt=attempt + 1)
                    if attempt == 2:
                        raise
                finally:
                    client._client.close()
        else:
            public, order = packet(task, rows, ns.seed + tid)
            selector_messages = [
                {'role': 'system', 'content': INSTRUCTION},
                {'role': 'user', 'content': json.dumps(public, ensure_ascii=False)},
            ]
            for attempt in range(3):
                _, key = keys[(index + attempt * ns.workers) % len(keys)]
                client = base.make_client(key, URL, selector_extra)
                try:
                    response = base.create_chat_completion_with_retry(client, max_retries=1, retry_delay=5,
                        model=MODEL, messages=selector_messages,
                        tools=[TOOL], tool_choice={'type': 'function', 'function': {'name': 'select_trajectory'}},
                        max_tokens=source_config.get('selector_max_tokens', 16384), seed=source_config.get('selector_seed', 77113)+tid+attempt,
                        **({'stream': False} if attempt else {}))
                    decision = parse_four_way_selection(response, order)
                    save(choice_path, decision)
                    break
                except Exception as exc:
                    emit(task_id=tid, selector_error=type(exc).__name__,
                         selector_error_detail=str(exc), selector_attempt=attempt + 1)
                    if attempt == 2:
                        raise
                    selector_messages.append({'role': 'user', 'content': (
                        'The previous response was invalid: ' + str(exc) + '. '
                        'Call select_trajectory exactly once. Return candidate_index as one integer in {0,1,2,3} and a nonempty reason.'
                    )})
                finally:
                    client._client.close()
        selected = rows[decision['selected']]
        old_result = None
        if ns.candidate_source:
            old_path = ns.candidate_source / str(tid) / 'result.json'
            if old_path.exists():
                old_result = json.loads(old_path.read_text(encoding='utf-8'))
        if ns.defer_verification:
            verdict = dict(is_correct=None, verification_valid=False,
                           feedback='Verification deferred until candidate pool completes',
                           source='deferred')
        elif selected['trajectory'] == anchor['trajectory']:
            verdict = historical[tid, 'A']
        elif (old_result and old_result['decision']['selected'] == decision['selected']
              and type(old_result['bestof4'].get('is_correct')) is bool
              and old_result['bestof4'].get('feedback') != 'Verification error'):
            verdict = old_result['bestof4']
        else:
            config = registry.get(task['env_name'])
            initial = config.get('task_initial_state', '{}')
            if isinstance(initial, dict):
                initial = json.dumps(initial, ensure_ascii=False)
            for attempt in range(8):
                _, key = keys[(index + attempt * ns.workers) % len(keys)]
                client = create_client(key, URL)
                try:
                    verdict = Verifier(MODEL, client=client, num_votes=1).check(
                        task_scenario_name=task['task_scenario_name'], task_initial_state=initial,
                        state_description=config.get('state_description', ''),
                        agent_instruction=task['agent_instruction'], verification_plan=task['verification_plan'],
                        trajectory=selected['trajectory'])
                finally:
                    client.close()
                if verdict.get('feedback') != 'Verification error' and type(verdict.get('is_correct')) is bool:
                    break
            else:
                raise RuntimeError('Verifier exhausted retries; selection saved for resume')
        save(out / 'result.json', dict(task_id=tid, decision=decision, bestof4=verdict,
             vanilla=historical.get((tid, 'A')),
             eds_eca=historical.get((tid, 'FINAL'))))
        emit(task_id=tid, status='complete', selected=decision['selected'])

    with ThreadPoolExecutor(max_workers=ns.workers) as pool:
        futures = {pool.submit(run, i, tid): tid for i, tid in enumerate(ids)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                emit(task_id=futures[future], status='error', error=type(exc).__name__)
            results = [json.loads(p.read_text(encoding='utf-8')) for p in ns.out.glob('*/result.json')]
            save(ns.out / 'summary.json', summarize_results(results, len(ids)))


if __name__ == '__main__':
    main()
