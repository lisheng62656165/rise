"""No paid API: exercise the real CLI, HTTP adapter, environments and scorers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aggregate_metrics import aggregate, expected_keys

ROOT = Path(__file__).resolve().parent


def test_full_coverage_and_pass_power_five():
    keys = sorted(expected_keys())
    assert len(keys) == 150
    good = dict(task_completion_pass=1, state_requirements_met=1, task_requirements_met=1, ux_score=4)
    runs = {str(i): {k: dict(good) for k in keys} for i in range(5)}
    runs['0'][keys[0]]['task_completion_pass'] = 0
    result = aggregate(runs, 'eds_eca')
    assert result['pass_at_1'] == 749 / 750
    assert result['pass_power_5'] == 149 / 150
    del runs['1'][keys[1]]
    with pytest.raises(ValueError, match='coverage'):
        aggregate(runs, 'eds_eca')
    assert aggregate(runs, 'eds_eca', allow_partial=True)['pass_power_5'] is None


def test_all_bundled_tasks_load():
    sys.path.insert(0, str(ROOT / 'benchmark'))
    from state_bench.domain import get_domain_config
    from state_bench.env_loader import load_task_environment
    from state_bench.schemas import TaskDefinition
    from state_bench.paths import domain_tasks_dir
    from phase0_runner import DOMAINS, read_split_ids
    for domain_name in DOMAINS:
        domain = get_domain_config(domain_name)
        parts = [read_split_ids(ROOT / 'benchmark', domain_name, s) for s in ('train', 'dev', 'test')]
        assert list(map(len, parts)) == [70, 30, 50]
        assert len(set(sum(parts, []))) == 150
        for task_id in sum(parts, []):
            task = TaskDefinition.load(domain_tasks_dir(domain_name) / f'{task_id}.json')
            data, _ = load_task_environment(domain, task)
            assert domain.environment_class(data.deep_copy(), now=task.now).tool_handlers


@pytest.mark.parametrize('tasks_per_domain,seeds', [
    (1, [42]), (0, [42]), (0, [42, 142, 242, 342, 442]),
    (1, [42, 142, 242, 342, 442]),
], ids=['smoke3', 'full150', 'full150-five-runs', 'smoke3-five-run-seed-arrays'])
def test_cli_pipeline_and_resume(tmp_path, tasks_per_domain, seeds):
    calls = []
    selectors = []
    orm_requests = []
    score_requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(request)
            tools = request.get('tools') or []
            names = [t['function']['name'] for t in tools]
            messages = request['messages']
            message = {'role': 'assistant', 'content': 'Fixture response.'}
            if 'select_listwise_trajectory' in names:
                packet = json.loads(messages[-1]['content'])
                selectors.append(packet)
                # No private grading fields may enter either selector.
                assert 'state_diff' not in json.dumps(packet)
                assert 'task_completion_pass' not in json.dumps(packet)
                args = {'index': 1, 'analysis': 'Fixture public comparison.'}
                message['tool_calls'] = [{'id': 'select_1', 'type': 'function', 'function': {
                    'name': names[0], 'arguments': json.dumps(args)}}]
            elif tools and not any(m['role'] == 'tool' for m in messages):
                tool = next(t['function'] for t in tools if t['function']['name'].startswith('get_'))
                name = tool['name']
                args = {field: f"fixture_missing_{request.get('seed', 0)}" for field in tool['parameters'].get('required', [])}
                message['tool_calls'] = [{'id': 'lookup_1', 'type': 'function', 'function': {
                    'name': name, 'arguments': json.dumps(args)}}]
            elif not tools:
                system = messages[0]['content'].lower()
                if 'evaluation_guidelines:' in system:
                    from oagents_official_orm import load_prompt
                    assert messages[0]['content'] == load_prompt()
                    assert messages[-1]['content'].count('---Trajectory - ') == 4
                    assert request['temperature'] == 0 and request['max_tokens'] == 2048
                    assert request.get('stream') is not True
                    assert 'state_diff' not in messages[-1]['content']
                    assert 'task_completion_pass' not in messages[-1]['content']
                    orm_requests.append(request)
                    message['content'] = json.dumps({'index': 1, 'analysis': 'Fixture public comparison.'})
                elif 'respond as the customer' in messages[-1]['content'].lower():
                    message['content'] = '[TASK_DONE]'
                elif 'user experience (ux)' in system:
                    score_requests.append(request)
                    message['content'] = json.dumps(dict(user_control=4, user_effort=4, response_density=4))
                else:
                    if 'evaluation_guidelines:' in system:
                        score_requests.append(request)
                    message['content'] = json.dumps({'details': []})
            payload = json.dumps({'id': 'fixture', 'object': 'chat.completion', 'created': 0,
                                  'model': 'fixture', 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': message}],
                                  'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {k: v for k, v in os.environ.items() if not k.startswith('MIMO_') and k not in {'PYTHONPATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'}}
    env.update(OPENAI_API_KEY='local-test-not-a-secret', MIMO_RETRY_BACKOFF_SECONDS='0',
               PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-X', 'utf8', str(ROOT / 'run_experiment.py'), '--output-dir', str(tmp_path / 'run'),
               '--model', 'fixture', '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
               '--tasks-per-domain', str(tasks_per_domain), '--workers', '10', '--seeds', *map(str, seeds)]
    if len(seeds) == 5:
        selector_seeds = [77121, 77122, 77123, 77124, 77125]
        judge_seeds = [88021, 88022, 88023, 88024, 88025]
        command.extend(['--selector-seeds', *map(str, selector_seeds),
                        '--judge-seeds', *map(str, judge_seeds)])
    try:
        result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, encoding='utf-8', timeout=600)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        report = json.loads((tmp_path / 'run/comparison.json').read_text())
        comparison = (tmp_path / 'run/comparison.md').read_text(encoding='utf-8')
        assert '| Method | Task Completion pass@1 | pass^5 | Mean UX |' in comparison
        assert comparison.index('| Vanilla |') < comparison.index('| Modified official ORM Best-of-4 |')
        assert comparison.index('| Modified official ORM Best-of-4 |') < comparison.index('| Paper-aligned StateTrace-EDS-ECA |')
        config = json.loads((tmp_path / 'run/config.json').read_text())
        assert config['orm_selector_max_tokens'] == 2048
        assert config['task_judge_max_tokens'] == 16384
        assert config['ux_judge_max_tokens'] == 16384
        assert config['judge_streaming'] is False
        if len(seeds) == 5:
            assert config['selector_seeds'] == selector_seeds
            assert config['judge_seeds'] == judge_seeds
        expected_per_run = 3 if tasks_per_domain else 150
        expected = expected_per_run * len(seeds)
        assert all(r['num_scored_rows'] == expected and r['partial'] == bool(tasks_per_domain) for r in report.values())
        from aggregate_metrics import load_run
        for method, metrics in report.items():
            runs = [load_run(f'{seed}={tmp_path / "run" / f"seed_{seed}" / "scores" / method}')[1]
                    for seed in seeds]
            assert all(len(rows) == expected_per_run for rows in runs)
            outcomes = [row for rows in runs for row in rows.values()]
            assert metrics['pass_at_1'] == sum(r['task_completion_pass'] for r in outcomes) / expected
            assert metrics['ux_mean'] == pytest.approx(sum(r['ux_score'] for r in outcomes) / expected)
            if len(seeds) == 5 and not tasks_per_domain:
                all_five = sum(all(rows[key]['task_completion_pass'] for rows in runs) for key in runs[0])
                assert metrics['pass_power_5_count'] == all_five
                assert metrics['pass_power_5'] == all_five / 150
            else:
                assert metrics['pass_power_5'] is None
        assert {len(p['candidates']) for p in selectors} == {2}
        assert len(orm_requests) == expected
        assert score_requests
        assert all(r['max_tokens'] == 16384 and r.get('stream') is not True for r in score_requests), score_requests
        before = len(calls)
        result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, encoding='utf-8', timeout=120)
        assert result.returncode == 0, result.stderr
        assert len(calls) == before, 'Resume regenerated completed work'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
