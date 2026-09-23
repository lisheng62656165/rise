import json
from unittest.mock import patch
from pathlib import Path
from src.config import Settings
from src.llm_client import LLMClient
from src.appworld_mimo import extract_python_code
from summarize import metrics
from scripts.select_appworld_oagents_bon4 import ORM_LIST_WISE_PROMPT, selector_input


def test_complete_scenario_only():
    ids = ['s_1', 's_2', 's_3', 't_1', 't_2', 't_3']
    result = metrics({'s_1': True, 's_2': True, 's_3': True, 't_1': True}, ids)
    assert result['TGC'] == 100
    assert result['SGC'] == 100 and result['scenarios'] == 1
    assert metrics({'s_1': True}, ids)['SGC'] is None


def test_no_outcome_in_oagents_prompt():
    candidate = {'trace': [{'assistant_text': 'hello', 'code': 'print(1)', 'execution_output': '1'}],
                 'evaluation_success': True, 'evaluation': {'secret_gold': 123}}
    text = selector_input([candidate] * 4)
    assert 'secret_gold' not in text and 'evaluation_success' not in text
    assert text.count('---Trajectory - ') == 4


def test_upstream_prompt_verbatim():
    import yaml
    path = Path(__file__).resolve().parents[1] / 'third_party/OAgents/ORM_list_wise.yaml'
    assert yaml.safe_load(path.read_text())['prompt'].strip() == ORM_LIST_WISE_PROMPT.strip()


def test_python_extraction():
    assert extract_python_code('reason\n```python\nprint(1)\n```') == 'print(1)'


def test_openai_reasoning_payload(monkeypatch):
    monkeypatch.setenv('OPENAI_REASONING', 'auto')
    settings = Settings('https://api.openai.com/v1', 'fake', 'gpt-5', seed=42)
    class Limiter:
        def acquire(self, *args): pass
        def record(self, *args): pass
        def release(self): pass
    class Response:
        status_code = 200
        text = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]})
        def raise_for_status(self): pass
    with patch('src.llm_client.requests.post', return_value=Response()) as call:
        result = LLMClient(settings, Limiter()).complete_messages_with_trace(
            [{'role': 'user', 'content': 'hi'}], {'max_tokens': 3000, 'temperature': 0})
    payload = call.call_args.kwargs['json']
    assert payload['max_completion_tokens'] == 3000
    assert not {'seed', 'temperature', 'max_tokens'} & payload.keys()
    assert result.text == 'ok'


def test_faithful_keeps_public_credit_not_outcomes(monkeypatch):
    from types import SimpleNamespace
    from scripts import run_appworld_state_trace_eds_eca as runner
    from src.appworld_state_trace_eds_eca_adapter import compile_appworld_public_events
    from src.appworld_state_trace_eds_eca import proposal_messages
    left = {'public_events': compile_appworld_public_events([
        {'code': 'apis.mail.send(to="x")', 'execution_output': 'Error: invalid target'}]),
        'evaluation_success': True, 'evaluation': {'private_label': 'FORBIDDEN_SENTINEL'}}
    right = {'public_events': compile_appworld_public_events([
        {'code': 'apis.mail.search(query="x")', 'execution_output': '[{"id":1}]'}]),
        'evaluation_success': False, 'evaluation': {'private_label': 'FORBIDDEN_SENTINEL'}}
    class FakeClient:
        def __init__(self, settings): pass
        def complete_messages_with_trace(self, messages, **kwargs):
            assert 'FORBIDDEN_SENTINEL' not in str(messages)
            packet = json.loads(messages[1]['content'])
            a = packet['candidate_A']['public_events'][0]['event_id']
            b = packet['candidate_B']['public_events'][0]['event_id']
            answer = {'choice': 'B', 'supporting_event_ids': [b], 'preserve_event_ids': [b],
                      'avoid_event_ids': [a], 'unresolved_issue_type': 'FAILURE_RECOVERY', 'reason': 'public evidence'}
            return SimpleNamespace(text=json.dumps(answer), request={}, response={})
    monkeypatch.setattr(runner, 'LLMClient', FakeClient)
    monkeypatch.setattr(runner, 'settings', lambda seed: Settings('https://api.openai.com/v1', 'fake', 'gpt-4.1'))
    selected, credit, details = runner.choose('send mail', left, right, 0, 2048, 0, 0, None, None)
    assert selected is right and details['choice'] == 'B'
    assert not details['fallback'] and credit['available']
    assert len(credit['preserve_events']) == len(credit['avoid_events']) == 1
    assert 'arguments' not in credit['preserve_events'][0]
    assert 'explicit_result' not in credit['avoid_events'][0]
    assert credit['active_intervention']['issue_type'] == 'FAILURE_RECOVERY'
    assert 'FORBIDDEN_SENTINEL' not in str(proposal_messages([{'role': 'user', 'content': 'task'}], credit, 'C'))


def test_shared_vanilla_mismatch_detected(tmp_path):
    from run import share_a
    import pytest
    a, b = tmp_path / 'a', tmp_path / 'b'
    (a / 't').mkdir(parents=True)
    (a / 't/candidate_a.json').write_text('{"value":1}')
    share_a(a, b, ['t'])
    assert (b / 't/candidate_a.json').read_bytes() == (a / 't/candidate_a.json').read_bytes()
    (b / 't/candidate_a.json').write_text('{"value":2}')
    with pytest.raises(RuntimeError):
        share_a(a, b, ['t'])


def test_main_experiment_seed_defaults_match_challenge_record(tmp_path, monkeypatch):
    import run as driver
    import sys
    from types import ModuleType

    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'test_challenge_main.json'
    monkeypatch.setattr('sys.argv', [
        'run.py', '--config', str(config_path), '--output', str(tmp_path / 'out'), '--method', 'faithful-v3'
    ])
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('MODEL_NAME', 'deepseek-v4.1-flash')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.invalid/v1')

    commands = []

    def fake_execute(script, arguments):
        commands.append((script, arguments))
        output_dir = Path(arguments[arguments.index('--output-dir') + 1])
        (output_dir / 'task_1').mkdir(parents=True, exist_ok=True)
        (output_dir / 'task_1' / 'final.json').write_text(
            json.dumps({'selected_success': True, 'vanilla_success': True})
        )

    monkeypatch.setattr(driver, 'execute', fake_execute)
    monkeypatch.setattr(driver.subprocess, 'run', lambda *args, **kwargs: None)

    fake_appworld = ModuleType('appworld')
    fake_appworld.load_task_ids = lambda split: ['task_1']
    monkeypatch.setitem(sys.modules, 'appworld', fake_appworld)
    monkeypatch.setattr('src.config.get_settings', lambda: Settings(
        'https://example.invalid/v1', 'test-only', 'deepseek-v4.1-flash'))
    driver.main()

    record = json.loads((tmp_path / 'out/test_challenge/experiment.json').read_text())
    assert record['seed_a'] == 53403
    assert record['proposal_seeds'] == [64639, 64640, 64641]
    assert record['faithful_selector_seed'] == 77113
    assert record['oagents_selector_seed'] == 53403
    assert record['config_name'] == 'appworld-test-challenge-deepseek-main'


def test_normal_and_challenge_share_main_config():
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'appworld_main.json').read_text())
    assert config['split'] == 'both'
    assert config['task_counts'] == {'test_normal': 168, 'test_challenge': 417}
    assert config['scenario_counts'] == {'test_normal': 56, 'test_challenge': 139}
    assert config['seeds']['vanilla_anchor_and_oagents_A'] == 53403
    assert config['seeds']['faithful_proposals_BCD'] == [64639, 64640, 64641]
    assert config['seeds']['faithful_selector_base'] == 77113
    assert config['seeds']['oagents_selector_base'] == 53403
    assert config['metric_columns'] == ['Test-N TGC', 'Test-N SGC', 'Test-C TGC', 'Test-C SGC']
