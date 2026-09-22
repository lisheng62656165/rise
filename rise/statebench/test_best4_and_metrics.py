import json
import copy
from types import SimpleNamespace

import pytest

from aggregate_metrics import load_run
from run_oagents_best_of4 import select_one
from oagents_official_orm import build_messages, load_prompt, parse_choice, select_with_retries, PROTOCOL


def test_best4_selector_is_a_separate_four_way_protocol():
    import run_oagents_best_of4 as module

    assert module.PROTOCOL == 'oagents_official_orm_listwise_statebench_v1'
    assert 'Evaluation_Guidelines:' in load_prompt()
    assert not hasattr(module, 'OAGENTS_BON_LISTWISE_INSTRUCTION')


def test_official_packet_is_public_and_fixed_order():
    rows = [dict(conversation=[{'role': 'user', 'content': f'candidate-{i}'}],
                 state_diff='PRIVATE', task_completion_pass=1, public_trace_ir='EDS_FEATURE') for i in range(4)]
    messages = build_messages(load_prompt(), rows)
    assert messages[0]['content'] == load_prompt()
    text = messages[1]['content']
    assert text.endswith('you can start!')
    assert text.count('---Trajectory - ') == 4
    assert [text.index(f'candidate-{i}') for i in range(4)] == sorted(text.index(f'candidate-{i}') for i in range(4))
    assert 'PRIVATE' not in text and 'EDS_FEATURE' not in text


@pytest.mark.parametrize('text', [
    '{"index":true,"analysis":"x"}', '{"index":4,"analysis":"x"}',
    '{"index":-1,"analysis":"x"}', '{"index":0,"analysis":""}',
    '{"index":0,"analysis":"x"}{"index":1,"analysis":"y"}', '{}',
])
def test_invalid_official_choices_do_not_default_to_anchor(text):
    with pytest.raises(ValueError):
        parse_choice(text)


def test_official_selector_same_input_retry(tmp_path):
    calls = []

    class Client:
        def complete_chat(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return 'invalid' if len(calls) == 1 else '```json\n{"index":3,"analysis":"best"}\n```'

    choice, attempts = select_with_retries(Client(), build_messages(load_prompt(), [{'conversation': []}] * 4), tmp_path / 'audit.json')
    assert choice['index'] == 3 and attempts == 2
    assert calls[0] == calls[1]
    assert calls[0]['max_tokens'] == 2048 and calls[0]['temperature'] == 0
    assert 'tools' not in calls[0]


def test_select_one_uses_official_protocol(monkeypatch, tmp_path):
    import reliable_mimo_client
    closed = []
    client = SimpleNamespace(model='fixture', complete_chat=lambda **kwargs: '{"index":2,"analysis":"best"}',
                             _client=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(reliable_mimo_client.ReliableMiMoClient, 'from_env', lambda: client)
    args = SimpleNamespace(statebench_root=tmp_path, output_dir=tmp_path, selector_seed=11, selector_retries=2)
    result = select_one(args, 'statebench::travel::t1', [{'conversation': []}] * 4, 7)
    assert result['selector_protocol'] == PROTOCOL and result['selected_candidate'] == 2
    assert result['selector_seed'] == 18 and result['shown_order'] == [0, 1, 2, 3]
    assert closed == [True]


def test_old_best4_finals_are_not_reused(monkeypatch, tmp_path):
    import sys
    import run_oagents_best_of4 as module
    monkeypatch.setenv('MIMO_API_KEY', 'local-test-not-a-secret')
    monkeypatch.setattr(sys, 'argv', ['runner', '--seed', '42', '--output-dir', str(tmp_path)])
    monkeypatch.setattr(module, 'selected_jobs', lambda args: [('travel', 't1')])
    monkeypatch.setattr(module, 'read_split_ids', lambda *args: ['t1'])
    final = tmp_path / 'final/travel__t1.json'
    final.parent.mkdir()
    final.write_text(json.dumps({'best_of4_selection': {'selected_candidate': 0}}))
    with pytest.raises(ValueError, match='older selector'):
        module.main()


def test_aggregate_loader_accepts_score_rows(tmp_path):
    row = {
        "task_key": "statebench::travel::t1",
        "task_completion_pass": 1,
        "state_requirements_met": 1,
        "task_requirements_met": 1,
        "ux_score": 4.0,
    }
    (tmp_path / "t1.json").write_text(json.dumps(row), encoding="utf-8")
    label, rows = load_run(f"seed1={tmp_path}")
    assert label == "seed1"
    assert rows["statebench::travel::t1"]["ux_score"] == 4.0


def test_metrics_use_all_five_runs_and_ux_rows():
    from aggregate_metrics import aggregate, expected_keys
    keys = sorted(expected_keys())
    runs = {str(seed): {key: dict(task_completion_pass=int(seed != 0),
                                state_requirements_met=1, task_requirements_met=1,
                                ux_score=seed + 1) for key in keys} for seed in range(5)}
    result = aggregate(runs, 'best4')
    assert result['num_scored_rows'] == 750
    assert result['pass_at_1'] == 0.8
    assert result['pass_power_5'] == 0
    assert result['ux_mean'] == 3
    assert aggregate({'0': runs['0']}, 'best4')['pass_power_5'] is None
