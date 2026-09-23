from types import SimpleNamespace as NS

import pytest
from oagents_bon_selector import messages, parse, format_retry_messages


def response(content, finish='stop'):
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish)])


def test_official_text_json():
    assert parse(response('```json\n{"index": 3, "analysis": "ok"}\n```'))['selected'] == 3


@pytest.mark.parametrize('content', ['', '{}', '{"index": true}', '{"index": 4}', '{"index": "1"}'])
def test_invalid_never_selects_zero(content):
    with pytest.raises(ValueError):
        parse(response(content))


def test_truncated_output_rejected():
    with pytest.raises(ValueError):
        parse(response('{"index": 0}', 'length'))


def test_public_only_and_official_prompt():
    rows = [{'trajectory': f'public-{i}', 'reward': 'SECRET'} for i in range(4)]
    result = messages({'agent_instruction': 'visible task', 'verification_plan': 'SECRET'}, rows)
    assert 'SECRET' not in str(result)
    assert 'Evaluation_Guidelines:' in result[0]['content']
    assert result[1]['content'].count('---Trajectory - ') == 4
    assert result[1]['content'].endswith('you can start!')


def test_format_retry_keeps_evidence_and_criteria():
    original = messages({'agent_instruction': 'visible'}, [{'trajectory': str(i)} for i in range(4)])
    retried = format_retry_messages(original)
    assert len(original) == 2
    assert retried[:2] == original
    assert 'original Evaluation_Guidelines' in retried[-1]['content']
