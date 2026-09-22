from types import SimpleNamespace

from reliable_mimo_client import ReliableMiMoClient


def test_provider_conversion_is_retry_safe(monkeypatch):
    monkeypatch.setenv('MIMO_SEND_SEED', '0')
    monkeypatch.setenv('MIMO_SEND_TEMPERATURE', '0')
    monkeypatch.setenv('MIMO_TOKEN_PARAMETER', 'max_completion_tokens')
    client = object.__new__(ReliableMiMoClient)
    request = dict(max_tokens=128, seed=42, temperature=0, model='fixture')
    for _ in range(3):
        client._compatible_kwargs(request)
    assert request == dict(max_completion_tokens=128, model='fixture')


def test_chat_retry_preserves_provider_parameters(monkeypatch):
    monkeypatch.setenv('MIMO_TOKEN_PARAMETER', 'max_completion_tokens')
    monkeypatch.setenv('MIMO_STREAMING', '0')
    monkeypatch.setenv('MIMO_JSON_RESPONSE_FORMAT', '0')
    client = object.__new__(ReliableMiMoClient)
    client.model = 'fixture'
    client.max_tokens = 128
    client.temperature = 0
    client.seed = 42
    client.reasoning_effort = None
    client.thinking_enabled = False
    client.retry_attempts = 2
    client.retry_backoff_seconds = 0
    client._hard_timeout_seconds = 0
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            raise TimeoutError('timed out')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='OK'))])

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert client.complete_chat([{'role': 'user', 'content': 'hello'}]) == 'OK'
    assert requests[0] == requests[1]
    assert requests[1]['max_completion_tokens'] == 128
