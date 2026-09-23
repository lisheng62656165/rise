from types import SimpleNamespace
import pytest
from occubench_request_monitor import RequestMonitor


def client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)


def test_request_exception_releases_slot(tmp_path):
    monitor = RequestMonitor(1, tmp_path/'requests.jsonl')
    def fail(**kwargs):
        raise TimeoutError()
    with pytest.raises(TimeoutError):
        monitor.wrap(client(fail)).chat.completions.create()
    assert monitor.slots.acquire(blocking=False)


def test_stream_holds_slot_until_consumed(tmp_path):
    monitor = RequestMonitor(1, tmp_path/'requests.jsonl')
    stream = monitor.wrap(client(lambda **kw: iter_stream())).chat.completions.create(stream=True)
    assert not monitor.slots.acquire(blocking=False)
    assert list(stream) == [1, 2]
    assert monitor.slots.acquire(blocking=False)


def iter_stream():
    yield 1
    yield 2
