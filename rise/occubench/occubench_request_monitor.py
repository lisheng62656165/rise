"""Bound in-flight requests, logging stream completion rather than just headers."""
import json
import threading
import time
from types import SimpleNamespace


class RequestMonitor:
    def __init__(self, limit, path):
        self.slots = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.path = path
        self.next_id = 0

    def log(self, **data):
        with self.lock:
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(time=time.time(), thread=threading.current_thread().name, **data))+'\n')

    def wrap(self, client):
        monitor = self

        class Completions:
            def create(self, **kwargs):
                with monitor.lock:
                    monitor.next_id += 1
                    rid = monitor.next_id
                monitor.log(request=rid, status='queued')
                monitor.slots.acquire()
                start = time.monotonic()
                monitor.log(request=rid, status='started', model=kwargs.get('model'))
                try:
                    response = client.chat.completions.create(**kwargs)
                except BaseException as exc:
                    monitor.log(request=rid, status='error', error=type(exc).__name__,
                                overloaded='temporarily overloaded' in str(exc).lower())
                    monitor.slots.release()
                    raise
                if not kwargs.get('stream'):
                    monitor.log(request=rid, status='complete', seconds=time.monotonic()-start)
                    monitor.slots.release()
                    return response

                def chunks():
                    count = 0
                    try:
                        for chunk in response:
                            count += 1
                            if count == 1 or count % 256 == 0:
                                monitor.log(request=rid, status='streaming', chunks=count)
                            yield chunk
                        monitor.log(request=rid, status='complete', seconds=time.monotonic()-start, chunks=count)
                    except BaseException as exc:
                        monitor.log(request=rid, status='error', error=type(exc).__name__, chunks=count,
                                    overloaded='temporarily overloaded' in str(exc).lower())
                        raise
                    finally:
                        try:
                            response.close()
                        finally:
                            monitor.slots.release()
                return chunks()

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()),
                               close=client.close, _client=client)
