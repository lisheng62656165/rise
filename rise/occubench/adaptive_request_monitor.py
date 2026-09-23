"""Conservative request-level ramp with shared rate-limit cooldown."""
import threading
import time

from occubench_request_monitor import RequestMonitor


class AdaptiveSlots:
    def __init__(self, initial, maximum, clock=time.monotonic):
        if not 1 <= initial <= maximum:
            raise ValueError('Expected 1 <= initial <= maximum')
        self.limit, self.maximum, self.active = initial, maximum, 0
        self.clock = clock
        self.window = clock()
        self.successes = 0
        self.cooldown = 0
        self.condition = threading.Condition()

    def acquire(self):
        with self.condition:
            while self.active >= self.limit or self.clock() < self.cooldown:
                self.condition.wait(timeout=1)
            self.active += 1

    def release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def observe(self, status, error=None, overloaded=False):
        with self.condition:
            now, old = self.clock(), self.limit
            if status == 'error':
                self.successes, self.window = 0, now
                if error == 'RateLimitError' or overloaded:
                    if now >= self.cooldown:
                        self.limit = max(1, self.limit // 2)
                    self.cooldown = max(self.cooldown, now + 60)
            elif status == 'complete':
                self.successes += 1
                if now - self.window >= 180 and self.successes >= 30:
                    self.limit = min(self.maximum, self.limit + 2)
                    self.successes, self.window = 0, now
            self.condition.notify_all()
            return (old, self.limit) if old != self.limit else None


class AdaptiveRequestMonitor(RequestMonitor):
    def __init__(self, initial, maximum, path):
        super().__init__(initial, path)
        self.slots = AdaptiveSlots(initial, maximum)

    def log(self, **data):
        change = self.slots.observe(data.get('status'), data.get('error'), data.get('overloaded', False))
        super().log(**data)
        if change:
            super().log(status='concurrency_adjusted', previous=change[0], limit=change[1])
