from adaptive_request_monitor import AdaptiveSlots


def test_ramp_and_rate_limit():
    now = [0]
    slots = AdaptiveSlots(10, 12, clock=lambda: now[0])
    for _ in range(30):
        slots.observe('complete')
    assert slots.limit == 10
    now[0] = 181
    assert slots.observe('complete') == (10, 12)
    assert slots.observe('error', 'RateLimitError') == (12, 6)
    assert slots.cooldown == 241
    slots.observe('error', 'RateLimitError')
    assert slots.limit == 6


def test_error_resets_ramp():
    now = [0]
    slots = AdaptiveSlots(10, 30, clock=lambda: now[0])
    now[0] = 200
    slots.observe('error', 'APIError')
    for _ in range(35):
        slots.observe('complete')
    assert slots.limit == 10
    slots.acquire()
    assert slots.active == 1
    slots.release()
    assert slots.active == 0


def test_overload_cooldown_and_recovery():
    now = [100]
    slots = AdaptiveSlots(4, 4, clock=lambda: now[0])
    assert slots.observe('error', 'APIError', overloaded=True) == (4, 2)
    assert slots.cooldown == 160
    slots.observe('error', 'APIError', overloaded=True)
    assert slots.limit == 2
    now[0] = 161
    slots.acquire()
    slots.release()
    now[0] = 281
    for _ in range(30):
        slots.observe('complete')
    assert slots.limit == 4
