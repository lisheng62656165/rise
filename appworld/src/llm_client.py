from __future__ import annotations
import copy
import json
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any
import requests
from openai import OpenAI
from .config import Settings


def _steady_time() -> float:
    # APPWorld freezes Python's wall and monotonic clocks while simulating a
    # task. os.times() is backed by the OS elapsed clock and keeps advancing.
    return os.times().elapsed


def _diagnostic(event: str, **fields: Any) -> None:
    if os.getenv("LLM_RETRY_DIAGNOSTICS", "0").lower() not in {"1", "true", "yes"}:
        return
    values = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        f"LLM_DIAG steady={_steady_time():.3f} pid={os.getpid()} event={event} {values}".rstrip(),
        file=sys.stderr,
        flush=True,
    )
@dataclass(frozen=True)
class LLMResult:
    text: str
    text_source: str
    request: dict
    response: dict
    message: dict


class _DictChunk:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def model_dump(self) -> dict[str, Any]:
        return self.data


class GlobalRateLimiter:
    def __init__(self, max_rpm=80, max_tpm=8000000):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self._lock = threading.Lock()
        self._request_times = deque()
        self._token_times = deque()
        self._request_semaphore = threading.Semaphore(max_rpm)
    def _clean_old_entries(self, now):
        while self._request_times and self._request_times[0] < now - 60:
            self._request_times.popleft()
        while self._token_times and self._token_times[0][0] < now - 60:
            self._token_times.popleft()
    def acquire(self, estimated_tokens=5000):
        while True:
            with self._lock:
                # APPWorld freezes Python wall and monotonic clocks inside a
                # task sandbox. The limiter needs an unaffected OS clock.
                now = _steady_time()
                self._clean_old_entries(now)
                current_rpm = len(self._request_times)
                current_tpm = sum(t for _, t in self._token_times)
                if current_rpm < self.max_rpm and current_tpm + estimated_tokens <= self.max_tpm:
                    # Reserve the request while holding the lock. The old
                    # fast path only checked the counters, so all workers
                    # could pass simultaneously and trigger API 429 storms.
                    self._request_times.append(now)
                    self._token_times.append((now, estimated_tokens))
                    return
                waits = []
                if current_rpm >= self.max_rpm and self._request_times:
                    waits.append(60 - (now - self._request_times[0]))
                if current_tpm + estimated_tokens > self.max_tpm and self._token_times:
                    oldest = self._token_times[0][0]
                    waits.append(60 - (now - oldest))
                if waits:
                    wait_time = max(1, min(waits))
                else:
                    wait_time = 1
            _diagnostic("rate_limit_sleep", seconds=f"{wait_time:.3f}", rpm=current_rpm, tpm=current_tpm)
            time.sleep(wait_time)
    def record(self, prompt_tokens, completion_tokens):
        # acquire() already reserves an estimated token budget. Appending
        # actual usage here would double-count every request and can stall
        # all workers under the TPM limit.
        return None
    def release(self):
        # Kept for API compatibility with existing callers. Request slots
        # are timestamp reservations and expire from _request_times.
        return None
_global_rate_limiter = None
_rate_limiter_lock = threading.Lock()
def get_global_rate_limiter(settings=None):
    global _global_rate_limiter
    with _rate_limiter_lock:
        if _global_rate_limiter is None:
            max_rpm = getattr(settings, "max_rpm", 80) if settings else 80
            max_tpm = getattr(settings, "max_tpm", 8000000) if settings else 8000000
            phase_window = float(os.getenv("RATE_LIMIT_START_JITTER_SECONDS", "0"))
            if phase_window > 0:
                # Process workers start together; phase them once so their
                # independent RPM limiters do not create synchronized bursts.
                phase = ((os.getpid() * 7919) % 10000) / 10000 * phase_window
                time.sleep(phase)
            _global_rate_limiter = GlobalRateLimiter(max_rpm=max_rpm, max_tpm=max_tpm)
        return _global_rate_limiter
def reset_global_rate_limiter():
    global _global_rate_limiter
    with _rate_limiter_lock:
        _global_rate_limiter = None
class LLMClient:
    def __init__(self, settings, rate_limiter=None):
        # Start each process at a different key; retryable NVIDIA failures may
        # advance within the pool without changing model output parameters.
        self._api_keys = [settings.api_key]
        self._api_key_offset = 0
        if os.getenv('NVIDIA_KEY_POOL', '0').lower() in {'1', 'true', 'yes'}:
            key_file = os.getenv('NVIDIA_KEY_FILE', '')
            try:
                keys = [line.strip() for line in open(key_file, encoding='utf-8') if line.strip()]
                if keys:
                    self._api_keys = keys
                    self._api_key_offset = os.getpid() % len(keys)
                    selected = keys[self._api_key_offset]
                    settings = dataclass_replace(settings, api_key=selected)
            except (OSError, TypeError):
                pass
        self.settings = settings
        self.endpoint = settings.base_url.rstrip('/') + '/chat/completions'
        self.rate_limiter = rate_limiter or get_global_rate_limiter(settings)

    def _api_key_for_attempt(self, attempt: int) -> str:
        return self._api_keys[(self._api_key_offset + attempt) % len(self._api_keys)]

    def _nvidia_sse_chunks(self, payload: dict[str, Any]):
        last_error = None
        for attempt in range(self.settings.request_retries + 1):
            headers = {
                'Authorization': f'Bearer {self._api_key_for_attempt(attempt)}',
                'Content-Type': 'application/json',
            }
            try:
                with requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=self.settings.request_timeout,
                ) as response:
                    if response.status_code in {408, 429, 500, 502, 503, 504}:
                        if attempt < self.settings.request_retries:
                            time.sleep(min(60, 2 ** attempt * 5))
                            continue
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith('data:'):
                            continue
                        value = line[5:].strip()
                        if value == '[DONE]':
                            return
                        yield _DictChunk(json.loads(value))
                    return
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.settings.request_retries:
                    raise
                time.sleep(min(60, 2 ** attempt * 5))
        if last_error is not None:
            raise last_error

    def complete_messages_with_trace(self, messages, extra_payload=None, allow_reasoning_fallback=False):
        omit_temperature = bool((extra_payload or {}).get('_omit_temperature'))
        max_token_field = 'max_completion_tokens' if self.settings.model_name.startswith('mimo-') else 'max_tokens'
        payload = {'model': self.settings.model_name, 'messages': copy.deepcopy(messages), max_token_field: self.settings.max_completion_tokens}
        if not omit_temperature:
            payload['temperature'] = self.settings.temperature
        if self.settings.top_p is not None:
            payload['top_p'] = self.settings.top_p
        if self.settings.frequency_penalty is not None:
            payload['frequency_penalty'] = self.settings.frequency_penalty
        if self.settings.presence_penalty is not None:
            payload['presence_penalty'] = self.settings.presence_penalty
        if self.settings.seed is not None:
            payload['seed'] = self.settings.seed
        if extra_payload:
            payload.update({k: v for k, v in extra_payload.items() if not k.startswith('_')})
        # Normalize after selector overrides so actor and verifier use the same
        # provider dialect. Reasoning models do not accept sampling/seed fields.
        reasoning = os.getenv('OPENAI_REASONING', 'auto').lower()
        is_reasoning = reasoning == '1' or (reasoning == 'auto' and
            self.settings.model_name.lower().startswith(('o1', 'o3', 'o4', 'gpt-5')))
        if 'api.openai.com' in self.settings.base_url or is_reasoning:
            budget = payload.pop('max_tokens', payload.get('max_completion_tokens', self.settings.max_completion_tokens))
            payload['max_completion_tokens'] = budget
        if is_reasoning:
            for key in ('temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'seed'):
                payload.pop(key, None)
            if os.getenv('REASONING_EFFORT'):
                payload['reasoning_effort'] = os.environ['REASONING_EFFORT']
        if os.getenv('SEND_SEED', '1') == '0':
            payload.pop('seed', None)
        original_messages = copy.deepcopy(messages)
        if os.getenv('NVIDIA_CONTEXT_COMPACT', '0').lower() in {'1', 'true', 'yes'}:
            # Keep the prompt bounded before the provider rejects it. This is
            # opt-in for targeted repair runs so the main benchmark is unchanged.
            encoded_size = len(json.dumps(payload.get('messages', []), ensure_ascii=False))
            if encoded_size > 48000 and len(payload.get('messages', [])) > 13:
                payload['messages'] = payload['messages'][:1] + payload['messages'][-12:]
                _diagnostic('context_messages_pretrim', old_size=encoded_size, new_count=len(payload['messages']))
        self.rate_limiter.acquire(5000)
        if 'integrate.api.nvidia.com' in self.settings.base_url:
            sdk_payload = copy.deepcopy(payload)
            extra_body = {}
            chat_template_kwargs = sdk_payload.pop('chat_template_kwargs', None)
            thinking = sdk_payload.pop('thinking', None)
            if thinking is not None:
                disabled = isinstance(thinking, dict) and str(thinking.get('type', '')).lower() in {'disabled', 'off', 'none'}
                chat_template_kwargs = {'enable_thinking': not disabled}
            elif os.getenv('NVIDIA_ENABLE_THINKING', '0').lower() in {'1', 'true', 'yes'}:
                chat_template_kwargs = {'enable_thinking': True}
            if chat_template_kwargs is not None:
                extra_body['chat_template_kwargs'] = chat_template_kwargs
            reasoning_budget = sdk_payload.pop('reasoning_budget', None)
            if reasoning_budget is None and chat_template_kwargs and chat_template_kwargs.get('enable_thinking'):
                reasoning_budget = int(os.getenv('NVIDIA_REASONING_BUDGET', '16384'))
            if reasoning_budget is not None:
                extra_body['reasoning_budget'] = reasoning_budget
            try:
                use_stream = os.getenv('NVIDIA_STREAM', '0').lower() in {'1', 'true', 'yes'}
                if use_stream:
                    stream_payload = copy.deepcopy(sdk_payload)
                    stream_payload.update(extra_body)
                    stream_payload['stream'] = True
                    completion = self._nvidia_sse_chunks(stream_payload)
                else:
                    last_error = None
                    for attempt in range(self.settings.request_retries + 1):
                        request_started = _steady_time()
                        _diagnostic("request_start", attempt=attempt)
                        try:
                            with OpenAI(
                                base_url=self.settings.base_url,
                                api_key=self._api_key_for_attempt(attempt),
                                timeout=self.settings.request_timeout,
                                max_retries=0,
                            ) as client:
                                completion = client.chat.completions.create(
                                    **sdk_payload,
                                    **({'extra_body': extra_body} if extra_body else {}),
                                    stream=False,
                                )
                            _diagnostic(
                                "request_success",
                                attempt=attempt,
                                elapsed=f"{_steady_time() - request_started:.3f}",
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            status = getattr(exc, 'status_code', None)
                            # NVIDIA returns this variant when the prompt consumes the
                            # available context before validating max_tokens.  Treat it
                            # like context overflow and retry with a smaller completion.
                            error_text = str(exc)
                            if status == 400 and 'max_tokens must be at least 1' in error_text:
                                kept = original_messages[:1] + original_messages[-12:]
                                if len(kept) < len(original_messages):
                                    sdk_payload['messages'] = kept
                                    sdk_payload['max_tokens'] = min(int(sdk_payload.get('max_tokens') or 4096), 1024)
                                    _diagnostic('context_messages_trim_retry', old_count=len(original_messages), new_count=len(kept))
                                    continue
                            if (
                                status == 400
                                and 'max_tokens must be at least 1' in error_text
                                and int(sdk_payload.get('max_tokens') or 0) > 1
                            ):
                                current_max = int(sdk_payload['max_tokens'])
                                sdk_payload['max_tokens'] = max(1, current_max // 2)
                                _diagnostic(
                                    'context_overflow_retry',
                                    attempt=attempt,
                                    old_max_tokens=current_max,
                                    new_max_tokens=sdk_payload['max_tokens'],
                                )
                                continue
                            if status == 400 and ('context' in error_text.lower() or 'maximum context' in error_text.lower()):
                                # Preserve system context and the latest turn while
                                # dropping only the oldest conversational history.
                                kept = original_messages[:1] + original_messages[-12:]
                                if len(kept) < len(original_messages):
                                    sdk_payload['messages'] = kept
                                    sdk_payload['max_tokens'] = max(1, min(int(sdk_payload.get('max_tokens') or 1), 1024))
                                    _diagnostic('context_messages_trim_retry', old_count=len(original_messages), new_count=len(kept))
                                    continue
                            _diagnostic(
                                "request_error",
                                attempt=attempt,
                                elapsed=f"{_steady_time() - request_started:.3f}",
                                error=type(exc).__name__,
                                status=status,
                            )
                            retryable_exception = type(exc).__name__ in {
                                'APITimeoutError', 'APIConnectionError', 'ConnectTimeout',
                                'ReadTimeout', 'RemoteProtocolError',
                            }
                            if (status not in (408, 429, 500, 502, 503, 504)
                                    and not retryable_exception) or attempt >= self.settings.request_retries:
                                raise
                            sleep_seconds = min(60, 2 ** attempt * 5)
                            _diagnostic("retry_sleep", attempt=attempt, seconds=sleep_seconds)
                            time.sleep(sleep_seconds)
                    else:
                        raise last_error
                if not use_stream:
                    data = completion.model_dump()
                    choices = data.get('choices') or []
                    if not choices:
                        raise RuntimeError('NVIDIA SDK response has no choices')
                    message = choices[0].get('message') or {}
                    content = message.get('content') or ''
                    reasoning_content = message.get('reasoning_content') or message.get('reasoning') or ''
                    if content or not allow_reasoning_fallback:
                        text, text_source = content, 'content'
                    else:
                        text, text_source = reasoning_content, 'reasoning_content_fallback'
                    usage = data.get('usage') or {}
                    self.rate_limiter.record(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
                    return LLMResult(
                        text=text,
                        text_source=text_source,
                        message=message,
                        request={'endpoint': self.endpoint, 'headers': {'api-key': '***', 'Authorization': 'Bearer ***'}, 'json': payload},
                        response={'status_code': 200, 'usage': usage, 'choices': choices, 'id': data.get('id'), 'created': data.get('created')},
                    )
                content_parts = []
                reasoning_parts = []
                tool_calls = {}
                finish_reason = None
                usage = {}
                response_id = None
                response_model = None
                created = None
                for chunk in completion:
                    chunk_data = chunk.model_dump()
                    response_id = response_id or chunk_data.get('id')
                    response_model = response_model or chunk_data.get('model')
                    created = created or chunk_data.get('created')
                    usage = chunk_data.get('usage') or usage
                    for choice in chunk_data.get('choices') or []:
                        finish_reason = choice.get('finish_reason') or finish_reason
                        delta = choice.get('delta') or {}
                        if delta.get('content') is not None:
                            content_parts.append(delta['content'])
                        if delta.get('reasoning_content') is not None:
                            reasoning_parts.append(delta['reasoning_content'])
                        for call in delta.get('tool_calls') or []:
                            index = call.get('index', len(tool_calls))
                            item = tool_calls.setdefault(index, {
                                'id': call.get('id'),
                                'type': call.get('type') or 'function',
                                'function': {'name': '', 'arguments': ''},
                            })
                            if call.get('id'):
                                item['id'] = call['id']
                            if call.get('type'):
                                item['type'] = call['type']
                            fn = call.get('function') or {}
                            if fn.get('name'):
                                item['function']['name'] += fn['name']
                            if fn.get('arguments'):
                                item['function']['arguments'] += fn['arguments']
                message = {'role': 'assistant', 'content': ''.join(content_parts)}
                if reasoning_parts:
                    message['reasoning_content'] = ''.join(reasoning_parts)
                if tool_calls:
                    message['tool_calls'] = [tool_calls[i] for i in sorted(tool_calls)]
                choices = [{'index': 0, 'message': message, 'finish_reason': finish_reason}]
                data = {'id': response_id, 'model': response_model, 'created': created, 'choices': choices, 'usage': usage}
                if not choices:
                    raise RuntimeError('NVIDIA SDK response has no choices')
                message = choices[0].get('message') or {}
                content = message.get('content') or ''
                reasoning_content = message.get('reasoning_content') or message.get('reasoning') or ''
                if content or not allow_reasoning_fallback:
                    text, text_source = content, 'content'
                else:
                    text, text_source = reasoning_content, 'reasoning_content_fallback'
                usage = data.get('usage') or {}
                self.rate_limiter.record(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
                return LLMResult(
                    text=text,
                    text_source=text_source,
                    message=message,
                    request={'endpoint': self.endpoint, 'headers': {'api-key': '***', 'Authorization': 'Bearer ***'}, 'json': payload},
                    response={'status_code': 200, 'usage': usage, 'choices': choices, 'id': data.get('id'), 'created': data.get('created')},
                )
            finally:
                self.rate_limiter.release()
        last_error = None
        for attempt in range(self.settings.request_retries + 1):
            try:
                headers = {
                    'api-key': self.settings.api_key,
                    'Authorization': f'Bearer {self.settings.api_key}',
                    'Content-Type': 'application/json',
                }
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.settings.request_timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.settings.request_retries:
                    retry_after = response.headers.get('Retry-After')
                    try:
                        wait_seconds = float(retry_after) if retry_after else 2 * (attempt + 1)
                    except ValueError:
                        wait_seconds = 2 * (attempt + 1)
                    time.sleep(max(1.0, wait_seconds))
                    continue
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.settings.request_retries:
                    raise
                time.sleep(2 * (attempt + 1))
        if response.status_code == 400 and self.settings.context_overflow_retry:
            adjusted = self._context_overflow_adjusted_max_tokens(response.text, payload)
            if adjusted is not None:
                retry_payload = copy.deepcopy(payload)
                retry_payload['max_tokens'] = adjusted
                headers = {
                    'api-key': self.settings.api_key,
                    'Authorization': f'Bearer {self.settings.api_key}',
                    'Content-Type': 'application/json',
                }
                response = requests.post(self.endpoint, headers=headers, json=retry_payload, timeout=self.settings.request_timeout)
                payload = retry_payload
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:1000] if response is not None else ""
            raise requests.HTTPError(f"{exc}; body={body}", response=response) from exc
        for decode_attempt in range(self.settings.request_retries + 1):
            try:
                data = self._decode_json_response(response)
                break
            except requests.exceptions.JSONDecodeError:
                if decode_attempt >= self.settings.request_retries:
                    self.rate_limiter.release()
                    raise
                time.sleep(2 * (decode_attempt + 1))
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.request_timeout,
                )
                response.raise_for_status()
        choices = data.get('choices') if isinstance(data, dict) else None
        if not choices:
            self.rate_limiter.release()
            error_value = data.get('error') if isinstance(data, dict) else None
            raise requests.HTTPError(
                "HTTP 200 response missing choices; "
                f"keys={sorted(data) if isinstance(data, dict) else type(data).__name__}; "
                f"error={str(error_value)[:500]}",
                response=response,
            )
        message = choices[0]['message']
        content = message.get('content') or ''
        reasoning_content = message.get('reasoning_content') or ''
        if content or not allow_reasoning_fallback:
            text = content
            text_source = 'content'
        else:
            text = reasoning_content
            text_source = 'reasoning_content_fallback'
        usage = data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        self.rate_limiter.record(prompt_tokens, completion_tokens)
        self.rate_limiter.release()
        return LLMResult(text=text, text_source=text_source, message=message, request={'endpoint': self.endpoint, 'headers': {'api-key': '***', 'Authorization': 'Bearer ***'}, 'json': payload}, response={'status_code': response.status_code, 'usage': usage, 'choices': data.get('choices'), 'id': data.get('id'), 'created': data.get('created')})

    @staticmethod
    def _decode_json_response(response):
        """Ignore Packy non-stream keep-alive whitespace before the JSON body."""
        text = response.text
        stripped = text.lstrip("\ufeff \t\r\n")
        if not stripped:
            raise requests.exceptions.JSONDecodeError(
                "Response contained keep-alive whitespace but no final JSON body",
                text,
                0,
            )
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            if start < 0:
                raise
            value, _ = json.JSONDecoder().raw_decode(stripped[start:])
            return value
    def complete_with_trace(self, prompt, allow_reasoning_fallback=False):
        messages = [{'role': 'system', 'content': 'You are MiMo, an AI assistant developed by Xiaomi.'}, {'role': 'user', 'content': prompt}]
        return self.complete_messages_with_trace(messages, allow_reasoning_fallback=allow_reasoning_fallback)
    def complete(self, prompt):
        return self.complete_with_trace(prompt).text
    def _context_overflow_adjusted_max_tokens(self, response_text, payload):
        current = int(payload.get('max_tokens') or self.settings.max_completion_tokens)
        match = re.search(r'maximum context length is (\d+) tokens.*requested \d+ tokens \((\d+) in the messages, (\d+) in the completion\)', response_text)
        if not match:
            return None
        context_window = min(self.settings.context_window, int(match.group(1)))
        message_tokens = int(match.group(2))
        min_completion = max(1, self.settings.context_overflow_min_completion_tokens)
        adjusted = context_window - message_tokens
        if adjusted < min_completion or adjusted >= current:
            return None
        return adjusted
