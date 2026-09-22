"""MIMO 瞬时传输错误的有界重试封装。

Inference retries preserve the prompt, seed, task, and environment, but a
provider may return different samples and charge for each request. Model errors are re-raised
immediately so this wrapper cannot hide invalid tool decisions.

本文件是 transport 基础设施，不参与 EDS-ECA 打分。它创建
OpenAI-compatible client，处理瞬时失败，可按配置轮换 key，保留 seeded
request，并提供 MiMoAgent 所需的 response/token 接口。
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from openai import OpenAI
from clients.mimo_client import MiMoClient as _MiMoClient


TRANSIENT_EXCEPTION_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
}


# 函数作用：识别适合重试的 MIMO 网络、限流和服务端瞬时错误。
def is_transient_mimo_error(error: BaseException) -> bool:
    name = type(error).__name__
    if name in TRANSIENT_EXCEPTION_NAMES:
        return True
    text = str(error).lower()
    return any(token in text for token in (
        "connection reset", "connection error", "timed out",
        "temporarily overloaded", "service overloaded", "rate limit",
    ))


class ReliableMiMoClient(_MiMoClient):
    """MiMo client with bounded retries for transport/service interruptions."""

    def complete_json(self, prompt, system_prompt=None, max_tokens=None, reasoning_effort=None):
        from clients.deepseek_flash_client import _parse_json_object
        messages = [{'role': 'system', 'content': (system_prompt or '') + '\nReturn one valid JSON object.'},
                    {'role': 'user', 'content': prompt}]
        for attempt in range(3):
            try:
                return _parse_json_object(self.complete_chat(
                    messages, max_tokens=max_tokens, temperature=0, reasoning_effort=reasoning_effort))
            except (ValueError, TypeError):
                if attempt == 2:
                    raise

    # A reverse proxy can accept many TCP connections while the upstream MIMO
    # service queues long SSE generations.  Limit active requests globally
    # across per-worker clients; task-level concurrency remains independent.
    _active_requests = threading.BoundedSemaphore(
        max(1, int(os.environ.get("MIMO_ACTIVE_REQUESTS", "64")))
    )
    _key_lock = threading.Lock()
    _key_index = 0

    # 函数作用：从环境变量构造带统一超时和重试配置的 MIMO 客户端。
    @classmethod
    def from_env(cls) -> "ReliableMiMoClient":
        """Build a client, optionally rotating across a secure API-key pool."""
        pool_file = os.environ.get("MIMO_API_KEYS_FILE")
        if not pool_file:
            return super().from_env()
        keys = [line.strip() for line in Path(pool_file).read_text().splitlines() if line.strip()]
        if not keys:
            raise ValueError(f"No API keys in MIMO_API_KEYS_FILE: {pool_file}")
        with cls._key_lock:
            api_key = keys[cls._key_index % len(keys)]
            cls._key_index += 1
        return cls(
            api_key=api_key,
            base_url=os.environ.get("MIMO_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("MIMO_MODEL", "gpt-4.1"),
            max_tokens=int(os.environ.get("MIMO_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("MIMO_TEMPERATURE", "0")),
            seed=int(os.environ["MIMO_SEED"]) if os.environ.get("MIMO_SEED") else None,
            reasoning_effort=os.environ.get("MIMO_REASONING_EFFORT") or None,
            thinking_enabled=os.environ.get("MIMO_THINKING", "").lower()
            in {"1", "true", "yes", "on"},
            timeout_seconds=float(os.environ.get("MIMO_TIMEOUT_SECONDS", "180")),
        )

    # Keep the historical default, but let long-running batch jobs choose a
    # bounded per-request budget without changing algorithm behavior.
    retry_attempts = max(1, int(os.environ.get("MIMO_RETRY_ATTEMPTS", "3")))
    retry_backoff_seconds = float(os.environ.get("MIMO_RETRY_BACKOFF_SECONDS", "5"))

    # 函数作用：保存客户端配置并初始化 OpenAI 兼容传输层。
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._hard_timeout_seconds = float(
            os.environ.get("MIMO_HARD_TIMEOUT_SECONDS", "0")
        )
        if os.environ.get("MIMO_DISABLE_KEEPALIVE", "0").lower() not in {"1", "true", "yes", "on"}:
            return
        # Reverse proxies can leave a pooled socket half-closed while the
        # caller is waiting on a later tool turn.  Use one short-lived socket
        # per request for diagnostic and batch runs; the default remains the
        # normal OpenAI connection pool.
        old_client = getattr(self, "_client", None)
        close = getattr(old_client, "close", None)
        if close is not None:
            close()
        timeout_seconds = float(
            kwargs.get("timeout_seconds", os.environ.get("MIMO_TIMEOUT_SECONDS", 300.0))
        )
        # A long SSE response occupies a socket for its entire generation.
        # Size the client pool from the actual worker count instead of the old
        # four-connection diagnostic default; otherwise workers fail in the
        # pool wait before the upstream request has even started.
        worker_count = max(1, int(os.environ.get("MIMO_WORKERS", os.environ.get("WORKERS", "20"))))
        max_connections = max(worker_count + 4, int(os.environ.get("MIMO_MAX_CONNECTIONS", "32")))
        proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("ALL_PROXY")
        )
        socket_options = []
        if hasattr(socket, "TCP_KEEPIDLE"):
            socket_options.extend(
                [
                    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30),
                    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
                    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
                ]
            )
        # Bound a silently black-holed forwarded connection.  This is a
        # kernel-level liveness limit, separate from the HTTP read timeout.
        # Keep it below the request budget so retries can recover before a
        # worker is stuck indefinitely behind the reverse tunnel.
        if hasattr(socket, "TCP_USER_TIMEOUT"):
            socket_options.append(
                (socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT,
                 int(min(timeout_seconds, 240.0) * 1000))
            )
        # 函数作用：根据当前 endpoint 和 key 建立底层同步 API 客户端。
        def build_transport_client() -> OpenAI:
            return OpenAI(
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
                # Keep long reads alive, but do not let a saturated pool wait
                # for the full read timeout. Multiple short-lived sockets also
                # make this resilient to a half-closed reverse-proxy connection.
                timeout=httpx.Timeout(
                    timeout_seconds,
                    connect=min(30.0, timeout_seconds),
                    pool=30.0,
                ),
                max_retries=0,
                http_client=httpx.Client(
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=0,
                    ),
                    transport=httpx.HTTPTransport(socket_options=socket_options),
                    # Explicitly pass the reverse-proxy URL. With a custom
                    # transport, relying on trust_env alone can fall back to
                    # a direct SYN to the public API and hang behind the
                    # firewall.
                    proxy=proxy,
                ),
            )

        self._build_transport_client = build_transport_client
        self._client = build_transport_client()

    # 函数作用：在瞬时错误上执行带退避的统一重试包装。
    def _retry(self, operation):
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                with self._active_requests:
                    if self._hard_timeout_seconds <= 0:
                        return operation()
                    # A blocking response iterator cannot observe a deadline
                    # until its next chunk. Run it in a daemon helper so a
                    # half-open reverse-tunnel request can be abandoned and
                    # retried without pinning the batch worker forever.
                    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

                    # 函数作用：调用一个操作并在认证问题时轮换可用 key 后重试。
                    def invoke() -> None:
                        try:
                            result_queue.put((True, operation()))
                        except BaseException as error:  # propagate in caller
                            result_queue.put((False, error))

                    helper = threading.Thread(target=invoke, daemon=True)
                    helper.start()
                    helper.join(self._hard_timeout_seconds)
                    if helper.is_alive():
                        close = getattr(self._client, "close", None)
                        if close is not None:
                            close()
                        raise httpx.ReadTimeout("MIMO request hard timeout")
                    ok, value = result_queue.get_nowait()
                    if ok:
                        return value
                    raise value
            except Exception as error:
                last_error = error
                if not is_transient_mimo_error(error) or attempt + 1 >= self.retry_attempts:
                    raise
                # An EOF can leave the underlying pooled client unusable. Do
                # not retry through that socket: close it and rebuild the same
                # explicitly proxied client before backing off.
                if hasattr(self, "_build_transport_client"):
                    close = getattr(self._client, "close", None)
                    if close is not None:
                        close()
                    self._client = self._build_transport_client()
                delay = self.retry_backoff_seconds * (2 ** attempt)
                time.sleep(delay)
        raise last_error  # pragma: no cover

    # 函数作用：按 StateBench agent 所需接口生成带可选工具调用的回复。
    def generate(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        if os.environ.get("MIMO_STREAMING", "0").lower() in {"1", "true", "yes", "on"}:
            return self._retry(
                lambda: self._generate_streaming(messages=messages, tools=tools)
            )
        return self._retry(lambda: self._generate_once(messages=messages, tools=tools))

    # 函数作用：执行普通 chat completion，供 selector 和评分器使用。
    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """Run simulator chat through the same deadline/retry path as agent calls."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        extra = self._provider_extra_body()
        if extra:
            kwargs["extra_body"] = extra
        elif reasoning_effort or self.reasoning_effort:
            kwargs['reasoning_effort'] = reasoning_effort or self.reasoning_effort
        elif self.thinking_enabled:
            kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}
        if os.environ.get("MIMO_JSON_RESPONSE_FORMAT", "0").lower() in {"1", "true", "yes", "on"}:
            kwargs["response_format"] = {"type": "json_object"}

        # 函数作用：向当前 provider 发起一次 OpenAI 兼容请求。
        def request() -> str:
            self._compatible_kwargs(kwargs)
            if os.environ.get("MIMO_STREAMING", "0").lower() in {"1", "true", "yes", "on"}:
                response = self._client.chat.completions.create(**kwargs, stream=True)
                parts = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        parts.append(content)
                return "".join(parts)
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        return self._retry(request)

    # 函数作用：生成特定服务商需要的额外请求字段。
    def _provider_extra_body(self) -> dict[str, Any] | None:
        nemotron_thinking = os.environ.get("MIMO_NEMOTRON_THINKING")
        if nemotron_thinking is not None:
            enabled = nemotron_thinking.lower() in {"1", "true", "yes", "on"}
            extra: dict[str, Any] = {
                "chat_template_kwargs": {"enable_thinking": enabled},
            }
            if enabled:
                extra["reasoning_budget"] = int(
                    os.environ.get("MIMO_REASONING_BUDGET", "16384")
                )
            return extra
        if os.environ.get("MIMO_CHAT_TEMPLATE_THINKING", "0").lower() in {"1", "true", "yes", "on"}:
            return {"chat_template_kwargs": {"thinking": True, "reasoning_effort": os.environ.get("MIMO_REASONING_EFFORT", "high")}}
        return None

    def _compatible_kwargs(self, kwargs):
        """Explicit provider switches; never retry by silently changing semantics."""
        if os.environ.get('MIMO_SEND_SEED', '1') == '0':
            kwargs.pop('seed', None)
        if os.environ.get('MIMO_SEND_TEMPERATURE', '1') == '0':
            kwargs.pop('temperature', None)
        if os.environ.get('MIMO_TOKEN_PARAMETER', 'max_tokens') == 'max_completion_tokens' and 'max_tokens' in kwargs:
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
        return kwargs

    # 函数作用：执行一次非流式 agent 生成并规范化返回格式。
    def _generate_once(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model, "messages": messages, "tools": tools,
            "tool_choice": "auto", "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        extra = self._provider_extra_body()
        if extra:
            kwargs["extra_body"] = extra
        elif self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif self.thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        response = self._client.chat.completions.create(**self._compatible_kwargs(kwargs))
        message = response.choices[0].message
        calls = [{"id": call.id, "name": call.function.name, "arguments": call.function.arguments or "{}"} for call in (message.tool_calls or [])]
        return SimpleNamespace(text=message.content or "", tool_calls=calls,
                               reasoning_content=getattr(message, "reasoning_content", None),
                               usage=getattr(response, "usage", None))

    # 函数作用：执行流式生成并拼接文本、工具调用和 token 用量。
    def _generate_streaming(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        """Consume SSE incrementally while preserving the normal response API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        extra = self._provider_extra_body()
        if extra:
            kwargs["extra_body"] = extra
        elif self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if self.seed is not None:
            kwargs["seed"] = self.seed

        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage = None
        deadline = None
        if self._hard_timeout_seconds > 0:
            deadline = time.monotonic() + self._hard_timeout_seconds
        for chunk in self._client.chat.completions.create(**self._compatible_kwargs(kwargs)):
            if deadline is not None and time.monotonic() >= deadline:
                raise httpx.ReadTimeout("MIMO streaming hard timeout")
            usage = getattr(chunk, "usage", None) or usage
            choices = getattr(chunk, "choices", None) or ()
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                content.append(str(text))
            thought = getattr(delta, "reasoning_content", None)
            if thought:
                reasoning.append(str(thought))
            for raw_call in getattr(delta, "tool_calls", None) or ():
                index = int(getattr(raw_call, "index", 0) or 0)
                call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                call_id = getattr(raw_call, "id", None)
                if call_id:
                    call["id"] = str(call_id)
                function = getattr(raw_call, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    if name:
                        call["name"] = str(name)
                    if arguments:
                        call["arguments"] += str(arguments)
        return SimpleNamespace(
            text="".join(content),
            tool_calls=[calls[index] for index in sorted(calls)],
            reasoning_content="".join(reasoning) or None,
            usage=usage,
        )
