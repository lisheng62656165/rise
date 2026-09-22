"""Agent interfaces used by the benchmark harness."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from state_bench.schemas import TokenUsage


@dataclass(slots=True)
class AgentRuntimeContext:
    """Per-run context passed to custom agents at construction time.

    This keeps the benchmark memory-agnostic while giving BYO agents access to
    stable task/runtime metadata and an optional free-form config payload.
    """

    task_id: str
    user_id: str
    domain: str
    now: str
    output_dir: str | None = None
    run_idx: int | None = None
    task_summary: str | None = None
    state_requirements: list[dict[str, Any]] = field(default_factory=list)
    task_requirements: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentToolCallRequest:
    """Provider-agnostic tool call requested by a custom agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentTurnResponse:
    """Provider-agnostic response returned by a harness-executed agent turn."""

    text: str = ""
    tool_calls: list[AgentToolCallRequest | dict[str, Any]] = field(default_factory=list)


class BaseAgent(ABC):
    """Base harness interface for agent implementations."""

    total_output_tokens: int = 0

    def __init__(self, runtime_context: AgentRuntimeContext | None = None):
        self.runtime_context = runtime_context
        self.token_usage = TokenUsage()

    def add_token_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        reasoning_output_tokens: int | None = None,
        category: str = "agent_turn",
    ) -> None:
        """Accumulate provider-reported token usage telemetry.

        Custom clients should pass provider-reported token counts here when
        available. If either input or output tokens are missing, no usage is
        recorded. Cost is intentionally reported separately because provider
        token buckets are not billing-compatible across SDKs.
        """
        if input_tokens is None or output_tokens is None:
            return

        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        cached_input_tokens = int(cached_input_tokens or 0)
        reasoning_output_tokens = int(reasoning_output_tokens or 0)
        total_tokens = input_tokens + output_tokens

        self.token_usage.input_tokens += input_tokens
        self.token_usage.cached_input_tokens += cached_input_tokens
        self.token_usage.output_tokens += output_tokens
        self.token_usage.reasoning_output_tokens += reasoning_output_tokens
        self.token_usage.total_tokens += total_tokens

    def add_cost_usd(self, amount: float, *, category: str = "agent_turn") -> None:
        """Accumulate user-reported agent cost for this trajectory."""
        amount = float(amount)
        if amount < 0:
            raise ValueError("cost amount must be >= 0")

        if category == "agent_turn":
            self.token_usage.agent_turn_cost_usd += amount
        elif category == "memory_ingestion":
            self.token_usage.memory_ingestion_cost_usd += amount
        elif category == "memory_retrieval":
            self.token_usage.memory_retrieval_cost_usd += amount
        else:
            self.token_usage.other_llm_cost_usd += amount
        self.token_usage.total_cost_usd += amount

    def add_response_usage(self, usage: Any, *, category: str = "other_llm") -> None:
        """Accumulate agent-side Responses API usage for later cost reporting."""
        if not usage:
            return

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached_input_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
        reasoning_output_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
        self.add_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            category=category,
        )

    def act(self, conversation: list[Any]) -> tuple[str, list[dict[str, Any]], list[Any]]:
        """Execute one legacy agent turn.

        Args:
            conversation: Full conversation as Responses API input items
                (stateless chaining pattern — includes message dicts,
                function_call items, and function_call_output items).

        Returns:
            A 3-tuple of:
            - text: The agent's final text response for this turn
            - tool_calls: List of tool call records, each a dict with
              keys {name, arguments, result}
            - raw_items: Raw API output items (for conversation chaining)
        """
        raise NotImplementedError(
            "Implement act() for legacy self-executed agents, or implement "
            "generate_next_turn() to let the benchmark harness execute tools."
        )

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse | dict[str, Any]:
        """Generate the next assistant step with a user-owned LLM client.

        Custom public-release agents should implement this method when they
        want to own their LLM client/provider/settings while letting the
        benchmark execute and record domain tools canonically.

        Return either AgentTurnResponse or a dict with:
        - text: assistant text
        - tool_calls: list of {name, arguments} requests
        """
        raise NotImplementedError

    def uses_harness_tool_execution(self) -> bool:
        """Whether this agent uses generate_next_turn() instead of legacy act()."""
        return type(self).generate_next_turn is not BaseAgent.generate_next_turn

    def memory_tool_schemas(self) -> list[dict[str, Any]]:
        """Optional read-only memory retrieval tools exposed to the agent.

        Official runs allow benchmark domain tools plus memory retrieval tools
        declared by the agent. Memory tools should not mutate benchmark state.
        """
        return []

    def memory_tool_handlers(self) -> dict[str, Any]:
        """Handlers for optional memory retrieval tools declared by the agent."""
        return {}

    def prepare_conversation(self, conversation: list[Any]) -> list[Any]:
        """Optional pre-turn hook for retrieval or system-message injection.

        Custom agents can override this to retrieve memories and inject a
        `{"role": "system", "content": ...}` message into the turn input
        without mutating the benchmark's canonical conversation transcript.
        """
        return conversation

    def inject_system_message(
        self,
        conversation: list[Any],
        content: str,
        *,
        before_last_user: bool = True,
    ) -> list[Any]:
        """Return a copy of the turn input with an injected system message."""
        if not content:
            return conversation

        system_item = {"role": "system", "content": content}
        if not before_last_user or not conversation:
            return [*conversation, system_item]
        return [*conversation[:-1], system_item, conversation[-1]]

    def ingest_trajectory(self, trajectory: Any) -> None:
        """Optional post-run hook for BYO ingestion.

        Called once after a trajectory is produced, before it is written to disk.
        Override this to ingest the just-finished conversation into a custom
        memory store or analytics pipeline.
        """
        return None
