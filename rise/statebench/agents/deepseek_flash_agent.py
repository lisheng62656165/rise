"""STATE-Bench agent adapter for DeepSeek Flash Chat Completions."""

from __future__ import annotations

import json
from typing import Any

from state_bench.agents.base import AgentToolCallRequest, AgentTurnResponse, BaseAgent


class DeepSeekFlashAgent(BaseAgent):
    def __init__(self, client, system_prompt, tools, tool_handlers, runtime_context=None, **kwargs):
        super().__init__(runtime_context=runtime_context)
        self.client = client
        self.system_prompt = system_prompt
        self.tools = [self._to_openai_tool(tool) for tool in tools]

    @staticmethod
    def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
        if "function" in tool:
            return tool
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _to_chat_messages(self, system_prompt: str, conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        call_index = 0

        for message in conversation:
            role = message.get("role")
            if role == "tool":
                # Tool results are stored on the assistant message that requested
                # them, which lets us synthesize stable Chat Completions call ids.
                continue
            if role not in {"user", "assistant", "system"}:
                continue

            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if role != "assistant" or not tool_calls:
                messages.append({"role": role, "content": content})
                continue

            openai_calls = []
            tool_messages = []
            for call in tool_calls:
                call_index += 1
                call_id = f"call_state_bench_{call_index}"
                args = call.get("arguments") or {}
                openai_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": self._json_dumps(args),
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": call["name"],
                        "content": self._json_dumps(call.get("result")),
                    }
                )
            messages.append({"role": "assistant", "content": content, "tool_calls": openai_calls})
            messages.extend(tool_messages)

        return messages

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"DeepSeek returned invalid tool arguments JSON: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek tool arguments must decode to an object.")
        return parsed

    def generate_next_turn(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurnResponse:
        response = self.client.generate(
            messages=self._to_chat_messages(system_prompt, conversation),
            tools=[self._to_openai_tool(tool) for tool in tools],
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.add_token_usage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )
        return AgentTurnResponse(
            text=response.text,
            tool_calls=[
                AgentToolCallRequest(name=call["name"], arguments=self._parse_arguments(call["arguments"]))
                for call in response.tool_calls
            ],
        )
