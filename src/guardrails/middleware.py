"""
Middleware for enforcing tool usage policies, including allow-listing, loop detection, and bounded tool output.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from tools._format import error, truncate

# Built-in Deep Agents harness tools that the supervisor may use besides our own tools.
HARNESS_TOOLS = frozenset(
    {"ls", "read_file", "glob", "grep", "write_file", "edit_file", "write_todos", "task", "delete"}
)


def _signature(call: Mapping[str, Any]) -> str:
    """Generate a unique signature for a tool call based on its name and arguments."""
    return f"{call.get('name')}:{json.dumps(call.get('args', {}), sort_keys=True, default=str)}"


def repeated_call_count(messages: Iterable[Any], call: Mapping[str, Any]) -> int:
    """Count how many times an identical tool call has already happened since the last human message."""
    recent: list[Any] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            recent = []
        recent.append(m)
    sig = _signature(call)
    matches = [
        c
        for m in recent
        if isinstance(m, AIMessage)
        for c in m.tool_calls
        if c.get("id") != call.get("id") and _signature(c) == sig
    ]
    return len(matches)


class ToolPolicyMiddleware(AgentMiddleware):
    """Action guardrail: allow-list, loop detection and bounded tool output."""

    def __init__(
        self,
        allowed_tools: Iterable[str],
        *,
        repeat_limit: int = 2,
        max_output_chars: int = 6000,
    ) -> None:
        super().__init__()
        self.allowed = set(allowed_tools) | set(HARNESS_TOOLS)
        self.repeat_limit = repeat_limit
        self.max_output_chars = max_output_chars

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = call.get("name", "")
        if name not in self.allowed:
            return self._reject(
                call, error(f"tool '{name}' is not available", "Use only the listed tools.")
            )
        messages = request.state.get("messages", []) if isinstance(request.state, dict) else []
        if repeated_call_count(messages, call) >= self.repeat_limit:
            return self._reject(
                call,
                error(
                    "identical tool call repeated",
                    "You already made this exact call. Change the query/arguments or answer with what you have.",
                ),
            )
        result = handler(request)
        if (
            isinstance(result, ToolMessage)
            and isinstance(result.content, str)
            and len(result.content) > self.max_output_chars
        ):
            result = result.model_copy(
                update={"content": truncate(result.content, self.max_output_chars)}
            )
        return result

    @staticmethod
    def _reject(call: Mapping[str, Any], content: str) -> ToolMessage:
        return ToolMessage(
            content=content, tool_call_id=call.get("id", ""), name=call.get("name"), status="error"
        )
