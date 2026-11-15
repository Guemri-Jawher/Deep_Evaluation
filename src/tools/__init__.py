"""Agent tools. Contract for every tool: typed input, bounded structured output, actionable errors."""

from tools.registry import build_tools

__all__ = ["build_tools"]
