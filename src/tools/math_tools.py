from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from tools._format import error, ok

_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _round(x: float, digits: float = 0) -> float:
    return float(round(x, int(digits)))


_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": _round,
    "min": min,
    "max": max,
}
MAX_EXPONENT = 1000


def safe_eval(expression: str) -> float:
    """Evaluate a numeric expression without `eval`. Raises ValueError on anything else."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError("exponent too large")
            return float(_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](walk(node.operand)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return float(_FUNCS[node.func.id](*(walk(a) for a in node.args)))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return walk(ast.parse(expression.strip(), mode="eval"))


class CalculatorInput(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic only, e.g. '0.8**3' or 'round(1-(1-0.7)**5, 4)'. Functions: sqrt, log, exp, abs, round, min, max.",
    )


class PassKInput(BaseModel):
    success_rate: float = Field(
        ge=0.0, le=1.0, description="Per-attempt success probability p in [0, 1]."
    )
    k: int = Field(ge=1, le=100, description="Number of attempts k (1-100).")


def calculator() -> BaseTool:
    def run(expression: str) -> str:
        try:
            return ok(expression=expression, result=safe_eval(expression))
        except ZeroDivisionError:
            return error("division by zero", "Check the denominator in your expression.")
        except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
            return error(
                str(exc), "Use only numbers, + - * / ** % //, parentheses and the listed functions."
            )

    return StructuredTool.from_function(
        func=run,
        name="calculator",
        description=(
            "Evaluate a plain arithmetic expression exactly. Use for cost, latency or ratio math. "
            "Do NOT use for pass@k / pass^k (use pass_k) or to look up facts (use search_docs). "
            "Read-only. Returns JSON: status, result."
        ),
        args_schema=CalculatorInput,
    )


def pass_k() -> BaseTool:
    def run(success_rate: float, k: int) -> str:
        return ok(
            success_rate=success_rate,
            k=k,
            pass_at_k=round(1 - (1 - success_rate) ** k, 6),
            pass_pow_k=round(success_rate**k, 6),
            note="pass@k = at least one of k attempts succeeds; pass^k = all k attempts succeed.",
        )

    return StructuredTool.from_function(
        func=run,
        name="pass_k",
        description=(
            "Compute agent reliability metrics from a per-attempt success rate p over k attempts: "
            "pass@k (at least one success) and pass^k (all succeed). Use when asked what happens to "
            "reliability as k grows. Read-only. Returns JSON: pass_at_k, pass_pow_k."
        ),
        args_schema=PassKInput,
    )
