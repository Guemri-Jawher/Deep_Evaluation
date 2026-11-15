from __future__ import annotations

import json
from typing import Any


def ok(**data: Any) -> str:
    return json.dumps({"status": "ok", **data}, ensure_ascii=False)


def error(message: str, hint: str) -> str:
    """Errors teach the model how to recover instead of just failing."""
    return json.dumps({"status": "error", "error": message, "hint": hint}, ensure_ascii=False)


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15].rstrip() + " …[truncated]"
