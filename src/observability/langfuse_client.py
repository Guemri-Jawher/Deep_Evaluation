"""Langfuse client, tracing context and score helpers.

Tracing is optional: without keys every helper degrades to a no-op so unit tests and
offline development never depend on the network.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

from utils.config import get_settings

log = logging.getLogger(__name__)


def _export_env() -> None:
    """The Langfuse SDK reads credentials from the process environment."""
    s = get_settings()
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key.get_secret_value())
    os.environ.setdefault("LANGFUSE_BASE_URL", s.langfuse_base_url)
    os.environ.setdefault("LANGFUSE_HOST", s.langfuse_base_url)


def get_langfuse() -> Any | None:
    """Return the Langfuse singleton, or None when keys are not configured."""
    if not get_settings().langfuse_enabled:
        return None
    _export_env()
    from langfuse import get_client

    return get_client()


def get_callback_handler() -> Any | None:
    """LangChain/LangGraph callback handler that turns every run into a Langfuse trace."""
    if get_langfuse() is None:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


@contextlib.contextmanager
def trace_run(
    name: str,
    *,
    input: Any = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    version: str | None = None,
) -> Iterator[Any | None]:
    """Open a root observation and propagate trace attributes to every child span.

    Yields the root span (or None when tracing is disabled). Wrap one logical request in it
    so that LangChain callbacks, tool calls and scores all land on a single trace.
    """
    lf = get_langfuse()
    if lf is None:
        yield None
        return
    from langfuse import propagate_attributes

    attrs: dict[str, Any] = {"trace_name": name}
    if session_id:
        attrs["session_id"] = session_id
    if user_id:
        attrs["user_id"] = user_id
    if tags:
        attrs["tags"] = tags
    if metadata:
        attrs["metadata"] = metadata
    if version:
        attrs["version"] = version
    with (
        lf.start_as_current_observation(as_type="span", name=name, input=input) as span,
        propagate_attributes(**attrs),
    ):
        yield span


def current_trace_id() -> str | None:
    lf = get_langfuse()
    return lf.get_current_trace_id() if lf else None


def score_trace(
    trace_id: str | None,
    name: str,
    value: float | bool | str,
    *,
    comment: str | None = None,
    data_type: str | None = None,
) -> None:
    """Attach an evaluation score to a trace (no-op without Langfuse)."""
    lf = get_langfuse()
    if lf is None or trace_id is None:
        return
    kwargs: dict[str, Any] = {"trace_id": trace_id, "name": name, "value": value}
    if comment:
        kwargs["comment"] = comment
    if data_type:
        kwargs["data_type"] = data_type
    try:
        lf.create_score(**kwargs)
    except Exception:  # scoring must never break the request path
        log.exception("failed to post score %s", name)


def flush() -> None:
    lf = get_langfuse()
    if lf is not None:
        lf.flush()
