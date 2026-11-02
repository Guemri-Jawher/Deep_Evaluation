"""Shared result types for every evaluation suite."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class MetricSummary(BaseModel):
    name: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


class CaseResult(BaseModel):
    case_id: str
    suite: str
    category: str = ""
    split: str = "dev"
    scores: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    tokens: int = 0
    error: str | None = None
    trace_id: str | None = None


class SuiteReport(BaseModel):
    suite: str
    dataset_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    config: dict[str, Any] = Field(default_factory=dict)
    n: int = 0
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)

    def metric(self, name: str) -> MetricSummary | None:
        return self.metrics.get(name)

    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]
