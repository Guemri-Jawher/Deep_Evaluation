"""LLM-as-a-judge: reason-then-score rubrics, robust parsing, caching, ensembles, pairwise swap."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from observability.langfuse_client import get_callback_handler
from prompts.registry import PromptHandle, get_prompt
from utils.config import DATA_DIR

CACHE_PATH = DATA_DIR / "cache" / "judge_cache.jsonl"
PASS_THRESHOLD = 4  # a 4 or 5 on the 1-5 rubric counts as a pass


@dataclass
class Verdict:
    """One judge result: a 1-5 `score` (None if unparseable) and its `reasoning`.

    `normalized` maps 1-5 to 0-1, `passed` means score >= 4, `cached` means the reply came from disk.
    Pairwise verdicts carry a `winner` ("A", "B" or "tie") instead of a score.
    """

    score: int | None
    reasoning: str
    raw: str = ""
    winner: str | None = None  # pairwise only
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.score is not None or self.winner is not None

    @property
    def normalized(self) -> float:
        """Map 1-5 to 0-1."""
        return float("nan") if self.score is None else (self.score - 1) / 4

    @property
    def passed(self) -> bool:
        return self.score is not None and self.score >= PASS_THRESHOLD


def parse_json_verdict(text: str) -> dict[str, Any] | None:
    """Pull the last JSON object that carries a 'score' or 'winner' key out of model output."""
    candidates = re.findall(r"\{[^{}]*\}", text, re.S)
    for blob in reversed(candidates):
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        if isinstance(data, dict) and ("score" in data or "winner" in data):
            return data
    return None


class _DiskCache:
    """Append-only JSONL cache so reruns on a rate-limited free tier are cheap and reproducible."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._data[row["k"]] = row["v"]
                except (ValueError, KeyError):
                    continue

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": key, "v": value}) + "\n")


@dataclass
class LLMJudge:
    """Scores one rubric (a prompt in the registry) with a fixed model."""

    rubric: str
    llm: BaseChatModel
    model_name: str = "judge"
    prompt_label: str | None = None
    cache: _DiskCache | None = field(default=None, repr=False)
    max_parse_retries: int = 1

    def __post_init__(self) -> None:
        self._prompt: PromptHandle | None = None

    @property
    def prompt(self) -> PromptHandle:
        if self._prompt is None:
            self._prompt = get_prompt(self.rubric, label=self.prompt_label)
        return self._prompt

    def _call(self, text: str) -> str:
        """Send the prompt to the model, or return the cached reply for the same model and prompt."""
        key = hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit
        handler = get_callback_handler()
        config: Any = {"callbacks": [handler]} if handler else {}
        reply = self.llm.invoke([HumanMessage(text)], config=config)
        out = reply.content if isinstance(reply.content, str) else str(reply.content)
        if self.cache is not None:
            self.cache.put(key, out)
        return out

    def score(self, **variables: object) -> Verdict:
        """Fill the rubric prompt with `variables`, ask the model, and parse a 1-5 score from its JSON.

        The prompt asks for reasoning first, then `{"reasoning": ..., "score": 1-5}`. If no valid
        score is found, it asks once more for the JSON only (`max_parse_retries`); after that it
        returns `Verdict(None, "unparseable judge output")`.
        """
        text = self.prompt.compile(**variables)
        raw = ""
        for attempt in range(self.max_parse_retries + 1):
            raw = self._call(text if attempt == 0 else text + "\n\nReturn ONLY the JSON object.")
            data = parse_json_verdict(raw)
            if data is not None and isinstance(data.get("score"), int | float):
                score = round(float(data["score"]))
                if 1 <= score <= 5:
                    return Verdict(score, str(data.get("reasoning", "")), raw)
        return Verdict(None, "unparseable judge output", raw)

    def score_pairwise(self, **variables: object) -> Verdict:
        """Pairwise rubric: returns a Verdict whose `winner` is 'A', 'B' or 'tie'."""
        raw = self._call(self.prompt.compile(**variables))
        data = parse_json_verdict(raw)
        if data and str(data.get("winner", "")) in {"A", "B", "tie"}:
            return Verdict(None, str(data.get("reasoning", "")), raw, winner=str(data["winner"]))
        return Verdict(None, "unparseable judge output", raw)


def disk_cache() -> _DiskCache:
    return _DiskCache(CACHE_PATH)


def aggregate(verdicts: Sequence[Verdict]) -> tuple[Verdict, bool]:
    """Median of an ensemble; the bool flags disagreement (spread >= 2) for human review."""
    scored = [v for v in verdicts if v.score is not None]
    if not scored:
        return Verdict(None, "all judges failed"), True
    scores = [v.score for v in scored if v.score is not None]
    median = round(statistics.median(scores))
    spread = max(scores) - min(scores)
    return Verdict(median, f"ensemble scores={scores}"), spread >= 2


@dataclass
class PairwiseResult:
    winner: str  # "good"/"bad" labels are resolved by the caller; here "first"|"second"|"tie"|"inconsistent"
    consistent: bool
    raw: tuple[str, str] = ("", "")


def compare_pair(
    judge: LLMJudge, *, question: str, reference: str, first: str, second: str
) -> PairwiseResult:
    """Judge both orderings and require the same verdict (controls position bias)."""
    v1 = judge.score_pairwise(
        question=question, reference=reference, answer_a=first, answer_b=second
    )
    v2 = judge.score_pairwise(
        question=question, reference=reference, answer_a=second, answer_b=first
    )
    w1 = {"A": "first", "B": "second", "tie": "tie"}.get(v1.winner or "", "invalid")
    w2 = {"A": "second", "B": "first", "tie": "tie"}.get(v2.winner or "", "invalid")
    if "invalid" in (w1, w2):
        return PairwiseResult("inconsistent", False, (v1.raw, v2.raw))
    if w1 == w2:
        return PairwiseResult(w1, True, (v1.raw, v2.raw))
    return PairwiseResult("inconsistent", False, (v1.raw, v2.raw))
