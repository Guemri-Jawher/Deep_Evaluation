"""
Guardrails for input and output checking, including prompt injection detection,
PII masking, secret redaction, and prompt leakage detection.

This module provides functions and classes to enforce guardrails on both input and output text,
ensuring that sensitive information is protected and that the agent operates within defined safety constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

REFUSAL_MESSAGE = (
    "I can't help with that request. I can answer questions about the evaluation field guide and the "
    "agentic design patterns guide, with citations."
)
MAX_INPUT_CHARS = 4000

_INJECTION = [
    r"ignore (?:all |any |the |your )?(?:previous|prior|above|earlier)?\s*(?:instructions|rules|prompts?)",
    r"disregard (?:all |any |the |your )?(?:previous|prior|above|earlier)?\s*(?:instructions|rules|prompts?)",
    r"forget (?:all |any |your )?(?:previous|prior|earlier)?\s*(?:instructions|rules)",
    r"(?:reveal|show|print|repeat|output|display|leak)\b.{0,40}\b(?:system|hidden|initial|developer) (?:prompt|message|instructions)",
    r"\b(?:system|developer) prompt\b.{0,20}\b(?:verbatim|word for word|exactly)",
    r"you are now (?:in )?(?:dan|developer mode|an? unrestricted)",
    r"\bdeveloper mode\b|\bjailbreak\b|\bdo anything now\b",
    r"pretend (?:you have|there are) no (?:rules|restrictions|guidelines)",
    r"override (?:your|the) (?:safety|rules|guardrails|instructions)",
]
_INJECTION_RE = [re.compile(p, re.I | re.S) for p in _INJECTION]

_PII = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\w)"),
    "card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}
_SECRETS = {
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "langfuse_key": re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9-]{8,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"),
}


@dataclass
class GuardResult:
    action: Literal["allow", "sanitize", "block"] = "allow"
    text: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"


def _luhn(number: str) -> bool:
    """Check if a number passes the Luhn algorithm (commonly used for credit card validation)."""
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Mask personally identifiable information (PII) in the text and return the masked text along with a list of found PII types."""
    found: list[str] = []

    def sub_card(m: re.Match[str]) -> str:
        if _luhn(m.group(0)):
            found.append("card")
            return "[CARD]"
        return m.group(0)

    out = _PII["card"].sub(sub_card, text)
    for name in ("email", "phone"):
        out, n = _PII[name].subn(f"[{name.upper()}]", out)
        if n:
            found.append(name)
    return out, found


def check_input(text: str) -> GuardResult:
    """Block oversize input and known injection phrasing; mask PII before the agent sees it."""
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult("block", "", ["input_too_long"])
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            return GuardResult("block", "", ["prompt_injection"])
    masked, found = mask_pii(text)
    if found:
        return GuardResult("sanitize", masked, [f"pii:{name}" for name in found])
    return GuardResult("allow", text)


def overlaps_prompt(output: str, prompt: str, n: int = 8) -> bool:
    """True when `output` repeats `n` consecutive words of the system prompt (prompt leakage)."""
    words = re.findall(r"\w+", prompt.lower())
    out = " ".join(re.findall(r"\w+", output.lower()))
    return any(" ".join(words[i : i + n]) in out for i in range(0, max(0, len(words) - n + 1)))


def check_output(
    text: str,
    *,
    system_prompt: str | None = None,
    retrieved_ids: set[str] | None = None,
    cited_ids: list[str] | None = None,
) -> GuardResult:
    """Redact secrets, detect prompt leakage and fabricated citations before the user sees text."""
    reasons: list[str] = []
    redacted = text
    for name, pattern in _SECRETS.items():
        redacted, n = pattern.subn("[REDACTED]", redacted)
        if n:
            reasons.append(f"secret:{name}")
    if system_prompt and overlaps_prompt(text, system_prompt):
        reasons.append("prompt_leak")
    if retrieved_ids is not None and cited_ids:
        fabricated = sorted({c for c in cited_ids if c not in retrieved_ids})
        if fabricated:
            reasons.append(f"fabricated_citation:{','.join(fabricated)}")
    if "prompt_leak" in reasons:
        return GuardResult("block", REFUSAL_MESSAGE, reasons)
    if reasons:
        return GuardResult("sanitize", redacted, reasons)
    return GuardResult("allow", text)
