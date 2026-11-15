"""
Static validation of SKILL.md files against the Agent Skills spec and Deep Agents guidance.

Used by `scripts/lint_skills.py` and by the unit tests, so a malformed skill can never reach the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils.config import SKILLS_DIR

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_BODY_LINES = 500
MAX_BODY_TOKENS = 5000  # rough estimate: 4 characters per token
LINK_RE = re.compile(r"\]\(((?:references|assets|scripts)/[^)\s]+)\)")


@dataclass
class SkillReport:
    path: Path
    name: str = ""
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def parse_skill(text: str) -> tuple[dict[str, object], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not match:
        raise ValueError("missing YAML frontmatter delimited by ---")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a mapping")
    return meta, match.group(2)


def lint_skill(skill_dir: Path) -> SkillReport:
    report = SkillReport(skill_dir)
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        report.problems.append("SKILL.md not found")
        return report
    try:
        meta, body = parse_skill(skill_file.read_text(encoding="utf-8"))
    except ValueError as exc:
        report.problems.append(str(exc))
        return report

    name = str(meta.get("name", ""))
    report.name = name
    if not name or not NAME_RE.match(name) or len(name) > MAX_NAME:
        report.problems.append(f"name '{name}' must be lowercase-hyphen, <= {MAX_NAME} chars")
    if name != skill_dir.name:
        report.problems.append(f"name '{name}' must match directory '{skill_dir.name}'")

    description = str(meta.get("description", "")).strip()
    if not description:
        report.problems.append("description is required")
    elif len(description) > MAX_DESCRIPTION:
        report.problems.append(f"description is {len(description)} chars (max {MAX_DESCRIPTION})")
    elif not re.search(r"\buse (?:when|for|only)\b", description, re.I):
        report.warnings.append("description should say WHEN to use the skill ('Use when ...')")

    metadata = meta.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        report.problems.append("metadata must be a mapping")
    elif (
        isinstance(metadata, dict)
        and "include_tools" in metadata
        and not isinstance(metadata["include_tools"], str)
    ):
        report.problems.append(
            "metadata.include_tools must be a space-separated string, not a list"
        )

    lines = body.strip().splitlines()
    if len(lines) > MAX_BODY_LINES:
        report.problems.append(f"body has {len(lines)} lines (max {MAX_BODY_LINES})")
    if len(body) // 4 > MAX_BODY_TOKENS:
        report.problems.append(f"body is ~{len(body) // 4} tokens (max {MAX_BODY_TOKENS})")

    for rel in LINK_RE.findall(body):
        if not (skill_dir / rel).is_file():
            report.problems.append(f"referenced file missing: {rel}")
        if rel.count("/") != 1:
            report.problems.append(f"reference must be one level deep: {rel}")
    return report


def lint_skills_root(root: Path | None = None) -> list[SkillReport]:
    """Lint `<root>/<scope>/<skill>/SKILL.md`; each scope dir is one `skills=[...]` source."""
    root = root or SKILLS_DIR
    reports: list[SkillReport] = []
    visible = lambda p: p.is_dir() and not p.name.startswith(("_", "."))  # noqa: E731  skip __pycache__
    for scope in sorted(p for p in root.iterdir() if visible(p)):
        skill_dirs = sorted(p for p in scope.iterdir() if visible(p))
        if not skill_dirs:
            bad = SkillReport(scope)
            bad.problems.append("source directory contains no skill directories")
            reports.append(bad)
        reports.extend(lint_skill(d) for d in skill_dirs)
    return reports
