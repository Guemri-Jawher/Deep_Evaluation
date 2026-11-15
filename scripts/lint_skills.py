"""Check every SKILL.md against the Agent Skills spec; exit 1 if any skill has a problem.

Needs no LLM, no network and no Langfuse, so it is safe for a pre-commit hook or CI.

Examples:
    uv run python scripts/lint_skills.py
    uv run python scripts/lint_skills.py --strict          # warnings fail too
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(p))

from skills.skills_lint import lint_skills_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="skills directory (default: src/skills)")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()

    reports = lint_skills_root(args.root)
    failed = 0
    for r in reports:
        bad = bool(r.problems) or (args.strict and bool(r.warnings))
        failed += bad
        label = r.name or r.path.name
        print(f"[{'FAIL' if bad else 'OK'}] {r.path.parent.name}/{label}")
        for text in r.problems:
            print(f"    problem: {text}")
        for text in r.warnings:
            print(f"    warning: {text}")
    print(f"\n{len(reports) - failed}/{len(reports)} skills passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
