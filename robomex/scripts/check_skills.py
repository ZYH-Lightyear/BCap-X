"""Validate a RoboMEx skill library from the command line."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from robomex.skills.lint import collect_library_errors

DEFAULT_ROOT = Path(__file__).parents[1] / "skills" / "builtin"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="skill library root (default: RoboMEx builtin library)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors = collect_library_errors(root)
    if errors:
        print(f"Skill library check failed ({len(errors)} error(s)):")
        for error in errors:
            print(f"- {error}")
        return 1
    count = len(list(root.glob("*/*/SKILL.md")))
    print(f"Skill library check passed: {count} package(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
