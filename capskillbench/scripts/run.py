#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CaP SkillBench across supported environment profiles.")
    parser.add_argument(
        "--env-profile",
        choices=["libero"],
        default="libero",
        help="Environment profile to run. Only libero is implemented in this scaffold.",
    )
    parser.add_argument("profile_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the profile runner.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    forwarded = args.profile_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.env_profile == "libero":
        from capskillbench.scripts import run_libero

        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *forwarded]
            return run_libero.main()
        finally:
            sys.argv = old_argv

    raise ValueError(f"unsupported env profile: {args.env_profile}")


if __name__ == "__main__":
    raise SystemExit(main())
