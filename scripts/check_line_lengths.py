#!/usr/bin/env python3
"""Verify that tracked files respect the 500-line limit (REQ-072)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MAX_LINES = 500


def _run_git_command(args: list[str]) -> list[str]:
    """Execute a git command and return its output lines."""

    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _staged_files() -> list[Path]:
    return [Path(p) for p in _run_git_command(["diff", "--cached", "--name-only"])]


def _tracked_files() -> list[Path]:
    return [Path(p) for p in _run_git_command(["ls-files"])]


def _is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(1024)
        return b"\0" not in chunk
    except OSError:
        return False


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _check_files(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        if not _is_text_file(path):
            continue
        line_count = _count_lines(path)
        if line_count > MAX_LINES:
            failures.append(f"{path} has {line_count} lines (limit {MAX_LINES})")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all tracked files instead of just the staged ones.",
    )
    args = parser.parse_args(argv)

    targets = _tracked_files() if args.all else _staged_files()
    failures = _check_files(targets)
    if failures:
        print("The following files exceed the 500-line limit:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print("\nREF-072: Please split or refactor the files before committing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
