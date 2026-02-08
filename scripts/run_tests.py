"""Run the Aksha test suite from Python."""

from __future__ import annotations

import argparse
import sys

import pytest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aksha tests")
    parser.add_argument(
        "-m",
        "--mark",
        dest="mark_expr",
        help="Only run tests matching the given marker expression (e.g., 'not integration')",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments passed through to pytest",
    )

    args = parser.parse_args(argv)

    pytest_cmd: list[str] = ["-s"]  # show test stdout/stderr; don't capture
    if args.mark_expr:
        pytest_cmd += ["-m", args.mark_expr]
    pytest_cmd += args.pytest_args or ["tests"]

    return pytest.main(pytest_cmd)


if __name__ == "__main__":
    sys.exit(main())
