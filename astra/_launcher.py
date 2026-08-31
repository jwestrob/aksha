"""Minimal console launcher for Astra's high-thread CPU allocator policy.

This module must remain free of imports from :mod:`astra.main`, PyHMMER, or
other allocation-heavy runtime modules.  The console entry point can then set
the glibc arena limit and replace itself with a fresh interpreter before any
of those modules are imported.  Programmatic/library entry points bypass this
module and retain their existing behavior.
"""

from __future__ import annotations

import os
import sys


_GLIBC_ARENA_ENV = "MALLOC_ARENA_MAX"
_ASTRA_ARENA_ENV = "ASTRA_CPU_MALLOC_ARENA_MAX"
_REEXEC_SENTINEL_ENV = "_ASTRA_CPU_ALLOCATOR_REEXEC"
_DEFAULT_ARENA_MAX = 24
_MINIMUM_AUTO_THREADS = 64
_MAXIMUM_ARENA_MAX = (1 << 31) - 1


def _is_linux_glibc() -> bool:
    """Return whether the current interpreter uses glibc on Linux."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        version = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        return False
    return isinstance(version, str) and version.startswith("glibc ")


def _requested_threads(arguments: tuple[str, ...]) -> int | None:
    """Return the CLI worker count, or ``None`` for malformed input.

    Astra's parser owns diagnostics for malformed ``--threads`` arguments.
    The launcher simply declines to re-exec in that case so it cannot mask the
    existing command-line error or alter its ordering.
    """
    value = "1"
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--threads":
            index += 1
            if index >= len(arguments):
                return None
            value = arguments[index]
        elif argument.startswith("--threads="):
            value = argument.partition("=")[2]
        index += 1
    try:
        return int(value)
    except ValueError:
        return None


def _is_cpu_search(arguments: tuple[str, ...]) -> bool:
    """Return whether raw arguments select the validated CPU search path."""
    if not arguments or arguments[0] != "search":
        return False
    if any(
        len(option := argument.partition("=")[0]) > 2
        and option.startswith("--")
        and "--gpu-manifest".startswith(option)
        for argument in arguments[1:]
    ):
        return False
    threads = _requested_threads(arguments[1:])
    return threads is not None and threads >= _MINIMUM_AUTO_THREADS


def _arena_max(environment: dict[str, str]) -> int | None:
    """Parse the Astra-specific arena control.

    Zero is an explicit opt-out.  A standard ``MALLOC_ARENA_MAX`` value is
    handled by the caller before this parser and always takes precedence.
    """
    value = environment.get(_ASTRA_ARENA_ENV)
    if value is None:
        return _DEFAULT_ARENA_MAX
    if value == "0":
        return None
    if (
        not value
        or not value.isascii()
        or not value.isdigit()
        or value[0] == "0"
    ):
        raise ValueError(
            f"{_ASTRA_ARENA_ENV} must be 0 or a canonical positive integer"
        )
    parsed = int(value)
    if parsed > _MAXIMUM_ARENA_MAX:
        raise ValueError(
            f"{_ASTRA_ARENA_ENV} must be at most {_MAXIMUM_ARENA_MAX}"
        )
    return parsed


def _reexec_environment(
    arguments: tuple[str, ...], environment: dict[str, str]
) -> dict[str, str] | None:
    """Return the environment for a required allocator re-exec, if any."""
    # The standard glibc setting is entirely user-owned, including an empty or
    # otherwise unusual value.  Never reinterpret or overwrite it.
    if _GLIBC_ARENA_ENV in environment:
        return None
    if _REEXEC_SENTINEL_ENV in environment:
        return None
    if not _is_cpu_search(arguments) or not _is_linux_glibc():
        return None
    arena_max = _arena_max(environment)
    if arena_max is None:
        return None
    updated = environment.copy()
    updated[_GLIBC_ARENA_ENV] = str(arena_max)
    updated[_REEXEC_SENTINEL_ENV] = "1"
    return updated


def _dispatch() -> object:
    """Import and run the existing CLI only after allocator routing."""
    from .main import main as astra_main

    return astra_main()


def main() -> object:
    """Run the Astra console command with the validated CPU arena policy."""
    arguments = tuple(sys.argv[1:])
    try:
        environment = _reexec_environment(arguments, dict(os.environ))
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if environment is not None:
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "astra.main", *arguments],
            environment,
        )
        raise RuntimeError("os.execvpe returned unexpectedly")
    return _dispatch()
