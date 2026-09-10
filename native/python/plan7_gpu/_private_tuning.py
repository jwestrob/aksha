"""Private, default-preserving production tuning contracts.

These environment variables intentionally remain outside the public Python
API.  Keeping their parsing here makes the exact integer decision boundaries
host-testable without constructing CUDA state.
"""

from __future__ import annotations

from fractions import Fraction
import os


CONTINUATION_SHARD_TRIGGER_ENV = (
    "PLAN7_GPU_CONTINUATION_SHARD_TRIGGER"
)
FORWARD_CPU_MAX_CELLS_ENV = "PLAN7_GPU_FORWARD_CPU_MAX_CELLS"

_DEFAULT_CONTINUATION_SHARD_TRIGGER = Fraction(2, 1)
_MIN_CONTINUATION_SHARD_TRIGGER = Fraction(1, 1)
_MAX_CONTINUATION_SHARD_TRIGGER = Fraction(2, 1)
_DEFAULT_FORWARD_CPU_MAX_CELLS = 200_000
_UINT64_MAX = (1 << 64) - 1


def continuation_shard_trigger() -> tuple[int, int]:
    """Return the exact private shard-trigger ratio as numerator/denominator."""
    raw = os.environ.get(CONTINUATION_SHARD_TRIGGER_ENV)
    if raw is None:
        trigger = _DEFAULT_CONTINUATION_SHARD_TRIGGER
    else:
        try:
            trigger = Fraction(raw)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"{CONTINUATION_SHARD_TRIGGER_ENV} must be a finite ratio "
                "from 1.0 through 2.0"
            ) from error
    if not (
        _MIN_CONTINUATION_SHARD_TRIGGER
        <= trigger
        <= _MAX_CONTINUATION_SHARD_TRIGGER
    ):
        raise ValueError(
            f"{CONTINUATION_SHARD_TRIGGER_ENV} must be from 1.0 through 2.0"
        )
    return trigger.numerator, trigger.denominator


def exceeds_ratio(
    value: int, reference: int, ratio: tuple[int, int]
) -> bool:
    """Compare ``value > reference * ratio`` using exact integer arithmetic."""
    numerator, denominator = ratio
    return value * denominator > reference * numerator


def auto_forward_cpu_max_cells() -> int:
    """Return the retained auto-ownership cutoff, defaulting to 200,000 cells."""
    raw = os.environ.get(FORWARD_CPU_MAX_CELLS_ENV)
    if raw is None:
        return _DEFAULT_FORWARD_CPU_MAX_CELLS
    try:
        cells = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{FORWARD_CPU_MAX_CELLS_ENV} must be a positive integer"
        ) from error
    if cells <= 0 or cells > _UINT64_MAX:
        raise ValueError(
            f"{FORWARD_CPU_MAX_CELLS_ENV} must be a positive uint64 integer"
        )
    return cells
