"""Ordered candidate-aware ``hmmsearch`` for Astra's pressed-database path.

This module intentionally stops at the same boundary as
``astra_pyhmmer.hmmsearch``: :func:`hmmsearch` accepts already-loaded pressed
profiles and an already-parsed target/candidate batch, then yields one
``TopHits`` object per query.  Astra remains responsible for parsing targets,
grouping model-specific cutoffs, and writing results.

A :class:`~plan7_gpu.adapter.SequenceBatch` is convenient when the caller
wants this function to build the candidate rows.  Passing a precomputed
:class:`~plan7_gpu.adapter.CandidateBatch` avoids filtering twice; its bound
profile identities and F1 threshold are checked before any CPU search starts.
Parallel searches use small contiguous row chunks and one exclusively owned,
exact-base PyHMMER ``Pipeline`` per worker thread. The retained scheduler
refills on any completed cost-balanced task through one bounded reorder window,
while still yielding or raising strictly in canonical task order. The prior
oldest-completion/fixed-row policy remains forceable for audit and rollback.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import operator
import os
from threading import Lock, local
import time
from typing import Any, TYPE_CHECKING

import astra_pyhmmer

from .adapter import (
    CandidateBatch,
    PressedProfilePair,
    SequenceBatch,
    _candidate_state,
)
from ._private_tuning import continuation_shard_trigger, exceeds_ratio

if TYPE_CHECKING:
    from .telemetry_report import TelemetryCollector


_MAX_ROWS_PER_TASK = 8
_TASKS_PER_WORKER_WINDOW = 2
_COMPLETION_REORDER_WINDOWS = 1
_CONTINUATION_SCHEDULER_ENV = "PLAN7_GPU_CONTINUATION_SCHEDULER"
_CONTINUATION_TASK_POLICY_ENV = "PLAN7_GPU_CONTINUATION_TASK_POLICY"
_CONTINUATION_PROFILE_ENV = "PLAN7_GPU_CONTINUATION_PROFILE"
_CONTINUATION_PIPELINE_MADVISE_WORK_HINT_ENV = (
    "PLAN7_GPU_CONTINUATION_PIPELINE_MADVISE_WORK_HINT"
)
_SCHEDULER_OLDEST = "oldest"
_SCHEDULER_COMPLETION = "completion"
_TASK_POLICY_FIXED = "fixed"
_TASK_POLICY_BALANCED = "balanced"
_TASK_POLICY_SHARDED = "sharded"
_scheduler_statistics_lock = Lock()
_scheduler_statistics: dict[str, Any] = {
    "schema_version": 1,
    "call_count": 0,
    "oldest_call_count": 0,
    "completion_call_count": 0,
    "fixed_task_call_count": 0,
    "balanced_task_call_count": 0,
    "sharded_task_call_count": 0,
    "task_count": 0,
    "shard_task_count": 0,
    "task_wall_ns": 0,
    "scheduler_wait_ns": 0,
    "oldest_pending_wait_ns": 0,
    "maximum_pending_tasks": 0,
    "maximum_completed_tasks": 0,
    "maximum_active_workers": 0,
    "pipeline_madvise_count": 0,
    "pipeline_madvise_bytes": 0,
    "task_records": [],
}


# Private cross-package contract consumed by Astra.  Keep this synchronized
# with ``_pipeline._ASTRA_TSV_RENDERER_ABI``; Astra deliberately falls back to
# public TopHits iteration for every other version.
_ASTRA_TSV_RENDERER_ABI = 2


class _AstraTSVRows:
    """Private marker for exact rows rendered before scheduler buffering."""

    __slots__ = ("rows", "row_count", "byte_count")

    def __init__(self, rows: str) -> None:
        if type(rows) is not str:
            raise TypeError("Astra TSV renderer must return exactly str")
        self.rows = rows
        self.row_count = rows.count("\n")
        self.byte_count = len(rows.encode("utf-8"))


def _render_search_result(
    result: Any, renderer: Any, telemetry: bool
) -> Any:
    if renderer is None:
        return result
    if telemetry:
        if type(result) is not tuple or len(result) != 2:
            raise RuntimeError("instrumented search result shape changed")
        hits, evidence = result
        return _AstraTSVRows(renderer(hits)), evidence
    return _AstraTSVRows(renderer(result))


def _continuation_scheduler_mode() -> str:
    mode = os.environ.get(_CONTINUATION_SCHEDULER_ENV, _SCHEDULER_COMPLETION)
    if mode not in (_SCHEDULER_OLDEST, _SCHEDULER_COMPLETION):
        raise ValueError(
            f"{_CONTINUATION_SCHEDULER_ENV} must be "
            f"{_SCHEDULER_OLDEST!r} or {_SCHEDULER_COMPLETION!r}"
        )
    return mode


def _continuation_task_policy() -> str:
    policy = os.environ.get(
        _CONTINUATION_TASK_POLICY_ENV, _TASK_POLICY_BALANCED
    )
    if policy not in (
        _TASK_POLICY_FIXED,
        _TASK_POLICY_BALANCED,
        _TASK_POLICY_SHARDED,
    ):
        raise ValueError(
            f"{_CONTINUATION_TASK_POLICY_ENV} must be "
            f"{_TASK_POLICY_FIXED!r}, {_TASK_POLICY_BALANCED!r}, or "
            f"{_TASK_POLICY_SHARDED!r}"
        )
    return policy


def _balanced_task_bounds(
    work_hints: tuple[int, ...], max_rows: int
) -> tuple[tuple[int, int, int], ...]:
    """Partition canonical rows by immutable work while preserving order."""
    if not work_hints:
        return ()
    if max_rows <= 0:
        raise ValueError("maximum task rows must be positive")
    if any(type(value) is not int or value <= 0 for value in work_hints):
        raise ValueError("continuation work hints must be positive integers")
    fixed_task_count = (len(work_hints) + max_rows - 1) // max_rows
    target = (sum(work_hints) + fixed_task_count - 1) // fixed_task_count
    bounds: list[tuple[int, int, int]] = []
    start = 0
    accumulated = 0
    for row, hint in enumerate(work_hints):
        if row > start and (
            row - start >= max_rows or accumulated + hint > target
        ):
            bounds.append((start, row, accumulated))
            start = row
            accumulated = 0
        accumulated += hint
    bounds.append((start, len(work_hints), accumulated))
    return tuple(bounds)


def _partition_exception_hints(
    work_hints: tuple[int, ...], target: int
) -> tuple[tuple[int, int, int], ...]:
    """Partition one profile at exact exception boundaries."""
    if not work_hints:
        return ()
    if target <= 0:
        raise ValueError("continuation shard target must be positive")
    if any(type(value) is not int or value <= 0 for value in work_hints):
        raise ValueError("continuation shard hints must be positive integers")
    bounds: list[tuple[int, int, int]] = []
    start = 0
    accumulated = 0
    for index, hint in enumerate(work_hints):
        if index > start and accumulated + hint > target:
            bounds.append((start, index, accumulated))
            start = index
            accumulated = 0
        accumulated += hint
    bounds.append((start, len(work_hints), accumulated))
    return tuple(bounds)


def _sharded_task_bounds(
    profile_hints: tuple[int, ...],
    exception_hints: tuple[tuple[int, ...], ...],
    max_rows: int,
    shard_trigger: tuple[int, int] = (2, 1),
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Split only pathological single-profile tasks into exact shards."""
    if len(profile_hints) != len(exception_hints):
        raise ValueError("profile and exception work-hint counts differ")
    base = _balanced_task_bounds(profile_hints, max_rows)
    if not base:
        return ()
    fixed_task_count = (len(profile_hints) + max_rows - 1) // max_rows
    target = (sum(profile_hints) + fixed_task_count - 1) // fixed_task_count
    tasks: list[tuple[int, int, int, int, int, int]] = []
    for start, stop, work_hint in base:
        row_hints = exception_hints[start] if stop == start + 1 else ()
        shards = (
            _partition_exception_hints(row_hints, target)
            if exceeds_ratio(work_hint, target, shard_trigger)
            and len(row_hints) > 1
            else ()
        )
        if len(shards) <= 1:
            tasks.append((start, stop, work_hint, -1, -1, 1))
            continue
        for shard_index, (begin, end, shard_hint) in enumerate(shards):
            tasks.append(
                (
                    start,
                    stop,
                    shard_hint,
                    begin,
                    end,
                    int(shard_index + 1 == len(shards)),
                )
            )
    return tuple(tasks)


def _continuation_task_bounds(
    candidates: CandidateBatch,
    max_rows: int,
    policy: str,
    shard_trigger: tuple[int, int] | None = None,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    if policy in (_TASK_POLICY_BALANCED, _TASK_POLICY_SHARDED):
        state = _candidate_state(candidates)
        if state.sealed_postfilter is not None:
            from . import _pipeline  # type: ignore[attr-defined]

            helper = getattr(
                _pipeline, "_sealed_continuation_work_hints_bound", None
            )
            if callable(helper):
                hints = tuple(helper(state.sealed_postfilter))
                if len(hints) != len(candidates):
                    raise RuntimeError(
                        "continuation work-hint profile count changed"
                    )
                if policy == _TASK_POLICY_SHARDED:
                    shard_helper = getattr(
                        _pipeline,
                        "_sealed_continuation_shard_work_hints_bound",
                        None,
                    )
                    if not callable(shard_helper):
                        raise RuntimeError(
                            "sharded continuation work hints are unavailable"
                        )
                    shard_hints = tuple(shard_helper(state.sealed_postfilter))
                    return _sharded_task_bounds(
                        hints,
                        shard_hints,
                        max_rows,
                        (
                            continuation_shard_trigger()
                            if shard_trigger is None
                            else shard_trigger
                        ),
                    )
                return tuple(
                    (start, stop, work, -1, -1, 1)
                    for start, stop, work in _balanced_task_bounds(
                        hints, max_rows
                    )
                )
    return tuple(
        (start, min(start + max_rows, len(candidates)), 0, -1, -1, 1)
        for start in range(0, len(candidates), max_rows)
    )


def _continuation_profile_enabled() -> bool:
    value = os.environ.get(_CONTINUATION_PROFILE_ENV)
    return value is not None and value not in ("", "0", "false", "False")


def _continuation_pipeline_madvise_work_hint() -> int | None:
    """Return the strict private threshold for releasing invalid DP pages."""
    value = os.environ.get(_CONTINUATION_PIPELINE_MADVISE_WORK_HINT_ENV)
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError(
            f"{_CONTINUATION_PIPELINE_MADVISE_WORK_HINT_ENV} must be a "
            "positive integer"
        )
    threshold = int(value, 10)
    if threshold <= 0:
        raise ValueError(
            f"{_CONTINUATION_PIPELINE_MADVISE_WORK_HINT_ENV} must be a "
            "positive integer"
        )
    return threshold


def _reset_continuation_scheduler_statistics() -> None:
    """Reset private opt-in scheduler measurements."""
    with _scheduler_statistics_lock:
        _scheduler_statistics.update(
            call_count=0,
            oldest_call_count=0,
            completion_call_count=0,
            fixed_task_call_count=0,
            balanced_task_call_count=0,
            sharded_task_call_count=0,
            task_count=0,
            shard_task_count=0,
            task_wall_ns=0,
            scheduler_wait_ns=0,
            oldest_pending_wait_ns=0,
            maximum_pending_tasks=0,
            maximum_completed_tasks=0,
            maximum_active_workers=0,
            pipeline_madvise_count=0,
            pipeline_madvise_bytes=0,
            task_records=[],
        )


def _continuation_scheduler_statistics() -> dict[str, Any]:
    """Return a defensive snapshot of private scheduler measurements."""
    with _scheduler_statistics_lock:
        snapshot = dict(_scheduler_statistics)
        snapshot["task_records"] = tuple(
            dict(record) for record in _scheduler_statistics["task_records"]
        )
    return snapshot


def _record_continuation_scheduler_call(
    *,
    mode: str,
    task_policy: str,
    task_records: list[dict[str, int]],
    scheduler_wait_ns: int,
    oldest_pending_wait_ns: int,
    maximum_pending_tasks: int,
    maximum_completed_tasks: int,
    maximum_active_workers: int,
) -> None:
    with _scheduler_statistics_lock:
        _scheduler_statistics["call_count"] += 1
        _scheduler_statistics[f"{mode}_call_count"] += 1
        _scheduler_statistics[f"{task_policy}_task_call_count"] += 1
        _scheduler_statistics["task_count"] += len(task_records)
        _scheduler_statistics["shard_task_count"] += sum(
            record.get("exception_begin", -1) >= 0
            for record in task_records
        )
        _scheduler_statistics["task_wall_ns"] += sum(
            record["finished_ns"] - record["started_ns"]
            for record in task_records
        )
        _scheduler_statistics["scheduler_wait_ns"] += scheduler_wait_ns
        _scheduler_statistics["oldest_pending_wait_ns"] += (
            oldest_pending_wait_ns
        )
        _scheduler_statistics["maximum_pending_tasks"] = max(
            _scheduler_statistics["maximum_pending_tasks"],
            maximum_pending_tasks,
        )
        _scheduler_statistics["maximum_completed_tasks"] = max(
            _scheduler_statistics["maximum_completed_tasks"],
            maximum_completed_tasks,
        )
        _scheduler_statistics["maximum_active_workers"] = max(
            _scheduler_statistics["maximum_active_workers"],
            maximum_active_workers,
        )
        _scheduler_statistics["task_records"].extend(task_records)


class _ContinuationPool:
    """Private request-scoped continuation workers reused across chunks.

    Astra creates one pool for each immutable set of Pipeline options and closes
    it after the last profile chunk. Calls remain sequential by default.  The
    private ``allow_concurrent_calls`` experiment permits several ordered chunk
    coordinators to submit work to the same fixed worker set, so a later chunk
    can occupy workers while an earlier chunk drains a pathological tail.
    """

    def __init__(
        self,
        cpus: int,
        *,
        allow_concurrent_calls: bool = False,
        task_policy: str | None = None,
        shard_trigger: tuple[int, int] | None = None,
        pipeline_madvise_work_hint: int | None = None,
    ) -> None:
        self.cpus = _positive_cpus(cpus)
        if type(allow_concurrent_calls) is not bool:
            raise TypeError("allow_concurrent_calls must be bool")
        if task_policy is not None and task_policy not in (
            _TASK_POLICY_FIXED,
            _TASK_POLICY_BALANCED,
            _TASK_POLICY_SHARDED,
        ):
            raise ValueError(
                "task_policy must be 'fixed', 'balanced', 'sharded', or None"
            )
        if shard_trigger is not None:
            if (
                type(shard_trigger) is not tuple
                or len(shard_trigger) != 2
                or any(type(value) is not int for value in shard_trigger)
            ):
                raise TypeError(
                    "shard_trigger must be an exact integer ratio tuple "
                    "or None"
                )
            numerator, denominator = shard_trigger
            if (
                denominator <= 0
                or not denominator <= numerator <= 2 * denominator
            ):
                raise ValueError("shard_trigger must be from 1.0 through 2.0")
            if task_policy != _TASK_POLICY_SHARDED:
                raise ValueError(
                    "shard_trigger requires request-local sharded task policy"
                )
        if pipeline_madvise_work_hint is not None and (
            type(pipeline_madvise_work_hint) is not int
            or pipeline_madvise_work_hint <= 0
        ):
            raise ValueError(
                "pipeline_madvise_work_hint must be a positive integer or None"
            )
        self._allow_concurrent_calls = allow_concurrent_calls
        self._task_policy = task_policy
        self._shard_trigger = shard_trigger
        self._pipeline_madvise_work_hint = (
            _continuation_pipeline_madvise_work_hint()
            if pipeline_madvise_work_hint is None
            else pipeline_madvise_work_hint
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.cpus,
            thread_name_prefix="plan7-gpu-astra",
        )
        self._worker_state = local()
        self._lock = Lock()
        self._statistics_lock = Lock()
        self._pipeline_options: dict[str, Any] | None = None
        self._active_call_count = 0
        self._maximum_active_call_count = 0
        self._closed = False
        self._call_count = 0
        self._pipeline_count = 0
        self._pipeline_madvise_count = 0
        self._pipeline_madvise_bytes = 0

    def _acquire(self, cpus: int, pipeline_options: dict[str, Any]) -> None:
        if cpus != self.cpus:
            raise ValueError("continuation pool CPU count does not match search")
        with self._lock:
            if self._closed:
                raise RuntimeError("continuation pool is closed")
            if self._active_call_count and not self._allow_concurrent_calls:
                raise RuntimeError("continuation pool is already in use")
            if self._pipeline_options is None:
                self._pipeline_options = dict(pipeline_options)
            elif self._pipeline_options != pipeline_options:
                raise ValueError(
                    "continuation pool Pipeline options changed between chunks"
                )
            self._active_call_count += 1
            self._maximum_active_call_count = max(
                self._maximum_active_call_count,
                self._active_call_count,
            )
            self._call_count += 1

    def _release(self) -> None:
        with self._lock:
            if self._active_call_count <= 0:
                raise RuntimeError("continuation pool call accounting underflow")
            self._active_call_count -= 1

    def _pipeline(self, pipeline_options: dict[str, Any]) -> Any:
        pipeline = getattr(self._worker_state, "pipeline", None)
        if pipeline is None:
            pipeline = astra_pyhmmer.plan7.Pipeline(**pipeline_options)
            self._worker_state.pipeline = pipeline
            with self._statistics_lock:
                self._pipeline_count += 1
        return pipeline

    def _madvise_pipeline_after(self, pipeline: Any, work_hint: int) -> int:
        threshold = self._pipeline_madvise_work_hint
        if threshold is None or work_hint <= threshold:
            return 0
        if getattr(self._worker_state, "pipeline", None) is not pipeline:
            raise RuntimeError("continuation worker Pipeline ownership changed")
        from . import _pipeline  # type: ignore[attr-defined]

        release = getattr(_pipeline, "_madvise_pipeline_pages_bound", None)
        if not callable(release):
            raise RuntimeError("Pipeline DP-page release is unavailable")
        released = release(pipeline)
        if type(released) is not int or released < 0:
            raise RuntimeError("Pipeline DP-page release accounting changed")
        with self._statistics_lock:
            self._pipeline_madvise_count += 1
            self._pipeline_madvise_bytes += released
        with _scheduler_statistics_lock:
            _scheduler_statistics["pipeline_madvise_count"] += 1
            _scheduler_statistics["pipeline_madvise_bytes"] += released
        return released

    @property
    def statistics(self) -> dict[str, int | bool | None]:
        with self._lock:
            active_call_count = self._active_call_count
            maximum_active_call_count = self._maximum_active_call_count
            closed = self._closed
            call_count = self._call_count
        with self._statistics_lock:
            pipeline_count = self._pipeline_count
            pipeline_madvise_count = self._pipeline_madvise_count
            pipeline_madvise_bytes = self._pipeline_madvise_bytes
        return {
            "schema_version": 2,
            "cpus": self.cpus,
            "call_count": call_count,
            "pipeline_count": pipeline_count,
            "active": bool(active_call_count),
            "active_call_count": active_call_count,
            "maximum_active_call_count": maximum_active_call_count,
            "allow_concurrent_calls": self._allow_concurrent_calls,
            "pipeline_madvise_work_hint_threshold": (
                self._pipeline_madvise_work_hint
            ),
            "pipeline_madvise_count": pipeline_madvise_count,
            "pipeline_madvise_bytes": pipeline_madvise_bytes,
            "closed": closed,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active_call_count:
                raise RuntimeError("cannot close an active continuation pool")
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "_ContinuationPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _positive_cpus(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("cpus must be a positive integer, not bool")
    try:
        cpus = operator.index(value)
    except TypeError as error:
        raise TypeError("cpus must be a positive integer") from error
    if cpus <= 0:
        raise ValueError(f"cpus must be strictly positive, not {cpus!r}")
    return cpus


def _pressed_pairs(
    profile_pairs: Iterable[PressedProfilePair],
) -> tuple[PressedProfilePair, ...]:
    try:
        pairs = tuple(profile_pairs)
    except TypeError as error:
        raise TypeError("profile_pairs must be an iterable") from error
    for pair in pairs:
        if type(pair) is not PressedProfilePair:
            raise TypeError(
                "profile_pairs must contain only objects from load_pressed_profiles"
            )
    return pairs


def _same_bound_pairs(
    pairs: tuple[PressedProfilePair, ...], candidates: CandidateBatch
) -> bool:
    bound_pairs = _candidate_state(candidates).pairs
    return len(pairs) == len(bound_pairs) and all(
        supplied is bound for supplied, bound in zip(pairs, bound_pairs, strict=True)
    )


def _forward_generation_options(
    pipeline_options: dict[str, Any], alphabet: Any
) -> tuple[float, float, bool] | None:
    """Normalize only the Pipeline fields that bind Forward provenance."""
    known = {
        name: pipeline_options[name]
        for name in ("F2", "F3", "bias_filter")
        if name in pipeline_options
    }
    try:
        probe = astra_pyhmmer.plan7.Pipeline(alphabet, **known)
    except (TypeError, ValueError, OverflowError):
        # Preserve the bridge's existing lazy Pipeline error: the real worker
        # construction will report invalid user options when iteration starts.
        return None
    return float(probe.F2), float(probe.F3), bool(probe.bias_filter)


def _forward_augmentation_available() -> bool:
    from . import _native, _pipeline  # type: ignore[attr-defined]

    return (
        callable(getattr(_pipeline, "_filter_scores_seam_available", None))
        and _pipeline._filter_scores_seam_available()
        and callable(
            getattr(_pipeline, "_filter_and_forward_scores_seam_available", None)
        )
        and _pipeline._filter_and_forward_scores_seam_available()
        and getattr(_native, "ForwardProfiles", None) is not None
        and hasattr(_native.SequenceBatch, "forward_candidates_many_raw")
    )


def _prepare_candidates(
    pairs: tuple[PressedProfilePair, ...],
    batch: SequenceBatch | CandidateBatch,
    pipeline_options: dict[str, Any],
    postfilter: bool,
) -> CandidateBatch:
    if type(batch) is SequenceBatch:
        alphabet = batch.alphabet
        f1 = pipeline_options.get("F1", 0.02)
        if postfilter:
            forward_options = None
            if _forward_augmentation_available():
                forward_options = _forward_generation_options(
                    pipeline_options, alphabet
                )
            if forward_options is None:
                candidates = batch.postfilter_batch(pairs, F1=f1)
            else:
                candidates = batch._postfilter_forward_batch(
                    pairs, f1, *forward_options
                )
        else:
            candidates = batch.candidate_batch(pairs, F1=f1)
    elif type(batch) is CandidateBatch:
        candidates = batch
        if not _same_bound_pairs(pairs, candidates):
            raise ValueError(
                "candidate batch is not bound to the supplied profile pairs "
                "in the same order"
            )
        alphabet = _candidate_state(candidates).targets.alphabet
        if "F1" in pipeline_options:
            try:
                requested_f1 = float(pipeline_options["F1"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("pipeline F1 must match candidate batch F1") from error
            if requested_f1 != candidates.F1:
                raise ValueError(
                    f"pipeline F1 {requested_f1!r} does not match "
                    f"candidate batch F1 {candidates.F1!r}"
                )
    else:
        raise TypeError("batch must be exactly SequenceBatch or CandidateBatch")

    # Use the exact normalized threshold stored with the candidate rows.  This
    # matters when a precomputed batch used a non-default F1 and the caller did
    # not otherwise need to repeat that option.
    pipeline_options["F1"] = candidates.F1
    if "alphabet" in pipeline_options:
        if pipeline_options["alphabet"] != alphabet:
            raise ValueError(
                "pipeline alphabet does not match the bound target alphabet"
            )
    else:
        pipeline_options["alphabet"] = alphabet
    return candidates


def _search_row(
    candidates: CandidateBatch,
    row: int,
    pipeline: Any,
    telemetry: bool,
    collector: TelemetryCollector | None = None,
    profile_ordinal: int | None = None,
) -> Any:
    continuation_to_record = None
    try:
        if collector is None:
            result = candidates.search(
                row, pipeline, return_telemetry=telemetry
            )
        else:
            result = candidates.search(
                row, pipeline, return_telemetry=True
            )
            if type(result) is not tuple or len(result) != 2:
                raise RuntimeError("instrumented candidate search shape changed")
            if profile_ordinal is None:
                raise RuntimeError("collector search lacks a profile ordinal")
            hits, continuation = result
            continuation_to_record = (profile_ordinal, continuation)
            result = result if telemetry else hits
        # A row is not committed to the collector until the reusable pipeline
        # has completed its required cleanup.  A final-row clear failure must
        # therefore leave the report incomplete rather than falsely complete.
        pipeline.clear()
        if continuation_to_record is not None:
            collector.record_continuation(*continuation_to_record)
    except BaseException:
        # Match PyHMMER's worker lifecycle and leave no dirty pipeline queued
        # for another row if the failure happened after pipeline mutation.
        try:
            pipeline.clear()
        except BaseException:
            pass
        raise
    return result


def _serial_hmmsearch(
    candidates: CandidateBatch,
    pipeline_options: dict[str, Any],
    telemetry: bool,
    collector: TelemetryCollector | None = None,
    profile_ordinals: tuple[int, ...] | None = None,
    result_renderer: Any = None,
) -> Iterator[Any]:
    if collector is not None and profile_ordinals is None:
        raise RuntimeError("collector search lacks profile ordinals")
    pipeline = astra_pyhmmer.plan7.Pipeline(**pipeline_options)
    for row in range(len(candidates)):
        if collector is None:
            result = _search_row(candidates, row, pipeline, telemetry)
        else:
            result = _search_row(
                candidates,
                row,
                pipeline,
                telemetry,
                collector,
                profile_ordinals[row],
            )
        yield _render_search_result(result, result_renderer, telemetry)


def _threaded_hmmsearch(
    candidates: CandidateBatch,
    cpus: int,
    pipeline_options: dict[str, Any],
    telemetry: bool,
    collector: TelemetryCollector | None = None,
    profile_ordinals: tuple[int, ...] | None = None,
    continuation_pool: _ContinuationPool | None = None,
    result_renderer: Any = None,
) -> Iterator[Any]:
    if collector is not None and profile_ordinals is None:
        raise RuntimeError("collector search lacks profile ordinals")
    worker_count = min(cpus, len(candidates))
    worker_state = local() if continuation_pool is None else None
    owns_executor = continuation_pool is None
    scheduler_mode = _continuation_scheduler_mode()
    task_policy = (
        continuation_pool._task_policy
        if continuation_pool is not None
        and continuation_pool._task_policy is not None
        else _continuation_task_policy()
    )
    shard_trigger = (
        continuation_pool._shard_trigger
        if continuation_pool is not None
        else None
    )
    if task_policy == _TASK_POLICY_SHARDED and (
        telemetry or collector is not None
    ):
        raise ValueError(
            "sharded continuation is an uninstrumented experimental path"
        )
    collect_profile = _continuation_profile_enabled()
    active_lock = Lock()
    active_workers = 0
    maximum_active_workers = 0
    task_records: list[dict[str, int]] = []
    scheduler_wait_ns = 0
    oldest_pending_wait_ns = 0
    maximum_pending_tasks = 0
    maximum_completed_tasks = 0

    # Eight rows is enough to amortize Future creation for the many cheap
    # reject-only rows common after GPU filtering, while keeping retained
    # TopHits and query-skew latency small.  Shrink chunks for short searches
    # so there are up to two initial tasks per worker instead of idling most
    # workers behind one oversized chunk.
    chunk_size = min(
        _MAX_ROWS_PER_TASK,
        max(
            1,
            (len(candidates) + _TASKS_PER_WORKER_WINDOW * worker_count - 1)
            // (_TASKS_PER_WORKER_WINDOW * worker_count),
        ),
    )
    task_bounds = _continuation_task_bounds(
        candidates, chunk_size, task_policy, shard_trigger
    )
    candidate_state = _candidate_state(candidates)
    sealed_postfilter = candidate_state.sealed_postfilter
    pending_shard_row: int | None = None
    pending_shard_hits: list[Any] = []

    if task_policy == _TASK_POLICY_SHARDED and sealed_postfilter is None:
        raise RuntimeError("sharded continuation requires a sealed sparse batch")

    def search_chunk(
        task_ordinal: int,
        start: int,
        stop: int,
        work_hint: int,
        exception_begin: int,
        exception_end: int,
        final_shard: int,
        submitted_ns: int,
    ) -> tuple[list[Any], BaseException | None, dict[str, int]]:
        nonlocal active_workers, maximum_active_workers
        started_ns = time.perf_counter_ns() if collect_profile else 0
        if collect_profile:
            with active_lock:
                active_workers += 1
                maximum_active_workers = max(
                    maximum_active_workers, active_workers
                )
        if continuation_pool is None:
            pipeline = getattr(worker_state, "pipeline", None)
            if pipeline is None:
                pipeline = astra_pyhmmer.plan7.Pipeline(**pipeline_options)
                worker_state.pipeline = pipeline
        else:
            pipeline = continuation_pool._pipeline(pipeline_options)
        hits = []
        error: BaseException | None = None
        try:
            if exception_begin >= 0:
                if stop != start + 1 or sealed_postfilter is None:
                    raise RuntimeError("continuation shard task is malformed")
                from . import _pipeline  # type: ignore[attr-defined]

                try:
                    hits.append(
                        _pipeline._search_hmm_sealed_sparse_journal_v3_shard_bound(
                            sealed_postfilter,
                            start,
                            exception_begin,
                            exception_end,
                            pipeline,
                        )
                    )
                    pipeline.clear()
                except BaseException:
                    try:
                        pipeline.clear()
                    except BaseException:
                        pass
                    raise
            else:
                for row in range(start, stop):
                    if collector is None:
                        result = _search_row(
                            candidates, row, pipeline, telemetry
                        )
                    else:
                        result = _search_row(
                            candidates,
                            row,
                            pipeline,
                            telemetry,
                            collector,
                            profile_ordinals[row],
                        )
                    hits.append(
                        _render_search_result(
                            result, result_renderer, telemetry
                        )
                    )
        except BaseException as caught:
            # A task preserves row-wise failure order: successful rows before
            # the failing row are yielded before this exact error.
            error = caught
        finally:
            if continuation_pool is not None and error is None:
                continuation_pool._madvise_pipeline_after(
                    pipeline, work_hint
                )
            finished_ns = time.perf_counter_ns() if collect_profile else 0
            if collect_profile:
                with active_lock:
                    active_workers -= 1
        return hits, error, {
            "task_ordinal": task_ordinal,
            "row_begin": start,
            "row_end": stop,
            "work_hint": work_hint,
            "exception_begin": exception_begin,
            "exception_end": exception_end,
            "final_shard": final_shard,
            "submitted_ns": submitted_ns,
            "started_ns": started_ns,
            "finished_ns": finished_ns,
        }

    executor = (
        ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="plan7-gpu-astra",
        )
        if continuation_pool is None
        else continuation_pool._executor
    )

    def submit(task_ordinal: int) -> tuple[Future[Any], tuple[int, ...]]:
        (
            start,
            stop,
            work_hint,
            exception_begin,
            exception_end,
            final_shard,
        ) = task_bounds[task_ordinal]
        submitted_ns = time.perf_counter_ns() if collect_profile else 0
        try:
            future = executor.submit(
                search_chunk,
                task_ordinal,
                start,
                stop,
                work_hint,
                exception_begin,
                exception_end,
                final_shard,
                submitted_ns,
            )
        except BaseException as error:
            failed: Future[Any] = Future()
            failed.set_exception(error)
            future = failed
        return future, (
            task_ordinal,
            start,
            stop,
            work_hint,
            exception_begin,
            exception_end,
            final_shard,
            submitted_ns,
        )

    def collect_future(
        future: Future[Any], metadata: tuple[int, ...]
    ) -> tuple[list[Any], BaseException | None, dict[str, int]]:
        (
            task_ordinal,
            start,
            stop,
            work_hint,
            exception_begin,
            exception_end,
            final_shard,
            submitted_ns,
        ) = metadata
        try:
            return future.result()
        except BaseException as error:
            now_ns = time.perf_counter_ns() if collect_profile else 0
            return [], error, {
                "task_ordinal": task_ordinal,
                "row_begin": start,
                "row_end": stop,
                "work_hint": work_hint,
                "exception_begin": exception_begin,
                "exception_end": exception_end,
                "final_shard": final_shard,
                "submitted_ns": submitted_ns,
                "started_ns": now_ns,
                "finished_ns": now_ns,
            }

    def canonical_chunk_hits(
        chunk_hits: list[Any], metadata: tuple[int, ...]
    ) -> tuple[Any, ...]:
        """Merge completed shards only when their canonical row is complete."""
        nonlocal pending_shard_row, pending_shard_hits
        start = metadata[1]
        exception_begin = metadata[4]
        final_shard = metadata[6]
        if exception_begin < 0:
            if pending_shard_row is not None:
                raise RuntimeError("continuation shard sequence is incomplete")
            return tuple(chunk_hits)
        if len(chunk_hits) != 1:
            raise RuntimeError("continuation shard returned the wrong result count")
        if pending_shard_row is None:
            pending_shard_row = start
        elif pending_shard_row != start:
            raise RuntimeError("continuation shard profile order changed")
        pending_shard_hits.append(chunk_hits[0])
        if not final_shard:
            return ()
        base = pending_shard_hits[-1]
        merged = (
            base.merge(*pending_shard_hits[:-1])
            if len(pending_shard_hits) > 1
            else base
        )
        pending_shard_row = None
        pending_shard_hits = []
        return (
            _render_search_result(merged, result_renderer, telemetry),
        )

    active_task_limit = min(
        len(task_bounds), _TASKS_PER_WORKER_WINDOW * worker_count
    )
    pool_acquired = False
    pending_oldest: deque[tuple[Future[Any], tuple[int, ...]]] = deque()
    pending_completion: dict[Future[Any], tuple[int, ...]] = {}
    try:
        if continuation_pool is not None:
            continuation_pool._acquire(cpus, pipeline_options)
            pool_acquired = True
        if scheduler_mode == _SCHEDULER_OLDEST:
            next_task_ordinal = 0
            while len(pending_oldest) < active_task_limit:
                pending_oldest.append(submit(next_task_ordinal))
                next_task_ordinal += 1
            maximum_pending_tasks = len(pending_oldest)

            while pending_oldest:
                future, metadata = pending_oldest.popleft()
                was_pending = not future.done()
                if collect_profile:
                    wait_start_ns = time.perf_counter_ns()
                    chunk_hits, error, record = collect_future(future, metadata)
                    waited_ns = time.perf_counter_ns() - wait_start_ns
                    scheduler_wait_ns += waited_ns
                    if was_pending:
                        oldest_pending_wait_ns += waited_ns
                else:
                    chunk_hits, error, record = collect_future(future, metadata)
                if collect_profile:
                    task_records.append(record)
                # Refill before yielding, preserving the retained scheduler's
                # exact failure and buffering behavior.
                if error is None and next_task_ordinal < len(task_bounds):
                    pending_oldest.append(
                        submit(next_task_ordinal)
                    )
                    next_task_ordinal += 1
                    maximum_pending_tasks = max(
                        maximum_pending_tasks, len(pending_oldest)
                    )
                if error is not None and metadata[4] >= 0:
                    raise error
                for hits in canonical_chunk_hits(chunk_hits, metadata):
                    yield hits
                if error is not None:
                    raise error
        else:
            completed: dict[
                int, tuple[list[Any], BaseException | None, dict[str, int]]
            ] = {}
            task_metadata: dict[int, tuple[int, ...]] = {}
            next_task_ordinal = 0
            next_yield_ordinal = 0
            known_failure_ordinal: int | None = None
            total_task_limit = active_task_limit * (
                1 + _COMPLETION_REORDER_WINDOWS
            )

            def refill_completion() -> None:
                nonlocal next_task_ordinal, maximum_pending_tasks
                while (
                    known_failure_ordinal is None
                    and next_task_ordinal < len(task_bounds)
                    and len(pending_completion) < active_task_limit
                    and len(pending_completion) + len(completed)
                        < total_task_limit
                ):
                    future, metadata = submit(next_task_ordinal)
                    pending_completion[future] = metadata
                    next_task_ordinal += 1
                    maximum_pending_tasks = max(
                        maximum_pending_tasks, len(pending_completion)
                    )

            refill_completion()
            while pending_completion or completed:
                ready = completed.pop(next_yield_ordinal, None)
                if ready is not None:
                    chunk_hits, error, _record = ready
                    metadata = task_metadata.pop(next_yield_ordinal)
                    next_yield_ordinal += 1
                    refill_completion()
                    if error is not None and metadata[4] >= 0:
                        raise error
                    for hits in canonical_chunk_hits(chunk_hits, metadata):
                        yield hits
                    if error is not None:
                        raise error
                    continue
                if not pending_completion:
                    raise RuntimeError(
                        "completion scheduler lost canonical task order"
                    )

                wait_start_ns = (
                    time.perf_counter_ns() if collect_profile else 0
                )
                done, _ = wait(
                    tuple(pending_completion),
                    return_when=FIRST_COMPLETED,
                )
                if collect_profile:
                    scheduler_wait_ns += (
                        time.perf_counter_ns() - wait_start_ns
                    )
                for future in sorted(
                    done,
                    key=lambda item: pending_completion[item][0],
                ):
                    metadata = pending_completion.pop(future)
                    chunk_hits, error, record = collect_future(future, metadata)
                    task_ordinal = metadata[0]
                    completed[task_ordinal] = chunk_hits, error, record
                    task_metadata[task_ordinal] = metadata
                    if collect_profile:
                        task_records.append(record)
                    if error is not None and (
                        known_failure_ordinal is None
                        or task_ordinal < known_failure_ordinal
                    ):
                        known_failure_ordinal = task_ordinal
                maximum_completed_tasks = max(
                    maximum_completed_tasks, len(completed)
                )
                refill_completion()
        if pending_shard_row is not None:
            raise RuntimeError("continuation shard sequence is incomplete")
    finally:
        pending_to_cancel = (
            pending_oldest
            if scheduler_mode == _SCHEDULER_OLDEST
            else pending_completion
        )
        for item in pending_to_cancel:
            future = item[0] if scheduler_mode == _SCHEDULER_OLDEST else item
            future.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)
        elif pool_acquired:
            continuation_pool._release()
        if collect_profile:
            _record_continuation_scheduler_call(
                mode=scheduler_mode,
                task_policy=task_policy,
                task_records=task_records,
                scheduler_wait_ns=scheduler_wait_ns,
                oldest_pending_wait_ns=oldest_pending_wait_ns,
                maximum_pending_tasks=maximum_pending_tasks,
                maximum_completed_tasks=maximum_completed_tasks,
                maximum_active_workers=maximum_active_workers,
            )


def _hmmsearch_impl(
    profile_pairs: Iterable[PressedProfilePair],
    batch: SequenceBatch | CandidateBatch,
    *,
    cpus: int = 1,
    postfilter: bool = False,
    telemetry: bool = False,
    telemetry_collector: TelemetryCollector | None = None,
    profile_ordinals: Iterable[int] | None = None,
    profile_keys: Iterable[str | None] | None = None,
    _continuation_pool: _ContinuationPool | None = None,
    _result_renderer: Any = None,
    **pipeline_options: Any,
) -> Iterator[Any]:
    """Yield candidate-aware HMM searches in supplied query order.

    Parameters mirror the narrow portion of ``astra_pyhmmer.hmmsearch`` replaced in
    Astra's installed pressed-database bulk path.  ``pipeline_options`` are
    forwarded unchanged to exact base :class:`astra_pyhmmer.plan7.Pipeline` objects,
    except that ``alphabet`` is inferred when absent and ``F1`` is normalized
    to the threshold bound to the candidate rows.

    ``cpus`` is deliberately stricter than PyHMMER's convenience function: it
    must be a positive integer, with ``1`` executing lazily in the consuming
    thread.  Values of ``0`` are rejected instead of consulting host topology,
    so Astra's explicit ``--threads`` allocation cannot be exceeded silently.
    Threaded iteration keeps at most twice the effective worker count in-flight
    chunks of at most eight rows each, bounding both the reorder buffer and
    queued work.  The returned iterator owns all pipelines and worker threads
    until it is exhausted or closed.

    Telemetry is deliberately narrower than ordinary candidate generation.
    ``telemetry=True`` or an explicit ``telemetry_collector`` requires a
    precomputed sealed fused :class:`CandidateBatch` whose generation sidecar
    is already present; unsupported ``SequenceBatch`` or legacy batches are
    rejected synchronously before filtering or searching begins. A collector
    records validated continuation evidence while preserving ordinary
    ``TopHits`` output unless ``telemetry=True`` also requests the historical
    ``(TopHits, evidence)`` opt-in shape.
    """
    worker_count = _positive_cpus(cpus)
    if (
        _continuation_pool is not None
        and type(_continuation_pool) is not _ContinuationPool
    ):
        raise TypeError("_continuation_pool must be exactly _ContinuationPool")
    if type(postfilter) is not bool:
        raise TypeError("postfilter must be bool")
    if type(telemetry) is not bool:
        raise TypeError("telemetry must be bool")
    if _result_renderer is not None and not callable(_result_renderer):
        raise TypeError("_result_renderer must be callable or None")
    if telemetry_collector is not None:
        from .telemetry_report import TelemetryCollector

        if type(telemetry_collector) is not TelemetryCollector:
            raise TypeError(
                "telemetry_collector must be exactly TelemetryCollector"
            )
    if telemetry_collector is None and (
        profile_ordinals is not None or profile_keys is not None
    ):
        raise ValueError(
            "profile ordinals and keys require an explicit telemetry collector"
        )
    telemetry_requested = telemetry or telemetry_collector is not None
    if telemetry_requested:
        if type(batch) is not CandidateBatch:
            raise ValueError(
                "telemetry requires a precomputed sealed fused CandidateBatch; "
                "SequenceBatch generation is not instrumented by this bridge"
            )
        if batch.generation_statistics is None:
            raise ValueError(
                "telemetry requires a precomputed CandidateBatch generated "
                "with telemetry=True"
            )
    pairs = _pressed_pairs(profile_pairs)
    options = dict(pipeline_options)
    candidates = _prepare_candidates(pairs, batch, options, postfilter)

    if len(candidates) != len(pairs):
        # This should be guaranteed by CandidateBatch construction, but keep
        # the bridge's row/query contract explicit at its trust boundary.
        raise ValueError("candidate row count does not match profile pair count")
    ordinal_values: tuple[int, ...] | None = None
    if telemetry_collector is not None:
        ordinal_values = (
            tuple(pair.ordinal for pair in pairs)
            if profile_ordinals is None
            else tuple(profile_ordinals)
        )
        key_values = None if profile_keys is None else tuple(profile_keys)
        generation_statistics = candidates.generation_statistics
        if generation_statistics is None:
            raise RuntimeError("instrumented generation statistics disappeared")
        telemetry_collector.record_generation(
            generation_statistics,
            ordinal_values,
            profile_keys=key_values,
        )
    if not pairs:
        return iter(())
    if worker_count == 1 and _continuation_pool is None:
        if telemetry_collector is None:
            return _serial_hmmsearch(
                candidates,
                options,
                telemetry,
                result_renderer=_result_renderer,
            )
        return _serial_hmmsearch(
            candidates,
            options,
            telemetry,
            telemetry_collector,
            ordinal_values,
            _result_renderer,
        )
    if telemetry_collector is None:
        return _threaded_hmmsearch(
            candidates,
            worker_count,
            options,
            telemetry,
            continuation_pool=_continuation_pool,
            result_renderer=_result_renderer,
        )
    return _threaded_hmmsearch(
        candidates,
        worker_count,
        options,
        telemetry,
        telemetry_collector,
        ordinal_values,
        _continuation_pool,
        _result_renderer,
    )


def hmmsearch(
    profile_pairs: Iterable[PressedProfilePair],
    batch: SequenceBatch | CandidateBatch,
    *,
    cpus: int = 1,
    postfilter: bool = False,
    telemetry: bool = False,
    telemetry_collector: TelemetryCollector | None = None,
    profile_ordinals: Iterable[int] | None = None,
    profile_keys: Iterable[str | None] | None = None,
    **pipeline_options: Any,
) -> Iterator[Any]:
    """Yield exact candidate-aware searches through the stable public bridge."""
    return _hmmsearch_impl(
        profile_pairs,
        batch,
        cpus=cpus,
        postfilter=postfilter,
        telemetry=telemetry,
        telemetry_collector=telemetry_collector,
        profile_ordinals=profile_ordinals,
        profile_keys=profile_keys,
        **pipeline_options,
    )


def _hmmsearch_with_continuation_pool(
    profile_pairs: Iterable[PressedProfilePair],
    batch: SequenceBatch | CandidateBatch,
    *,
    continuation_pool: _ContinuationPool,
    cpus: int = 1,
    postfilter: bool = False,
    telemetry: bool = False,
    telemetry_collector: TelemetryCollector | None = None,
    profile_ordinals: Iterable[int] | None = None,
    profile_keys: Iterable[str | None] | None = None,
    **pipeline_options: Any,
) -> Iterator[Any]:
    """Private Astra entry retaining worker Pipelines across profile chunks."""
    return _hmmsearch_impl(
        profile_pairs,
        batch,
        cpus=cpus,
        postfilter=postfilter,
        telemetry=telemetry,
        telemetry_collector=telemetry_collector,
        profile_ordinals=profile_ordinals,
        profile_keys=profile_keys,
        _continuation_pool=continuation_pool,
        **pipeline_options,
    )


def _astra_tsv_renderer() -> Any:
    from . import _pipeline  # type: ignore[attr-defined]

    if getattr(_pipeline, "_ASTRA_TSV_RENDERER_ABI", None) != (
        _ASTRA_TSV_RENDERER_ABI
    ):
        raise RuntimeError("native Astra TSV renderer ABI is unavailable")
    renderer = getattr(_pipeline, "_astra_tsv_rows_bound", None)
    if not callable(renderer):
        raise RuntimeError("native Astra TSV renderer is unavailable")
    return renderer


def _hmmsearch_astra_tsv(
    profile_pairs: Iterable[PressedProfilePair],
    batch: SequenceBatch | CandidateBatch,
    *,
    cpus: int = 1,
    postfilter: bool = False,
    telemetry: bool = False,
    telemetry_collector: TelemetryCollector | None = None,
    profile_ordinals: Iterable[int] | None = None,
    profile_keys: Iterable[str | None] | None = None,
    **pipeline_options: Any,
) -> Iterator[Any]:
    """Private Astra path rendering exact rows inside continuation workers."""
    return _hmmsearch_impl(
        profile_pairs,
        batch,
        cpus=cpus,
        postfilter=postfilter,
        telemetry=telemetry,
        telemetry_collector=telemetry_collector,
        profile_ordinals=profile_ordinals,
        profile_keys=profile_keys,
        _result_renderer=_astra_tsv_renderer(),
        **pipeline_options,
    )


def _hmmsearch_astra_tsv_with_continuation_pool(
    profile_pairs: Iterable[PressedProfilePair],
    batch: SequenceBatch | CandidateBatch,
    *,
    continuation_pool: _ContinuationPool,
    cpus: int = 1,
    postfilter: bool = False,
    telemetry: bool = False,
    telemetry_collector: TelemetryCollector | None = None,
    profile_ordinals: Iterable[int] | None = None,
    profile_keys: Iterable[str | None] | None = None,
    **pipeline_options: Any,
) -> Iterator[Any]:
    """Worker-rendering Astra path retaining request-scoped Pipelines."""
    return _hmmsearch_impl(
        profile_pairs,
        batch,
        cpus=cpus,
        postfilter=postfilter,
        telemetry=telemetry,
        telemetry_collector=telemetry_collector,
        profile_ordinals=profile_ordinals,
        profile_keys=profile_keys,
        _continuation_pool=continuation_pool,
        _result_renderer=_astra_tsv_renderer(),
        **pipeline_options,
    )


__all__ = ["hmmsearch"]
