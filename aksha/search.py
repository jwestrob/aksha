
import gc
import inspect
import io
import os
import sys
import time
import logging
import shutil
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Condition, Event, Lock, Thread
from typing import NamedTuple
from tqdm import tqdm
import astra_pyhmmer
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from aksha import initialize
from aksha import rp16 as rp16_module


PRESSED_SUFFIXES = ('h3m', 'h3i', 'h3f', 'h3p')
HMM_CHUNK_SIZE = 2000
CPU_STREAM_PRESSED_ENV = "ASTRA_CPU_STREAM_PRESSED"
CPU_STREAM_PRESSED_AUTO = "auto"
GPU_CELL_CAP = 100_000_000
GPU_PROFILE_CELL_CAP_ENV = 'ASTRA_GPU_PROFILE_CELL_CAP'
GPU_PROFILE_CELL_CAPS = (100_000_000, 200_000_000, 300_000_000, 400_000_000)
GPU_CHUNK_LOCAL_PROFILE_PACK_ENV = 'ASTRA_GPU_CHUNK_LOCAL_PROFILE_PACK'
GPU_CHUNK_LOCAL_PROFILE_PACK_MIN_PRESSED_BYTES = 4 << 30
GPU_TIMING_ENV = 'ASTRA_GPU_OVERLAP_TIMING'
GPU_SERIAL_ENV = 'ASTRA_GPU_PROFILE_SERIAL'
GPU_LEGACY_OVERLAP_ENV = 'ASTRA_GPU_PROFILE_LEGACY_OVERLAP'
GPU_LEGACY_OVERLAP_VALUE = 'single-prefetch'
GPU_READY_QUEUE_DEPTH_ENV = 'ASTRA_GPU_READY_QUEUE_DEPTH'
GPU_READY_QUEUE_BYTES_ENV = 'ASTRA_GPU_READY_QUEUE_BYTES'
GPU_CONTINUATION_POOL_ENV = 'ASTRA_GPU_CONTINUATION_POOL'
GPU_CONTINUATION_WINDOW_ENV = 'ASTRA_GPU_CONTINUATION_WINDOW'
GPU_CONTINUATION_WORKERS_ENV = 'ASTRA_GPU_CONTINUATION_WORKERS'
GPU_INTRAROW_RELEASE_MIN_BYTES_ENV = (
    'PLAN7_GPU_INTRAROW_RELEASE_MIN_BYTES'
)
GPU_DOMAIN_GUARD = 2.0e-4
GPU_READY_QUEUE_CAPACITY = 1
GPU_PRODUCER_LOOKAHEAD_CAPACITY = GPU_READY_QUEUE_CAPACITY + 1
GPU_LIVE_CANDIDATE_CAPACITY = GPU_READY_QUEUE_CAPACITY + 2
GPU_READY_QUEUE_DEPTHS = (1, 2, 4)
GPU_CONTINUATION_WINDOWS = (1, 2, 4)
GPU_READY_QUEUE_MAX_BYTES = (1 << 63) - 1
GPU_PRODUCTION_TARGET_MINIMUM = 65_536
GPU_PRODUCTION_THREAD_COUNT = 64
GPU_PRODUCTION_CONTINUATION_WINDOW = 4
GPU_PRODUCTION_SHARD_TRIGGER = (3, 2)
GPU_PRODUCTION_PIPELINE_MADVISE_WORK_HINT = 1_300_000_000
GPU_PRODUCTION_INTRAROW_RELEASE_MIN_BYTES = 16 << 20
GPU_PRODUCTION_FORWARD_CPU_MAX_CELLS = 200_000
GPU_PRODUCTION_EVALUE = 1e-15
GPU_PRODUCTION_OVERRIDE_ENVS = (
    GPU_PROFILE_CELL_CAP_ENV,
    GPU_CHUNK_LOCAL_PROFILE_PACK_ENV,
    GPU_SERIAL_ENV,
    GPU_LEGACY_OVERLAP_ENV,
    GPU_READY_QUEUE_DEPTH_ENV,
    GPU_READY_QUEUE_BYTES_ENV,
    GPU_CONTINUATION_POOL_ENV,
    GPU_CONTINUATION_WINDOW_ENV,
    GPU_CONTINUATION_WORKERS_ENV,
    'PLAN7_GPU_CONTINUATION_SCHEDULER',
    'PLAN7_GPU_CONTINUATION_TASK_POLICY',
    'PLAN7_GPU_CONTINUATION_SHARD_TRIGGER',
    'PLAN7_GPU_CONTINUATION_ONESHOT_PIPELINE_WORK_HINT',
    'PLAN7_GPU_CONTINUATION_PIPELINE_MADVISE_WORK_HINT',
    'PLAN7_GPU_AVX512_TAIL_MADVISE',
    GPU_INTRAROW_RELEASE_MIN_BYTES_ENV,
    'PLAN7_GPU_FORWARD_OWNERSHIP',
    'PLAN7_GPU_FORWARD_CPU_MIN_CELLS',
    'PLAN7_GPU_FORWARD_CPU_MIN_LENGTH',
    'PLAN7_GPU_FORWARD_CPU_MAX_CELLS',
    'PLAN7_GPU_DOMAIN_OWNERSHIP',
    'PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP',
    'PLAN7_GPU_F1_RAW_XE',
    'PLAN7_GPU_SSV_IDENTITY_PADDING',
    'PLAN7_GPU_VIT_LENGTH_CACHE',
    'PLAN7_GPU_FULL_MSV_ARITHMETIC',
    'PLAN7_GPU_FULL_MSV_POLICY',
    'PLAN7_GPU_SSV_LENGTH_METADATA',
    'PLAN7_GPU_SSV_PROFILE_POLICY',
    'PLAN7_GPU_FILTER_TAIL_SIMD',
    'PLAN7_GPU_FILTER_TAIL_SIMD_TEST_FALLBACK',
)
_NATIVE_TSV_ROWS_UNRESOLVED = object()
_native_tsv_rows = _NATIVE_TSV_ROWS_UNRESOLVED
_ASTRA_TSV_RENDERER_ABI = 2
_gpu_request_page_release_lock = Lock()


class GPUConfigurationError(ValueError):
    """Raised when an explicit installed-database GPU request is invalid."""


class _GPURequestTuning(NamedTuple):
    """Immutable, request-local selection of the measured large-GPU path."""

    automatic: bool
    pressed_bytes: int
    reason: str


def _disabled_gpu_request_tuning(reason, pressed_bytes=0):
    return _GPURequestTuning(False, pressed_bytes, reason)


def _gpu_automatic_sparse_journal_v3(
    request_tuning, filter_tail_simd=False,
):
    """Select sparse journal v3 only as part of the exact automatic bundle."""
    return _gpu_automatic_continuation_tuning(
        request_tuning, filter_tail_simd
    )


def _gpu_automatic_continuation_tuning(
    request_tuning, filter_tail_simd=False,
):
    """Select the measured continuation stack for KOFAM or PFAM."""
    if type(filter_tail_simd) is not bool:
        raise TypeError("filter_tail_simd must be bool")
    if request_tuning is None:
        return filter_tail_simd
    if type(request_tuning) is not _GPURequestTuning:
        raise TypeError("request_tuning must be exactly _GPURequestTuning or None")
    return request_tuning.automatic or filter_tail_simd


def gpu_filter_tail_simd_request(
    options, threads, target_count, *, installed_attested=False,
    single_mapped_database=False, profile_count=None, cache_enabled=False,
):
    """Select the exact retained PFAM gathering-cutoff SIMD request shape."""
    if not installed_attested or not single_mapped_database:
        return False
    if type(threads) is not int or threads != GPU_PRODUCTION_THREAD_COUNT:
        return False
    if type(target_count) is not int or target_count <= GPU_PRODUCTION_TARGET_MINIMUM:
        return False
    if type(profile_count) is not int or profile_count < 256:
        return False
    if cache_enabled is not False:
        return False
    if any(name in os.environ for name in GPU_PRODUCTION_OVERRIDE_ENVS):
        return False
    try:
        kwargs = define_kwargs(options)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return kwargs == {'bit_cutoffs': 'gathering'}


def _gpu_continuation_pool_enabled(default=False):
    if type(default) is not bool:
        raise TypeError("default continuation-pool policy must be bool")
    value = os.environ.get(GPU_CONTINUATION_POOL_ENV)
    if value is None:
        return default
    if value == '0':
        return False
    if value == '1':
        return True
    raise GPUConfigurationError(
        f"{GPU_CONTINUATION_POOL_ENV} must be exactly '0' or '1'"
    )


def _gpu_continuation_window(default=1):
    if default not in GPU_CONTINUATION_WINDOWS:
        raise ValueError("default continuation window is unsupported")
    value = os.environ.get(GPU_CONTINUATION_WINDOW_ENV)
    if value is None:
        return default
    if value not in tuple(str(item) for item in GPU_CONTINUATION_WINDOWS):
        allowed = ', '.join(str(item) for item in GPU_CONTINUATION_WINDOWS)
        raise GPUConfigurationError(
            f"{GPU_CONTINUATION_WINDOW_ENV} accepts only {allowed}"
        )
    return int(value)


def _gpu_continuation_worker_count(available, profile_session, pool_enabled):
    """Apply the private continuation-only cap after producer reservation."""
    value = os.environ.get(GPU_CONTINUATION_WORKERS_ENV)
    if value is None:
        return available
    if not value.isascii() or not value.isdecimal() or value.startswith('0'):
        raise GPUConfigurationError(
            f"{GPU_CONTINUATION_WORKERS_ENV} must be a canonical positive integer"
        )
    requested = int(value)
    if profile_session is None or not pool_enabled:
        raise GPUConfigurationError(
            f"{GPU_CONTINUATION_WORKERS_ENV} requires an active GPU profile "
            "session and continuation pool"
        )
    if requested > available:
        raise GPUConfigurationError(
            f"{GPU_CONTINUATION_WORKERS_ENV} cannot exceed the {available} "
            "continuation workers available after producer reservation"
        )
    return requested


def _new_gpu_continuation_pools(chunks, threads, enabled,
                                continuation_window=1,
                                task_policy=None,
                                shard_trigger=None,
                                pipeline_madvise_work_hint=None):
    if not enabled:
        return None
    try:
        from plan7_gpu.astra_search import _ContinuationPool
    except (ImportError, AttributeError) as error:
        raise GPUConfigurationError(
            "installed plan7_gpu lacks request-scoped continuation pools"
        ) from error
    pools = {}
    try:
        for pipeline_options in _gpu_pipeline_option_groups(chunks):
            key = id(pipeline_options)
            if key not in pools:
                pools[key] = _ContinuationPool(
                    threads,
                    allow_concurrent_calls=continuation_window > 1,
                    task_policy=task_policy,
                    shard_trigger=shard_trigger,
                    pipeline_madvise_work_hint=(
                        pipeline_madvise_work_hint
                    ),
                )
    except BaseException:
        for pool in pools.values():
            pool.close()
        raise
    return pools


def _close_gpu_continuation_pools(pools, gpu_metrics):
    if pools is None:
        return
    active_error = sys.exc_info()[0] is not None
    close_error = None
    for pool in pools.values():
        try:
            pool.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
        statistics = pool.statistics
        if gpu_metrics is not None:
            gpu_metrics.continuation_pool_call_count += statistics['call_count']
            gpu_metrics.continuation_pipeline_count += statistics['pipeline_count']
    if close_error is not None and not active_error:
        raise close_error


def _gpu_intrarow_release_min_bytes(default=None):
    value = os.environ.get(GPU_INTRAROW_RELEASE_MIN_BYTES_ENV)
    if value is None:
        return default
    if not value.isascii() or not value.isdecimal():
        raise GPUConfigurationError(
            f"{GPU_INTRAROW_RELEASE_MIN_BYTES_ENV} must be a nonnegative "
            "integer"
        )
    parsed = int(value, 10)
    if parsed > (1 << 64) - 1:
        raise GPUConfigurationError(
            f"{GPU_INTRAROW_RELEASE_MIN_BYTES_ENV} exceeds uint64"
        )
    return parsed


def _configure_gpu_request_page_release(
    request_tuning, filter_tail_simd=False,
):
    """Apply and return scoped core hooks without mutating the environment."""
    if type(filter_tail_simd) is not bool:
        raise TypeError("filter_tail_simd must be bool")
    from plan7_gpu import _pipeline

    automatic = bool(
        request_tuning is not None and request_tuning.automatic
    )
    intrarow_min_bytes = _gpu_intrarow_release_min_bytes(
        GPU_PRODUCTION_INTRAROW_RELEASE_MIN_BYTES if automatic else None
    )
    swap_avx = getattr(
        _pipeline, '_swap_avx512_tail_madvise_bound', None
    )
    swap_intrarow = getattr(
        _pipeline, '_swap_intrarow_page_release_bound', None
    )
    swap_filter_tail = getattr(
        _pipeline, '_swap_filter_tail_simd_bound', None
    )
    if not automatic and intrarow_min_bytes is None and not filter_tail_simd:
        return None
    previous_avx = None
    previous_intrarow = None
    previous_filter_tail = None
    avx_configured = False
    intrarow_configured = False
    filter_tail_configured = False
    _gpu_request_page_release_lock.acquire()
    try:
        if automatic:
            if not callable(swap_avx):
                raise GPUConfigurationError(
                    "installed plan7_gpu lacks request-scoped AVX page release"
                )
            previous_avx = swap_avx(True)
            avx_configured = True
        if intrarow_min_bytes is not None:
            if not callable(swap_intrarow):
                raise GPUConfigurationError(
                    "installed plan7_gpu lacks logical intra-row page release"
                )
            previous_intrarow = swap_intrarow(intrarow_min_bytes)
            intrarow_configured = True
        if filter_tail_simd:
            if not callable(swap_filter_tail):
                raise GPUConfigurationError(
                    "installed plan7_gpu lacks request-scoped filter-tail SIMD"
                )
            previous_filter_tail = swap_filter_tail(True)
            filter_tail_configured = True
    except BaseException:
        try:
            if filter_tail_configured:
                swap_filter_tail(previous_filter_tail)
        finally:
            try:
                if intrarow_configured:
                    swap_intrarow(previous_intrarow)
            finally:
                try:
                    if avx_configured:
                        swap_avx(previous_avx)
                finally:
                    _gpu_request_page_release_lock.release()
        raise
    return (
        swap_avx,
        swap_intrarow,
        avx_configured,
        intrarow_configured,
        previous_avx,
        previous_intrarow,
        swap_filter_tail,
        filter_tail_configured,
        previous_filter_tail,
    )


def _restore_gpu_request_page_release(configuration):
    if configuration is None:
        return
    (
        swap_avx,
        swap_intrarow,
        avx_configured,
        intrarow_configured,
        previous_avx,
        previous_intrarow,
        swap_filter_tail,
        filter_tail_configured,
        previous_filter_tail,
    ) = configuration
    try:
        if filter_tail_configured:
            swap_filter_tail(previous_filter_tail)
    finally:
        try:
            if intrarow_configured:
                swap_intrarow(previous_intrarow)
        finally:
            try:
                if avx_configured:
                    swap_avx(previous_avx)
            finally:
                _gpu_request_page_release_lock.release()


class GPUOverlapMetrics:
    """Opt-in counters for the bounded profile-generation pipeline."""

    def __init__(self):
        self._lock = Lock()
        self.requested_thread_count = 0
        self.profile_worker_count = 0
        self.profile_build_worker_count = 0
        self.profile_selection_worker_count = 0
        self.producer_slot_count = 0
        self.continuation_worker_count = 0
        self.continuation_window = 1
        self.continuation_window_high_water = 0
        self.continuation_maximum_chunk_output_bytes = 0
        self.profile_overlap_enabled = False
        self.scheduler_mode = 'none'
        self.profile_host_bytes = 0
        self.profile_chunk_local_pack = False
        self.profile_streamed = False
        self.profile_stream_session_count = 0
        self.profile_stream_max_pairs = 0
        self.profile_pressed_bytes = 0
        self.profile_pointer_bytes = 0
        self.profile_identity_token_bytes = 0
        self.profile_background_bytes = 0
        self.chunk_count = 0
        self.generated_chunk_count = 0
        self.consumed_chunk_count = 0
        self.ready_without_wait_count = 0
        self.ready_queue_capacity = 0
        self.ready_queue_high_water = 0
        self.ready_queue_byte_capacity = 0
        self.ready_queue_byte_high_water = 0
        self.ready_queue_final_count = 0
        self.ready_queue_final_bytes = 0
        self.generated_candidate_bytes = 0
        self.consumed_candidate_bytes = 0
        self.maximum_candidate_bytes = 0
        self.producer_lookahead_capacity = 0
        self.producer_idle_count = 0
        self.producer_idle_seconds = 0.0
        self.producer_lookahead_high_water = 0
        self.producer_lookahead_start_count = 0
        self.live_candidate_capacity = 0
        self.profile_load_seconds = 0.0
        self.session_build_seconds = 0.0
        self.profile_cache_enabled = False
        self.profile_cache_hit = False
        self.profile_cache_validation_seconds = 0.0
        self.profile_session_id = 0
        self.profile_session_selection_count_start = 0
        self.target_batch_seconds = 0.0
        self.preflight_seconds = 0.0
        self.generation_seconds = 0.0
        self.selection_seconds = 0.0
        self.generation_wait_seconds = 0.0
        self.initial_generation_wait_seconds = 0.0
        self.pipeline_stall_seconds = 0.0
        self.continuation_seconds = 0.0
        self.continuation_call_seconds = 0.0
        self.continuation_pool_enabled = False
        self.continuation_pool_call_count = 0
        self.continuation_pipeline_count = 0
        self.tsv_worker_rendered_profile_count = 0
        self.tsv_worker_rendered_row_count = 0
        self.tsv_worker_rendered_bytes = 0
        self.tsv_consumer_fallback_profile_count = 0
        self.overlap_seconds = 0.0
        self.pipeline_wall_seconds = 0.0
        self.generation_records = []
        self.continuation_records = []

    def snapshot(self):
        return {
            'requested_thread_count': self.requested_thread_count,
            'profile_worker_count': self.profile_worker_count,
            'profile_build_worker_count': self.profile_build_worker_count,
            'profile_selection_worker_count': (
                self.profile_selection_worker_count
            ),
            'producer_slot_count': self.producer_slot_count,
            'continuation_worker_count': self.continuation_worker_count,
            'continuation_window': self.continuation_window,
            'continuation_window_high_water': (
                self.continuation_window_high_water
            ),
            'continuation_maximum_chunk_output_bytes': (
                self.continuation_maximum_chunk_output_bytes
            ),
            'profile_overlap_enabled': self.profile_overlap_enabled,
            'scheduler_mode': self.scheduler_mode,
            'profile_host_bytes': self.profile_host_bytes,
            'profile_chunk_local_pack': self.profile_chunk_local_pack,
            'profile_streamed': self.profile_streamed,
            'profile_stream_session_count': self.profile_stream_session_count,
            'profile_stream_max_pairs': self.profile_stream_max_pairs,
            'profile_pressed_bytes': self.profile_pressed_bytes,
            'profile_pointer_bytes': self.profile_pointer_bytes,
            'profile_identity_token_bytes': self.profile_identity_token_bytes,
            'profile_background_bytes': self.profile_background_bytes,
            'chunk_count': self.chunk_count,
            'generated_chunk_count': self.generated_chunk_count,
            'consumed_chunk_count': self.consumed_chunk_count,
            'ready_without_wait_count': self.ready_without_wait_count,
            'ready_queue_capacity': self.ready_queue_capacity,
            'ready_queue_high_water': self.ready_queue_high_water,
            'ready_queue_byte_capacity': self.ready_queue_byte_capacity,
            'ready_queue_byte_high_water': self.ready_queue_byte_high_water,
            'ready_queue_final_count': self.ready_queue_final_count,
            'ready_queue_final_bytes': self.ready_queue_final_bytes,
            'generated_candidate_bytes': self.generated_candidate_bytes,
            'consumed_candidate_bytes': self.consumed_candidate_bytes,
            'maximum_candidate_bytes': self.maximum_candidate_bytes,
            'producer_lookahead_capacity': self.producer_lookahead_capacity,
            'producer_idle_count': self.producer_idle_count,
            'producer_idle_seconds': self.producer_idle_seconds,
            'producer_lookahead_high_water': (
                self.producer_lookahead_high_water
            ),
            'producer_lookahead_start_count': (
                self.producer_lookahead_start_count
            ),
            'live_candidate_capacity': self.live_candidate_capacity,
            'profile_load_seconds': self.profile_load_seconds,
            'session_build_seconds': self.session_build_seconds,
            'profile_cache_enabled': self.profile_cache_enabled,
            'profile_cache_hit': self.profile_cache_hit,
            'profile_cache_validation_seconds': (
                self.profile_cache_validation_seconds
            ),
            'profile_session_id': self.profile_session_id,
            'profile_session_selection_count_start': (
                self.profile_session_selection_count_start
            ),
            'target_batch_seconds': self.target_batch_seconds,
            'preflight_seconds': self.preflight_seconds,
            'generation_seconds': self.generation_seconds,
            'selection_seconds': self.selection_seconds,
            'generation_wait_seconds': self.generation_wait_seconds,
            'initial_generation_wait_seconds': self.initial_generation_wait_seconds,
            'pipeline_stall_seconds': self.pipeline_stall_seconds,
            'continuation_seconds': self.continuation_seconds,
            'continuation_call_seconds': self.continuation_call_seconds,
            'continuation_pool_enabled': self.continuation_pool_enabled,
            'continuation_pool_call_count': self.continuation_pool_call_count,
            'continuation_pipeline_count': self.continuation_pipeline_count,
            'tsv_worker_rendered_profile_count': (
                self.tsv_worker_rendered_profile_count
            ),
            'tsv_worker_rendered_row_count': (
                self.tsv_worker_rendered_row_count
            ),
            'tsv_worker_rendered_bytes': self.tsv_worker_rendered_bytes,
            'tsv_consumer_fallback_profile_count': (
                self.tsv_consumer_fallback_profile_count
            ),
            'overlap_seconds': self.overlap_seconds,
            'pipeline_wall_seconds': self.pipeline_wall_seconds,
            'generation_records': [
                dict(record) for record in self.generation_records
            ],
            'continuation_records': [
                dict(record) for record in self.continuation_records
            ],
        }


def gpu_profile_cell_cap():
    """Return the private exact profile-by-target chunk ceiling."""
    value = os.environ.get(GPU_PROFILE_CELL_CAP_ENV)
    if value is None:
        return GPU_CELL_CAP
    allowed = tuple(str(cap) for cap in GPU_PROFILE_CELL_CAPS)
    if value not in allowed:
        raise GPUConfigurationError(
            f"{GPU_PROFILE_CELL_CAP_ENV} accepts only {', '.join(allowed)}"
        )
    return int(value)


def _pressed_profile_payload_bytes(pressed_base):
    return sum(
        Path(f"{pressed_base}.{suffix}").stat().st_size
        for suffix in PRESSED_SUFFIXES
    )


def gpu_production_request_tuning(
    pressed_base,
    profile_count,
    options,
    threads,
    target_count,
    *,
    installed_attested=False,
    cache_enabled=False,
):
    """Select only the exact large installed-GPU shape measured on KOFAM."""
    if not installed_attested:
        return _disabled_gpu_request_tuning("not-installed-attested")
    if cache_enabled:
        return _disabled_gpu_request_tuning("persistent-cache")
    if any(name in os.environ for name in GPU_PRODUCTION_OVERRIDE_ENVS):
        return _disabled_gpu_request_tuning("explicit-override")
    if not _gpu_production_evalue_options(options):
        return _disabled_gpu_request_tuning("not-evalue-only")
    if type(threads) is not int or threads != GPU_PRODUCTION_THREAD_COUNT:
        return _disabled_gpu_request_tuning("thread-count")
    if type(target_count) is not int or target_count <= GPU_PRODUCTION_TARGET_MINIMUM:
        return _disabled_gpu_request_tuning("target-count")
    if type(profile_count) is not int or profile_count <= 0:
        return _disabled_gpu_request_tuning("profile-count")
    pressed_bytes = _pressed_profile_payload_bytes(pressed_base)
    if pressed_bytes < GPU_CHUNK_LOCAL_PROFILE_PACK_MIN_PRESSED_BYTES:
        return _disabled_gpu_request_tuning(
            "pressed-payload", pressed_bytes
        )
    return _GPURequestTuning(True, pressed_bytes, "eligible-large-gpu")


def gpu_chunk_local_profile_pack_configuration(
    pressed_base, profile_count, *, cache_enabled=False, automatic=False,
):
    """Return the private chunk-local-pack decision and pressed byte count.

    The experiment is both explicitly requested and shape-gated.  Consequently
    the normal path performs no extra filesystem work, while an opt-in still
    leaves PFAM and small installed databases on the established eager pack.
    """
    if type(automatic) is not bool:
        raise TypeError("automatic chunk-local policy must be bool")
    value = os.environ.get(GPU_CHUNK_LOCAL_PROFILE_PACK_ENV)
    if value is None:
        if not automatic:
            return False, 0
    elif value == '0':
        return False, 0
    elif value != '1':
        raise GPUConfigurationError(
            f"{GPU_CHUNK_LOCAL_PROFILE_PACK_ENV} must be exactly '0' or '1'"
        )
    if cache_enabled:
        raise GPUConfigurationError(
            f"{GPU_CHUNK_LOCAL_PROFILE_PACK_ENV}=1 is not compatible with "
            "persistent GPU profile caching"
        )
    if isinstance(profile_count, bool) or not isinstance(profile_count, int):
        raise TypeError("profile_count must be an integer")
    if profile_count < 0:
        raise ValueError("profile_count must be nonnegative")
    pressed_bytes = _pressed_profile_payload_bytes(pressed_base)
    enabled = (
        profile_count > 0
        and pressed_bytes >= GPU_CHUNK_LOCAL_PROFILE_PACK_MIN_PRESSED_BYTES
    )
    return enabled, pressed_bytes


def _gpu_evalue_only_options(options):
    """Recognize the one fixed-threshold shape audited for GPU streaming."""
    if type(options) is not dict:
        return False
    try:
        kwargs = define_kwargs(options)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return set(kwargs) == {'E'}


def _gpu_production_evalue_options(options):
    """Recognize the exact finite E=1e-15 production threshold."""
    if not _gpu_evalue_only_options(options):
        return False
    return define_kwargs(options)['E'] == GPU_PRODUCTION_EVALUE


def gpu_pressed_profile_stream_configuration(
    pressed_base,
    profile_count,
    options,
    threads,
    *,
    cache_enabled=False,
    automatic=False,
):
    """Select the private large, E-only, overlapping GPU stream shape.

    The ordinary eager session remains authoritative unless every condition is
    present.  In particular, small/PFAM databases and benchmark controls keep
    their established profile lifetime and scheduler.
    """
    if not _gpu_evalue_only_options(options):
        return False, 0
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads < 2
        or os.environ.get(GPU_SERIAL_ENV) == '1'
        or GPU_LEGACY_OVERLAP_ENV in os.environ
    ):
        return False, 0
    return gpu_chunk_local_profile_pack_configuration(
        pressed_base,
        profile_count,
        cache_enabled=cache_enabled,
        automatic=automatic,
    )


class _PressedGPUProfileStream:
    """A lazy attested pressed database consumed by one GPU request."""

    def __init__(
        self,
        pressed_base,
        manifest_path,
        profile_count,
        pressed_bytes,
        build_workers,
        iterator_factory,
        session_factory,
    ):
        if isinstance(profile_count, bool) or not isinstance(profile_count, int):
            raise TypeError("streamed profile count must be an integer")
        if profile_count <= 0:
            raise ValueError("streamed profile count must be positive")
        if not callable(iterator_factory):
            raise TypeError("pressed profile chunk iterator is unavailable")
        if not callable(session_factory):
            raise TypeError("profile session factory is unavailable")
        self.pressed_base = Path(pressed_base).resolve(strict=False)
        self.manifest_path = manifest_path
        self.profile_count = profile_count
        self.pressed_bytes = pressed_bytes
        self.build_workers = build_workers
        self._iterator_factory = iterator_factory
        self._session_factory = session_factory
        self._lock = Lock()
        self._closed = False
        self._session_count = 0
        self._selection_count = 0
        self._maximum_session_statistics = {
            'host_bytes': 0,
            'profile_pointer_bytes': 0,
            'identity_token_bytes': 0,
            'background_bytes': 0,
            'worker_count': 0,
            'build_worker_count': 0,
            'selection_worker_count': 0,
        }

    def __len__(self):
        return self.profile_count

    @property
    def closed(self):
        with self._lock:
            return self._closed

    @property
    def statistics(self):
        with self._lock:
            result = dict(self._maximum_session_statistics)
            result.update({
                'session_id': id(self),
                'selection_count': self._selection_count,
                'chunk_local_pack': True,
                'streamed': True,
                'stream_session_count': self._session_count,
            })
            return result

    def chunk_specs(self, chunk_size, pipeline_options):
        with self._lock:
            if self._closed:
                raise RuntimeError("pressed GPU profile stream is closed")
        return _PressedGPUProfileChunks(self, chunk_size, pipeline_options)

    def open_chunk_session(self, pairs):
        with self._lock:
            if self._closed:
                raise RuntimeError("pressed GPU profile stream is closed")
        session = self._session_factory(
            pairs,
            build_workers=self.build_workers,
            selection_workers=0,
            _chunk_local_pack=True,
        )
        statistics = session.statistics
        with self._lock:
            if self._closed:
                session.close()
                raise RuntimeError("pressed GPU profile stream closed during build")
            self._session_count += 1
            for name in self._maximum_session_statistics:
                value = statistics.get(name, 0)
                if type(value) is int:
                    self._maximum_session_statistics[name] = max(
                        self._maximum_session_statistics[name], value
                    )
        return session

    def record_selection(self):
        with self._lock:
            self._selection_count += 1

    def close(self):
        with self._lock:
            self._closed = True


class _PressedGPUProfileChunks:
    """One-pass canonical chunk specs backed by a pinned lockstep stream."""

    def __init__(self, owner, chunk_size, pipeline_options):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("GPU profile chunk size must be an integer")
        if chunk_size <= 0:
            raise ValueError("GPU profile chunk size must be positive")
        if type(pipeline_options) is not dict:
            raise TypeError("GPU profile pipeline options must be a dict")
        self.owner = owner
        self.chunk_size = chunk_size
        self.pipeline_options = pipeline_options
        self.pipeline_option_groups = (pipeline_options,)
        self._started = False
        self._closed = False
        self._raw_iterator = None

    def __len__(self):
        return (
            self.owner.profile_count + self.chunk_size - 1
        ) // self.chunk_size

    def __iter__(self):
        if self._started:
            raise RuntimeError("pressed GPU profile chunks are one-pass")
        if self._closed:
            raise RuntimeError("pressed GPU profile chunks are closed")
        self._started = True
        raw_iterator = self.owner._iterator_factory(
            self.owner.pressed_base,
            self.chunk_size,
            manifest=self.owner.manifest_path,
        )
        self._raw_iterator = raw_iterator
        expected_ordinal = 0
        chunk_index = 0
        try:
            for pairs in raw_iterator:
                pairs = tuple(pairs)
                ordinals = tuple(pair.ordinal for pair in pairs)
                expected = tuple(
                    range(expected_ordinal, expected_ordinal + len(pairs))
                )
                if not pairs or ordinals != expected:
                    raise RuntimeError(
                        "pressed GPU profile stream changed canonical order"
                    )
                chunk_index += 1
                if chunk_index > len(self):
                    raise RuntimeError(
                        "pressed GPU profile stream exceeds manifest count"
                    )
                yield (
                    chunk_index,
                    pairs,
                    ordinals,
                    self.pipeline_options,
                )
                expected_ordinal += len(pairs)
            if expected_ordinal != self.owner.profile_count:
                raise RuntimeError(
                    "pressed GPU profile stream count differs from manifest"
                )
            if chunk_index != len(self):
                raise RuntimeError("pressed GPU profile stream chunk count changed")
        finally:
            close = getattr(raw_iterator, 'close', None)
            if close is not None:
                close()
            self._raw_iterator = None
            self._closed = True

    def close(self):
        raw_iterator = self._raw_iterator
        if raw_iterator is not None:
            close = getattr(raw_iterator, 'close', None)
            if close is not None:
                close()
            self._raw_iterator = None
        self._closed = True


def _gpu_pipeline_option_groups(chunks):
    if type(chunks) is _PressedGPUProfileChunks:
        return chunks.pipeline_option_groups
    groups = []
    seen = set()
    for spec in chunks:
        key = id(spec[3])
        if key not in seen:
            seen.add(key)
            groups.append(spec[3])
    return tuple(groups)


def _merged_time_intervals(records, started_key, finished_key):
    intervals = sorted(
        (record[started_key], record[finished_key])
        for record in records
    )
    merged = []
    for started, finished in intervals:
        if finished < started:
            raise RuntimeError("timing interval finished before it started")
        if merged and started <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], finished))
        else:
            merged.append((started, finished))
    return merged


def _interval_intersection_seconds(left, right):
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_started, left_finished = left[left_index]
        right_started, right_finished = right[right_index]
        total += max(
            0.0,
            min(left_finished, right_finished)
            - max(left_started, right_started),
        )
        if left_finished <= right_finished:
            left_index += 1
        else:
            right_index += 1
    return total


def gpu_hmm_chunk_size(sequence_count):
    """Bound one GPU candidate matrix while retaining Aksha's 2,000-HMM cap."""
    if sequence_count > GPU_CELL_CAP:
        raise GPUConfigurationError(
            f"explicit GPU search supports at most {GPU_CELL_CAP:,} targets; "
            f"received {sequence_count:,}"
        )
    profile_cell_cap = gpu_profile_cell_cap()
    return min(
        HMM_CHUNK_SIZE,
        max(1, profile_cell_cap // max(1, sequence_count)),
    )


def gpu_profile_worker_allocation(threads, overlap_requested):
    """Return overlap and compute-worker slots under Aksha's CLI convention.

    ``--threads`` counts compute/search workers, as in Aksha's CPU path; the
    common main writer is excluded. Serial control reserves the same producer
    slot as overlap so the continuation width remains identical.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise GPUConfigurationError(
            "GPU profile sessions require a positive integer compute-worker budget"
        )
    if type(overlap_requested) is not bool:
        raise TypeError("overlap_requested must be bool")
    producer_slots = int(threads >= 2)
    overlap_enabled = overlap_requested and bool(producer_slots)
    return overlap_enabled, producer_slots, threads - producer_slots


def gpu_profile_scheduler_mode(profile_session, overlap_enabled):
    """Select the production queue or the exact benchmark control."""
    value = os.environ.get(GPU_LEGACY_OVERLAP_ENV)
    if value is None:
        return 'bounded-ready-queue' if overlap_enabled else 'serial'
    if value != GPU_LEGACY_OVERLAP_VALUE:
        raise GPUConfigurationError(
            f"{GPU_LEGACY_OVERLAP_ENV} accepts only "
            f"{GPU_LEGACY_OVERLAP_VALUE!r}"
        )
    if profile_session is None or not overlap_enabled:
        raise GPUConfigurationError(
            f"{GPU_LEGACY_OVERLAP_ENV} requires an active overlapping "
            "GPU profile session"
        )
    return 'single-prefetch'


def gpu_ready_queue_configuration():
    """Return the exact experimental ready-queue depth and byte ceiling.

    With neither variable set this deliberately returns the audited one-slot,
    item-bounded configuration.  Deeper queues require an explicit byte limit
    so enabling lookahead can never silently turn into an unbounded host-memory
    experiment.
    """
    depth_text = os.environ.get(GPU_READY_QUEUE_DEPTH_ENV)
    byte_text = os.environ.get(GPU_READY_QUEUE_BYTES_ENV)

    if depth_text is None:
        depth = GPU_READY_QUEUE_CAPACITY
    elif depth_text not in tuple(str(value) for value in GPU_READY_QUEUE_DEPTHS):
        allowed = ', '.join(str(value) for value in GPU_READY_QUEUE_DEPTHS)
        raise GPUConfigurationError(
            f"{GPU_READY_QUEUE_DEPTH_ENV} accepts only {allowed}"
        )
    else:
        depth = int(depth_text)

    if byte_text is None:
        byte_capacity = None
    else:
        if (
            not byte_text
            or not byte_text.isascii()
            or not byte_text.isdigit()
            or byte_text[0] == '0'
        ):
            raise GPUConfigurationError(
                f"{GPU_READY_QUEUE_BYTES_ENV} must be a canonical positive "
                "decimal byte count"
            )
        try:
            byte_capacity = int(byte_text)
        except ValueError as error:
            raise GPUConfigurationError(
                f"{GPU_READY_QUEUE_BYTES_ENV} is outside the supported "
                "integer range"
            ) from error
        if byte_capacity > GPU_READY_QUEUE_MAX_BYTES:
            raise GPUConfigurationError(
                f"{GPU_READY_QUEUE_BYTES_ENV} must be at most "
                f"{GPU_READY_QUEUE_MAX_BYTES}"
            )

    if depth != GPU_READY_QUEUE_CAPACITY and byte_capacity is None:
        raise GPUConfigurationError(
            f"{GPU_READY_QUEUE_DEPTH_ENV}={depth} requires an explicit "
            f"{GPU_READY_QUEUE_BYTES_ENV}"
        )
    return depth, byte_capacity


def _gpu_candidate_resident_bytes(candidates):
    """Read the adapter's exact incremental CandidateBatch byte charge."""
    try:
        resident_bytes = candidates.resident_bytes
    except AttributeError as error:
        raise GPUConfigurationError(
            f"{GPU_READY_QUEUE_BYTES_ENV} requires a plan7_gpu CandidateBatch "
            "with exact resident_bytes support"
        ) from error
    if type(resident_bytes) is not int or resident_bytes < 0:
        raise RuntimeError(
            "CandidateBatch.resident_bytes must be an exact nonnegative integer"
        )
    return resident_bytes


class _GPUByteBoundedReadyQueue:
    """A strict FIFO bounded by queued-item count and exact queued bytes.

    The ceiling deliberately excludes the batch currently owned by the
    consumer and the batch being generated by the producer.  Those two live
    objects are reported by the separate live/lookahead capacity metrics.
    """

    def __init__(self, max_items, max_bytes):
        if type(max_items) is not int or max_items <= 0:
            raise ValueError("max_items must be a positive integer")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._items = deque()
        self._resident_bytes = 0
        self._item_high_water = 0
        self._byte_high_water = 0
        self._condition = Condition()

    def _validate_weight(self, weight):
        if type(weight) is not int or weight < 0:
            raise ValueError("queue item bytes must be a nonnegative integer")
        if weight > self._max_bytes:
            raise ValueError(
                f"queue item requires {weight} bytes, exceeding the "
                f"{self._max_bytes}-byte ready-queue ceiling"
            )

    def _can_put(self, weight):
        return (
            len(self._items) < self._max_items
            and self._resident_bytes + weight <= self._max_bytes
        )

    def _append(self, item, weight):
        self._items.append((item, weight))
        self._resident_bytes += weight
        self._item_high_water = max(self._item_high_water, len(self._items))
        self._byte_high_water = max(
            self._byte_high_water, self._resident_bytes
        )
        self._condition.notify_all()

    def put_nowait(self, item, weight):
        self._validate_weight(weight)
        with self._condition:
            if not self._can_put(weight):
                raise Full
            self._append(item, weight)

    def put(self, item, weight, timeout, stop=None):
        self._validate_weight(weight)
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._can_put(weight):
                if stop is not None and stop.is_set():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise Full
                self._condition.wait(remaining)
            if stop is not None and stop.is_set():
                return False
            self._append(item, weight)
            return True

    def _popleft(self):
        item, weight = self._items.popleft()
        self._resident_bytes -= weight
        self._condition.notify_all()
        return item

    def get_nowait(self):
        with self._condition:
            if not self._items:
                raise Empty
            return self._popleft()

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise Empty
                self._condition.wait(remaining)
            return self._popleft()

    def state(self):
        with self._condition:
            return (
                len(self._items),
                self._resident_bytes,
                self._item_high_water,
                self._byte_high_water,
            )

    def high_water(self):
        return self.state()[2:]

    def wake_all(self):
        with self._condition:
            self._condition.notify_all()


def parse_gpu_manifest_mappings(entries):
    """Parse repeatable ``DB=PATH`` values without accepting ambiguity."""
    mappings = {}
    for entry in entries or ():
        if not isinstance(entry, str):
            raise GPUConfigurationError("--gpu-manifest values must be DB=PATH strings")
        db_name, separator, manifest_path = entry.partition('=')
        db_name = db_name.strip()
        manifest_path = manifest_path.strip()
        if not separator or not db_name or not manifest_path:
            raise GPUConfigurationError(
                f"malformed --gpu-manifest value {entry!r}; expected DB=PATH"
            )
        if db_name in mappings:
            raise GPUConfigurationError(
                f"duplicate --gpu-manifest mapping for database {db_name!r}"
            )
        mappings[db_name] = os.path.expandvars(os.path.expanduser(manifest_path))
    return mappings


def discover_pressed_base(installation_dir):
    """Return the unambiguous complete pressed base in an installation directory.

    Aksha's own press operation normally names the base after the directory.
    Older installations such as HydDB instead keep a source-derived base name,
    so they are accepted when exactly one complete pressed set is present.
    """
    try:
        raw_directory = os.fspath(installation_dir)
    except TypeError as error:
        raise GPUConfigurationError("installed database has no installation directory") from error
    if not raw_directory:
        raise GPUConfigurationError("installed database has no installation directory")

    db_dir = Path(os.path.expandvars(os.path.expanduser(raw_directory)))
    if not db_dir.is_dir():
        raise GPUConfigurationError(
            f"installed database directory does not exist: {db_dir}"
        )
    db_dir = db_dir.resolve()

    def complete(base):
        return all(Path(f"{base}.{suffix}").is_file() for suffix in PRESSED_SUFFIXES)

    candidates = set()
    for member in db_dir.iterdir():
        if member.suffix in {f'.{suffix}' for suffix in PRESSED_SUFFIXES}:
            base = member.with_suffix('')
            if complete(base):
                candidates.add(base)

    if not candidates:
        raise GPUConfigurationError(
            f"no complete pressed HMM set found in {db_dir}; expected .h3m/.h3i/.h3f/.h3p"
        )
    if len(candidates) != 1:
        names = ', '.join(sorted(base.name for base in candidates))
        raise GPUConfigurationError(
            f"ambiguous pressed HMM sets in {db_dir}: {names}"
        )
    return candidates.pop()


def _resolve_installed_hmm_names(installed_hmms, parsed_json):
    names = installed_hmms.split(',') if ',' in installed_hmms else [installed_hmms]
    if 'all_prot' in names:
        names = [
            db['name'] for db in parsed_json['db_urls']
            if db['molecule_type'] == 'protein' and db['installed']
        ]
    return names


def validate_gpu_configuration(mappings, installed_hmm_names, parsed_json,
                               threads, macsyfinder_enabled):
    """Reject explicit GPU mappings that cannot be consumed exactly once."""
    if not mappings:
        return
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise GPUConfigurationError(
            "--threads must be a positive integer when --gpu-manifest is used"
        )
    if macsyfinder_enabled:
        raise GPUConfigurationError(
            "--write_macsyfinder cannot be combined with --gpu-manifest"
        )

    requested_names = list(installed_hmm_names or ())
    seen = set()
    duplicates = set()
    for name in requested_names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise GPUConfigurationError(
            "duplicate installed HMM database(s) with GPU mappings: "
            + ', '.join(sorted(duplicates))
        )

    requested = set(requested_names)
    eligible = {
        db['name'] for db in parsed_json['db_urls']
        if (db.get('name') in requested
            and db.get('installed')
            and db.get('molecule_type') == 'protein'
            and db.get('installation_dir'))
    }
    unused = sorted(set(mappings) - eligible)
    if unused:
        raise GPUConfigurationError(
            "unused --gpu-manifest mapping(s): " + ', '.join(unused)
        )


def preflight_gpu_databases(mappings, installed_hmm_names, parsed_json,
                            all_sequences, threads, gpu_metrics=None,
                            profile_session_cache=None, search_options=None,
                            request_tuning_by_db=None,
                            filter_tail_simd_by_db=None):
    """Load every attested mapped database and initialize one target batch."""
    if not mappings:
        return {}, None, False

    preflight_started = time.perf_counter()
    from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
    try:
        from plan7_gpu.adapter import _iter_pressed_profile_chunks
    except (ImportError, AttributeError):
        _iter_pressed_profile_chunks = None
    from plan7_gpu.pressed_manifest import validate_pressed_manifest

    if profile_session_cache is not None:
        from aksha.gpu_profile_cache import GPUProfileSessionCache

        if not isinstance(profile_session_cache, GPUProfileSessionCache):
            raise TypeError(
                "profile_session_cache must be GPUProfileSessionCache or None"
            )
    if request_tuning_by_db is not None and type(request_tuning_by_db) is not dict:
        raise TypeError("request_tuning_by_db must be exactly dict or None")
    if (
        filter_tail_simd_by_db is not None
        and type(filter_tail_simd_by_db) is not dict
    ):
        raise TypeError("filter_tail_simd_by_db must be exactly dict or None")

    database_specs = {}
    for db_name in installed_hmm_names:
        manifest_path = mappings.get(db_name)
        if manifest_path is None:
            continue
        database = next(
            item for item in parsed_json['db_urls'] if item['name'] == db_name
        )
        pressed_base = discover_pressed_base(database['installation_dir'])
        validation = validate_pressed_manifest(pressed_base, manifest_path)
        database_specs[db_name] = (
            pressed_base,
            manifest_path,
            validation,
        )

    if profile_session_cache is not None and len(database_specs) != 1:
        raise GPUConfigurationError(
            "persistent GPU profile caching currently requires exactly one "
            "mapped database"
        )

    selected_filter_tail_simd = False
    for db_name, (_, _, validation) in database_specs.items():
        filter_tail_simd = gpu_filter_tail_simd_request(
            search_options,
            threads,
            len(all_sequences),
            installed_attested=True,
            single_mapped_database=(len(database_specs) == 1),
            profile_count=getattr(validation, 'model_count', None),
            cache_enabled=(profile_session_cache is not None),
        )
        selected_filter_tail_simd = (
            selected_filter_tail_simd or filter_tail_simd
        )
        if filter_tail_simd_by_db is not None:
            filter_tail_simd_by_db[db_name] = filter_tail_simd

    databases = {}
    selected_request_tuning = None
    if profile_session_cache is None:
        for db_name, (
            pressed_base,
            manifest_path,
            validation,
        ) in database_specs.items():
            profile_count = getattr(validation, 'model_count', None)
            request_tuning = (
                gpu_production_request_tuning(
                    pressed_base,
                    profile_count,
                    search_options,
                    threads,
                    len(all_sequences),
                    installed_attested=True,
                )
                if len(database_specs) == 1
                else _disabled_gpu_request_tuning(
                    "multiple-mapped-databases"
                )
            )
            selected_request_tuning = request_tuning
            if request_tuning_by_db is not None:
                request_tuning_by_db[db_name] = request_tuning
            stream_enabled, pressed_bytes = (
                gpu_pressed_profile_stream_configuration(
                    pressed_base,
                    profile_count,
                    search_options,
                    threads,
                    automatic=request_tuning.automatic,
                )
            )
            if stream_enabled:
                if _iter_pressed_profile_chunks is None:
                    raise GPUConfigurationError(
                        "installed plan7_gpu lacks pressed-profile streaming"
                    )
                stream = _PressedGPUProfileStream(
                    pressed_base,
                    manifest_path,
                    profile_count,
                    pressed_bytes,
                    threads,
                    _iter_pressed_profile_chunks,
                    ProfileSession,
                )
                databases[db_name] = (pressed_base, stream, stream)
                if gpu_metrics is not None:
                    metrics = gpu_metrics[db_name]
                    metrics.profile_streamed = True
                    metrics.profile_chunk_local_pack = True
                    metrics.profile_pressed_bytes = pressed_bytes
                continue
            load_started = time.perf_counter()
            pairs = load_pressed_profiles(pressed_base, manifest=manifest_path)
            if gpu_metrics is not None:
                gpu_metrics[db_name].profile_load_seconds += (
                    time.perf_counter() - load_started
                )
            databases[db_name] = (pressed_base, pairs, None)
    else:
        for db_name, (pressed_base, _, _) in database_specs.items():
            if request_tuning_by_db is not None:
                request_tuning_by_db[db_name] = _disabled_gpu_request_tuning(
                    "persistent-cache"
                )
            databases[db_name] = (pressed_base, (), None)

    postfilter = gpu_postfilter_available()
    batch = None
    cache_reservation = None
    try:
        if profile_session_cache is not None:
            # Claim the one-entry cache before allocating a target-dependent
            # CUDA batch. Concurrent long-lived requests therefore fail while
            # still host-only instead of transiently doubling device memory.
            cache_reservation = profile_session_cache.reserve()
        batch_started = time.perf_counter()
        batch_arguments = {
            "alphabet": astra_pyhmmer.easel.Alphabet.amino(),
        }
        if (
            (
                selected_request_tuning is not None
                and selected_request_tuning.automatic
            )
            or selected_filter_tail_simd
        ):
            batch_arguments["_forward_cpu_max_cells"] = (
                GPU_PRODUCTION_FORWARD_CPU_MAX_CELLS
            )
        batch = SequenceBatch(all_sequences, **batch_arguments)
        batch_seconds = time.perf_counter() - batch_started
        if gpu_metrics is not None:
            for metrics in gpu_metrics.values():
                metrics.target_batch_seconds = batch_seconds
        # A fresh producer thread begins on CUDA ordinal 0. Until plan7_gpu
        # exposes a scoped device bind, keep nonzero batches on the existing
        # same-thread path instead of creating an unusable ProfileSession.
        # Build workers retire before ProfileSession returns. Keeping the
        # persistent selection pool at zero makes eager multi-database
        # sessions threadless and selections synchronous before continuation.
        session_supported = (
            postfilter
            and gpu_profile_forward_available()
            and batch.memory_snapshot['device_ordinal'] == 0
            and callable(
                getattr(batch, '_postfilter_forward_selection', None)
            )
        )
        if session_supported:
            for db_name, (pressed_base, pairs, _) in tuple(databases.items()):
                if profile_session_cache is None:
                    if type(pairs) is _PressedGPUProfileStream:
                        session = pairs
                        session_build_seconds = 0.0
                        pressed_bytes = pairs.pressed_bytes
                        databases[db_name] = (
                            pressed_base,
                            pairs,
                            session,
                        )
                        if gpu_metrics is not None:
                            metrics = gpu_metrics[db_name]
                            session_statistics = session.statistics
                            metrics.profile_streamed = True
                            metrics.profile_chunk_local_pack = True
                            metrics.profile_pressed_bytes = pressed_bytes
                            metrics.profile_session_id = session_statistics[
                                'session_id'
                            ]
                            metrics.profile_session_selection_count_start = (
                                session_statistics['selection_count']
                            )
                        continue
                    if not pairs:
                        continue
                    chunk_local_pack, pressed_bytes = (
                        gpu_chunk_local_profile_pack_configuration(
                            pressed_base,
                            len(pairs),
                        )
                    )
                    session_started = time.perf_counter()
                    session_kwargs = {
                        'build_workers': threads,
                        'selection_workers': 0,
                    }
                    if chunk_local_pack:
                        session_kwargs['_chunk_local_pack'] = True
                    session = ProfileSession(pairs, **session_kwargs)
                    session_build_seconds = time.perf_counter() - session_started
                else:
                    _, pressed_bytes = (
                        gpu_chunk_local_profile_pack_configuration(
                            pressed_base,
                            0,
                            cache_enabled=True,
                        )
                    )
                    manifest_path = database_specs[db_name][1]
                    session = profile_session_cache.acquire(
                        pressed_base,
                        manifest_path,
                        device_key=(
                            'cuda-ordinal',
                            batch.memory_snapshot['device_ordinal'],
                        ),
                        build_workers=threads,
                        selection_workers=0,
                        reservation=cache_reservation,
                    )
                    cache_reservation = None
                    pairs = session.profile_pairs
                    session_build_seconds = session.session_build_seconds
                databases[db_name] = (
                    pressed_base,
                    pairs,
                    session,
                )
                if gpu_metrics is not None:
                    metrics = gpu_metrics[db_name]
                    metrics.session_build_seconds += session_build_seconds
                    session_statistics = session.statistics
                    metrics.profile_host_bytes = session_statistics['host_bytes']
                    metrics.profile_chunk_local_pack = session_statistics.get(
                        'chunk_local_pack', False
                    )
                    metrics.profile_pressed_bytes = pressed_bytes
                    metrics.profile_pointer_bytes = session_statistics.get(
                        'profile_pointer_bytes', 0
                    )
                    metrics.profile_identity_token_bytes = (
                        session_statistics.get('identity_token_bytes', 0)
                    )
                    metrics.profile_background_bytes = session_statistics.get(
                        'background_bytes', 0
                    )
                    metrics.profile_session_id = session_statistics['session_id']
                    metrics.profile_session_selection_count_start = (
                        session_statistics['selection_count']
                    )
                    if profile_session_cache is not None:
                        metrics.profile_cache_enabled = True
                        metrics.profile_cache_hit = session.reused
                        metrics.profile_cache_validation_seconds += (
                            session.validation_seconds
                        )
                        metrics.profile_load_seconds += (
                            session.profile_load_seconds
                        )
        else:
            if profile_session_cache is not None:
                cache_reservation.close()
                cache_reservation = None
            for db_name, (pressed_base, pairs, _) in tuple(databases.items()):
                if (
                    profile_session_cache is None
                    and type(pairs) is not _PressedGPUProfileStream
                ):
                    continue
                manifest_path = database_specs[db_name][1]
                load_started = time.perf_counter()
                eager_pairs = load_pressed_profiles(
                    pressed_base, manifest=manifest_path
                )
                if type(pairs) is _PressedGPUProfileStream:
                    pairs.close()
                databases[db_name] = (pressed_base, eager_pairs, None)
                if gpu_metrics is not None:
                    metrics = gpu_metrics[db_name]
                    metrics.profile_load_seconds += (
                        time.perf_counter() - load_started
                    )
                    metrics.profile_streamed = False
        if gpu_metrics is not None:
            preflight_seconds = time.perf_counter() - preflight_started
            for metrics in gpu_metrics.values():
                metrics.preflight_seconds = preflight_seconds
        return databases, batch, postfilter
    except BaseException:
        batch_close_failed = False
        if batch is not None:
            try:
                batch.close()
            except BaseException:
                batch_close_failed = True
        if batch_close_failed and profile_session_cache is not None:
            # A target batch whose close failed may still own device memory.
            # Retire the cache before returning its lease so no next request
            # can allocate another target batch in this process.
            try:
                profile_session_cache.close()
            except BaseException:
                pass
        for _, _, session in databases.values():
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    pass
        if cache_reservation is not None:
            try:
                cache_reservation.close()
            except BaseException:
                pass
        raise


def gpu_postfilter_available():
    """Return whether plan7_gpu can resume HMMER after exact GPU filters."""
    from plan7_gpu import _pipeline

    probe = getattr(_pipeline, '_filter_scores_seam_available', None)
    return callable(probe) and probe() is True


def gpu_profile_forward_available():
    """Return whether selection generation can include exact CUDA Forward."""
    from plan7_gpu import SequenceBatch, _pipeline

    probe = getattr(
        _pipeline, '_filter_and_forward_scores_seam_available', None
    )
    return (
        callable(probe)
        and probe() is True
        and hasattr(SequenceBatch, '_postfilter_forward_selection')
    )


def _signature_matches(method, names, *, keyword_only=(), defaulted=()):
    """Match the callable layout used by one sealed continuation ABI."""
    try:
        parameters = tuple(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return False
    if tuple(parameter.name for parameter in parameters) != tuple(names):
        return False
    keyword_only = frozenset(keyword_only)
    defaulted = frozenset(defaulted)
    for parameter in parameters:
        expected_kind = (
            inspect.Parameter.KEYWORD_ONLY
            if parameter.name in keyword_only
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if parameter.kind is not expected_kind:
            return False
        has_default = parameter.default is not inspect.Parameter.empty
        if has_default != (parameter.name in defaulted):
            return False
    return True


def _profile_continuation_capabilities():
    """Return the safe guarded-domain and compact-domain capabilities."""
    from plan7_gpu import SequenceBatch, _native, _pipeline

    filter_probe = getattr(_pipeline, '_filter_scores_seam_available', None)
    forward_probe = getattr(
        _pipeline, '_filter_and_forward_scores_seam_available', None
    )
    seam_probe = getattr(_pipeline, '_simple_regions_seam_available', None)
    seal = getattr(
        _pipeline, '_seal_profile_selection_continuation_bound', None
    )
    sparse_enabled = getattr(
        _pipeline, '_sealed_sparse_journal_v3_enabled_bound', None
    )
    sparse_search = getattr(
        _pipeline, '_search_hmm_sealed_sparse_journal_v3_bound', None
    )
    selection_method = getattr(
        SequenceBatch, '_postfilter_forward_selection', None
    )
    domain_method = getattr(
        SequenceBatch, '_postfilter_forward_domain_selection', None
    )
    native_method = getattr(
        _native.SequenceBatch,
        '_postfilter_forward_domain_selection_sealed',
        None,
    )
    if not (
        callable(selection_method)
        and callable(domain_method)
        and callable(native_method)
        and callable(seal)
        and callable(filter_probe)
        and filter_probe() is True
        and callable(forward_probe)
        and forward_probe() is True
        and callable(seam_probe)
        and seam_probe() is True
    ):
        return False, False, False

    selection_prefix = (
        'self', 'selection', 'F1', 'F2', 'F3', 'bias_filter',
        'pipeline', 'domain_guard',
    )
    domain_prefix = (
        'self', 'selection', 'F1', 'f2', 'f3', 'bias_filter',
        'pipeline', 'domain_guard',
    )
    native_prefix = (
        'self', 'selection', 'f1', 'f2', 'f3', 'guard_band',
        'gathered_byte_budget',
    )
    selection_options = ('pipeline', 'domain_guard')
    legacy_adapter = (
        _signature_matches(
            selection_method,
            selection_prefix,
            keyword_only=selection_options,
            defaulted=selection_options,
        )
        and _signature_matches(domain_method, domain_prefix)
    )
    compact_selection_suffix = (
        '_rescore_compact_byte_budget',
        '_rescore_matrix_byte_budget',
        '_rescore_trace_byte_budget',
        '_rescore_test_fault',
    )
    compact_domain_suffix = (
        'rescore_compact_byte_budget',
        'rescore_matrix_byte_budget',
        'rescore_trace_byte_budget',
        'rescore_test_fault',
    )
    compact_adapter = (
        _signature_matches(
            selection_method,
            selection_prefix + compact_selection_suffix,
            keyword_only=selection_options + compact_selection_suffix,
            defaulted=selection_options + compact_selection_suffix,
        )
        and _signature_matches(
            domain_method, domain_prefix + compact_domain_suffix
        )
    )
    telemetry_adapter = (
        _signature_matches(
            selection_method,
            selection_prefix + compact_selection_suffix + ('telemetry',),
            keyword_only=(
                selection_options
                + compact_selection_suffix
                + ('telemetry',)
            ),
            defaulted=(
                selection_options
                + compact_selection_suffix
                + ('telemetry',)
            ),
        )
        and _signature_matches(
            domain_method,
            domain_prefix + compact_domain_suffix + ('telemetry',),
        )
    )
    sparse_journal_v3_adapter_v1 = (
        _signature_matches(
            selection_method,
            selection_prefix
            + compact_selection_suffix
            + ('telemetry', 'sparse_journal_v3'),
            keyword_only=(
                selection_options
                + compact_selection_suffix
                + ('telemetry', 'sparse_journal_v3')
            ),
            defaulted=(
                selection_options
                + compact_selection_suffix
                + ('telemetry', 'sparse_journal_v3')
            ),
        )
        and _signature_matches(
            domain_method,
            domain_prefix
            + compact_domain_suffix
            + ('telemetry', 'sparse_journal_v3'),
        )
    )
    ga_pruning_adapter = (
        _signature_matches(
            selection_method,
            selection_prefix
            + compact_selection_suffix
            + ('telemetry', 'sparse_journal_v3', '_ga_pruning'),
            keyword_only=(
                selection_options
                + compact_selection_suffix
                + ('telemetry', 'sparse_journal_v3', '_ga_pruning')
            ),
            defaulted=(
                selection_options
                + compact_selection_suffix
                + ('telemetry', 'sparse_journal_v3', '_ga_pruning')
            ),
        )
        and _signature_matches(
            domain_method,
            domain_prefix
            + compact_domain_suffix
            + ('telemetry', 'sparse_journal_v3', 'ga_pruning'),
        )
    )
    sparse_journal_v3_adapter = (
        sparse_journal_v3_adapter_v1 or ga_pruning_adapter
    )
    sparse_journal_v3_pipeline = (
        _signature_matches(
            seal,
            (
                'queries', 'optimized_profiles', 'sequences',
                'residue_offsets', 'f1', 'background_fingerprint',
                'continuation_journal', 'selection_identity',
                'selection_identity_tokens', 'profile_fingerprints',
                'batch_generation', 'sequence_content_fingerprint',
                'pipeline', 'guard_band', 'native_stage_timings',
                'generation_statistics', 'sparse_journal_v3',
            ),
            defaulted=(
                'native_stage_timings', 'generation_statistics',
                'sparse_journal_v3',
            ),
        )
        and _signature_matches(sparse_enabled, ('sealed_object',))
        and _signature_matches(
            sparse_search,
            (
                'sealed_object', 'row', 'pipeline',
                '_return_route_statistics',
            ),
            defaulted=('_return_route_statistics',),
        )
    )
    legacy_native = _signature_matches(
        native_method,
        native_prefix,
        defaulted=('guard_band', 'gathered_byte_budget'),
    )
    compact_native_suffix = (
        'rescore_simple_diagnostic',
        'rescore_matrix_byte_budget',
        'rescore_trace_byte_budget',
        'rescore_compact_byte_budget',
        '_rescore_test_fault',
        'generation_tail_fingerprint',
    )
    compact_native = _signature_matches(
        native_method,
        native_prefix + compact_native_suffix,
        defaulted=(
            'guard_band',
            'gathered_byte_budget',
        ) + compact_native_suffix,
    )
    timing_native = _signature_matches(
        native_method,
        native_prefix + compact_native_suffix + ('_return_stage_timings',),
        defaulted=(
            ('guard_band', 'gathered_byte_budget')
            + compact_native_suffix
            + ('_return_stage_timings',)
        ),
    )
    telemetry_native_suffix = (
        '_return_stage_timings',
        '_return_generation_statistics',
    )
    telemetry_native = _signature_matches(
        native_method,
        native_prefix + compact_native_suffix + telemetry_native_suffix,
        defaulted=(
            ('guard_band', 'gathered_byte_budget')
            + compact_native_suffix
            + telemetry_native_suffix
        ),
    )
    direct_sparse_v3_native_v1 = _signature_matches(
        native_method,
        native_prefix
        + compact_native_suffix
        + telemetry_native_suffix
        + ('_direct_sparse_v3',),
        defaulted=(
            ('guard_band', 'gathered_byte_budget')
            + compact_native_suffix
            + telemetry_native_suffix
            + ('_direct_sparse_v3',)
        ),
    )
    ga_pruning_native = _signature_matches(
        native_method,
        native_prefix
        + compact_native_suffix
        + telemetry_native_suffix
        + ('_direct_sparse_v3', '_ga_target_cutoffs'),
        defaulted=(
            ('guard_band', 'gathered_byte_budget')
            + compact_native_suffix
            + telemetry_native_suffix
            + ('_direct_sparse_v3', '_ga_target_cutoffs')
        ),
    )
    direct_sparse_v3_native = (
        direct_sparse_v3_native_v1 or ga_pruning_native
    )

    compact_probe = getattr(
        _pipeline, '_compact_domains_seam_available', None
    )
    compact_tail = getattr(
        _pipeline, '_compact_tail_fingerprint_bound', None
    )
    legacy_pipeline = compact_probe is None and compact_tail is None
    compact_pipeline = callable(compact_probe) and callable(compact_tail)

    # V1 producers and consumers must remain paired. A V1 Python adapter is
    # also backward-compatible with the complete V2 native/pipeline pair: it
    # omits the optional compact fingerprint and therefore generates the same
    # guarded journal it did before V2 existed.
    if legacy_adapter and legacy_native and legacy_pipeline:
        return True, False, False
    if not compact_pipeline:
        return False, False, False
    if legacy_adapter:
        if not (
            compact_native
            or timing_native
            or telemetry_native
            or direct_sparse_v3_native
        ):
            return False, False, False
        compact_seam = compact_probe()
        if compact_seam is not True and compact_seam is not False:
            return False, False, False
        return True, False, False
    if sparse_journal_v3_adapter:
        if not direct_sparse_v3_native:
            return False, False, False
    elif telemetry_adapter:
        if not (telemetry_native or direct_sparse_v3_native):
            return False, False, False
    elif compact_adapter:
        if not (
            compact_native
            or timing_native
            or telemetry_native
            or direct_sparse_v3_native
        ):
            return False, False, False
    else:
        return False, False, False
    compact_seam = compact_probe()
    if compact_seam is not True and compact_seam is not False:
        return False, False, False
    if legacy_adapter:
        return True, False, False
    return (
        True,
        compact_seam,
        sparse_journal_v3_adapter and sparse_journal_v3_pipeline,
    )


def gpu_profile_domain_available():
    """Return whether a session selection can carry a domain journal."""
    return _profile_continuation_capabilities()[0]


def gpu_profile_compact_available():
    """Return whether a session selection can carry compact DEVICE domains."""
    return _profile_continuation_capabilities()[1]


def gpu_profile_sparse_journal_v3_available():
    """Return whether fused session generation supports sparse journal v3."""
    return _profile_continuation_capabilities()[2]


def gpu_profile_ga_pruning_available():
    """Return whether exact gathering-cutoff GA pruning is available."""
    from plan7_gpu import SequenceBatch, _native

    selection_method = getattr(
        SequenceBatch, '_postfilter_forward_selection', None
    )
    domain_method = getattr(
        SequenceBatch, '_postfilter_forward_domain_selection', None
    )
    native_method = getattr(
        _native.SequenceBatch,
        '_postfilter_forward_domain_selection_sealed',
        None,
    )
    if not gpu_profile_sparse_journal_v3_available():
        return False
    try:
        selection_names = tuple(inspect.signature(selection_method).parameters)
        domain_names = tuple(inspect.signature(domain_method).parameters)
        native_names = tuple(inspect.signature(native_method).parameters)
    except (TypeError, ValueError):
        return False
    return (
        selection_names[-1:] == ('_ga_pruning',)
        and domain_names[-1:] == ('ga_pruning',)
        and native_names[-1:] == ('_ga_target_cutoffs',)
    )


def has_thresholds(x):
    """Check if an HMM has any bitscore cutoffs available."""
    return (x.cutoffs.gathering_available() or
            x.cutoffs.noise_available() or
            x.cutoffs.trusted_available())


def write_macsyfinder_hit(hits, macsyfinder_dir, hmm_name_to_filename=None):
    """Write one HMM's search results as a hmmsearch-format text file for MacSyFinder.

    MacSyFinder's ``--previous-run`` expects per-gene ``.search_hmm.out`` files
    inside an ``hmmer_results/`` directory.  This function writes a minimal but
    parser-compatible file from a astra_pyhmmer ``TopHits`` object.

    Called once per HMM per input FASTA file.  When ``prot_in`` is a directory
    with multiple ``.faa`` files, the same HMM file is appended to across
    successive FASTA files.  The header is written only on first call; the
    ``//`` end-of-query marker is added by ``finalize_macsyfinder_files()``.

    Parameters
    ----------
    hits : astra_pyhmmer.plan7.TopHits
        Results of searching one HMM against the sequence database.
    macsyfinder_dir : str
        Path to the output directory (will contain ``hmmer_results/``).
    hmm_name_to_filename : dict, optional
        Mapping from HMM internal NAME to the HMM filename stem.
        When provided, the output file is named by the filename stem
        (which matches what MacSyFinder expects) rather than the
        internal NAME field (which may differ).
    """
    hmm_name = hits.query.name
    hmm_length = hits.query.M

    # Use filename stem if mapping is available, fall back to internal NAME
    file_stem = hmm_name
    if hmm_name_to_filename and hmm_name in hmm_name_to_filename:
        file_stem = hmm_name_to_filename[hmm_name]

    hmmer_dir = os.path.join(macsyfinder_dir, "hmmer_results")
    os.makedirs(hmmer_dir, exist_ok=True)
    out_path = os.path.join(hmmer_dir, f"{file_stem}.search_hmm.out")

    file_exists = os.path.exists(out_path)

    with open(out_path, "a") as fh:
        # Write header only on first call for this HMM
        if not file_exists:
            fh.write("# hmmsearch :: search profile(s) against a sequence database\n")
            fh.write("# HMMER 3.4 (astra_pyhmmer); http://hmmer.org/\n")
            fh.write("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
            fh.write(f"Query:       {hmm_name}  [M={hmm_length}]\n\n")

        for hit in hits:
            if not hit.included:
                continue
            hit_name = hit.name
            hit_desc = hit.description if hit.description else ""

            fh.write(f">> {hit_name}\n")
            fh.write("   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to"
                     "    alifrom  ali to    envfrom  env to     acc\n")
            fh.write(" ---   ------ ----- --------- --------- ------- -------"
                     "    ------- -------    ------- -------    ----\n")

            for dom_idx, domain in enumerate(hit.domains.reported, start=1):
                aln = domain.alignment
                h_from = aln.hmm_from if aln else 0
                h_to   = aln.hmm_to   if aln else 0
                t_from = aln.target_from if aln else domain.env_from
                t_to   = aln.target_to   if aln else domain.env_to

                fh.write(
                    f"  {dom_idx:>3d} ! {domain.score:>7.1f} {domain.bias:>5.1f}"
                    f"  {domain.c_evalue:>9.2e}  {domain.i_evalue:>9.2e}"
                    f"  {h_from:>7d} {h_to:>7d} .."
                    f"  {t_from:>7d} {t_to:>7d} .."
                    f"  {domain.env_from:>7d} {domain.env_to:>7d} .. 0.00\n"
                )

            fh.write("\n")


def finalize_macsyfinder_files(macsyfinder_dir):
    """Append ``//`` end-of-query markers to all MacSyFinder hmmsearch output files.

    Must be called once after all ``write_macsyfinder_hit()`` calls are complete.
    """
    hmmer_dir = os.path.join(macsyfinder_dir, "hmmer_results")
    if not os.path.isdir(hmmer_dir):
        return
    for fname in os.listdir(hmmer_dir):
        if fname.endswith(".search_hmm.out"):
            fpath = os.path.join(hmmer_dir, fname)
            with open(fpath, "a") as fh:
                fh.write("//\n")

def extract_sequences(results_or_ids, protein_dict_or_outdir, outdir=None):
    """Extract hit sequences and write per-HMM FASTAs.

    Supports two calling conventions:
      - New: extract_sequences(hit_ids_by_hmm, protein_dict, outdir)
        where hit_ids_by_hmm is dict[str, set[str]]
      - Legacy (scan.py): extract_sequences(results_dataframes_dict, outdir)
        where results_dataframes_dict is dict[str, DataFrame]
    """
    import pandas as pd

    if outdir is None:
        # Legacy call: extract_sequences(results_dataframes_dict, outdir)
        # results_or_ids is a dict of DataFrames, protein_dict_or_outdir is outdir
        outdir = protein_dict_or_outdir
        fastas_dir = os.path.join(outdir, 'fastas')
        os.makedirs(fastas_dir, exist_ok=True)
        for genome_file, df in results_or_ids.items():
            for hmm_name in df['hmm_name'].unique():
                ids = df[df['hmm_name'] == hmm_name]['sequence_id'].tolist()
                hits_fasta = os.path.join(fastas_dir, f"{hmm_name}.faa")
                with open(hits_fasta, 'a') as fh:
                    # Legacy path: re-read from disk (scan.py doesn't keep seqs in memory)
                    with astra_pyhmmer.easel.SequenceFile(genome_file, digital=True,
                                                     alphabet=astra_pyhmmer.easel.Alphabet.amino()) as sf:
                        for seq in sf:
                            if seq.name in ids:
                                text_seq = seq.textize()
                                fh.write(f">{text_seq.name}\n{text_seq.sequence}\n")
        return

    # New call: extract_sequences(hit_ids_by_hmm, protein_dict, outdir)
    hit_ids_by_hmm = results_or_ids
    protein_dict = protein_dict_or_outdir
    fastas_dir = os.path.join(outdir, 'fastas')
    os.makedirs(fastas_dir, exist_ok=True)

    # Build a flat name → sequence lookup (once, not per-HMM)
    seq_lookup = {}
    for sequences in protein_dict.values():
        for seq in sequences:
            seq_lookup[seq.name] = seq

    for hmm_name, seq_ids in hit_ids_by_hmm.items():
        hits_fasta = os.path.join(fastas_dir, f"{hmm_name}.faa")
        with open(hits_fasta, 'w') as fh:
            for sid in seq_ids:
                seq = seq_lookup.get(sid)
                if seq is not None:
                    text_seq = seq.textize()
                    fh.write(f">{text_seq.name}\n{text_seq.sequence}\n")


def _gpu_astra_tsv_capability():
    """Return the exact worker-rendered ABI objects, or ``(None, None)``."""
    try:
        from plan7_gpu import astra_search as gpu_search_module
    except ImportError:
        return None, None
    if (
        getattr(gpu_search_module, '_ASTRA_TSV_RENDERER_ABI', None)
        != _ASTRA_TSV_RENDERER_ABI
    ):
        return None, None
    rows_type = getattr(gpu_search_module, '_AstraTSVRows', None)
    if not isinstance(rows_type, type):
        return None, None
    return gpu_search_module, rows_type


def _gpu_hmmsearch_entrypoint(continuation_pool=False):
    """Feature-detect the exact native sink and otherwise use public rows."""
    from plan7_gpu import astra_search as gpu_search_module

    native_module, _ = _gpu_astra_tsv_capability()
    if continuation_pool:
        native_name = '_hmmsearch_astra_tsv_with_continuation_pool'
        fallback_name = '_hmmsearch_with_continuation_pool'
    else:
        native_name = '_hmmsearch_astra_tsv'
        fallback_name = 'hmmsearch'
    if native_module is not None:
        native = getattr(native_module, native_name, None)
        if callable(native):
            return native
    return getattr(gpu_search_module, fallback_name)


def _consume_gpu_candidate_chunk(spec, candidates, total_chunks, threads, fh,
                                 gpu_hmmsearch, gpu_metrics,
                                 telemetry_collector=None,
                                 continuation_pool=None,
                                 commit_progress=True):
    chunk_index, hmm_chunk, _, kwargs = spec
    if commit_progress:
        print(f"  Chunk {chunk_index}/{total_chunks} "
              f"({len(hmm_chunk)} HMMs)...", end="", flush=True)
    hit_iterator = None
    hits = None
    completed = False
    _, rendered_rows_type = _gpu_astra_tsv_capability()
    started = time.perf_counter()
    try:
        continuation_options = {}
        if continuation_pool is not None:
            continuation_options['continuation_pool'] = continuation_pool
        if telemetry_collector is None:
            hit_iterator = gpu_hmmsearch(
                hmm_chunk,
                candidates,
                cpus=threads,
                postfilter=True,
                **continuation_options,
                **kwargs,
            )
        else:
            hit_iterator = gpu_hmmsearch(
                hmm_chunk,
                candidates,
                cpus=threads,
                postfilter=True,
                telemetry_collector=telemetry_collector,
                profile_ordinals=tuple(spec[2]),
                **continuation_options,
                **kwargs,
            )
        for hits in hit_iterator:
            if (
                rendered_rows_type is not None
                and type(hits) is rendered_rows_type
            ):
                fh.write(hits.rows)
                if gpu_metrics is not None:
                    with gpu_metrics._lock:
                        gpu_metrics.tsv_worker_rendered_profile_count += 1
                        gpu_metrics.tsv_worker_rendered_row_count += (
                            hits.row_count
                        )
                        gpu_metrics.tsv_worker_rendered_bytes += (
                            hits.byte_count
                        )
            else:
                if gpu_metrics is not None:
                    with gpu_metrics._lock:
                        gpu_metrics.tsv_consumer_fallback_profile_count += 1
                process_hits_to_file(hits, fh)
        completed = True
    except BaseException:
        if hit_iterator is not None:
            close = getattr(hit_iterator, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException:
                    pass
        raise
    finally:
        finished = time.perf_counter()
        if gpu_metrics is not None:
            with gpu_metrics._lock:
                gpu_metrics.continuation_seconds += finished - started
                gpu_metrics.continuation_call_seconds += finished - started
                gpu_metrics.continuation_records.append({
                    'chunk_index': chunk_index,
                    'profile_indices': tuple(spec[2]),
                    'profile_count': len(hmm_chunk),
                    'started_monotonic_seconds': started,
                    'finished_monotonic_seconds': finished,
                    'duration_seconds': finished - started,
                    'completed': completed,
                })
        hit_iterator = None
        hits = None
    if gpu_metrics is not None and commit_progress:
        with gpu_metrics._lock:
            gpu_metrics.consumed_chunk_count += 1
    if commit_progress:
        print(" done")
    return started, finished


def _generate_gpu_profile_candidates(sequence_batch, selection, kwargs,
                                     domain_continuation=False,
                                     telemetry=False,
                                     sparse_journal_v3=False,
                                     ga_pruning=False):
    """Run the same bounded CUDA stages as the live post-filter product path."""
    F1 = kwargs.get('F1', 0.02)
    F2 = kwargs.get('F2', 0.001)
    F3 = kwargs.get('F3', 0.00001)
    bias_filter = kwargs.get('bias_filter', True)
    if type(telemetry) is not bool:
        raise TypeError("telemetry must be bool")
    if type(sparse_journal_v3) is not bool:
        raise TypeError("sparse_journal_v3 must be bool")
    if type(ga_pruning) is not bool:
        raise TypeError("ga_pruning must be bool")
    if telemetry and (not domain_continuation or bias_filter is not True):
        raise GPUConfigurationError(
            "route telemetry requires fused domain continuation with bias filtering"
        )
    if sparse_journal_v3 and (
        not domain_continuation or bias_filter is not True
    ):
        raise GPUConfigurationError(
            "sparse journal v3 requires fused domain continuation with bias filtering"
        )
    if ga_pruning and not sparse_journal_v3:
        raise GPUConfigurationError("GA pruning requires sparse journal v3")
    if domain_continuation and bias_filter is True:
        # This configuration-only pipeline is private to the producer call.
        # CPU continuation workers still create and exclusively own their
        # search pipelines lazily inside plan7_gpu.astra_search.
        generation_pipeline = astra_pyhmmer.plan7.Pipeline(
            sequence_batch.alphabet,
            **kwargs,
        )
        generation_options = {
            'pipeline': generation_pipeline,
            'domain_guard': GPU_DOMAIN_GUARD,
        }
        if telemetry:
            generation_options['telemetry'] = True
        if sparse_journal_v3:
            generation_options['sparse_journal_v3'] = True
        if ga_pruning:
            generation_options['_ga_pruning'] = True
        return sequence_batch._postfilter_forward_selection(
            selection,
            F1,
            F2,
            F3,
            bias_filter,
            **generation_options,
        )
    return sequence_batch._postfilter_forward_selection(
        selection, F1, F2, F3, bias_filter
    )


def _generate_gpu_profile_candidates_for_run(
        sequence_batch, selection, kwargs, domain_continuation,
        telemetry_collector, sparse_journal_v3=False, ga_pruning=False):
    """Preserve the exact default call and add only explicit opt-ins."""
    if telemetry_collector is None and not sparse_journal_v3 and not ga_pruning:
        return _generate_gpu_profile_candidates(
            sequence_batch, selection, kwargs, domain_continuation
        )
    generation_options = {}
    if telemetry_collector is not None:
        generation_options['telemetry'] = True
    if sparse_journal_v3:
        generation_options['sparse_journal_v3'] = True
    if ga_pruning:
        generation_options['ga_pruning'] = True
    return _generate_gpu_profile_candidates(
        sequence_batch, selection, kwargs, domain_continuation,
        **generation_options,
    )


def _consume_gpu_candidate_chunk_for_run(
        spec, candidates, total_chunks, threads, fh, gpu_hmmsearch,
        gpu_metrics, telemetry_collector, continuation_pools=None,
        commit_progress=True):
    """Preserve ordinary TopHits consumption unless telemetry is explicit."""
    if telemetry_collector is None:
        return _consume_gpu_candidate_chunk(
            spec,
            candidates,
            total_chunks,
            threads,
            fh,
            gpu_hmmsearch,
            gpu_metrics,
            continuation_pool=(
                None
                if continuation_pools is None
                else continuation_pools[id(spec[3])]
            ),
            commit_progress=commit_progress,
        )
    return _consume_gpu_candidate_chunk(
        spec,
        candidates,
        total_chunks,
        threads,
        fh,
        gpu_hmmsearch,
        gpu_metrics,
        telemetry_collector=telemetry_collector,
        continuation_pool=(
            None
            if continuation_pools is None
            else continuation_pools[id(spec[3])]
        ),
        commit_progress=commit_progress,
    )


def _run_gpu_profile_serial(chunks, profile_session, sequence_batch,
                            threads, fh, gpu_metrics=None,
                            domain_continuation=False,
                            telemetry_collector=None,
                            sparse_journal_v3=False, ga_pruning=False,
                            continuation_pools=None):
    """Run sealed GPU-through-Forward generation as the serial control."""
    gpu_hmmsearch = _gpu_hmmsearch_entrypoint(
        continuation_pool=continuation_pools is not None
    )

    if gpu_metrics is not None:
        statistics = profile_session.statistics
        gpu_metrics.scheduler_mode = 'serial'
        gpu_metrics.chunk_count += len(chunks)
        gpu_metrics.profile_worker_count = statistics['worker_count']
        gpu_metrics.profile_build_worker_count = statistics.get(
            'build_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_selection_worker_count = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_host_bytes = statistics['host_bytes']
    pipeline_started = time.perf_counter()
    try:
        for spec in chunks:
            selection_started = time.perf_counter()
            selection = profile_session.select(spec[2])
            if gpu_metrics is not None:
                gpu_metrics.selection_seconds += (
                    time.perf_counter() - selection_started
                )
            generation_started = time.perf_counter()
            try:
                candidates = _generate_gpu_profile_candidates_for_run(
                    sequence_batch,
                    selection,
                    spec[3],
                    domain_continuation,
                    telemetry_collector,
                    sparse_journal_v3,
                    ga_pruning,
                )
            except BaseException:
                try:
                    selection.close()
                except BaseException:
                    pass
                raise
            selection.close()
            generation_finished = time.perf_counter()
            if gpu_metrics is not None:
                gpu_metrics.generated_chunk_count += 1
                gpu_metrics.generation_seconds += (
                    generation_finished - generation_started
                )
            _consume_gpu_candidate_chunk_for_run(
                spec,
                candidates,
                len(chunks),
                threads,
                fh,
                gpu_hmmsearch,
                gpu_metrics,
                telemetry_collector,
                continuation_pools,
            )
            candidates = None
    finally:
        if gpu_metrics is not None:
            gpu_metrics.pipeline_wall_seconds += (
                time.perf_counter() - pipeline_started
            )


def _run_gpu_profile_single_prefetch(chunks, profile_session, sequence_batch,
                                     threads, fh, gpu_metrics=None,
                                     domain_continuation=False,
                                     telemetry_collector=None,
                                     sparse_journal_v3=False,
                                     ga_pruning=False,
                                     continuation_pools=None):
    """Retain the original one-future overlap scheduler as a control."""
    if not chunks:
        return

    gpu_hmmsearch = _gpu_hmmsearch_entrypoint(
        continuation_pool=continuation_pools is not None
    )

    if gpu_metrics is not None:
        statistics = profile_session.statistics
        gpu_metrics.scheduler_mode = 'single-prefetch'
        gpu_metrics.chunk_count += len(chunks)
        gpu_metrics.profile_worker_count = statistics['worker_count']
        gpu_metrics.profile_build_worker_count = statistics.get(
            'build_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_selection_worker_count = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_host_bytes = statistics['host_bytes']

    def generate(spec, selection):
        started = time.perf_counter()
        try:
            candidates = _generate_gpu_profile_candidates_for_run(
                sequence_batch,
                selection,
                spec[3],
                domain_continuation,
                telemetry_collector,
                sparse_journal_v3,
                ga_pruning,
            )
        except BaseException:
            try:
                selection.close()
            except BaseException:
                pass
            raise
        selection.close()
        return candidates, started, time.perf_counter()

    pipeline_started = time.perf_counter()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix='aksha-gpu-generate',
    )
    pending_future = None
    pending_error = None
    pending_selection = None
    def start_generation(spec):
        selection_started = time.perf_counter()
        try:
            selection = profile_session.select(spec[2])
        except BaseException as error:
            if gpu_metrics is not None:
                gpu_metrics.selection_seconds += (
                    time.perf_counter() - selection_started
                )
            return None, error, None
        if gpu_metrics is not None:
            gpu_metrics.selection_seconds += (
                time.perf_counter() - selection_started
            )
        try:
            return executor.submit(generate, spec, selection), None, selection
        except BaseException as error:
            try:
                selection.close()
            except BaseException:
                pass
            return None, error, None

    pending_future, pending_error, pending_selection = start_generation(chunks[0])
    previous_consumption = None
    try:
        for position, spec in enumerate(chunks):
            if pending_error is not None:
                raise pending_error
            completed_future = pending_future
            pending_future = None
            pending_selection = None
            ready_without_wait = completed_future.done()
            wait_started = time.perf_counter()
            candidates, generation_started, generation_finished = (
                completed_future.result()
            )
            wait_seconds = time.perf_counter() - wait_started
            if gpu_metrics is not None:
                gpu_metrics.generated_chunk_count += 1
                gpu_metrics.ready_without_wait_count += int(ready_without_wait)
                gpu_metrics.generation_seconds += (
                    generation_finished - generation_started
                )
                gpu_metrics.generation_wait_seconds += wait_seconds
                if position == 0:
                    gpu_metrics.initial_generation_wait_seconds += wait_seconds
                else:
                    gpu_metrics.pipeline_stall_seconds += wait_seconds
                if previous_consumption is not None:
                    overlap_start = max(
                        generation_started, previous_consumption[0]
                    )
                    overlap_finish = min(
                        generation_finished, previous_consumption[1]
                    )
                    gpu_metrics.overlap_seconds += max(
                        0.0, overlap_finish - overlap_start
                    )

            # Preselection is synchronous and the session has zero pack
            # workers, so its copy cannot contend with CPU continuation.
            # A prefetch failure is deferred until this ready chunk is emitted.
            if position + 1 < len(chunks):
                (
                    pending_future,
                    pending_error,
                    pending_selection,
                ) = start_generation(chunks[position + 1])

            previous_consumption = _consume_gpu_candidate_chunk_for_run(
                spec,
                candidates,
                len(chunks),
                threads,
                fh,
                gpu_hmmsearch,
                gpu_metrics,
                telemetry_collector,
                continuation_pools,
            )
            candidates = None
    finally:
        if pending_future is not None:
            cancelled = pending_future.cancel()
            if cancelled and pending_selection is not None:
                try:
                    pending_selection.close()
                except BaseException:
                    pass
        active_error = sys.exc_info()[0] is not None
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            if not active_error:
                raise
        if gpu_metrics is not None:
            gpu_metrics.pipeline_wall_seconds += (
                time.perf_counter() - pipeline_started
            )


def _run_gpu_profile_pipeline(chunks, profile_session, sequence_batch,
                              threads, fh, gpu_metrics=None,
                              domain_continuation=False,
                              ready_queue_configuration=None,
                              telemetry_collector=None,
                              sparse_journal_v3=False, ga_pruning=False,
                              continuation_pools=None,
                              continuation_window=1):
    """Continuously produce ordered GPU batches into a bounded ready queue.

    The producer owns selection and CUDA generation.  While the caller consumes
    chunk ``i``, a depth-``d`` queue may retain the next ``d`` chunks and the
    producer may work on one further chunk.  The default remains the audited
    single ready slot.  Experimental deeper queues require an exact byte cap
    and exact immutable ``CandidateBatch.resident_bytes`` charges.
    """
    if not chunks:
        return

    if ready_queue_configuration is None:
        ready_queue_configuration = gpu_ready_queue_configuration()
    if (
        type(ready_queue_configuration) is not tuple
        or len(ready_queue_configuration) != 2
    ):
        raise TypeError(
            "ready_queue_configuration must be an exact (depth, bytes) tuple"
        )
    ready_depth, ready_byte_capacity = ready_queue_configuration
    if type(ready_depth) is not int or ready_depth not in GPU_READY_QUEUE_DEPTHS:
        raise ValueError("ready queue depth must be exactly 1, 2, or 4")
    if ready_byte_capacity is not None and (
        type(ready_byte_capacity) is not int or ready_byte_capacity <= 0
    ):
        raise ValueError("ready queue byte capacity must be positive or None")
    if ready_depth != GPU_READY_QUEUE_CAPACITY and ready_byte_capacity is None:
        raise ValueError("deeper ready queues require an exact byte capacity")
    if (
        type(continuation_window) is not int
        or continuation_window not in GPU_CONTINUATION_WINDOWS
    ):
        raise ValueError("continuation window must be exactly 1, 2, or 4")
    if continuation_window > 1:
        if continuation_pools is None:
            raise ValueError("concurrent continuation requires request pools")
        if telemetry_collector is not None:
            raise ValueError(
                "concurrent continuation does not support route telemetry"
            )
        if len(_gpu_pipeline_option_groups(chunks)) != 1:
            raise ValueError(
                "concurrent continuation requires one Pipeline option group"
            )

    gpu_hmmsearch = _gpu_hmmsearch_entrypoint(
        continuation_pool=continuation_pools is not None
    )

    if gpu_metrics is not None:
        statistics = profile_session.statistics
        gpu_metrics.scheduler_mode = 'bounded-ready-queue'
        gpu_metrics.ready_queue_capacity = ready_depth
        gpu_metrics.ready_queue_byte_capacity = ready_byte_capacity or 0
        gpu_metrics.producer_lookahead_capacity = (
            ready_depth + 1
        )
        gpu_metrics.live_candidate_capacity = ready_depth + 2
        gpu_metrics.continuation_window = continuation_window
        gpu_metrics.live_candidate_capacity = (
            ready_depth + continuation_window + 1
        )
        gpu_metrics.chunk_count += len(chunks)
        gpu_metrics.profile_worker_count = statistics['worker_count']
        gpu_metrics.profile_build_worker_count = statistics.get(
            'build_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_selection_worker_count = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_host_bytes = statistics['host_bytes']

    if ready_byte_capacity is None:
        # Preserve the audited default implementation, not merely its nominal
        # capacity.  This path is selected only for the default depth of one.
        ready = Queue(maxsize=ready_depth)
    else:
        ready = _GPUByteBoundedReadyQueue(
            ready_depth, ready_byte_capacity
        )
    stop = Event()
    producer_done = Event()
    producer_state = {
        'idle_count': 0,
        'idle_seconds': 0.0,
        'ready_high_water': 0,
        'terminal_error': None,
    }

    def put_nowait(item, item_bytes):
        if ready_byte_capacity is None:
            ready.put_nowait(item)
        else:
            ready.put_nowait(item, item_bytes)

    def put_with_timeout(item, item_bytes):
        if ready_byte_capacity is None:
            ready.put(item, timeout=0.05)
            return True
        else:
            return ready.put(
                item, item_bytes, timeout=0.05, stop=stop
            )

    def publish(item, item_bytes):
        if stop.is_set():
            return False
        try:
            put_nowait(item, item_bytes)
        except Full:
            producer_state['idle_count'] += 1
            idle_started = time.perf_counter()
            while not stop.is_set():
                try:
                    inserted = put_with_timeout(item, item_bytes)
                    producer_state['idle_seconds'] += (
                        time.perf_counter() - idle_started
                    )
                    if not inserted:
                        return False
                    if ready_byte_capacity is None:
                        producer_state['ready_high_water'] = 1
                    return True
                except Full:
                    pass
            producer_state['idle_seconds'] += (
                time.perf_counter() - idle_started
            )
            return False
        if ready_byte_capacity is None:
            producer_state['ready_high_water'] = 1
        return True

    streamed_profiles = type(profile_session) is _PressedGPUProfileStream

    def close_generation_context(selection, chunk_session):
        close_error = None
        for resource in (selection, chunk_session):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        return close_error

    def record_stream_session(pair_count, build_seconds):
        if gpu_metrics is None:
            return
        statistics = profile_session.statistics
        with gpu_metrics._lock:
            gpu_metrics.profile_streamed = True
            gpu_metrics.profile_chunk_local_pack = True
            gpu_metrics.profile_stream_session_count = statistics[
                'stream_session_count'
            ]
            gpu_metrics.profile_stream_max_pairs = max(
                gpu_metrics.profile_stream_max_pairs,
                pair_count,
            )
            gpu_metrics.session_build_seconds += build_seconds
            gpu_metrics.profile_host_bytes = max(
                gpu_metrics.profile_host_bytes,
                statistics['host_bytes'],
            )
            gpu_metrics.profile_worker_count = max(
                gpu_metrics.profile_worker_count,
                statistics['worker_count'],
            )
            gpu_metrics.profile_build_worker_count = max(
                gpu_metrics.profile_build_worker_count,
                statistics.get(
                    'build_worker_count', statistics['worker_count']
                ),
            )
            gpu_metrics.profile_pointer_bytes = max(
                gpu_metrics.profile_pointer_bytes,
                statistics.get('profile_pointer_bytes', 0),
            )
            gpu_metrics.profile_identity_token_bytes = max(
                gpu_metrics.profile_identity_token_bytes,
                statistics.get('identity_token_bytes', 0),
            )
            gpu_metrics.profile_background_bytes = max(
                gpu_metrics.profile_background_bytes,
                statistics.get('background_bytes', 0),
            )

    def produce():
        try:
            for position, spec in enumerate(chunks):
                if stop.is_set():
                    return
                if position >= len(chunks):
                    raise RuntimeError(
                        "GPU profile producer exceeded declared chunk count"
                    )
                selection = None
                chunk_session = None
                candidates = None
                if streamed_profiles:
                    session_started = time.perf_counter()
                    try:
                        chunk_session = profile_session.open_chunk_session(
                            spec[1]
                        )
                    except BaseException as error:
                        selection_finished = time.perf_counter()
                        item = (
                            position, spec, None, error,
                            session_started, selection_finished, None, None, 0,
                        )
                        publish(item, 0)
                        item = None
                        return
                    session_finished = time.perf_counter()
                    record_stream_session(
                        len(spec[1]), session_finished - session_started
                    )
                    selection_indices = range(len(spec[1]))
                else:
                    selection_indices = spec[2]
                selection_started = time.perf_counter()
                try:
                    active_session = (
                        chunk_session if streamed_profiles else profile_session
                    )
                    selection = active_session.select(selection_indices)
                    if streamed_profiles:
                        profile_session.record_selection()
                except BaseException as error:
                    close_generation_context(None, chunk_session)
                    chunk_session = None
                    selection_finished = time.perf_counter()
                    item = (
                        position, spec, None, error,
                        selection_started, selection_finished, None, None, 0,
                    )
                    publish(item, 0)
                    item = None
                    return
                selection_finished = time.perf_counter()
                generation_started = time.perf_counter()
                try:
                    candidates = _generate_gpu_profile_candidates_for_run(
                        sequence_batch,
                        selection,
                        spec[3],
                        domain_continuation,
                        telemetry_collector,
                        sparse_journal_v3,
                        ga_pruning,
                    )
                except BaseException as error:
                    close_generation_context(selection, chunk_session)
                    selection = None
                    chunk_session = None
                    item = (
                        position, spec, None, error,
                        selection_started, selection_finished,
                        generation_started, None, 0,
                    )
                    publish(item, 0)
                    item = None
                    return
                close_error = close_generation_context(
                    selection, chunk_session
                )
                selection = None
                chunk_session = None
                if close_error is not None:
                    candidates = None
                    item = (
                        position, spec, None, close_error,
                        selection_started, selection_finished,
                        generation_started, None, 0,
                    )
                    publish(item, 0)
                    item = None
                    return
                generation_finished = time.perf_counter()
                candidate_bytes = 0
                if (
                    ready_byte_capacity is not None
                    or continuation_window > 1
                ):
                    try:
                        candidate_bytes = _gpu_candidate_resident_bytes(
                            candidates
                        )
                        if (
                            ready_byte_capacity is not None
                            and candidate_bytes > ready_byte_capacity
                        ):
                            raise GPUConfigurationError(
                                f"generated CandidateBatch requires "
                                f"{candidate_bytes} bytes, exceeding "
                                f"{GPU_READY_QUEUE_BYTES_ENV}="
                                f"{ready_byte_capacity}"
                            )
                    except BaseException as error:
                        candidates = None
                        item = (
                            position, spec, None, error,
                            selection_started, selection_finished,
                            generation_started, generation_finished, 0,
                        )
                        publish(item, 0)
                        item = None
                        return
                item = (
                    position, spec, candidates, None,
                    selection_started, selection_finished,
                    generation_started, generation_finished,
                    candidate_bytes,
                )
                if not publish(item, candidate_bytes):
                    item = None
                    candidates = None
                    return
                item = None
                candidates = None
        except BaseException as error:
            producer_state['terminal_error'] = error
        finally:
            close_chunks = getattr(chunks, 'close', None)
            if close_chunks is not None:
                try:
                    close_chunks()
                except BaseException as error:
                    if producer_state['terminal_error'] is None:
                        producer_state['terminal_error'] = error
            producer_done.set()

    def next_ready():
        try:
            return ready.get_nowait(), True
        except Empty:
            pass
        while True:
            try:
                return ready.get(timeout=0.05), False
            except Empty:
                if producer_done.is_set():
                    error = producer_state['terminal_error']
                    if error is not None:
                        raise error
                    raise RuntimeError(
                        "GPU profile producer stopped before publishing "
                        "every ordered chunk"
                    )

    def discard_ready():
        while True:
            try:
                item = ready.get_nowait()
            except Empty:
                return
            item = None

    pipeline_started = time.perf_counter()
    producer = Thread(
        target=produce,
        name='aksha-gpu-generate_0',
        daemon=False,
    )
    producer_started = False
    consumption_intervals = []
    continuation_executor = None
    pending_continuations = deque()
    pending_pool_key = None

    def finish_oldest_continuation():
        nonlocal pending_pool_key
        (
            _position,
            future,
            output_buffer,
            candidate_bytes,
            spec,
        ) = pending_continuations[0]
        continuation_error = None
        committed_entry = False
        try:
            try:
                interval = future.result()
            except BaseException as error:
                if not future.done():
                    # The wait, rather than the coordinator task, was
                    # interrupted.  Keep the future and its buffer in the
                    # pending deque: cleanup owns both until the task reaches
                    # a terminal state.
                    raise
                try:
                    interval = future.result()
                except BaseException as worker_error:
                    interval = None
                    continuation_error = worker_error
                else:
                    # The worker completed successfully while the coordinator
                    # wait itself was interrupted.  Preserve the interruption
                    # and leave the buffer owned by cleanup; never commit a
                    # speculative partial/complete chunk after cancellation.
                    raise error
            committed = pending_continuations.popleft()
            committed_entry = True
            if committed[1] is not future:
                raise RuntimeError(
                    "continuation window changed canonical future order"
                )
            rendered = output_buffer.getvalue()
            fh.write(rendered)
            if gpu_metrics is not None:
                gpu_metrics.consumed_candidate_bytes += candidate_bytes
                gpu_metrics.continuation_maximum_chunk_output_bytes = max(
                    gpu_metrics.continuation_maximum_chunk_output_bytes,
                    len(rendered.encode('utf-8')),
                )
            if interval is not None:
                consumption_intervals.append(interval)
            if continuation_error is None:
                if gpu_metrics is not None:
                    with gpu_metrics._lock:
                        gpu_metrics.consumed_chunk_count += 1
                print(
                    f"  Chunk {spec[0]}/{len(chunks)} "
                    f"({len(spec[1])} HMMs)... done"
                )
        finally:
            if committed_entry:
                output_buffer.close()
                if not pending_continuations:
                    pending_pool_key = None
        if continuation_error is not None:
            raise continuation_error

    def finish_all_pending_continuations():
        while pending_continuations:
            finish_oldest_continuation()

    def raise_after_pending_continuations(error):
        finish_all_pending_continuations()
        raise error

    try:
        if continuation_window > 1:
            continuation_executor = ThreadPoolExecutor(
                max_workers=continuation_window,
                thread_name_prefix='aksha-gpu-continue',
            )
        producer.start()
        producer_started = True
        for position in range(len(chunks)):
            wait_started = time.perf_counter()
            try:
                item, ready_without_wait = next_ready()
            except BaseException as error:
                raise_after_pending_continuations(error)
            ready_dequeued = time.perf_counter()
            wait_seconds = ready_dequeued - wait_started
            (
                produced_position,
                produced_spec,
                candidates,
                producer_error,
                selection_started,
                selection_finished,
                generation_started,
                generation_finished,
                candidate_bytes,
            ) = item
            item = None
            expected_spec = (
                None if type(chunks) is _PressedGPUProfileChunks
                else chunks[position]
            )
            if (
                produced_position != position
                or (
                    expected_spec is not None
                    and produced_spec is not expected_spec
                )
            ):
                candidates = None
                raise_after_pending_continuations(
                    RuntimeError("GPU profile producer changed chunk order")
                )
            spec = produced_spec
            if gpu_metrics is not None:
                gpu_metrics.selection_seconds += (
                    selection_finished - selection_started
                )
            if producer_error is not None:
                raise_after_pending_continuations(producer_error)
            if generation_started is None or generation_finished is None:
                candidates = None
                raise_after_pending_continuations(
                    RuntimeError(
                        "GPU profile producer omitted generation timing"
                    )
                )
            if ready_byte_capacity is not None:
                try:
                    observed_candidate_bytes = _gpu_candidate_resident_bytes(
                        candidates
                    )
                except BaseException as error:
                    candidates = None
                    raise_after_pending_continuations(error)
                if observed_candidate_bytes != candidate_bytes:
                    candidates = None
                    raise_after_pending_continuations(
                        RuntimeError(
                            "CandidateBatch.resident_bytes changed while "
                            f"queued: {candidate_bytes} -> "
                            f"{observed_candidate_bytes}"
                        )
                    )
            if gpu_metrics is not None:
                gpu_metrics.generated_chunk_count += 1
                gpu_metrics.generated_candidate_bytes += candidate_bytes
                gpu_metrics.maximum_candidate_bytes = max(
                    gpu_metrics.maximum_candidate_bytes,
                    candidate_bytes,
                )
                gpu_metrics.generation_records.append({
                    'position': position,
                    'chunk_index': spec[0],
                    'profile_indices': tuple(spec[2]),
                    'profile_count': len(spec[1]),
                    'selection_started_monotonic_seconds': selection_started,
                    'selection_finished_monotonic_seconds': selection_finished,
                    'selection_duration_seconds': (
                        selection_finished - selection_started
                    ),
                    'generation_started_monotonic_seconds': generation_started,
                    'generation_finished_monotonic_seconds': generation_finished,
                    'generation_duration_seconds': (
                        generation_finished - generation_started
                    ),
                    'ready_wait_started_monotonic_seconds': wait_started,
                    'ready_dequeued_monotonic_seconds': ready_dequeued,
                    'ready_wait_duration_seconds': wait_seconds,
                    'ready_without_wait': ready_without_wait,
                    'candidate_resident_bytes': (
                        candidate_bytes
                        if (
                            ready_byte_capacity is not None
                            or continuation_window > 1
                        )
                        else None
                    ),
                })
                gpu_metrics.ready_without_wait_count += int(ready_without_wait)
                gpu_metrics.generation_seconds += (
                    generation_finished - generation_started
                )
                gpu_metrics.generation_wait_seconds += wait_seconds
                if position == 0:
                    gpu_metrics.initial_generation_wait_seconds += wait_seconds
                else:
                    gpu_metrics.pipeline_stall_seconds += wait_seconds
                generation_lookahead = 0
                for consumed_position, (
                    consumption_started,
                    consumption_finished,
                ) in enumerate(consumption_intervals):
                    overlap_start = max(
                        generation_started, consumption_started
                    )
                    overlap_finish = min(
                        generation_finished, consumption_finished
                    )
                    gpu_metrics.overlap_seconds += max(
                        0.0, overlap_finish - overlap_start
                    )
                    # A generation may begin in the narrow handoff before
                    # continuation records its start.  It is still useful
                    # lookahead when it began before that older chunk finished.
                    if generation_started < consumption_finished:
                        generation_lookahead = max(
                            generation_lookahead,
                            position - consumed_position,
                        )
                gpu_metrics.producer_lookahead_high_water = max(
                    gpu_metrics.producer_lookahead_high_water,
                    generation_lookahead,
                )
                gpu_metrics.producer_lookahead_start_count += int(
                    generation_lookahead >= 2
                )

            if continuation_executor is None:
                consumption_interval = _consume_gpu_candidate_chunk_for_run(
                    spec,
                    candidates,
                    len(chunks),
                    threads,
                    fh,
                    gpu_hmmsearch,
                    gpu_metrics,
                    telemetry_collector,
                    continuation_pools,
                )
                if gpu_metrics is not None:
                    gpu_metrics.consumed_candidate_bytes += candidate_bytes
                candidates = None
                consumption_intervals.append(consumption_interval)
            else:
                pool_key = id(spec[3])
                if (
                    pending_pool_key is not None
                    and pending_pool_key != pool_key
                ):
                    finish_all_pending_continuations()
                if pending_pool_key is None:
                    pending_pool_key = pool_key
                output_buffer = io.StringIO()
                try:
                    future = continuation_executor.submit(
                        _consume_gpu_candidate_chunk_for_run,
                        spec,
                        candidates,
                        len(chunks),
                        threads,
                        output_buffer,
                        gpu_hmmsearch,
                        gpu_metrics,
                        telemetry_collector,
                        continuation_pools,
                        False,
                    )
                except BaseException as error:
                    output_buffer.close()
                    candidates = None
                    raise_after_pending_continuations(error)
                pending_continuations.append((
                    position,
                    future,
                    output_buffer,
                    candidate_bytes,
                    spec,
                ))
                candidates = None
                if gpu_metrics is not None:
                    gpu_metrics.continuation_window_high_water = max(
                        gpu_metrics.continuation_window_high_water,
                        len(pending_continuations),
                    )
                if len(pending_continuations) >= continuation_window:
                    finish_oldest_continuation()
        finish_all_pending_continuations()
        if continuation_executor is not None and gpu_metrics is not None:
            with gpu_metrics._lock:
                gpu_metrics.continuation_records.sort(
                    key=lambda record: record['chunk_index']
                )
                generation_intervals = _merged_time_intervals(
                    gpu_metrics.generation_records,
                    'generation_started_monotonic_seconds',
                    'generation_finished_monotonic_seconds',
                )
                continuation_intervals = _merged_time_intervals(
                    gpu_metrics.continuation_records,
                    'started_monotonic_seconds',
                    'finished_monotonic_seconds',
                )
                gpu_metrics.continuation_seconds = sum(
                    finished - started
                    for started, finished in continuation_intervals
                )
                gpu_metrics.overlap_seconds = (
                    _interval_intersection_seconds(
                        generation_intervals,
                        continuation_intervals,
                    )
                )
                lookahead_high_water = 0
                lookahead_start_count = 0
                for generation_record in gpu_metrics.generation_records:
                    position = generation_record['position']
                    generation_started = generation_record[
                        'generation_started_monotonic_seconds'
                    ]
                    generation_lookahead = 0
                    for consumed_position, continuation_record in enumerate(
                        gpu_metrics.continuation_records
                    ):
                        if consumed_position >= position:
                            break
                        if generation_started < continuation_record[
                            'finished_monotonic_seconds'
                        ]:
                            generation_lookahead = max(
                                generation_lookahead,
                                position - consumed_position,
                            )
                    lookahead_high_water = max(
                        lookahead_high_water,
                        generation_lookahead,
                    )
                    lookahead_start_count += int(
                        generation_lookahead >= 2
                    )
                gpu_metrics.producer_lookahead_high_water = (
                    lookahead_high_water
                )
                gpu_metrics.producer_lookahead_start_count = (
                    lookahead_start_count
                )
    finally:
        active_error = sys.exc_info()[1]
        stop.set()
        if ready_byte_capacity is not None:
            ready.wake_all()
        discard_ready()
        join_error = None
        if producer_started:
            # Thread.is_alive() cannot prove target termination after an
            # interrupted join: CPython may mark the Thread stopped while its
            # target is still running.  The target-owned event is its final
            # action and therefore seals all session and CandidateBatch use.
            while not producer_done.is_set():
                try:
                    producer_done.wait(timeout=0.05)
                except BaseException as error:
                    if join_error is None:
                        join_error = error
            # Retire the Python thread too.  A transient join interruption is
            # recorded, but cleanup is retried to completion before the outer
            # scope may close the session or target batch.
            while True:
                try:
                    producer.join()
                except BaseException as error:
                    if join_error is None:
                        join_error = error
                    continue
                break
        if continuation_executor is not None:
            # A wait on Future.result() may itself be interrupted while its
            # coordinator task is still using the StringIO.  Cancel work that
            # has not started, then retire every still-tracked future before
            # closing any of its buffers.  Futures removed from the deque have
            # already returned from result() and are therefore terminal.
            for _position, future, _buffer, _bytes, _spec in (
                pending_continuations
            ):
                try:
                    future.cancel()
                except BaseException as error:
                    if join_error is None:
                        join_error = error
            while True:
                try:
                    continuation_executor.shutdown(
                        wait=True,
                        cancel_futures=True,
                    )
                except BaseException as error:
                    if join_error is None:
                        join_error = error
                    continue
                break
            for _position, future, _buffer, _bytes, _spec in (
                pending_continuations
            ):
                while not future.done():
                    try:
                        future.result(timeout=0.05)
                    except FutureTimeoutError:
                        continue
                    except BaseException as error:
                        # A terminal exception belongs to the worker and is
                        # already ordered behind the active pipeline error.
                        # Only preserve interruptions of this cleanup wait.
                        if not future.done() and join_error is None:
                            join_error = error
            while pending_continuations:
                _position, future, output_buffer, _bytes, _spec = (
                    pending_continuations.popleft()
                )
                output_buffer.close()
        discard_ready()
        if ready_byte_capacity is None:
            final_ready_count = ready.qsize()
            final_ready_bytes = 0
        else:
            (
                final_ready_count,
                final_ready_bytes,
                ready_high_water,
                byte_high_water,
            ) = ready.state()
        if final_ready_count or final_ready_bytes:
            final_queue_error = RuntimeError(
                "GPU ready queue retained items or bytes after producer join"
            )
        else:
            final_queue_error = None
        if gpu_metrics is not None:
            if ready_byte_capacity is not None:
                producer_state['ready_high_water'] = ready_high_water
                gpu_metrics.ready_queue_byte_high_water = max(
                    gpu_metrics.ready_queue_byte_high_water,
                    byte_high_water,
                )
            gpu_metrics.ready_queue_final_count = final_ready_count
            gpu_metrics.ready_queue_final_bytes = final_ready_bytes
            gpu_metrics.ready_queue_high_water = max(
                gpu_metrics.ready_queue_high_water,
                producer_state['ready_high_water'],
            )
            gpu_metrics.producer_idle_count += producer_state['idle_count']
            gpu_metrics.producer_idle_seconds += producer_state['idle_seconds']
            gpu_metrics.pipeline_wall_seconds += (
                time.perf_counter() - pipeline_started
            )
        if join_error is not None and active_error is None:
            raise join_error
        if final_queue_error is not None and active_error is None:
            raise final_queue_error



def hmmsearch(protein_dict, hmms, threads, options, db_name=None,
              macsyfinder_dir=None, hmm_name_to_filename=None,
              all_sequences=None, gpu_sequence_batch=None,
              gpu_postfilter=None, gpu_profile_session=None,
              gpu_metrics=None, gpu_profile_overlap=True,
              gpu_request_tuning=None,
              gpu_filter_tail_simd=False,
              telemetry_collector=None, sparse_journal_v3=False,
              ga_pruning=False):
    if type(sparse_journal_v3) is not bool:
        raise TypeError("sparse_journal_v3 must be bool")
    if type(ga_pruning) is not bool:
        raise TypeError("ga_pruning must be bool")
    if type(gpu_filter_tail_simd) is not bool:
        raise TypeError("gpu_filter_tail_simd must be bool")
    if (
        gpu_request_tuning is not None
        and type(gpu_request_tuning) is not _GPURequestTuning
    ):
        raise TypeError(
            "gpu_request_tuning must be exactly _GPURequestTuning or None"
        )
    hmmsearch_kwargs = define_kwargs(options)
    streamed_gpu_profiles = (
        type(gpu_profile_session) is _PressedGPUProfileStream
    )

    if telemetry_collector is not None:
        from plan7_gpu.telemetry_report import TelemetryCollector

        if type(telemetry_collector) is not TelemetryCollector:
            raise TypeError(
                "telemetry_collector must be exactly TelemetryCollector"
            )

    if gpu_sequence_batch is not None and macsyfinder_dir is not None:
        raise GPUConfigurationError(
            "MacSyFinder output is unavailable for an explicitly GPU-mapped database"
        )
    if gpu_postfilter is None:
        gpu_postfilter = (
            gpu_postfilter_available() if gpu_sequence_batch is not None else False
        )
    elif type(gpu_postfilter) is not bool:
        raise TypeError("gpu_postfilter must be bool or None")
    if gpu_postfilter and gpu_sequence_batch is None:
        raise GPUConfigurationError(
            "exact GPU post-filter mode requires a GPU sequence batch"
        )
    if gpu_profile_session is not None:
        if gpu_sequence_batch is None or not gpu_postfilter:
            raise GPUConfigurationError(
                "GPU profile sessions require exact post-filter mode"
            )
        if gpu_profile_session.closed:
            raise GPUConfigurationError("GPU profile session is closed")
        if len(gpu_profile_session) != len(hmms):
            raise GPUConfigurationError(
                "GPU profile session does not cover the supplied profiles"
            )
        if streamed_gpu_profiles and hmms is not gpu_profile_session:
            raise GPUConfigurationError(
                "pressed GPU stream must be its supplied profile collection"
            )
    elif sparse_journal_v3:
        raise GPUConfigurationError(
            "sparse journal v3 requires an explicit GPU profile session"
        )
    if ga_pruning and not sparse_journal_v3:
        raise GPUConfigurationError("GA pruning requires sparse journal v3")
    if gpu_filter_tail_simd and not sparse_journal_v3:
        raise GPUConfigurationError(
            "filter-tail SIMD requires sparse journal v3"
        )
    if gpu_metrics is not None and not isinstance(gpu_metrics, GPUOverlapMetrics):
        raise TypeError("gpu_metrics must be GPUOverlapMetrics or None")
    if type(gpu_profile_overlap) is not bool:
        raise TypeError("gpu_profile_overlap must be bool")
    profile_overlap_enabled = False
    profile_domain_continuation = False
    continuation_threads = threads
    if gpu_profile_session is not None:
        (
            profile_overlap_enabled,
            producer_slots,
            continuation_threads,
        ) = gpu_profile_worker_allocation(threads, gpu_profile_overlap)
        statistics = gpu_profile_session.statistics
        selection_workers = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        if selection_workers:
            raise GPUConfigurationError(
                "GPU profile sessions require zero persistent selection "
                "workers"
            )
        profile_domain_continuation = gpu_profile_domain_available()
        if gpu_metrics is not None:
            gpu_metrics.requested_thread_count = threads
            gpu_metrics.producer_slot_count = producer_slots
            gpu_metrics.continuation_worker_count = continuation_threads
            gpu_metrics.profile_overlap_enabled = profile_overlap_enabled
    if sparse_journal_v3:
        if not profile_domain_continuation:
            raise GPUConfigurationError(
                "sparse journal v3 requires the fused domain-continuation path"
            )
        if not gpu_profile_sparse_journal_v3_available():
            raise GPUConfigurationError(
                "installed plan7_gpu does not support sparse journal v3"
            )
        if hmmsearch_kwargs.get('bias_filter', True) is not True:
            raise GPUConfigurationError(
                "sparse journal v3 requires fused domain continuation with "
                "bias filtering"
            )
    if ga_pruning:
        if not gpu_profile_ga_pruning_available():
            raise GPUConfigurationError(
                "installed plan7_gpu does not support exact GA pruning"
            )
        if hmmsearch_kwargs.get('bit_cutoffs') != 'gathering':
            raise GPUConfigurationError(
                "GA pruning requires gathering bit cutoffs"
            )
    if telemetry_collector is not None:
        if gpu_profile_session is None:
            raise GPUConfigurationError(
                "route telemetry requires an explicit GPU profile session"
            )
        if not profile_domain_continuation:
            raise GPUConfigurationError(
                "route telemetry requires the fused domain-continuation path"
            )
        telemetry_collector.bind_expected_profiles(range(len(hmms)))
    profile_scheduler_mode = gpu_profile_scheduler_mode(
        gpu_profile_session, profile_overlap_enabled
    )
    if streamed_gpu_profiles:
        if profile_scheduler_mode != 'bounded-ready-queue':
            raise GPUConfigurationError(
                "pressed GPU profile streaming requires the bounded "
                "ready-queue scheduler"
            )
        if set(hmmsearch_kwargs) != {'E'}:
            raise GPUConfigurationError(
                "pressed GPU profile streaming requires an E-value-only search"
            )
        if telemetry_collector is not None or ga_pruning:
            raise GPUConfigurationError(
                "pressed GPU profile streaming does not support telemetry or "
                "GA pruning"
            )
    ready_queue_configuration = None
    if profile_scheduler_mode == 'bounded-ready-queue':
        # Parse the experimental contract before creating output directories or
        # files.  Invalid depth/byte requests therefore fail without partial
        # search output.
        ready_queue_configuration = gpu_ready_queue_configuration()
    elif gpu_profile_session is not None and (
        GPU_READY_QUEUE_DEPTH_ENV in os.environ
        or GPU_READY_QUEUE_BYTES_ENV in os.environ
    ):
        raise GPUConfigurationError(
            f"{GPU_READY_QUEUE_DEPTH_ENV} and {GPU_READY_QUEUE_BYTES_ENV} "
            "require the active bounded-ready-queue scheduler"
        )
    automatic_gpu_tuning = bool(
        gpu_request_tuning is not None
        and gpu_request_tuning.automatic
    )
    continuation_tuning = _gpu_automatic_continuation_tuning(
        gpu_request_tuning, gpu_filter_tail_simd
    )
    continuation_pool_enabled = _gpu_continuation_pool_enabled(
        continuation_tuning
    )
    continuation_window = _gpu_continuation_window(
        GPU_PRODUCTION_CONTINUATION_WINDOW if continuation_tuning else 1
    )
    if continuation_pool_enabled and gpu_profile_session is None:
        raise GPUConfigurationError(
            f"{GPU_CONTINUATION_POOL_ENV}=1 requires a GPU profile session"
        )
    if continuation_window > 1:
        if not continuation_pool_enabled or gpu_profile_session is None:
            raise GPUConfigurationError(
                f"{GPU_CONTINUATION_WINDOW_ENV}>1 requires "
                f"{GPU_CONTINUATION_POOL_ENV}=1 and a GPU profile session"
            )
        if profile_scheduler_mode != 'bounded-ready-queue':
            raise GPUConfigurationError(
                f"{GPU_CONTINUATION_WINDOW_ENV}>1 requires the bounded "
                "ready-queue scheduler"
            )
        if telemetry_collector is not None:
            raise GPUConfigurationError(
                f"{GPU_CONTINUATION_WINDOW_ENV}>1 does not support route "
                "telemetry"
            )
    continuation_threads = _gpu_continuation_worker_count(
        continuation_threads,
        gpu_profile_session,
        continuation_pool_enabled,
    )
    if gpu_metrics is not None:
        gpu_metrics.continuation_pool_enabled = continuation_pool_enabled
        gpu_metrics.continuation_window = continuation_window
        if gpu_profile_session is not None:
            gpu_metrics.continuation_worker_count = continuation_threads

    # Always write to temp files — bulk mode is faster and avoids
    # keeping huge result lists in memory.  The per-genome loop is
    # only needed when MacSyFinder output requires per-genome provenance.
    tmp_dir = os.path.join(options['outdir'], 'tmp_results')
    os.makedirs(tmp_dir, exist_ok=True)

    def cutoff_available(query, cutoff):
        if gpu_sequence_batch is not None:
            # PressedProfilePair.cutoffs is the immutable snapshot captured
            # while the manifest-authenticated pressed streams were pinned.
            return getattr(query.cutoffs, cutoff) is not None
        return getattr(query.cutoffs, f"{cutoff}_available")()

    def get_best_cutoff(query):
        if options['cascade']:
            cutoff_order = [
                hmmsearch_kwargs.get('preferred_cutoff', 'trusted'),
                'trusted', 'gathering', 'noise'
            ]
            for cutoff in cutoff_order:
                if cutoff_available(query, cutoff):
                    return cutoff
        elif 'bit_cutoffs' in hmmsearch_kwargs:
            if cutoff_available(query, hmmsearch_kwargs['bit_cutoffs']):
                return hmmsearch_kwargs['bit_cutoffs']
        return None

    if isinstance(hmms, _PressedHMMStream):
        if (gpu_sequence_batch is not None or gpu_profile_session is not None
                or macsyfinder_dir is not None or options['cascade']):
            raise GPUConfigurationError(
                "streamed pressed profiles require ordinary bulk CPU search "
                "with a fixed threshold policy"
            )
        if all_sequences is None:
            all_sequences = []
            for sequences in protein_dict.values():
                all_sequences.extend(sequences)

        kwargs = hmmsearch_kwargs.copy()
        kwargs.pop('preferred_cutoff', None)
        requested_cutoff = kwargs.get('bit_cutoffs')
        if (requested_cutoff is not None
                and not hmms.all_have_cutoff(requested_cutoff)):
            # The eager implementation globally groups mixed cutoff
            # availability. Re-enter it unchanged so profile/output and
            # failure ordering stay authoritative for that uncommon case.
            print(
                "  Pressed profile stream has mixed cutoff availability; "
                "using the ordinary eager CPU path"
            )
            with astra_pyhmmer.plan7.HMMFile(hmms.pressed_base) as hmm_file:
                eager_hmms = list(hmm_file)
            return hmmsearch(
                protein_dict,
                eager_hmms,
                threads,
                options,
                db_name=db_name,
                all_sequences=all_sequences,
            )
        out_file = os.path.join(tmp_dir, "bulk_results.tsv")
        print(
            f"Bulk search: {len(all_sequences)} sequences × streamed pressed "
            "HMMs (bounded profile residency)"
        )
        with open(out_file, 'w') as fh:
            fh.write(
                "sequence_id\thmm_name\tbitscore\tevalue\tc_evalue\ti_evalue\t"
                "env_from\tenv_to\tdom_bitscore\tali_from\tali_to\thmm_from\thmm_to\n"
            )
            profile_chunks = hmms.chunks(HMM_CHUNK_SIZE)
            try:
                for chunk_index, hmm_chunk in enumerate(profile_chunks, 1):
                    print(
                        f"  Chunk {chunk_index} ({len(hmm_chunk)} HMMs)...",
                        end="",
                        flush=True,
                    )
                    hit_iterator = astra_pyhmmer.hmmsearch(
                        hmm_chunk, all_sequences, cpus=threads, **kwargs
                    )
                    hits = None
                    try:
                        for hits in hit_iterator:
                            process_hits_to_file(hits, fh)
                    finally:
                        close = getattr(hit_iterator, "close", None)
                        if close is not None:
                            close()
                        hit_iterator = None
                        hits = None
                        hmm_chunk = None
                    gc.collect()
                    print(" done")
            finally:
                profile_chunks.close()
        gc.collect()
        return tmp_dir

    if streamed_gpu_profiles:
        # The private stream is admitted only for a single E-value policy, so
        # grouping cannot change either thresholds or canonical model order.
        group_kwargs_list = None
    else:
        # Pre-compute HMM groups ONCE — grouping depends only on HMM cutoff
        # availability, not on per-genome data.  Previously this was inside the
        # per-genome loop, wasting len(hmms) * len(protein_dict) iterations.
        hmm_groups = {}
        for hmm_index, hmm in enumerate(hmms):
            best_cutoff = get_best_cutoff(hmm)
            hmm_groups.setdefault(best_cutoff, []).append((hmm_index, hmm))

        # Build per-group kwargs once (avoids re-copying per genome)
        group_kwargs_list = []
        for cutoff, indexed_hmm_group in hmm_groups.items():
            kwargs = hmmsearch_kwargs.copy()
            if cutoff:
                kwargs['bit_cutoffs'] = cutoff
            else:
                # No bitscore threshold available for these HMMs.
                # In cascade mode, fall back to E-value 1e-15 (the intended
                # cascade behavior) instead of astra_pyhmmer's permissive default (10.0).
                if 'bit_cutoffs' in kwargs:
                    del kwargs['bit_cutoffs']
                if options['cascade']:
                    kwargs.setdefault('E', 1e-15)

            # Remove internal-only keys before passing to astra_pyhmmer
            kwargs.pop('preferred_cutoff', None)
            group_kwargs_list.append((
                [hmm for _, hmm in indexed_hmm_group],
                tuple(index for index, _ in indexed_hmm_group),
                kwargs,
            ))

    # For large datasets without MacSyFinder output, flatten all sequences
    # and search once against the full pool.  This turns N_genomes * N_chunks
    # astra_pyhmmer.hmmsearch() calls into just N_chunks calls — e.g. 54 instead of
    # 192,456 for KOFAM on DPANN (3,564 genomes × 54 chunks).
    bulk_mode = not macsyfinder_dir

    HEADER = ("sequence_id\thmm_name\tbitscore\tevalue\tc_evalue\ti_evalue\t"
              "env_from\tenv_to\tdom_bitscore\tali_from\tali_to\thmm_from\thmm_to\n")

    if bulk_mode:
        # Use pre-flattened list if provided, otherwise flatten now
        if all_sequences is None:
            all_sequences = []
            for sequences in protein_dict.values():
                all_sequences.extend(sequences)
        grouped_count = (
            len(hmms)
            if streamed_gpu_profiles
            else sum(len(g) for g, _, _ in group_kwargs_list)
        )
        print(f"Bulk search: {len(all_sequences)} sequences × {len(hmms)} HMMs "
              f"({grouped_count} grouped)")

        hmm_chunk_size = HMM_CHUNK_SIZE
        if gpu_sequence_batch is not None:
            hmm_chunk_size = gpu_hmm_chunk_size(len(all_sequences))

        # Single output file — keep handle open across all chunks
        out_file = os.path.join(tmp_dir, "bulk_results.tsv")
        total_chunks = (
            (len(hmms) + hmm_chunk_size - 1) // hmm_chunk_size
            if streamed_gpu_profiles
            else sum(
                (len(g) + hmm_chunk_size - 1) // hmm_chunk_size
                for g, _, _ in group_kwargs_list
            )
        )
        chunk_idx = 0
        with open(out_file, 'w') as fh:
            fh.write(HEADER)
            if streamed_gpu_profiles:
                chunks = gpu_profile_session.chunk_specs(
                    hmm_chunk_size,
                    hmmsearch_kwargs.copy(),
                )
                if len(chunks) != total_chunks:
                    raise RuntimeError(
                        "pressed GPU profile stream chunk count changed"
                    )
            else:
                chunks = []
                for hmm_group, group_indices, kwargs in group_kwargs_list:
                    for chunk_start in range(0, len(hmm_group), hmm_chunk_size):
                        hmm_chunk = hmm_group[
                            chunk_start:chunk_start + hmm_chunk_size
                        ]
                        chunk_indices = group_indices[
                            chunk_start:chunk_start + hmm_chunk_size
                        ]
                        chunk_idx += 1
                        chunks.append((
                            chunk_idx,
                            hmm_chunk,
                            chunk_indices,
                            kwargs,
                        ))

            if gpu_profile_session is not None:
                profile_run_options = {}
                if telemetry_collector is not None:
                    profile_run_options['telemetry_collector'] = (
                        telemetry_collector
                    )
                if sparse_journal_v3:
                    profile_run_options['sparse_journal_v3'] = True
                if ga_pruning:
                    profile_run_options['ga_pruning'] = True
                release_configuration = (
                    _configure_gpu_request_page_release(
                        gpu_request_tuning, gpu_filter_tail_simd
                    )
                )
                continuation_pools = None
                try:
                    continuation_pools = _new_gpu_continuation_pools(
                        chunks,
                        continuation_threads,
                        continuation_pool_enabled,
                        continuation_window,
                        task_policy=(
                            'sharded' if continuation_tuning else None
                        ),
                        shard_trigger=(
                            GPU_PRODUCTION_SHARD_TRIGGER
                            if continuation_tuning else None
                        ),
                        pipeline_madvise_work_hint=(
                            GPU_PRODUCTION_PIPELINE_MADVISE_WORK_HINT
                            if automatic_gpu_tuning else None
                        ),
                    )
                    profile_run_options['continuation_pools'] = (
                        continuation_pools
                    )
                    if profile_scheduler_mode == 'bounded-ready-queue':
                        _run_gpu_profile_pipeline(
                            chunks,
                            gpu_profile_session,
                            gpu_sequence_batch,
                            continuation_threads,
                            fh,
                            gpu_metrics,
                            profile_domain_continuation,
                            ready_queue_configuration,
                            continuation_window=continuation_window,
                            **profile_run_options,
                        )
                    elif profile_scheduler_mode == 'single-prefetch':
                        _run_gpu_profile_single_prefetch(
                            chunks,
                            gpu_profile_session,
                            gpu_sequence_batch,
                            continuation_threads,
                            fh,
                            gpu_metrics,
                            profile_domain_continuation,
                            **profile_run_options,
                        )
                    else:
                        _run_gpu_profile_serial(
                            chunks,
                            gpu_profile_session,
                            gpu_sequence_batch,
                            continuation_threads,
                            fh,
                            gpu_metrics,
                            profile_domain_continuation,
                            **profile_run_options,
                        )
                finally:
                    try:
                        _close_gpu_continuation_pools(
                            continuation_pools, gpu_metrics
                        )
                    finally:
                        _restore_gpu_request_page_release(
                            release_configuration
                        )
            else:
                for chunk_index, hmm_chunk, _, kwargs in chunks:
                    print(f"  Chunk {chunk_index}/{total_chunks} "
                          f"({len(hmm_chunk)} HMMs)...", end="", flush=True)
                    if gpu_sequence_batch is None:
                        hit_iterator = astra_pyhmmer.hmmsearch(
                            hmm_chunk, all_sequences, cpus=threads, **kwargs
                        )
                        rendered_rows_type = None
                    else:
                        gpu_hmmsearch = _gpu_hmmsearch_entrypoint()
                        _, rendered_rows_type = _gpu_astra_tsv_capability()
                        if gpu_postfilter:
                            hit_iterator = gpu_hmmsearch(
                                hmm_chunk, gpu_sequence_batch, cpus=threads,
                                postfilter=True, **kwargs
                            )
                        else:
                            hit_iterator = gpu_hmmsearch(
                                hmm_chunk, gpu_sequence_batch, cpus=threads, **kwargs
                            )
                    try:
                        for hits in hit_iterator:
                            if (
                                rendered_rows_type is not None
                                and type(hits) is rendered_rows_type
                            ):
                                fh.write(hits.rows)
                                if gpu_metrics is not None:
                                    gpu_metrics.tsv_worker_rendered_profile_count += 1
                                    gpu_metrics.tsv_worker_rendered_row_count += (
                                        hits.row_count
                                    )
                                    gpu_metrics.tsv_worker_rendered_bytes += (
                                        hits.byte_count
                                    )
                            else:
                                if (
                                    gpu_sequence_batch is not None
                                    and gpu_metrics is not None
                                ):
                                    gpu_metrics.tsv_consumer_fallback_profile_count += 1
                                process_hits_to_file(hits, fh)
                    except BaseException:
                        if gpu_sequence_batch is not None:
                            close = getattr(hit_iterator, "close", None)
                            if close is not None:
                                try:
                                    close()
                                except BaseException:
                                    pass
                        raise
                    finally:
                        if gpu_sequence_batch is not None:
                            hit_iterator = None
                            hits = None
                    if gpu_sequence_batch is None:
                        gc.collect()
                    print(" done")

        gc.collect()

    else:
        # Per-genome loop — only used when MacSyFinder output is needed
        # (requires per-genome provenance tracking).
        for fasta_file, sequences in tqdm(protein_dict.items()):
            safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.basename(fasta_file))
            tmp_file = os.path.join(tmp_dir, f"{safe_filename}_results.tsv")
            with open(tmp_file, 'w') as fh:
                fh.write(HEADER)
                for hmm_group, _, kwargs in group_kwargs_list:
                    for chunk_start in range(0, len(hmm_group), HMM_CHUNK_SIZE):
                        hmm_chunk = hmm_group[chunk_start:chunk_start + HMM_CHUNK_SIZE]
                        for hits in astra_pyhmmer.hmmsearch(hmm_chunk, sequences,
                                                      cpus=threads, **kwargs):
                            process_hits_to_file(hits, fh)
                            write_macsyfinder_hit(hits, macsyfinder_dir, hmm_name_to_filename)
            gc.collect()

    return tmp_dir


def process_hits_to_file(hits, fh):
    """Write hits to an already-open file handle *fh*."""
    global _native_tsv_rows
    if _native_tsv_rows is _NATIVE_TSV_ROWS_UNRESOLVED:
        try:
            from plan7_gpu import _pipeline
        except ImportError:
            _native_tsv_rows = None
        else:
            renderer = getattr(_pipeline, '_astra_tsv_rows_bound', None)
            unsupported = getattr(
                _pipeline, '_AstraTSVRendererUnsupported', None
            )
            if (
                getattr(_pipeline, '_ASTRA_TSV_RENDERER_ABI', None)
                == _ASTRA_TSV_RENDERER_ABI
                and callable(renderer)
                and isinstance(unsupported, type)
                and issubclass(unsupported, ValueError)
            ):
                _native_tsv_rows = (renderer, unsupported)
            else:
                _native_tsv_rows = None
    if _native_tsv_rows is not None:
        renderer, unsupported = _native_tsv_rows
        try:
            rows = renderer(hits)
        except unsupported:
            # The native ABI deliberately covers only ordinary, sort-key
            # ordered protein searches.  Preserve the public wrapper path for
            # every other TopHits shape.
            pass
        else:
            fh.write(rows)
            return

    cog = hits.query.name
    for hit in hits:
        if hit.included:
            hit_name = hit.name
            full_bitscore = hit.score
            full_evalue = hit.evalue
            for domain in hit.domains.reported:
                aln = domain.alignment
                ali_from = aln.target_from if aln else domain.env_from
                ali_to = aln.target_to if aln else domain.env_to
                hmm_from = aln.hmm_from if aln else ""
                hmm_to = aln.hmm_to if aln else ""
                fh.write(f"{hit_name}\t{cog}\t{full_bitscore:.2f}\t{full_evalue:.2e}\t{domain.c_evalue:.2e}\t"
                         f"{domain.i_evalue:.2e}\t{domain.env_from}\t{domain.env_to}\t{domain.score:.2f}\t"
                         f"{ali_from}\t{ali_to}\t{hmm_from}\t{hmm_to}\n")


def extract_sequences_from_tmp(tmp_dir, protein_dict, outdir):
    """Read hit IDs from temp result files and extract sequences from memory."""
    # Collect hit IDs per HMM from all result files (no pandas needed)
    hit_ids_by_hmm = {}
    for filename in os.listdir(tmp_dir):
        if filename.endswith('_results.tsv'):
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.split('\t', 3)  # only need first two columns
                    seq_id, hmm_name = parts[0], parts[1]
                    hit_ids_by_hmm.setdefault(hmm_name, set()).add(seq_id)

    if hit_ids_by_hmm:
        extract_sequences(hit_ids_by_hmm, protein_dict, outdir)
        print(f"Extracted sequences for {len(hit_ids_by_hmm)} HMMs → {outdir}/fastas/")

def cleanup_temp_files(temp_dir):
    shutil.rmtree(temp_dir)
    print(f"Temporary files removed from {temp_dir}")



def parse_single_hmm(hmm_path):
    """Single-file parser for fallback when no pressed DB exists."""
    with astra_pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
        return hmm_file.read()


class _PressedHMMStream:
    """Re-openable, bounded-memory view of a pressed HMM database.

    This is deliberately private and used only by the eligible installed-DB
    CPU bulk path.  Each iteration owns its ``HMMFile`` and releases every
    completed chunk before reading the next one; no public Aksha/PyHMMER API
    changes.
    """

    __slots__ = (
        "pressed_base",
        "_cutoff_availability",
        "_large_profile_audit",
    )

    def __init__(self, pressed_base):
        self.pressed_base = os.fspath(pressed_base)
        self._cutoff_availability = {}
        self._large_profile_audit = None

    def chunks(self, chunk_size):
        if (isinstance(chunk_size, bool)
                or not isinstance(chunk_size, int)
                or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer")
        with astra_pyhmmer.plan7.HMMFile(self.pressed_base) as hmm_file:
            chunk = []
            for hmm in hmm_file:
                chunk.append(hmm)
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

    def all_have_cutoff(self, cutoff):
        if cutoff in self._cutoff_availability:
            return self._cutoff_availability[cutoff]
        method_name = f"{cutoff}_available"
        with astra_pyhmmer.plan7.HMMFile(self.pressed_base) as hmm_file:
            for hmm in hmm_file:
                method = getattr(hmm.cutoffs, method_name, None)
                if method is None or not method():
                    self._cutoff_availability[cutoff] = False
                    return False
        self._cutoff_availability[cutoff] = True
        return True

    def large_enough_for_streaming(self):
        """Return ``(large, inspected)`` for the current chunk threshold."""
        threshold = HMM_CHUNK_SIZE
        cached = self._large_profile_audit
        if cached is not None and cached[0] == threshold:
            return cached[1], cached[2]
        inspected = 0
        with astra_pyhmmer.plan7.HMMFile(self.pressed_base) as hmm_file:
            for _ in hmm_file:
                inspected += 1
                if inspected > threshold:
                    result = (threshold, True, inspected)
                    self._large_profile_audit = result
                    return True, inspected
        result = (threshold, False, inspected)
        self._large_profile_audit = result
        return False, inspected


class _PressedCPUStreamDecision(NamedTuple):
    """Machine-readable telemetry for installed pressed-profile routing."""

    policy: str
    enabled: bool
    reason: str
    cutoff: object
    profiles_inspected: int


def _stream_pressed_cpu_policy():
    value = os.environ.get(CPU_STREAM_PRESSED_ENV, CPU_STREAM_PRESSED_AUTO)
    if value not in (CPU_STREAM_PRESSED_AUTO, "0", "1"):
        raise ValueError(
            f"{CPU_STREAM_PRESSED_ENV} must be auto, 0, or 1"
        )
    return value


def _select_pressed_cpu_stream(
        pressed_base, options, *, installed=True, gpu=False,
        macsyfinder=False):
    """Select the proven bounded CPU source or explain an eager fallback.

    ``ASTRA_CPU_STREAM_PRESSED=1`` requests the optimization but never forces
    an unproven semantic path.  Both explicit and automatic selection retain
    the eager implementation for custom/unpressed inputs, GPU execution,
    cascade thresholds, MacSyFinder output, mixed cutoff availability, and
    databases no larger than one ordinary profile chunk.
    """
    if not installed:
        return None, _PressedCPUStreamDecision(
            CPU_STREAM_PRESSED_AUTO, False, "custom-source", None, 0
        )
    policy = _stream_pressed_cpu_policy()
    if pressed_base is None:
        return None, _PressedCPUStreamDecision(
            policy, False, "unpressed-source", None, 0
        )
    if policy == "0":
        return None, _PressedCPUStreamDecision(
            policy, False, "disabled-by-environment", None, 0
        )
    if gpu:
        return None, _PressedCPUStreamDecision(
            policy, False, "gpu-search", None, 0
        )
    if macsyfinder:
        return None, _PressedCPUStreamDecision(
            policy, False, "macsyfinder-output", None, 0
        )
    if options.get("cascade"):
        return None, _PressedCPUStreamDecision(
            policy, False, "cascade-thresholds", None, 0
        )

    fixed_cutoffs = tuple(
        cutoff
        for selected, cutoff in (
            (options.get("cut_ga"), "gathering"),
            (options.get("cut_nc"), "noise"),
            (options.get("cut_tc"), "trusted"),
        )
        if selected
    )
    if len(fixed_cutoffs) > 1:
        return None, _PressedCPUStreamDecision(
            policy, False, "multiple-cutoff-families", None, 0
        )
    cutoff = fixed_cutoffs[0] if fixed_cutoffs else None
    stream = _PressedHMMStream(pressed_base)
    large, inspected = stream.large_enough_for_streaming()
    if not large:
        return None, _PressedCPUStreamDecision(
            policy, False, "small-profile-database", cutoff, inspected
        )
    if cutoff is not None and not stream.all_have_cutoff(cutoff):
        return None, _PressedCPUStreamDecision(
            policy,
            False,
            f"mixed-{cutoff}-availability",
            cutoff,
            inspected,
        )
    return stream, _PressedCPUStreamDecision(
        policy,
        True,
        "eligible-fixed-profile-cutoff" if cutoff else "eligible-fixed-global",
        cutoff,
        inspected,
    )


def _report_pressed_cpu_stream_decision(decision):
    status = "enabled" if decision.enabled else "eager-fallback"
    cutoff = decision.cutoff if decision.cutoff is not None else "global"
    message = (
        "CPU pressed-profile stream "
        f"policy={decision.policy} status={status} reason={decision.reason} "
        f"cutoff={cutoff} profiles_inspected={decision.profiles_inspected}"
    )
    print(f"  {message}")
    logging.info(message)


def _stream_pressed_cpu_enabled():
    """Compatibility helper reporting whether policy permits consideration."""
    return _stream_pressed_cpu_policy() != "0"

def _find_pressed_db(db_dir):
    """Check if a pressed HMM database exists in *db_dir*.

    Returns the base path (without extension) if all four pressed files
    (``.h3m``, ``.h3i``, ``.h3f``, ``.h3p``) exist, otherwise ``None``.
    """
    db_name = os.path.basename(db_dir.rstrip('/'))
    pressed_base = os.path.join(db_dir, db_name)
    if all(os.path.exists(f"{pressed_base}.{ext}") for ext in ('h3m', 'h3i', 'h3f', 'h3p')):
        return pressed_base
    return None


def parse_hmms(hmm_in):
    #Checks first whether HMMs are provided as a single file or as a directory.

    hmms = []  # Initialize an empty list to store parsed HMMs
    # Mapping from HMM internal NAME → filename stem, for MacSyFinder compat.
    # Many HMM databases (DefenseFinder, TXSScan) have HMM files whose internal
    # NAME field differs from the filename. MacSyFinder expects filenames, so
    # we track the mapping here and pass it through to write_macsyfinder_hit().
    hmm_name_to_filename = {}
    print("Parsing HMMs...")
    t0 = time.perf_counter()

    # Check if hmm_in is a directory or a single file
    if os.path.isdir(hmm_in):
        if not os.listdir(hmm_in):
            print("hmm_in directory is empty.")
            logging.info('hmm_in directory is empty.')
            sys.exit(1)

        # Prefer pressed database if available (~50x faster than individual files)
        pressed_base = _find_pressed_db(hmm_in)
        if pressed_base:
            print(f"  Loading from pressed database: {pressed_base}")
            with astra_pyhmmer.plan7.HMMFile(pressed_base) as hmm_file:
                hmms = list(hmm_file)
            elapsed = time.perf_counter() - t0
            print(f"HMMs parsed: {len(hmms)} models in {elapsed:.1f}s (pressed DB)")
            return list(hmms), hmm_name_to_filename

        hmm_files = list(filter(lambda x: x.endswith(('.hmm', '.HMM')), os.listdir(hmm_in)))
        if len(hmm_files) == 0:
            print("No .hmm files found in directory.")
            logging.info('No .hmm files found in directory.')
            sys.exit(1)
        elif len(hmm_files) == 1:
            #Only one HMM file in input directory
            #Get full path to file
            hmm_path = os.path.join(hmm_in, hmm_files[0])
            with astra_pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
                #Works in case of single-model or multi-model HMM file
                hmms = list(hmm_file)

        else:
            hmm_paths = [os.path.join(hmm_in, hmm_file) for hmm_file in hmm_files]

            #I have tried!! Every possible method! To parallelize this!
            #It does not work. SINGLE THREADED IT IS!
            hmms = list(tqdm(map(parse_single_hmm, hmm_paths)))

            # Build internal NAME → filename stem mapping for MacSyFinder compat
            for hmm_file, hmm_obj in zip(hmm_files, hmms):
                filename_stem = hmm_file.rsplit('.hmm', 1)[0].rsplit('.HMM', 1)[0]
                internal_name = hmm_obj.name
                if internal_name != filename_stem:
                    hmm_name_to_filename[internal_name] = filename_stem

            if hmm_name_to_filename:
                print(f"  {len(hmm_name_to_filename)} HMMs have internal NAME != filename (will use filename for MacSyFinder)")

    elif os.path.isfile(hmm_in):
        if os.path.getsize(hmm_in) == 0:
            print("hmm_in file is empty.")
            logging.info('hmm_in file is empty.')
            sys.exit(1)
        # Parse the single HMM file; handles multi-model files
        with astra_pyhmmer.plan7.HMMFile(hmm_in) as hmm_file:
            hmms = list(hmm_file)
    else:
        print("Invalid HMM input.")
        logging.info("Invalid HMM input.")
        print("If you used pre-installed HMMs, check hmm_databases.json")
        logging.info("If you used pre-installed HMMs, check hmm_databases.json")
        print("Which is located in the databases directory.")
        logging.info("Which is located in the databases directory.")

        print("Thing that threw the error: {}".format(hmm_in))
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"HMMs parsed: {len(hmms)} models in {elapsed:.1f}s")

    return list(hmms), hmm_name_to_filename

def process_fasta(fasta_file):
    # Function to handle each file for parallelism
    with astra_pyhmmer.easel.SequenceFile(fasta_file, digital=True, alphabet=astra_pyhmmer.easel.Alphabet.amino()) as seq_file:
        sequences = seq_file.read_block()
    return fasta_file, sequences

def parse_protein_input(prot_in, threads):
    print("Parsing protein input sequences...")
    protein_dict = {}  # Initialize an empty dictionary to store parsed proteins
    
    # Check if prot_in is a directory or a single file
    if os.path.isdir(prot_in):
        if not os.listdir(prot_in):
            print("prot_in directory is empty.")
            logging.info("prot_in directory is empty.")
            sys.exit(1)

        # Initialize an empty dictionary to hold protein sequences
        protein_dict = {}

        fasta_paths = [os.path.join(prot_in, x) for x in os.listdir(prot_in)]

        # astra_pyhmmer sequence objects can't pickle (no ProcessPoolExecutor), but
        # the GIL is released during astra_pyhmmer's C-level I/O, so threads work.
        # For small file counts the overhead isn't worth it; threshold at 8.
        if len(fasta_paths) >= 8:
            n_workers = min(threads, len(fasta_paths))
            print(f"  Loading {len(fasta_paths)} files with {n_workers} threads...")
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(tqdm(executor.map(process_fasta, fasta_paths),
                                    total=len(fasta_paths)))
        else:
            results = list(map(process_fasta, tqdm(fasta_paths)))

        # Populate the protein_dict
        for fasta_path, sequences in results:
            protein_dict[fasta_path] = sequences
    elif os.path.isfile(prot_in):
        if os.path.getsize(prot_in) == 0:
            print("prot_in file is empty.")
            logging.info("prot_in file is empty.")
            sys.exit(1)
        # Parse the single protein FASTA file
        with astra_pyhmmer.easel.SequenceFile(prot_in, digital=True) as seq_file:
            sequences = seq_file.read_block()
        protein_dict[prot_in] = sequences
    else:
        print("Invalid input for prot_in.")
        logging.info("Invalid input for prot_in.")
        sys.exit(1)
    
    return protein_dict

def define_kwargs(options):
    kwargs = {}
    
    if options['cascade']:
        # Cascade mode: per-HMM adaptive thresholds.  HMM grouping in
        # hmmsearch() assigns the best available bitscore cutoff to each
        # profile (preferred → trusted → gathering → noise), falling back
        # to E-value 1e-15 for profiles with no thresholds at all.
        # Do NOT set bit_cutoffs here — it is set per-group in hmmsearch().
        if options['cut_tc']:
            kwargs['preferred_cutoff'] = 'trusted'
        elif options['cut_ga']:
            kwargs['preferred_cutoff'] = 'gathering'
        elif options['cut_nc']:
            kwargs['preferred_cutoff'] = 'noise'
        else:
            kwargs['preferred_cutoff'] = 'trusted'
    elif options['cut_ga']:
        kwargs['bit_cutoffs'] = 'gathering'
    elif options['cut_nc']:
        kwargs['bit_cutoffs'] = 'noise'
    elif options['cut_tc']:
        kwargs['bit_cutoffs'] = 'trusted'


    #Numerical threshold parameters
    if options['bitscore'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['bitscore'], float):
            try:
                kwargs['T'] = float(options['bitscore'])
            except ValueError:
                print("Error: bitscore threshold must be a float or castable as a float.")
                logging.info("Error: bitscore threshold must be a float or castable as a float.")

    if options['domE'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['domE'], float):
            try:
                kwargs['domE'] = float(options['domE'])
            except ValueError:
                print("Error: domE must be a float or castable to float.")
                logging.info("Error: domE must be a float or castable to float.")

    if options['domT'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['domT'], float):
            try:
                kwargs['domT'] = float(options['domT'])
            except ValueError:
                print("Error: domT must be a float or castable to float.")
                logging.info("Error: domT must be a float or castable to float.")

    if options['incE'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incE'], float):
            try:
                kwargs['incE'] = float(options['incE'])
            except ValueError:
                print("Error: domT must be a float or castable to float.")
                logging.error("Error: domT must be a float or castable to float.")

    if options['incT'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incT'], float):
            try:
                kwargs['incT'] = float(options['incT'])
            except ValueError:
                print("Error: incT must be a float or castable to float.")
                logging.error("Error: incT must be a float or castable to float.")

    if options['incdomE'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incdomE'], float):
            try:
                kwargs['incdomE'] = float(options['incdomE'])
            except ValueError:
                print("Error: incdomE must be a float or castable to float.")
                logging.error("Error: incdomE must be a float or castable to float.")

    if options['incdomT'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incdomT'], float):
            try:
                kwargs['incdomT'] = float(options['incdomT'])
            except ValueError:
                print("Error: incdomT must be a float or castable to float.")
                logging.error("Error: incdomT must be a float or castable to float.")

    if options['evalue'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['evalue'], float):
            try:
                kwargs['E'] = float(options['evalue'])
            except ValueError:
                print("Error: evalue must be a float or castable to float.")
                logging.error("Error: evalue must be a float or castable to float.")

    return kwargs

def combine_results(tmp_dir, output_file):
    """Combine temp result files into a single output file.

    If the temp directory contains a single file (bulk mode), it is simply
    moved to *output_file* — no copying needed.  For multiple files
    (MacSyFinder per-genome mode) the files are streamed together.
    """
    tmp_files = sorted(
        f for f in os.listdir(tmp_dir) if f.endswith('_results.tsv')
    )
    if not tmp_files:
        print("No results found to combine.")
        return

    if len(tmp_files) == 1:
        # Bulk mode — single file, just move it
        src = os.path.join(tmp_dir, tmp_files[0])
        shutil.move(src, output_file)
        print(f"Results → {output_file}")
        return

    # Multiple files (MacSyFinder per-genome mode) — stream-combine
    total_rows = 0
    header_written = False

    with open(output_file, 'w') as out:
        for filename in tqdm(tmp_files, desc="Combining"):
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path) as inp:
                header = inp.readline()
                if not header_written:
                    out.write(header)
                    header_written = True
                for line in inp:
                    out.write(line)
                    total_rows += 1

    print(f"Combined {total_rows:,} hits from {len(tmp_files)} files → {output_file}")

def _write_macsyfinder_conf(macsyfinder_dir, prot_in):
    """Write a minimal macsyfinder.conf for --previous-run compatibility.

    If *prot_in* is a directory, concatenates all .faa files into a single
    FASTA inside *macsyfinder_dir* so MacSyFinder can index it.
    """
    conf_path = os.path.join(macsyfinder_dir, "macsyfinder.conf")
    abs_prot = os.path.abspath(prot_in)

    if os.path.isdir(abs_prot):
        faa_files = sorted(
            f for f in os.listdir(abs_prot)
            if f.endswith(('.faa', '.fa', '.fasta'))
        )
        if len(faa_files) == 1:
            sequence_db = os.path.join(abs_prot, faa_files[0])
        else:
            # Concatenate all FASTA files into one
            concat_path = os.path.join(macsyfinder_dir, "all_proteins.faa")
            with open(concat_path, "w") as out:
                for faa in faa_files:
                    with open(os.path.join(abs_prot, faa)) as inp:
                        out.write(inp.read())
            sequence_db = concat_path
    else:
        sequence_db = abs_prot

    with open(conf_path, "w") as fh:
        fh.write("[base]\n")
        fh.write(f"sequence_db = {sequence_db}\n")
        fh.write("db_type = ordered_replicon\n")
        fh.write("hmmer = hmmsearch\n\n")
        fh.write("[hmmer]\n")
        fh.write("e_value_search = 0.1\n")
    print(f"MacSyFinder config written to {conf_path}")


def main(args, *, gpu_profile_session_cache=None):
    t1 = time.time()
    hmm_in = args.hmm_in
    prot_in = args.prot_in
    outdir = args.outdir
    gpu_manifests = parse_gpu_manifest_mappings(
        getattr(args, 'gpu_manifest', ())
    )
    gpu_parsed_json = None
    gpu_installed_hmm_names = None
    log_file_path = os.path.join(outdir, 'astra_search_log.txt')

    # --- Ribosomal-protein marker mode ---------------------------------
    # ``--16rp``/``--15rp`` are stored under non-identifier dests, so read
    # them via getattr.  RP mode points hmm_in at the RP16 marker set and
    # triggers genome-aware extraction after the search (see end of main).
    rp16_mode = getattr(args, '16rp', False)
    rp15_mode = getattr(args, '15rp', False)
    if rp15_mode:
        msg = ("--15rp (archaea-only 15-marker set) is not yet implemented. "
               "Use --16rp, which covers Bacteria and Archaea.")
        print(msg)
        logging.error(msg)
        sys.exit(1)
    synteny_threshold = getattr(args, 'synteny', None)
    if rp16_mode:
        if hmm_in is not None or args.installed_hmms is not None:
            print("--16rp uses its own marker set; ignoring "
                  "--hmm_in/--installed_hmms.")
        try:
            hmm_in = rp16_module.rp16_hmm_dir()
        except FileNotFoundError as exc:
            print(exc)
            logging.error(str(exc))
            sys.exit(1)
        args.installed_hmms = None
        # The markers all carry GA cutoffs; default to them when the
        # user did not request any explicit threshold.
        if not (args.cut_ga or args.cut_nc or args.cut_tc or args.cascade
                or args.evalue or args.bitscore):
            print("--16rp: no threshold specified, defaulting to --cut_ga.")
            args.cut_ga = True
    elif synteny_threshold is not None:
        print("--synteny is only used with --16rp/--15rp; ignoring.")
        synteny_threshold = None

    if gpu_manifests:
        if args.installed_hmms is None:
            gpu_parsed_json = {'db_urls': []}
            gpu_installed_hmm_names = []
        else:
            gpu_parsed_json = initialize.load_config()
            gpu_installed_hmm_names = _resolve_installed_hmm_names(
                args.installed_hmms, gpu_parsed_json
            )
        validate_gpu_configuration(
            gpu_manifests,
            gpu_installed_hmm_names,
            gpu_parsed_json,
            args.threads,
            getattr(args, 'write_macsyfinder', False),
        )

    hmmsearch_options = {
        "cascade": args.cascade,
        "cut_ga": args.cut_ga,
        "cut_nc": args.cut_nc,
        "cut_tc": args.cut_tc,
        "evalue": args.evalue,
        "bitscore": args.bitscore,
        "domE": args.domE,
        "domT": args.domT,
        "incE": args.incE,
        "incT": args.incT,
        "incdomE": args.incdomE,
        "incdomT": args.incdomT,
        "outdir": outdir,
        "meta": args.meta
    }

    if hmm_in is None and args.installed_hmms is None:
        error_out = "Either a user-provided or pre-installed HMM database must be specified."
        print(error_out)
        logging.error(error_out)
        sys.exit(1)

    protein_dict = parse_protein_input(prot_in, args.threads)

    # Pre-flatten sequences once for all database searches (avoids
    # re-allocating for each DB in multi-DB runs).
    all_sequences = []
    for seqs in protein_dict.values():
        all_sequences.extend(seqs)
    print(f"Total sequences loaded: {len(all_sequences)}")

    # Free protein_dict if we don't need per-file provenance.
    # write_seqs needs it for sequence extraction; MacSyFinder needs it
    # for per-genome loop.  Otherwise it's dead weight.
    needs_protein_dict = (args.write_seqs or rp16_mode
                          or getattr(args, 'write_macsyfinder', False))
    if not needs_protein_dict:
        del protein_dict
        gc.collect()
        protein_dict = None  # keep the name bound for the code paths that check it

    gpu_databases = {}
    gpu_sequence_batch = None
    gpu_postfilter = False
    gpu_metrics_by_db = None
    gpu_request_tuning_by_db = {}
    gpu_filter_tail_simd_by_db = {}
    if gpu_manifests:
        # This check must happen before constructing a CUDA SequenceBatch or
        # creating any result path. With one HMM, a larger target set already
        # exceeds the bounded profile-by-target candidate matrix.
        gpu_hmm_chunk_size(len(all_sequences))
        if os.environ.get(GPU_TIMING_ENV) == '1':
            gpu_metrics_by_db = {
                db_name: GPUOverlapMetrics()
                for db_name in gpu_installed_hmm_names
                if db_name in gpu_manifests
            }
        gpu_databases, gpu_sequence_batch, gpu_postfilter = preflight_gpu_databases(
            gpu_manifests,
            gpu_installed_hmm_names,
            gpu_parsed_json,
            all_sequences,
            args.threads,
            gpu_metrics_by_db,
            gpu_profile_session_cache,
            search_options=hmmsearch_options,
            request_tuning_by_db=gpu_request_tuning_by_db,
            filter_tail_simd_by_db=gpu_filter_tail_simd_by_db,
        )

    try:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            if args.write_seqs:
                os.makedirs(os.path.join(outdir, 'fastas'))

        # Create temporary directory for results only after every explicit GPU
        # database and the shared target batch have completed preflight.
        tmp_results_dir = os.path.join(outdir, 'tmp_results')
        os.makedirs(tmp_results_dir, exist_ok=True)

        logging.basicConfig(filename=log_file_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    except BaseException:
        batch_close_failed = False
        if gpu_sequence_batch is not None:
            try:
                gpu_sequence_batch.close()
            except BaseException:
                batch_close_failed = True
        if batch_close_failed and gpu_profile_session_cache is not None:
            try:
                gpu_profile_session_cache.close()
            except BaseException:
                pass
        for _, _, session in gpu_databases.values():
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    pass
        raise

    # MacSyFinder-compatible output directory (per-HMM hmmsearch text files)
    macsyfinder_dir = None
    if getattr(args, 'write_macsyfinder', False):
        macsyfinder_dir = os.path.join(outdir, 'macsyfinder_compat')
        hmmer_results_dir = os.path.join(macsyfinder_dir, 'hmmer_results')
        # Clean previous run's files to avoid stale appends
        if os.path.isdir(hmmer_results_dir):
            shutil.rmtree(hmmer_results_dir)
        os.makedirs(hmmer_results_dir, exist_ok=True)
        print(f"MacSyFinder-compatible output enabled → {macsyfinder_dir}/")
        logging.info(f"MacSyFinder-compatible output enabled → {macsyfinder_dir}/")

    try:
        if hmm_in is not None:
            print("Searching with user-provided HMM(s)...")
            logging.info("Searching with user-provided HMM(s)...")
            user_hmms, user_name_map = parse_hmms(hmm_in)
            results = hmmsearch(protein_dict, user_hmms, args.threads, hmmsearch_options,
                                macsyfinder_dir=macsyfinder_dir, hmm_name_to_filename=user_name_map,
                                all_sequences=all_sequences)
            if args.write_seqs:
                extract_sequences_from_tmp(results, protein_dict, outdir)
            hits_tsv = os.path.join(outdir,
                                    'rp16_raw_hits.tsv' if rp16_mode else 'user_hmms_hits_df.tsv')
            combine_results(results, hits_tsv)
            if rp16_mode:
                rp16_module.process(hits_tsv, protein_dict, outdir, synteny_threshold)
            del user_hmms

        if args.installed_hmms is not None:
            installed_hmm_names = args.installed_hmms.split(',') if ',' in args.installed_hmms else [args.installed_hmms]
            print(f"Searching with pre-installed HMMs: {', '.join(installed_hmm_names)}")
            logging.info(f"Searching with pre-installed HMMs: {', '.join(installed_hmm_names)}")

            parsed_json = (
                gpu_parsed_json
                if gpu_parsed_json is not None
                else initialize.load_config()
            )

            if 'all_prot' in installed_hmm_names:
                installed_hmm_names = [db['name'] for db in parsed_json['db_urls'] if db['molecule_type'] == 'protein' and db['installed']]

            for hmm_db in installed_hmm_names:
                installed_hmm_in = next((item for item in parsed_json['db_urls'] if item["name"] == hmm_db), None)
                if installed_hmm_in is not None:
                    installation_dir = installed_hmm_in['installation_dir']
                    manifest_path = gpu_manifests.get(hmm_db)
                    if manifest_path is None:
                        pressed_base = _find_pressed_db(installation_dir)
                        db_hmms, stream_decision = (
                            _select_pressed_cpu_stream(
                                pressed_base,
                                hmmsearch_options,
                                installed=True,
                                gpu=False,
                                macsyfinder=macsyfinder_dir is not None,
                            )
                        )
                        _report_pressed_cpu_stream_decision(stream_decision)
                        if db_hmms is not None:
                            print(
                                "  Streaming pressed profiles with bounded "
                                f"residency: {pressed_base}"
                            )
                            db_name_map = {}
                        else:
                            db_hmms, db_name_map = parse_hmms(installation_dir)
                        tmp_dir = hmmsearch(
                            protein_dict, db_hmms, args.threads, hmmsearch_options,
                            hmm_db, macsyfinder_dir=macsyfinder_dir,
                            hmm_name_to_filename=db_name_map,
                            all_sequences=all_sequences,
                        )
                    else:
                        (
                            pressed_base,
                            db_hmms,
                            gpu_profile_session,
                        ) = gpu_databases[hmm_db]
                        try:
                            gpu_profile_overlap_requested = (
                                os.environ.get(GPU_SERIAL_ENV) != '1'
                            )
                            if gpu_profile_session is not None:
                                gpu_profile_overlap, _, _ = (
                                    gpu_profile_worker_allocation(
                                        args.threads,
                                        gpu_profile_overlap_requested,
                                    )
                                )
                                scheduling = (
                                    "overlap" if gpu_profile_overlap else "serial"
                                )
                                gpu_mode = (
                                    f"exact GPU-through-Forward {scheduling}"
                                )
                            elif gpu_postfilter:
                                gpu_profile_overlap = False
                                gpu_mode = "exact post-filter + Forward"
                            else:
                                gpu_profile_overlap = False
                                gpu_mode = "legacy SSV"
                            print(
                                f"  GPU search for {hmm_db} ({gpu_mode}): "
                                f"{pressed_base}"
                            )
                            logging.info(
                                f"GPU search for {hmm_db} ({gpu_mode}): "
                                f"{pressed_base}"
                            )
                            db_metrics = (
                                gpu_metrics_by_db[hmm_db]
                                if gpu_metrics_by_db is not None
                                else None
                            )
                            request_tuning = (
                                gpu_request_tuning_by_db.get(hmm_db)
                            )
                            filter_tail_simd = (
                                gpu_filter_tail_simd_by_db.get(
                                    hmm_db, False
                                )
                            )
                            tmp_dir = hmmsearch(
                                protein_dict, db_hmms, args.threads,
                                hmmsearch_options, hmm_db,
                                all_sequences=all_sequences,
                                gpu_sequence_batch=gpu_sequence_batch,
                                gpu_postfilter=gpu_postfilter,
                                gpu_profile_session=gpu_profile_session,
                                gpu_metrics=(
                                    db_metrics
                                    if gpu_profile_session is not None
                                    else None
                                ),
                                gpu_profile_overlap=gpu_profile_overlap,
                                gpu_request_tuning=request_tuning,
                                sparse_journal_v3=(
                                    _gpu_automatic_sparse_journal_v3(
                                        request_tuning, filter_tail_simd
                                    )
                                ),
                                gpu_filter_tail_simd=filter_tail_simd,
                            )
                            if (
                                db_metrics is not None
                                and gpu_profile_session is not None
                            ):
                                timing = db_metrics.snapshot()
                                timing_line = (
                                    f"  GPU-through-Forward {scheduling} timing: "
                                    f"phase-workers=build:"
                                    f"{timing['profile_build_worker_count']}, "
                                    f"search=selection:"
                                    f"{timing['profile_selection_worker_count']}"
                                    f"/producer:{timing['producer_slot_count']}"
                                    f"/continuation:"
                                    f"{timing['continuation_worker_count']}"
                                    f"/{timing['requested_thread_count']}, "
                                    f"preflight={timing['preflight_seconds']:.3f}s, "
                                    f"selection={timing['selection_seconds']:.3f}s, "
                                    f"generation={timing['generation_seconds']:.3f}s, "
                                    f"wait={timing['generation_wait_seconds']:.3f}s, "
                                    f"CPU/output={timing['continuation_seconds']:.3f}s, "
                                    f"measured-overlap={timing['overlap_seconds']:.3f}s"
                                )
                                print(timing_line)
                                logging.info(timing_line.strip())
                        finally:
                            active_gpu_error = sys.exc_info()[0] is not None
                            session_close_error = None
                            cached_profile_lease = (
                                gpu_profile_session_cache is not None
                                and gpu_profile_session is not None
                            )
                            if (
                                gpu_profile_session is not None
                                and not cached_profile_lease
                            ):
                                try:
                                    gpu_profile_session.close()
                                except BaseException as error:
                                    session_close_error = error
                            if not cached_profile_lease:
                                gpu_databases.pop(hmm_db, None)
                            if (
                                session_close_error is not None
                                and not active_gpu_error
                            ):
                                raise session_close_error
                    if args.write_seqs:
                        extract_sequences_from_tmp(tmp_dir, protein_dict, outdir)
                    combine_results(tmp_dir, os.path.join(outdir, f'{hmm_db}_hits_df.tsv'))
                    del db_hmms
                    gc.collect()
                else:
                    print(f"No installation_dir specified for db {hmm_db}")
                    logging.info(f"No installation_dir specified for db {hmm_db}")
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error = None
        batch_close_failed = False
        if gpu_sequence_batch is not None:
            try:
                gpu_sequence_batch.close()
            except BaseException as error:
                batch_close_failed = True
                if cleanup_error is None:
                    cleanup_error = error
        if batch_close_failed and gpu_profile_session_cache is not None:
            try:
                gpu_profile_session_cache.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        for _, _, session in gpu_databases.values():
            if session is not None:
                try:
                    session.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
        if cleanup_error is not None and not active_error:
            raise cleanup_error

    # Write MacSyFinder config file and finalize hmmsearch output if enabled
    if macsyfinder_dir:
        finalize_macsyfinder_files(macsyfinder_dir)
        _write_macsyfinder_conf(macsyfinder_dir, prot_in)

    # Clean up temporary directory if it exists
    if os.path.exists(os.path.join(outdir, 'tmp_results')):
        cleanup_temp_files(os.path.join(outdir, 'tmp_results'))

    time_printout = f"Process took {time.time()-t1} seconds."
    print(time_printout)
    logging.info(time_printout)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="ASTRA search tool")
    args = parser.parse_args()
    main(args)
