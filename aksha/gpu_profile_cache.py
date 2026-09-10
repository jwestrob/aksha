"""Explicit, bounded reuse of immutable plan7 GPU profile sessions.

The cache is deliberately process-local and opt-in.  A normal one-shot Aksha
CLI invocation therefore retains its existing ownership and cleanup behavior.
Long-lived callers may own one cache, pass it to successive searches, and close
it deterministically when the worker retires.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Hashable, Optional, Union


PROFILE_SEMANTICS = "canonical-amino-local-multihit-exact-forward-v1"


class GPUProfileCacheBusyError(RuntimeError):
    """Raised when a cache entry already has an active search lease."""


@dataclass(frozen=True)
class Plan7RuntimeIdentity:
    """Code and private-ABI identity for one loaded plan7 runtime."""

    pyhmmer_version: str
    pyhmmer_private_abi_sha256: str
    adapter_sha256: str
    native_extension_sha256: str
    pipeline_extension_sha256: str


@dataclass(frozen=True)
class GPUProfileCacheKey:
    """Every input that can change or invalidate a cached profile session."""

    canonical_base: Path
    manifest_sha256: str
    pressed_stat_token: Hashable
    runtime: Plan7RuntimeIdentity
    device_key: Hashable
    build_workers: int
    selection_workers: int
    profile_semantics: str


@dataclass
class _CacheEntry:
    key: GPUProfileCacheKey
    pairs: tuple[Any, ...]
    session: Any
    leased: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan7_runtime_identity() -> Plan7RuntimeIdentity:
    """Fingerprint the already-loaded plan7/PyHMMER implementation once.

    The cache cannot cross a process boundary, but recording the exact Python
    adapter and native extension artifacts makes its compatibility contract
    explicit and auditable instead of relying only on a package version.
    """
    import astra_pyhmmer
    from plan7_gpu import _native, _pipeline
    from plan7_gpu import adapter
    from plan7_gpu._abi import pyhmmer_abi_fingerprint

    def module_digest(module: Any) -> str:
        origin = getattr(module, "__file__", None)
        if not origin:
            raise RuntimeError("plan7 runtime module has no filesystem origin")
        return _sha256_file(Path(origin).resolve(strict=True))

    return Plan7RuntimeIdentity(
        pyhmmer_version=str(astra_pyhmmer.__version__),
        pyhmmer_private_abi_sha256=pyhmmer_abi_fingerprint(),
        adapter_sha256=module_digest(adapter),
        native_extension_sha256=module_digest(_native),
        pipeline_extension_sha256=module_digest(_pipeline),
    )


class GPUProfileSessionLease:
    """Exclusive access to one cached immutable profile session."""

    __slots__ = (
        "_cache",
        "_entry",
        "_released",
        "reused",
        "validation_seconds",
        "profile_load_seconds",
        "session_build_seconds",
    )

    def __init__(
        self,
        cache: GPUProfileSessionCache,
        entry: _CacheEntry,
        *,
        reused: bool,
        validation_seconds: float,
        profile_load_seconds: float,
        session_build_seconds: float,
    ) -> None:
        self._cache = cache
        self._entry = entry
        self._released = False
        self.reused = reused
        self.validation_seconds = validation_seconds
        self.profile_load_seconds = profile_load_seconds
        self.session_build_seconds = session_build_seconds

    def _require_open(self) -> _CacheEntry:
        if self._released or self._entry is None:
            raise RuntimeError("GPU profile-session lease is closed")
        return self._entry

    @property
    def profile_pairs(self) -> tuple[Any, ...]:
        return self._require_open().pairs

    @property
    def cache_key(self) -> GPUProfileCacheKey:
        return self._require_open().key

    @property
    def closed(self) -> bool:
        entry = self._entry
        return self._released or entry is None or bool(entry.session.closed)

    @property
    def statistics(self) -> dict[str, int]:
        return self._require_open().session.statistics

    def __len__(self) -> int:
        return len(self._require_open().session)

    def select(self, indices: Any) -> Any:
        return self._require_open().session.select(indices)

    def close(self) -> None:
        if not self._released:
            cache = self._cache
            entry = self._entry
            try:
                cache._release(entry)
            finally:
                # A retired lease must not keep the cache, its 1.15 GB profile
                # tuple, or its native session alive through stale references.
                self._released = True
                self._cache = None
                self._entry = None

    def __enter__(self) -> GPUProfileSessionLease:
        self._require_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class GPUProfileSessionReservation:
    """An exclusive pre-allocation claim for one upcoming cache lease."""

    __slots__ = ("_cache", "_active")

    def __init__(self, cache: GPUProfileSessionCache) -> None:
        self._cache = cache
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def close(self) -> None:
        if self._active:
            cache = self._cache
            try:
                cache._cancel_reservation(self)
            finally:
                if not self._active:
                    self._cache = None

    def __enter__(self) -> GPUProfileSessionReservation:
        if not self._active:
            raise RuntimeError("GPU profile-session reservation is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class GPUProfileSessionCache:
    """A one-entry, exclusive-lease cache for very large profile sessions.

    PFAM's immutable host snapshot is roughly 1.15 GB, so this first production
    seam intentionally permits only one resident database and one active search
    at a time.  A different key evicts the idle entry before allocating its
    replacement; two full sessions are never retained by the cache itself.
    """

    def __init__(
        self,
        *,
        runtime_identity: Optional[Plan7RuntimeIdentity] = None,
        validator: Optional[Callable[..., Any]] = None,
        loader: Optional[Callable[..., Any]] = None,
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if validator is None or loader is None or session_factory is None:
            from plan7_gpu import ProfileSession, load_pressed_profiles
            from plan7_gpu.pressed_manifest import validate_pressed_manifest

            if validator is None:
                validator = validate_pressed_manifest
            if loader is None:
                loader = load_pressed_profiles
            if session_factory is None:
                session_factory = ProfileSession
        self._runtime = runtime_identity or plan7_runtime_identity()
        self._validator = validator
        self._loader = loader
        self._session_factory = session_factory
        self._lock = RLock()
        self._entry: Optional[_CacheEntry] = None
        self._reservation: Optional[GPUProfileSessionReservation] = None
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def resident(self) -> bool:
        with self._lock:
            return self._entry is not None

    @property
    def resident_key(self) -> Optional[GPUProfileCacheKey]:
        with self._lock:
            return None if self._entry is None else self._entry.key

    def reserve(self) -> GPUProfileSessionReservation:
        """Claim the cache before allocating a per-search target batch."""
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU profile-session cache is closed")
            if (
                self._reservation is not None
                or (self._entry is not None and self._entry.leased)
            ):
                raise GPUProfileCacheBusyError(
                    "GPU profile-session cache already has an active search lease"
                )
            reservation = GPUProfileSessionReservation(self)
            self._reservation = reservation
            return reservation

    def acquire(
        self,
        pressed_base: Union[str, os.PathLike[str]],
        manifest_path: Union[str, os.PathLike[str]],
        *,
        device_key: Hashable,
        build_workers: int,
        selection_workers: int = 0,
        profile_semantics: str = PROFILE_SEMANTICS,
        reservation: Optional[GPUProfileSessionReservation] = None,
    ) -> GPUProfileSessionLease:
        if isinstance(build_workers, bool) or not isinstance(build_workers, int):
            raise TypeError("build_workers must be a positive integer")
        if build_workers <= 0:
            raise ValueError("build_workers must be a positive integer")
        if (
            isinstance(selection_workers, bool)
            or not isinstance(selection_workers, int)
        ):
            raise TypeError("selection_workers must be a nonnegative integer")
        if selection_workers < 0:
            raise ValueError("selection_workers must be a nonnegative integer")
        if not isinstance(profile_semantics, str) or not profile_semantics:
            raise TypeError("profile_semantics must be a non-empty string")
        try:
            hash(device_key)
        except TypeError as error:
            raise TypeError("device_key must be hashable") from error

        canonical_base = Path(pressed_base).resolve(strict=False)
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU profile-session cache is closed")
            if reservation is None:
                if (
                    self._reservation is not None
                    or (self._entry is not None and self._entry.leased)
                ):
                    raise GPUProfileCacheBusyError(
                        "GPU profile-session cache already has an active search lease"
                    )
            elif (
                reservation._cache is not self
                or not reservation._active
                or self._reservation is not reservation
            ):
                raise RuntimeError("GPU profile-session cache reservation is invalid")

            validation_started = time.perf_counter()
            validation = self._validator(canonical_base, manifest_path)
            validation_seconds = time.perf_counter() - validation_started
            if validation.canonical_base != canonical_base:
                raise ValueError("pressed manifest canonical database mismatch")
            manifest_sha256 = validation.manifest_sha256
            if (
                not isinstance(manifest_sha256, str)
                or len(manifest_sha256) != 64
                or any(character not in "0123456789abcdef" for character in manifest_sha256)
            ):
                raise RuntimeError("pressed manifest has an invalid SHA256 identity")
            try:
                hash(validation.stat_token)
            except TypeError as error:
                raise RuntimeError("pressed database stat token is not hashable") from error

            key = GPUProfileCacheKey(
                canonical_base=canonical_base,
                manifest_sha256=manifest_sha256,
                pressed_stat_token=validation.stat_token,
                runtime=self._runtime,
                device_key=device_key,
                build_workers=build_workers,
                selection_workers=selection_workers,
                profile_semantics=profile_semantics,
            )
            entry = self._entry
            if entry is not None and entry.key == key and not entry.session.closed:
                entry.leased = True
                self._consume_reservation(reservation)
                return GPUProfileSessionLease(
                    self,
                    entry,
                    reused=True,
                    validation_seconds=validation_seconds,
                    profile_load_seconds=0.0,
                    session_build_seconds=0.0,
                )

            if entry is not None:
                self._entry = None
                try:
                    entry.session.close()
                finally:
                    # Do not carry the old PFAM tuple into replacement load:
                    # even this local entry would otherwise overlap two large
                    # immutable profile sets during key invalidation.
                    entry.pairs = ()
                    entry.session = None
                entry = None

            load_started = time.perf_counter()
            pairs = tuple(self._loader(canonical_base, manifest=manifest_path))
            profile_load_seconds = time.perf_counter() - load_started
            if len(pairs) != validation.model_count:
                raise RuntimeError(
                    "loaded profile count differs from validated manifest"
                )
            session_started = time.perf_counter()
            session = self._session_factory(
                pairs,
                build_workers=build_workers,
                selection_workers=selection_workers,
            )
            session_build_seconds = time.perf_counter() - session_started
            try:
                if session.closed:
                    raise RuntimeError("new GPU profile session is already closed")
                if len(session) != len(pairs):
                    raise RuntimeError(
                        "GPU profile session does not cover loaded profiles"
                    )
            except BaseException:
                session.close()
                raise
            entry = _CacheEntry(key=key, pairs=pairs, session=session, leased=True)
            self._entry = entry
            self._consume_reservation(reservation)
            return GPUProfileSessionLease(
                self,
                entry,
                reused=False,
                validation_seconds=validation_seconds,
                profile_load_seconds=profile_load_seconds,
                session_build_seconds=session_build_seconds,
            )

    def _consume_reservation(
        self,
        reservation: Optional[GPUProfileSessionReservation],
    ) -> None:
        if reservation is not None:
            if self._reservation is not reservation or not reservation._active:
                raise RuntimeError("GPU profile-session cache reservation is invalid")
            self._reservation = None
            reservation._active = False
            reservation._cache = None

    def _cancel_reservation(
        self, reservation: GPUProfileSessionReservation
    ) -> None:
        with self._lock:
            if self._reservation is reservation and reservation._active:
                self._reservation = None
                reservation._active = False
                reservation._cache = None
                return
            if reservation._active:
                raise RuntimeError("GPU profile-session cache reservation is invalid")

    def _release(self, entry: _CacheEntry) -> None:
        with self._lock:
            if self._entry is not entry or not entry.leased:
                raise RuntimeError("GPU profile-session cache lease is invalid")
            entry.leased = False
            if self._closed:
                self._entry = None
                entry.session.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            reservation = self._reservation
            if reservation is not None:
                self._reservation = None
                reservation._active = False
                reservation._cache = None
            entry = self._entry
            if entry is not None and not entry.leased:
                self._entry = None
                entry.session.close()

    def __enter__(self) -> GPUProfileSessionCache:
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU profile-session cache is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "GPUProfileCacheBusyError",
    "GPUProfileCacheKey",
    "GPUProfileSessionCache",
    "GPUProfileSessionLease",
    "GPUProfileSessionReservation",
    "PROFILE_SEMANTICS",
    "Plan7RuntimeIdentity",
    "plan7_runtime_identity",
]
