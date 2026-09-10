"""Thread-safe collection and fail-closed export for Phase 0 telemetry."""

from __future__ import annotations

import copy
import ctypes
import csv
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from threading import Lock
from typing import Any, Iterable

from ._telemetry import (
    defensive_continuation_statistics,
    validate_generation_statistics,
)


REPORT_SCHEMA_VERSION = 2
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class TelemetryReportCommittedError(RuntimeError):
    """Report publication committed, but final durability confirmation failed.

    ``target`` names the immutable report that now exists.  Callers must audit
    that target rather than retrying the no-replace publication.
    """

    def __init__(self, target: Path, error: OSError) -> None:
        self.target = target
        self.os_error = error
        super().__init__(
            f"telemetry report is committed at {target}, but final fsync "
            f"failed: {error}"
        )


def _exact_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer")
    return value


def _profile_ordinals(values: Iterable[int], expected: int) -> tuple[int, ...]:
    try:
        ordinals = tuple(values)
    except TypeError as error:
        raise TypeError("profile_ordinals must be iterable") from error
    if len(ordinals) != expected:
        raise ValueError("profile ordinal count differs from generation rows")
    for index, ordinal in enumerate(ordinals):
        _exact_nonnegative_int(ordinal, f"profile_ordinals[{index}]")
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("profile ordinals are not unique within the chunk")
    return ordinals


def _expected_profile_ordinals(values: Iterable[int]) -> tuple[int, ...]:
    try:
        ordinals = tuple(values)
    except TypeError as error:
        raise TypeError("expected profile ordinals must be iterable") from error
    for index, ordinal in enumerate(ordinals):
        _exact_nonnegative_int(ordinal, f"expected_profile_ordinals[{index}]")
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("expected profile ordinals are not unique")
    return ordinals


def _profile_keys(
    values: Iterable[str | None] | None, expected: int
) -> tuple[str | None, ...]:
    if values is None:
        return (None,) * expected
    try:
        keys = tuple(values)
    except TypeError as error:
        raise TypeError("profile_keys must be iterable") from error
    if len(keys) != expected:
        raise ValueError("profile key count differs from generation rows")
    if any(key is not None and type(key) is not str for key in keys):
        raise TypeError("profile keys must be exact strings or None")
    return keys


def _publish_directory_noreplace(stage: Path, target: Path) -> None:
    """Linux-atomically publish a directory without replacing any target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace report publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        _AT_FDCWD,
        os.fsencode(stage),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    ) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)


_NOREPLACE_UNSUPPORTED = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    }
)


def _discard_report_directory(path: Path) -> None:
    """Remove one private, uncommitted report directory."""
    path.chmod(0o700)
    for child in path.iterdir():
        child.chmod(0o600)
    shutil.rmtree(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256_at(directory_fd: int, name: str) -> str:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _same_directory_identity(path: Path, descriptor: int) -> bool:
    try:
        observed = path.lstat()
    except OSError:
        return False
    expected = os.fstat(descriptor)
    return (
        stat.S_ISDIR(observed.st_mode)
        and observed.st_dev == expected.st_dev
        and observed.st_ino == expected.st_ino
    )


def _publish_directory_portable(stage: Path, target: Path) -> None:
    """Publish through an exclusive directory reservation.

    ``mkdir`` provides the portable no-overwrite primitive, including on NFS.
    The reserved target remains mode 0700 while its already-fsynced members
    are copied.  Changing it to mode 0555 is the commit point.  Consequently a
    process crash leaves either no target, a visibly incomplete 0700 target,
    or the complete immutable report; every state continues to block an
    accidental overwrite.  Live failures after reservation deliberately leave
    the same 0700 evidence instead of deleting by pathname: a replacement
    directory can never be mistaken for, and removed as, this publisher's
    reservation.  Member creation and the commit chmod stay bound to one open
    directory descriptor, with pathname identity checked on both sides of the
    commit.  The portable protocol assumes cooperating publishers do not
    rename an empty reservation during the single ``mkdir``-to-``open``
    acquisition window; POSIX exposes no atomic mkdir-and-open operation.  A
    nonempty substitution is rejected before any member or mode is changed,
    and all substitutions after acquisition are identity-checked.
    """
    os.mkdir(target, 0o700)
    target_fd = os.open(
        target,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        if (
            not _same_directory_identity(target, target_fd)
            or os.listdir(target_fd)
        ):
            raise RuntimeError("portable report reservation identity changed")
        os.fchmod(target_fd, 0o700)
        members = sorted(
            stage.iterdir(),
            key=lambda path: (path.name == "artifact.sha256", path.name),
        )
        for source in members:
            source_stat = source.lstat()
            if source.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
                raise RuntimeError("staged report member is not a regular file")
            destination_fd = os.open(
                source.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
                dir_fd=target_fd,
            )
            with source.open("rb") as input_stream, os.fdopen(
                destination_fd, "wb"
            ) as output:
                shutil.copyfileobj(input_stream, output)
                output.flush()
                os.fchmod(output.fileno(), source_stat.st_mode & 0o777)
                os.fsync(output.fileno())
            destination_stat = os.stat(
                source.name, dir_fd=target_fd, follow_symlinks=False
            )
            if (
                destination_stat.st_size != source_stat.st_size
                or _file_sha256_at(target_fd, source.name)
                != _file_sha256(source)
            ):
                raise OSError(
                    errno.EIO,
                    "portable report publication changed staged content",
                    target / source.name,
                )
        os.fsync(target_fd)
        if not _same_directory_identity(target, target_fd):
            raise RuntimeError("portable report reservation identity changed")
        _discard_report_directory(stage)
        os.fchmod(target_fd, 0o555)
        try:
            os.fsync(target_fd)
        except OSError as error:
            raise TelemetryReportCommittedError(target, error) from error
        if not _same_directory_identity(target, target_fd):
            raise RuntimeError("committed report reservation identity changed")
    finally:
        os.close(target_fd)


def _publish_report_directory(stage: Path, target: Path) -> None:
    """Use atomic no-replace when supported, otherwise the NFS-safe protocol."""
    try:
        _publish_directory_noreplace(stage, target)
    except RuntimeError:
        _publish_directory_portable(stage, target)
    except OSError as error:
        if error.errno not in _NOREPLACE_UNSUPPORTED:
            raise
        _publish_directory_portable(stage, target)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _reason_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    reason_bits = profile["_reason_fact_bits"]
    reason_counts = profile["generation"]["reason_counts"]
    reason_cells = profile["generation"]["reason_logical_cells"]
    for stage, facts in reason_bits.items():
        counts = dict(reason_counts[stage])
        cells = dict(reason_cells[stage])
        for reason, _bit in facts:
            rows.append(
                {
                    "profile_ordinal": profile["profile_ordinal"],
                    "profile_key": profile["profile_key"],
                    "stage": stage,
                    "reason": reason,
                    "rows": counts.get(reason, 0),
                    "logical_cells": cells.get(reason, 0),
                }
            )
    return rows


class TelemetryCollector:
    """Join generation and continuation sidecars by exact global ordinal.

    The collector is inert until passed explicitly to an integration boundary.
    Generation rows may arrive in multiple chunks and continuation records may
    arrive concurrently or out of order. Duplicate ordinals, missing rows, and
    local/global identity drift fail closed before an export is created.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._chunks: list[dict[str, Any]] = []
        self._profiles: dict[int, dict[str, Any]] = {}
        self._batch_identities: set[tuple[int, int, int]] = set()
        self._expected_profile_ordinals: tuple[int, ...] | None = None

    def bind_expected_profiles(self, profile_ordinals: Iterable[int]) -> None:
        """Bind the complete intended report universe before search work."""
        ordinals = _expected_profile_ordinals(profile_ordinals)
        expected = frozenset(ordinals)
        with self._lock:
            if self._expected_profile_ordinals is not None:
                if self._expected_profile_ordinals != ordinals:
                    raise ValueError("expected profile ordinals changed")
                return
            extras = sorted(set(self._profiles) - expected)
            if extras:
                raise ValueError(
                    "recorded profile ordinals are outside the expected universe"
                )
            self._expected_profile_ordinals = ordinals

    def record_generation(
        self,
        generation_statistics: dict[str, Any],
        profile_ordinals: Iterable[int],
        *,
        profile_keys: Iterable[str | None] | None = None,
    ) -> int:
        """Record one instrumented generation chunk and its global mapping."""
        generation = validate_generation_statistics(generation_statistics)
        batch_identity = generation["batch_identity"]
        if batch_identity is None:
            raise ValueError(
                "collector generation evidence requires a sealed batch identity"
            )
        count = generation["profile_count"]
        identity_tuple = (
            batch_identity["session_id"],
            batch_identity["selection_id"],
            batch_identity["batch_generation"],
        )
        ordinals = _profile_ordinals(profile_ordinals, count)
        keys = _profile_keys(profile_keys, count)
        with self._lock:
            if self._expected_profile_ordinals is None:
                raise RuntimeError(
                    "expected profile universe must be bound before generation"
                )
            expected = frozenset(self._expected_profile_ordinals)
            extras = sorted(set(ordinals) - expected)
            if extras:
                raise ValueError(
                    "generation profile ordinals are outside the expected universe"
                )
            duplicate = next(
                (ordinal for ordinal in ordinals if ordinal in self._profiles),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"profile ordinal {duplicate} was already recorded")
            if identity_tuple in self._batch_identities:
                raise ValueError("sealed batch identity was already recorded")
            chunk_index = len(self._chunks)
            shared = {
                name: copy.deepcopy(value)
                for name, value in generation.items()
                if name != "profiles"
            }
            self._chunks.append(
                {
                    "chunk_index": chunk_index,
                    "profile_ordinals": ordinals,
                    "profile_keys": keys,
                    "generation": shared,
                }
            )
            self._batch_identities.add(identity_tuple)
            reason_fact_bits = generation["reason_fact_bits"]
            for local_index, (ordinal, key, row) in enumerate(
                zip(ordinals, keys, generation["profiles"], strict=True)
            ):
                self._profiles[ordinal] = {
                    "profile_ordinal": ordinal,
                    "profile_key": key,
                    "chunk_index": chunk_index,
                    "chunk_profile_index": local_index,
                    "target_count": generation["target_count"],
                    "target_residues": generation["total_target_residues"],
                    "batch_identity": copy.deepcopy(batch_identity),
                    "generation": copy.deepcopy(row),
                    "continuation": None,
                    "_reason_fact_bits": copy.deepcopy(reason_fact_bits),
                }
            return chunk_index

    def record_continuation(
        self,
        profile_ordinal: int,
        continuation_statistics: dict[str, Any],
    ) -> None:
        """Bind one validated CPU-continuation record to its generation row."""
        ordinal = _exact_nonnegative_int(profile_ordinal, "profile_ordinal")
        continuation = defensive_continuation_statistics(
            continuation_statistics
        )
        with self._lock:
            profile = self._profiles.get(ordinal)
            if profile is None:
                raise ValueError(
                    f"profile ordinal {ordinal} has no generation record"
                )
            if profile["continuation"] is not None:
                raise ValueError(
                    f"profile ordinal {ordinal} continuation was already recorded"
                )
            if (
                continuation["identity"]["profile_index"]
                != profile["chunk_profile_index"]
            ):
                raise ValueError(
                    "continuation local profile identity differs from generation"
                )
            generation = profile["generation"]
            if continuation["target_count"] != profile["target_count"]:
                raise ValueError(
                    "continuation target count differs from generation"
                )
            if (
                continuation["postfilter_record_count"]
                != generation["counts"]["f1_candidate_count"]
            ):
                raise ValueError(
                    "continuation post-filter count differs from generation"
                )
            generation_identity = profile["batch_identity"]
            if continuation["batch_identity"] != generation_identity:
                raise ValueError(
                    "continuation batch identity differs from generation"
                )
            if continuation["path"] == "journal":
                row_start = sum(
                    other["generation"]["journal"]["row_count"]
                    for other in self._profiles.values()
                    if (
                        other["chunk_index"] == profile["chunk_index"]
                        and other["chunk_profile_index"]
                        < profile["chunk_profile_index"]
                    )
                )
                row_stop = row_start + generation["journal"]["row_count"]
                identity = continuation["identity"]
                if (
                    identity["journal_row_start"] != row_start
                    or identity["journal_row_stop"] != row_stop
                    or continuation["journal"]["match_count"]
                    != generation["journal"]["row_count"]
                    or continuation["journal"]["cpu_required_count"]
                    != generation["counts"]["backward_cpu_required_count"]
                    or continuation["journal"]["no_region_count"]
                    != generation["counts"]["backward_no_region_count"]
                    or continuation["journal"]["simple_count"]
                    != generation["counts"]["backward_simple_count"]
                    or continuation["source_routes"]["forward_count"]
                    != (
                        generation["counts"]["forward_reject_count"]
                        + continuation["journal"]["cpu_required_count"]
                    )
                ):
                    raise ValueError(
                        "continuation journal attribution differs from generation"
                    )
            profile["continuation"] = continuation

    def snapshot(self, *, require_complete: bool = True) -> dict[str, Any]:
        """Return canonical raw records, reason totals, and a CPU-wall Pareto."""
        if type(require_complete) is not bool:
            raise TypeError("require_complete must be bool")
        with self._lock:
            chunks = copy.deepcopy(self._chunks)
            stored_profiles = copy.deepcopy(self._profiles)
            expected_ordinals = self._expected_profile_ordinals

        missing_continuation = sorted(
            ordinal
            for ordinal, profile in stored_profiles.items()
            if profile["continuation"] is None
        )
        missing_generation = (
            ()
            if expected_ordinals is None
            else tuple(
                ordinal
                for ordinal in expected_ordinals
                if ordinal not in stored_profiles
            )
        )
        if require_complete and expected_ordinals is None:
            raise RuntimeError("expected profile universe is not bound")
        if require_complete and (missing_generation or missing_continuation):
            raise RuntimeError(
                "Phase 0 telemetry is incomplete: missing generation="
                + ",".join(str(value) for value in missing_generation[:16])
                + "; missing continuation="
                + ",".join(str(value) for value in missing_continuation[:16])
            )

        profiles = []
        reason_rows = []
        for ordinal in sorted(stored_profiles):
            profile = stored_profiles[ordinal]
            reason_rows.extend(_reason_rows(profile))
            profile.pop("_reason_fact_bits")
            continuation = profile["continuation"]
            profile["continuation_cpu_wall_ns"] = (
                None if continuation is None else continuation["wall_ns"]
            )
            profiles.append(profile)

        reason_summary: dict[tuple[str, str], list[int]] = {}
        for row in reason_rows:
            key = (row["stage"], row["reason"])
            totals = reason_summary.setdefault(key, [0, 0])
            totals[0] += row["rows"]
            totals[1] += row["logical_cells"]
        reason_totals = [
            {
                "stage": stage,
                "reason": reason,
                "rows": totals[0],
                "logical_cells": totals[1],
            }
            for (stage, reason), totals in reason_summary.items()
        ]

        complete_profiles = [
            profile
            for profile in profiles
            if profile["continuation_cpu_wall_ns"] is not None
        ]
        ranked = sorted(
            complete_profiles,
            key=lambda profile: (
                -profile["continuation_cpu_wall_ns"],
                profile["profile_ordinal"],
            ),
        )
        total_wall_ns = sum(
            profile["continuation_cpu_wall_ns"] for profile in ranked
        )
        cumulative = 0
        pareto = []
        for rank, profile in enumerate(ranked, 1):
            cumulative += profile["continuation_cpu_wall_ns"]
            pareto.append(
                {
                    "rank": rank,
                    "profile_ordinal": profile["profile_ordinal"],
                    "profile_key": profile["profile_key"],
                    "continuation_cpu_wall_ns": (
                        profile["continuation_cpu_wall_ns"]
                    ),
                    "cumulative_cpu_wall_ns": cumulative,
                    "total_cpu_wall_ns": total_wall_ns,
                    "cumulative_fraction_numerator": cumulative,
                    "cumulative_fraction_denominator": total_wall_ns,
                }
            )

        complete = (
            expected_ordinals is not None
            and not missing_generation
            and not missing_continuation
        )
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "scope": "phase0_route_telemetry_report",
            "complete": complete,
            "expected_profile_ordinals": expected_ordinals,
            "missing_generation_profile_ordinals": tuple(missing_generation),
            "missing_continuation_profile_ordinals": tuple(
                missing_continuation
            ),
            "chunk_count": len(chunks),
            "expected_profile_count": (
                None if expected_ordinals is None else len(expected_ordinals)
            ),
            "profile_count": len(profiles),
            "continuation_cpu_wall_ns": total_wall_ns,
            "chunks": tuple(chunks),
            "profiles": tuple(profiles),
            "reason_rows": tuple(reason_rows),
            "reason_totals": tuple(reason_totals),
            "pareto": tuple(pareto),
        }

    def export(self, directory: os.PathLike[str] | str) -> dict[str, Path]:
        """No-replace publish one immutable, crash-detectable report."""
        snapshot = self.snapshot(require_complete=True)
        target = Path(directory)
        if not target.name:
            raise ValueError("report directory must have a final path component")
        parent = target.parent
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError("report parent must be an existing real directory")
        if os.path.lexists(target):
            raise FileExistsError(target)

        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
        try:
            payloads = self._render(snapshot)
            hashes = []
            for name, payload in payloads.items():
                path = stage / name
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                path.chmod(0o444)
                hashes.append((name, hashlib.sha256(payload).hexdigest()))
            manifest = "".join(
                f"{digest}  {name}\n" for name, digest in sorted(hashes)
            ).encode("ascii")
            manifest_path = stage / "artifact.sha256"
            with manifest_path.open("xb") as stream:
                stream.write(manifest)
                stream.flush()
                os.fsync(stream.fileno())
            manifest_path.chmod(0o444)
            _fsync_directory(stage)
            stage.chmod(0o555)
            _publish_report_directory(stage, target)
            try:
                _fsync_directory(parent)
            except OSError as error:
                raise TelemetryReportCommittedError(target, error) from error
        except BaseException:
            if stage.exists():
                _discard_report_directory(stage)
            raise
        return {
            name: target / name
            for name in (*self._render_names(), "artifact.sha256")
        }

    @staticmethod
    def _render_names() -> tuple[str, ...]:
        return (
            "phase0-raw.json",
            "phase0-profiles.tsv",
            "phase0-reasons.tsv",
            "phase0-reason-summary.tsv",
            "phase0-pareto.tsv",
        )

    @staticmethod
    def _render(snapshot: dict[str, Any]) -> dict[str, bytes]:
        raw = (
            json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        profiles = snapshot["profiles"]
        count_names = tuple(profiles[0]["generation"]["counts"]) if profiles else ()
        cell_names = (
            tuple(profiles[0]["generation"]["logical_cells"])
            if profiles
            else ()
        )
        route_names = (
            tuple(profiles[0]["continuation"]["routes"])
            if profiles
            else ()
        )
        profile_header = (
            "profile_ordinal",
            "profile_key",
            "chunk_index",
            "chunk_profile_index",
            "batch_session_id",
            "batch_selection_id",
            "batch_generation",
            "model_length",
            "target_count",
            "target_residues",
            *(f"generation.{name}" for name in count_names),
            *(f"generation.{name}" for name in cell_names),
            "generation.journal_row_count",
            "generation.journal_region_count",
            "continuation_cpu_wall_ns",
            "continuation.path",
            *(f"continuation.{name}" for name in route_names),
        )
        profile_rows = []
        for profile in profiles:
            generation = profile["generation"]
            continuation = profile["continuation"]
            profile_rows.append(
                (
                    profile["profile_ordinal"],
                    profile["profile_key"] or "",
                    profile["chunk_index"],
                    profile["chunk_profile_index"],
                    profile["batch_identity"]["session_id"],
                    profile["batch_identity"]["selection_id"],
                    profile["batch_identity"]["batch_generation"],
                    generation["model_length"],
                    profile["target_count"],
                    profile["target_residues"],
                    *(generation["counts"][name] for name in count_names),
                    *(generation["logical_cells"][name] for name in cell_names),
                    generation["journal"]["row_count"],
                    generation["journal"]["region_count"],
                    profile["continuation_cpu_wall_ns"],
                    continuation["path"],
                    *(continuation["routes"][name] for name in route_names),
                )
            )

        reason_header = (
            "profile_ordinal",
            "profile_key",
            "stage",
            "reason",
            "rows",
            "logical_cells",
        )
        reason_rows = [
            tuple(row[name] for name in reason_header)
            for row in snapshot["reason_rows"]
        ]
        summary_header = ("stage", "reason", "rows", "logical_cells")
        summary_rows = [
            tuple(row[name] for name in summary_header)
            for row in snapshot["reason_totals"]
        ]
        pareto_header = (
            "rank",
            "profile_ordinal",
            "profile_key",
            "continuation_cpu_wall_ns",
            "cumulative_cpu_wall_ns",
            "total_cpu_wall_ns",
            "cumulative_fraction_numerator",
            "cumulative_fraction_denominator",
        )
        pareto_rows = [
            tuple(row[name] if row[name] is not None else "" for name in pareto_header)
            for row in snapshot["pareto"]
        ]

        def tsv(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> bytes:
            output = io.StringIO(newline="")
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
            return output.getvalue().encode("utf-8")

        return {
            "phase0-raw.json": raw,
            "phase0-profiles.tsv": tsv(profile_header, profile_rows),
            "phase0-reasons.tsv": tsv(reason_header, reason_rows),
            "phase0-reason-summary.tsv": tsv(summary_header, summary_rows),
            "phase0-pareto.tsv": tsv(pareto_header, pareto_rows),
        }


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "TelemetryCollector",
    "TelemetryReportCommittedError",
]
