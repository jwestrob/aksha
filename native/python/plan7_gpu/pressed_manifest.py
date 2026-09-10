"""Audit and attest a pressed HMM database.

The manifest is a trust anchor for loading a pressed database without rebuilding
every optimized profile.  Creation performs the expensive check once: each HMM
is optimized with HMMER's canonical background and compared byte-for-byte with
the corresponding serialized pressed filter and profile.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import datetime as dt
import hashlib
import io
import json
import os
from itertools import zip_longest
from pathlib import Path
import platform
import stat
import struct
import sys
import tempfile
from typing import Any, NamedTuple, cast

import astra_pyhmmer


PYHMMER_VERSION = cast(str, getattr(astra_pyhmmer, "__version__"))
SCHEMA = "plan7-gpu-pressed-profile-manifest"
SCHEMA_VERSION = 1
TOOL_NAME = "plan7_gpu.pressed_manifest"
TOOL_VERSION = 1
AUDIT_ALGORITHM = "canonical-hmmpress-serialized-oprofile-v1"
PRIVATE_ABI_ALGORITHM = "plan7-gpu-astra_pyhmmer-private-abi-v1"
VERIFICATION_REFERENCE = "HMM.to_profile(default Background, L=pressed L)"
VERIFICATION_COMPARISON = (
    "exact serialized h3f and h3p bytes after pressed-offset normalization"
)
PRESSED_SUFFIXES = ("h3m", "h3i", "h3f", "h3p")
_STAT_FIELDS = ("device", "inode", "size", "mtime_ns", "ctime_ns")
_MISSING = object()
_MAX_MANIFEST_BYTES = 1024 * 1024
_SOURCE_PATH = Path(__file__).resolve(strict=True)


class PressedManifestError(ValueError):
    """A pressed database failed audit or manifest validation."""


class PressedProfileAuditError(PressedManifestError):
    """An HMM and its pressed optimized profile are not equivalent."""


class PressedFileStat(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class PressedStatToken(NamedTuple):
    h3m: PressedFileStat
    h3i: PressedFileStat
    h3f: PressedFileStat
    h3p: PressedFileStat


@dataclass(frozen=True, slots=True)
class ManifestValidation:
    """A stable database snapshot authenticated by a manifest."""

    canonical_base: Path
    model_count: int
    stat_token: PressedStatToken
    used_content_hashes: bool
    manifest_sha256: str


def _canonical_base(path: str | os.PathLike[str]) -> Path:
    return Path(path).resolve(strict=False)


def _member_path(base: Path, suffix: str) -> Path:
    return Path(f"{base}.{suffix}")


def _stat_value(value: os.stat_result, label: Path) -> PressedFileStat:
    if not stat.S_ISREG(value.st_mode):
        raise PressedManifestError(
            f"pressed database member is not a regular file: {label}"
        )
    return PressedFileStat(
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_stat(path: Path) -> PressedFileStat:
    return _stat_value(path.stat(), path)


def pressed_stat_token(
    path: str | os.PathLike[str],
) -> PressedStatToken:
    """Capture the identity and mutation timestamps of all four members."""
    base = _canonical_base(path)
    return PressedStatToken(
        *(_file_stat(_member_path(base, suffix)) for suffix in PRESSED_SUFFIXES)
    )


@contextmanager
def _pinned_pressed_database(
    path: str | os.PathLike[str],
) -> Iterator[tuple[Path, PressedStatToken]]:
    """Pin all pressed members and expose them under one private alias base.

    The caller must exclude in-place writers for the duration of this context.
    Reopening every source path on exit detects ordinary concurrent drift and
    refreshes close-to-open attributes on network filesystems, but it is not a
    substitute for writer coordination.
    """
    if sys.platform != "linux" or not hasattr(os, "O_NOFOLLOW"):
        raise PressedManifestError(
            "pinned pressed-database loading requires Linux O_NOFOLLOW support"
        )

    base = _canonical_base(path)
    descriptors: list[tuple[str, int]] = []
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        for suffix in PRESSED_SUFFIXES:
            member = _member_path(base, suffix)
            descriptor = os.open(member, flags)
            descriptors.append((suffix, descriptor))

        before = PressedStatToken(
            *(
                _stat_value(os.fstat(descriptor), _member_path(base, suffix))
                for suffix, descriptor in descriptors
            )
        )
        try:
            with tempfile.TemporaryDirectory(prefix="plan7-gpu-pressed-") as directory:
                alias_base = Path(directory) / "database"
                for suffix, descriptor in descriptors:
                    os.symlink(
                        f"/proc/self/fd/{descriptor}",
                        _member_path(alias_base, suffix),
                    )
                yield alias_base, before
        finally:
            pinned_after = PressedStatToken(
                *(
                    _stat_value(os.fstat(descriptor), _member_path(base, suffix))
                    for suffix, descriptor in descriptors
                )
            )
            reopened: list[tuple[str, int]] = []
            try:
                for suffix in PRESSED_SUFFIXES:
                    member = _member_path(base, suffix)
                    descriptor = os.open(member, flags)
                    reopened.append((suffix, descriptor))
                path_after = PressedStatToken(
                    *(
                        _stat_value(os.fstat(descriptor), _member_path(base, suffix))
                        for suffix, descriptor in reopened
                    )
                )
            finally:
                for _, descriptor in reopened:
                    os.close(descriptor)
            if pinned_after != before or path_after != before:
                raise PressedManifestError(
                    "pressed database changed while its members were pinned"
                )
    finally:
        for _, descriptor in descriptors:
            os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_LOADED_SOURCE_FINGERPRINT = (_SOURCE_PATH.name, _sha256_file(_SOURCE_PATH))


def _serialized_profile(profile: Any) -> tuple[bytes, bytes]:
    filter_output = io.BytesIO()
    profile_output = io.BytesIO()
    profile.write(filter_output, profile_output)
    return filter_output.getvalue(), profile_output.getvalue()


def _identity_matches(hmm: Any, optimized: Any) -> bool:
    return bool(
        hmm.alphabet == optimized.alphabet
        and hmm.M == optimized.M
        and hmm.name == optimized.name
        and hmm.accession == optimized.accession
        and hmm.consensus == optimized.consensus
    )


def _audit_profiles(base: Path) -> int:
    count = 0
    try:
        with (
            astra_pyhmmer.plan7.HMMFile(base) as hmm_file,
            astra_pyhmmer.plan7.HMMPressedFile(base) as pressed_file,
            astra_pyhmmer.easel.SSIReader(_member_path(base, "h3i")) as index,
        ):
            primary_keys = set(index.primary_keys)
            indexed_count = len(index.primary_keys)
            if len(primary_keys) != indexed_count:
                raise PressedProfileAuditError(
                    "pressed index contains duplicate primary keys"
                )
            records = zip_longest(hmm_file, pressed_file, fillvalue=_MISSING)
            for ordinal, (hmm, optimized) in enumerate(records):
                if hmm is _MISSING or optimized is _MISSING:
                    raise PressedProfileAuditError(
                        "pressed HMM and optimized-profile streams have "
                        f"different lengths at ordinal {ordinal}"
                    )
                hmm = cast(Any, hmm)
                optimized = cast(Any, optimized)
                if not _identity_matches(hmm, optimized):
                    raise PressedProfileAuditError(
                        f"pressed HMM/profile identity mismatch at ordinal {ordinal}"
                    )
                if hmm.name not in primary_keys:
                    raise PressedProfileAuditError(
                        f"pressed index is missing model name at ordinal {ordinal}"
                    )
                try:
                    name_entry = index.find_name(hmm.name)
                    if name_entry.record_offset != optimized.offsets.model or (
                        hmm.accession is not None
                        and index.find_name(hmm.accession) != name_entry
                    ):
                        raise PressedProfileAuditError(
                            f"pressed index offset/alias mismatch at ordinal {ordinal}"
                        )
                except KeyError as error:
                    raise PressedProfileAuditError(
                        f"pressed index key mismatch at ordinal {ordinal}"
                    ) from error

                background = astra_pyhmmer.plan7.Background(hmm.alphabet)
                reference = hmm.to_profile(background, L=optimized.L).to_optimized()
                for field in ("model", "filter", "profile"):
                    setattr(
                        reference.offsets,
                        field,
                        getattr(optimized.offsets, field),
                    )
                reference_filter, reference_profile = _serialized_profile(reference)
                pressed_filter, pressed_profile = _serialized_profile(optimized)
                if reference_filter != pressed_filter:
                    raise PressedProfileAuditError(
                        "pressed HMM/profile score mismatch in h3f "
                        f"at ordinal {ordinal}"
                    )
                if reference_profile != pressed_profile:
                    raise PressedProfileAuditError(
                        "pressed HMM/profile score mismatch in h3p "
                        f"at ordinal {ordinal}"
                    )
                count += 1
            if count != indexed_count:
                raise PressedProfileAuditError(
                    "pressed index and profile streams have different lengths"
                )
    except PressedManifestError:
        raise
    except Exception as error:
        raise PressedProfileAuditError(
            f"failed to audit pressed database {base}: {error}"
        ) from error

    if count == 0:
        raise PressedProfileAuditError("pressed database contains no models")
    return count


def _private_abi_fingerprint() -> str:
    from ._abi import (  # imported lazily so this module has no CUDA dependency
        pyhmmer_abi_fingerprint,
        validate_private_abi_platform,
    )

    validate_private_abi_platform()
    return cast(str, pyhmmer_abi_fingerprint())


def _source_fingerprint() -> tuple[str, str]:
    return _LOADED_SOURCE_FINGERPRINT


def _require_loaded_source_unchanged() -> None:
    current = (_SOURCE_PATH.name, _sha256_file(_SOURCE_PATH))
    if current != _LOADED_SOURCE_FINGERPRINT:
        raise PressedManifestError(
            "pressed-manifest audit source changed after it was imported"
        )


def _stat_json(value: PressedFileStat) -> dict[str, int]:
    return dict(zip(_STAT_FIELDS, value, strict=True))


def _timestamp() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def build_pressed_manifest(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Fully audit a pressed database and return its versioned manifest."""
    _require_loaded_source_unchanged()
    base = _canonical_base(path)
    source_filename, source_sha256 = _source_fingerprint()
    private_abi = _private_abi_fingerprint()
    with _pinned_pressed_database(base) as (snapshot, stat_token):
        model_count = _audit_profiles(snapshot)
        hashes = {
            suffix: _sha256_file(_member_path(snapshot, suffix))
            for suffix in PRESSED_SUFFIXES
        }
    _require_loaded_source_unchanged()

    stat_values = dict(zip(PRESSED_SUFFIXES, stat_token, strict=True))
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _timestamp(),
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "audit_algorithm": AUDIT_ALGORITHM,
            "source": {
                "filename": source_filename,
                "sha256": source_sha256,
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "pyhmmer_version": PYHMMER_VERSION,
            "private_abi": {
                "algorithm": PRIVATE_ABI_ALGORITHM,
                "sha256": private_abi,
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "byteorder": sys.byteorder,
                "pointer_bits": struct.calcsize("P") * 8,
            },
        },
        "database": {
            "base_name": base.name,
            "model_count": model_count,
            "artifacts": {
                suffix: {
                    "filename": _member_path(base, suffix).name,
                    "size_bytes": stat_values[suffix].size,
                    "sha256": hashes[suffix],
                }
                for suffix in PRESSED_SUFFIXES
            },
            "stat_token": {
                suffix: _stat_json(stat_values[suffix]) for suffix in PRESSED_SUFFIXES
            },
        },
        "verification": {
            "result": "passed",
            "models_compared": model_count,
            "reference": VERIFICATION_REFERENCE,
            "comparison": VERIFICATION_COMPARISON,
        },
    }


def _encoded_manifest(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, data: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"manifest already exists: {path}") from error
            temporary.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_pressed_manifest(
    database: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Audit ``database`` and atomically write a manifest to ``output``."""
    base = _canonical_base(database)
    requested_output = Path(output)
    if requested_output.is_symlink():
        raise PressedManifestError("manifest output must not be a symbolic link")
    destination = requested_output.parent.resolve(strict=False) / requested_output.name
    if destination == base:
        raise PressedManifestError(
            "manifest output must not overwrite the pressed database base"
        )
    if destination in {_member_path(base, suffix) for suffix in PRESSED_SUFFIXES}:
        raise PressedManifestError(
            "manifest output must not overwrite a pressed database member"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {destination}")
    manifest = build_pressed_manifest(base)
    _atomic_write(destination, _encoded_manifest(manifest), overwrite)
    return manifest


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PressedManifestError(f"duplicate manifest key: {key}")
        value[key] = item
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PressedManifestError(f"manifest {label} must be an object")
    return value


def _exact_fields(
    value: dict[str, Any], expected: set[str], label: str
) -> dict[str, Any]:
    if set(value) != expected:
        raise PressedManifestError(f"manifest {label} has invalid fields")
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PressedManifestError(f"manifest {label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise PressedManifestError(f"manifest {label} must be a {qualifier} integer")
    return cast(int, value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PressedManifestError(f"manifest {label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise PressedManifestError(
            f"manifest {label} must be a lowercase SHA256 digest"
        )
    return text


def _utc_timestamp(value: Any, label: str) -> str:
    text = _string(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as error:
        raise PressedManifestError(
            f"manifest {label} must be a canonical UTC timestamp"
        ) from error
    if parsed.utcoffset() != dt.timedelta(0):
        raise PressedManifestError(
            f"manifest {label} must be a canonical UTC timestamp"
        )
    canonical = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if text != canonical:
        raise PressedManifestError(
            f"manifest {label} must be a canonical UTC timestamp"
        )
    return text


def _parse_stat(value: Any, label: str) -> PressedFileStat:
    mapping = _mapping(value, label)
    if set(mapping) != set(_STAT_FIELDS):
        raise PressedManifestError(f"manifest {label} has invalid stat fields")
    return PressedFileStat(
        *(_integer(mapping[field], f"{label}.{field}") for field in _STAT_FIELDS)
    )


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise PressedManifestError(f"manifest is not a regular file: {path}")
    if file_stat.st_size > _MAX_MANIFEST_BYTES:
        raise PressedManifestError("manifest is unexpectedly large")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except PressedManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PressedManifestError(f"invalid manifest JSON: {error}") from error
    return _mapping(value, "root"), raw


def _parse_manifest(
    manifest: dict[str, Any], base: Path
) -> tuple[int, PressedStatToken, dict[str, str]]:
    _exact_fields(
        manifest,
        {
            "schema",
            "schema_version",
            "created_at_utc",
            "tool",
            "database",
            "verification",
        },
        "root",
    )
    if manifest.get("schema") != SCHEMA:
        raise PressedManifestError("unsupported pressed-manifest schema")
    if _integer(manifest.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise PressedManifestError("unsupported pressed-manifest schema version")
    _utc_timestamp(manifest.get("created_at_utc"), "created_at_utc")

    tool = _exact_fields(
        _mapping(manifest.get("tool"), "tool"),
        {
            "name",
            "version",
            "audit_algorithm",
            "source",
            "python",
            "pyhmmer_version",
            "private_abi",
            "platform",
        },
        "tool",
    )
    expected_tool_values = {
        "name": TOOL_NAME,
        "audit_algorithm": AUDIT_ALGORITHM,
        "pyhmmer_version": PYHMMER_VERSION,
    }
    for key, expected in expected_tool_values.items():
        if tool.get(key) != expected:
            raise PressedManifestError(
                f"manifest tool.{key} does not match this runtime"
            )
    if _integer(tool.get("version"), "tool.version") != TOOL_VERSION:
        raise PressedManifestError("manifest tool.version does not match this runtime")
    source_metadata = _exact_fields(
        _mapping(tool.get("source"), "tool.source"),
        {"filename", "sha256"},
        "tool.source",
    )
    source_filename, source_sha256 = _source_fingerprint()
    if (
        source_metadata.get("filename") != source_filename
        or _sha256(source_metadata.get("sha256"), "tool.source.sha256") != source_sha256
    ):
        raise PressedManifestError("manifest audit source does not match this runtime")
    python_metadata = _exact_fields(
        _mapping(tool.get("python"), "tool.python"),
        {"implementation", "version"},
        "tool.python",
    )
    expected_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    if python_metadata != expected_python:
        raise PressedManifestError(
            "manifest Python runtime does not match this runtime"
        )
    platform_metadata = _exact_fields(
        _mapping(tool.get("platform"), "tool.platform"),
        {"system", "machine", "byteorder", "pointer_bits"},
        "tool.platform",
    )
    expected_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
    }
    if platform_metadata != expected_platform:
        raise PressedManifestError("manifest platform does not match this runtime")
    private_abi = _exact_fields(
        _mapping(tool.get("private_abi"), "tool.private_abi"),
        {"algorithm", "sha256"},
        "tool.private_abi",
    )
    if private_abi.get("algorithm") != PRIVATE_ABI_ALGORITHM:
        raise PressedManifestError("unsupported private ABI fingerprint algorithm")
    recorded_abi = _sha256(private_abi.get("sha256"), "tool.private_abi.sha256")
    if recorded_abi != _private_abi_fingerprint():
        raise PressedManifestError(
            "manifest PyHMMER private ABI does not match this runtime"
        )

    database = _exact_fields(
        _mapping(manifest.get("database"), "database"),
        {"base_name", "model_count", "artifacts", "stat_token"},
        "database",
    )
    base_name = _string(database.get("base_name"), "database.base_name")
    if (
        base_name != Path(base_name).name
        or "/" in base_name
        or "\\" in base_name
        or base.name != base_name
    ):
        raise PressedManifestError(
            "manifest database base name does not match the requested database"
        )
    model_count = _integer(
        database.get("model_count"), "database.model_count", positive=True
    )
    artifacts = _mapping(database.get("artifacts"), "database.artifacts")
    stat_values = _mapping(database.get("stat_token"), "database.stat_token")
    if set(artifacts) != set(PRESSED_SUFFIXES):
        raise PressedManifestError("manifest has an invalid artifact set")
    if set(stat_values) != set(PRESSED_SUFFIXES):
        raise PressedManifestError("manifest has an invalid stat-token set")

    hashes: dict[str, str] = {}
    stats: list[PressedFileStat] = []
    for suffix in PRESSED_SUFFIXES:
        artifact = _exact_fields(
            _mapping(artifacts[suffix], f"database.artifacts.{suffix}"),
            {"filename", "size_bytes", "sha256"},
            f"database.artifacts.{suffix}",
        )
        if artifact.get("filename") != f"{base_name}.{suffix}":
            raise PressedManifestError(
                f"manifest artifact filename mismatch for {suffix}"
            )
        size = _integer(
            artifact.get("size_bytes"),
            f"database.artifacts.{suffix}.size_bytes",
        )
        hashes[suffix] = _sha256(
            artifact.get("sha256"),
            f"database.artifacts.{suffix}.sha256",
        )
        recorded_stat = _parse_stat(
            stat_values[suffix], f"database.stat_token.{suffix}"
        )
        if size != recorded_stat.size:
            raise PressedManifestError(
                f"manifest size and stat token disagree for {suffix}"
            )
        stats.append(recorded_stat)

    verification = _exact_fields(
        _mapping(manifest.get("verification"), "verification"),
        {"result", "models_compared", "reference", "comparison"},
        "verification",
    )
    if verification.get("result") != "passed":
        raise PressedManifestError("manifest does not record a passed audit")
    if verification.get("reference") != VERIFICATION_REFERENCE:
        raise PressedManifestError("manifest verification reference is unsupported")
    if verification.get("comparison") != VERIFICATION_COMPARISON:
        raise PressedManifestError("manifest verification comparison is unsupported")
    if (
        _integer(
            verification.get("models_compared"),
            "verification.models_compared",
            positive=True,
        )
        != model_count
    ):
        raise PressedManifestError(
            "manifest model count and verification count disagree"
        )
    return model_count, PressedStatToken(*stats), hashes


def validate_pressed_manifest(
    database: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    allow_hash_fallback: bool = True,
) -> ManifestValidation:
    """Authenticate a stable pressed database against a prior full audit.

    An exact stat-token match is the fast path.  If the database was copied,
    content hashes can authenticate the identical bytes under a new stat token.
    The returned token must be checked by the loader before and after it reads
    the database, closing the validation/load race.
    """
    _require_loaded_source_unchanged()
    base = _canonical_base(database)
    manifest, raw = _read_manifest(Path(manifest_path).resolve(strict=True))
    model_count, recorded_token, hashes = _parse_manifest(manifest, base)

    with _pinned_pressed_database(base) as (snapshot, current_token):
        used_hashes = current_token != recorded_token
        if used_hashes:
            if not allow_hash_fallback:
                raise PressedManifestError(
                    "pressed database stat token does not match the manifest"
                )
            for suffix, current_stat in zip(
                PRESSED_SUFFIXES, current_token, strict=True
            ):
                artifact = _member_path(snapshot, suffix)
                database_entry = manifest["database"]["artifacts"][suffix]
                if current_stat.size != database_entry["size_bytes"]:
                    raise PressedManifestError(
                        f"pressed database size mismatch for {suffix}"
                    )
                if _sha256_file(artifact) != hashes[suffix]:
                    raise PressedManifestError(
                        f"pressed database SHA256 mismatch for {suffix}"
                    )

    _require_loaded_source_unchanged()
    return ManifestValidation(
        canonical_base=base,
        model_count=model_count,
        stat_token=current_token,
        used_content_hashes=used_hashes,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "ManifestValidation",
    "PressedFileStat",
    "PressedManifestError",
    "PressedProfileAuditError",
    "PressedStatToken",
    "build_pressed_manifest",
    "create_pressed_manifest",
    "pressed_stat_token",
    "validate_pressed_manifest",
]
