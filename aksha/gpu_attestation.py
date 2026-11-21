"""Trust in the GPU bias math, and the GPU readiness check.

The exact GPU path evaluates ``log`` on the device while the CPU pipeline
calls the host libm ``log``.  GPU results equal CPU results only where the two
agree, so ``plan7_gpu`` gates every bias-dependent GPU stage on an attested
environment.  Two kinds of attestation exist:

* built-in: the exact glibc/libm build and CPU signature that were audited
  when a GPU target was admitted (compiled into the CUDA extension);
* self-attestation: the same exhaustive audit -- every positive finite
  binary32 input, device versus host ``log``, zero mismatches required -- run
  by ``plan7_gpu`` itself the first time Aksha meets a new environment.  The
  result is cached under a key made of everything the audit depends on (CUDA
  extension, libm build ID, glibc, CPU identity, device product, CUDA runtime
  and driver), so an operating-system or driver update triggers a new audit
  instead of silently disabling the GPU.

An environment that is not attested still completes every search exactly:
the GPU runs only the SSV filter and every bias-dependent stage runs on the
CPU, with one warning line naming what is untrusted and how to restore GPU
use.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path

SELF_ATTEST_ENV = "AKSHA_GPU_SELF_ATTEST"
CACHE_DIR_ENV = "AKSHA_GPU_ATTESTATION_DIR"
RECORD_SCHEMA = "aksha-gpu-bias-attestation-v1"
ATTEST_HINT = (
    "to restore GPU use, run `aksha gpu-check --attest` inside a GPU allocation"
)
_STATUS_HINTS = {
    "cuda-device": (
        "GPU bias math needs an admitted device (NVIDIA H200, or RTX 2080 Ti) "
        "with CUDA runtime 12.5"
    ),
    "float-environment": (
        "keep the process floating-point mode at round-to-nearest with FTZ "
        "and DAZ off"
    ),
    "host-features": (
        "GPU bias math needs an Intel x86-64 host with AVX2 and AVX-512 "
        "F/DQ/CD/BW/VL"
    ),
    "libm-unavailable": "the host libm build ID must be readable",
    "unsupported-platform": "GPU bias math needs Linux x86-64 with glibc",
}

_MODE_ALIASES = {
    "": "auto", "auto": "auto", "1": "auto", "yes": "auto", "on": "auto",
    "never": "never", "0": "never", "no": "never", "off": "never",
    "force": "force",
}


def self_attest_mode(environ=None):
    """Return ``auto`` (default), ``never`` or ``force`` from the environment."""
    environ = os.environ if environ is None else environ
    raw = environ.get(SELF_ATTEST_ENV, "auto")
    mode = _MODE_ALIASES.get(raw.strip().lower())
    if mode is None:
        raise ValueError(
            f"{SELF_ATTEST_ENV} must be auto, never or force (got {raw!r})"
        )
    return mode


def cache_directory(environ=None):
    """Directory of cached self-attestation records."""
    environ = os.environ if environ is None else environ
    explicit = environ.get(CACHE_DIR_ENV)
    if explicit:
        return Path(explicit).expanduser()
    base = environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "aksha" / "gpu-attestation"


def native_module():
    """Return ``plan7_gpu._native`` or ``None`` when the CUDA wheel is absent."""
    try:
        from plan7_gpu import _native
    except ImportError:
        return None
    return _native


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_key(native, provenance=None):
    """Everything the exhaustive audit result depends on, or ``None``.

    A change in any field (a rebuilt CUDA extension, a new libm or glibc, a
    different CPU, device product, CUDA runtime or driver) selects a different
    cache record and therefore a fresh audit.
    """
    if provenance is None:
        provenance = native.bias_environment_provenance()
    cuda = provenance.get("cuda")
    host = provenance.get("host")
    build_id = (provenance.get("libm") or {}).get("gnu_build_id")
    if cuda is None or host is None or build_id is None:
        return None
    try:
        glibc = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        glibc = None
    return {
        "schema": RECORD_SCHEMA,
        "algorithm": native.BIAS_ATTESTATION_ALGORITHM,
        "native_extension_sha256": _sha256_file(native.__file__),
        "glibc": glibc,
        "libm_build_id": build_id,
        "host": host,
        "cuda": {
            "target": provenance.get("target"),
            "name": cuda["name"],
            "compute_capability": cuda["compute_capability"],
            "multiprocessor_count": cuda["multiprocessor_count"],
            "runtime_version": cuda["runtime_version"],
            "driver_version": cuda["driver_version"],
            "cudart_version": cuda["cudart_version"],
            "nvcc": cuda["nvcc"],
        },
    }


def key_digest(key):
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_path(key, environ=None):
    return cache_directory(environ) / f"{key_digest(key)}.json"


def load_record(path, key):
    """Return ``(record, why_not)`` for a cache file that matches ``key``."""
    try:
        record = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None, "no cached self-attestation"
    except (OSError, ValueError) as error:
        return None, f"unreadable cached self-attestation {path}: {error}"
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        return None, f"cached self-attestation {path} has another schema"
    if record.get("key") != key:
        return None, f"cached self-attestation {path} is for another environment"
    audit = record.get("audit")
    required = (
        "libm_build_id", "host", "cuda_target", "algorithm_version",
        "inputs_checked", "float_mismatches",
    )
    if not isinstance(audit, dict) or any(name not in audit for name in required):
        return None, f"cached self-attestation {path} is incomplete"
    return record, ""


def write_record(path, key, audit):
    """Atomically write an audit record (successful or not)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cuda = audit.get("cuda") or {}
    record = {
        "schema": RECORD_SCHEMA,
        "key": key,
        "audit": audit,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "hostname": socket.gethostname(),
            "device_uuid": cuda.get("uuid"),
            "device_pci_bus_address": cuda.get("pci_bus_address"),
        },
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=".attestation-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, indent=1, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return record


@dataclass
class BiasTrust:
    """Outcome of the bias-math trust evaluation for the current CUDA device.

    ``state`` is ``trusted``, ``untrusted``, ``pending`` (a search would run
    the one-time audit; readiness checks only) or ``unavailable`` (no CUDA
    extension or no trust probe: GPU routing is left unchanged).
    """

    state: str
    source: str | None
    reason: str
    hint: str = ""
    record_path: str | None = None
    audit: dict | None = None

    @property
    def trusted(self):
        return {"trusted": True, "untrusted": False}.get(self.state)

    def search_line(self):
        """The single line a GPU search prints about bias-math trust."""
        if self.state == "trusted":
            return f"GPU bias math trusted ({self.source})."
        if self.state == "untrusted":
            hint = self.hint or ATTEST_HINT
            return (
                "WARNING: GPU bias math is untrusted in this environment "
                f"({self.reason}). The GPU runs only the SSV filter; the bias "
                "filter, Viterbi, Forward and domain stages run on the exact "
                f"CPU path; {hint}."
            )
        if self.state == "pending":
            return f"GPU bias math pending self-attestation ({self.reason})."
        return ""


def _audit_summary(audit):
    text = (
        f"{audit['float_mismatches']:,} binary32 mismatches over "
        f"{audit['inputs_checked']:,} inputs"
    )
    first = audit.get("first_float_mismatch")
    if first:
        text += (
            f", first at input {first['input']} (host {first['host']}, "
            f"device {first['device']})"
        )
    return text


def evaluate_bias_trust(*, attest="auto", threads=1, announce=print,
                        environ=None):
    """Decide whether the GPU bias math may be used on the current device.

    ``attest``: ``auto`` runs the exhaustive audit when no valid cached record
    exists; ``force`` always reruns it; ``never`` never runs it; ``report``
    (readiness check) never runs it and reports ``pending`` where a search
    would.
    """
    if attest not in ("auto", "force", "never", "report"):
        raise ValueError(f"unknown attestation mode {attest!r}")
    native = native_module()
    if native is None or not hasattr(native, "bias_environment_attested"):
        return BiasTrust(
            "unavailable", None,
            "the plan7_gpu CUDA extension does not expose bias-math trust",
        )
    if not hasattr(native, "bias_environment_status"):
        attested, reason = native.bias_environment_attested()
        if attested:
            return BiasTrust("trusted", "built-in attestation", "")
        return BiasTrust(
            "untrusted", None, reason,
            hint=(
                "the installed aksha-cuda12 cannot self-attest a new "
                "environment; install one that can"
            ),
        )
    status = native.bias_environment_status()
    if status["attested"]:
        source = (
            "built-in attestation"
            if status["source"] == "built-in"
            else "self-attested earlier in this process"
        )
        return BiasTrust("trusted", source, "")
    if not status["self_attestation_applicable"]:
        return BiasTrust(
            "untrusted", None, status["reason"],
            hint=_STATUS_HINTS.get(status["name"], ATTEST_HINT),
        )

    provenance = native.bias_environment_provenance()
    key = environment_key(native, provenance)
    if key is None:
        return BiasTrust("untrusted", None, status["reason"])
    path = record_path(key, environ)
    stale = ""
    if attest != "force":
        record, _ = load_record(path, key)
        if record is not None:
            audit = record["audit"]
            registered, why = native.bias_register_attestation(
                audit["libm_build_id"],
                audit["host"],
                int(audit["cuda_target"]),
                int(audit["algorithm_version"]),
                int(audit["inputs_checked"]),
                int(audit["float_mismatches"]),
            )
            if registered:
                return BiasTrust(
                    "trusted",
                    f"self-attested {record.get('created_at_utc', '?')}, "
                    f"cached at {path}",
                    "", record_path=os.fspath(path), audit=audit,
                )
            if int(audit["float_mismatches"]) != 0:
                return BiasTrust(
                    "untrusted", None,
                    f"{status['reason']}; its self-attestation found "
                    f"{_audit_summary(audit)} (record {path})",
                    hint=(
                        "the device log differs from this host libm, so GPU "
                        "use needs a different libm or device; a forced "
                        f"re-audit is {SELF_ATTEST_ENV}=force"
                    ),
                    record_path=os.fspath(path), audit=audit,
                )
            stale = why
    if attest == "never":
        return BiasTrust(
            "untrusted", None,
            f"{status['reason']}; self-attestation is disabled by "
            f"{SELF_ATTEST_ENV}=never",
            hint=(
                f"to restore GPU use, unset {SELF_ATTEST_ENV} or run "
                "`aksha gpu-check --attest` inside a GPU allocation"
            ),
        )
    if attest == "report":
        detail = f"; cached record rejected: {stale}" if stale else ""
        return BiasTrust(
            "pending", None,
            f"{status['reason']}{detail}; a GPU search runs the one-time "
            "exhaustive self-attestation first (a few seconds)",
            record_path=os.fspath(path),
        )
    announce(
        "GPU bias math: running the one-time exhaustive self-attestation "
        f"for this environment ({native.BIAS_ATTESTATION_INPUT_COUNT:,} "
        "inputs, device versus host log)..."
    )
    host_threads = max(1, min(int(threads), os.cpu_count() or 1, 1024))
    try:
        audit = native.bias_self_attest(host_threads)
    except RuntimeError as error:
        return BiasTrust(
            "untrusted", None,
            f"{status['reason']}; self-attestation could not run: {error}",
        )
    note = ""
    try:
        write_record(path, key, audit)
    except OSError as error:
        note = f"; the record could not be cached ({error})"
    if audit["attested"]:
        return BiasTrust(
            "trusted",
            f"self-attested now in {audit['seconds']:.1f} s: "
            f"{_audit_summary(audit)}; record {path}{note}",
            "", record_path=os.fspath(path), audit=audit,
        )
    return BiasTrust(
        "untrusted", None,
        f"{status['reason']}; self-attestation found {_audit_summary(audit)}"
        f"{note}",
        hint=(
            "the device log differs from this host libm, so GPU use needs a "
            "different libm or device"
        ),
        record_path=os.fspath(path), audit=audit,
    )
