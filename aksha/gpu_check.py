"""``aksha gpu-check``: would a GPU search run on the GPU, and if not, why.

The check reports, for every database manifest and every visible CUDA device,
which route ``aksha search --gpu-manifest`` would take:

* ``GPU``: the exact GPU path (bias math trusted, manifest authenticates the
  pressed files, device used by searches);
* ``GPU after self-attestation``: a search first runs the one-time exhaustive
  bias-math audit for this environment (``--attest`` runs it now);
* ``SSV only``: the GPU runs the SSV filter and every bias-dependent stage runs
  on the exact CPU path (bias math untrusted);
* ``refused``: the search would stop before searching (manifest or device).

The bias-math evaluation is the same function a GPU search calls at start.
Exit status: 0 every database ready on GPU, 2 pending self-attestation,
1 any fallback or refusal.
"""
from __future__ import annotations

import json
import os
import sys

from aksha import gpu_attestation

READY, PENDING, FALLBACK = "READY", "PENDING", "FALLBACK"


def _default_threads():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _route_when_trusted(search, device_ordinal):
    postfilter = search.gpu_postfilter_available()
    forward = postfilter and search.gpu_profile_forward_available()
    if postfilter and forward and device_ordinal == 0:
        return "exact GPU-through-Forward"
    if postfilter:
        return "exact post-filter + Forward"
    return "GPU SSV filter (installed runtime lacks the post-filter seam)"


def _manifest_rows(manifests, parsed_json, search):
    from plan7_gpu.pressed_manifest import validate_pressed_manifest

    rows = []
    entries = {entry.get("name"): entry for entry in parsed_json.get("db_urls", ())}
    for db_name, manifest_path in manifests.items():
        row = {"database": db_name, "manifest": manifest_path, "ok": False}
        entry = entries.get(db_name)
        if (entry is None or not entry.get("installed")
                or entry.get("molecule_type") != "protein"):
            row["error"] = f"{db_name} is not an installed protein database"
            rows.append(row)
            continue
        try:
            pressed_base = search.discover_pressed_base(entry["installation_dir"])
            validation = validate_pressed_manifest(pressed_base, manifest_path)
        except Exception as error:  # report every manifest problem, keep going
            row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            continue
        row.update(
            ok=True,
            pressed_base=os.fspath(pressed_base),
            model_count=getattr(validation, "model_count", None),
        )
        rows.append(row)
    return rows


def readiness(manifests, *, attest=False, threads=None, announce=print):
    """Return the readiness document (also used by tests)."""
    from aksha import initialize, search

    threads = threads or _default_threads()
    parsed_json = initialize.load_config()
    trust = gpu_attestation.evaluate_bias_trust(
        attest="auto" if attest else "report",
        threads=threads,
        announce=announce,
    )
    native = gpu_attestation.native_module()
    devices = []
    device_reason = ""
    provenance = None
    if native is not None and hasattr(native, "bias_cuda_devices"):
        listing = native.bias_cuda_devices()
        devices = listing["devices"]
        device_reason = listing["reason"]
    if native is not None:
        try:
            provenance = native.bias_environment_provenance()
        except Exception as error:
            device_reason = device_reason or str(error)
    manifest_rows = _manifest_rows(manifests, parsed_json, search) if manifests else []

    verdicts = []
    for row in manifest_rows:
        for device in devices or [None]:
            ordinal = None if device is None else device["device_ordinal"]
            verdict = {"database": row["database"], "device_ordinal": ordinal}
            if not row["ok"]:
                verdict.update(status=FALLBACK, route="refused",
                               why=f"manifest does not authenticate: {row['error']}")
            elif device is None:
                verdict.update(status=FALLBACK, route="refused",
                               why=f"no CUDA device visible ({device_reason or 'none'})")
            elif ordinal != 0:
                verdict.update(
                    status=None, route="not used",
                    why=("aksha search runs on CUDA device 0; select this "
                         "device with CUDA_VISIBLE_DEVICES"),
                )
            elif trust.state == "trusted":
                verdict.update(status=READY, route="GPU",
                               why=_route_when_trusted(search, ordinal))
            elif trust.state == "pending":
                verdict.update(status=PENDING, route="GPU after self-attestation",
                               why=trust.reason)
            elif trust.state == "untrusted":
                verdict.update(
                    status=FALLBACK, route="SSV only",
                    why=(f"bias math untrusted: {trust.reason}; "
                         f"{trust.hint or gpu_attestation.ATTEST_HINT}"),
                )
            else:
                verdict.update(status=READY, route="GPU",
                               why=f"{_route_when_trusted(search, ordinal)} "
                                   f"({trust.reason})")
            verdicts.append(verdict)

    statuses = [v["status"] for v in verdicts if v["status"] is not None]
    if not manifest_rows:
        # No manifest named: report the device and bias-math state alone.
        device0 = next((d for d in devices if d["device_ordinal"] == 0), None)
        if device0 is None:
            statuses = [FALLBACK]
        else:
            statuses = [{"trusted": READY, "pending": PENDING,
                         "unavailable": READY}.get(trust.state, FALLBACK)]
    if not statuses:
        overall = FALLBACK
    elif FALLBACK in statuses:
        overall = FALLBACK
    elif PENDING in statuses:
        overall = PENDING
    else:
        overall = READY
    return {
        "overall": overall,
        "bias_math": {
            "state": trust.state,
            "source": trust.source,
            "reason": trust.reason,
            "hint": trust.hint,
            "record": trust.record_path,
            "audit": trust.audit,
        },
        "host": None if provenance is None else {
            "glibc": _glibc(),
            "libm_build_id": (provenance.get("libm") or {}).get("gnu_build_id"),
            "cpu": provenance.get("host"),
        },
        "devices": devices,
        "device_reason": device_reason,
        "databases": manifest_rows,
        "verdicts": verdicts,
        "self_attest_mode": gpu_attestation.self_attest_mode(),
        "attestation_cache": os.fspath(gpu_attestation.cache_directory()),
    }


def _glibc():
    try:
        return os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        return None


def format_report(document):
    lines = ["Aksha GPU readiness"]
    host = document["host"]
    if host is not None:
        cpu = host.get("cpu") or {}
        lines.append(
            f"  host: {host.get('glibc')}, libm build ID {host.get('libm_build_id')}, "
            f"CPU family {cpu.get('family')} model {cpu.get('model')} "
            f"stepping {cpu.get('stepping')}"
        )
    bias = document["bias_math"]
    text = f"  bias math: {bias['state']}"
    if bias["source"]:
        text += f" ({bias['source']})"
    elif bias["reason"]:
        text += f" ({bias['reason']})"
    lines.append(text)
    if bias["state"] == "untrusted" and bias["hint"]:
        lines.append(f"    {bias['hint']}")
    lines.append(f"  self-attestation: {document['self_attest_mode']}, cache "
                 f"{document['attestation_cache']}")
    if document["devices"]:
        for device in document["devices"]:
            identity = device["identity"] or {}
            used = "  [used by aksha search]" if device["device_ordinal"] == 0 else ""
            cc = identity.get("compute_capability") or ["?", "?"]
            lines.append(
                f"  GPU {device['device_ordinal']}: {identity.get('name', '?')}, "
                f"CC {cc[0]}.{cc[1]}, target {device['target']}{used}"
            )
    else:
        lines.append(f"  GPU: none visible ({document['device_reason'] or 'no device'})")
    for row in document["databases"]:
        if row["ok"]:
            lines.append(f"  {row['database']}: manifest {row['manifest']} authenticates "
                         f"{row['pressed_base']} ({row['model_count']} models)")
        else:
            lines.append(f"  {row['database']}: manifest {row['manifest']}: {row['error']}")
    for verdict in document["verdicts"]:
        device = ("no GPU" if verdict["device_ordinal"] is None
                  else f"GPU {verdict['device_ordinal']}")
        lines.append(f"  {verdict['database']} on {device}: {verdict['route']} -- "
                     f"{verdict['why']}")
    lines.append(f"overall: {document['overall']}")
    return "\n".join(lines)


def main(args):
    from aksha import search

    manifests = search.parse_gpu_manifest_mappings(getattr(args, "gpu_manifest", ()))
    document = readiness(
        manifests,
        attest=args.attest,
        threads=args.threads,
        announce=(lambda line: print(line, file=sys.stderr)) if args.json else print,
    )
    if args.json:
        print(json.dumps(document, indent=1, sort_keys=True, default=str))
    else:
        print(format_report(document))
    return {READY: 0, PENDING: 2, FALLBACK: 1}[document["overall"]]
