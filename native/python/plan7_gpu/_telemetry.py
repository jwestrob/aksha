"""Validation and presentation for opt-in route telemetry.

The native extension transports fixed-width counters, never result records or
journal layouts.  This module gives that package-private tuple a versioned,
defensive Python representation.  Reason counters are multi-label facts: one
row may increment more than one reason.
"""

from __future__ import annotations

import copy
from typing import Any


GENERATION_TELEMETRY_SCHEMA_VERSION = 2
_UINT64_MAX = (1 << 64) - 1

_METRIC_NAMES = (
    "model_length",
    "target_count",
    "target_residues",
    "f1_candidate_count",
    "f1_reject_count",
    "f1_logical_cells",
    "postfilter_logical_cells",
    "f2_pass_count",
    "forward_logical_cells",
    "forward_cpu_required_count",
    "forward_reject_count",
    "forward_pass_count",
    "backward_logical_cells",
    "backward_cpu_required_count",
    "backward_no_region_count",
    "backward_simple_count",
    "journal_row_count",
    "journal_region_count",
    "rescore_logical_cells",
    "rescore_cpu_required_count",
    "rescore_device_count",
    "rescore_region_count",
    "rescore_certified_ga_count",
)

_COUNT_METRICS = (
    "f1_candidate_count",
    "f1_reject_count",
    "f2_pass_count",
    "forward_cpu_required_count",
    "forward_reject_count",
    "forward_pass_count",
    "backward_cpu_required_count",
    "backward_no_region_count",
    "backward_simple_count",
    "rescore_cpu_required_count",
    "rescore_device_count",
    "rescore_region_count",
    "rescore_certified_ga_count",
)

_CELL_METRICS = (
    "f1_logical_cells",
    "postfilter_logical_cells",
    "forward_logical_cells",
    "backward_logical_cells",
    "rescore_logical_cells",
)

_REASON_FACTS = {
    "postfilter": (
        ("raw_f1_reject", 0x0001),
        ("msv_range_state", 0x0002),
        ("candidate_state_cpu", 0x0004),
        ("bias_input_status_nonzero", 0x0008),
        ("bias_filter_score_failed", 0x0010),
        ("bias_score_nonfinite", 0x0020),
        ("bias_cutoff_unresolved", 0x0040),
        ("viterbi_erange", 0x0080),
        ("viterbi_no_result_or_other_status", 0x0100),
        ("final_cpu_required", 0x0200),
        ("final_reject", 0x0400),
        ("final_pass", 0x0800),
        ("other_cpu_required", 0x1000),
        ("contract_fallback", 0x2000),
        ("full_msv_executed", 0x4000),
        ("viterbi_executed", 0x8000),
    ),
    "f2": (
        ("postfilter_not_pass_or_host_environment_unattested", 0x01),
        ("input_invalid", 0x02),
        ("msv_threshold_exceeded", 0x04),
        ("viterbi_threshold_exceeded", 0x08),
        ("pass", 0x10),
    ),
    "forward": (
        ("kernel_status_non_ok", 0x0001),
        ("target_empty", 0x0002),
        ("fwdsc_nonfinite", 0x0004),
        ("filtersc_nonfinite", 0x0008),
        ("tau_nonfinite", 0x0010),
        ("lambda_invalid", 0x0020),
        ("f3_reject", 0x0040),
        ("output_cap", 0x0080),
        ("survivor_gathered", 0x0100),
        ("other_cpu_required", 0x0200),
    ),
    "backward_domain": (
        ("target_empty", 0x0001),
        ("forward_special_nonfinite", 0x0002),
        ("forward_scale_invalid", 0x0004),
        ("host_float_environment_invalid", 0x0008),
        ("mode_or_nj_unsupported", 0x0010),
        ("work_cap", 0x0020),
        ("workspace_cap", 0x0040),
        ("terminal_score_invalid", 0x0080),
        ("posterior_or_backward_score_nonfinite", 0x0100),
        ("nexpected_invalid", 0x0200),
        ("has_own_scales", 0x0400),
        ("threshold_uncertain", 0x0800),
        ("multidomain", 0x1000),
        ("no_regions", 0x2000),
        ("simple", 0x4000),
        ("region_output_cap", 0x8000),
        ("other_cpu_required", 0x00010000),
        ("final_cpu_required", 0x00020000),
    ),
    "rescore": (
        ("global_compact_budget", 0x00000001),
        ("own_scales", 0x00000002),
        ("region_work_cap", 0x00000004),
        ("row_work_cap", 0x00000008),
        ("matrix_cap", 0x00000010),
        ("trace_cap", 0x00000020),
        ("run_work_cap", 0x00000040),
        ("forward_score_invalid", 0x00000080),
        ("backward_score_invalid", 0x00000100),
        ("scaleproduct_invalid", 0x00000200),
        ("null2_or_correction_invalid", 0x00000400),
        ("oa_score_invalid", 0x00000800),
        ("trace_capacity_exhausted", 0x00001000),
        ("trace_iteration_invalid", 0x00002000),
        ("trace_predecessor_invalid", 0x00004000),
        ("trace_coordinates_invalid", 0x00008000),
        ("identity_mismatch", 0x00010000),
        ("host_result_invalid", 0x00020000),
        ("host_trace_invalid", 0x00040000),
        ("host_null2_invalid", 0x00080000),
        ("row_atomic_propagation", 0x00100000),
        ("device_result", 0x00200000),
        ("other_cpu_required", 0x00400000),
        ("upstream_backward_own_scales", 0x00800000),
        ("final_cpu_required", 0x01000000),
        ("certified_ga_reject", 0x02000000),
    ),
}

_STAGES = tuple(_REASON_FACTS)


def validate_generation_reason_fact_layout(value: Any) -> tuple[tuple[int, ...], ...]:
    """Bind compiled native enum values to the version-1 Python registry."""
    expected = tuple(
        tuple(bit for _name, bit in _REASON_FACTS[stage])
        for stage in _STAGES
    )
    if type(value) is not tuple or any(type(row) is not tuple for row in value):
        raise ImportError("native generation reason layout is not immutable")
    if value != expected:
        raise ImportError("native generation reason layout differs from registry")
    return expected


def _uint64(value: Any, field: str) -> int:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        raise ValueError(f"{field} must be an exact uint64 integer")
    return value


def _checked_sum(values: list[int], field: str) -> int:
    total = sum(values)
    if total > _UINT64_MAX:
        raise OverflowError(f"{field} total exceeds uint64")
    return total


def _batch_identity(
    value: Any, field: str, *, allow_unbound: bool
) -> tuple[int, int, int] | None:
    if value is None:
        if allow_unbound:
            return None
        raise ValueError(f"{field} is unbound")
    if type(value) is not tuple or len(value) != 3:
        raise ValueError(f"{field} must be an exact identity tuple")
    identity = tuple(
        _uint64(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if any(item == 0 for item in identity):
        raise ValueError(f"{field} contains a zero identity")
    return identity


def _batch_identity_record(
    identity: tuple[int, int, int] | None,
) -> dict[str, int] | None:
    if identity is None:
        return None
    return {
        "session_id": identity[0],
        "selection_id": identity[1],
        "batch_generation": identity[2],
    }


def _batch_identity_tuple(value: Any, field: str) -> tuple[int, int, int] | None:
    if value is None:
        return None
    names = ("session_id", "selection_id", "batch_generation")
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{field} fields changed")
    return _batch_identity(
        tuple(value[name] for name in names), field, allow_unbound=False
    )


def build_generation_statistics(
    schema_version: int,
    profile_count: int,
    target_count: int,
    total_target_residues: int,
    profile_records: tuple[
        tuple[
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[tuple[int, ...], ...],
        ],
        ...,
    ],
    forward_call_reason_facts: int,
    journal_allocation_bytes: int,
    native_totals: dict[str, Any],
    batch_identity: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Validate the fixed native transport and return canonical evidence."""
    if (
        type(schema_version) is not int
        or schema_version != GENERATION_TELEMETRY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported generation telemetry schema")
    bound_identity = _batch_identity(
        batch_identity, "generation batch identity", allow_unbound=True
    )
    profile_count = _uint64(profile_count, "profile_count")
    target_count = _uint64(target_count, "target_count")
    total_target_residues = _uint64(
        total_target_residues, "total_target_residues"
    )
    if type(profile_records) is not tuple or len(profile_records) != profile_count:
        raise ValueError("generation telemetry profile count changed")
    if (
        type(forward_call_reason_facts) is not int
        or not 0 <= forward_call_reason_facts <= 0x01
    ):
        raise ValueError("invalid Forward call reason facts")
    journal_allocation_bytes = _uint64(
        journal_allocation_bytes, "journal_allocation_bytes"
    )
    if type(native_totals) is not dict:
        raise TypeError("native_totals must be a dict")

    profiles: list[dict[str, Any]] = []
    metric_columns: dict[str, list[int]] = {name: [] for name in _METRIC_NAMES}
    total_reasons = {
        stage: [0] * len(_REASON_FACTS[stage]) for stage in _STAGES
    }
    total_reason_cells = {
        stage: [0] * len(_REASON_FACTS[stage]) for stage in _STAGES
    }
    for profile_index, record in enumerate(profile_records):
        if type(record) is not tuple or len(record) != 3:
            raise ValueError("invalid generation telemetry profile record")
        metrics, reason_rows, reason_cell_rows = record
        if type(metrics) is not tuple or len(metrics) != len(_METRIC_NAMES):
            raise ValueError("invalid generation metric row")
        if type(reason_rows) is not tuple or len(reason_rows) != len(_STAGES):
            raise ValueError("invalid generation reason row")
        if (
            type(reason_cell_rows) is not tuple
            or len(reason_cell_rows) != len(_STAGES)
        ):
            raise ValueError("invalid generation reason-cell row")
        metric = {
            name: _uint64(value, f"profiles[{profile_index}].{name}")
            for name, value in zip(_METRIC_NAMES, metrics, strict=True)
        }
        if metric["target_count"] != target_count:
            raise ValueError("per-profile target count changed")
        if metric["target_residues"] != total_target_residues:
            raise ValueError("per-profile target residue count changed")
        if metric["f1_candidate_count"] + metric["f1_reject_count"] != target_count:
            raise ValueError("F1 candidate partition changed")
        if metric["f2_pass_count"] > metric["f1_candidate_count"]:
            raise ValueError("F2 pass count exceeds F1 candidates")
        if (
            metric["forward_cpu_required_count"]
            + metric["forward_reject_count"]
            + metric["forward_pass_count"]
            != metric["f2_pass_count"]
        ):
            raise ValueError("Forward route partition changed")
        if (
            metric["backward_cpu_required_count"]
            + metric["backward_no_region_count"]
            + metric["backward_simple_count"]
            != metric["forward_pass_count"]
        ):
            raise ValueError("Backward/domain route partition changed")
        if metric["journal_row_count"] != metric["forward_pass_count"]:
            raise ValueError("continuation journal row attribution changed")
        if (
            metric["rescore_cpu_required_count"]
            + metric["rescore_device_count"]
            + metric["rescore_certified_ga_count"]
            != metric["rescore_region_count"]
        ):
            raise ValueError("rescore route partition changed")
        if metric["rescore_region_count"] not in (0, metric["journal_region_count"]):
            raise ValueError("rescore/journal region attribution changed")

        reason_counts: dict[str, tuple[tuple[str, int], ...]] = {}
        reason_logical_cells: dict[str, tuple[tuple[str, int], ...]] = {}
        populations = {
            "postfilter": metric["f1_candidate_count"],
            "f2": metric["f1_candidate_count"],
            "forward": metric["f2_pass_count"],
            "backward_domain": metric["forward_pass_count"],
            "rescore": metric["rescore_region_count"],
        }
        stage_cells = {
            "postfilter": metric["postfilter_logical_cells"],
            "f2": 0,
            "forward": metric["forward_logical_cells"],
            "backward_domain": metric["backward_logical_cells"],
            "rescore": metric["rescore_logical_cells"],
        }
        for stage_index, stage in enumerate(_STAGES):
            row = reason_rows[stage_index]
            cell_row = reason_cell_rows[stage_index]
            facts = _REASON_FACTS[stage]
            if type(row) is not tuple or len(row) != len(facts):
                raise ValueError(f"invalid {stage} reason row")
            if type(cell_row) is not tuple or len(cell_row) != len(facts):
                raise ValueError(f"invalid {stage} reason-cell row")
            named: list[tuple[str, int]] = []
            named_cells: list[tuple[str, int]] = []
            for reason_index, ((name, _bit), value, cells) in enumerate(
                zip(facts, row, cell_row, strict=True)
            ):
                count = _uint64(
                    value,
                    f"profiles[{profile_index}].{stage}.{name}",
                )
                if count > populations[stage]:
                    raise ValueError(f"{stage} reason count exceeds its population")
                total_reasons[stage][reason_index] += count
                if total_reasons[stage][reason_index] > _UINT64_MAX:
                    raise OverflowError(f"{stage} reason total exceeds uint64")
                cells = _uint64(
                    cells,
                    f"profiles[{profile_index}].{stage}.{name}.logical_cells",
                )
                if cells > stage_cells[stage]:
                    raise ValueError(
                        f"{stage} reason cells exceed stage work cells"
                    )
                total_reason_cells[stage][reason_index] += cells
                if total_reason_cells[stage][reason_index] > _UINT64_MAX:
                    raise OverflowError(
                        f"{stage} reason-cell total exceeds uint64"
                    )
                if count:
                    named.append((name, count))
                if cells:
                    named_cells.append((name, cells))
            reason_counts[stage] = tuple(named)
            reason_logical_cells[stage] = tuple(named_cells)

        post = dict(reason_counts["postfilter"])
        if (
            post.get("final_cpu_required", 0)
            + post.get("final_reject", 0)
            + post.get("final_pass", 0)
            != metric["f1_candidate_count"]
        ):
            raise ValueError("postfilter final-action facts do not partition rows")
        full_msv_cells = dict(reason_logical_cells["postfilter"]).get(
            "full_msv_executed", 0
        )
        viterbi_cells = dict(reason_logical_cells["postfilter"]).get(
            "viterbi_executed", 0
        )
        if full_msv_cells + viterbi_cells != metric["postfilter_logical_cells"]:
            raise ValueError("postfilter execution facts do not partition work")
        f2_reasons = dict(reason_counts["f2"])
        # ``msv_threshold_exceeded`` is a transition fact, not a terminal
        # outcome: a row can fail MSV and subsequently pass Viterbi.  The
        # other four facts are the exact terminal partition.
        if (
            f2_reasons.get(
                "postfilter_not_pass_or_host_environment_unattested", 0
            )
            + f2_reasons.get("input_invalid", 0)
            + f2_reasons.get("viterbi_threshold_exceeded", 0)
            + f2_reasons.get("pass", 0)
            != metric["f1_candidate_count"]
        ):
            raise ValueError("F2 terminal source branches do not partition rows")
        if (
            f2_reasons.get("viterbi_threshold_exceeded", 0)
            > f2_reasons.get("msv_threshold_exceeded", 0)
            or f2_reasons.get("msv_threshold_exceeded", 0)
            > (
                f2_reasons.get("viterbi_threshold_exceeded", 0)
                + f2_reasons.get("pass", 0)
            )
        ):
            raise ValueError("F2 MSV/Viterbi transition attribution changed")
        if f2_reasons.get("pass", 0) != metric["f2_pass_count"]:
            raise ValueError("F2 pass facts changed")
        forward_reasons = dict(reason_counts["forward"])
        if forward_reasons.get("f3_reject", 0) != metric["forward_reject_count"]:
            raise ValueError("Forward reject facts changed")
        if forward_reasons.get("survivor_gathered", 0) != metric["forward_pass_count"]:
            raise ValueError("Forward survivor facts changed")
        backward_reasons = dict(reason_counts["backward_domain"])
        if (
            backward_reasons.get("final_cpu_required", 0)
            != metric["backward_cpu_required_count"]
            or backward_reasons.get("no_regions", 0)
            != metric["backward_no_region_count"]
            or backward_reasons.get("simple", 0)
            != metric["backward_simple_count"]
        ):
            raise ValueError("Backward terminal route facts changed")
        explained_backward_cpu = sum(
            count
            for name, count in backward_reasons.items()
            if name not in {
                "final_cpu_required", "no_regions", "simple"
            }
        )
        if (
            metric["backward_cpu_required_count"] != 0
            and explained_backward_cpu == 0
        ):
            raise ValueError("Backward CPU routes lack source explanations")
        rescore_reasons = dict(reason_counts["rescore"])
        if (
            rescore_reasons.get("device_result", 0)
            != metric["rescore_device_count"]
            or rescore_reasons.get("final_cpu_required", 0)
            != metric["rescore_cpu_required_count"]
            or rescore_reasons.get("certified_ga_reject", 0)
            != metric["rescore_certified_ga_count"]
        ):
            raise ValueError("rescore terminal route facts changed")
        for name, value in metric.items():
            metric_columns[name].append(value)
        profiles.append(
            {
                "profile_index": profile_index,
                "model_length": metric["model_length"],
                "counts": {name: metric[name] for name in _COUNT_METRICS},
                "logical_cells": {name: metric[name] for name in _CELL_METRICS},
                "journal": {
                    "row_count": metric["journal_row_count"],
                    "region_count": metric["journal_region_count"],
                },
                "reason_counts": reason_counts,
                "reason_logical_cells": reason_logical_cells,
            }
        )

    totals = {
        name: _checked_sum(values, name)
        for name, values in metric_columns.items()
        if name not in {"model_length", "target_count", "target_residues"}
    }
    reason_totals = {
        stage: tuple(
            (name, total_reasons[stage][index])
            for index, (name, _bit) in enumerate(_REASON_FACTS[stage])
            if total_reasons[stage][index]
        )
        for stage in _STAGES
    }
    reason_cell_totals = {
        stage: tuple(
            (name, total_reason_cells[stage][index])
            for index, (name, _bit) in enumerate(_REASON_FACTS[stage])
            if total_reason_cells[stage][index]
        )
        for stage in _STAGES
    }
    postfilter_native = native_totals.get("postfilter")
    forward_native = native_totals.get("forward")
    backward_native = native_totals.get("backward_domain")
    rescore_native = native_totals.get("rescore")
    if (
        type(postfilter_native) is not dict
        or set(postfilter_native) != {
            "candidate_count",
            "full_msv_execution_count",
            "viterbi_execution_count",
            "full_msv_work_cells",
            "viterbi_work_cells",
            "work_cells",
        }
        or type(forward_native) is not dict
        or set(forward_native) != {
            "candidate_count",
            "survivor_count",
            "work_cells",
            "output_cap_fallback_count",
        }
        or type(backward_native) is not dict
        or set(backward_native) != {
            "candidate_count",
            "device_result_count",
            "cpu_required_count",
            "work_cells",
        }
        or set(native_totals) != {
            "postfilter", "forward", "backward_domain", "rescore"
        }
    ):
        raise ValueError("native generation totals are incomplete")
    post_reasons = dict(reason_totals["postfilter"])
    post_cells = dict(reason_cell_totals["postfilter"])
    if _uint64(
        postfilter_native.get("candidate_count"),
        "native postfilter candidates",
    ) != totals["f1_candidate_count"]:
        raise ValueError("native postfilter candidate total changed")
    if _uint64(
        postfilter_native.get("full_msv_execution_count"),
        "native full-MSV executions",
    ) != post_reasons.get("full_msv_executed", 0):
        raise ValueError("native full-MSV execution attribution changed")
    if _uint64(
        postfilter_native.get("viterbi_execution_count"),
        "native Viterbi executions",
    ) != post_reasons.get("viterbi_executed", 0):
        raise ValueError("native Viterbi execution attribution changed")
    if _uint64(
        postfilter_native.get("full_msv_work_cells"),
        "native full-MSV cells",
    ) != post_cells.get("full_msv_executed", 0):
        raise ValueError("native full-MSV work-cell attribution changed")
    if _uint64(
        postfilter_native.get("viterbi_work_cells"),
        "native Viterbi cells",
    ) != post_cells.get("viterbi_executed", 0):
        raise ValueError("native Viterbi work-cell attribution changed")
    if _uint64(
        postfilter_native.get("work_cells"),
        "native postfilter cells",
    ) != totals["postfilter_logical_cells"]:
        raise ValueError("native postfilter work-cell attribution changed")
    if _uint64(forward_native.get("candidate_count"), "native Forward candidates") != totals["f2_pass_count"]:
        raise ValueError("native Forward candidate total changed")
    if _uint64(forward_native.get("survivor_count"), "native Forward survivors") != totals["forward_pass_count"]:
        raise ValueError("native Forward survivor total changed")
    if _uint64(forward_native.get("work_cells"), "native Forward cells") != totals["forward_logical_cells"]:
        raise ValueError("native Forward work-cell attribution changed")
    if _uint64(
        forward_native.get("output_cap_fallback_count"),
        "native Forward output-cap fallbacks",
    ) != dict(reason_totals["forward"]).get("output_cap", 0):
        raise ValueError("native Forward output-cap attribution changed")
    if _uint64(backward_native.get("candidate_count"), "native Backward candidates") != totals["forward_pass_count"]:
        raise ValueError("native Backward candidate total changed")
    if _uint64(
        backward_native.get("cpu_required_count"),
        "native Backward CPU routes",
    ) != totals["backward_cpu_required_count"]:
        raise ValueError("native Backward CPU-route attribution changed")
    if _uint64(
        backward_native.get("device_result_count"),
        "native Backward device routes",
    ) != (
        totals["backward_no_region_count"]
        + totals["backward_simple_count"]
    ):
        raise ValueError("native Backward device-route attribution changed")
    if _uint64(backward_native.get("work_cells"), "native Backward cells") != totals["backward_logical_cells"]:
        raise ValueError("native Backward work-cell attribution changed")
    if rescore_native is None:
        if totals["rescore_region_count"] != 0:
            raise ValueError("rescore telemetry exists without native totals")
    else:
        if type(rescore_native) is not dict or set(rescore_native) != {
            "region_count",
            "device_result_count",
            "cpu_required_count",
            "certified_ga_region_count",
            "certified_ga_row_count",
            "certified_ga_skipped_work_cells",
            "ga_classification_ms",
            "work_cells",
        }:
            raise ValueError("native rescore totals are invalid")
        if _uint64(rescore_native.get("region_count"), "native rescore regions") != totals["rescore_region_count"]:
            raise ValueError("native rescore region total changed")
        if _uint64(
            rescore_native.get("cpu_required_count"),
            "native rescore CPU routes",
        ) != totals["rescore_cpu_required_count"]:
            raise ValueError("native rescore CPU-route attribution changed")
        if _uint64(
            rescore_native.get("device_result_count"),
            "native rescore device routes",
        ) != totals["rescore_device_count"]:
            raise ValueError("native rescore device-route attribution changed")
        if _uint64(
            rescore_native.get("certified_ga_region_count"),
            "native rescore GA rejects",
        ) != totals["rescore_certified_ga_count"]:
            raise ValueError("native rescore GA-reject attribution changed")
        ga_rows = _uint64(
            rescore_native.get("certified_ga_row_count"),
            "native rescore GA rows",
        )
        ga_cells = _uint64(
            rescore_native.get("certified_ga_skipped_work_cells"),
            "native rescore GA skipped cells",
        )
        work_cells = _uint64(
            rescore_native.get("work_cells"), "native rescore cells"
        )
        ga_ms = rescore_native.get("ga_classification_ms")
        if ga_rows > totals["rescore_certified_ga_count"]:
            raise ValueError("native rescore GA row count exceeds region count")
        if ga_cells > work_cells:
            raise ValueError("native rescore GA skipped cells exceed admitted work")
        if type(ga_ms) is not float or ga_ms != ga_ms or not 0.0 <= ga_ms < float("inf"):
            raise ValueError("native rescore GA classification time is invalid")
        if work_cells != totals["rescore_logical_cells"]:
            raise ValueError("native rescore work-cell attribution changed")

    return {
        "schema_version": GENERATION_TELEMETRY_SCHEMA_VERSION,
        "scope": "generation",
        "batch_identity": _batch_identity_record(bound_identity),
        "profile_count": profile_count,
        "target_count": target_count,
        "total_target_residues": total_target_residues,
        "reason_fact_bits": copy.deepcopy(_REASON_FACTS),
        "call_reason_facts": {
            "forward": (
                (("contract_fallback", 0x01),)
                if forward_call_reason_facts & 0x01
                else ()
            )
        },
        "profiles": tuple(profiles),
        "totals": totals,
        "reason_totals": reason_totals,
        "reason_logical_cell_totals": reason_cell_totals,
        "journal": {
            "allocation_bytes": journal_allocation_bytes,
            "profile_attribution": "row and region counts only",
        },
        "native_totals": copy.deepcopy(native_totals),
    }


def validate_generation_statistics(value: Any) -> dict[str, Any]:
    """Fail closed on a stored sidecar and return a private canonical copy."""
    if (
        type(value) is not dict
        or value.get("schema_version")
        != GENERATION_TELEMETRY_SCHEMA_VERSION
    ):
        raise ValueError("invalid generation statistics schema")
    required = {
        "schema_version",
        "scope",
        "batch_identity",
        "profile_count",
        "target_count",
        "total_target_residues",
        "reason_fact_bits",
        "call_reason_facts",
        "profiles",
        "totals",
        "reason_totals",
        "reason_logical_cell_totals",
        "journal",
        "native_totals",
    }
    if set(value) != required or value.get("scope") != "generation":
        raise ValueError("generation statistics fields changed")
    if value.get("reason_fact_bits") != _REASON_FACTS:
        raise ValueError("generation reason fact registry changed")
    profiles = value.get("profiles")
    if type(profiles) is not tuple:
        raise ValueError("generation profile records are not immutable")
    if len(profiles) != value.get("profile_count"):
        raise ValueError("generation profile records changed")

    profile_count = _uint64(value.get("profile_count"), "profile_count")
    target_count = _uint64(value.get("target_count"), "target_count")
    total_target_residues = _uint64(
        value.get("total_target_residues"), "total_target_residues"
    )
    batch_identity = _batch_identity_tuple(
        value.get("batch_identity"), "generation batch identity"
    )

    def expand_sparse(
        sparse: Any, stage: str, field: str
    ) -> tuple[int, ...]:
        if type(sparse) is not tuple:
            raise ValueError(f"{stage} {field} is not immutable")
        expected_names = tuple(name for name, _bit in _REASON_FACTS[stage])
        observed: dict[str, int] = {}
        order: list[str] = []
        for entry in sparse:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(f"invalid {stage} {field} entry")
            name, count = entry
            if type(name) is not str or name not in expected_names:
                raise ValueError(f"unknown {stage} {field} name")
            if name in observed:
                raise ValueError(f"duplicate {stage} {field} name")
            count = _uint64(count, f"{stage}.{name}.{field}")
            if count == 0:
                raise ValueError(f"zero {stage} {field} entry is not canonical")
            observed[name] = count
            order.append(name)
        if tuple(order) != tuple(name for name in expected_names if name in observed):
            raise ValueError(f"{stage} {field} order changed")
        return tuple(observed.get(name, 0) for name in expected_names)

    profile_records = []
    expected_profile_fields = {
        "profile_index",
        "model_length",
        "counts",
        "logical_cells",
        "journal",
        "reason_counts",
        "reason_logical_cells",
    }
    for profile_index, profile in enumerate(profiles):
        if (
            type(profile) is not dict
            or set(profile) != expected_profile_fields
            or profile.get("profile_index") != profile_index
        ):
            raise ValueError("generation profile fields or order changed")
        counts = profile.get("counts")
        cells = profile.get("logical_cells")
        journal = profile.get("journal")
        reason_counts = profile.get("reason_counts")
        reason_cells = profile.get("reason_logical_cells")
        if type(counts) is not dict or set(counts) != set(_COUNT_METRICS):
            raise ValueError("generation profile count fields changed")
        if type(cells) is not dict or set(cells) != set(_CELL_METRICS):
            raise ValueError("generation profile cell fields changed")
        if type(journal) is not dict or set(journal) != {
            "row_count",
            "region_count",
        }:
            raise ValueError("generation profile journal fields changed")
        if type(reason_counts) is not dict or set(reason_counts) != set(_STAGES):
            raise ValueError("generation profile reason fields changed")
        if type(reason_cells) is not dict or set(reason_cells) != set(_STAGES):
            raise ValueError("generation profile reason-cell fields changed")
        metric = {
            "model_length": profile.get("model_length"),
            "target_count": target_count,
            "target_residues": total_target_residues,
            **counts,
            **cells,
            "journal_row_count": journal.get("row_count"),
            "journal_region_count": journal.get("region_count"),
        }
        reason_rows = tuple(
            expand_sparse(reason_counts[stage], stage, "count")
            for stage in _STAGES
        )
        reason_cell_rows = tuple(
            expand_sparse(reason_cells[stage], stage, "logical_cells")
            for stage in _STAGES
        )
        profile_records.append(
            (
                tuple(metric[name] for name in _METRIC_NAMES),
                reason_rows,
                reason_cell_rows,
            )
        )

    call_reason_facts = value.get("call_reason_facts")
    if type(call_reason_facts) is not dict or set(call_reason_facts) != {
        "forward"
    }:
        raise ValueError("generation call reason fields changed")
    forward_call = call_reason_facts["forward"]
    if forward_call == ():
        forward_call_bits = 0
    elif forward_call == (("contract_fallback", 0x01),):
        forward_call_bits = 0x01
    else:
        raise ValueError("Forward call reason facts changed")
    journal_total = value.get("journal")
    if type(journal_total) is not dict or set(journal_total) != {
        "allocation_bytes",
        "profile_attribution",
    }:
        raise ValueError("generation journal fields changed")
    if journal_total.get("profile_attribution") != "row and region counts only":
        raise ValueError("generation journal attribution scope changed")

    rebuilt = build_generation_statistics(
        GENERATION_TELEMETRY_SCHEMA_VERSION,
        profile_count,
        target_count,
        total_target_residues,
        tuple(profile_records),
        forward_call_bits,
        journal_total.get("allocation_bytes"),
        value.get("native_totals"),
        batch_identity,
    )
    if rebuilt != value:
        raise ValueError("generation telemetry aggregate reconciliation changed")
    return rebuilt


def bind_generation_statistics_identity(
    value: dict[str, Any], batch_identity: tuple[int, int, int]
) -> dict[str, Any]:
    """Bind native generation evidence to its validated sealed journal."""
    canonical = validate_generation_statistics(value)
    identity = _batch_identity(
        batch_identity, "generation batch identity", allow_unbound=False
    )
    existing = _batch_identity_tuple(
        canonical["batch_identity"], "generation batch identity"
    )
    if existing is not None and existing != identity:
        raise ValueError("generation batch identity changed")
    canonical["batch_identity"] = _batch_identity_record(identity)
    return validate_generation_statistics(canonical)


def defensive_generation_statistics(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a deep copy so callers cannot mutate sealed batch evidence."""
    return None if value is None else copy.deepcopy(value)


def build_continuation_statistics(
    schema_version: int,
    path: str,
    wall_ns: int,
    target_count: int,
    postfilter_record_count: int,
    route_counts: tuple[int, ...],
    journal_counts: tuple[int, ...],
    compact_counts: tuple[int, ...],
    source_counts: tuple[int, ...],
    decision_counts: tuple[int, ...],
    identity: tuple[int, ...],
    batch_identity: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Validate one opt-in CPU-consumption record and name every route."""
    if (
        type(schema_version) is not int
        or schema_version != GENERATION_TELEMETRY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported continuation telemetry schema")
    bound_identity = _batch_identity(
        batch_identity, "continuation batch identity", allow_unbound=True
    )
    if path not in {"postfilter", "forward", "journal"}:
        raise ValueError("invalid continuation telemetry path")
    wall_ns = _uint64(wall_ns, "wall_ns")
    target_count = _uint64(target_count, "target_count")
    postfilter_record_count = _uint64(
        postfilter_record_count, "postfilter_record_count"
    )
    if postfilter_record_count > target_count:
        raise ValueError("post-filter record count exceeds targets")
    if type(route_counts) is not tuple or len(route_counts) != 7:
        raise ValueError("invalid continuation route transport")
    routes = tuple(
        _uint64(value, f"route_counts[{index}]")
        for index, value in enumerate(route_counts)
    )
    if routes[0] != target_count - postfilter_record_count:
        raise ValueError("F1 reject attribution changed")
    if _checked_sum(list(routes), "continuation routes") != target_count:
        raise ValueError("continuation routes do not partition targets")
    if path == "postfilter" and any(routes[index] for index in (4, 5, 6)):
        raise ValueError("post-filter path contains downstream routes")
    if path == "forward" and any(routes[index] for index in (5, 6)):
        raise ValueError("Forward path contains journal-only routes")

    if type(journal_counts) is not tuple or len(journal_counts) != 4:
        raise ValueError("invalid continuation journal transport")
    journal = tuple(
        _uint64(value, f"journal_counts[{index}]")
        for index, value in enumerate(journal_counts)
    )
    if journal[0] > postfilter_record_count:
        raise ValueError("journal matches exceed post-filter records")
    if journal[1] + journal[2] + journal[3] != journal[0]:
        raise ValueError("journal routes do not partition matches")
    if path != "journal" and any(journal):
        raise ValueError("non-journal path contains journal routes")

    if type(compact_counts) is not tuple or len(compact_counts) != 8:
        raise ValueError("invalid continuation compact transport")
    compact = tuple(
        _uint64(value, f"compact_counts[{index}]")
        for index, value in enumerate(compact_counts)
    )
    if compact[0] != compact[1] + compact[2] + compact[3]:
        raise ValueError("compact outcomes do not partition attempts")
    if compact[1] != routes[6]:
        raise ValueError("compact accepted route attribution changed")
    if compact[0] == 0 and any(compact[4:]):
        raise ValueError("empty compact census has a first-attempt record")

    if type(source_counts) is not tuple or len(source_counts) != 6:
        raise ValueError("invalid continuation source transport")
    source = tuple(
        _uint64(value, f"source_counts[{index}]")
        for index, value in enumerate(source_counts)
    )
    if routes[1] != source[0] + compact[3]:
        raise ValueError("CPU pipeline source attribution changed")
    if routes[2] != source[1]:
        raise ValueError("definite-reject source attribution changed")
    if routes[3] != source[2]:
        raise ValueError("filter-continuation source attribution changed")
    if routes[4] != source[3] + compact[2]:
        raise ValueError("Forward-continuation source attribution changed")
    if routes[5] != source[5]:
        raise ValueError("simple-continuation source attribution changed")
    if source[4] != source[5] + compact[0]:
        raise ValueError("journal eligible rows do not partition decisions")

    if type(decision_counts) is not tuple or len(decision_counts) != 12:
        raise ValueError("invalid continuation decision transport")
    decisions = tuple(
        _uint64(value, f"decision_counts[{index}]")
        for index, value in enumerate(decision_counts)
    )
    if any(value > 1 for value in decisions[:8]):
        raise ValueError("path decision facts must be exact booleans")
    if path == "postfilter":
        if not any(decisions[:5]) or any(decisions[5:8]):
            raise ValueError("postfilter path decision facts changed")
    elif path == "forward":
        if any(decisions[:5]) or not any(decisions[5:8]):
            raise ValueError("Forward path decision facts changed")
    elif any(decisions[:8]):
        raise ValueError("journal path contains bypass facts")
    for value in decisions[8:]:
        if value > source[5]:
            raise ValueError("compact bypass fact exceeds bypass rows")
    if source[5] and sum(decisions[8:]) < source[5]:
        raise ValueError("compact bypass rows lack source facts")

    if type(identity) is not tuple or len(identity) != 3:
        raise ValueError("invalid continuation identity transport")
    identity_values = tuple(
        _uint64(value, f"identity[{index}]")
        for index, value in enumerate(identity)
    )
    if path == "journal":
        if identity_values[1] > identity_values[2]:
            raise ValueError("journal row bounds are not monotone")
        if journal[0] != identity_values[2] - identity_values[1]:
            raise ValueError("journal row bounds changed")
        if source[4] != journal[2] + journal[3]:
            raise ValueError("journal eligible route attribution changed")
        if source[3] < journal[1]:
            raise ValueError(
                "journal CPU routes exceed Forward continuation sources"
            )
        if compact[0] > journal[3]:
            raise ValueError("compact attempts exceed SIMPLE journal rows")
    elif any(identity_values[1:]):
        raise ValueError("non-journal path contains journal row bounds")
    if compact[0]:
        if (
            not identity_values[1] <= compact[4] < identity_values[2]
            or compact[5] != identity_values[0]
            or compact[6] >= target_count
            or compact[7] == 0
        ):
            raise ValueError("compact first-attempt identity changed")

    return {
        "schema_version": GENERATION_TELEMETRY_SCHEMA_VERSION,
        "scope": "continuation",
        "batch_identity": _batch_identity_record(bound_identity),
        "path": path,
        "wall_ns": wall_ns,
        "target_count": target_count,
        "postfilter_record_count": postfilter_record_count,
        "routes": {
            "f1_reject_count": routes[0],
            "cpu_pipeline_count": routes[1],
            "definite_reject_count": routes[2],
            "filter_continuation_count": routes[3],
            "forward_continuation_count": routes[4],
            "simple_continuation_count": routes[5],
            "compact_accepted_count": routes[6],
        },
        "journal": {
            "match_count": journal[0],
            "cpu_required_count": journal[1],
            "no_region_count": journal[2],
            "simple_count": journal[3],
        },
        "compact": {
            "attempt_count": compact[0],
            "accepted_count": compact[1],
            "invalid_retry_count": compact[2],
            "threshold_retry_count": compact[3],
            "first_attempt": None if compact[0] == 0 else compact[4:],
        },
        "source_routes": {
            "postfilter_cpu_count": source[0],
            "definite_reject_count": source[1],
            "filter_count": source[2],
            "forward_count": source[3],
            "journal_eligible_count": source[4],
            "simple_bypass_count": source[5],
        },
        "decision_facts": {
            "forward": {
                "row_external_unavailable": decisions[0],
                "seam_unavailable": decisions[1],
                "f2_changed": decisions[2],
                "f3_changed": decisions[3],
                "bias_filter_changed": decisions[4],
            },
            "journal": {
                "storage_unavailable": decisions[5],
                "simple_regions_seam_unavailable": decisions[6],
                "tail_options_changed": decisions[7],
            },
            "compact": {
                "route_not_device": decisions[8],
                "empty": decisions[9],
                "tail_options_changed": decisions[10],
                "rebase_unavailable": decisions[11],
            },
        },
        "identity": {
            "profile_index": identity_values[0],
            "journal_row_start": identity_values[1],
            "journal_row_stop": identity_values[2],
        },
    }


def validate_continuation_statistics(value: Any) -> dict[str, Any]:
    """Fail closed on stored continuation evidence and rebuild it exactly."""
    required = {
        "schema_version",
        "scope",
        "batch_identity",
        "path",
        "wall_ns",
        "target_count",
        "postfilter_record_count",
        "routes",
        "journal",
        "compact",
        "source_routes",
        "decision_facts",
        "identity",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("continuation statistics fields changed")
    if value.get("scope") != "continuation":
        raise ValueError("continuation statistics scope changed")
    batch_identity = _batch_identity_tuple(
        value.get("batch_identity"), "continuation batch identity"
    )

    routes = value.get("routes")
    route_names = (
        "f1_reject_count",
        "cpu_pipeline_count",
        "definite_reject_count",
        "filter_continuation_count",
        "forward_continuation_count",
        "simple_continuation_count",
        "compact_accepted_count",
    )
    journal = value.get("journal")
    journal_names = (
        "match_count",
        "cpu_required_count",
        "no_region_count",
        "simple_count",
    )
    compact = value.get("compact")
    compact_names = (
        "attempt_count",
        "accepted_count",
        "invalid_retry_count",
        "threshold_retry_count",
        "first_attempt",
    )
    source = value.get("source_routes")
    source_names = (
        "postfilter_cpu_count",
        "definite_reject_count",
        "filter_count",
        "forward_count",
        "journal_eligible_count",
        "simple_bypass_count",
    )
    if type(routes) is not dict or set(routes) != set(route_names):
        raise ValueError("continuation route fields changed")
    if type(journal) is not dict or set(journal) != set(journal_names):
        raise ValueError("continuation journal fields changed")
    if type(compact) is not dict or set(compact) != set(compact_names):
        raise ValueError("continuation compact fields changed")
    if type(source) is not dict or set(source) != set(source_names):
        raise ValueError("continuation source-route fields changed")

    first_attempt = compact["first_attempt"]
    if first_attempt is None:
        first_attempt_values = (0, 0, 0, 0)
    elif type(first_attempt) is tuple and len(first_attempt) == 4:
        first_attempt_values = first_attempt
    else:
        raise ValueError("continuation compact first attempt changed")

    decisions = value.get("decision_facts")
    decision_names = {
        "forward": (
            "row_external_unavailable",
            "seam_unavailable",
            "f2_changed",
            "f3_changed",
            "bias_filter_changed",
        ),
        "journal": (
            "storage_unavailable",
            "simple_regions_seam_unavailable",
            "tail_options_changed",
        ),
        "compact": (
            "route_not_device",
            "empty",
            "tail_options_changed",
            "rebase_unavailable",
        ),
    }
    if type(decisions) is not dict or set(decisions) != set(decision_names):
        raise ValueError("continuation decision fields changed")
    decision_values: list[int] = []
    for stage, names in decision_names.items():
        stage_values = decisions.get(stage)
        if type(stage_values) is not dict or set(stage_values) != set(names):
            raise ValueError(f"continuation {stage} decision fields changed")
        decision_values.extend(stage_values[name] for name in names)

    identity = value.get("identity")
    identity_names = (
        "profile_index",
        "journal_row_start",
        "journal_row_stop",
    )
    if type(identity) is not dict or set(identity) != set(identity_names):
        raise ValueError("continuation identity fields changed")

    rebuilt = build_continuation_statistics(
        value.get("schema_version"),
        value.get("path"),
        value.get("wall_ns"),
        value.get("target_count"),
        value.get("postfilter_record_count"),
        tuple(routes[name] for name in route_names),
        tuple(journal[name] for name in journal_names),
        (
            *(compact[name] for name in compact_names[:-1]),
            *first_attempt_values,
        ),
        tuple(source[name] for name in source_names),
        tuple(decision_values),
        tuple(identity[name] for name in identity_names),
        batch_identity,
    )
    if rebuilt != value:
        raise ValueError("continuation telemetry reconciliation changed")
    return rebuilt


def defensive_continuation_statistics(
    value: dict[str, Any],
) -> dict[str, Any]:
    """Validate and return a private copy of continuation evidence."""
    return copy.deepcopy(validate_continuation_statistics(value))
