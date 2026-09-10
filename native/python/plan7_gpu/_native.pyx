# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stddef cimport size_t
from libc.limits cimport INT_MAX
from libc.stdint cimport int16_t, int32_t, uintptr_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.math cimport isfinite, isnan
from libc.stdlib cimport calloc, free
from libc.string cimport memcmp, memcpy, memset
from cpython.array cimport array as carray, clone
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
)
from cpython.pyport cimport PY_SSIZE_T_MAX
from libcpp.vector cimport vector
from libeasel cimport eslCONST_LOG2
from astra_pyhmmer.plan7 cimport OptimizedProfile

import array as _array
import os as _os
import astra_pyhmmer as _pyhmmer
import time as _time

from . import _abi as _abi_module
from . import _telemetry as _telemetry_module
from ._fingerprint import (
    optimized_profile_fingerprint as _profile_fingerprint,
    sequence_content_fingerprint as _sequence_content_fingerprint,
)
from ._private_tuning import (
    auto_forward_cpu_max_cells as _auto_forward_cpu_max_cells,
)


PYHMMER_PRIVATE_ABI = "0.12.0"
PYHMMER_PRIVATE_ABI_SHA256 = PYHMMER_ABI_SHA256
if _pyhmmer.__version__ != PYHMMER_PRIVATE_ABI:
    raise ImportError(
        f"plan7_gpu._native requires PyHMMER {PYHMMER_PRIVATE_ABI}, "
        f"found {_pyhmmer.__version__}"
    )
_abi_module.validate_private_abi_platform()
_runtime_abi_sha256 = _abi_module.pyhmmer_abi_fingerprint()
if _runtime_abi_sha256 != PYHMMER_PRIVATE_ABI_SHA256:
    raise ImportError(
        "plan7_gpu._native was built against a different PyHMMER private ABI "
        f"({PYHMMER_PRIVATE_ABI_SHA256} != {_runtime_abi_sha256})"
    )


cdef carray _UINT32_ARRAY_TEMPLATE = _array.array("I")
cdef carray _UINT16_ARRAY_TEMPLATE = _array.array("H")
cdef carray _UINT64_ARRAY_TEMPLATE = _array.array("Q")
cdef carray _FLOAT_ARRAY_TEMPLATE = _array.array("f")
cdef uint64_t _sealed_journal_build_count = 0
cdef uint64_t _sealed_journal_payload_bytes = 0
cdef uint64_t _sealed_journal_duplicate_python_bytes = 0
cdef uint64_t _sealed_journal_validation_ns = 0
cdef uint64_t _sealed_journal_emit_ns = 0
cdef uint64_t _direct_v3_staging_build_count = 0
cdef uint64_t _direct_v3_eliminated_v2_bytes = 0
cdef uint64_t _direct_v3_staging_payload_bytes = 0
cdef uint64_t _direct_v3_staging_build_ns = 0
cdef uint64_t _direct_v3_source_validation_ns = 0
cdef uint64_t _resident_f2_compaction_run_count = 0
cdef uint64_t _resident_f2_source_count = 0
cdef uint64_t _resident_f2_selected_count = 0
cdef uint64_t _resident_f2_compiled_profile_count = 0
cdef uint64_t _resident_f2_unsupported_profile_count = 0
cdef uint64_t _resident_f2_selected_d2h_bytes = 0
cdef double _resident_f2_compile_milliseconds = 0.0
cdef double _resident_f2_upload_milliseconds = 0.0
cdef double _resident_f2_kernel_milliseconds = 0.0
cdef double _resident_f2_scan_milliseconds = 0.0
cdef double _resident_f2_download_milliseconds = 0.0
cdef double _resident_f2_total_milliseconds = 0.0
cdef uint64_t _resident_forward_f2_call_count = 0
cdef uint64_t _resident_forward_f2_candidate_count = 0
cdef uint64_t _resident_forward_f2_eliminated_h2d_bytes = 0
cdef double _resident_forward_f2_gather_milliseconds = 0.0
cdef uint64_t _resident_forward_call_count = 0
cdef uint64_t _resident_forward_requested_bytes = 0
cdef uint64_t _resident_forward_allocated_bytes = 0
cdef uint64_t _resident_forward_materialized_bytes = 0
cdef uint64_t _resident_forward_allocation_fallback_count = 0
cdef double _resident_forward_allocation_milliseconds = 0.0
cdef double _resident_forward_materialization_milliseconds = 0.0
cdef uint64_t _resident_backward_call_count = 0
cdef uint64_t _resident_backward_forward_h2d_bytes = 0
cdef uint64_t _resident_backward_eliminated_forward_h2d_bytes = 0
cdef double _resident_backward_forward_upload_milliseconds = 0.0
cdef uint64_t _resident_backward_region_requested_bytes = 0
cdef uint64_t _resident_backward_region_allocated_bytes = 0
cdef uint64_t _resident_backward_region_materialized_bytes = 0
cdef uint64_t _resident_backward_region_allocation_fallback_count = 0
cdef double _resident_backward_region_allocation_milliseconds = 0.0
cdef double _resident_backward_region_materialization_milliseconds = 0.0
cdef uint64_t _resident_rescore_call_count = 0
cdef uint64_t _resident_rescore_upstream_h2d_bytes = 0
cdef uint64_t _resident_rescore_eliminated_upstream_h2d_bytes = 0
cdef uint64_t _resident_rescore_selection_h2d_bytes = 0
cdef double _resident_rescore_upstream_upload_milliseconds = 0.0
cdef double _resident_rescore_prepare_milliseconds = 0.0
cdef uint64_t _f3_compiled_profile_count = 0
cdef uint64_t _f3_unsupported_profile_count = 0
cdef uint64_t _f3_host_audit_count = 0
cdef uint64_t _f3_host_decision_avoided_count = 0
cdef uint64_t _f3_device_decision_count = 0
cdef uint64_t _f3_device_reject_count = 0
cdef uint64_t _f3_device_pass_count = 0
cdef uint64_t _f3_host_fallback_count = 0
cdef uint64_t _f3_decision_mismatch_count = 0
cdef uint64_t _f3_device_compaction_run_count = 0
cdef uint64_t _f3_device_compaction_candidate_count = 0
cdef uint64_t _f3_device_compacted_survivor_count = 0
cdef uint64_t _f3_survivor_upload_avoided_bytes = 0
cdef uint64_t _subwarp_call_count = 0
cdef uint64_t _subwarp_auto_call_count = 0
cdef uint64_t _subwarp_width1_call_count = 0
cdef uint64_t _subwarp_width2_call_count = 0
cdef uint64_t _subwarp_width4_call_count = 0
cdef uint64_t _subwarp_width8_call_count = 0
cdef uint64_t _subwarp_no_kernel_count = 0
cdef uint64_t _subwarp_forced_count = 0
cdef uint64_t _subwarp_sparse_width1_count = 0
cdef uint64_t _subwarp_short_width4_count = 0
cdef uint64_t _subwarp_short_width2_count = 0
cdef uint64_t _subwarp_long_width4_count = 0
cdef uint64_t _subwarp_long_width2_count = 0
cdef uint64_t _subwarp_long_saturated_width1_count = 0
cdef uint64_t _subwarp_divergent_width1_count = 0
cdef uint64_t _subwarp_kernel_launch_count = 0
cdef uint64_t _subwarp_scheduled_warp_count = 0
cdef uint64_t _subwarp_candidate_count = 0
cdef uint64_t _subwarp_active_lane_slots = 0
cdef uint64_t _subwarp_issued_lane_slots = 0
SEALED_STAGE_TIMING_SCHEMA_VERSION = 1
GENERATION_TELEMETRY_SCHEMA_VERSION = 2
DIRECT_V3_STAGING_SCHEMA_VERSION = 3

# Host F2 decisions are made in this Cython translation unit, so their exact
# version-1 facts intentionally live beside that source predicate.
F2_REASON_POSTFILTER_NOT_PASS_OR_HOST_ENVIRONMENT_UNATTESTED = 0x01
F2_REASON_INPUT_INVALID = 0x02
F2_REASON_MSV_THRESHOLD_EXCEEDED = 0x04
F2_REASON_VITERBI_THRESHOLD_EXCEEDED = 0x08
F2_REASON_PASS = 0x10

_DEVICE_CAPACITY_NAMES = (
    "input_residues",
    "input_offsets",
    "input_null_scores",
    "length_tjb",
    "results",
    "compact_scores",
    "profiles",
    "f1_profiles",
    "candidate_words",
    "bias_profiles",
    "bias_candidates",
    "bias_ssv_inputs",
    "bias_results",
    "bias_logp",
    "bias_log1mp",
    "postfilter_states",
    "postfilter_bias_inputs",
    "postfilter_bias_results",
    "postfilter_viterbi_results",
    "postfilter_length_transitions",
    "postfilter_msv_offsets",
    "postfilter_viterbi_offsets",
    "postfilter_dp",
    "postfilter_results",
    "forward_candidate_profiles",
    "forward_candidate_sequences",
    "forward_filter_scores",
    "forward_f3_thresholds",
    "forward_length_transitions",
    "forward_dp_offsets",
    "forward_x_offsets",
    "forward_dp",
    "forward_xmx",
    "forward_results",
    "forward_survivor_candidates",
    "forward_survivor_offsets",
    "forward_gathered",
    "candidate_word_counts",
    "candidate_word_offsets",
    "candidate_profile_offsets",
    "candidate_scan_workspace",
    "f1_profile_packed_scores",
    "f1_profile_packed_quartets",
    "f1_scalar_profile_indices",
    "length_class_indices",
    "f1_compact_tjb",
    "f1_raw_xe",
)


cdef extern from * nogil:
    """
    extern "C" {
    #include <esl_gumbel.h>
    }
    static inline unsigned plan7_popcount_u32(uint32_t value) {
      return (unsigned) __builtin_popcount(value);
    }
    static inline unsigned plan7_ctz_u32(uint32_t value) {
      return (unsigned) __builtin_ctz(value);
    }
    """
    unsigned plan7_popcount_u32(uint32_t value)
    unsigned plan7_ctz_u32(uint32_t value)
    double esl_gumbel_surv(double, double, double)


cdef extern from "bias_cuda.h" nogil:
    cdef enum plan7_bias_action:
        PLAN7_BIAS_CPU_REQUIRED
        PLAN7_BIAS_DEFINITE_REJECT
        PLAN7_BIAS_DEFINITE_PASS

    cdef enum plan7_bias_cutoff_mode:
        PLAN7_BIAS_CUTOFF_INVALID
        PLAN7_BIAS_CUTOFF_SCORE
        PLAN7_BIAS_CUTOFF_ALWAYS_REJECT
        PLAN7_BIAS_CUTOFF_ALWAYS_PASS

    cdef enum plan7_bias_cuda_target:
        PLAN7_BIAS_CUDA_UNATTESTED
        PLAN7_BIAS_CUDA_SM75_RTX2080_TI
        PLAN7_BIAS_CUDA_SM90_H200

    enum:
        PLAN7_BIAS_LIBM_BUILD_ID_SIZE

    ctypedef struct plan7_bias_cuda_identity:
        int32_t device_ordinal
        int32_t runtime_version
        int32_t driver_version
        int32_t cudart_version
        int32_t nvcc_major
        int32_t nvcc_minor
        int32_t nvcc_build
        int32_t compute_major
        int32_t compute_minor
        int32_t multiprocessor_count
        int32_t pci_domain_id
        int32_t pci_bus_id
        int32_t pci_device_id
        uint64_t total_global_memory
        uint8_t uuid[16]
        char pci_bus_address[32]
        char name[256]

    ctypedef struct plan7_bias_host_identity:
        uint32_t vendor_ebx
        uint32_t vendor_edx
        uint32_t vendor_ecx
        uint32_t family
        uint32_t model
        uint32_t stepping
        uint32_t leaf1_ecx
        uint32_t leaf1_edx
        uint32_t leaf7_ebx
        uint32_t xcr0_low
        uint32_t xcr0_high

    ctypedef struct plan7_bias_profile:
        float t10
        float t11
        float scale
        float cutoff_bit_score
        int32_t cutoff_mode
        uint32_t reserved
        float pi0
        float pi1
        float t02
        float t12
        float eo[29][2]

    ctypedef struct plan7_bias_candidate:
        uint32_t profile_index
        uint32_t sequence_index

    ctypedef struct plan7_bias_result:
        uint32_t sequence_index
        float filtersc
        int16_t ssv_numerator
        uint8_t ssv_status
        uint8_t action

    int plan7_bias_pack_amino_profile(
        const float *background,
        const float *composition,
        int model_length,
        float scale,
        int cutoff_mode,
        float cutoff_bit_score,
        plan7_bias_profile *profile,
        char *error,
        size_t error_size,
    )

    int plan7_bias_filter_score_host(
        const plan7_bias_profile *profile,
        const uint8_t *residues,
        uint64_t length,
        float *filtersc,
    )

    int plan7_bias_rebias_decision(
        uint8_t ssv_status,
        int16_t ssv_numerator,
        float scale,
        float filtersc,
        int cutoff_mode,
        float cutoff_bit_score,
        float *bit_score,
    )

    int plan7_bias_host_environment_attested()
    int plan7_bias_current_cuda_identity(
        plan7_bias_cuda_identity *identity,
        char *reason,
        size_t reason_size,
    )
    int plan7_bias_current_host_identity(
        plan7_bias_host_identity *identity,
        char *reason,
        size_t reason_size,
    )
    int plan7_bias_cuda_identity_target(
        const plan7_bias_cuda_identity *identity,
        char *reason,
        size_t reason_size,
    )
    int plan7_bias_host_identity_attested(
        const plan7_bias_host_identity *identity,
        int cuda_target,
        char *reason,
        size_t reason_size,
    )
    int plan7_bias_current_libm_build_id(uint8_t *build_id)
    int plan7_bias_libm_build_id_attested(
        const uint8_t *build_id,
        size_t build_id_size,
        int cuda_target,
    )
    int plan7_bias_environment_attested(char *reason, size_t reason_size)


cdef extern from "ssv_cuda.h" nogil:
    cdef enum plan7_ssv_status:
        PLAN7_SSV_OK
        PLAN7_SSV_ERANGE
        PLAN7_SSV_ENORESULT
        PLAN7_SSV_EMPTY

    cdef enum plan7_f1_action:
        PLAN7_F1_CPU_REQUIRED
        PLAN7_F1_DEFINITE_REJECT

    cdef enum plan7_f1_cutoff_mode:
        PLAN7_F1_CUTOFF_INVALID
        PLAN7_F1_CUTOFF_SCORE
        PLAN7_F1_CUTOFF_ALWAYS_REJECT
        PLAN7_F1_CUTOFF_ALWAYS_CPU

    cdef enum plan7_gpu_execution_policy:
        PLAN7_GPU_EXECUTION_POLICY_AUTO
        PLAN7_GPU_EXECUTION_POLICY_SIMPLE
        PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT

    cdef enum plan7_gpu_execution_policy_abi:
        PLAN7_GPU_EXECUTION_POLICY_VERSION
        PLAN7_GPU_EXECUTION_POLICY_FORWARD_CANDIDATES_PER_WARP

    ctypedef struct plan7_ssv_result:
        uint8_t xE
        uint8_t status
        uint8_t tjb
        uint8_t reserved
        int16_t numerator

    ctypedef struct plan7_ssv_profile:
        uint64_t score_offset
        uint64_t score_count
        int32_t score_stride
        int32_t model_length
        uint8_t tbm
        uint8_t tec
        uint8_t base
        uint8_t bias
        float scale

    ctypedef struct plan7_ssv_sequence_batch:
        pass

    ctypedef struct plan7_ssv_sequence_batch_view:
        uint64_t generation_id
        int device_ordinal
        int alphabet_size
        int host_float_environment_valid
        size_t sequence_count
        const uint64_t *host_lengths
        const uint8_t *device_residues
        const uint64_t *device_offsets
        uint64_t input_device_bytes

    ctypedef struct plan7_ssv_workspace_statistics:
        uint64_t f1_device_compaction_run_count
        uint64_t f1_host_expansion_run_count
        uint64_t f1_candidate_upload_count
        uint64_t f1_candidate_upload_avoided_count
        uint64_t f1_profile_packed_run_count
        uint64_t f1_profile_packed_quartet_count
        uint64_t f1_profile_packed_profile_count
        uint64_t f1_profile_scalar_profile_count
        uint64_t f1_profile_packed_score_bytes
        uint64_t f1_identity_padding_run_count
        uint64_t f1_identity_padding_quartet_count
        uint64_t f1_identity_padding_profile_count
        uint64_t f1_length_class_run_count
        uint64_t f1_length_class_value_count
        uint64_t f1_length_compact_h2d_bytes
        uint64_t f1_length_dense_h2d_bytes_avoided
        uint64_t f1_length_dense_materialized_bytes
        uint64_t postfilter_device_bytes
        uint64_t postfilter_dp_capacity_bytes
        uint64_t postfilter_growth_count
        uint64_t postfilter_run_count
        uint64_t full_msv_compaction_run_count
        uint64_t full_msv_compaction_chunk_count
        uint64_t full_msv_compaction_source_count
        uint64_t full_msv_compaction_selected_count
        uint64_t full_msv_legacy_run_count
        uint64_t full_msv_launch_candidate_count
        uint64_t full_msv_launch_candidate_avoided_count
        uint64_t full_msv_index_d2h_bytes
        uint64_t full_msv_packed_run_count
        uint64_t full_msv_packed_group_count
        uint64_t full_msv_packed_candidate_count
        uint64_t full_msv_scalar_candidate_count
        uint64_t vit_length_cache_run_count
        uint64_t vit_length_cache_entry_count
        uint64_t vit_length_cache_candidate_count
        uint64_t vit_length_direct_candidate_count
        uint64_t vit_length_cache_build_ns
        uint64_t vit_length_candidate_plan_ns
        uint64_t forward_device_bytes
        uint64_t forward_dp_capacity_bytes
        uint64_t forward_xmx_capacity_bytes
        uint64_t forward_gather_capacity_bytes
        uint64_t forward_growth_count
        uint64_t forward_event_create_count
        uint64_t forward_run_count
        uint64_t f1_raw_xe_run_count
        uint64_t f1_raw_xe_logical_pair_count
        uint64_t f1_raw_xe_sidecar_bytes_written
        uint64_t f1_raw_xe_candidate_gather_count
        uint64_t f1_candidate_ssv_replay_count
        uint64_t f1_candidate_ssv_replay_avoided_count
        uint64_t f1_raw_xe_fallback_run_count

    ctypedef struct plan7_gpu_execution_policy_statistics:
        uint32_t version
        uint32_t mode
        uint64_t target_count
        uint64_t length_class_count
        uint64_t f1_run_count
        uint64_t forward_candidates_per_warp

    ctypedef struct plan7_profile_footprint:
        uint64_t profile_count
        uint64_t ssv_device_bytes
        uint64_t viterbi_device_bytes
        uint64_t viterbi_exact_rbv_upper_bytes
        uint64_t forward_device_bytes
        uint64_t bias_device_bytes
        uint64_t minimum_device_bytes
        uint64_t maximum_device_bytes

    ctypedef struct plan7_allocation_simulation:
        uint64_t peak_additional_bytes
        uint64_t final_additional_bytes
        uint64_t final_free_bytes
        uint64_t growth_count
        uint64_t first_unfit_index
        int32_t fits
        int32_t reserved

    cdef enum plan7_ssv_device_capacity:
        PLAN7_SSV_DEVICE_CAPACITY_COUNT

    ctypedef struct plan7_ssv_memory_snapshot:
        int32_t device_ordinal
        int32_t reserved
        uint64_t cuda_free_bytes
        uint64_t cuda_total_bytes
        uint64_t persistent_device_bytes
        uint64_t device_capacity_bytes[47]

    ctypedef struct plan7_ssv_f1_candidate_view:
        size_t profile_count
        size_t candidate_count
        const size_t *candidate_offsets
        const plan7_bias_candidate *candidates

    ctypedef struct plan7_f0_profile_statistics:
        uint64_t logical_pair_count
        uint64_t exact_candidate_count
        uint64_t coarse_candidate_count
        uint64_t certified_reject_count
        uint64_t false_reject_count
        uint64_t logical_cell_count
        uint64_t survivor_exact_cell_count

    ctypedef struct plan7_f0_evaluation_statistics:
        uint64_t profile_count
        uint64_t sequence_count
        uint64_t class_count
        uint64_t logical_pair_count
        uint64_t exact_candidate_count
        uint64_t coarse_candidate_count
        uint64_t certified_reject_count
        uint64_t false_reject_count
        uint64_t logical_cell_count
        uint64_t survivor_exact_cell_count
        uint64_t coarse_table_bytes
        uint64_t temporary_device_bytes
        double exact_generation_milliseconds
        double coarse_table_build_milliseconds
        double coarse_upload_milliseconds
        double coarse_kernel_milliseconds
        double analysis_milliseconds

    ctypedef struct plan7_seed_profile_statistics:
        uint64_t logical_pair_count
        uint64_t exact_candidate_count
        uint64_t seed_candidate_count
        uint64_t certified_reject_count
        uint64_t false_reject_count
        uint64_t unsupported_pair_count
        uint64_t logical_cell_count
        uint64_t survivor_exact_cell_count

    ctypedef struct plan7_seed_evaluation_statistics:
        uint64_t profile_count
        uint64_t sequence_count
        uint64_t maximum_word_length
        uint64_t logical_pair_count
        uint64_t exact_candidate_count
        uint64_t seed_candidate_count
        uint64_t certified_reject_count
        uint64_t false_reject_count
        uint64_t unsupported_pair_count
        uint64_t logical_cell_count
        uint64_t survivor_exact_cell_count
        uint64_t temporary_device_bytes
        double exact_generation_milliseconds
        double seed_kernel_milliseconds
        double analysis_milliseconds

    int plan7_cuda_device_count(char *error, size_t error_size)
    int plan7_cuda_memory_info(
        int *device_ordinal,
        uint64_t *free_bytes,
        uint64_t *total_bytes,
        char *error,
        size_t error_size,
    )
    int plan7_validate_device_ordinal(
        int owner_device,
        int current_device,
        char *error,
        size_t error_size,
    )
    int plan7_profile_footprint_compute(
        const uint32_t *model_lengths,
        size_t profile_count,
        plan7_profile_footprint *footprint,
        char *error,
        size_t error_size,
    )
    int plan7_profile_slice_cell_count(
        uint64_t profile_count,
        uint64_t target_count,
        uint64_t cell_limit,
        uint64_t *cell_count,
        char *error,
        size_t error_size,
    )
    int plan7_simulate_allocate_before_free(
        const uint64_t *current_capacities,
        const uint64_t *required_capacities,
        size_t capacity_count,
        uint64_t free_bytes,
        uint64_t *final_capacities,
        plan7_allocation_simulation *simulation,
        char *error,
        size_t error_size,
    )
    int plan7_tjb_for_length(float scale, uint64_t length)
    int plan7_ssv_f1_decision(
        uint8_t status,
        int16_t numerator,
        uint64_t length,
        float scale,
        float m_mu,
        float m_lambda,
        double f1,
        double *ret_p,
    )

    int plan7_ssv_f1_cutoff(
        float m_mu,
        float m_lambda,
        double f1,
        float *ret_bit_score,
    )

    int plan7_ssv_f1_cutoff_decision(
        uint8_t status,
        int16_t numerator,
        uint64_t length,
        float scale,
        int cutoff_mode,
        float cutoff_bit_score,
    )

    int plan7_ssv_sequence_batch_create(
        const uint8_t *residues,
        size_t residue_count,
        const uint64_t *offsets,
        size_t offset_count,
        int alphabet_size,
        plan7_ssv_sequence_batch **batch,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_destroy(
        plan7_ssv_sequence_batch **batch,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_set_execution_policy(
        plan7_ssv_sequence_batch *batch,
        int policy,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_execution_policy_statistics(
        const plan7_ssv_sequence_batch *batch,
        plan7_gpu_execution_policy_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_view(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_sequence_batch_view *view,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_workspace_statistics(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_workspace_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_memory_snapshot(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_memory_snapshot *snapshot,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_filter(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *striped_scores,
        size_t striped_score_count,
        int score_stride,
        int model_length,
        int alphabet_size,
        uint8_t tbm,
        uint8_t tec,
        uint8_t base,
        uint8_t bias,
        float scale,
        plan7_ssv_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_filter_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        plan7_ssv_result *profile_major_results,
        size_t result_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_f1_candidates_many(
        const plan7_ssv_sequence_batch *batch,
        const plan7_ssv_result *profile_major_results,
        size_t result_count,
        const float *scales,
        const float *m_mu,
        const float *m_lambda,
        size_t profile_count,
        double f1,
        const size_t *candidate_offsets,
        uint32_t *candidate_indices,
        size_t candidate_index_count,
        size_t *candidate_counts,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_f1_mask_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        const float *m_mu,
        const float *m_lambda,
        double f1,
        uint32_t *profile_major_candidate_words,
        size_t candidate_word_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_f1_compact_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        const float *m_mu,
        const float *m_lambda,
        double f1,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_f1_candidate_view(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_f1_candidate_view *view,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_evaluate_f0_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        const float *m_mu,
        const float *m_lambda,
        double f1,
        const uint8_t *residue_classes,
        size_t residue_class_count,
        size_t class_count,
        plan7_f0_profile_statistics *profile_statistics,
        size_t profile_statistics_count,
        plan7_f0_evaluation_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_evaluate_seed_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        const float *m_mu,
        const float *m_lambda,
        double f1,
        size_t maximum_word_length,
        size_t indexed_alphabet_size,
        plan7_seed_profile_statistics *profile_statistics,
        size_t profile_statistics_count,
        plan7_seed_evaluation_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_bias_candidates_many(
        plan7_ssv_sequence_batch *batch,
        const plan7_bias_profile *bias_profiles,
        size_t profile_count,
        const size_t *candidate_offsets,
        const uint32_t *candidate_indices,
        size_t candidate_count,
        plan7_bias_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )


cdef extern from "postfilter_cuda.h" nogil:
    cdef enum plan7_postfilter_abi:
        PLAN7_POSTFILTER_RECORD_VERSION
        PLAN7_POSTFILTER_RECORD_SIZE

    ctypedef struct plan7_postfilter_result:
        uint32_t sequence_index
        float filtersc
        int16_t msv_numerator
        uint8_t msv_status
        uint8_t action
        float vfsc

    cdef enum plan7_postfilter_f2_fact:
        PLAN7_POSTFILTER_F2_NOT_PASS_OR_UNATTESTED
        PLAN7_POSTFILTER_F2_INPUT_INVALID
        PLAN7_POSTFILTER_F2_MSV_THRESHOLD_EXCEEDED
        PLAN7_POSTFILTER_F2_VITERBI_THRESHOLD_EXCEEDED
        PLAN7_POSTFILTER_F2_PASS

    ctypedef struct plan7_postfilter_f2_statistics:
        uint64_t source_count
        uint64_t selected_count
        uint64_t compiled_profile_count
        uint64_t unsupported_profile_count
        uint64_t mask_word_count
        uint64_t selected_d2h_bytes
        uint64_t run_count
        float compile_milliseconds
        float upload_milliseconds
        float kernel_milliseconds
        float scan_milliseconds
        float download_milliseconds
        float total_milliseconds

    ctypedef struct plan7_postfilter_f2_resident_view:
        uint64_t batch_generation
        uint64_t workspace_generation
        uint64_t selected_source_hash
        int32_t device_ordinal
        uint32_t supported
        size_t profile_count
        size_t source_count
        size_t selected_count
        const uint32_t *host_selected_sources
        const plan7_bias_candidate *host_candidates
        const plan7_postfilter_result *host_results
        const plan7_bias_candidate *device_candidates
        const plan7_postfilter_result *device_results
        const uint32_t *device_selected_sources
        const void *owner
        plan7_postfilter_f2_statistics statistics

    cdef enum plan7_postfilter_reason_fact:
        PLAN7_POSTFILTER_REASON_RAW_F1_REJECT
        PLAN7_POSTFILTER_REASON_MSV_RANGE_STATE
        PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU
        PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO
        PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED
        PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE
        PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED
        PLAN7_POSTFILTER_REASON_VITERBI_ERANGE
        PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS
        PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED
        PLAN7_POSTFILTER_REASON_FINAL_REJECT
        PLAN7_POSTFILTER_REASON_FINAL_PASS
        PLAN7_POSTFILTER_REASON_OTHER_CPU_REQUIRED
        PLAN7_POSTFILTER_REASON_CONTRACT_FALLBACK
        PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED
        PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED

    ctypedef struct plan7_postfilter_reason_statistics:
        uint64_t candidate_count
        uint64_t full_msv_execution_count
        uint64_t viterbi_execution_count
        uint64_t full_msv_work_cells
        uint64_t viterbi_work_cells
        uint64_t work_cells

    ctypedef struct plan7_viterbi_database:
        pass

    ctypedef struct plan7_profile_session:
        pass

    ctypedef struct plan7_profile_selection:
        pass

    ctypedef struct plan7_forward_database:
        pass

    ctypedef struct plan7_profile_selection_view:
        uint64_t session_id
        uint64_t selection_id
        size_t profile_count
        const uint8_t *packed_scores
        size_t packed_score_count
        const plan7_ssv_profile *profiles
        const float *m_mu
        const float *m_lambda
        const float *v_mu
        const float *v_lambda
        const plan7_bias_profile *bias_templates
        const uintptr_t *identity_tokens
        uint64_t host_bytes

    ctypedef struct plan7_profile_session_statistics:
        uint64_t session_id
        uint64_t profile_count
        uint64_t worker_count
        uint64_t build_worker_count
        uint64_t selection_worker_count
        uint64_t selection_count
        uint64_t parallel_run_count
        uint64_t build_parallel_run_count
        uint64_t selection_parallel_run_count
        uint64_t host_bytes
        uint64_t ssv_score_bytes
        uint64_t bias_profile_bytes
        uint64_t viterbi_descriptor_bytes
        uint64_t viterbi_emission_bytes
        uint64_t viterbi_transition_bytes
        uint64_t viterbi_exact_rbv_bytes
        uint64_t forward_descriptor_bytes
        uint64_t forward_emission_bytes
        uint64_t forward_transition_bytes
        uint64_t chunk_local_pack
        uint64_t profile_pointer_bytes
        uint64_t identity_token_bytes
        uint64_t background_bytes

    int plan7_viterbi_database_create(
        const uintptr_t *profile_pointers,
        size_t profile_count,
        plan7_viterbi_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_viterbi_database_destroy(
        plan7_viterbi_database **database,
        char *error,
        size_t error_size,
    )

    size_t plan7_viterbi_database_profile_count(
        const plan7_viterbi_database *database,
    )

    int plan7_profile_session_create(
        const uintptr_t *profile_pointers,
        size_t profile_count,
        const float *background,
        size_t background_count,
        size_t build_worker_count,
        size_t selection_worker_count,
        int chunk_local_pack,
        plan7_profile_session **session,
        char *error,
        size_t error_size,
    )

    int plan7_profile_session_destroy(
        plan7_profile_session **session,
        char *error,
        size_t error_size,
    )

    int plan7_profile_session_get_statistics(
        const plan7_profile_session *session,
        plan7_profile_session_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_profile_session_select(
        plan7_profile_session *session,
        const size_t *profile_indices,
        size_t profile_count,
        plan7_profile_selection **selection,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_destroy(
        plan7_profile_selection **selection,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_get_view(
        const plan7_profile_selection *selection,
        plan7_profile_selection_view *view,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_snapshot_equal_for_test(
        const plan7_profile_selection *left,
        const plan7_profile_selection *right,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_stage_viterbi(
        const plan7_profile_selection *selection,
        plan7_viterbi_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_stage_forward(
        const plan7_profile_selection *selection,
        plan7_forward_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_postfilter_candidates_many(
        plan7_ssv_sequence_batch *batch,
        const plan7_bias_profile *bias_profiles,
        size_t profile_count,
        const size_t *candidate_offsets,
        const uint32_t *candidate_indices,
        size_t candidate_count,
        const uintptr_t *source_profile_pointers,
        const plan7_viterbi_database *viterbi_database,
        plan7_postfilter_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_postfilter_candidates_many_reason_facts(
        plan7_ssv_sequence_batch *batch,
        const plan7_bias_profile *bias_profiles,
        size_t profile_count,
        const size_t *candidate_offsets,
        const uint32_t *candidate_indices,
        size_t candidate_count,
        const uintptr_t *source_profile_pointers,
        const plan7_viterbi_database *viterbi_database,
        plan7_postfilter_result *results,
        size_t result_count,
        uint16_t *reason_facts,
        size_t reason_count,
        plan7_postfilter_reason_statistics *reason_statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_postfilter_candidates_many_fixed_bias(
        plan7_ssv_sequence_batch *batch,
        const plan7_bias_profile *bias_profiles,
        size_t profile_count,
        const size_t *candidate_offsets,
        const uint32_t *candidate_indices,
        size_t candidate_count,
        const uintptr_t *source_profile_pointers,
        const plan7_viterbi_database *viterbi_database,
        plan7_postfilter_result *results,
        size_t result_count,
        uint16_t *reason_facts,
        size_t reason_count,
        plan7_postfilter_reason_statistics *reason_statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_compact_postfilter_f2(
        plan7_ssv_sequence_batch *batch,
        const plan7_ssv_profile *profiles,
        const float *m_mu,
        const float *m_lambda,
        const float *v_mu,
        const float *v_lambda,
        size_t profile_count,
        double f2,
        plan7_postfilter_f2_resident_view *view,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_filter_cuda(
        const uint8_t *striped_scores,
        size_t striped_score_count,
        int score_stride,
        int model_length,
        int alphabet_size,
        const uint8_t *residues,
        size_t residue_count,
        const uint64_t *offsets,
        size_t offset_count,
        size_t sequence_count,
        uint8_t tbm,
        uint8_t tec,
        uint8_t base,
        uint8_t bias,
        float scale,
        plan7_ssv_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )


cdef extern from "forward_cuda.h" nogil:
    cdef enum plan7_forward_abi:
        PLAN7_FORWARD_RECORD_VERSION
        PLAN7_FORWARD_RECORD_SIZE
        PLAN7_FORWARD_MAX_GATHERED_BYTES

    cdef enum plan7_forward_action:
        PLAN7_FORWARD_CPU_REQUIRED
        PLAN7_FORWARD_DEFINITE_REJECT
        PLAN7_FORWARD_DEFINITE_PASS

    cdef enum plan7_forward_status:
        PLAN7_FORWARD_OK
        PLAN7_FORWARD_ERANGE
        PLAN7_FORWARD_ENORESULT
        PLAN7_FORWARD_EMPTY

    cdef enum plan7_forward_reason_fact:
        PLAN7_FORWARD_REASON_KERNEL_STATUS_NON_OK
        PLAN7_FORWARD_REASON_TARGET_EMPTY
        PLAN7_FORWARD_REASON_FWDSC_NONFINITE
        PLAN7_FORWARD_REASON_FILTERSC_NONFINITE
        PLAN7_FORWARD_REASON_TAU_NONFINITE
        PLAN7_FORWARD_REASON_LAMBDA_INVALID
        PLAN7_FORWARD_REASON_F3_REJECT
        PLAN7_FORWARD_REASON_OUTPUT_CAP
        PLAN7_FORWARD_REASON_SURVIVOR_GATHERED
        PLAN7_FORWARD_REASON_OTHER_CPU_REQUIRED

    cdef enum plan7_forward_call_reason_fact:
        PLAN7_FORWARD_CALL_REASON_CONTRACT_FALLBACK

    cdef enum plan7_forward_subwarp_policy_reason:
        PLAN7_FORWARD_SUBWARP_POLICY_NO_KERNEL
        PLAN7_FORWARD_SUBWARP_POLICY_FORCED
        PLAN7_FORWARD_SUBWARP_POLICY_SPARSE_WIDTH1
        PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH4
        PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH2
        PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH4
        PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH2
        PLAN7_FORWARD_SUBWARP_POLICY_LONG_SATURATED_WIDTH1
        PLAN7_FORWARD_SUBWARP_POLICY_DIVERGENT_WIDTH1

    ctypedef struct plan7_forward_result:
        uint32_t sequence_index
        float fwdsc
        uint8_t status
        uint8_t action
        uint16_t reserved

    ctypedef struct plan7_forward_statistics:
        uint64_t generation_f3_bits
        uint64_t candidate_count
        uint64_t survivor_count
        uint64_t work_cells
        uint64_t dp_workspace_bytes
        uint64_t xmx_workspace_bytes
        uint64_t gather_workspace_bytes
        uint64_t gathered_xmx_bytes
        uint64_t output_byte_limit
        uint64_t output_cap_fallback_count
        float kernel_milliseconds
        float classification_milliseconds
        float gather_milliseconds
        float download_milliseconds
        float total_milliseconds

    ctypedef struct plan7_forward_f3_device_statistics:
        uint64_t compiled_profile_count
        uint64_t unsupported_profile_count
        uint64_t host_audit_count
        uint64_t host_decision_avoided_count
        uint64_t device_decision_count
        uint64_t device_reject_count
        uint64_t device_pass_count
        uint64_t host_fallback_count
        uint64_t decision_mismatch_count
        uint64_t device_compaction_run_count
        uint64_t device_compaction_candidate_count
        uint64_t device_compacted_survivor_count
        uint64_t survivor_upload_avoided_bytes

    ctypedef struct plan7_forward_provenance:
        uint64_t database_generation
        uint64_t batch_generation
        uint64_t row_hash
        uint64_t special_hash
        uint64_t continuation_hash
        uint64_t pass_count
        uint64_t special_count
        uint64_t generation_f3_bits
        uint64_t integrity_tag

    ctypedef struct plan7_forward_residency_statistics:
        uint64_t requested_bytes
        uint64_t allocated_bytes
        uint64_t materialized_bytes
        uint64_t allocation_fallback_count
        float allocation_milliseconds
        float materialization_milliseconds

    ctypedef struct plan7_forward_input_residency_statistics:
        uint64_t resident_f2_call_count
        uint64_t resident_f2_candidate_count
        uint64_t eliminated_candidate_h2d_bytes
        float gather_milliseconds

    ctypedef struct plan7_forward_subwarp_statistics:
        uint32_t policy_version
        uint32_t requested_candidates_per_warp
        uint32_t candidates_per_warp
        uint32_t policy_reason
        uint32_t multiprocessor_count
        uint32_t reserved
        uint64_t l2_cache_bytes
        uint64_t policy_tile_candidate_count
        uint64_t model_length_sum
        uint64_t target_length_sum
        uint64_t average_model_length
        uint64_t average_target_length
        uint64_t maximum_model_length
        uint64_t maximum_target_length
        uint64_t maximum_candidate_work_cells
        uint64_t average_work_cells
        uint64_t short_width4_workspace_limit_bytes
        uint64_t long_packed_workspace_limit_bytes
        uint64_t policy_xmx_workspace_bytes
        uint64_t minimum_cta_count
        uint64_t width1_cta_count
        uint64_t width2_cta_count
        uint64_t width4_cta_count
        uint64_t kernel_launch_count
        uint64_t scheduled_warp_count
        uint64_t candidate_subwarp_count
        uint64_t active_lane_slots
        uint64_t issued_lane_slots

    ctypedef struct plan7_forward_resident_view:
        uint64_t database_generation
        uint64_t batch_generation
        uint64_t pass_count
        uint64_t special_count
        int32_t device_ordinal
        uint32_t reserved
        const float *specials

    ctypedef struct plan7_forward_output:
        pass

    ctypedef struct plan7_forward_snapshot_profile:
        uint64_t emission_offset
        uint64_t transition_offset
        uint32_t q
        uint32_t model_length
        float e_move
        float e_loop
        float f_tau
        float f_lambda
        float nj
        int32_t mode

    int plan7_forward_database_create(
        const uintptr_t *profile_pointers,
        size_t profile_count,
        plan7_forward_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_forward_database_destroy(
        plan7_forward_database **database,
        char *error,
        size_t error_size,
    )

    size_t plan7_forward_database_profile_count(
        const plan7_forward_database *database,
    )

    uint64_t plan7_forward_database_device_bytes(
        const plan7_forward_database *database,
    )

    float plan7_forward_database_pack_milliseconds(
        const plan7_forward_database *database,
    )

    float plan7_forward_database_upload_milliseconds(
        const plan7_forward_database *database,
    )

    int plan7_forward_database_get_profile_snapshot(
        const plan7_forward_database *database,
        size_t profile_index,
        plan7_forward_snapshot_profile *profile,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace_variant(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        int candidates_per_warp,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace_reason_facts(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace_resident(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace_resident_reason_facts(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace_postfilter_resident(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        const plan7_postfilter_f2_resident_view *postfilter_view,
        double f3,
        uint64_t gathered_byte_budget,
        int collect_reason_facts,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace_f3_audit(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_output_destroy(
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    size_t plan7_forward_output_result_count(
        const plan7_forward_output *output,
    )

    const plan7_forward_result *plan7_forward_output_results(
        const plan7_forward_output *output,
    )

    size_t plan7_forward_output_reason_count(
        const plan7_forward_output *output,
    )

    const uint16_t *plan7_forward_output_reason_facts(
        const plan7_forward_output *output,
    )

    const uint64_t *plan7_forward_output_special_offsets(
        const plan7_forward_output *output,
    )

    size_t plan7_forward_output_special_count(
        const plan7_forward_output *output,
    )

    const float *plan7_forward_output_specials(
        const plan7_forward_output *output,
    )

    const plan7_forward_statistics *plan7_forward_output_statistics(
        const plan7_forward_output *output,
    )

    const plan7_forward_residency_statistics *plan7_forward_output_residency_statistics(
        const plan7_forward_output *output,
    )

    const plan7_forward_input_residency_statistics *plan7_forward_output_input_residency_statistics(
        const plan7_forward_output *output,
    )

    const plan7_forward_subwarp_statistics *plan7_forward_output_subwarp_statistics(
        const plan7_forward_output *output,
    )

    int plan7_forward_output_get_resident_view(
        const plan7_forward_output *output,
        plan7_forward_resident_view *view,
        char *error,
        size_t error_size,
    )

    const plan7_forward_f3_device_statistics *plan7_forward_output_f3_device_statistics(
        const plan7_forward_output *output,
    )

    float plan7_forward_output_upload_milliseconds(
        const plan7_forward_output *output,
    )

    float plan7_forward_output_total_milliseconds(
        const plan7_forward_output *output,
    )

    int plan7_forward_output_contract_fallback(
        const plan7_forward_output *output,
    )

    const plan7_forward_provenance *plan7_forward_output_provenance(
        const plan7_forward_output *output,
    )


cdef extern from "f3_threshold.h" nogil:
    cdef enum plan7_f3_threshold_reason:
        PLAN7_F3_THRESHOLD_REASON_NONE
        PLAN7_F3_THRESHOLD_REASON_INVALID_PARAMETERS
        PLAN7_F3_THRESHOLD_REASON_NO_NUMERIC_PASS
        PLAN7_F3_THRESHOLD_REASON_CERTIFICATE_FAILED

    ctypedef struct plan7_f3_threshold:
        uint32_t tau_bits
        uint32_t lambda_bits
        uint64_t f3_bits
        uint32_t threshold_bits
        uint32_t predecessor_bits
        uint32_t successor_bits
        uint8_t supported
        uint8_t reason
        uint8_t has_predecessor
        uint8_t has_successor
        uint8_t negative_infinity_pass
        uint8_t predecessor_pass
        uint8_t threshold_pass
        uint8_t successor_pass
        uint8_t positive_infinity_pass
        uint8_t quiet_nan_oracle_pass
        uint8_t nan_requires_fallback
        uint8_t reserved

    ctypedef struct plan7_f2_threshold:
        uint32_t mu_bits
        uint32_t lambda_bits
        uint64_t f2_bits
        uint32_t threshold_bits
        uint32_t predecessor_bits
        uint32_t successor_bits
        uint8_t supported
        uint8_t reason
        uint8_t has_predecessor
        uint8_t has_successor
        uint8_t negative_infinity_pass
        uint8_t predecessor_pass
        uint8_t threshold_pass
        uint8_t successor_pass
        uint8_t positive_infinity_pass
        uint8_t quiet_nan_oracle_pass
        uint8_t nan_requires_fallback
        uint8_t reserved

    int plan7_forward_compile_f3_threshold(
        float tau,
        float lambda_,
        double f3,
        plan7_f3_threshold *result,
    )

    int plan7_forward_f3_oracle_pass_bits(
        uint32_t bit_score_bits,
        float tau,
        float lambda_,
        double f3,
    )

    int plan7_postfilter_compile_f2_threshold(
        float mu,
        float lambda_,
        double f2,
        plan7_f2_threshold *result,
    )

    int plan7_postfilter_f2_oracle_pass_bits(
        uint32_t bit_score_bits,
        float mu,
        float lambda_,
        double f2,
    )


cdef extern from "backward_domain_cuda.h" nogil:
    cdef enum plan7_backward_domain_abi:
        PLAN7_BACKWARD_DOMAIN_RECORD_VERSION
        PLAN7_BACKWARD_DOMAIN_RECORD_SIZE
        PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE
        PLAN7_BACKWARD_DOMAIN_REGION_SIZE
        PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES
        PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS
        PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS

    cdef enum plan7_backward_domain_route:
        PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED
        PLAN7_BACKWARD_DOMAIN_NO_REGIONS
        PLAN7_BACKWARD_DOMAIN_SIMPLE

    cdef enum plan7_backward_domain_status:
        PLAN7_BACKWARD_DOMAIN_OK
        PLAN7_BACKWARD_DOMAIN_ERANGE
        PLAN7_BACKWARD_DOMAIN_ENORESULT
        PLAN7_BACKWARD_DOMAIN_EMPTY

    cdef enum plan7_backward_domain_reason_fact:
        PLAN7_BACKWARD_DOMAIN_REASON_TARGET_EMPTY
        PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SPECIAL_NONFINITE
        PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SCALE_INVALID
        PLAN7_BACKWARD_DOMAIN_REASON_HOST_FLOAT_ENV_INVALID
        PLAN7_BACKWARD_DOMAIN_REASON_MODE_OR_NJ_UNSUPPORTED
        PLAN7_BACKWARD_DOMAIN_REASON_WORK_CAP
        PLAN7_BACKWARD_DOMAIN_REASON_WORKSPACE_CAP
        PLAN7_BACKWARD_DOMAIN_REASON_TERMINAL_SCORE_INVALID
        PLAN7_BACKWARD_DOMAIN_REASON_POSTERIOR_OR_BACKWARD_SCORE_NONFINITE
        PLAN7_BACKWARD_DOMAIN_REASON_NEXPECTED_INVALID
        PLAN7_BACKWARD_DOMAIN_REASON_HAS_OWN_SCALES
        PLAN7_BACKWARD_DOMAIN_REASON_THRESHOLD_UNCERTAIN
        PLAN7_BACKWARD_DOMAIN_REASON_MULTIDOMAIN
        PLAN7_BACKWARD_DOMAIN_REASON_NO_REGIONS
        PLAN7_BACKWARD_DOMAIN_REASON_SIMPLE
        PLAN7_BACKWARD_DOMAIN_REASON_REGION_OUTPUT_CAP
        PLAN7_BACKWARD_DOMAIN_REASON_OTHER_CPU_REQUIRED
        PLAN7_BACKWARD_DOMAIN_REASON_FINAL_CPU_REQUIRED

    cdef enum plan7_backward_domain_test_fault:
        PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH
        PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH
        PLAN7_BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE

    ctypedef struct plan7_backward_domain_candidate:
        uint32_t profile_index
        uint32_t sequence_index

    ctypedef struct plan7_backward_domain_result:
        uint32_t profile_index
        uint32_t sequence_index
        float backward_score
        float nexpected
        uint32_t uncertain_count
        uint32_t region_count
        uint32_t multidomain_count
        uint8_t status
        uint8_t route
        uint8_t has_own_scales
        uint8_t reserved

    ctypedef struct plan7_domain_posterior:
        float btot
        float etot
        float mocc

    ctypedef struct plan7_simple_region:
        uint32_t begin
        uint32_t end

    ctypedef struct plan7_backward_domain_provenance:
        plan7_forward_provenance forward
        uint64_t threshold_hash
        uint64_t result_hash
        uint64_t region_hash
        uint64_t candidate_count
        uint64_t region_count

    ctypedef struct plan7_backward_domain_statistics:
        uint64_t candidate_count
        uint64_t device_result_count
        uint64_t cpu_required_count
        uint64_t work_cells
        uint64_t dp_workspace_bytes
        uint64_t backward_special_workspace_bytes
        uint64_t forward_special_workspace_bytes
        uint64_t posterior_bytes
        uint64_t simple_region_bytes
        uint64_t output_byte_limit
        uint64_t output_cap_fallback_count
        uint64_t work_cap_fallback_count
        uint64_t posterior_omitted_count
        uint64_t own_scale_count
        uint64_t threshold_uncertain_count
        uint64_t no_region_count
        uint64_t simple_count
        uint64_t multidomain_fallback_count
        float kernel_milliseconds
        float upload_milliseconds
        float download_milliseconds
        float total_milliseconds

    ctypedef struct plan7_backward_domain_residency_statistics:
        uint64_t forward_special_h2d_bytes
        uint64_t eliminated_forward_special_h2d_bytes
        uint64_t resident_input_count
        float forward_special_upload_milliseconds
        uint64_t resident_region_requested_bytes
        uint64_t resident_region_allocated_bytes
        uint64_t resident_region_materialized_bytes
        uint64_t resident_region_allocation_fallback_count
        float resident_region_allocation_milliseconds
        float resident_region_materialization_milliseconds

    ctypedef struct plan7_backward_domain_output:
        pass

    int plan7_backward_domain_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const plan7_forward_provenance *provenance,
        const uint64_t *forward_offsets,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_run_with_reason_facts(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const plan7_forward_provenance *provenance,
        const uint64_t *forward_offsets,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_run_from_forward_output(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const uint64_t *forward_offsets,
        const plan7_forward_output *forward_output,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_run_from_forward_output_with_reason_facts(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const uint64_t *forward_offsets,
        const plan7_forward_output *forward_output,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_route_all_cpu_from_forward_output(
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const uint64_t *forward_offsets,
        const plan7_forward_output *forward_output,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        int collect_reason_facts,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_unsealed_test_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const uint64_t *forward_offsets,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_output_destroy(
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    size_t plan7_backward_domain_output_result_count(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_result *plan7_backward_domain_output_results(
        const plan7_backward_domain_output *output,
    )

    const uint64_t *plan7_backward_domain_output_posterior_offsets(
        const plan7_backward_domain_output *output,
    )

    size_t plan7_backward_domain_output_posterior_count(
        const plan7_backward_domain_output *output,
    )

    const plan7_domain_posterior *plan7_backward_domain_output_posteriors(
        const plan7_backward_domain_output *output,
    )

    const uint64_t *plan7_backward_domain_output_region_offsets(
        const plan7_backward_domain_output *output,
    )

    size_t plan7_backward_domain_output_region_count(
        const plan7_backward_domain_output *output,
    )

    const plan7_simple_region *plan7_backward_domain_output_regions(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_provenance *plan7_backward_domain_output_provenance(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_statistics *plan7_backward_domain_output_statistics(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_residency_statistics *plan7_backward_domain_output_residency_statistics(
        const plan7_backward_domain_output *output,
    )

    size_t plan7_backward_domain_output_reason_count(
        const plan7_backward_domain_output *output,
    )

    const uint32_t *plan7_backward_domain_output_reason_facts(
        const plan7_backward_domain_output *output,
    )

    int c_plan7_backward_domain_merge_reason_facts_for_test \
            "plan7_backward_domain_merge_reason_facts_for_test"(
        const uint64_t *active_sources,
        const uint32_t *active_facts,
        size_t active_count,
        uint32_t *source_facts,
        size_t source_count,
    )

    int plan7_backward_domain_output_apply_test_fault(
        plan7_backward_domain_output *output,
        int fault,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_cpu_oracle(
        uintptr_t source_profile_pointer,
        const uint8_t *residues,
        size_t residue_count,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        plan7_backward_domain_result *result,
        plan7_domain_posterior *posteriors,
        size_t posterior_count,
        plan7_simple_region *regions,
        size_t region_capacity,
        size_t *region_count,
        char *error,
        size_t error_size,
    )


cdef extern from "domain_rescore_cuda.h" nogil:
    cdef enum plan7_domain_rescore_abi:
        PLAN7_DOMAIN_RESCORE_RECORD_VERSION
        PLAN7_DOMAIN_RESCORE_RECORD_SIZE
        PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE
        PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        PLAN7_DOMAIN_RESCORE_MAX_MATRIX_BYTES
        PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES

    cdef enum plan7_domain_rescore_action:
        PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
        PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT

    cdef enum plan7_domain_rescore_status:
        PLAN7_DOMAIN_RESCORE_OK
        PLAN7_DOMAIN_RESCORE_ERANGE
        PLAN7_DOMAIN_RESCORE_ENORESULT
        PLAN7_DOMAIN_RESCORE_ECAP
        PLAN7_DOMAIN_RESCORE_EMPTY

    cdef enum plan7_domain_rescore_reason_fact:
        PLAN7_DOMAIN_RESCORE_REASON_GLOBAL_COMPACT_BUDGET
        PLAN7_DOMAIN_RESCORE_REASON_OWN_SCALES
        PLAN7_DOMAIN_RESCORE_REASON_REGION_WORK_CAP
        PLAN7_DOMAIN_RESCORE_REASON_ROW_WORK_CAP
        PLAN7_DOMAIN_RESCORE_REASON_MATRIX_CAP
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAP
        PLAN7_DOMAIN_RESCORE_REASON_RUN_WORK_CAP
        PLAN7_DOMAIN_RESCORE_REASON_FORWARD_SCORE_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_BACKWARD_SCORE_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_SCALEPRODUCT_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_NULL2_OR_CORRECTION_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_OA_SCORE_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAPACITY_EXHAUSTED
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_ITERATION_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_PREDECESSOR_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_COORDINATES_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_IDENTITY_MISMATCH
        PLAN7_DOMAIN_RESCORE_REASON_HOST_RESULT_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_HOST_TRACE_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_HOST_NULL2_INVALID
        PLAN7_DOMAIN_RESCORE_REASON_ROW_ATOMIC_PROPAGATION
        PLAN7_DOMAIN_RESCORE_REASON_DEVICE_RESULT
        PLAN7_DOMAIN_RESCORE_REASON_OTHER_CPU_REQUIRED
        PLAN7_DOMAIN_RESCORE_REASON_UPSTREAM_OWN_SCALES
        PLAN7_DOMAIN_RESCORE_REASON_FINAL_CPU_REQUIRED
        PLAN7_DOMAIN_RESCORE_REASON_CERTIFIED_GA_REJECT

    ctypedef struct plan7_domain_rescore_result:
        uint32_t row_index
        uint32_t profile_index
        uint32_t sequence_index
        uint32_t envelope_begin
        uint32_t envelope_end
        uint32_t alignment_begin
        uint32_t alignment_end
        uint32_t model_begin
        uint32_t model_end
        float forward_score
        float backward_score
        float oa_score
        float domain_correction
        float score_consistency
        uint8_t status
        uint8_t action
        uint8_t has_own_scales
        uint8_t reserved
        uint32_t reserved2

    ctypedef struct plan7_domain_rescore_trace_step:
        uint32_t sequence_position
        uint32_t model_position
        float posterior
        uint8_t state
        uint8_t reserved[3]

    ctypedef struct plan7_domain_rescore_provenance:
        plan7_backward_domain_provenance backward
        uint64_t result_hash
        uint64_t trace_hash
        uint64_t null2_hash
        uint64_t result_count
        uint64_t trace_count
        uint64_t null2_count

    ctypedef struct plan7_domain_rescore_statistics:
        uint64_t upstream_row_count
        uint64_t simple_row_count
        uint64_t region_count
        uint64_t device_result_count
        uint64_t cpu_required_count
        uint64_t numeric_fallback_count
        uint64_t cap_fallback_count
        uint64_t global_cpu_fallback_count
        uint64_t work_cells
        uint64_t forward_matrix_bytes
        uint64_t posterior_matrix_bytes
        uint64_t special_workspace_bytes
        uint64_t trace_workspace_bytes
        uint64_t compact_output_byte_limit
        uint64_t compact_output_bytes
        float kernel_milliseconds
        float upload_milliseconds
        float download_milliseconds
        float total_milliseconds
        uint64_t certified_ga_row_count
        uint64_t certified_ga_region_count
        uint64_t certified_ga_skipped_work_cells
        float ga_classification_milliseconds

    ctypedef struct plan7_domain_rescore_residency_statistics:
        uint64_t upstream_h2d_bytes
        uint64_t eliminated_upstream_h2d_bytes
        uint64_t resident_selection_h2d_bytes
        uint64_t resident_input_count
        float upstream_upload_milliseconds
        float resident_prepare_milliseconds

    ctypedef struct plan7_domain_rescore_output:
        pass

    int plan7_domain_rescore_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_output *upstream,
        uint64_t compact_byte_budget,
        uint64_t matrix_byte_budget,
        uint64_t trace_byte_budget,
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_domain_rescore_run_with_reason_facts(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_output *upstream,
        uint64_t compact_byte_budget,
        uint64_t matrix_byte_budget,
        uint64_t trace_byte_budget,
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_domain_rescore_run_ga(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_output *upstream,
        const float *whole_forward_scores,
        size_t whole_forward_score_count,
        const float *target_ga_cutoffs,
        size_t target_ga_cutoff_count,
        uint64_t compact_byte_budget,
        uint64_t matrix_byte_budget,
        uint64_t trace_byte_budget,
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_domain_rescore_run_ga_with_reason_facts(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_output *upstream,
        const float *whole_forward_scores,
        size_t whole_forward_score_count,
        const float *target_ga_cutoffs,
        size_t target_ga_cutoff_count,
        uint64_t compact_byte_budget,
        uint64_t matrix_byte_budget,
        uint64_t trace_byte_budget,
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_domain_rescore_output_destroy(
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    size_t plan7_domain_rescore_output_result_count(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_result *plan7_domain_rescore_output_results(
        const plan7_domain_rescore_output *output,
    )

    const uint64_t *plan7_domain_rescore_output_trace_offsets(
        const plan7_domain_rescore_output *output,
    )

    size_t plan7_domain_rescore_output_trace_count(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_trace_step *plan7_domain_rescore_output_traces(
        const plan7_domain_rescore_output *output,
    )

    size_t plan7_domain_rescore_output_null2_count(
        const plan7_domain_rescore_output *output,
    )

    const float *plan7_domain_rescore_output_null2(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_provenance *plan7_domain_rescore_output_provenance(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_statistics *plan7_domain_rescore_output_statistics(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_residency_statistics *plan7_domain_rescore_output_residency_statistics(
        const plan7_domain_rescore_output *output,
    )

    size_t plan7_domain_rescore_output_reason_count(
        const plan7_domain_rescore_output *output,
    )

    const uint32_t *plan7_domain_rescore_output_reason_facts(
        const plan7_domain_rescore_output *output,
    )

    int c_plan7_domain_rescore_merge_reason_facts_for_test \
            "plan7_domain_rescore_merge_reason_facts_for_test"(
        const uint32_t *active_result_indices,
        const uint32_t *active_facts,
        size_t active_count,
        uint32_t *source_facts,
        size_t source_count,
    )

    int c_plan7_domain_rescore_own_scale_required_for_test \
            "plan7_domain_rescore_own_scale_required_for_test"(float xB)
    int c_plan7_domain_rescore_oatrace_j_predecessor_for_test \
            "plan7_domain_rescore_oatrace_j_predecessor_for_test"(
        float jpath,
        float epath,
        int j_loop_enabled,
        int e_loop_enabled,
    )

    int plan7_domain_rescore_cpu_oracle(
        uintptr_t source_profile_pointer,
        const uint8_t *residues,
        size_t residue_count,
        uint32_t envelope_begin,
        uint32_t envelope_end,
        plan7_domain_rescore_result *result,
        float *null2,
        size_t null2_count,
        plan7_domain_rescore_trace_step *trace,
        size_t trace_capacity,
        size_t *trace_count,
        char *error,
        size_t error_size,
    )


cdef extern from "continuation_journal.h":
    const char *PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME

    cdef enum plan7_continuation_journal_abi:
        PLAN7_CONTINUATION_JOURNAL_VERSION
        PLAN7_CONTINUATION_JOURNAL_MAGIC
        PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE

    cdef enum plan7_continuation_compact_route:
        PLAN7_CONTINUATION_COMPACT_NONE
        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
        PLAN7_CONTINUATION_COMPACT_DEVICE

    cdef enum plan7_continuation_journal_v3_source_stage:
        PLAN7_CONTINUATION_V3_RAW_F1_REJECT
        PLAN7_CONTINUATION_V3_BIAS_REJECT
        PLAN7_CONTINUATION_V3_F2_REJECT
        PLAN7_CONTINUATION_V3_F3_REJECT
        PLAN7_CONTINUATION_V3_DOMAIN_NO_REGIONS
        PLAN7_CONTINUATION_V3_CPU_REQUIRED
        PLAN7_CONTINUATION_V3_F2_SURVIVOR
        PLAN7_CONTINUATION_V3_F3_SURVIVOR
        PLAN7_CONTINUATION_V3_DOMAIN_CPU_REQUIRED
        PLAN7_CONTINUATION_V3_DOMAIN_SIMPLE
        PLAN7_CONTINUATION_V3_DOMAIN_COMPACT

    cdef enum plan7_continuation_journal_v3_exception_route:
        PLAN7_CONTINUATION_V3_FULL_PIPELINE
        PLAN7_CONTINUATION_V3_FILTER_SCORES
        PLAN7_CONTINUATION_V3_FORWARD_SCORES
        PLAN7_CONTINUATION_V3_SIMPLE_REGIONS
        PLAN7_CONTINUATION_V3_COMPACT_DOMAINS

    ctypedef struct plan7_continuation_journal_row:
        uint32_t profile_index
        uint32_t sequence_index
        float usc
        float filtersc
        float vfsc
        float fwdsc
        float backward_score
        float nexpected
        uint32_t uncertain_count
        uint32_t region_count
        uint32_t multidomain_count
        uint8_t postfilter_status
        uint8_t postfilter_action
        uint8_t forward_status
        uint8_t forward_action
        uint8_t domain_status
        uint8_t domain_route
        uint8_t has_own_scales
        uint8_t reserved
        uint32_t compact_result_count
        uint8_t compact_route
        uint8_t reserved2[3]
        uint32_t reserved3

    ctypedef struct plan7_continuation_journal:
        uint32_t magic
        uint16_t version
        uint16_t header_size
        uint32_t row_size
        uint32_t region_size
        uint32_t compact_result_size
        uint32_t compact_trace_step_size
        uint32_t compact_null2_stride
        uint64_t total_bytes
        uint64_t session_id
        uint64_t selection_id
        uint64_t profile_count
        uint64_t postfilter_count
        uint64_t forward_count
        uint64_t row_count
        uint64_t special_count
        uint64_t region_count
        uint64_t compact_result_count
        uint64_t compact_trace_offset_count
        uint64_t compact_trace_count
        uint64_t compact_null2_count
        uint64_t generation_tail_fingerprint
        uint64_t rescore_simple_row_count
        uint64_t rescore_device_result_count
        uint64_t rescore_cpu_required_count
        uint64_t rescore_numeric_fallback_count
        uint64_t rescore_cap_fallback_count
        uint64_t rescore_global_cpu_fallback_count
        uint64_t rescore_compact_output_byte_limit
        uint64_t rescore_compact_output_bytes
        uint64_t generation_f1_bits
        uint64_t generation_f2_bits
        uint64_t generation_f3_bits
        uint32_t rt1_bits
        uint32_t rt2_bits
        uint32_t rt3_bits
        uint32_t guard_band_bits
        uint8_t generation_bias_filter
        uint8_t generation_compact_domains
        uint8_t compact_global_fallback
        uint8_t sequence_content_fingerprint[32]
        uint64_t postfilter_offsets_offset
        uint64_t postfilter_records_offset
        uint64_t forward_offsets_offset
        uint64_t forward_records_offset
        uint64_t forward_special_offsets_offset
        uint64_t profile_offsets_offset
        uint64_t identity_tokens_offset
        uint64_t profile_fingerprints_offset
        uint64_t rows_offset
        uint64_t special_offsets_offset
        uint64_t specials_offset
        uint64_t region_offsets_offset
        uint64_t regions_offset
        uint64_t compact_row_offsets_offset
        uint64_t compact_results_offset
        uint64_t compact_trace_offsets_offset
        uint64_t compact_traces_offset
        uint64_t compact_null2_offset
        plan7_forward_provenance forward
        plan7_backward_domain_provenance backward
        plan7_domain_rescore_provenance rescore
        uint64_t integrity_tag

    uint64_t plan7_continuation_journal_integrity(
        const plan7_continuation_journal *journal,
    ) nogil

    uint64_t plan7_continuation_journal_v3_decision_term(
        uint64_t source_index,
        uint8_t decision,
    ) nogil

    uint64_t plan7_continuation_journal_v3_decision_seed(
        uint64_t postfilter_count,
        uint64_t profile_count,
        uint64_t session_id,
        uint64_t selection_id,
        uint64_t batch_generation,
        uint64_t f1_bits,
        uint64_t f2_bits,
        uint64_t f3_bits,
        uint64_t generation_tail_fingerprint,
    ) nogil

    uint64_t plan7_continuation_journal_v3_decision_finish(
        uint64_t state,
        uint64_t exception_count,
        uint64_t special_count,
        uint64_t region_count,
        uint64_t compact_result_count,
        uint64_t compact_trace_count,
        uint64_t compact_null2_count,
    ) nogil


cdef union float_bits:
    float value
    uint32_t bits


cdef union double_bits:
    double value
    uint64_t bits


cdef void _continuation_journal_capsule_destroy(object capsule) noexcept:
    cdef void *pointer
    if PyCapsule_IsValid(capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME):
        pointer = PyCapsule_GetPointer(
            capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
        )
        if pointer != NULL:
            free(pointer)


cdef bint _journal_append_storage(
    size_t *cursor,
    size_t count,
    size_t item_size,
    uint64_t *offset,
) noexcept:
    cdef size_t aligned
    cdef size_t byte_count
    if cursor[0] > (<size_t> -1) - 7:
        return False
    aligned = (cursor[0] + 7) & ~<size_t> 7
    if count != 0 and item_size > (<size_t> -1) // count:
        return False
    byte_count = count * item_size
    if aligned > (<size_t> -1) - byte_count:
        return False
    offset[0] = <uint64_t> aligned
    cursor[0] = aligned + byte_count
    return True


cdef bytes _copy_native_bytes(
    const void *source,
    size_t count,
    size_t item_size,
    const char *label,
):
    cdef size_t byte_count
    cdef bytes storage
    if count != 0 and item_size > (<size_t> -1) // count:
        raise OverflowError((<bytes> label).decode() + " size overflows size_t")
    byte_count = count * item_size
    if byte_count > <size_t> PY_SSIZE_T_MAX:
        raise OverflowError((<bytes> label).decode() + " exceeds Python limits")
    storage = PyBytes_FromStringAndSize(NULL, byte_count)
    if byte_count:
        if source == NULL:
            raise RuntimeError((<bytes> label).decode() + " source is null")
        memcpy(PyBytes_AS_STRING(storage), source, byte_count)
    return storage


cdef inline uint8_t _direct_v3_decision(
    uint8_t stage,
    uint8_t route,
) noexcept nogil:
    return <uint8_t> stage | (<uint8_t> route << 4)


cdef inline void _direct_v3_plan_initial(
    uint8_t *plan,
    size_t source_index,
    uint8_t decision,
    uint64_t *terms,
    uint64_t *exception_count,
) noexcept nogil:
    plan[source_index] = decision
    terms[0] ^= plan7_continuation_journal_v3_decision_term(
        <uint64_t> source_index, decision
    )
    if decision >> 4:
        exception_count[0] += 1


cdef inline void _direct_v3_plan_replace(
    uint8_t *plan,
    size_t source_index,
    uint8_t decision,
    uint64_t *terms,
) noexcept nogil:
    terms[0] ^= plan7_continuation_journal_v3_decision_term(
        <uint64_t> source_index, plan[source_index]
    )
    plan[source_index] = decision
    terms[0] ^= plan7_continuation_journal_v3_decision_term(
        <uint64_t> source_index, decision
    )


cdef object _build_continuation_journal_capsule(
    const plan7_profile_selection_view *view,
    const uint8_t *profile_fingerprints,
    const uint8_t *sequence_content_fingerprint,
    const plan7_postfilter_result *postfilter_records,
    size_t postfilter_count,
    const uint64_t *postfilter_offsets,
    const plan7_postfilter_result *candidate_records,
    const float *uncorrected_scores,
    const uint32_t *candidate_profiles,
    const size_t *pass_sources,
    const uint64_t *forward_profile_offsets,
    const uint64_t *profile_offsets,
    size_t pass_count,
    const uint64_t *pass_special_offsets,
    const plan7_forward_output *forward_output,
    const plan7_backward_domain_output *domain_output,
    const plan7_domain_rescore_output *rescore_output,
    uint64_t generation_tail_fingerprint,
    double f1,
    double f2,
    double f3,
    bint bias_filter,
    float rt1,
    float rt2,
    float rt3,
    float guard_band,
    uint64_t *journal_total_bytes,
    object postfilter_owner=None,
    bint direct_sparse_v3=False,
    object direct_decision_plan=None,
    uint64_t direct_decision_terms=0,
    uint64_t direct_exception_count=0,
    uint64_t direct_special_count=0,
    uint64_t direct_region_count=0,
    uint64_t direct_compact_result_count=0,
    uint64_t direct_compact_trace_count=0,
    uint64_t direct_compact_null2_count=0,
):
    cdef const plan7_forward_result *forward_results
    cdef const float *forward_specials
    cdef const plan7_forward_provenance *forward_provenance
    cdef const plan7_backward_domain_result *domain_results
    cdef const uint64_t *domain_region_offsets
    cdef const plan7_simple_region *domain_regions
    cdef const plan7_backward_domain_provenance *domain_provenance
    cdef const plan7_domain_rescore_result *rescore_results = NULL
    cdef const uint64_t *rescore_trace_offsets = NULL
    cdef const plan7_domain_rescore_trace_step *rescore_traces = NULL
    cdef const float *rescore_null2 = NULL
    cdef const plan7_domain_rescore_provenance *rescore_provenance = NULL
    cdef const plan7_domain_rescore_statistics *rescore_statistics = NULL
    cdef size_t forward_count
    cdef size_t special_count
    cdef size_t domain_count
    cdef size_t region_count
    cdef size_t compact_result_count = 0
    cdef size_t compact_trace_count = 0
    cdef size_t compact_null2_count = 0
    cdef size_t cursor = sizeof(plan7_continuation_journal)
    cdef size_t row
    cdef size_t source
    cdef size_t profile
    cdef size_t simple_row_count = 0
    cdef size_t device_result_count = 0
    cdef size_t cpu_required_count = 0
    cdef size_t certified_ga_count = 0
    cdef size_t compact_bytes = 0
    cdef size_t region_begin
    cdef size_t region_end
    cdef uint8_t row_action
    cdef const plan7_domain_rescore_result *compact_result
    cdef uint64_t profile_offsets_offset = 0
    cdef uint64_t identity_tokens_offset = 0
    cdef uint64_t profile_fingerprints_offset = 0
    cdef uint64_t postfilter_offsets_offset = 0
    cdef uint64_t postfilter_records_offset = 0
    cdef uint64_t forward_offsets_offset = 0
    cdef uint64_t forward_records_offset = 0
    cdef uint64_t forward_special_offsets_offset = 0
    cdef uint64_t rows_offset = 0
    cdef uint64_t special_offsets_offset = 0
    cdef uint64_t specials_offset = 0
    cdef uint64_t region_offsets_offset = 0
    cdef uint64_t regions_offset = 0
    cdef uint64_t compact_row_offsets_offset = 0
    cdef uint64_t compact_results_offset = 0
    cdef uint64_t compact_trace_offsets_offset = 0
    cdef uint64_t compact_traces_offset = 0
    cdef uint64_t compact_null2_offset = 0
    cdef plan7_continuation_journal *journal = NULL
    cdef plan7_continuation_journal_row *rows
    cdef uint64_t *target_u64
    cdef double_bits double_encoded
    cdef float_bits float_encoded
    cdef object capsule
    cdef object validation_start_ns = _time.perf_counter_ns()
    cdef object validation_elapsed_ns
    cdef object dense_emit_start_ns
    cdef object direct_start_ns
    cdef bytes direct_postfilter_storage
    cdef bytes direct_postfilter_offsets
    cdef bytes direct_forward_records
    cdef bytes direct_forward_offsets
    cdef bytes direct_forward_special_offsets
    cdef bytes direct_profile_offsets
    cdef bytes direct_rows
    cdef bytes direct_specials
    cdef bytes direct_region_offsets
    cdef bytes direct_regions
    cdef bytes direct_compact_row_offsets
    cdef bytes direct_compact_results
    cdef bytes direct_compact_trace_offsets
    cdef bytes direct_compact_traces
    cdef bytes direct_compact_null2
    cdef bytes direct_identity_tokens
    cdef bytes direct_profile_fingerprints
    cdef bytes direct_sequence_fingerprint
    cdef bytes direct_forward_provenance
    cdef bytes direct_domain_provenance
    cdef bytes direct_rescore_provenance
    cdef bytes direct_decision_storage
    cdef tuple direct_decision_counts
    cdef uint64_t direct_decision_tag = 0
    cdef double_bits direct_f1_encoded
    cdef double_bits direct_f2_encoded
    cdef double_bits direct_f3_encoded
    cdef uint64_t direct_staging_bytes = 0
    global _sealed_journal_build_count
    global _sealed_journal_payload_bytes
    global _sealed_journal_validation_ns
    global _sealed_journal_emit_ns
    global _direct_v3_staging_build_count
    global _direct_v3_eliminated_v2_bytes
    global _direct_v3_staging_payload_bytes
    global _direct_v3_staging_build_ns
    global _direct_v3_source_validation_ns

    forward_count = plan7_forward_output_result_count(forward_output)
    special_count = plan7_forward_output_special_count(forward_output)
    domain_count = plan7_backward_domain_output_result_count(domain_output)
    region_count = plan7_backward_domain_output_region_count(domain_output)
    if domain_count != pass_count:
        raise RuntimeError("Backward/domain journal result count changed")
    if rescore_output != NULL:
        compact_result_count = plan7_domain_rescore_output_result_count(
            rescore_output
        )
        compact_trace_count = plan7_domain_rescore_output_trace_count(
            rescore_output
        )
        compact_null2_count = plan7_domain_rescore_output_null2_count(
            rescore_output
        )
        rescore_results = plan7_domain_rescore_output_results(rescore_output)
        rescore_trace_offsets = (
            plan7_domain_rescore_output_trace_offsets(rescore_output)
        )
        rescore_traces = plan7_domain_rescore_output_traces(rescore_output)
        rescore_null2 = plan7_domain_rescore_output_null2(rescore_output)
        rescore_provenance = (
            plan7_domain_rescore_output_provenance(rescore_output)
        )
        rescore_statistics = (
            plan7_domain_rescore_output_statistics(rescore_output)
        )
        if generation_tail_fingerprint == 0:
            raise RuntimeError("compact-domain tail fingerprint is zero")
    elif generation_tail_fingerprint != 0:
        raise RuntimeError("compact-domain tail fingerprint has no output")
    if sizeof(uintptr_t) != sizeof(uint64_t):
        raise RuntimeError("profile identity tokens require a 64-bit process")
    if not (
        _journal_append_storage(
            &cursor, view.profile_count + 1, sizeof(uint64_t),
            &postfilter_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, postfilter_count, sizeof(plan7_postfilter_result),
            &postfilter_records_offset,
        )
        and _journal_append_storage(
            &cursor, view.profile_count + 1, sizeof(uint64_t),
            &forward_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, forward_count, sizeof(plan7_forward_result),
            &forward_records_offset,
        )
        and _journal_append_storage(
            &cursor, forward_count + 1, sizeof(uint64_t),
            &forward_special_offsets_offset,
        )
        and
        _journal_append_storage(
            &cursor, view.profile_count + 1, sizeof(uint64_t),
            &profile_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, view.profile_count, sizeof(uint64_t),
            &identity_tokens_offset,
        )
        and _journal_append_storage(
            &cursor,
            view.profile_count,
            PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
            &profile_fingerprints_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count, sizeof(plan7_continuation_journal_row),
            &rows_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count + 1, sizeof(uint64_t),
            &special_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, special_count, sizeof(float), &specials_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count + 1, sizeof(uint64_t),
            &region_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, region_count, sizeof(plan7_simple_region),
            &regions_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count + 1, sizeof(uint64_t),
            &compact_row_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, compact_result_count,
            sizeof(plan7_domain_rescore_result),
            &compact_results_offset,
        )
        and _journal_append_storage(
            &cursor, compact_result_count + 1, sizeof(uint64_t),
            &compact_trace_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step),
            &compact_traces_offset,
        )
        and _journal_append_storage(
            &cursor, compact_null2_count, sizeof(float),
            &compact_null2_offset,
        )
        and cursor <= <size_t> PY_SSIZE_T_MAX
    ):
        raise OverflowError("continuation journal size overflows size_t")

    forward_results = plan7_forward_output_results(forward_output)
    forward_specials = plan7_forward_output_specials(forward_output)
    forward_provenance = plan7_forward_output_provenance(forward_output)
    domain_results = plan7_backward_domain_output_results(domain_output)
    domain_region_offsets = plan7_backward_domain_output_region_offsets(
        domain_output
    )
    domain_regions = plan7_backward_domain_output_regions(domain_output)
    domain_provenance = plan7_backward_domain_output_provenance(domain_output)
    if (
        forward_provenance == NULL
        or domain_provenance == NULL
        or domain_region_offsets == NULL
        or (forward_count and forward_results == NULL)
        or (special_count and forward_specials == NULL)
        or (pass_count and domain_results == NULL)
        or (region_count and domain_regions == NULL)
    ):
        raise RuntimeError("continuation journal source storage is incomplete")
    if memcmp(
        forward_provenance,
        &domain_provenance.forward,
        sizeof(plan7_forward_provenance),
    ) != 0:
        raise RuntimeError("Forward and Backward/domain provenance differ")
    if (
        forward_provenance.pass_count != pass_count
        or forward_provenance.special_count != special_count
        or domain_provenance.candidate_count != pass_count
        or domain_provenance.region_count != region_count
        or pass_special_offsets[pass_count] != special_count
        or domain_region_offsets[pass_count] != region_count
    ):
        raise RuntimeError("continuation journal provenance counts differ")

    for row in range(pass_count):
        if domain_results[row].route == PLAN7_BACKWARD_DOMAIN_SIMPLE:
            simple_row_count += 1
    if rescore_output != NULL:
        if (
            rescore_provenance == NULL
            or rescore_statistics == NULL
            or rescore_trace_offsets == NULL
            or (compact_result_count and rescore_results == NULL)
            or (compact_trace_count and rescore_traces == NULL)
            or (compact_null2_count and rescore_null2 == NULL)
            or compact_result_count
            > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
            // sizeof(plan7_domain_rescore_result)
            or compact_trace_count
            > PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES
            // sizeof(plan7_domain_rescore_trace_step)
            or compact_result_count
            > (<size_t> -1) // PLAN7_DOMAIN_RESCORE_NULL2_COUNT
            or compact_null2_count
            != compact_result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
            or compact_result_count not in (0, region_count)
            or memcmp(
                &rescore_provenance.backward,
                domain_provenance,
                sizeof(plan7_backward_domain_provenance),
            ) != 0
            or rescore_provenance.result_count != compact_result_count
            or rescore_provenance.trace_count != compact_trace_count
            or rescore_provenance.null2_count != compact_null2_count
            or rescore_statistics.upstream_row_count != pass_count
            or rescore_statistics.simple_row_count != simple_row_count
            or rescore_statistics.region_count != region_count
            or rescore_statistics.device_result_count
            + rescore_statistics.cpu_required_count
            + rescore_statistics.certified_ga_region_count != region_count
            or rescore_statistics.numeric_fallback_count
            + rescore_statistics.cap_fallback_count
            != rescore_statistics.cpu_required_count
            or rescore_statistics.global_cpu_fallback_count
            not in (0, region_count)
            or rescore_statistics.compact_output_byte_limit < sizeof(uint64_t)
            or rescore_statistics.compact_output_byte_limit
            > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
            or rescore_statistics.compact_output_bytes
            > rescore_statistics.compact_output_byte_limit
        ):
            raise RuntimeError("compact-domain provenance or caps differ")
        compact_bytes = (
            compact_result_count * sizeof(plan7_domain_rescore_result)
            + (compact_result_count + 1) * sizeof(uint64_t)
            + compact_trace_count * sizeof(plan7_domain_rescore_trace_step)
            + compact_null2_count * sizeof(float)
        )
        if rescore_statistics.compact_output_bytes != compact_bytes:
            raise RuntimeError("compact-domain output byte count differs")
        if compact_result_count == 0:
            if (
                rescore_statistics.device_result_count != 0
                or rescore_statistics.cpu_required_count != region_count
                or rescore_statistics.certified_ga_region_count != 0
                or rescore_statistics.global_cpu_fallback_count != region_count
                or compact_trace_count != 0
            ):
                raise RuntimeError("compact-domain global fallback differs")
        else:
            if rescore_statistics.global_cpu_fallback_count != 0:
                raise RuntimeError("compact-domain fallback accounting differs")
            for row in range(pass_count):
                region_begin = domain_region_offsets[row]
                region_end = domain_region_offsets[row + 1]
                if region_begin == region_end:
                    continue
                row_action = rescore_results[region_begin].action
                for source in range(region_begin, region_end):
                    compact_result = &rescore_results[source]
                    if (
                        compact_result.row_index != row
                        or compact_result.profile_index
                        != domain_results[row].profile_index
                        or compact_result.sequence_index
                        != domain_results[row].sequence_index
                        or compact_result.envelope_begin
                        != domain_regions[source].begin
                        or compact_result.envelope_end
                        != domain_regions[source].end
                        or compact_result.action != row_action
                        or compact_result.action not in (
                            PLAN7_DOMAIN_RESCORE_CPU_REQUIRED,
                            PLAN7_DOMAIN_RESCORE_DEVICE_RESULT,
                            PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT,
                        )
                    ):
                        raise RuntimeError(
                            "compact-domain result order or row atomicity differs"
                        )
                    if compact_result.action == (
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                    ):
                        device_result_count += 1
                    elif compact_result.action == (
                        PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT
                    ):
                        certified_ga_count += 1
                    else:
                        cpu_required_count += 1
            if (
                device_result_count
                != rescore_statistics.device_result_count
                or cpu_required_count
                != rescore_statistics.cpu_required_count
                or certified_ga_count
                != rescore_statistics.certified_ga_region_count
            ):
                raise RuntimeError("compact-domain action counts differ")

    validation_elapsed_ns = _time.perf_counter_ns() - validation_start_ns
    if direct_sparse_v3:
        _direct_v3_source_validation_ns += <uint64_t> validation_elapsed_ns
        # Phase 1B deliberately avoids the dense v2 allocation.  The native
        # outputs are still live here, so copy only the segmented generation
        # facts needed by the already-oracled v3 planner.  The large
        # post-filter byte string is reused without a second copy.
        direct_start_ns = _time.perf_counter_ns()
        if type(postfilter_owner) is not bytes:
            raise TypeError("direct v3 post-filter storage must be bytes")
        if len(postfilter_owner) != postfilter_count * sizeof(
            plan7_postfilter_result
        ):
            raise ValueError("direct v3 post-filter storage size changed")
        if type(direct_decision_plan) is not bytes:
            raise TypeError("direct v3 decision plan must be immutable bytes")
        if len(direct_decision_plan) != postfilter_count:
            raise ValueError("direct v3 decision plan size changed")
        if (
            direct_exception_count > postfilter_count
            or direct_special_count > special_count
            or direct_region_count > region_count
            or direct_compact_result_count > compact_result_count
            or direct_compact_trace_count > compact_trace_count
            or direct_compact_null2_count > compact_null2_count
        ):
            raise ValueError("direct v3 sparse payload counts exceed sources")
        direct_postfilter_storage = postfilter_owner
        direct_decision_storage = direct_decision_plan
        direct_postfilter_offsets = _copy_native_bytes(
            postfilter_offsets,
            view.profile_count + 1,
            sizeof(uint64_t),
            b"direct v3 post-filter offsets",
        )
        direct_forward_records = _copy_native_bytes(
            forward_results,
            forward_count,
            sizeof(plan7_forward_result),
            b"direct v3 Forward records",
        )
        direct_forward_offsets = _copy_native_bytes(
            forward_profile_offsets,
            view.profile_count + 1,
            sizeof(uint64_t),
            b"direct v3 Forward profile offsets",
        )
        direct_forward_special_offsets = _copy_native_bytes(
            plan7_forward_output_special_offsets(forward_output),
            forward_count + 1,
            sizeof(uint64_t),
            b"direct v3 Forward special offsets",
        )
        direct_profile_offsets = _copy_native_bytes(
            profile_offsets,
            view.profile_count + 1,
            sizeof(uint64_t),
            b"direct v3 domain profile offsets",
        )
        direct_specials = _copy_native_bytes(
            forward_specials,
            special_count,
            sizeof(float),
            b"direct v3 Forward specials",
        )
        direct_region_offsets = _copy_native_bytes(
            domain_region_offsets,
            pass_count + 1,
            sizeof(uint64_t),
            b"direct v3 region offsets",
        )
        direct_regions = _copy_native_bytes(
            domain_regions,
            region_count,
            sizeof(plan7_simple_region),
            b"direct v3 regions",
        )
        if compact_result_count:
            direct_compact_row_offsets = _copy_native_bytes(
                domain_region_offsets,
                pass_count + 1,
                sizeof(uint64_t),
                b"direct v3 compact row offsets",
            )
        else:
            direct_compact_row_offsets = bytes(
                (pass_count + 1) * sizeof(uint64_t)
            )
        direct_compact_results = _copy_native_bytes(
            rescore_results,
            compact_result_count,
            sizeof(plan7_domain_rescore_result),
            b"direct v3 compact results",
        )
        if rescore_output != NULL:
            direct_compact_trace_offsets = _copy_native_bytes(
                rescore_trace_offsets,
                compact_result_count + 1,
                sizeof(uint64_t),
                b"direct v3 compact trace offsets",
            )
        else:
            direct_compact_trace_offsets = bytes(sizeof(uint64_t))
        direct_compact_traces = _copy_native_bytes(
            rescore_traces,
            compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step),
            b"direct v3 compact traces",
        )
        direct_compact_null2 = _copy_native_bytes(
            rescore_null2,
            compact_null2_count,
            sizeof(float),
            b"direct v3 compact null2",
        )
        direct_rows = PyBytes_FromStringAndSize(
            NULL, pass_count * sizeof(plan7_continuation_journal_row)
        )
        if pass_count:
            memset(
                PyBytes_AS_STRING(direct_rows),
                0,
                pass_count * sizeof(plan7_continuation_journal_row),
            )
        rows = <plan7_continuation_journal_row *> PyBytes_AS_STRING(
            direct_rows
        )
        for row in range(pass_count):
            source = pass_sources[row]
            if source >= forward_count:
                raise RuntimeError("direct v3 source index changed")
            if (
                forward_results[source].action
                != PLAN7_FORWARD_DEFINITE_PASS
                or domain_results[row].profile_index
                != candidate_profiles[source]
                or domain_results[row].sequence_index
                != forward_results[source].sequence_index
            ):
                raise RuntimeError("direct v3 row identity changed")
            rows[row].profile_index = domain_results[row].profile_index
            rows[row].sequence_index = domain_results[row].sequence_index
            rows[row].usc = uncorrected_scores[source]
            rows[row].filtersc = candidate_records[source].filtersc
            rows[row].vfsc = candidate_records[source].vfsc
            rows[row].fwdsc = forward_results[source].fwdsc
            rows[row].backward_score = domain_results[row].backward_score
            rows[row].nexpected = domain_results[row].nexpected
            rows[row].uncertain_count = domain_results[row].uncertain_count
            rows[row].region_count = domain_results[row].region_count
            rows[row].multidomain_count = domain_results[row].multidomain_count
            rows[row].postfilter_status = candidate_records[source].msv_status
            rows[row].postfilter_action = candidate_records[source].action
            rows[row].forward_status = forward_results[source].status
            rows[row].forward_action = forward_results[source].action
            rows[row].domain_status = domain_results[row].status
            rows[row].domain_route = domain_results[row].route
            rows[row].has_own_scales = domain_results[row].has_own_scales
            rows[row].reserved = domain_results[row].reserved
            region_begin = domain_region_offsets[row]
            region_end = domain_region_offsets[row + 1]
            if region_end - region_begin > <size_t> 0xffffffff:
                raise RuntimeError("direct v3 compact row count exceeds uint32")
            if compact_result_count:
                rows[row].compact_result_count = <uint32_t> (
                    region_end - region_begin
                )
                if region_begin != region_end:
                    if rescore_results[region_begin].action == (
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                    ):
                        rows[row].compact_route = (
                            PLAN7_CONTINUATION_COMPACT_DEVICE
                        )
                    elif rescore_results[region_begin].action == (
                        PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
                    ):
                        rows[row].compact_route = (
                            PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                        )
                    elif rescore_results[region_begin].action != (
                        PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT
                    ):
                        raise RuntimeError(
                            "direct v3 compact action is invalid"
                        )
            elif (
                rescore_output != NULL
                and rescore_statistics.global_cpu_fallback_count != 0
                and region_begin != region_end
            ):
                rows[row].compact_route = (
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                )

        direct_identity_tokens = PyBytes_FromStringAndSize(
            NULL, view.profile_count * sizeof(uint64_t)
        )
        target_u64 = <uint64_t *> PyBytes_AS_STRING(direct_identity_tokens)
        for profile in range(view.profile_count):
            target_u64[profile] = <uint64_t> view.identity_tokens[profile]
        direct_profile_fingerprints = _copy_native_bytes(
            profile_fingerprints,
            view.profile_count,
            PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
            b"direct v3 profile fingerprints",
        )
        direct_sequence_fingerprint = _copy_native_bytes(
            sequence_content_fingerprint,
            32,
            1,
            b"direct v3 sequence fingerprint",
        )
        direct_forward_provenance = _copy_native_bytes(
            forward_provenance,
            1,
            sizeof(plan7_forward_provenance),
            b"direct v3 Forward provenance",
        )
        direct_domain_provenance = _copy_native_bytes(
            domain_provenance,
            1,
            sizeof(plan7_backward_domain_provenance),
            b"direct v3 domain provenance",
        )
        if rescore_provenance != NULL:
            direct_rescore_provenance = _copy_native_bytes(
                rescore_provenance,
                1,
                sizeof(plan7_domain_rescore_provenance),
                b"direct v3 rescore provenance",
            )
        else:
            direct_rescore_provenance = bytes(
                sizeof(plan7_domain_rescore_provenance)
            )

        direct_staging_bytes = (
            len(direct_postfilter_offsets)
            + len(direct_forward_records)
            + len(direct_forward_offsets)
            + len(direct_forward_special_offsets)
            + len(direct_profile_offsets)
            + len(direct_rows)
            + len(direct_specials)
            + len(direct_region_offsets)
            + len(direct_regions)
            + len(direct_compact_row_offsets)
            + len(direct_compact_results)
            + len(direct_compact_trace_offsets)
            + len(direct_compact_traces)
            + len(direct_compact_null2)
            + len(direct_identity_tokens)
            + len(direct_profile_fingerprints)
            + len(direct_sequence_fingerprint)
            + len(direct_forward_provenance)
            + len(direct_domain_provenance)
            + len(direct_rescore_provenance)
            + len(direct_decision_storage)
        )
        if journal_total_bytes != NULL:
            # Counterfactual exact v2 allocation size, calculated by the same
            # checked layout code but never allocated on this path.
            journal_total_bytes[0] = <uint64_t> cursor
        float_encoded.value = guard_band
        double_encoded.value = f1
        direct_f1_encoded.value = f1
        direct_f2_encoded.value = f2
        direct_f3_encoded.value = f3
        direct_decision_tag = plan7_continuation_journal_v3_decision_seed(
            postfilter_count,
            view.profile_count,
            view.session_id,
            view.selection_id,
            forward_provenance.batch_generation,
            direct_f1_encoded.bits,
            direct_f2_encoded.bits,
            direct_f3_encoded.bits,
            generation_tail_fingerprint,
        ) ^ direct_decision_terms
        direct_decision_tag = plan7_continuation_journal_v3_decision_finish(
            direct_decision_tag,
            direct_exception_count,
            direct_special_count,
            direct_region_count,
            direct_compact_result_count,
            direct_compact_trace_count,
            direct_compact_null2_count,
        )
        direct_decision_counts = (
            direct_exception_count,
            direct_special_count,
            direct_region_count,
            direct_compact_result_count,
            direct_compact_trace_count,
            direct_compact_null2_count,
            direct_decision_tag,
        )
        direct_start_ns = _time.perf_counter_ns() - direct_start_ns
        _direct_v3_staging_build_count += 1
        _direct_v3_eliminated_v2_bytes += <uint64_t> cursor
        _direct_v3_staging_payload_bytes += direct_staging_bytes
        _direct_v3_staging_build_ns += <uint64_t> direct_start_ns
        return (
            DIRECT_V3_STAGING_SCHEMA_VERSION,
            direct_postfilter_storage,
            direct_postfilter_offsets,
            direct_forward_records,
            direct_forward_offsets,
            direct_forward_special_offsets,
            direct_specials,
            direct_profile_offsets,
            direct_rows,
            direct_region_offsets,
            direct_regions,
            direct_compact_row_offsets,
            direct_compact_results,
            direct_compact_trace_offsets,
            direct_compact_traces,
            direct_compact_null2,
            float_encoded.bits,
            generation_tail_fingerprint,
            bool(rescore_output != NULL),
            simple_row_count,
            (
                rescore_statistics.device_result_count
                if rescore_statistics != NULL else 0
            ),
            (
                rescore_statistics.cpu_required_count
                if rescore_statistics != NULL else 0
            ),
            (
                rescore_statistics.numeric_fallback_count
                if rescore_statistics != NULL else 0
            ),
            (
                rescore_statistics.cap_fallback_count
                if rescore_statistics != NULL else 0
            ),
            (
                rescore_statistics.global_cpu_fallback_count
                if rescore_statistics != NULL else 0
            ),
            <uint64_t> cursor,
            direct_start_ns,
            direct_staging_bytes,
            view.session_id,
            view.selection_id,
            forward_provenance.batch_generation,
            direct_identity_tokens,
            direct_profile_fingerprints,
            direct_sequence_fingerprint,
            double_encoded.bits,
            f2,
            f3,
            bool(bias_filter),
            direct_forward_provenance,
            direct_domain_provenance,
            direct_rescore_provenance,
            validation_elapsed_ns,
            direct_decision_storage,
            direct_decision_counts,
            (
                rescore_statistics.certified_ga_region_count
                if rescore_statistics != NULL else 0
            ),
        )

    _sealed_journal_validation_ns += <uint64_t> validation_elapsed_ns
    dense_emit_start_ns = _time.perf_counter_ns()
    journal = <plan7_continuation_journal *> calloc(1, cursor)
    if journal == NULL:
        raise MemoryError("continuation journal allocation failed")
    try:
        journal.magic = PLAN7_CONTINUATION_JOURNAL_MAGIC
        journal.version = PLAN7_CONTINUATION_JOURNAL_VERSION
        journal.header_size = sizeof(plan7_continuation_journal)
        journal.row_size = sizeof(plan7_continuation_journal_row)
        journal.region_size = sizeof(plan7_simple_region)
        journal.compact_result_size = sizeof(plan7_domain_rescore_result)
        journal.compact_trace_step_size = sizeof(
            plan7_domain_rescore_trace_step
        )
        journal.compact_null2_stride = PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        journal.total_bytes = cursor
        journal.session_id = view.session_id
        journal.selection_id = view.selection_id
        journal.profile_count = view.profile_count
        journal.postfilter_count = postfilter_count
        journal.forward_count = forward_count
        journal.row_count = pass_count
        journal.special_count = special_count
        journal.region_count = region_count
        journal.compact_result_count = compact_result_count
        journal.compact_trace_offset_count = compact_result_count + 1
        journal.compact_trace_count = compact_trace_count
        journal.compact_null2_count = compact_null2_count
        journal.generation_tail_fingerprint = generation_tail_fingerprint
        if rescore_output != NULL:
            journal.rescore_simple_row_count = (
                rescore_statistics.simple_row_count
            )
            journal.rescore_device_result_count = (
                rescore_statistics.device_result_count
            )
            journal.rescore_cpu_required_count = (
                rescore_statistics.cpu_required_count
            )
            journal.rescore_numeric_fallback_count = (
                rescore_statistics.numeric_fallback_count
            )
            journal.rescore_cap_fallback_count = (
                rescore_statistics.cap_fallback_count
            )
            journal.rescore_global_cpu_fallback_count = (
                rescore_statistics.global_cpu_fallback_count
            )
            journal.rescore_compact_output_byte_limit = (
                rescore_statistics.compact_output_byte_limit
            )
            journal.rescore_compact_output_bytes = (
                rescore_statistics.compact_output_bytes
            )
        double_encoded.value = f1
        journal.generation_f1_bits = double_encoded.bits
        double_encoded.value = f2
        journal.generation_f2_bits = double_encoded.bits
        double_encoded.value = f3
        journal.generation_f3_bits = double_encoded.bits
        float_encoded.value = rt1
        journal.rt1_bits = float_encoded.bits
        float_encoded.value = rt2
        journal.rt2_bits = float_encoded.bits
        float_encoded.value = rt3
        journal.rt3_bits = float_encoded.bits
        float_encoded.value = guard_band
        journal.guard_band_bits = float_encoded.bits
        journal.generation_bias_filter = <uint8_t> bias_filter
        journal.generation_compact_domains = <uint8_t> (
            rescore_output != NULL
        )
        if rescore_output != NULL:
            journal.compact_global_fallback = <uint8_t> (
                rescore_statistics.global_cpu_fallback_count != 0
            )
        memcpy(
            journal.sequence_content_fingerprint,
            sequence_content_fingerprint,
            32,
        )
        journal.postfilter_offsets_offset = postfilter_offsets_offset
        journal.postfilter_records_offset = postfilter_records_offset
        journal.forward_offsets_offset = forward_offsets_offset
        journal.forward_records_offset = forward_records_offset
        journal.forward_special_offsets_offset = forward_special_offsets_offset
        journal.profile_offsets_offset = profile_offsets_offset
        journal.identity_tokens_offset = identity_tokens_offset
        journal.profile_fingerprints_offset = profile_fingerprints_offset
        journal.rows_offset = rows_offset
        journal.special_offsets_offset = special_offsets_offset
        journal.specials_offset = specials_offset
        journal.region_offsets_offset = region_offsets_offset
        journal.regions_offset = regions_offset
        journal.compact_row_offsets_offset = compact_row_offsets_offset
        journal.compact_results_offset = compact_results_offset
        journal.compact_trace_offsets_offset = compact_trace_offsets_offset
        journal.compact_traces_offset = compact_traces_offset
        journal.compact_null2_offset = compact_null2_offset
        journal.forward = forward_provenance[0]
        journal.backward = domain_provenance[0]
        if rescore_output != NULL:
            journal.rescore = rescore_provenance[0]

        memcpy(
            <uint8_t *> journal + postfilter_offsets_offset,
            postfilter_offsets,
            (view.profile_count + 1) * sizeof(uint64_t),
        )
        if postfilter_count:
            memcpy(
                <uint8_t *> journal + postfilter_records_offset,
                postfilter_records,
                postfilter_count * sizeof(plan7_postfilter_result),
            )
        memcpy(
            <uint8_t *> journal + forward_offsets_offset,
            forward_profile_offsets,
            (view.profile_count + 1) * sizeof(uint64_t),
        )
        if forward_count:
            memcpy(
                <uint8_t *> journal + forward_records_offset,
                forward_results,
                forward_count * sizeof(plan7_forward_result),
            )
        memcpy(
            <uint8_t *> journal + forward_special_offsets_offset,
            plan7_forward_output_special_offsets(forward_output),
            (forward_count + 1) * sizeof(uint64_t),
        )
        memcpy(
            <uint8_t *> journal + profile_offsets_offset,
            profile_offsets,
            (view.profile_count + 1) * sizeof(uint64_t),
        )
        target_u64 = <uint64_t *> (
            <uint8_t *> journal + identity_tokens_offset
        )
        for profile in range(view.profile_count):
            target_u64[profile] = <uint64_t> view.identity_tokens[profile]
        if view.profile_count:
            memcpy(
                <uint8_t *> journal + profile_fingerprints_offset,
                profile_fingerprints,
                view.profile_count
                * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
            )
        memcpy(
            <uint8_t *> journal + special_offsets_offset,
            pass_special_offsets,
            (pass_count + 1) * sizeof(uint64_t),
        )
        if special_count:
            memcpy(
                <uint8_t *> journal + specials_offset,
                forward_specials,
                special_count * sizeof(float),
            )
        memcpy(
            <uint8_t *> journal + region_offsets_offset,
            domain_region_offsets,
            (pass_count + 1) * sizeof(uint64_t),
        )
        if region_count:
            memcpy(
                <uint8_t *> journal + regions_offset,
                domain_regions,
                region_count * sizeof(plan7_simple_region),
            )
        if compact_result_count:
            memcpy(
                <uint8_t *> journal + compact_row_offsets_offset,
                domain_region_offsets,
                (pass_count + 1) * sizeof(uint64_t),
            )
            memcpy(
                <uint8_t *> journal + compact_results_offset,
                rescore_results,
                compact_result_count * sizeof(plan7_domain_rescore_result),
            )
        if rescore_output != NULL:
            memcpy(
                <uint8_t *> journal + compact_trace_offsets_offset,
                rescore_trace_offsets,
                (compact_result_count + 1) * sizeof(uint64_t),
            )
        if compact_trace_count:
            memcpy(
                <uint8_t *> journal + compact_traces_offset,
                rescore_traces,
                compact_trace_count
                * sizeof(plan7_domain_rescore_trace_step),
            )
        if compact_null2_count:
            memcpy(
                <uint8_t *> journal + compact_null2_offset,
                rescore_null2,
                compact_null2_count * sizeof(float),
            )

        rows = <plan7_continuation_journal_row *> (
            <uint8_t *> journal + rows_offset
        )
        for row in range(pass_count):
            source = pass_sources[row]
            if source >= forward_count:
                raise RuntimeError("continuation journal source index changed")
            if (
                forward_results[source].action
                != PLAN7_FORWARD_DEFINITE_PASS
                or domain_results[row].profile_index
                != candidate_profiles[source]
                or domain_results[row].sequence_index
                != forward_results[source].sequence_index
            ):
                raise RuntimeError("continuation journal row identity changed")
            rows[row].profile_index = domain_results[row].profile_index
            rows[row].sequence_index = domain_results[row].sequence_index
            rows[row].usc = uncorrected_scores[source]
            rows[row].filtersc = candidate_records[source].filtersc
            rows[row].vfsc = candidate_records[source].vfsc
            rows[row].fwdsc = forward_results[source].fwdsc
            rows[row].backward_score = domain_results[row].backward_score
            rows[row].nexpected = domain_results[row].nexpected
            rows[row].uncertain_count = domain_results[row].uncertain_count
            rows[row].region_count = domain_results[row].region_count
            rows[row].multidomain_count = domain_results[row].multidomain_count
            rows[row].postfilter_status = candidate_records[source].msv_status
            rows[row].postfilter_action = candidate_records[source].action
            rows[row].forward_status = forward_results[source].status
            rows[row].forward_action = forward_results[source].action
            rows[row].domain_status = domain_results[row].status
            rows[row].domain_route = domain_results[row].route
            rows[row].has_own_scales = domain_results[row].has_own_scales
            rows[row].reserved = domain_results[row].reserved
            region_begin = domain_region_offsets[row]
            region_end = domain_region_offsets[row + 1]
            if region_end - region_begin > <size_t> 0xffffffff:
                raise RuntimeError("compact-domain row count exceeds uint32")
            if compact_result_count:
                rows[row].compact_result_count = <uint32_t> (
                    region_end - region_begin
                )
                if region_begin != region_end:
                    if rescore_results[region_begin].action == (
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                    ):
                        rows[row].compact_route = (
                            PLAN7_CONTINUATION_COMPACT_DEVICE
                        )
                    else:
                        rows[row].compact_route = (
                            PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                        )
            elif (
                rescore_output != NULL
                and rescore_statistics.global_cpu_fallback_count != 0
                and region_begin != region_end
            ):
                rows[row].compact_route = (
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                )

        journal.integrity_tag = plan7_continuation_journal_integrity(journal)
        if journal_total_bytes != NULL:
            journal_total_bytes[0] = <uint64_t> cursor
        capsule = PyCapsule_New(
            journal,
            PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME,
            _continuation_journal_capsule_destroy,
        )
        _sealed_journal_build_count += 1
        _sealed_journal_payload_bytes += <uint64_t> cursor
        _sealed_journal_emit_ns += <uint64_t> (
            _time.perf_counter_ns() - dense_emit_start_ns
        )
        journal = NULL
        return capsule
    finally:
        if journal != NULL:
            free(journal)


cdef enum generation_metric_index:
    GENERATION_MODEL_LENGTH = 0
    GENERATION_TARGET_COUNT = 1
    GENERATION_TARGET_RESIDUES = 2
    GENERATION_F1_CANDIDATE_COUNT = 3
    GENERATION_F1_REJECT_COUNT = 4
    GENERATION_F1_LOGICAL_CELLS = 5
    GENERATION_POSTFILTER_LOGICAL_CELLS = 6
    GENERATION_F2_PASS_COUNT = 7
    GENERATION_FORWARD_LOGICAL_CELLS = 8
    GENERATION_FORWARD_CPU_COUNT = 9
    GENERATION_FORWARD_REJECT_COUNT = 10
    GENERATION_FORWARD_PASS_COUNT = 11
    GENERATION_BACKWARD_LOGICAL_CELLS = 12
    GENERATION_BACKWARD_CPU_COUNT = 13
    GENERATION_BACKWARD_NO_REGION_COUNT = 14
    GENERATION_BACKWARD_SIMPLE_COUNT = 15
    GENERATION_JOURNAL_ROW_COUNT = 16
    GENERATION_JOURNAL_REGION_COUNT = 17
    GENERATION_RESCORE_LOGICAL_CELLS = 18
    GENERATION_RESCORE_CPU_COUNT = 19
    GENERATION_RESCORE_DEVICE_COUNT = 20
    GENERATION_RESCORE_REGION_COUNT = 21
    GENERATION_RESCORE_CERTIFIED_GA_COUNT = 22
    GENERATION_METRIC_COUNT = 23


cdef enum generation_reason_width:
    GENERATION_POSTFILTER_REASON_COUNT = 16
    GENERATION_F2_REASON_COUNT = 5
    GENERATION_FORWARD_REASON_COUNT = 10
    GENERATION_BACKWARD_REASON_COUNT = 18
    GENERATION_RESCORE_REASON_COUNT = 26


GENERATION_REASON_FACT_LAYOUT = (
    (
        PLAN7_POSTFILTER_REASON_RAW_F1_REJECT,
        PLAN7_POSTFILTER_REASON_MSV_RANGE_STATE,
        PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU,
        PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO,
        PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED,
        PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE,
        PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED,
        PLAN7_POSTFILTER_REASON_VITERBI_ERANGE,
        PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS,
        PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED,
        PLAN7_POSTFILTER_REASON_FINAL_REJECT,
        PLAN7_POSTFILTER_REASON_FINAL_PASS,
        PLAN7_POSTFILTER_REASON_OTHER_CPU_REQUIRED,
        PLAN7_POSTFILTER_REASON_CONTRACT_FALLBACK,
        PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED,
        PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED,
    ),
    (
        F2_REASON_POSTFILTER_NOT_PASS_OR_HOST_ENVIRONMENT_UNATTESTED,
        F2_REASON_INPUT_INVALID,
        F2_REASON_MSV_THRESHOLD_EXCEEDED,
        F2_REASON_VITERBI_THRESHOLD_EXCEEDED,
        F2_REASON_PASS,
    ),
    (
        PLAN7_FORWARD_REASON_KERNEL_STATUS_NON_OK,
        PLAN7_FORWARD_REASON_TARGET_EMPTY,
        PLAN7_FORWARD_REASON_FWDSC_NONFINITE,
        PLAN7_FORWARD_REASON_FILTERSC_NONFINITE,
        PLAN7_FORWARD_REASON_TAU_NONFINITE,
        PLAN7_FORWARD_REASON_LAMBDA_INVALID,
        PLAN7_FORWARD_REASON_F3_REJECT,
        PLAN7_FORWARD_REASON_OUTPUT_CAP,
        PLAN7_FORWARD_REASON_SURVIVOR_GATHERED,
        PLAN7_FORWARD_REASON_OTHER_CPU_REQUIRED,
    ),
    tuple(1 << index for index in range(GENERATION_BACKWARD_REASON_COUNT)),
    tuple(1 << index for index in range(GENERATION_RESCORE_REASON_COUNT)),
)
GENERATION_REASON_FACT_LAYOUT = (
    _telemetry_module.validate_generation_reason_fact_layout(
        GENERATION_REASON_FACT_LAYOUT
    )
)


cdef inline void _count_postfilter_reason(
    vector[uint64_t]& counts, size_t base, uint16_t facts,
) noexcept nogil:
    if facts & PLAN7_POSTFILTER_REASON_RAW_F1_REJECT: counts[base] += 1
    if facts & PLAN7_POSTFILTER_REASON_MSV_RANGE_STATE: counts[base + 1] += 1
    if facts & PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU: counts[base + 2] += 1
    if facts & PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO: counts[base + 3] += 1
    if facts & PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED: counts[base + 4] += 1
    if facts & PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE: counts[base + 5] += 1
    if facts & PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED: counts[base + 6] += 1
    if facts & PLAN7_POSTFILTER_REASON_VITERBI_ERANGE: counts[base + 7] += 1
    if facts & PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS: counts[base + 8] += 1
    if facts & PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED: counts[base + 9] += 1
    if facts & PLAN7_POSTFILTER_REASON_FINAL_REJECT: counts[base + 10] += 1
    if facts & PLAN7_POSTFILTER_REASON_FINAL_PASS: counts[base + 11] += 1
    if facts & PLAN7_POSTFILTER_REASON_OTHER_CPU_REQUIRED: counts[base + 12] += 1
    if facts & PLAN7_POSTFILTER_REASON_CONTRACT_FALLBACK: counts[base + 13] += 1
    if facts & PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED: counts[base + 14] += 1
    if facts & PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED: counts[base + 15] += 1


cdef inline uint64_t _postfilter_execution_count(uint16_t facts) noexcept nogil:
    return (
        (1 if facts & PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED else 0)
        + (1 if facts & PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED else 0)
    )


def postfilter_execution_cells_for_test(
    uint64_t sequence_length, uint64_t model_length, uint16_t facts,
):
    """Exercise the exact opt-in 0/1/2-times-L*M work attribution."""
    cdef uint64_t execution_count = _postfilter_execution_count(facts)
    cdef uint64_t base_cells
    if model_length != 0 and sequence_length > (<uint64_t> -1) // model_length:
        raise OverflowError("post-filter test cell product overflows uint64")
    base_cells = sequence_length * model_length
    if execution_count != 0 and base_cells > (<uint64_t> -1) // execution_count:
        raise OverflowError("post-filter test execution cells overflow uint64")
    return base_cells * execution_count


def compile_f3_threshold_for_test(float tau, float lambda_, double f3):
    """Compile the exact non-NaN binary32 HMMER F3 decision boundary."""
    cdef plan7_f3_threshold threshold
    cdef int status = plan7_forward_compile_f3_threshold(
        tau, lambda_, f3, &threshold
    )
    if status != 0:
        raise RuntimeError("F3 threshold compiler rejected its output buffer")
    return {
        "supported": threshold.supported != 0,
        "reason": threshold.reason,
        "tau_bits": threshold.tau_bits,
        "lambda_bits": threshold.lambda_bits,
        "f3_bits": threshold.f3_bits,
        "threshold_bits": (
            threshold.threshold_bits if threshold.supported else None
        ),
        "predecessor_bits": (
            threshold.predecessor_bits
            if threshold.supported and threshold.has_predecessor
            else None
        ),
        "successor_bits": (
            threshold.successor_bits
            if threshold.supported and threshold.has_successor
            else None
        ),
        "negative_infinity_pass": threshold.negative_infinity_pass != 0,
        "predecessor_pass": (
            threshold.predecessor_pass != 0
            if threshold.has_predecessor
            else None
        ),
        "threshold_pass": threshold.threshold_pass != 0,
        "successor_pass": (
            threshold.successor_pass != 0
            if threshold.has_successor
            else None
        ),
        "positive_infinity_pass": threshold.positive_infinity_pass != 0,
        "quiet_nan_oracle_pass": threshold.quiet_nan_oracle_pass != 0,
        "nan_requires_fallback": threshold.nan_requires_fallback != 0,
    }


def f3_oracle_pass_bits_for_test(
    uint32_t bit_score_bits, float tau, float lambda_, double f3,
):
    """Evaluate HMMER's exact host F3 predicate for raw binary32 score bits."""
    return plan7_forward_f3_oracle_pass_bits(
        bit_score_bits, tau, lambda_, f3
    ) != 0


def compile_f2_threshold_for_test(float mu, float lambda_, double f2):
    """Compile the exact non-NaN binary32 HMMER Gumbel/F2 boundary."""
    cdef plan7_f2_threshold threshold
    cdef int status = plan7_postfilter_compile_f2_threshold(
        mu, lambda_, f2, &threshold
    )
    if status != 0:
        raise RuntimeError("F2 threshold compiler rejected its output buffer")
    return {
        "supported": threshold.supported != 0,
        "reason": threshold.reason,
        "mu_bits": threshold.mu_bits,
        "lambda_bits": threshold.lambda_bits,
        "f2_bits": threshold.f2_bits,
        "threshold_bits": (
            threshold.threshold_bits if threshold.supported else None
        ),
        "predecessor_bits": (
            threshold.predecessor_bits
            if threshold.supported and threshold.has_predecessor
            else None
        ),
        "successor_bits": (
            threshold.successor_bits
            if threshold.supported and threshold.has_successor
            else None
        ),
        "negative_infinity_pass": threshold.negative_infinity_pass != 0,
        "predecessor_pass": (
            threshold.predecessor_pass != 0
            if threshold.has_predecessor
            else None
        ),
        "threshold_pass": threshold.threshold_pass != 0,
        "successor_pass": (
            threshold.successor_pass != 0
            if threshold.has_successor
            else None
        ),
        "positive_infinity_pass": threshold.positive_infinity_pass != 0,
        "quiet_nan_oracle_pass": threshold.quiet_nan_oracle_pass != 0,
        "nan_requires_fallback": threshold.nan_requires_fallback != 0,
    }


def f2_oracle_pass_bits_for_test(
    uint32_t bit_score_bits, float mu, float lambda_, double f2,
):
    """Evaluate HMMER's exact host F2 predicate for raw binary32 bits."""
    return plan7_postfilter_f2_oracle_pass_bits(
        bit_score_bits, mu, lambda_, f2
    ) != 0


cdef inline void _count_forward_reason(
    vector[uint64_t]& counts, size_t base, uint16_t facts,
) noexcept nogil:
    if facts & PLAN7_FORWARD_REASON_KERNEL_STATUS_NON_OK: counts[base] += 1
    if facts & PLAN7_FORWARD_REASON_TARGET_EMPTY: counts[base + 1] += 1
    if facts & PLAN7_FORWARD_REASON_FWDSC_NONFINITE: counts[base + 2] += 1
    if facts & PLAN7_FORWARD_REASON_FILTERSC_NONFINITE: counts[base + 3] += 1
    if facts & PLAN7_FORWARD_REASON_TAU_NONFINITE: counts[base + 4] += 1
    if facts & PLAN7_FORWARD_REASON_LAMBDA_INVALID: counts[base + 5] += 1
    if facts & PLAN7_FORWARD_REASON_F3_REJECT: counts[base + 6] += 1
    if facts & PLAN7_FORWARD_REASON_OUTPUT_CAP: counts[base + 7] += 1
    if facts & PLAN7_FORWARD_REASON_SURVIVOR_GATHERED: counts[base + 8] += 1
    if facts & PLAN7_FORWARD_REASON_OTHER_CPU_REQUIRED: counts[base + 9] += 1


cdef inline void _count_backward_reason(
    vector[uint64_t]& counts, size_t base, uint32_t facts,
) noexcept nogil:
    cdef uint32_t bit = 1
    cdef size_t index
    for index in range(GENERATION_BACKWARD_REASON_COUNT):
        if facts & bit: counts[base + index] += 1
        bit <<= 1


cdef inline void _count_rescore_reason(
    vector[uint64_t]& counts, size_t base, uint32_t facts,
) noexcept nogil:
    cdef uint32_t bit = 1
    cdef size_t index
    for index in range(GENERATION_RESCORE_REASON_COUNT):
        if facts & bit: counts[base + index] += 1
        bit <<= 1


cdef inline bint _rescore_reason_admitted_work(uint32_t facts) noexcept nogil:
    return not (
        facts & (
            PLAN7_DOMAIN_RESCORE_REASON_GLOBAL_COMPACT_BUDGET
            | PLAN7_DOMAIN_RESCORE_REASON_UPSTREAM_OWN_SCALES
            | PLAN7_DOMAIN_RESCORE_REASON_REGION_WORK_CAP
            | PLAN7_DOMAIN_RESCORE_REASON_ROW_WORK_CAP
            | PLAN7_DOMAIN_RESCORE_REASON_MATRIX_CAP
            | PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAP
            | PLAN7_DOMAIN_RESCORE_REASON_RUN_WORK_CAP
        )
    )


def domain_rescore_reason_admitted_work_for_test(uint32_t facts):
    """Host boundary for exact rescore preflight-versus-active facts."""
    if facts & <uint32_t> 0xfc000000:
        raise ValueError("rescore reason facts contain unknown bits")
    return bool(_rescore_reason_admitted_work(facts))


cdef inline void _count_reason_cells16(
    vector[uint64_t]& cells,
    size_t base,
    size_t width,
    uint16_t facts,
    uint64_t logical_cells,
) noexcept nogil:
    cdef uint16_t bit = 1
    cdef size_t index
    for index in range(width):
        if facts & bit: cells[base + index] += logical_cells
        bit <<= 1


cdef inline void _count_reason_cells32(
    vector[uint64_t]& cells,
    size_t base,
    size_t width,
    uint32_t facts,
    uint64_t logical_cells,
) noexcept nogil:
    cdef uint32_t bit = 1
    cdef size_t index
    for index in range(width):
        if facts & bit: cells[base + index] += logical_cells
        bit <<= 1


def bias_environment_attested():
    cdef char reason[512]
    cdef int attested
    reason[0] = 0
    with nogil:
        attested = plan7_bias_environment_attested(reason, sizeof(reason))
    return bool(attested), reason.decode("utf-8", "replace")


cdef object _bias_cuda_identity_dict(plan7_bias_cuda_identity *identity):
    cdef bytes uuid = PyBytes_FromStringAndSize(
        <char *> &identity.uuid[0], sizeof(identity.uuid)
    )
    return {
        "device_ordinal": identity.device_ordinal,
        "runtime_version": identity.runtime_version,
        "driver_version": identity.driver_version,
        "cudart_version": identity.cudart_version,
        "nvcc": {
            "major": identity.nvcc_major,
            "minor": identity.nvcc_minor,
            "build": identity.nvcc_build,
        },
        "compute_capability": [
            identity.compute_major, identity.compute_minor
        ],
        "multiprocessor_count": identity.multiprocessor_count,
        "pci_domain_id": identity.pci_domain_id,
        "pci_bus_id": identity.pci_bus_id,
        "pci_device_id": identity.pci_device_id,
        "total_global_memory": identity.total_global_memory,
        "uuid": (
            "GPU-"
            + uuid.hex()[0:8]
            + "-"
            + uuid.hex()[8:12]
            + "-"
            + uuid.hex()[12:16]
            + "-"
            + uuid.hex()[16:20]
            + "-"
            + uuid.hex()[20:32]
        ),
        "pci_bus_address": identity.pci_bus_address.decode(
            "ascii", "strict"
        ),
        "name": identity.name.decode("utf-8", "replace"),
    }


cdef object _bias_host_identity_dict(plan7_bias_host_identity *identity):
    return {
        "vendor_words": [
            identity.vendor_ebx,
            identity.vendor_edx,
            identity.vendor_ecx,
        ],
        "family": identity.family,
        "model": identity.model,
        "stepping": identity.stepping,
        "leaf1_ecx": identity.leaf1_ecx,
        "leaf1_edx": identity.leaf1_edx,
        "leaf7_ebx": identity.leaf7_ebx,
        "xcr0_low": identity.xcr0_low,
        "xcr0_high": identity.xcr0_high,
    }


def bias_environment_provenance():
    """Return the exact host, toolkit, driver-API, and device gate inputs."""
    cdef plan7_bias_cuda_identity cuda_identity
    cdef plan7_bias_host_identity host_identity
    cdef uint8_t libm_build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE]
    cdef char cuda_reason[512]
    cdef char host_reason[512]
    cdef char attestation_reason[512]
    cdef int cuda_status
    cdef int host_status
    cdef int libm_status
    cdef int libm_attested = 0
    cdef int target = PLAN7_BIAS_CUDA_UNATTESTED
    cdef int attested
    cuda_reason[0] = 0
    host_reason[0] = 0
    attestation_reason[0] = 0
    with nogil:
        cuda_status = plan7_bias_current_cuda_identity(
            &cuda_identity, cuda_reason, sizeof(cuda_reason)
        )
        host_status = plan7_bias_current_host_identity(
            &host_identity, host_reason, sizeof(host_reason)
        )
        if cuda_status == 0:
            target = plan7_bias_cuda_identity_target(
                &cuda_identity, cuda_reason, sizeof(cuda_reason)
            )
        libm_status = plan7_bias_current_libm_build_id(&libm_build_id[0])
        if libm_status == 0:
            libm_attested = plan7_bias_libm_build_id_attested(
                &libm_build_id[0], sizeof(libm_build_id), target
            )
        attested = plan7_bias_environment_attested(
            attestation_reason, sizeof(attestation_reason)
        )
    target_name = {
        PLAN7_BIAS_CUDA_SM75_RTX2080_TI: "sm75_rtx2080ti",
        PLAN7_BIAS_CUDA_SM90_H200: "sm90_h200",
    }.get(target, "unattested")
    return {
        "attested": bool(attested),
        "reason": attestation_reason.decode("utf-8", "replace"),
        "target": target_name,
        "target_code": target,
        "cuda": (
            _bias_cuda_identity_dict(&cuda_identity)
            if cuda_status == 0 else None
        ),
        "cuda_identity_reason": cuda_reason.decode("utf-8", "replace"),
        "host": (
            _bias_host_identity_dict(&host_identity)
            if host_status == 0 else None
        ),
        "host_identity_reason": host_reason.decode("utf-8", "replace"),
        "libm": {
            "gnu_build_id": (
                PyBytes_FromStringAndSize(
                    <char *> &libm_build_id[0], sizeof(libm_build_id)
                ).hex()
                if libm_status == 0 else None
            ),
            "attested_for_target": bool(libm_attested),
        },
    }


def _bias_cuda_identity_target_raw(
    str name,
    int compute_major,
    int compute_minor,
    int multiprocessor_count,
    uint64_t total_global_memory,
    bytes uuid,
    str pci_bus_address,
    int device_ordinal=0,
    int runtime_version=12050,
    int driver_version=12050,
    int cudart_version=12050,
    int nvcc_major=12,
    int nvcc_minor=5,
    int nvcc_build=82,
    int pci_domain_id=0,
    int pci_bus_id=1,
    int pci_device_id=0,
):
    """Classify a synthetic CUDA identity at the pure host unit boundary."""
    cdef plan7_bias_cuda_identity identity
    cdef char reason[512]
    cdef bytes encoded_name = name.encode("utf-8")
    cdef bytes encoded_pci = pci_bus_address.encode("ascii")
    cdef int target
    if len(uuid) != sizeof(identity.uuid):
        raise ValueError("CUDA UUID must contain exactly 16 bytes")
    if b"\0" in encoded_name or len(encoded_name) >= sizeof(identity.name):
        raise ValueError("CUDA device name does not fit the identity record")
    if b"\0" in encoded_pci or len(encoded_pci) >= sizeof(identity.pci_bus_address):
        raise ValueError("CUDA PCI address does not fit the identity record")
    memset(&identity, 0, sizeof(identity))
    identity.device_ordinal = device_ordinal
    identity.runtime_version = runtime_version
    identity.driver_version = driver_version
    identity.cudart_version = cudart_version
    identity.nvcc_major = nvcc_major
    identity.nvcc_minor = nvcc_minor
    identity.nvcc_build = nvcc_build
    identity.compute_major = compute_major
    identity.compute_minor = compute_minor
    identity.multiprocessor_count = multiprocessor_count
    identity.pci_domain_id = pci_domain_id
    identity.pci_bus_id = pci_bus_id
    identity.pci_device_id = pci_device_id
    identity.total_global_memory = total_global_memory
    memcpy(&identity.uuid[0], PyBytes_AS_STRING(uuid), sizeof(identity.uuid))
    memcpy(
        &identity.name[0], PyBytes_AS_STRING(encoded_name), len(encoded_name)
    )
    memcpy(
        &identity.pci_bus_address[0],
        PyBytes_AS_STRING(encoded_pci),
        len(encoded_pci),
    )
    reason[0] = 0
    with nogil:
        target = plan7_bias_cuda_identity_target(
            &identity, reason, sizeof(reason)
        )
    return target, reason.decode("utf-8", "replace")


def _bias_host_identity_attested_raw(
    int cuda_target,
    uint32_t family,
    uint32_t model,
    uint32_t stepping,
    uint32_t leaf1_ecx,
    uint32_t leaf1_edx,
    uint32_t leaf7_ebx,
    uint32_t xcr0_low,
    uint32_t xcr0_high=0,
    uint32_t vendor_ebx=0x756e6547,
    uint32_t vendor_edx=0x49656e69,
    uint32_t vendor_ecx=0x6c65746e,
):
    """Evaluate a synthetic CPU/accelerator pairing without a GPU."""
    cdef plan7_bias_host_identity identity
    cdef char reason[512]
    cdef int attested
    identity.vendor_ebx = vendor_ebx
    identity.vendor_edx = vendor_edx
    identity.vendor_ecx = vendor_ecx
    identity.family = family
    identity.model = model
    identity.stepping = stepping
    identity.leaf1_ecx = leaf1_ecx
    identity.leaf1_edx = leaf1_edx
    identity.leaf7_ebx = leaf7_ebx
    identity.xcr0_low = xcr0_low
    identity.xcr0_high = xcr0_high
    reason[0] = 0
    with nogil:
        attested = plan7_bias_host_identity_attested(
            &identity, cuda_target, reason, sizeof(reason)
        )
    return bool(attested), reason.decode("utf-8", "replace")


def _bias_libm_build_id_attested_raw(build_id, int cuda_target):
    """Evaluate a synthetic libm build ID at the pure host unit boundary."""
    cdef bytes encoded
    cdef const uint8_t *raw = NULL
    cdef size_t raw_size = 0
    cdef int attested
    if build_id is not None:
        if type(build_id) is not bytes:
            raise TypeError("libm build ID must be exactly bytes or None")
        encoded = build_id
        raw = <const uint8_t *> PyBytes_AS_STRING(encoded)
        raw_size = len(encoded)
    with nogil:
        attested = plan7_bias_libm_build_id_attested(
            raw, raw_size, cuda_target
        )
    return bool(attested)


def _sealed_journal_transport_statistics():
    """Return implementation-only fused transport allocation counters."""
    return {
        "build_count": _sealed_journal_build_count,
        "payload_bytes": _sealed_journal_payload_bytes,
        "duplicate_python_bytes": _sealed_journal_duplicate_python_bytes,
        "dense_v2_source_validation_ns": _sealed_journal_validation_ns,
        "dense_v2_emit_ns": _sealed_journal_emit_ns,
        "direct_v3_staging_build_count": _direct_v3_staging_build_count,
        "direct_v3_eliminated_v2_bytes": _direct_v3_eliminated_v2_bytes,
        "direct_v3_staging_payload_bytes": (
            _direct_v3_staging_payload_bytes
        ),
        "direct_v3_staging_build_ns": _direct_v3_staging_build_ns,
        "direct_v3_source_validation_ns": (
            _direct_v3_source_validation_ns
        ),
    }


def _forward_backward_residency_statistics():
    """Return cumulative exact transfer counters for resident continuation."""
    return {
        "forward_call_count": _resident_forward_call_count,
        "forward_requested_bytes": _resident_forward_requested_bytes,
        "forward_allocated_bytes": _resident_forward_allocated_bytes,
        "forward_materialized_bytes": _resident_forward_materialized_bytes,
        "forward_allocation_fallback_count": (
            _resident_forward_allocation_fallback_count
        ),
        "forward_allocation_ms": (
            _resident_forward_allocation_milliseconds
        ),
        "forward_materialization_ms": (
            _resident_forward_materialization_milliseconds
        ),
        "backward_call_count": _resident_backward_call_count,
        "backward_forward_h2d_bytes": (
            _resident_backward_forward_h2d_bytes
        ),
        "backward_eliminated_forward_h2d_bytes": (
            _resident_backward_eliminated_forward_h2d_bytes
        ),
        "backward_forward_upload_ms": (
            _resident_backward_forward_upload_milliseconds
        ),
        "backward_region_requested_bytes": (
            _resident_backward_region_requested_bytes
        ),
        "backward_region_allocated_bytes": (
            _resident_backward_region_allocated_bytes
        ),
        "backward_region_materialized_bytes": (
            _resident_backward_region_materialized_bytes
        ),
        "backward_region_allocation_fallback_count": (
            _resident_backward_region_allocation_fallback_count
        ),
        "backward_region_allocation_ms": (
            _resident_backward_region_allocation_milliseconds
        ),
        "backward_region_materialization_ms": (
            _resident_backward_region_materialization_milliseconds
        ),
        "rescore_call_count": _resident_rescore_call_count,
        "rescore_upstream_h2d_bytes": (
            _resident_rescore_upstream_h2d_bytes
        ),
        "rescore_eliminated_upstream_h2d_bytes": (
            _resident_rescore_eliminated_upstream_h2d_bytes
        ),
        "rescore_selection_h2d_bytes": (
            _resident_rescore_selection_h2d_bytes
        ),
        "rescore_upstream_upload_ms": (
            _resident_rescore_upstream_upload_milliseconds
        ),
        "rescore_prepare_ms": _resident_rescore_prepare_milliseconds,
    }


def _postfilter_forward_residency_statistics():
    """Return cumulative exact F2 compaction and resident-input counters."""
    return {
        "f2_compaction_run_count": _resident_f2_compaction_run_count,
        "f2_source_count": _resident_f2_source_count,
        "f2_selected_count": _resident_f2_selected_count,
        "f2_compiled_profile_count": _resident_f2_compiled_profile_count,
        "f2_unsupported_profile_count": (
            _resident_f2_unsupported_profile_count
        ),
        "f2_selected_d2h_bytes": _resident_f2_selected_d2h_bytes,
        "f2_compile_ms": _resident_f2_compile_milliseconds,
        "f2_upload_ms": _resident_f2_upload_milliseconds,
        "f2_kernel_ms": _resident_f2_kernel_milliseconds,
        "f2_scan_ms": _resident_f2_scan_milliseconds,
        "f2_download_ms": _resident_f2_download_milliseconds,
        "f2_total_ms": _resident_f2_total_milliseconds,
        "forward_resident_f2_call_count": _resident_forward_f2_call_count,
        "forward_resident_f2_candidate_count": (
            _resident_forward_f2_candidate_count
        ),
        "forward_eliminated_candidate_h2d_bytes": (
            _resident_forward_f2_eliminated_h2d_bytes
        ),
        "forward_resident_f2_gather_ms": (
            _resident_forward_f2_gather_milliseconds
        ),
    }


cdef void _accumulate_postfilter_f2_statistics(
    const plan7_postfilter_f2_statistics *statistics,
) noexcept:
    global _resident_f2_compaction_run_count
    global _resident_f2_source_count
    global _resident_f2_selected_count
    global _resident_f2_compiled_profile_count
    global _resident_f2_unsupported_profile_count
    global _resident_f2_selected_d2h_bytes
    global _resident_f2_compile_milliseconds
    global _resident_f2_upload_milliseconds
    global _resident_f2_kernel_milliseconds
    global _resident_f2_scan_milliseconds
    global _resident_f2_download_milliseconds
    global _resident_f2_total_milliseconds
    if statistics == NULL:
        return
    _resident_f2_compaction_run_count += statistics.run_count
    _resident_f2_source_count += statistics.source_count
    _resident_f2_selected_count += statistics.selected_count
    _resident_f2_compiled_profile_count += statistics.compiled_profile_count
    _resident_f2_unsupported_profile_count += (
        statistics.unsupported_profile_count
    )
    _resident_f2_selected_d2h_bytes += statistics.selected_d2h_bytes
    _resident_f2_compile_milliseconds += statistics.compile_milliseconds
    _resident_f2_upload_milliseconds += statistics.upload_milliseconds
    _resident_f2_kernel_milliseconds += statistics.kernel_milliseconds
    _resident_f2_scan_milliseconds += statistics.scan_milliseconds
    _resident_f2_download_milliseconds += statistics.download_milliseconds
    _resident_f2_total_milliseconds += statistics.total_milliseconds


cdef void _accumulate_forward_input_residency_statistics(
    const plan7_forward_input_residency_statistics *statistics,
) noexcept:
    global _resident_forward_f2_call_count
    global _resident_forward_f2_candidate_count
    global _resident_forward_f2_eliminated_h2d_bytes
    global _resident_forward_f2_gather_milliseconds
    if statistics == NULL:
        return
    _resident_forward_f2_call_count += statistics.resident_f2_call_count
    _resident_forward_f2_candidate_count += (
        statistics.resident_f2_candidate_count
    )
    _resident_forward_f2_eliminated_h2d_bytes += (
        statistics.eliminated_candidate_h2d_bytes
    )
    _resident_forward_f2_gather_milliseconds += statistics.gather_milliseconds


cdef void _accumulate_forward_f3_device_statistics(
    const plan7_forward_f3_device_statistics *statistics,
) noexcept:
    global _f3_compiled_profile_count
    global _f3_unsupported_profile_count
    global _f3_host_audit_count
    global _f3_host_decision_avoided_count
    global _f3_device_decision_count
    global _f3_device_reject_count
    global _f3_device_pass_count
    global _f3_host_fallback_count
    global _f3_decision_mismatch_count
    global _f3_device_compaction_run_count
    global _f3_device_compaction_candidate_count
    global _f3_device_compacted_survivor_count
    global _f3_survivor_upload_avoided_bytes
    if statistics == NULL:
        return
    _f3_compiled_profile_count += statistics.compiled_profile_count
    _f3_unsupported_profile_count += statistics.unsupported_profile_count
    _f3_host_audit_count += statistics.host_audit_count
    _f3_host_decision_avoided_count += (
        statistics.host_decision_avoided_count
    )
    _f3_device_decision_count += statistics.device_decision_count
    _f3_device_reject_count += statistics.device_reject_count
    _f3_device_pass_count += statistics.device_pass_count
    _f3_host_fallback_count += statistics.host_fallback_count
    _f3_decision_mismatch_count += statistics.decision_mismatch_count
    _f3_device_compaction_run_count += statistics.device_compaction_run_count
    _f3_device_compaction_candidate_count += (
        statistics.device_compaction_candidate_count
    )
    _f3_device_compacted_survivor_count += (
        statistics.device_compacted_survivor_count
    )
    _f3_survivor_upload_avoided_bytes += (
        statistics.survivor_upload_avoided_bytes
    )


def _forward_f3_device_statistics():
    """Return cumulative production Forward F3 decision/compaction counters."""
    return {
        "f3_compiled_profile_count": _f3_compiled_profile_count,
        "f3_unsupported_profile_count": _f3_unsupported_profile_count,
        "f3_host_audit_count": _f3_host_audit_count,
        "f3_host_decision_avoided_count": (
            _f3_host_decision_avoided_count
        ),
        "f3_device_decision_count": _f3_device_decision_count,
        "f3_device_reject_count": _f3_device_reject_count,
        "f3_device_pass_count": _f3_device_pass_count,
        "f3_host_fallback_count": _f3_host_fallback_count,
        "f3_decision_mismatch_count": _f3_decision_mismatch_count,
        "f3_device_compaction_run_count": _f3_device_compaction_run_count,
        "f3_device_compaction_candidate_count": (
            _f3_device_compaction_candidate_count
        ),
        "f3_device_compacted_survivor_count": (
            _f3_device_compacted_survivor_count
        ),
        "f3_survivor_upload_avoided_bytes": (
            _f3_survivor_upload_avoided_bytes
        ),
    }


cdef void _accumulate_forward_subwarp_statistics(
    const plan7_forward_subwarp_statistics *statistics,
) noexcept:
    global _subwarp_call_count
    global _subwarp_auto_call_count
    global _subwarp_width1_call_count
    global _subwarp_width2_call_count
    global _subwarp_width4_call_count
    global _subwarp_width8_call_count
    global _subwarp_no_kernel_count
    global _subwarp_forced_count
    global _subwarp_sparse_width1_count
    global _subwarp_short_width4_count
    global _subwarp_short_width2_count
    global _subwarp_long_width4_count
    global _subwarp_long_width2_count
    global _subwarp_long_saturated_width1_count
    global _subwarp_divergent_width1_count
    global _subwarp_kernel_launch_count
    global _subwarp_scheduled_warp_count
    global _subwarp_candidate_count
    global _subwarp_active_lane_slots
    global _subwarp_issued_lane_slots
    if statistics == NULL:
        return
    _subwarp_call_count += 1
    if statistics.requested_candidates_per_warp == 0:
        _subwarp_auto_call_count += 1
    if statistics.candidates_per_warp == 1:
        _subwarp_width1_call_count += 1
    elif statistics.candidates_per_warp == 2:
        _subwarp_width2_call_count += 1
    elif statistics.candidates_per_warp == 4:
        _subwarp_width4_call_count += 1
    elif statistics.candidates_per_warp == 8:
        _subwarp_width8_call_count += 1
    if statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_NO_KERNEL:
        _subwarp_no_kernel_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_FORCED:
        _subwarp_forced_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_SPARSE_WIDTH1:
        _subwarp_sparse_width1_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH4:
        _subwarp_short_width4_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH2:
        _subwarp_short_width2_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH4:
        _subwarp_long_width4_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH2:
        _subwarp_long_width2_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_LONG_SATURATED_WIDTH1:
        _subwarp_long_saturated_width1_count += 1
    elif statistics.policy_reason == PLAN7_FORWARD_SUBWARP_POLICY_DIVERGENT_WIDTH1:
        _subwarp_divergent_width1_count += 1
    _subwarp_kernel_launch_count += statistics.kernel_launch_count
    _subwarp_scheduled_warp_count += statistics.scheduled_warp_count
    _subwarp_candidate_count += statistics.candidate_subwarp_count
    _subwarp_active_lane_slots += statistics.active_lane_slots
    _subwarp_issued_lane_slots += statistics.issued_lane_slots


def _forward_subwarp_statistics():
    """Return cumulative production Forward subwarp policy counters."""
    return {
        "call_count": _subwarp_call_count,
        "auto_call_count": _subwarp_auto_call_count,
        "width1_call_count": _subwarp_width1_call_count,
        "width2_call_count": _subwarp_width2_call_count,
        "width4_call_count": _subwarp_width4_call_count,
        "width8_call_count": _subwarp_width8_call_count,
        "no_kernel_count": _subwarp_no_kernel_count,
        "forced_count": _subwarp_forced_count,
        "sparse_width1_count": _subwarp_sparse_width1_count,
        "short_width4_count": _subwarp_short_width4_count,
        "short_width2_count": _subwarp_short_width2_count,
        "long_width4_count": _subwarp_long_width4_count,
        "long_width2_count": _subwarp_long_width2_count,
        "long_saturated_width1_count": (
            _subwarp_long_saturated_width1_count
        ),
        "divergent_width1_count": _subwarp_divergent_width1_count,
        "kernel_launch_count": _subwarp_kernel_launch_count,
        "scheduled_warp_count": _subwarp_scheduled_warp_count,
        "candidate_count": _subwarp_candidate_count,
        "active_lane_slots": _subwarp_active_lane_slots,
        "issued_lane_slots": _subwarp_issued_lane_slots,
    }


cdef void _accumulate_forward_backward_residency_statistics(
    const plan7_forward_residency_statistics *forward,
    const plan7_backward_domain_residency_statistics *backward,
    const plan7_domain_rescore_residency_statistics *rescore,
) noexcept:
    global _resident_forward_call_count
    global _resident_forward_requested_bytes
    global _resident_forward_allocated_bytes
    global _resident_forward_materialized_bytes
    global _resident_forward_allocation_fallback_count
    global _resident_forward_allocation_milliseconds
    global _resident_forward_materialization_milliseconds
    global _resident_backward_call_count
    global _resident_backward_forward_h2d_bytes
    global _resident_backward_eliminated_forward_h2d_bytes
    global _resident_backward_forward_upload_milliseconds
    global _resident_backward_region_requested_bytes
    global _resident_backward_region_allocated_bytes
    global _resident_backward_region_materialized_bytes
    global _resident_backward_region_allocation_fallback_count
    global _resident_backward_region_allocation_milliseconds
    global _resident_backward_region_materialization_milliseconds
    global _resident_rescore_call_count
    global _resident_rescore_upstream_h2d_bytes
    global _resident_rescore_eliminated_upstream_h2d_bytes
    global _resident_rescore_selection_h2d_bytes
    global _resident_rescore_upstream_upload_milliseconds
    global _resident_rescore_prepare_milliseconds
    if forward != NULL:
        _resident_forward_call_count += 1
        _resident_forward_requested_bytes += forward.requested_bytes
        _resident_forward_allocated_bytes += forward.allocated_bytes
        _resident_forward_materialized_bytes += forward.materialized_bytes
        _resident_forward_allocation_fallback_count += (
            forward.allocation_fallback_count
        )
        _resident_forward_allocation_milliseconds += (
            forward.allocation_milliseconds
        )
        _resident_forward_materialization_milliseconds += (
            forward.materialization_milliseconds
        )
    if backward != NULL:
        _resident_backward_call_count += backward.resident_input_count
        _resident_backward_forward_h2d_bytes += (
            backward.forward_special_h2d_bytes
        )
        _resident_backward_eliminated_forward_h2d_bytes += (
            backward.eliminated_forward_special_h2d_bytes
        )
        _resident_backward_forward_upload_milliseconds += (
            backward.forward_special_upload_milliseconds
        )
        _resident_backward_region_requested_bytes += (
            backward.resident_region_requested_bytes
        )
        _resident_backward_region_allocated_bytes += (
            backward.resident_region_allocated_bytes
        )
        _resident_backward_region_materialized_bytes += (
            backward.resident_region_materialized_bytes
        )
        _resident_backward_region_allocation_fallback_count += (
            backward.resident_region_allocation_fallback_count
        )
        _resident_backward_region_allocation_milliseconds += (
            backward.resident_region_allocation_milliseconds
        )
        _resident_backward_region_materialization_milliseconds += (
            backward.resident_region_materialization_milliseconds
        )
    if rescore != NULL:
        _resident_rescore_call_count += rescore.resident_input_count
        _resident_rescore_upstream_h2d_bytes += rescore.upstream_h2d_bytes
        _resident_rescore_eliminated_upstream_h2d_bytes += (
            rescore.eliminated_upstream_h2d_bytes
        )
        _resident_rescore_selection_h2d_bytes += (
            rescore.resident_selection_h2d_bytes
        )
        _resident_rescore_upstream_upload_milliseconds += (
            rescore.upstream_upload_milliseconds
        )
        _resident_rescore_prepare_milliseconds += (
            rescore.resident_prepare_milliseconds
        )


def bias_host_environment_attested():
    cdef int attested
    with nogil:
        attested = plan7_bias_host_environment_attested()
    return bool(attested)


def pack_bias_profile_raw(
    const float[::1] background,
    const float[::1] composition,
    int model_length,
    float scale,
    int cutoff_mode,
    float cutoff_bit_score,
):
    cdef char error[512]
    cdef int status
    cdef plan7_bias_profile profile
    cdef bytearray output
    cdef uint8_t[::1] output_view
    if background.shape[0] != 20 or composition.shape[0] != 20:
        raise ValueError("bias background and composition must contain 20 values")
    error[0] = 0
    with nogil:
        status = plan7_bias_pack_amino_profile(
            &background[0],
            &composition[0],
            model_length,
            scale,
            cutoff_mode,
            cutoff_bit_score,
            &profile,
            error,
            sizeof(error),
        )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    output = bytearray(sizeof(plan7_bias_profile))
    output_view = output
    memcpy(&output_view[0], &profile, sizeof(plan7_bias_profile))
    return bytes(output)


def bias_filter_score_host_raw(
    const uint8_t[::1] packed_profile,
    const uint8_t[::1] residues,
):
    cdef plan7_bias_profile profile
    cdef float score
    cdef float_bits score_bits
    cdef int status
    if packed_profile.shape[0] != sizeof(plan7_bias_profile):
        raise ValueError("packed bias profile has the wrong size")
    if residues.shape[0] == 0:
        raise ValueError("bias filter does not score empty targets")
    memcpy(&profile, &packed_profile[0], sizeof(plan7_bias_profile))
    with nogil:
        status = plan7_bias_filter_score_host(
            &profile,
            &residues[0],
            <uint64_t> residues.shape[0],
            &score,
        )
    if status != 0:
        raise ValueError("invalid bias profile or target")
    score_bits.value = score
    return score_bits.bits


def bias_rebias_decision_raw(
    int ssv_status,
    int ssv_numerator,
    float scale,
    float filtersc,
    int cutoff_mode,
    float cutoff_bit_score,
):
    cdef float bit_score
    cdef float_bits encoded
    cdef int action
    if not 0 <= ssv_status <= 255:
        raise ValueError("SSV status must fit in uint8")
    if not -32768 <= ssv_numerator <= 32767:
        raise ValueError("SSV numerator must fit in int16")
    action = plan7_bias_rebias_decision(
        <uint8_t> ssv_status,
        <int16_t> ssv_numerator,
        scale,
        filtersc,
        cutoff_mode,
        cutoff_bit_score,
        &bit_score,
    )
    if isfinite(bit_score):
        encoded.value = bit_score
        return action, encoded.bits
    return action, None


cdef list _format_results(vector[plan7_ssv_result]& results, float scale):
    cdef size_t i
    cdef plan7_ssv_result result
    cdef float_bits score
    cdef object score_value
    cdef list output = []

    for i in range(results.size()):
        result = results[i]
        if result.status == PLAN7_SSV_OK:
            score.value = <float> result.numerator
            score.value /= scale
            score.value -= 3.0
            score_value = score.bits
        else:
            score_value = None
        output.append(
            (result.status, result.xE, result.tjb, result.numerator, score_value)
        )
    return output


cdef list _format_many_results(
    vector[plan7_ssv_result]& results,
    const float[::1] scales,
    size_t sequence_count,
):
    cdef size_t profile_index
    cdef size_t sequence_index
    cdef size_t result_index
    cdef plan7_ssv_result result
    cdef float_bits score
    cdef object score_value
    cdef list profile_results
    cdef list output = []

    for profile_index in range(<size_t> scales.shape[0]):
        profile_results = []
        for sequence_index in range(sequence_count):
            result_index = profile_index * sequence_count + sequence_index
            result = results[result_index]
            if result.status == PLAN7_SSV_OK:
                score.value = <float> result.numerator
                score.value /= scales[profile_index]
                score.value -= 3.0
                score_value = score.bits
            else:
                score_value = None
            profile_results.append(
                (result.status, result.xE, result.tjb, result.numerator, score_value)
            )
        output.append(profile_results)
    return output


def device_count():
    cdef char error[512]
    cdef int count
    error[0] = 0
    with nogil:
        count = plan7_cuda_device_count(error, sizeof(error))
    if count < 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    return count


def device_memory_info():
    cdef char error[512]
    cdef int device_ordinal = -1
    cdef uint64_t free_bytes = 0
    cdef uint64_t total_bytes = 0
    cdef int status
    error[0] = 0
    with nogil:
        status = plan7_cuda_memory_info(
            &device_ordinal, &free_bytes, &total_bytes, error, sizeof(error)
        )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    return {
        "device_ordinal": device_ordinal,
        "cuda_free_bytes": free_bytes,
        "cuda_total_bytes": total_bytes,
    }


def _validate_device_ordinal(int owner_device, int current_device):
    """Exercise the pure owner/current-device validation seam."""
    cdef char error[512]
    cdef int status
    error[0] = 0
    status = plan7_validate_device_ordinal(
        owner_device, current_device, error, sizeof(error)
    )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))


def profile_footprint(model_lengths):
    """Return amino profile byte bounds without packing or allocating."""
    cdef object lengths_array
    cdef const uint32_t[::1] lengths
    cdef plan7_profile_footprint footprint
    cdef char error[512]
    cdef int status
    try:
        lengths_array = _array.array("I", model_lengths)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model lengths must be unsigned 32-bit integers") from exc
    lengths = lengths_array
    error[0] = 0
    with nogil:
        status = plan7_profile_footprint_compute(
            &lengths[0] if lengths.shape[0] else NULL,
            <size_t> lengths.shape[0],
            &footprint,
            error,
            sizeof(error),
        )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    return {
        "profile_count": footprint.profile_count,
        "ssv_device_bytes": footprint.ssv_device_bytes,
        "viterbi_device_bytes": footprint.viterbi_device_bytes,
        "viterbi_exact_rbv_upper_bytes": (
            footprint.viterbi_exact_rbv_upper_bytes
        ),
        "forward_device_bytes": footprint.forward_device_bytes,
        "bias_device_bytes": footprint.bias_device_bytes,
        "minimum_device_bytes": footprint.minimum_device_bytes,
        "maximum_device_bytes": footprint.maximum_device_bytes,
    }


def profile_slice_cell_count(
    uint64_t profile_count,
    uint64_t target_count,
    uint64_t cell_limit=100_000_000,
):
    """Validate and return the profile-major planner cell count."""
    cdef uint64_t cell_count = 0
    cdef char error[512]
    cdef int status
    error[0] = 0
    status = plan7_profile_slice_cell_count(
        profile_count,
        target_count,
        cell_limit,
        &cell_count,
        error,
        sizeof(error),
    )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    return cell_count


def simulate_allocate_before_free(
    current_capacities,
    required_capacities,
    uint64_t free_bytes,
):
    """Simulate ordered high-water growth by allocate-new, then free-old."""
    cdef object current_array
    cdef object required_array
    cdef carray final_array
    cdef const uint64_t[::1] current
    cdef const uint64_t[::1] required
    cdef uint64_t[::1] final
    cdef plan7_allocation_simulation simulation
    cdef char error[512]
    cdef int status
    try:
        current_array = _array.array("Q", current_capacities)
        required_array = _array.array("Q", required_capacities)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("capacities must be unsigned 64-bit integers") from exc
    current = current_array
    required = required_array
    if current.shape[0] != required.shape[0]:
        raise ValueError("current and required capacities differ in length")
    final_array = clone(
        _UINT64_ARRAY_TEMPLATE, <Py_ssize_t> current.shape[0], zero=False
    )
    final = final_array
    error[0] = 0
    with nogil:
        status = plan7_simulate_allocate_before_free(
            &current[0] if current.shape[0] else NULL,
            &required[0] if required.shape[0] else NULL,
            <size_t> current.shape[0],
            free_bytes,
            &final[0] if final.shape[0] else NULL,
            &simulation,
            error,
            sizeof(error),
        )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    return {
        "fits": bool(simulation.fits),
        "peak_additional_bytes": simulation.peak_additional_bytes,
        "final_additional_bytes": simulation.final_additional_bytes,
        "final_free_bytes": (
            simulation.final_free_bytes if simulation.fits else None
        ),
        "growth_count": simulation.growth_count,
        "first_unfit_index": (
            None if simulation.fits else simulation.first_unfit_index
        ),
        "capacities": tuple(final_array),
    }


def tjb_for_lengths(float scale, const uint64_t[::1] lengths):
    cdef bytearray output = bytearray(lengths.shape[0])
    cdef uint8_t[::1] view = output
    cdef size_t i
    cdef int value
    if not isfinite(scale) or scale <= 0:
        raise ValueError("profile scale must be finite and positive")
    for i in range(<size_t> lengths.shape[0]):
        if lengths[i] > 100_000:
            raise ValueError("sequence length exceeds HMMER's protein limit")
    with nogil:
        for i in range(<size_t> lengths.shape[0]):
            value = plan7_tjb_for_length(scale, lengths[i])
            view[i] = <uint8_t> value
    return output


def pack_striped_scores(
    list striped_score_buffers,
    const int32_t[::1] score_strides,
    const int32_t[::1] model_lengths,
    int alphabet_size,
):
    """Transpose striped HMMER scores into compact ``[k][residue]`` rows."""
    cdef size_t profile_count = <size_t> len(striped_score_buffers)
    cdef size_t profile_index
    cdef size_t model_length
    cdef size_t profile_score_count
    cdef size_t total_score_count = 0
    cdef size_t output_offset = 0
    cdef size_t k
    cdef int residue
    cdef int q_count
    cdef int column
    cdef int score_stride
    cdef const uint8_t[::1] striped_scores
    cdef bytearray packed_scores
    cdef uint8_t[::1] packed_view

    if alphabet_size < 1:
        raise ValueError("alphabet size must be positive")
    if (
        <size_t> score_strides.shape[0] != profile_count
        or <size_t> model_lengths.shape[0] != profile_count
    ):
        raise ValueError("profile score metadata lengths differ")

    for profile_index in range(profile_count):
        if not 1 <= model_lengths[profile_index] <= 100_000:
            raise ValueError("invalid model length")
        model_length = <size_t> model_lengths[profile_index]
        if model_length > (<size_t> -1) // <size_t> alphabet_size:
            raise OverflowError("compact profile score count overflows size_t")
        profile_score_count = model_length * <size_t> alphabet_size
        if profile_score_count > (<size_t> -1) - total_score_count:
            raise OverflowError("packed profile score count overflows size_t")
        total_score_count += profile_score_count

    packed_scores = bytearray(total_score_count)
    packed_view = packed_scores
    for profile_index in range(profile_count):
        striped_scores = striped_score_buffers[profile_index]
        score_stride = score_strides[profile_index]
        model_length = <size_t> model_lengths[profile_index]
        q_count = max(2, (model_lengths[profile_index] + 15) // 16)
        if score_stride < 16 * (q_count + 17):
            raise ValueError("striped score stride is too short")
        if (
            <size_t> score_stride > (<size_t> -1) // <size_t> alphabet_size
            or <size_t> striped_scores.shape[0]
            != <size_t> score_stride * <size_t> alphabet_size
        ):
            raise ValueError("striped score buffer has the wrong size")
        with nogil:
            for k in range(model_length):
                column = 16 * (<int> k % q_count) + <int> k // q_count
                for residue in range(alphabet_size):
                    packed_view[
                        output_offset + k * <size_t> alphabet_size + residue
                    ] = striped_scores[
                        <size_t> residue * <size_t> score_stride + column
                    ]
        output_offset += model_length * <size_t> alphabet_size
    return packed_scores


cdef class ForwardProvenance:
    """Opaque identity binding gathered Forward rows to resident inputs."""

    cdef plan7_forward_provenance _value

    def __cinit__(self):
        self._value.database_generation = 0
        self._value.batch_generation = 0
        self._value.row_hash = 0
        self._value.special_hash = 0
        self._value.continuation_hash = 0
        self._value.pass_count = 0
        self._value.special_count = 0
        self._value.generation_f3_bits = 0
        self._value.integrity_tag = 0

    @property
    def pass_count(self):
        return self._value.pass_count

    @property
    def special_count(self):
        return self._value.special_count

    def __reduce__(self):
        raise TypeError("Forward provenance tokens cannot be serialized")

    def _tampered_for_test(self, field):
        """Return a corrupt copy for native provenance rejection tests."""
        cdef ForwardProvenance token
        token = ForwardProvenance.__new__(ForwardProvenance)
        token._value = self._value
        if field == "continuation_hash":
            token._value.continuation_hash ^= 1
        elif field == "generation_f3_bits":
            token._value.generation_f3_bits ^= 1
        elif field == "integrity_tag":
            token._value.integrity_tag ^= 1
        else:
            raise ValueError("unknown provenance field")
        return token


cdef ForwardProvenance _forward_provenance_from_output(
    const plan7_forward_output *output,
):
    cdef const plan7_forward_provenance *native
    cdef ForwardProvenance token
    native = plan7_forward_output_provenance(output)
    if native == NULL:
        raise RuntimeError("Forward provenance is null")
    token = ForwardProvenance.__new__(ForwardProvenance)
    token._value = native[0]
    return token


cdef dict _forward_f3_device_statistics_from_output(
    const plan7_forward_output *output,
):
    cdef const plan7_forward_f3_device_statistics *native
    native = plan7_forward_output_f3_device_statistics(output)
    if native == NULL:
        raise RuntimeError("Forward device F3 statistics are null")
    return {
        "f3_compiled_profile_count": native.compiled_profile_count,
        "f3_unsupported_profile_count": native.unsupported_profile_count,
        "f3_host_audit_count": native.host_audit_count,
        "f3_host_decision_avoided_count": native.host_decision_avoided_count,
        "f3_device_decision_count": native.device_decision_count,
        "f3_device_reject_count": native.device_reject_count,
        "f3_device_pass_count": native.device_pass_count,
        "f3_host_fallback_count": native.host_fallback_count,
        "f3_decision_mismatch_count": native.decision_mismatch_count,
        "f3_device_compaction_run_count": native.device_compaction_run_count,
        "f3_device_compaction_candidate_count": (
            native.device_compaction_candidate_count
        ),
        "f3_device_compacted_survivor_count": (
            native.device_compacted_survivor_count
        ),
        "f3_survivor_upload_avoided_bytes": (
            native.survivor_upload_avoided_bytes
        ),
    }


cdef class BackwardDomainProvenance:
    """Opaque seal for one compact Backward/domain continuation journal."""

    cdef plan7_backward_domain_provenance _value

    def __cinit__(self):
        self._value.candidate_count = 0
        self._value.region_count = 0

    @property
    def candidate_count(self):
        return self._value.candidate_count

    @property
    def region_count(self):
        return self._value.region_count

    def __reduce__(self):
        raise TypeError("Backward/domain provenance tokens cannot be serialized")


cdef BackwardDomainProvenance _backward_provenance_from_output(
    const plan7_backward_domain_output *output,
):
    cdef const plan7_backward_domain_provenance *native
    cdef BackwardDomainProvenance token
    native = plan7_backward_domain_output_provenance(output)
    if native == NULL:
        raise RuntimeError("Backward/domain provenance is null")
    token = BackwardDomainProvenance.__new__(BackwardDomainProvenance)
    token._value = native[0]
    return token


cdef object _backward_domain_route_payload_from_output(
    const plan7_backward_domain_output *output,
):
    """Copy the compact route/region journal for diagnostic audits only."""
    cdef size_t result_count
    cdef size_t result_bytes
    cdef size_t offset_bytes
    cdef size_t region_count
    cdef size_t region_bytes
    cdef bytes result_storage
    cdef bytes offset_storage
    cdef bytes region_storage
    cdef const plan7_backward_domain_result *results
    cdef const uint64_t *offsets
    cdef const plan7_simple_region *regions

    if output == NULL:
        raise RuntimeError("Backward/domain route output is null")
    result_count = plan7_backward_domain_output_result_count(output)
    region_count = plan7_backward_domain_output_region_count(output)
    if result_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_backward_domain_result)
    ):
        raise OverflowError("Backward/domain route results exceed Python limits")
    if result_count > (<size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)) - 1:
        raise OverflowError("Backward/domain route offsets exceed Python limits")
    if region_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_simple_region)
    ):
        raise OverflowError("Backward/domain route regions exceed Python limits")
    result_bytes = result_count * sizeof(plan7_backward_domain_result)
    offset_bytes = (result_count + 1) * sizeof(uint64_t)
    region_bytes = region_count * sizeof(plan7_simple_region)
    results = plan7_backward_domain_output_results(output)
    offsets = plan7_backward_domain_output_region_offsets(output)
    regions = plan7_backward_domain_output_regions(output)
    if (
        offsets == NULL
        or (result_count and results == NULL)
        or (region_count and regions == NULL)
    ):
        raise RuntimeError("Backward/domain route storage is incomplete")
    result_storage = PyBytes_FromStringAndSize(NULL, result_bytes)
    if result_bytes:
        memcpy(PyBytes_AS_STRING(result_storage), results, result_bytes)
    offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
    memcpy(PyBytes_AS_STRING(offset_storage), offsets, offset_bytes)
    region_storage = PyBytes_FromStringAndSize(NULL, region_bytes)
    if region_bytes:
        memcpy(PyBytes_AS_STRING(region_storage), regions, region_bytes)
    return (
        result_storage,
        memoryview(offset_storage).cast("Q"),
        memoryview(region_storage).cast("I"),
    )


cdef object _domain_rescore_payload_from_output(
    const plan7_domain_rescore_output *output,
):
    cdef size_t result_count
    cdef size_t result_bytes
    cdef size_t offset_bytes
    cdef size_t trace_count
    cdef size_t trace_bytes
    cdef size_t null2_count
    cdef size_t null2_bytes
    cdef bytes result_storage
    cdef bytes offset_storage
    cdef bytes trace_storage
    cdef bytes null2_storage
    cdef const plan7_domain_rescore_result *results
    cdef const uint64_t *offsets
    cdef const plan7_domain_rescore_trace_step *traces
    cdef const float *null2
    cdef const plan7_domain_rescore_statistics *statistics
    cdef const plan7_domain_rescore_provenance *provenance

    if output == NULL:
        raise RuntimeError("isolated-domain output is null")
    result_count = plan7_domain_rescore_output_result_count(output)
    trace_count = plan7_domain_rescore_output_trace_count(output)
    null2_count = plan7_domain_rescore_output_null2_count(output)
    if result_count > (<size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_rescore_result)):
        raise OverflowError("isolated-domain results exceed Python limits")
    if trace_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_rescore_trace_step)
    ):
        raise OverflowError("isolated-domain traces exceed Python limits")
    if null2_count > (<size_t> PY_SSIZE_T_MAX // sizeof(float)):
        raise OverflowError("isolated-domain null2 output exceeds Python limits")
    if result_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
    ) - 1:
        raise OverflowError("isolated-domain offsets exceed Python limits")
    result_bytes = result_count * sizeof(plan7_domain_rescore_result)
    offset_bytes = (result_count + 1) * sizeof(uint64_t)
    trace_bytes = trace_count * sizeof(plan7_domain_rescore_trace_step)
    null2_bytes = null2_count * sizeof(float)
    results = plan7_domain_rescore_output_results(output)
    offsets = plan7_domain_rescore_output_trace_offsets(output)
    traces = plan7_domain_rescore_output_traces(output)
    null2 = plan7_domain_rescore_output_null2(output)
    statistics = plan7_domain_rescore_output_statistics(output)
    provenance = plan7_domain_rescore_output_provenance(output)
    if (
        offsets == NULL
        or statistics == NULL
        or provenance == NULL
        or (result_count and results == NULL)
        or (trace_count and traces == NULL)
        or (null2_count and null2 == NULL)
    ):
        raise RuntimeError("isolated-domain output storage is incomplete")
    result_storage = PyBytes_FromStringAndSize(NULL, result_bytes)
    if result_bytes:
        memcpy(PyBytes_AS_STRING(result_storage), results, result_bytes)
    offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
    memcpy(PyBytes_AS_STRING(offset_storage), offsets, offset_bytes)
    trace_storage = PyBytes_FromStringAndSize(NULL, trace_bytes)
    if trace_bytes:
        memcpy(PyBytes_AS_STRING(trace_storage), traces, trace_bytes)
    null2_storage = PyBytes_FromStringAndSize(NULL, null2_bytes)
    if null2_bytes:
        memcpy(PyBytes_AS_STRING(null2_storage), null2, null2_bytes)
    return (
        result_storage,
        memoryview(offset_storage).cast("Q"),
        trace_storage,
        memoryview(null2_storage).cast("f"),
        {
            "upstream_row_count": statistics.upstream_row_count,
            "simple_row_count": statistics.simple_row_count,
            "region_count": statistics.region_count,
            "device_result_count": statistics.device_result_count,
            "cpu_required_count": statistics.cpu_required_count,
            "numeric_fallback_count": statistics.numeric_fallback_count,
            "cap_fallback_count": statistics.cap_fallback_count,
            "global_cpu_fallback_count": (
                statistics.global_cpu_fallback_count
            ),
            "work_cells": statistics.work_cells,
            "forward_matrix_bytes": statistics.forward_matrix_bytes,
            "posterior_matrix_bytes": statistics.posterior_matrix_bytes,
            "special_workspace_bytes": statistics.special_workspace_bytes,
            "trace_workspace_bytes": statistics.trace_workspace_bytes,
            "compact_output_byte_limit": (
                statistics.compact_output_byte_limit
            ),
            "compact_output_bytes": statistics.compact_output_bytes,
            "certified_ga_row_count": statistics.certified_ga_row_count,
            "certified_ga_region_count": (
                statistics.certified_ga_region_count
            ),
            "certified_ga_skipped_work_cells": (
                statistics.certified_ga_skipped_work_cells
            ),
            "ga_classification_ms": (
                statistics.ga_classification_milliseconds
            ),
            "kernel_ms": statistics.kernel_milliseconds,
            "upload_ms": statistics.upload_milliseconds,
            "download_ms": statistics.download_milliseconds,
            "total_ms": statistics.total_milliseconds,
            "result_hash": provenance.result_hash,
            "trace_hash": provenance.trace_hash,
            "null2_hash": provenance.null2_hash,
            "result_count": provenance.result_count,
            "trace_count": provenance.trace_count,
            "null2_count": provenance.null2_count,
        },
    )


def backward_domain_cpu_oracle_raw(
    OptimizedProfile profile,
    const uint8_t[::1] residues,
    const float[::1] forward_specials,
    float rt1=0.25,
    float rt2=0.10,
    float rt3=0.20,
    float guard_band=2.0e-4,
):
    """Run pristine HMMER BackwardParser + DomainDecoding for one row."""
    cdef plan7_backward_domain_result result
    cdef size_t posterior_count
    cdef size_t posterior_bytes
    cdef bytes result_storage
    cdef bytes posterior_storage
    cdef plan7_domain_posterior *posterior_pointer
    cdef size_t region_capacity
    cdef size_t region_count = 0
    cdef size_t region_bytes
    cdef bytes region_storage
    cdef plan7_simple_region *region_pointer
    cdef char error[512]
    cdef int status

    if residues.shape[0] == 0:
        raise ValueError("Backward/domain CPU oracle requires a nonempty target")
    if residues.shape[0] > (<size_t> -1) - 1:
        raise OverflowError("Backward/domain posterior length overflows size_t")
    posterior_count = <size_t> residues.shape[0] + 1
    if posterior_count > (<size_t> -1) // sizeof(plan7_domain_posterior):
        raise OverflowError("Backward/domain posterior size overflows size_t")
    posterior_bytes = posterior_count * sizeof(plan7_domain_posterior)
    if posterior_bytes > <size_t> PY_SSIZE_T_MAX:
        raise OverflowError("Backward/domain posterior exceeds Python limits")
    posterior_storage = PyBytes_FromStringAndSize(NULL, posterior_bytes)
    posterior_pointer = <plan7_domain_posterior *> PyBytes_AS_STRING(
        posterior_storage
    )
    region_capacity = <size_t> residues.shape[0]
    if region_capacity > (<size_t> PY_SSIZE_T_MAX // sizeof(plan7_simple_region)):
        raise OverflowError("Backward/domain region output exceeds Python limits")
    region_bytes = region_capacity * sizeof(plan7_simple_region)
    region_storage = PyBytes_FromStringAndSize(NULL, region_bytes)
    region_pointer = <plan7_simple_region *> PyBytes_AS_STRING(region_storage)
    error[0] = 0
    with nogil:
        status = plan7_backward_domain_cpu_oracle(
            <uintptr_t> profile._om,
            &residues[0],
            <size_t> residues.shape[0],
            &forward_specials[0] if forward_specials.shape[0] else NULL,
            <size_t> forward_specials.shape[0],
            rt1,
            rt2,
            rt3,
            guard_band,
            &result,
            posterior_pointer,
            posterior_count,
            region_pointer if region_capacity else NULL,
            region_capacity,
            &region_count,
            error,
            sizeof(error),
        )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    result_storage = PyBytes_FromStringAndSize(
        <const char *> &result, sizeof(plan7_backward_domain_result)
    )
    if region_count > region_capacity:
        raise RuntimeError("Backward/domain oracle region count exceeds capacity")
    region_storage = region_storage[: region_count * sizeof(plan7_simple_region)]
    return (
        result_storage,
        memoryview(posterior_storage).cast("f"),
        memoryview(region_storage).cast("I"),
    )


def domain_rescore_cpu_oracle_raw(
    OptimizedProfile profile,
    const uint8_t[::1] residues,
    uint32_t envelope_begin,
    uint32_t envelope_end,
):
    """Run pristine HMMER's complete isolated-envelope rescore path."""
    cdef plan7_domain_rescore_result result
    cdef size_t trace_capacity
    cdef size_t trace_count = 0
    cdef size_t trace_bytes
    cdef bytes result_storage
    cdef bytes null2_storage
    cdef bytes trace_storage
    cdef float *null2_pointer
    cdef plan7_domain_rescore_trace_step *trace_pointer
    cdef char error[512]
    cdef int status

    if residues.shape[0] == 0:
        raise ValueError("isolated-domain CPU oracle requires a nonempty target")
    if envelope_begin == 0 or envelope_begin > envelope_end:
        raise ValueError("isolated-domain envelope is invalid")
    if envelope_end > <uint32_t> residues.shape[0]:
        raise ValueError("isolated-domain envelope exceeds the target")
    if sizeof(plan7_domain_rescore_result) != PLAN7_DOMAIN_RESCORE_RECORD_SIZE:
        raise RuntimeError("isolated-domain result ABI size mismatch")
    if sizeof(plan7_domain_rescore_trace_step) != (
        PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE
    ):
        raise RuntimeError("isolated-domain trace ABI size mismatch")
    trace_capacity = (
        <size_t> (envelope_end - envelope_begin + 1)
        + <size_t> profile.M
        + 16
    )
    if trace_capacity > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_rescore_trace_step)
    ):
        raise OverflowError("isolated-domain trace exceeds Python limits")
    trace_bytes = trace_capacity * sizeof(plan7_domain_rescore_trace_step)
    trace_storage = PyBytes_FromStringAndSize(NULL, trace_bytes)
    trace_pointer = <plan7_domain_rescore_trace_step *> PyBytes_AS_STRING(
        trace_storage
    )
    null2_storage = PyBytes_FromStringAndSize(
        NULL, PLAN7_DOMAIN_RESCORE_NULL2_COUNT * sizeof(float)
    )
    null2_pointer = <float *> PyBytes_AS_STRING(null2_storage)
    error[0] = 0
    with nogil:
        status = plan7_domain_rescore_cpu_oracle(
            <uintptr_t> profile._om,
            &residues[0],
            <size_t> residues.shape[0],
            envelope_begin,
            envelope_end,
            &result,
            null2_pointer,
            PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
            trace_pointer,
            trace_capacity,
            &trace_count,
            error,
            sizeof(error),
        )
    if status != 0:
        message = error.decode("utf-8", "replace")
        raise RuntimeError(message or f"isolated-domain oracle status {status}")
    if trace_count > trace_capacity:
        raise RuntimeError("isolated-domain oracle trace count changed")
    result_storage = PyBytes_FromStringAndSize(
        <const char *> &result, sizeof(plan7_domain_rescore_result)
    )
    trace_storage = trace_storage[
        : trace_count * sizeof(plan7_domain_rescore_trace_step)
    ]
    return (
        result_storage,
        memoryview(null2_storage).cast("f"),
        trace_storage,
    )


def domain_rescore_own_scale_required_for_test(float xB):
    """Exercise the exact stock binary32/double scaling boundary."""
    return bool(c_plan7_domain_rescore_own_scale_required_for_test(xB))


def domain_rescore_oatrace_j_predecessor_for_test(
    float jpath,
    float epath,
    bint j_loop_enabled,
    bint e_loop_enabled,
):
    """Return whether stock OATrace's strict comparison chooses J."""
    return bool(c_plan7_domain_rescore_oatrace_j_predecessor_for_test(
        jpath,
        epath,
        j_loop_enabled,
        e_loop_enabled,
    ))


def backward_domain_merge_reason_facts_for_test(
    const uint64_t[::1] active_sources,
    const uint32_t[::1] active_facts,
    const uint32_t[::1] source_facts,
):
    """Exercise production's compact Backward-row reason remapping."""
    cdef carray output
    cdef int status
    if active_sources.shape[0] != active_facts.shape[0]:
        raise ValueError("Backward active reason rows differ")
    output = clone(_UINT32_ARRAY_TEMPLATE, source_facts.shape[0], False)
    if source_facts.shape[0]:
        memcpy(
            output.data.as_uints,
            &source_facts[0],
            source_facts.shape[0] * sizeof(uint32_t),
        )
    status = c_plan7_backward_domain_merge_reason_facts_for_test(
        &active_sources[0] if active_sources.shape[0] else NULL,
        &active_facts[0] if active_facts.shape[0] else NULL,
        active_sources.shape[0],
        output.data.as_uints if source_facts.shape[0] else NULL,
        source_facts.shape[0],
    )
    if status != 0:
        raise ValueError("invalid Backward active reason source mapping")
    return output


def domain_rescore_merge_reason_facts_for_test(
    const uint32_t[::1] active_result_indices,
    const uint32_t[::1] active_facts,
    const uint32_t[::1] source_facts,
):
    """Exercise production's compact rescore-region reason remapping."""
    cdef carray output
    cdef int status
    if active_result_indices.shape[0] != active_facts.shape[0]:
        raise ValueError("rescore active reason rows differ")
    output = clone(_UINT32_ARRAY_TEMPLATE, source_facts.shape[0], False)
    if source_facts.shape[0]:
        memcpy(
            output.data.as_uints,
            &source_facts[0],
            source_facts.shape[0] * sizeof(uint32_t),
        )
    status = c_plan7_domain_rescore_merge_reason_facts_for_test(
        &active_result_indices[0] if active_result_indices.shape[0] else NULL,
        &active_facts[0] if active_facts.shape[0] else NULL,
        active_result_indices.shape[0],
        output.data.as_uints if source_facts.shape[0] else NULL,
        source_facts.shape[0],
    )
    if status != 0:
        raise ValueError("invalid rescore active reason source mapping")
    return output


cdef class ProfileSelection:
    """Immutable pointer-free host pack for an ordered profile slice.

    Operations and ``close`` must be serialized by the public adapter.
    """

    cdef plan7_profile_selection *_selection
    cdef object _owner
    cdef tuple _fingerprints

    def __cinit__(self):
        self._selection = NULL
        self._owner = None
        self._fingerprints = ()

    def __dealloc__(self):
        if self._selection != NULL:
            plan7_profile_selection_destroy(&self._selection, NULL, 0)

    cdef plan7_profile_selection_view _view(self) except *:
        cdef plan7_profile_selection_view view
        cdef char error[512]
        cdef int status
        if self._selection == NULL:
            raise RuntimeError("profile selection is closed")
        error[0] = 0
        status = plan7_profile_selection_get_view(
            self._selection, &view, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return view

    def __len__(self):
        cdef plan7_profile_selection_view view = self._view()
        return view.profile_count

    @property
    def closed(self):
        return self._selection == NULL

    @property
    def identity(self):
        cdef plan7_profile_selection_view view = self._view()
        return view.session_id, view.selection_id

    def _identity_tokens_for_seal(self):
        """Return an immutable adapter-only identity snapshot."""
        cdef plan7_profile_selection_view view = self._view()
        cdef size_t byte_count
        if sizeof(uintptr_t) != sizeof(uint64_t):
            raise RuntimeError("profile identity tokens require a 64-bit process")
        if view.profile_count and view.identity_tokens == NULL:
            raise RuntimeError("profile selection identity storage is null")
        if view.profile_count > (<size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)):
            raise OverflowError("profile identity snapshot exceeds Python limits")
        byte_count = view.profile_count * sizeof(uint64_t)
        return PyBytes_FromStringAndSize(
            <const char *> view.identity_tokens, <Py_ssize_t> byte_count
        )

    def _fingerprints_for_seal(self):
        """Return adapter-only ordered immutable profile fingerprints."""
        return b"".join(self._fingerprints)

    @property
    def host_bytes(self):
        cdef plan7_profile_selection_view view = self._view()
        return view.host_bytes

    def _snapshot_equal_for_test(self, other):
        """Compare every arithmetic snapshot byte except identity tokens."""
        cdef ProfileSelection right
        cdef char error[512]
        cdef int status
        if self._selection == NULL:
            raise RuntimeError("profile selection is closed")
        if not isinstance(other, ProfileSelection):
            raise TypeError("snapshot comparison requires ProfileSelection")
        right = other
        if right._selection == NULL:
            raise RuntimeError("profile selection is closed")
        error[0] = 0
        with nogil:
            status = plan7_profile_selection_snapshot_equal_for_test(
                self._selection,
                right._selection,
                error,
                sizeof(error),
            )
        if status < 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return bool(status)

    def close(self):
        cdef plan7_profile_selection *selection = NULL
        cdef char error[512]
        cdef int status = 0
        if self._selection != NULL:
            selection = self._selection
            self._selection = NULL
            self._owner = None
            error[0] = 0
            with nogil:
                status = plan7_profile_selection_destroy(
                    &selection, error, sizeof(error)
                )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))


cdef class ProfileSession:
    """Host-owned exact SSV, bias, Viterbi, and Forward snapshots.

    Session operations and ``close`` must be serialized by the public adapter.
    Construction does not create a CUDA context or allocate device memory.
    The private chunk-local mode keeps source owners alive and snapshots only
    the profiles in each ordered selection.
    """

    cdef plan7_profile_session *_session
    cdef tuple _fingerprints
    cdef tuple _owners

    def __cinit__(self):
        self._session = NULL
        self._fingerprints = ()
        self._owners = ()

    def __init__(
        self,
        profiles,
        const float[::1] background,
        size_t build_worker_count,
        size_t selection_worker_count,
        profile_fingerprints=None,
        bint chunk_local_pack=False,
    ):
        cdef tuple owners = tuple(profiles)
        cdef tuple fingerprints
        cdef vector[uintptr_t] pointers
        cdef OptimizedProfile profile
        cdef object value
        cdef char error[512]
        cdef int status

        if self._session != NULL:
            raise RuntimeError("profile session is already initialized")
        if background.shape[0] != 20:
            raise ValueError("profile session background must have 20 residues")
        if (
            build_worker_count > <size_t> len(owners)
            or selection_worker_count > <size_t> len(owners)
        ):
            raise ValueError("profile session worker count exceeds profile count")
        pointers.reserve(len(owners))
        for value in owners:
            if not isinstance(value, OptimizedProfile):
                raise TypeError("profile sessions require OptimizedProfile objects")
            profile = value
            pointers.push_back(<uintptr_t> profile._om)
        error[0] = 0
        with nogil:
            status = plan7_profile_session_create(
                pointers.data() if pointers.size() else NULL,
                pointers.size(),
                &background[0],
                <size_t> background.shape[0],
                build_worker_count,
                selection_worker_count,
                1 if chunk_local_pack else 0,
                &self._session,
                error,
                sizeof(error),
            )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        try:
            if chunk_local_pack:
                self._owners = owners
            if profile_fingerprints is None:
                fingerprints = tuple(
                    _profile_fingerprint(value) for value in owners
                )
            else:
                fingerprints = tuple(profile_fingerprints)
                if len(fingerprints) != len(owners):
                    raise ValueError(
                        "profile fingerprint count differs from profile count"
                    )
            if any(
                type(value) is not bytes
                or len(value)
                != PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
                for value in fingerprints
            ):
                raise RuntimeError(
                    "optimized-profile fingerprint generation failed"
                )
            self._fingerprints = fingerprints
        except:
            plan7_profile_session_destroy(&self._session, NULL, 0)
            raise

    def __dealloc__(self):
        if self._session != NULL:
            plan7_profile_session_destroy(&self._session, NULL, 0)

    def __len__(self):
        return self.statistics["profile_count"]

    @property
    def closed(self):
        return self._session == NULL

    @property
    def statistics(self):
        cdef plan7_profile_session_statistics statistics
        cdef char error[512]
        cdef int status
        if self._session == NULL:
            raise RuntimeError("profile session is closed")
        error[0] = 0
        status = plan7_profile_session_get_statistics(
            self._session, &statistics, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return {
            "session_id": statistics.session_id,
            "profile_count": statistics.profile_count,
            "worker_count": statistics.worker_count,
            "build_worker_count": statistics.build_worker_count,
            "selection_worker_count": statistics.selection_worker_count,
            "selection_count": statistics.selection_count,
            "parallel_run_count": statistics.parallel_run_count,
            "build_parallel_run_count": (
                statistics.build_parallel_run_count
            ),
            "selection_parallel_run_count": (
                statistics.selection_parallel_run_count
            ),
            "host_bytes": statistics.host_bytes,
            "ssv_score_bytes": statistics.ssv_score_bytes,
            "bias_profile_bytes": statistics.bias_profile_bytes,
            "viterbi_descriptor_bytes": statistics.viterbi_descriptor_bytes,
            "viterbi_emission_bytes": statistics.viterbi_emission_bytes,
            "viterbi_transition_bytes": statistics.viterbi_transition_bytes,
            "viterbi_exact_rbv_bytes": statistics.viterbi_exact_rbv_bytes,
            "forward_descriptor_bytes": statistics.forward_descriptor_bytes,
            "forward_emission_bytes": statistics.forward_emission_bytes,
            "forward_transition_bytes": statistics.forward_transition_bytes,
            "chunk_local_pack": bool(statistics.chunk_local_pack),
            "profile_pointer_bytes": statistics.profile_pointer_bytes,
            "identity_token_bytes": statistics.identity_token_bytes,
            "background_bytes": statistics.background_bytes,
        }

    def _fingerprints_for_seal(self):
        """Return adapter-only immutable database profile fingerprints."""
        if self._session == NULL:
            raise RuntimeError("profile session is closed")
        return b"".join(self._fingerprints)

    def select(self, values):
        cdef tuple requested = tuple(values)
        cdef vector[size_t] indices
        cdef object value
        cdef object indexed
        cdef list selected_fingerprints = []
        cdef size_t index
        cdef ProfileSelection selection
        cdef char error[512]
        cdef int status

        if self._session == NULL:
            raise RuntimeError("profile session is closed")
        indices.reserve(len(requested))
        for value in requested:
            if isinstance(value, bool):
                raise TypeError("profile selection index must not be bool")
            try:
                indexed = value.__index__()
            except (AttributeError, TypeError) as exc:
                raise TypeError("profile selection index must be an integer") from exc
            if indexed < 0:
                raise IndexError("profile selection index is out of range")
            indices.push_back(<size_t> indexed)
        selection = ProfileSelection.__new__(ProfileSelection)
        error[0] = 0
        with nogil:
            status = plan7_profile_session_select(
                self._session,
                indices.data() if indices.size() else NULL,
                indices.size(),
                &selection._selection,
                error,
                sizeof(error),
            )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        selection._owner = self
        for index in range(indices.size()):
            selected_fingerprints.append(self._fingerprints[indices[index]])
        selection._fingerprints = tuple(selected_fingerprints)
        return selection

    def close(self):
        cdef plan7_profile_session *session = NULL
        cdef char error[512]
        cdef int status = 0
        if self._session != NULL:
            session = self._session
            self._session = NULL
            error[0] = 0
            with nogil:
                status = plan7_profile_session_destroy(
                    &session, error, sizeof(error)
                )
            self._owners = ()
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))


cdef class ViterbiProfiles:
    """Device-resident exact Viterbi profiles for the raw post-filter API.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_viterbi_database *_database
    cdef tuple _owners

    def __cinit__(self):
        self._database = NULL
        self._owners = ()

    def __init__(self, profiles):
        cdef tuple owners = tuple(profiles)
        cdef vector[uintptr_t] pointers
        cdef OptimizedProfile profile
        cdef object value
        cdef char error[512]
        cdef int status

        if self._database != NULL:
            raise RuntimeError("Viterbi profiles are already initialized")
        pointers.reserve(len(owners))
        for value in owners:
            if not isinstance(value, OptimizedProfile):
                raise TypeError("Viterbi profiles must be OptimizedProfile objects")
            profile = value
            pointers.push_back(<uintptr_t> profile._om)
        error[0] = 0
        status = plan7_viterbi_database_create(
            pointers.data() if pointers.size() else NULL,
            pointers.size(),
            &self._database,
            error,
            sizeof(error),
        )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        self._owners = owners

    def __dealloc__(self):
        if self._database != NULL:
            plan7_viterbi_database_destroy(&self._database, NULL, 0)

    def __len__(self):
        if self._database == NULL:
            return 0
        return plan7_viterbi_database_profile_count(self._database)

    @property
    def closed(self):
        return self._database == NULL

    def __enter__(self):
        if self._database == NULL:
            raise RuntimeError("Viterbi profiles are closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        cdef plan7_viterbi_database *database = NULL
        cdef char error[512]
        cdef int status = 0
        if self._database != NULL:
            database = self._database
            self._database = NULL
            self._owners = ()
            error[0] = 0
            with nogil:
                status = plan7_viterbi_database_destroy(
                    &database, error, sizeof(error)
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))


cdef class ForwardProfiles:
    """Device-resident exact Forward snapshots for F3 and later stages.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_forward_database *_database
    cdef tuple _owners

    def __cinit__(self):
        self._database = NULL
        self._owners = ()

    def __init__(self, profiles):
        cdef tuple owners
        cdef vector[uintptr_t] pointers
        cdef OptimizedProfile profile
        cdef ProfileSelection selection
        cdef object value
        cdef char error[512]
        cdef int status

        if self._database != NULL:
            raise RuntimeError("Forward profiles are already initialized")
        if isinstance(profiles, ProfileSelection):
            selection = profiles
            if selection._selection == NULL:
                raise RuntimeError("profile selection is closed")
            error[0] = 0
            with nogil:
                status = plan7_profile_selection_stage_forward(
                    selection._selection,
                    &self._database,
                    error,
                    sizeof(error),
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            self._owners = (selection,)
            return
        owners = tuple(profiles)
        pointers.reserve(len(owners))
        for value in owners:
            if not isinstance(value, OptimizedProfile):
                raise TypeError("Forward profiles must be OptimizedProfile objects")
            profile = value
            pointers.push_back(<uintptr_t> profile._om)
        error[0] = 0
        # Keep the GIL while native code takes its exact private-array snapshot.
        status = plan7_forward_database_create(
            pointers.data() if pointers.size() else NULL,
            pointers.size(),
            &self._database,
            error,
            sizeof(error),
        )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        self._owners = owners

    def __dealloc__(self):
        if self._database != NULL:
            plan7_forward_database_destroy(&self._database, NULL, 0)

    def __len__(self):
        if self._database == NULL:
            return 0
        return plan7_forward_database_profile_count(self._database)

    @property
    def closed(self):
        return self._database == NULL

    @property
    def statistics(self):
        if self._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        return {
            "device_bytes": plan7_forward_database_device_bytes(self._database),
            "pack_ms": plan7_forward_database_pack_milliseconds(self._database),
            "upload_ms": plan7_forward_database_upload_milliseconds(self._database),
        }

    def __enter__(self):
        if self._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        cdef plan7_forward_database *database = NULL
        cdef char error[512]
        cdef int status = 0
        if self._database != NULL:
            database = self._database
            self._database = NULL
            self._owners = ()
            error[0] = 0
            with nogil:
                status = plan7_forward_database_destroy(
                    &database, error, sizeof(error)
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))


cdef class SequenceBatch:
    """Device-resident target batch for raw CUDA operations.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_ssv_sequence_batch *_batch
    cdef vector[plan7_ssv_result] _results
    cdef vector[plan7_ssv_result] _many_results
    cdef vector[plan7_ssv_profile] _profiles
    cdef vector[uint32_t] _candidate_words
    cdef vector[uint32_t] _candidate_indices
    cdef vector[size_t] _candidate_offsets
    cdef vector[size_t] _candidate_counts
    cdef vector[size_t] _bias_candidate_offsets
    cdef vector[plan7_bias_profile] _bias_profiles
    cdef vector[plan7_bias_result] _bias_results
    cdef vector[plan7_postfilter_result] _postfilter_results
    cdef vector[uint64_t] _lengths
    cdef size_t _sequence_count
    cdef int _alphabet_size
    cdef bytes _content_fingerprint
    cdef bint _generation_ledger_enabled
    cdef bint _cpu_domain_route
    cdef bint _cpu_rescore_route
    cdef int _cpu_forward_route_mode
    cdef bint _cpu_forward_route_auto
    cdef uint64_t _cpu_forward_min_cells
    cdef uint64_t _cpu_forward_max_cells
    cdef uint64_t _cpu_forward_min_length
    cdef uint64_t _cpu_forward_selected_count
    cdef uint64_t _cpu_forward_selected_cells
    cdef uint64_t _gpu_forward_selected_count
    cdef uint64_t _gpu_forward_selected_cells
    cdef uint64_t _ledger_fused_call_count
    cdef uint64_t _ledger_fused_total_ns
    cdef uint64_t _ledger_f1_native_ns
    cdef uint64_t _ledger_f1_candidate_mirror_ns
    cdef uint64_t _ledger_postfilter_host_prepare_ns
    cdef uint64_t _ledger_viterbi_stage_ns
    cdef uint64_t _ledger_postfilter_native_ns
    cdef uint64_t _ledger_postfilter_materialize_ns
    cdef uint64_t _ledger_f2_control_ns
    cdef uint64_t _ledger_forward_stage_ns
    cdef uint64_t _ledger_forward_native_ns
    cdef uint64_t _ledger_backward_native_ns
    cdef uint64_t _ledger_rescore_native_ns

    def __cinit__(
        self,
        const uint8_t[::1] residues,
        const uint64_t[::1] offsets,
        int alphabet_size,
        int _execution_policy=PLAN7_GPU_EXECUTION_POLICY_AUTO,
        object _forward_cpu_max_cells=None,
    ):
        cdef char error[512]
        cdef int status
        cdef size_t i
        cdef object forward_ownership
        cdef object forward_threshold
        cdef tuple forward_override_environments = (
            "PLAN7_GPU_FORWARD_OWNERSHIP",
            "PLAN7_GPU_FORWARD_CPU_MIN_CELLS",
            "PLAN7_GPU_FORWARD_CPU_MIN_LENGTH",
            "PLAN7_GPU_FORWARD_CPU_MAX_CELLS",
            "ASTRA_GPU_CONTINUATION_POOL",
        )

        self._batch = NULL
        self._sequence_count = 0
        self._alphabet_size = alphabet_size
        self._content_fingerprint = b""
        self._generation_ledger_enabled = (
            _os.environ.get("PLAN7_GPU_GENERATION_LEDGER") == "1"
        )
        self._cpu_domain_route = (
            _os.environ.get("PLAN7_GPU_DOMAIN_OWNERSHIP") == "cpu"
            or (
                "PLAN7_GPU_DOMAIN_OWNERSHIP" not in _os.environ
                and _execution_policy != PLAN7_GPU_EXECUTION_POLICY_SIMPLE
                and offsets.shape[0] > 65536
            )
        )
        self._cpu_rescore_route = (
            _os.environ.get("PLAN7_GPU_DOMAIN_OWNERSHIP") == "cpu_rescore"
        )
        self._cpu_forward_route_mode = 0
        self._cpu_forward_route_auto = False
        self._cpu_forward_min_cells = 0
        self._cpu_forward_max_cells = 0
        self._cpu_forward_min_length = 0
        self._cpu_forward_selected_count = 0
        self._cpu_forward_selected_cells = 0
        self._gpu_forward_selected_count = 0
        self._gpu_forward_selected_cells = 0
        forward_ownership = _os.environ.get("PLAN7_GPU_FORWARD_OWNERSHIP")
        if _forward_cpu_max_cells is not None:
            if any(
                name in _os.environ for name in forward_override_environments
            ):
                raise ValueError(
                    "request-local Forward ownership conflicts with an "
                    "environment override"
                )
            if type(_forward_cpu_max_cells) is not int:
                raise TypeError(
                    "_forward_cpu_max_cells must be an integer or None"
                )
            if (
                _forward_cpu_max_cells <= 0
                or _forward_cpu_max_cells > 0xffffffffffffffff
            ):
                raise ValueError(
                    "_forward_cpu_max_cells is outside positive uint64 range"
                )
            if (
                _execution_policy == PLAN7_GPU_EXECUTION_POLICY_SIMPLE
                or offsets.shape[0] <= 65537
                or not self._cpu_domain_route
            ):
                raise ValueError(
                    "request-local Forward ownership requires a large "
                    "non-simple CPU-domain target batch"
                )
            self._cpu_forward_route_mode = 4
            self._cpu_forward_route_auto = True
            self._cpu_forward_max_cells = <uint64_t> _forward_cpu_max_cells
            forward_ownership = "hybrid_cells_below"
        elif (
            forward_ownership is None
            and _os.environ.get("ASTRA_GPU_CONTINUATION_POOL") == "1"
            and _execution_policy != PLAN7_GPU_EXECUTION_POLICY_SIMPLE
            and offsets.shape[0] > 65536
            and self._cpu_domain_route
        ):
            self._cpu_forward_route_mode = 4
            self._cpu_forward_route_auto = True
            self._cpu_forward_max_cells = _auto_forward_cpu_max_cells()
            forward_ownership = "hybrid_cells_below"
        elif forward_ownership is None:
            forward_ownership = "gpu"
        if self._cpu_forward_route_auto:
            pass
        elif forward_ownership == "cpu":
            self._cpu_forward_route_mode = 1
        elif forward_ownership == "hybrid_cells":
            forward_threshold = _os.environ.get(
                "PLAN7_GPU_FORWARD_CPU_MIN_CELLS"
            )
            if forward_threshold is None:
                raise ValueError(
                    "hybrid_cells Forward ownership requires "
                    "PLAN7_GPU_FORWARD_CPU_MIN_CELLS"
                )
            self._cpu_forward_min_cells = int(forward_threshold)
            if self._cpu_forward_min_cells == 0:
                raise ValueError("Forward CPU cell threshold must be positive")
            self._cpu_forward_route_mode = 2
        elif forward_ownership == "hybrid_cells_below":
            forward_threshold = _os.environ.get(
                "PLAN7_GPU_FORWARD_CPU_MAX_CELLS"
            )
            if forward_threshold is None:
                raise ValueError(
                    "hybrid_cells_below Forward ownership requires "
                    "PLAN7_GPU_FORWARD_CPU_MAX_CELLS"
                )
            self._cpu_forward_max_cells = int(forward_threshold)
            if self._cpu_forward_max_cells == 0:
                raise ValueError("Forward CPU cell threshold must be positive")
            self._cpu_forward_route_mode = 4
        elif forward_ownership == "hybrid_length":
            forward_threshold = _os.environ.get(
                "PLAN7_GPU_FORWARD_CPU_MIN_LENGTH"
            )
            if forward_threshold is None:
                raise ValueError(
                    "hybrid_length Forward ownership requires "
                    "PLAN7_GPU_FORWARD_CPU_MIN_LENGTH"
                )
            self._cpu_forward_min_length = int(forward_threshold)
            if self._cpu_forward_min_length == 0:
                raise ValueError("Forward CPU length threshold must be positive")
            self._cpu_forward_route_mode = 3
        elif forward_ownership != "gpu":
            raise ValueError("invalid PLAN7_GPU_FORWARD_OWNERSHIP")
        self._ledger_fused_call_count = 0
        self._ledger_fused_total_ns = 0
        self._ledger_f1_native_ns = 0
        self._ledger_f1_candidate_mirror_ns = 0
        self._ledger_postfilter_host_prepare_ns = 0
        self._ledger_viterbi_stage_ns = 0
        self._ledger_postfilter_native_ns = 0
        self._ledger_postfilter_materialize_ns = 0
        self._ledger_f2_control_ns = 0
        self._ledger_forward_stage_ns = 0
        self._ledger_forward_native_ns = 0
        self._ledger_backward_native_ns = 0
        self._ledger_rescore_native_ns = 0
        if offsets.shape[0] == 0:
            raise ValueError("offsets must contain an initial zero")
        if alphabet_size < 1:
            raise ValueError("alphabet size must be positive")
        if _execution_policy not in (
            PLAN7_GPU_EXECUTION_POLICY_AUTO,
            PLAN7_GPU_EXECUTION_POLICY_SIMPLE,
            PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT,
        ):
            raise ValueError("invalid GPU execution policy")
        self._content_fingerprint = _sequence_content_fingerprint(
            alphabet_size, residues, offsets
        )
        if len(self._content_fingerprint) != 32:
            raise RuntimeError("sequence content fingerprint generation failed")
        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_create(
                &residues[0] if residues.shape[0] else NULL,
                <size_t> residues.shape[0],
                &offsets[0],
                <size_t> offsets.shape[0],
                alphabet_size,
                &self._batch,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        error[0] = 0
        status = plan7_ssv_sequence_batch_set_execution_policy(
            self._batch,
            _execution_policy,
            error,
            sizeof(error),
        )
        if status != 0:
            plan7_ssv_sequence_batch_destroy(&self._batch, NULL, 0)
            raise RuntimeError(error.decode("utf-8", "replace"))
        self._sequence_count = <size_t> offsets.shape[0] - 1
        self._results.resize(self._sequence_count)
        self._lengths.resize(self._sequence_count)
        for i in range(self._sequence_count):
            self._lengths[i] = offsets[i + 1] - offsets[i]

    def __dealloc__(self):
        if self._batch != NULL:
            plan7_ssv_sequence_batch_destroy(&self._batch, NULL, 0)

    def __len__(self):
        return self._sequence_count

    @property
    def closed(self):
        return self._batch == NULL

    def _generation_and_content_for_seal(self):
        """Return adapter-only native batch identity and residue digest."""
        cdef plan7_ssv_sequence_batch_view view
        cdef char error[512]
        cdef int status
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_view(
            self._batch, &view, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return view.generation_id, self._content_fingerprint

    @property
    def workspace_statistics(self):
        """Return post-filter and Forward cache capacities for this batch.

        These values exclude the batch's input, SSV, and bias allocations.
        """
        cdef plan7_ssv_workspace_statistics statistics
        cdef plan7_gpu_execution_policy_statistics policy_statistics
        cdef char error[512]
        cdef int status
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_workspace_statistics(
            self._batch, &statistics, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_execution_policy_statistics(
            self._batch, &policy_statistics, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return {
            "f1_device_compaction_run_count": (
                statistics.f1_device_compaction_run_count
            ),
            "f1_host_expansion_run_count": statistics.f1_host_expansion_run_count,
            "f1_candidate_upload_count": statistics.f1_candidate_upload_count,
            "f1_candidate_upload_avoided_count": (
                statistics.f1_candidate_upload_avoided_count
            ),
            "f1_profile_packed_run_count": (
                statistics.f1_profile_packed_run_count
            ),
            "f1_profile_packed_quartet_count": (
                statistics.f1_profile_packed_quartet_count
            ),
            "f1_profile_packed_profile_count": (
                statistics.f1_profile_packed_profile_count
            ),
            "f1_profile_scalar_profile_count": (
                statistics.f1_profile_scalar_profile_count
            ),
            "f1_profile_packed_score_bytes": (
                statistics.f1_profile_packed_score_bytes
            ),
            "f1_identity_padding_run_count": (
                statistics.f1_identity_padding_run_count
            ),
            "f1_identity_padding_quartet_count": (
                statistics.f1_identity_padding_quartet_count
            ),
            "f1_identity_padding_profile_count": (
                statistics.f1_identity_padding_profile_count
            ),
            "f1_length_class_run_count": statistics.f1_length_class_run_count,
            "f1_length_class_value_count": (
                statistics.f1_length_class_value_count
            ),
            "f1_length_compact_h2d_bytes": (
                statistics.f1_length_compact_h2d_bytes
            ),
            "f1_length_dense_h2d_bytes_avoided": (
                statistics.f1_length_dense_h2d_bytes_avoided
            ),
            "f1_length_dense_materialized_bytes": (
                statistics.f1_length_dense_materialized_bytes
            ),
            "f1_raw_xe_run_count": statistics.f1_raw_xe_run_count,
            "f1_raw_xe_logical_pair_count": (
                statistics.f1_raw_xe_logical_pair_count
            ),
            "f1_raw_xe_sidecar_bytes_written": (
                statistics.f1_raw_xe_sidecar_bytes_written
            ),
            "f1_raw_xe_candidate_gather_count": (
                statistics.f1_raw_xe_candidate_gather_count
            ),
            "f1_candidate_ssv_replay_count": (
                statistics.f1_candidate_ssv_replay_count
            ),
            "f1_candidate_ssv_replay_avoided_count": (
                statistics.f1_candidate_ssv_replay_avoided_count
            ),
            "f1_raw_xe_fallback_run_count": (
                statistics.f1_raw_xe_fallback_run_count
            ),
            "postfilter_device_bytes": statistics.postfilter_device_bytes,
            "postfilter_dp_capacity_bytes": (
                statistics.postfilter_dp_capacity_bytes
            ),
            "postfilter_growth_count": statistics.postfilter_growth_count,
            "postfilter_run_count": statistics.postfilter_run_count,
            "full_msv_compaction_run_count": (
                statistics.full_msv_compaction_run_count
            ),
            "full_msv_compaction_chunk_count": (
                statistics.full_msv_compaction_chunk_count
            ),
            "full_msv_compaction_source_count": (
                statistics.full_msv_compaction_source_count
            ),
            "full_msv_compaction_selected_count": (
                statistics.full_msv_compaction_selected_count
            ),
            "full_msv_legacy_run_count": statistics.full_msv_legacy_run_count,
            "full_msv_launch_candidate_count": (
                statistics.full_msv_launch_candidate_count
            ),
            "full_msv_launch_candidate_avoided_count": (
                statistics.full_msv_launch_candidate_avoided_count
            ),
            "full_msv_index_d2h_bytes": statistics.full_msv_index_d2h_bytes,
            "full_msv_packed_run_count": statistics.full_msv_packed_run_count,
            "full_msv_packed_group_count": (
                statistics.full_msv_packed_group_count
            ),
            "full_msv_packed_candidate_count": (
                statistics.full_msv_packed_candidate_count
            ),
            "full_msv_scalar_candidate_count": (
                statistics.full_msv_scalar_candidate_count
            ),
            "vit_length_cache_run_count": statistics.vit_length_cache_run_count,
            "vit_length_cache_entry_count": (
                statistics.vit_length_cache_entry_count
            ),
            "vit_length_cache_candidate_count": (
                statistics.vit_length_cache_candidate_count
            ),
            "vit_length_direct_candidate_count": (
                statistics.vit_length_direct_candidate_count
            ),
            "vit_length_cache_build_ns": statistics.vit_length_cache_build_ns,
            "vit_length_candidate_plan_ns": (
                statistics.vit_length_candidate_plan_ns
            ),
            "forward_device_bytes": statistics.forward_device_bytes,
            "forward_dp_capacity_bytes": statistics.forward_dp_capacity_bytes,
            "forward_xmx_capacity_bytes": statistics.forward_xmx_capacity_bytes,
            "forward_gather_capacity_bytes": (
                statistics.forward_gather_capacity_bytes
            ),
            "forward_growth_count": statistics.forward_growth_count,
            "forward_event_create_count": statistics.forward_event_create_count,
            "forward_run_count": statistics.forward_run_count,
            "execution_policy_version": policy_statistics.version,
            "execution_policy_mode": policy_statistics.mode,
            "execution_policy_target_count": (
                policy_statistics.target_count
            ),
            "execution_policy_length_class_count": (
                policy_statistics.length_class_count
            ),
            "execution_policy_f1_run_count": (
                policy_statistics.f1_run_count
            ),
            "execution_policy_forward_candidates_per_warp": (
                policy_statistics.forward_candidates_per_warp
            ),
            "generation_ledger": {
                "schema_version": 1,
                "enabled": bool(self._generation_ledger_enabled),
                "units": "nanoseconds",
                "fused_call_count": self._ledger_fused_call_count,
                "fused_total_ns": self._ledger_fused_total_ns,
                "f1_native_ns": self._ledger_f1_native_ns,
                "f1_candidate_mirror_ns": (
                    self._ledger_f1_candidate_mirror_ns
                ),
                "postfilter_host_prepare_ns": (
                    self._ledger_postfilter_host_prepare_ns
                ),
                "viterbi_stage_ns": self._ledger_viterbi_stage_ns,
                "postfilter_native_ns": self._ledger_postfilter_native_ns,
                "postfilter_materialize_ns": (
                    self._ledger_postfilter_materialize_ns
                ),
                "f2_control_ns": self._ledger_f2_control_ns,
                "forward_stage_ns": self._ledger_forward_stage_ns,
                "forward_native_ns": self._ledger_forward_native_ns,
                "backward_native_ns": self._ledger_backward_native_ns,
                "rescore_native_ns": self._ledger_rescore_native_ns,
            },
            "cpu_domain_ownership": bool(self._cpu_domain_route),
            "cpu_rescore_ownership": bool(self._cpu_rescore_route),
            "cpu_forward_ownership_mode": self._cpu_forward_route_mode,
            "cpu_forward_ownership_auto": bool(self._cpu_forward_route_auto),
            "cpu_forward_min_cells": self._cpu_forward_min_cells,
            "cpu_forward_max_cells": self._cpu_forward_max_cells,
            "cpu_forward_min_length": self._cpu_forward_min_length,
            "cpu_forward_selected_count": self._cpu_forward_selected_count,
            "cpu_forward_selected_cells": self._cpu_forward_selected_cells,
            "gpu_forward_selected_count": self._gpu_forward_selected_count,
            "gpu_forward_selected_cells": self._gpu_forward_selected_cells,
        }

    @property
    def memory_snapshot(self):
        """Return current CUDA availability and persistent device capacities."""
        cdef plan7_ssv_memory_snapshot snapshot
        cdef char error[512]
        cdef int status
        cdef size_t i
        cdef dict capacities = {}
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if len(_DEVICE_CAPACITY_NAMES) != PLAN7_SSV_DEVICE_CAPACITY_COUNT:
            raise RuntimeError("native device capacity ABI changed")
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_memory_snapshot(
            self._batch, &snapshot, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        for i in range(PLAN7_SSV_DEVICE_CAPACITY_COUNT):
            capacities[_DEVICE_CAPACITY_NAMES[i]] = (
                snapshot.device_capacity_bytes[i]
            )
        return {
            "device_ordinal": snapshot.device_ordinal,
            "cuda_free_bytes": snapshot.cuda_free_bytes,
            "cuda_total_bytes": snapshot.cuda_total_bytes,
            "persistent_device_bytes": snapshot.persistent_device_bytes,
            "capacity_bytes": capacities,
        }

    def __enter__(self):
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        cdef plan7_ssv_sequence_batch *batch = NULL
        cdef char ssv_error[512]
        cdef int ssv_status = 0
        if self._batch != NULL:
            batch = self._batch
            self._batch = NULL
            ssv_error[0] = 0
            with nogil:
                ssv_status = plan7_ssv_sequence_batch_destroy(
                    &batch, ssv_error, sizeof(ssv_error)
                )
        if ssv_status != 0:
            raise RuntimeError(ssv_error.decode("utf-8", "replace"))

    cdef int _run_filter(
        self,
        const uint8_t[::1] striped_scores,
        int score_stride,
        int model_length,
        int alphabet_size,
        int tbm,
        int tec,
        int base,
        int bias,
        float scale,
    ) except -1:
        cdef char error[512]
        cdef int status

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if score_stride < 1 or alphabet_size < 1 or not 1 <= model_length <= 100_000:
            raise ValueError("invalid score dimensions")
        if alphabet_size != self._alphabet_size:
            raise ValueError("profile and sequence alphabets differ")
        if <size_t> striped_scores.shape[0] < <size_t> score_stride * alphabet_size:
            raise ValueError("striped score buffer is too short")
        if not 0 <= tbm <= 255 or not 0 <= tec <= 255:
            raise ValueError("transition costs must fit in uint8")
        if not 0 <= base <= 255 or not 0 <= bias <= 255:
            raise ValueError("profile byte constants must fit in uint8")
        if not isfinite(scale) or scale <= 0:
            raise ValueError("profile scale must be finite and positive")

        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_filter(
                self._batch,
                &striped_scores[0] if striped_scores.shape[0] else NULL,
                <size_t> striped_scores.shape[0],
                score_stride,
                model_length,
                alphabet_size,
                <uint8_t> tbm,
                <uint8_t> tec,
                <uint8_t> base,
                <uint8_t> bias,
                scale,
                self._results.data() if self._sequence_count else NULL,
                self._sequence_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return 0

    def filter_raw(
        self,
        const uint8_t[::1] striped_scores,
        int score_stride,
        int model_length,
        int alphabet_size,
        int tbm,
        int tec,
        int base,
        int bias,
        float scale,
    ):
        self._run_filter(
            striped_scores,
            score_stride,
            model_length,
            alphabet_size,
            tbm,
            tec,
            base,
            bias,
            scale,
        )
        return _format_results(self._results, scale)

    def cpu_candidates_raw(
        self,
        const uint8_t[::1] striped_scores,
        int score_stride,
        int model_length,
        int alphabet_size,
        int tbm,
        int tec,
        int base,
        int bias,
        float scale,
        float m_mu,
        float m_lambda,
        double f1,
    ):
        cdef size_t i
        cdef int action
        cdef list output = []

        self._run_filter(
            striped_scores,
            score_stride,
            model_length,
            alphabet_size,
            tbm,
            tec,
            base,
            bias,
            scale,
        )
        for i in range(self._sequence_count):
            action = plan7_ssv_f1_decision(
                self._results[i].status,
                self._results[i].numerator,
                self._lengths[i],
                scale,
                m_mu,
                m_lambda,
                f1,
                NULL,
            )
            if action == PLAN7_F1_CPU_REQUIRED:
                output.append(i)
        return output

    cdef size_t _prepare_profiles(
        self,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
    ) except? 0:
        cdef size_t i
        cdef size_t profile_count = <size_t> score_offsets.shape[0]

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if (
            <size_t> score_counts.shape[0] != profile_count
            or <size_t> score_strides.shape[0] != profile_count
            or <size_t> model_lengths.shape[0] != profile_count
            or <size_t> scales.shape[0] != profile_count
            or <size_t> constants.shape[0] != profile_count * 4
        ):
            raise ValueError("profile metadata lengths differ")

        self._profiles.resize(profile_count)
        for i in range(profile_count):
            self._profiles[i].score_offset = score_offsets[i]
            self._profiles[i].score_count = score_counts[i]
            self._profiles[i].score_stride = score_strides[i]
            self._profiles[i].model_length = model_lengths[i]
            self._profiles[i].tbm = constants[4 * i]
            self._profiles[i].tec = constants[4 * i + 1]
            self._profiles[i].base = constants[4 * i + 2]
            self._profiles[i].bias = constants[4 * i + 3]
            self._profiles[i].scale = scales[i]
        return profile_count

    cdef size_t _run_filter_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
    ) except? 0:
        cdef char error[512]
        cdef size_t profile_count
        cdef size_t result_count
        cdef int status

        profile_count = self._prepare_profiles(
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        if self._sequence_count and profile_count > (<size_t> -1) / self._sequence_count:
            raise OverflowError("multi-profile result count overflows size_t")
        result_count = profile_count * self._sequence_count
        self._many_results.resize(result_count)

        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_filter_many(
                self._batch,
                &packed_scores[0] if packed_scores.shape[0] else NULL,
                <size_t> packed_scores.shape[0],
                self._profiles.data() if profile_count else NULL,
                profile_count,
                self._many_results.data() if result_count else NULL,
                result_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return profile_count

    def filter_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
    ):
        self._run_filter_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        return _format_many_results(self._many_results, scales, self._sequence_count)

    cdef size_t _run_candidates_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        bint host_candidate_expansion,
    ) except? 0:
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t words_per_profile
        cdef size_t candidate_word_count
        cdef size_t word_index
        cdef size_t sequence_index
        cdef size_t candidate_count = 0
        cdef size_t output_index
        cdef uint32_t word
        cdef unsigned bit
        cdef int status
        cdef char error[512]
        cdef plan7_ssv_f1_candidate_view compact_view
        cdef plan7_bias_candidate mapping
        cdef uint64_t ledger_start_ns = 0

        profile_count = <size_t> score_offsets.shape[0]
        if (
            <size_t> m_mu.shape[0] != profile_count
            or <size_t> m_lambda.shape[0] != profile_count
        ):
            raise ValueError("e-value parameter lengths differ")
        profile_count = self._prepare_profiles(
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        if self._sequence_count > (<size_t> -1) - 31:
            raise OverflowError("candidate mask size overflows size_t")
        words_per_profile = (self._sequence_count + 31) // 32
        if words_per_profile and profile_count > (<size_t> -1) // words_per_profile:
            raise OverflowError("candidate mask size overflows size_t")
        candidate_word_count = profile_count * words_per_profile
        self._candidate_counts.resize(profile_count)
        if not host_candidate_expansion and candidate_word_count <= INT_MAX:
            self._candidate_words.clear()
            error[0] = 0
            if self._generation_ledger_enabled:
                ledger_start_ns = _time.perf_counter_ns()
            with nogil:
                status = plan7_ssv_sequence_batch_f1_compact_many(
                    self._batch,
                    &packed_scores[0] if packed_scores.shape[0] else NULL,
                    <size_t> packed_scores.shape[0],
                    self._profiles.data() if profile_count else NULL,
                    profile_count,
                    &m_mu[0] if profile_count else NULL,
                    &m_lambda[0] if profile_count else NULL,
                    f1,
                    error,
                    sizeof(error),
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            if self._generation_ledger_enabled:
                self._ledger_f1_native_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
                ledger_start_ns = _time.perf_counter_ns()
            error[0] = 0
            status = plan7_ssv_sequence_batch_get_f1_candidate_view(
                self._batch, &compact_view, error, sizeof(error)
            )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            if compact_view.profile_count != profile_count:
                raise RuntimeError("device candidate profile count changed")
            candidate_count = compact_view.candidate_count
            self._candidate_offsets.resize(profile_count)
            self._candidate_indices.resize(candidate_count)
            for profile_index in range(profile_count):
                if (
                    compact_view.candidate_offsets[profile_index]
                    > compact_view.candidate_offsets[profile_index + 1]
                    or compact_view.candidate_offsets[profile_index + 1]
                    > candidate_count
                ):
                    raise RuntimeError("device candidate offsets are invalid")
                self._candidate_offsets[profile_index] = (
                    compact_view.candidate_offsets[profile_index]
                )
                self._candidate_counts[profile_index] = (
                    compact_view.candidate_offsets[profile_index + 1]
                    - compact_view.candidate_offsets[profile_index]
                )
                for output_index in range(
                    compact_view.candidate_offsets[profile_index],
                    compact_view.candidate_offsets[profile_index + 1],
                ):
                    mapping = compact_view.candidates[output_index]
                    if (
                        mapping.profile_index != profile_index
                        or mapping.sequence_index >= self._sequence_count
                    ):
                        raise RuntimeError("device candidate mapping is invalid")
                    self._candidate_indices[output_index] = mapping.sequence_index
            if (
                profile_count
                and compact_view.candidate_offsets[profile_count] != candidate_count
            ):
                raise RuntimeError("device candidate count changed")
            if self._generation_ledger_enabled:
                self._ledger_f1_candidate_mirror_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
            return profile_count

        self._candidate_words.resize(candidate_word_count)
        error[0] = 0
        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()
        with nogil:
            status = plan7_ssv_sequence_batch_f1_mask_many(
                self._batch,
                &packed_scores[0] if packed_scores.shape[0] else NULL,
                <size_t> packed_scores.shape[0],
                self._profiles.data() if profile_count else NULL,
                profile_count,
                &m_mu[0] if profile_count else NULL,
                &m_lambda[0] if profile_count else NULL,
                f1,
                self._candidate_words.data() if candidate_word_count else NULL,
                candidate_word_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        if self._generation_ledger_enabled:
            self._ledger_f1_native_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
            ledger_start_ns = _time.perf_counter_ns()

        self._candidate_offsets.resize(profile_count)
        for profile_index in range(profile_count):
            self._candidate_counts[profile_index] = 0
            for word_index in range(words_per_profile):
                self._candidate_counts[profile_index] += plan7_popcount_u32(
                    self._candidate_words[
                        profile_index * words_per_profile + word_index
                    ]
                )
            self._candidate_offsets[profile_index] = candidate_count
            if self._candidate_counts[profile_index] > (<size_t> -1) - candidate_count:
                raise OverflowError("candidate count overflows size_t")
            candidate_count += self._candidate_counts[profile_index]

        self._candidate_indices.resize(candidate_count)
        for profile_index in range(profile_count):
            output_index = self._candidate_offsets[profile_index]
            for word_index in range(words_per_profile):
                word = self._candidate_words[
                    profile_index * words_per_profile + word_index
                ]
                while word:
                    bit = plan7_ctz_u32(word)
                    sequence_index = word_index * 32 + bit
                    if sequence_index >= self._sequence_count:
                        raise RuntimeError("candidate mask has trailing bits set")
                    self._candidate_indices[output_index] = <uint32_t> sequence_index
                    output_index += 1
                    word &= word - 1
            if output_index != (
                self._candidate_offsets[profile_index]
                + self._candidate_counts[profile_index]
            ):
                raise RuntimeError("candidate mask count changed")
        if self._generation_ledger_enabled:
            self._ledger_f1_candidate_mirror_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
        return profile_count

    def cpu_candidates_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        bint _host_candidate_expansion=False,
    ):
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t sequence_index
        cdef list candidates
        cdef list output = []

        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            _host_candidate_expansion,
        )
        for profile_index in range(profile_count):
            candidates = []
            for sequence_index in range(self._candidate_counts[profile_index]):
                candidates.append(
                    self._candidate_indices[
                        self._candidate_offsets[profile_index] + sequence_index
                    ]
                )
            output.append(candidates)
        return output

    def evaluate_f0_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] residue_classes,
        int class_count,
    ):
        """Evaluate a certified reduced-alphabet F0 outside production."""
        cdef size_t profile_count = <size_t> score_offsets.shape[0]
        cdef size_t profile_index
        cdef int status
        cdef char error[512]
        cdef vector[plan7_f0_profile_statistics] rows
        cdef plan7_f0_profile_statistics row
        cdef plan7_f0_evaluation_statistics summary
        cdef list profiles = []

        if (
            <size_t> m_mu.shape[0] != profile_count
            or <size_t> m_lambda.shape[0] != profile_count
        ):
            raise ValueError("e-value parameter lengths differ")
        profile_count = self._prepare_profiles(
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        if class_count < 2:
            raise ValueError("F0 class count must be at least two")
        rows.resize(profile_count)
        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_evaluate_f0_many(
                self._batch,
                &packed_scores[0] if packed_scores.shape[0] else NULL,
                <size_t> packed_scores.shape[0],
                self._profiles.data() if profile_count else NULL,
                profile_count,
                &m_mu[0] if profile_count else NULL,
                &m_lambda[0] if profile_count else NULL,
                f1,
                &residue_classes[0] if residue_classes.shape[0] else NULL,
                <size_t> residue_classes.shape[0],
                <size_t> class_count,
                rows.data() if profile_count else NULL,
                profile_count,
                &summary,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        for profile_index in range(profile_count):
            row = rows[profile_index]
            profiles.append(
                {
                    "profile_index": profile_index,
                    "logical_pair_count": row.logical_pair_count,
                    "exact_candidate_count": row.exact_candidate_count,
                    "coarse_candidate_count": row.coarse_candidate_count,
                    "certified_reject_count": row.certified_reject_count,
                    "false_reject_count": row.false_reject_count,
                    "logical_cell_count": row.logical_cell_count,
                    "survivor_exact_cell_count": (
                        row.survivor_exact_cell_count
                    ),
                }
            )
        return {
            "schema": "plan7_gpu.phase8_f0_evaluation.v1",
            "profile_count": summary.profile_count,
            "sequence_count": summary.sequence_count,
            "class_count": summary.class_count,
            "logical_pair_count": summary.logical_pair_count,
            "exact_candidate_count": summary.exact_candidate_count,
            "coarse_candidate_count": summary.coarse_candidate_count,
            "certified_reject_count": summary.certified_reject_count,
            "false_reject_count": summary.false_reject_count,
            "logical_cell_count": summary.logical_cell_count,
            "survivor_exact_cell_count": summary.survivor_exact_cell_count,
            "coarse_table_bytes": summary.coarse_table_bytes,
            "temporary_device_bytes": summary.temporary_device_bytes,
            "exact_generation_milliseconds": (
                summary.exact_generation_milliseconds
            ),
            "coarse_table_build_milliseconds": (
                summary.coarse_table_build_milliseconds
            ),
            "coarse_upload_milliseconds": summary.coarse_upload_milliseconds,
            "coarse_kernel_milliseconds": summary.coarse_kernel_milliseconds,
            "analysis_milliseconds": summary.analysis_milliseconds,
            "profiles": profiles,
        }

    def evaluate_mandatory_seed_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        int maximum_word_length,
        int indexed_alphabet_size=20,
    ):
        """Evaluate the certified bounded-word index outside production."""
        cdef size_t profile_count = <size_t> score_offsets.shape[0]
        cdef size_t profile_index
        cdef int status
        cdef char error[512]
        cdef vector[plan7_seed_profile_statistics] rows
        cdef plan7_seed_profile_statistics row
        cdef plan7_seed_evaluation_statistics summary
        cdef list profiles = []

        if (
            <size_t> m_mu.shape[0] != profile_count
            or <size_t> m_lambda.shape[0] != profile_count
        ):
            raise ValueError("e-value parameter lengths differ")
        if maximum_word_length not in (1, 2, 4, 8, 16, 32):
            raise ValueError("maximum seed word length must be 1/2/4/8/16/32")
        if indexed_alphabet_size <= 0:
            raise ValueError("indexed alphabet size must be positive")
        profile_count = self._prepare_profiles(
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        rows.resize(profile_count)
        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_evaluate_seed_many(
                self._batch,
                &packed_scores[0] if packed_scores.shape[0] else NULL,
                <size_t> packed_scores.shape[0],
                self._profiles.data() if profile_count else NULL,
                profile_count,
                &m_mu[0] if profile_count else NULL,
                &m_lambda[0] if profile_count else NULL,
                f1,
                <size_t> maximum_word_length,
                <size_t> indexed_alphabet_size,
                rows.data() if profile_count else NULL,
                profile_count,
                &summary,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        for profile_index in range(profile_count):
            row = rows[profile_index]
            profiles.append(
                {
                    "profile_index": profile_index,
                    "logical_pair_count": row.logical_pair_count,
                    "exact_candidate_count": row.exact_candidate_count,
                    "seed_candidate_count": row.seed_candidate_count,
                    "certified_reject_count": row.certified_reject_count,
                    "false_reject_count": row.false_reject_count,
                    "unsupported_pair_count": row.unsupported_pair_count,
                    "logical_cell_count": row.logical_cell_count,
                    "survivor_exact_cell_count": row.survivor_exact_cell_count,
                }
            )
        return {
            "schema": "plan7_gpu.phase10_mandatory_seed_evaluation.v1",
            "profile_count": summary.profile_count,
            "sequence_count": summary.sequence_count,
            "maximum_word_length": summary.maximum_word_length,
            "indexed_alphabet_size": indexed_alphabet_size,
            "logical_pair_count": summary.logical_pair_count,
            "exact_candidate_count": summary.exact_candidate_count,
            "seed_candidate_count": summary.seed_candidate_count,
            "certified_reject_count": summary.certified_reject_count,
            "false_reject_count": summary.false_reject_count,
            "unsupported_pair_count": summary.unsupported_pair_count,
            "logical_cell_count": summary.logical_cell_count,
            "survivor_exact_cell_count": summary.survivor_exact_cell_count,
            "temporary_device_bytes": summary.temporary_device_bytes,
            "exact_generation_milliseconds": (
                summary.exact_generation_milliseconds
            ),
            "seed_kernel_milliseconds": summary.seed_kernel_milliseconds,
            "analysis_milliseconds": summary.analysis_milliseconds,
            "profiles": profiles,
        }

    def cpu_candidates_many_csr_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        bint _host_candidate_expansion=False,
    ):
        """Return compact candidate rows as native uint32 data and uint64 offsets."""
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef carray indices
        cdef carray offsets

        if _UINT32_ARRAY_TEMPLATE.itemsize != sizeof(uint32_t):
            raise RuntimeError("array('I') is not native uint32")
        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")

        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            _host_candidate_expansion,
        )
        candidate_count = self._candidate_indices.size()
        indices = clone(_UINT32_ARRAY_TEMPLATE, candidate_count, False)
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        if candidate_count:
            memcpy(
                indices.data.as_uints,
                self._candidate_indices.data(),
                candidate_count * sizeof(uint32_t),
            )
        for profile_index in range(profile_count):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._candidate_offsets[profile_index]
            )
        offsets.data.as_ulonglongs[profile_count] = <uint64_t> candidate_count
        return indices, offsets

    cdef size_t _run_bias_candidates_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
    ) except? 0:
        cdef char error[512]
        cdef size_t profile_count
        cdef size_t candidate_count
        cdef size_t profile_index
        cdef size_t expected_bias_bytes
        cdef int status

        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            False,
        )
        if profile_count == 0:
            if packed_bias_profiles.shape[0] != 0:
                raise ValueError("packed bias profiles have trailing bytes")
            self._bias_profiles.clear()
            self._bias_results.clear()
            self._bias_candidate_offsets.resize(1)
            self._bias_candidate_offsets[0] = 0
            return 0
        if profile_count > (<size_t> -1) // sizeof(plan7_bias_profile):
            raise OverflowError("packed bias profile size overflows size_t")
        expected_bias_bytes = profile_count * sizeof(plan7_bias_profile)
        if <size_t> packed_bias_profiles.shape[0] != expected_bias_bytes:
            raise ValueError("packed bias profile buffer has the wrong size")

        self._bias_profiles.resize(profile_count)
        memcpy(
            self._bias_profiles.data(),
            &packed_bias_profiles[0],
            expected_bias_bytes,
        )
        candidate_count = self._candidate_indices.size()
        self._bias_results.resize(candidate_count)
        self._bias_candidate_offsets.resize(profile_count + 1)
        for profile_index in range(profile_count):
            self._bias_candidate_offsets[profile_index] = (
                self._candidate_offsets[profile_index]
            )
        self._bias_candidate_offsets[profile_count] = candidate_count

        if candidate_count == 0:
            return profile_count

        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_bias_candidates_many(
                self._batch,
                self._bias_profiles.data(),
                profile_count,
                self._bias_candidate_offsets.data(),
                self._candidate_indices.data() if candidate_count else NULL,
                candidate_count,
                self._bias_results.data() if candidate_count else NULL,
                candidate_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return profile_count

    def bias_candidates_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
    ):
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_index
        cdef plan7_bias_result result
        cdef float_bits score
        cdef object score_bits
        cdef list row
        cdef list output = []

        profile_count = self._run_bias_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            packed_bias_profiles,
        )
        for profile_index in range(profile_count):
            row = []
            for candidate_index in range(
                self._bias_candidate_offsets[profile_index],
                self._bias_candidate_offsets[profile_index + 1],
            ):
                result = self._bias_results[candidate_index]
                if isfinite(result.filtersc):
                    score.value = result.filtersc
                    score_bits = score.bits
                else:
                    score_bits = None
                row.append(
                    (
                        result.sequence_index,
                        result.action,
                        result.ssv_status,
                        result.ssv_numerator,
                        score_bits,
                    )
                )
            output.append(row)
        return output

    def bias_candidates_many_csr_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
    ):
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef size_t result_bytes
        cdef bytearray records
        cdef uint8_t[::1] record_view
        cdef carray offsets

        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        profile_count = self._run_bias_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            packed_bias_profiles,
        )
        candidate_count = self._bias_results.size()
        if candidate_count > (<size_t> -1) // sizeof(plan7_bias_result):
            raise OverflowError("bias result size overflows size_t")
        result_bytes = candidate_count * sizeof(plan7_bias_result)
        records = bytearray(result_bytes)
        if result_bytes:
            record_view = records
            memcpy(&record_view[0], self._bias_results.data(), result_bytes)
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._bias_candidate_offsets[profile_index]
            )
        return records, offsets

    def forward_candidates_many_raw(
        self,
        const uint64_t[::1] candidate_offsets,
        const uint32_t[::1] candidate_indices,
        const float[::1] filter_scores,
        double f3,
        source_profiles,
        ForwardProfiles forward_profiles,
        uint64_t gathered_byte_budget=PLAN7_FORWARD_MAX_GATHERED_BYTES,
        bint _f3_audit=False,
        int _candidates_per_warp=0,
    ):
        """Classify F3 and return only passing parser special-state rows."""
        cdef char error[512]
        cdef int status
        cdef size_t profile_count
        cdef size_t candidate_count = <size_t> candidate_indices.shape[0]
        cdef size_t result_count
        cdef size_t result_bytes
        cdef size_t offset_bytes
        cdef size_t special_count
        cdef size_t special_bytes
        cdef size_t profile_index
        cdef bint sealed_source = source_profiles is None
        cdef tuple owners = () if sealed_source else tuple(source_profiles)
        cdef vector[uintptr_t] source_pointers
        cdef OptimizedProfile source_profile
        cdef object value
        cdef plan7_forward_output *output = NULL
        cdef const plan7_forward_result *native_results
        cdef const uint16_t *native_forward_reasons
        cdef const uint64_t *native_offsets
        cdef const float *native_specials
        cdef const plan7_forward_statistics *native_statistics
        cdef const plan7_forward_subwarp_statistics *native_subwarp_statistics
        cdef bytes records
        cdef bytes offset_storage
        cdef bytes special_storage
        cdef object offsets
        cdef object specials
        cdef dict statistics
        cdef ForwardProvenance provenance

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if forward_profiles._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        if candidate_offsets.shape[0] == 0:
            raise ValueError("candidate offsets must contain an initial zero")
        profile_count = <size_t> candidate_offsets.shape[0] - 1
        if <size_t> filter_scores.shape[0] != candidate_count:
            raise ValueError("Forward candidate score lengths differ")
        if not sealed_source and len(owners) != profile_count:
            raise ValueError("source and Forward profile counts differ")
        if plan7_forward_database_profile_count(
            forward_profiles._database
        ) != profile_count:
            raise ValueError("Forward profile and candidate row counts differ")
        if not sealed_source:
            source_pointers.reserve(profile_count)
            for profile_index in range(profile_count):
                value = owners[profile_index]
                if not isinstance(value, OptimizedProfile):
                    raise TypeError(
                        "source profiles must be OptimizedProfile objects"
                    )
                if value is not forward_profiles._owners[profile_index]:
                    raise ValueError(
                        "source profile identity differs from Forward profile row"
                    )
                source_profile = value
                source_pointers.push_back(<uintptr_t> source_profile._om)
        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if _FLOAT_ARRAY_TEMPLATE.itemsize != sizeof(float):
            raise RuntimeError("array('f') is not native float32")
        if sizeof(plan7_forward_result) != PLAN7_FORWARD_RECORD_SIZE:
            raise RuntimeError("Forward result ABI size mismatch")
        if _candidates_per_warp not in (0, 1, 2, 4, 8):
            raise ValueError(
                "Forward candidates per warp must be auto, 1, 2, 4, or 8"
            )
        if _f3_audit and _candidates_per_warp != 0:
            raise ValueError(
                "Forward F3 audit requires automatic subwarp selection"
            )

        error[0] = 0
        # Keep the GIL while native code validates live private profile arrays.
        if _f3_audit:
            status = plan7_forward_run_batch_workspace_f3_audit(
                forward_profiles._database,
                self._batch,
                source_pointers.data() if source_pointers.size() else NULL,
                profile_count,
                &candidate_offsets[0],
                &candidate_indices[0] if candidate_count else NULL,
                &filter_scores[0] if candidate_count else NULL,
                candidate_count,
                f3,
                gathered_byte_budget,
                &output,
                error,
                sizeof(error),
            )
        elif _candidates_per_warp == 0:
            status = plan7_forward_run_batch_workspace_variant(
                forward_profiles._database,
                self._batch,
                source_pointers.data() if source_pointers.size() else NULL,
                profile_count,
                &candidate_offsets[0],
                &candidate_indices[0] if candidate_count else NULL,
                &filter_scores[0] if candidate_count else NULL,
                candidate_count,
                f3,
                gathered_byte_budget,
                0,
                &output,
                error,
                sizeof(error),
            )
        else:
            status = plan7_forward_run_batch_workspace_variant(
                forward_profiles._database,
                self._batch,
                source_pointers.data() if source_pointers.size() else NULL,
                profile_count,
                &candidate_offsets[0],
                &candidate_indices[0] if candidate_count else NULL,
                &filter_scores[0] if candidate_count else NULL,
                candidate_count,
                f3,
                gathered_byte_budget,
                _candidates_per_warp,
                &output,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        try:
            result_count = plan7_forward_output_result_count(output)
            if result_count != candidate_count:
                raise RuntimeError("Forward result count changed")
            if result_count > (<size_t> -1) // sizeof(plan7_forward_result):
                raise OverflowError("Forward result size overflows size_t")
            result_bytes = result_count * sizeof(plan7_forward_result)
            if result_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("Forward result size exceeds Python limits")
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            native_results = plan7_forward_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Forward result storage is null")
                memcpy(PyBytes_AS_STRING(records), native_results, result_bytes)

            native_offsets = plan7_forward_output_special_offsets(output)
            if native_offsets == NULL:
                raise RuntimeError("Forward special offsets are null")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
            ) - 1:
                raise OverflowError("Forward special offsets exceed Python limits")
            offset_bytes = (result_count + 1) * sizeof(uint64_t)
            offset_storage = PyBytes_FromStringAndSize(
                NULL,
                offset_bytes,
            )
            memcpy(
                PyBytes_AS_STRING(offset_storage),
                native_offsets,
                offset_bytes,
            )
            offsets = memoryview(offset_storage).cast("Q")

            special_count = plan7_forward_output_special_count(output)
            if special_count > (<size_t> -1) // sizeof(float):
                raise OverflowError("Forward special matrix size overflows size_t")
            special_bytes = special_count * sizeof(float)
            if special_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError(
                    "Forward special matrix size exceeds Python limits"
                )
            special_storage = PyBytes_FromStringAndSize(NULL, special_bytes)
            native_specials = plan7_forward_output_specials(output)
            if special_bytes:
                if native_specials == NULL:
                    raise RuntimeError("Forward special matrix is null")
                memcpy(
                    PyBytes_AS_STRING(special_storage),
                    native_specials,
                    special_bytes,
                )
            specials = memoryview(special_storage).cast("f")

            native_statistics = plan7_forward_output_statistics(output)
            if native_statistics == NULL:
                raise RuntimeError("Forward statistics are null")
            native_subwarp_statistics = (
                plan7_forward_output_subwarp_statistics(output)
            )
            if native_subwarp_statistics == NULL:
                raise RuntimeError("Forward subwarp statistics are null")
            statistics = {
                "generation_f3_bits": native_statistics.generation_f3_bits,
                "candidate_count": native_statistics.candidate_count,
                "survivor_count": native_statistics.survivor_count,
                "work_cells": native_statistics.work_cells,
                "dp_workspace_bytes": native_statistics.dp_workspace_bytes,
                "xmx_workspace_bytes": native_statistics.xmx_workspace_bytes,
                "gather_workspace_bytes": (
                    native_statistics.gather_workspace_bytes
                ),
                "gathered_xmx_bytes": native_statistics.gathered_xmx_bytes,
                "output_byte_limit": native_statistics.output_byte_limit,
                "output_cap_fallback_count": (
                    native_statistics.output_cap_fallback_count
                ),
                "kernel_ms": native_statistics.kernel_milliseconds,
                "classification_ms": (
                    native_statistics.classification_milliseconds
                ),
                "gather_ms": native_statistics.gather_milliseconds,
                "download_ms": native_statistics.download_milliseconds,
                "total_ms": native_statistics.total_milliseconds,
                "subwarp_policy_version": (
                    native_subwarp_statistics.policy_version
                ),
                "requested_candidates_per_warp": (
                    native_subwarp_statistics.requested_candidates_per_warp
                ),
                "candidates_per_warp": (
                    native_subwarp_statistics.candidates_per_warp
                ),
                "subwarp_policy_reason": (
                    native_subwarp_statistics.policy_reason
                ),
                "multiprocessor_count": (
                    native_subwarp_statistics.multiprocessor_count
                ),
                "l2_cache_bytes": native_subwarp_statistics.l2_cache_bytes,
                "policy_tile_candidate_count": (
                    native_subwarp_statistics.policy_tile_candidate_count
                ),
                "model_length_sum": native_subwarp_statistics.model_length_sum,
                "target_length_sum": native_subwarp_statistics.target_length_sum,
                "average_model_length": (
                    native_subwarp_statistics.average_model_length
                ),
                "average_target_length": (
                    native_subwarp_statistics.average_target_length
                ),
                "maximum_model_length": (
                    native_subwarp_statistics.maximum_model_length
                ),
                "maximum_target_length": (
                    native_subwarp_statistics.maximum_target_length
                ),
                "maximum_candidate_work_cells": (
                    native_subwarp_statistics.maximum_candidate_work_cells
                ),
                "average_work_cells": (
                    native_subwarp_statistics.average_work_cells
                ),
                "short_width4_workspace_limit_bytes": (
                    native_subwarp_statistics.short_width4_workspace_limit_bytes
                ),
                "long_packed_workspace_limit_bytes": (
                    native_subwarp_statistics.long_packed_workspace_limit_bytes
                ),
                "policy_xmx_workspace_bytes": (
                    native_subwarp_statistics.policy_xmx_workspace_bytes
                ),
                "minimum_cta_count": (
                    native_subwarp_statistics.minimum_cta_count
                ),
                "width1_cta_count": native_subwarp_statistics.width1_cta_count,
                "width2_cta_count": native_subwarp_statistics.width2_cta_count,
                "width4_cta_count": native_subwarp_statistics.width4_cta_count,
                "kernel_launch_count": (
                    native_subwarp_statistics.kernel_launch_count
                ),
                "scheduled_warp_count": (
                    native_subwarp_statistics.scheduled_warp_count
                ),
                "candidate_subwarp_count": (
                    native_subwarp_statistics.candidate_subwarp_count
                ),
                "active_lane_slots": (
                    native_subwarp_statistics.active_lane_slots
                ),
                "issued_lane_slots": (
                    native_subwarp_statistics.issued_lane_slots
                ),
            }
            statistics.update(_forward_f3_device_statistics_from_output(output))
            provenance = _forward_provenance_from_output(output)
            statistics.update({
                "database_generation": provenance._value.database_generation,
                "batch_generation": provenance._value.batch_generation,
                "row_hash": provenance._value.row_hash,
                "special_hash": provenance._value.special_hash,
                "continuation_hash": provenance._value.continuation_hash,
                "pass_count": provenance._value.pass_count,
                "special_count": provenance._value.special_count,
                "_provenance": provenance,
            })
            return records, offsets, specials, statistics
        finally:
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)

    def backward_domain_many_raw(
        self,
        const uint32_t[::1] candidate_profiles,
        const uint32_t[::1] candidate_indices,
        const uint64_t[::1] forward_offsets,
        const float[::1] forward_specials,
        ForwardProfiles forward_profiles,
        ForwardProvenance provenance,
        float rt1=0.25,
        float rt2=0.10,
        float rt3=0.20,
        float guard_band=2.0e-4,
        uint64_t posterior_byte_budget=(384 * 1024 * 1024),
        bint _unsealed_test=False,
    ):
        """Run CUDA BackwardParser and domain posterior decoding."""
        cdef size_t candidate_count = <size_t> candidate_indices.shape[0]
        cdef vector[plan7_backward_domain_candidate] candidates
        cdef size_t candidate
        cdef plan7_backward_domain_output *output = NULL
        cdef const plan7_backward_domain_result *native_results
        cdef const uint64_t *native_offsets
        cdef const plan7_domain_posterior *native_posteriors
        cdef const uint64_t *native_region_offsets
        cdef const plan7_simple_region *native_regions
        cdef const plan7_backward_domain_statistics *native_statistics
        cdef size_t result_count
        cdef size_t result_bytes
        cdef size_t offset_bytes
        cdef size_t posterior_count
        cdef size_t posterior_bytes
        cdef size_t region_count
        cdef size_t region_bytes
        cdef bytes records
        cdef bytes offset_storage
        cdef bytes posterior_storage
        cdef bytes region_offset_storage
        cdef bytes region_storage
        cdef object offsets
        cdef object posteriors
        cdef object region_offsets
        cdef object regions
        cdef dict statistics
        cdef BackwardDomainProvenance output_provenance
        cdef char error[512]
        cdef int status

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if forward_profiles._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        if <size_t> candidate_profiles.shape[0] != candidate_count:
            raise ValueError("Backward/domain candidate lengths differ")
        if <size_t> forward_offsets.shape[0] != candidate_count + 1:
            raise ValueError("Backward/domain Forward offsets have wrong length")
        if sizeof(plan7_backward_domain_result) != (
            PLAN7_BACKWARD_DOMAIN_RECORD_SIZE
        ):
            raise RuntimeError("Backward/domain result ABI size mismatch")
        if sizeof(plan7_domain_posterior) != (
            PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE
        ):
            raise RuntimeError("Backward/domain posterior ABI size mismatch")
        if sizeof(plan7_simple_region) != PLAN7_BACKWARD_DOMAIN_REGION_SIZE:
            raise RuntimeError("Backward/domain region ABI size mismatch")
        candidates.resize(candidate_count)
        for candidate in range(candidate_count):
            candidates[candidate].profile_index = candidate_profiles[candidate]
            candidates[candidate].sequence_index = candidate_indices[candidate]

        error[0] = 0
        if _unsealed_test:
            with nogil:
                status = plan7_backward_domain_unsealed_test_run(
                    forward_profiles._database,
                    self._batch,
                    candidates.data() if candidate_count else NULL,
                    candidate_count,
                    &forward_offsets[0],
                    &forward_specials[0] if forward_specials.shape[0] else NULL,
                    <size_t> forward_specials.shape[0],
                    rt1,
                    rt2,
                    rt3,
                    guard_band,
                    posterior_byte_budget,
                    &output,
                    error,
                    sizeof(error),
                )
        else:
            with nogil:
                status = plan7_backward_domain_run(
                    forward_profiles._database,
                    self._batch,
                    candidates.data() if candidate_count else NULL,
                    candidate_count,
                    &provenance._value,
                    &forward_offsets[0],
                    &forward_specials[0] if forward_specials.shape[0] else NULL,
                    <size_t> forward_specials.shape[0],
                    rt1,
                    rt2,
                    rt3,
                    guard_band,
                    posterior_byte_budget,
                    &output,
                    error,
                    sizeof(error),
                )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        try:
            result_count = plan7_backward_domain_output_result_count(output)
            if result_count != candidate_count:
                raise RuntimeError("Backward/domain result count changed")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(plan7_backward_domain_result)
            ):
                raise OverflowError("Backward/domain results exceed Python limits")
            result_bytes = result_count * sizeof(plan7_backward_domain_result)
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            native_results = plan7_backward_domain_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Backward/domain result storage is null")
                memcpy(PyBytes_AS_STRING(records), native_results, result_bytes)

            native_offsets = (
                plan7_backward_domain_output_posterior_offsets(output)
            )
            if native_offsets == NULL:
                raise RuntimeError("Backward/domain posterior offsets are null")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
            ) - 1:
                raise OverflowError(
                    "Backward/domain posterior offsets exceed Python limits"
                )
            offset_bytes = (result_count + 1) * sizeof(uint64_t)
            offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
            memcpy(
                PyBytes_AS_STRING(offset_storage),
                native_offsets,
                offset_bytes,
            )
            offsets = memoryview(offset_storage).cast("Q")

            posterior_count = (
                plan7_backward_domain_output_posterior_count(output)
            )
            if posterior_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_posterior)
            ):
                raise OverflowError(
                    "Backward/domain posterior output exceeds Python limits"
                )
            posterior_bytes = (
                posterior_count * sizeof(plan7_domain_posterior)
            )
            posterior_storage = PyBytes_FromStringAndSize(NULL, posterior_bytes)
            native_posteriors = plan7_backward_domain_output_posteriors(output)
            if posterior_bytes:
                if native_posteriors == NULL:
                    raise RuntimeError("Backward/domain posterior storage is null")
                memcpy(
                    PyBytes_AS_STRING(posterior_storage),
                    native_posteriors,
                    posterior_bytes,
                )
            posteriors = memoryview(posterior_storage).cast("f")

            native_region_offsets = (
                plan7_backward_domain_output_region_offsets(output)
            )
            if native_region_offsets == NULL:
                raise RuntimeError("Backward/domain region offsets are null")
            region_offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
            memcpy(
                PyBytes_AS_STRING(region_offset_storage),
                native_region_offsets,
                offset_bytes,
            )
            region_offsets = memoryview(region_offset_storage).cast("Q")

            region_count = plan7_backward_domain_output_region_count(output)
            if region_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(plan7_simple_region)
            ):
                raise OverflowError(
                    "Backward/domain region output exceeds Python limits"
                )
            region_bytes = region_count * sizeof(plan7_simple_region)
            region_storage = PyBytes_FromStringAndSize(NULL, region_bytes)
            native_regions = plan7_backward_domain_output_regions(output)
            if region_bytes:
                if native_regions == NULL:
                    raise RuntimeError("Backward/domain region storage is null")
                memcpy(
                    PyBytes_AS_STRING(region_storage),
                    native_regions,
                    region_bytes,
                )
            regions = memoryview(region_storage).cast("I")

            native_statistics = plan7_backward_domain_output_statistics(output)
            if native_statistics == NULL:
                raise RuntimeError("Backward/domain statistics are null")
            statistics = {
                "candidate_count": native_statistics.candidate_count,
                "device_result_count": native_statistics.device_result_count,
                "cpu_required_count": native_statistics.cpu_required_count,
                "work_cells": native_statistics.work_cells,
                "dp_workspace_bytes": native_statistics.dp_workspace_bytes,
                "backward_special_workspace_bytes": (
                    native_statistics.backward_special_workspace_bytes
                ),
                "forward_special_workspace_bytes": (
                    native_statistics.forward_special_workspace_bytes
                ),
                "posterior_bytes": native_statistics.posterior_bytes,
                "simple_region_bytes": native_statistics.simple_region_bytes,
                "output_byte_limit": native_statistics.output_byte_limit,
                "output_cap_fallback_count": (
                    native_statistics.output_cap_fallback_count
                ),
                "work_cap_fallback_count": (
                    native_statistics.work_cap_fallback_count
                ),
                "posterior_omitted_count": (
                    native_statistics.posterior_omitted_count
                ),
                "own_scale_count": native_statistics.own_scale_count,
                "threshold_uncertain_count": (
                    native_statistics.threshold_uncertain_count
                ),
                "no_region_count": native_statistics.no_region_count,
                "simple_count": native_statistics.simple_count,
                "multidomain_fallback_count": (
                    native_statistics.multidomain_fallback_count
                ),
                "kernel_ms": native_statistics.kernel_milliseconds,
                "upload_ms": native_statistics.upload_milliseconds,
                "download_ms": native_statistics.download_milliseconds,
                "total_ms": native_statistics.total_milliseconds,
                "unsealed_test": bool(_unsealed_test),
            }
            output_provenance = _backward_provenance_from_output(output)
            statistics.update({
                "threshold_hash": output_provenance._value.threshold_hash,
                "result_hash": output_provenance._value.result_hash,
                "region_hash": output_provenance._value.region_hash,
                "_provenance": output_provenance,
            })
            return (
                records,
                offsets,
                posteriors,
                region_offsets,
                regions,
                statistics,
            )
        finally:
            if output != NULL:
                plan7_backward_domain_output_destroy(&output, NULL, 0)

    cdef object _forward_profile_selection_raw(
        self,
        ProfileSelection selection,
        const uint8_t[::1] postfilter_records,
        const uint64_t[::1] postfilter_offsets,
        const uint64_t[::1] residue_offsets,
        double f2,
        double f3,
        uint64_t gathered_byte_budget,
        bint sealed_domain_journal,
        bint rescore_simple_diagnostic,
        uint64_t rescore_compact_byte_budget,
        uint64_t rescore_matrix_byte_budget,
        uint64_t rescore_trace_byte_budget,
        int rescore_test_fault,
        uint64_t generation_tail_fingerprint,
        double generation_f1,
        bint generation_bias_filter,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        object generation_telemetry_seed,
        object postfilter_owner,
        bint direct_sparse_v3,
        object ga_target_cutoffs,
    ):
        """Run selection-aware F2/F3 without reading a live optimized profile."""
        # CPU-rescore ownership deliberately exposes the pre-compact tail to
        # continuation.  Bind and seal that actual generation contract, not
        # the compact-tail fingerprint requested by the ordinary GPU path.
        if self._cpu_rescore_route:
            generation_tail_fingerprint = 0
        cdef plan7_profile_selection_view view = selection._view()
        cdef vector[uint64_t] candidate_offsets
        cdef vector[uint32_t] candidate_indices
        cdef vector[float] filter_scores
        cdef vector[float] uncorrected_scores
        cdef vector[plan7_postfilter_result] candidate_records
        cdef vector[uint32_t] candidate_profiles
        cdef vector[size_t] candidate_postfilter_sources
        cdef vector[plan7_backward_domain_candidate] domain_candidates
        cdef vector[size_t] pass_sources
        cdef vector[float] ga_whole_forward_scores
        cdef vector[uint64_t] pass_special_offsets
        cdef vector[uint64_t] journal_profile_offsets
        cdef vector[uint64_t] generation_metrics
        cdef vector[uint64_t] postfilter_reason_counts
        cdef vector[uint64_t] f2_reason_counts
        cdef vector[uint64_t] forward_reason_counts
        cdef vector[uint64_t] backward_reason_counts
        cdef vector[uint64_t] rescore_reason_counts
        cdef vector[uint64_t] postfilter_reason_cells
        cdef vector[uint64_t] f2_reason_cells
        cdef vector[uint64_t] forward_reason_cells
        cdef vector[uint64_t] backward_reason_cells
        cdef vector[uint64_t] rescore_reason_cells
        cdef vector[plan7_forward_snapshot_profile] forward_snapshots
        cdef plan7_backward_domain_candidate domain_candidate
        cdef plan7_postfilter_result record
        cdef float_bits vfsc_bits
        cdef double_bits generation_f3
        cdef float usc
        cdef float bit_score
        cdef double probability
        cdef uint16_t reason16
        cdef uint32_t reason32
        cdef uint16_t forward_facts
        cdef uint8_t forward_call_facts = 0
        cdef uint8_t direct_decision = 0
        cdef uint8_t direct_compact_route = PLAN7_CONTINUATION_COMPACT_NONE
        cdef uint32_t backward_preflight_mask
        cdef uint32_t previous
        cdef bint have_previous
        cdef bint host_attested
        cdef bint use_resident_f2 = False
        cdef bint resident_f2_pass = False
        cdef bint host_f2_pass = False
        cdef size_t profile_count = view.profile_count
        cdef size_t profile_index
        cdef size_t cursor
        cdef size_t start
        cdef size_t stop
        cdef size_t record_count
        cdef size_t candidate_count
        cdef uint64_t sequence_length
        cdef plan7_forward_database *database = NULL
        cdef plan7_forward_output *output = NULL
        cdef plan7_forward_resident_view resident_view
        cdef plan7_postfilter_f2_resident_view f2_resident_view
        cdef plan7_backward_domain_output *domain_output = NULL
        cdef plan7_domain_rescore_output *rescore_output = NULL
        cdef const plan7_forward_result *native_results
        cdef const uint64_t *native_offsets
        cdef const float *native_specials
        cdef const plan7_forward_provenance *native_provenance
        cdef const plan7_forward_statistics *native_statistics
        cdef const plan7_forward_residency_statistics *native_residency_statistics
        cdef const plan7_forward_input_residency_statistics *native_input_residency_statistics
        cdef const plan7_backward_domain_statistics *native_domain_statistics
        cdef const plan7_backward_domain_residency_statistics *native_domain_residency_statistics
        cdef const plan7_domain_rescore_statistics *native_rescore_statistics
        cdef const plan7_domain_rescore_residency_statistics *native_rescore_residency_statistics
        cdef const uint32_t *native_domain_reasons
        cdef const uint32_t *native_rescore_reasons
        cdef const plan7_backward_domain_result *native_domain_results = NULL
        cdef const uint64_t *native_domain_region_offsets = NULL
        cdef const plan7_simple_region *native_domain_regions = NULL
        cdef const plan7_domain_rescore_result *native_rescore_results = NULL
        cdef const uint64_t *native_rescore_trace_offsets = NULL
        cdef char error[512]
        cdef char destroy_error[512]
        cdef int status = 0
        cdef int resident_status = 0
        cdef int destroy_status = 0
        cdef size_t result_count
        cdef size_t result_bytes
        cdef size_t offset_bytes
        cdef size_t special_count
        cdef size_t special_bytes
        cdef size_t pass_count
        cdef size_t source
        cdef size_t reason_count
        cdef size_t region
        cdef size_t row
        cdef size_t reason_base = 0
        cdef size_t resident_f2_cursor = 0
        cdef size_t resident_f2_source = 0
        cdef uint64_t total_target_residues = 0
        cdef uint64_t journal_total_bytes = 0
        cdef uint64_t logical_cells
        cdef uint64_t postfilter_base_cells
        cdef uint64_t postfilter_execution_count
        cdef bint collect_generation_telemetry = (
            generation_telemetry_seed is not None
        )
        cdef bint direct_domain_safe
        cdef bint direct_no_region
        cdef bint direct_ga_reject
        cdef bint cpu_forward_selected
        cdef int cpu_forward_route_mode = self._cpu_forward_route_mode
        cdef uint64_t forward_work_cells
        cdef size_t direct_postfilter_source
        cdef size_t direct_region_begin
        cdef size_t direct_region_end
        cdef size_t direct_trace_begin
        cdef size_t direct_trace_end
        cdef size_t direct_compact_source_count = 0
        cdef uint64_t direct_payload_delta
        cdef uint64_t direct_decision_terms = 0
        cdef uint64_t direct_exception_count = 0
        cdef uint64_t direct_special_count = 0
        cdef uint64_t direct_region_count = 0
        cdef uint64_t direct_compact_result_count = 0
        cdef uint64_t direct_compact_trace_count = 0
        cdef uint64_t direct_compact_null2_count = 0
        cdef bytes records
        cdef bytes direct_decision_plan = b""
        cdef uint8_t *direct_decisions = NULL
        cdef bytes offset_storage
        cdef bytes special_storage
        cdef object special_offsets
        cdef object specials
        cdef carray row_offsets
        cdef carray expected_indices
        cdef dict statistics
        cdef ForwardProvenance provenance
        cdef object journal_capsule = None
        cdef object sealed_stage_timings = None
        cdef object rescore_payload = None
        cdef object upstream_payload = None
        cdef object generation_statistics = None
        cdef object profile_records
        cdef object metric_values
        cdef object reason_values
        cdef object reason_cell_values
        cdef object stage_reason_values
        cdef object stage_reason_cell_values
        cdef object native_totals
        cdef object postfilter_reason_storage = None
        cdef const uint16_t[::1] postfilter_reason_view
        cdef bytes profile_fingerprint_storage = b""
        cdef const uint8_t[::1] profile_fingerprint_view
        cdef const uint8_t[::1] sequence_fingerprint_view
        cdef const float[::1] ga_target_cutoff_view
        cdef bint ga_pruning = ga_target_cutoffs is not None
        cdef double_bits threshold_bits
        cdef float_bits threshold_float_bits
        cdef uint64_t ledger_start_ns = 0

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if self._cpu_forward_route_auto and (
            profile_count < 64
            or not sealed_domain_journal
            or not direct_sparse_v3
            or not self._cpu_domain_route
            or collect_generation_telemetry
        ):
            cpu_forward_route_mode = 0
        if cpu_forward_route_mode != 0:
            if not sealed_domain_journal or not direct_sparse_v3:
                raise ValueError(
                    "CPU Forward ownership requires direct sparse sealed generation"
                )
            if not self._cpu_domain_route:
                raise ValueError(
                    "CPU Forward ownership requires CPU domain ownership"
                )
            if collect_generation_telemetry:
                raise ValueError(
                    "CPU Forward ownership reason telemetry is not implemented"
                )
        memset(&f2_resident_view, 0, sizeof(f2_resident_view))
        if ga_pruning:
            if not sealed_domain_journal or not direct_sparse_v3:
                raise ValueError(
                    "GA pruning requires direct sparse sealed continuation"
                )
            ga_target_cutoff_view = ga_target_cutoffs
            if ga_target_cutoff_view.shape[0] != profile_count:
                raise ValueError("GA target cutoff count changed")
            for profile_index in range(profile_count):
                if not isfinite(ga_target_cutoff_view[profile_index]):
                    raise ValueError("GA target cutoff is not finite")
        if sizeof(plan7_postfilter_result) != PLAN7_POSTFILTER_RECORD_SIZE:
            raise RuntimeError("post-filter result ABI size mismatch")
        if sizeof(plan7_forward_result) != PLAN7_FORWARD_RECORD_SIZE:
            raise RuntimeError("Forward result ABI size mismatch")
        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if _UINT32_ARRAY_TEMPLATE.itemsize != sizeof(uint32_t):
            raise RuntimeError("array('I') is not native uint32")
        if sealed_domain_journal:
            if (
                not generation_bias_filter
                or not isfinite(generation_f1)
                or generation_f1 < 0.0
                or generation_f1 > 1.0
                or not isfinite(f2)
                or f2 < 0.0
                or f2 > 1.0
                or not isfinite(f3)
                or f3 < 0.0
                or f3 > 1.0
                or not isfinite(rt1)
                or not isfinite(rt2)
                or not isfinite(rt3)
                or not isfinite(guard_band)
                or guard_band < <float> 2.0e-4
                or guard_band > <float> 1.0
                or rt1 != <float> 0.25
                or rt2 != <float> 0.10
                or rt3 != <float> 0.20
            ):
                raise ValueError(
                    "sealed domain journals require finite canonical thresholds "
                    "with bias filtering and guard_band >= 2e-4"
                )
        if postfilter_offsets.shape[0] != profile_count + 1:
            raise ValueError("post-filter row-offset count differs from selection")
        if postfilter_offsets[0] != 0:
            raise ValueError("post-filter row offsets must start at zero")
        if <size_t> postfilter_records.shape[0] % sizeof(plan7_postfilter_result):
            raise ValueError("post-filter result storage has trailing bytes")
        record_count = (
            <size_t> postfilter_records.shape[0]
            // sizeof(plan7_postfilter_result)
        )
        if postfilter_offsets[profile_count] != record_count:
            raise ValueError("post-filter row offsets do not span result storage")
        if direct_sparse_v3:
            if record_count > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("direct v3 decision plan exceeds Python limits")
            direct_decision_plan = PyBytes_FromStringAndSize(NULL, record_count)
            direct_decisions = <uint8_t *> PyBytes_AS_STRING(
                direct_decision_plan
            )
        if residue_offsets.shape[0] != self._sequence_count + 1:
            raise ValueError("target residue-prefix length differs from targets")
        if residue_offsets[0] != 0:
            raise ValueError("target residue prefix must start at zero")
        if profile_count and (
            view.profiles == NULL
            or view.m_mu == NULL
            or view.m_lambda == NULL
            or view.v_mu == NULL
            or view.v_lambda == NULL
            or view.identity_tokens == NULL
        ):
            raise RuntimeError("profile selection storage is incomplete")

        if collect_generation_telemetry:
            if (
                not isinstance(generation_telemetry_seed, tuple)
                or len(generation_telemetry_seed) != 3
                or generation_telemetry_seed[0]
                    != GENERATION_TELEMETRY_SCHEMA_VERSION
                or not isinstance(generation_telemetry_seed[1], bytes)
                or not isinstance(generation_telemetry_seed[2], tuple)
                or len(generation_telemetry_seed[2]) != 6
            ):
                raise ValueError("invalid generation telemetry seed")
            postfilter_reason_storage = generation_telemetry_seed[1]
            if len(postfilter_reason_storage) != record_count * sizeof(uint16_t):
                raise ValueError("post-filter reason fact count changed")
            postfilter_reason_view = memoryview(
                postfilter_reason_storage
            ).cast("H")
            generation_metrics.resize(profile_count * GENERATION_METRIC_COUNT)
            postfilter_reason_counts.resize(
                profile_count * GENERATION_POSTFILTER_REASON_COUNT
            )
            f2_reason_counts.resize(
                profile_count * GENERATION_F2_REASON_COUNT
            )
            forward_reason_counts.resize(
                profile_count * GENERATION_FORWARD_REASON_COUNT
            )
            backward_reason_counts.resize(
                profile_count * GENERATION_BACKWARD_REASON_COUNT
            )
            rescore_reason_counts.resize(
                profile_count * GENERATION_RESCORE_REASON_COUNT
            )
            postfilter_reason_cells.resize(
                profile_count * GENERATION_POSTFILTER_REASON_COUNT
            )
            f2_reason_cells.resize(
                profile_count * GENERATION_F2_REASON_COUNT
            )
            forward_reason_cells.resize(
                profile_count * GENERATION_FORWARD_REASON_COUNT
            )
            backward_reason_cells.resize(
                profile_count * GENERATION_BACKWARD_REASON_COUNT
            )
            rescore_reason_cells.resize(
                profile_count * GENERATION_RESCORE_REASON_COUNT
            )
            forward_snapshots.resize(profile_count)
            for cursor in range(self._sequence_count):
                if self._lengths[cursor] > (<uint64_t> -1) - total_target_residues:
                    raise OverflowError("target residue total overflows uint64")
                total_target_residues += self._lengths[cursor]
            for profile_index in range(profile_count):
                reason_base = profile_index * GENERATION_METRIC_COUNT
                generation_metrics[reason_base + GENERATION_MODEL_LENGTH] = (
                    <uint64_t> view.profiles[profile_index].model_length
                )
                generation_metrics[reason_base + GENERATION_TARGET_COUNT] = (
                    self._sequence_count
                )
                generation_metrics[
                    reason_base + GENERATION_TARGET_RESIDUES
                ] = total_target_residues
                if (
                    view.profiles[profile_index].model_length > 0
                    and total_target_residues > (<uint64_t> -1) // (
                        <uint64_t> view.profiles[profile_index].model_length
                    )
                ):
                    raise OverflowError("F1 logical cell count overflows uint64")
                generation_metrics[
                    reason_base + GENERATION_F1_LOGICAL_CELLS
                ] = total_target_residues * (
                    <uint64_t> view.profiles[profile_index].model_length
                )

        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()
        host_attested = plan7_bias_host_environment_attested() == 1
        if sealed_domain_journal:
            error[0] = 0
            with nogil:
                status = plan7_ssv_sequence_batch_compact_postfilter_f2(
                    self._batch,
                    view.profiles,
                    view.m_mu,
                    view.m_lambda,
                    view.v_mu,
                    view.v_lambda,
                    profile_count,
                    f2,
                    &f2_resident_view,
                    error,
                    sizeof(error),
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            _accumulate_postfilter_f2_statistics(
                &f2_resident_view.statistics
            )
            use_resident_f2 = f2_resident_view.supported == 1
            if (
                use_resident_f2
                and f2_resident_view.source_count != record_count
            ):
                raise RuntimeError("resident F2 source count changed")
        candidate_offsets.reserve(profile_count + 1)
        candidate_offsets.push_back(0)
        for profile_index in range(profile_count):
            start = <size_t> postfilter_offsets[profile_index]
            stop = <size_t> postfilter_offsets[profile_index + 1]
            if start > stop or stop > record_count:
                raise ValueError("post-filter row offsets are not monotone")
            if collect_generation_telemetry:
                reason_base = profile_index * GENERATION_METRIC_COUNT
                if stop - start > self._sequence_count:
                    raise RuntimeError("F1 candidate count exceeds target count")
                generation_metrics[
                    reason_base + GENERATION_F1_CANDIDATE_COUNT
                ] = stop - start
                generation_metrics[
                    reason_base + GENERATION_F1_REJECT_COUNT
                ] = self._sequence_count - (stop - start)
            previous = 0
            have_previous = False
            for cursor in range(start, stop):
                resident_f2_pass = False
                if use_resident_f2:
                    if resident_f2_cursor < f2_resident_view.selected_count:
                        resident_f2_source = (
                            f2_resident_view.host_selected_sources[
                                resident_f2_cursor
                            ]
                        )
                        if resident_f2_source < cursor:
                            raise RuntimeError(
                                "resident F2 sources are not increasing"
                            )
                        resident_f2_pass = resident_f2_source == cursor
                memcpy(
                    &record,
                    &postfilter_records[cursor * sizeof(plan7_postfilter_result)],
                    sizeof(plan7_postfilter_result),
                )
                if record.sequence_index >= self._sequence_count:
                    raise IndexError("post-filter result sequence index out of range")
                if have_previous and record.sequence_index <= previous:
                    raise ValueError(
                        "post-filter result sequence indexes are not increasing"
                    )
                previous = record.sequence_index
                have_previous = True
                if direct_sparse_v3:
                    if record.action == PLAN7_BIAS_CPU_REQUIRED:
                        direct_decision = _direct_v3_decision(
                            PLAN7_CONTINUATION_V3_CPU_REQUIRED,
                            PLAN7_CONTINUATION_V3_FULL_PIPELINE,
                        )
                    elif isnan(record.filtersc):
                        direct_decision = _direct_v3_decision(
                            PLAN7_CONTINUATION_V3_RAW_F1_REJECT, 0
                        )
                    elif record.action == PLAN7_BIAS_DEFINITE_REJECT:
                        direct_decision = _direct_v3_decision(
                            PLAN7_CONTINUATION_V3_BIAS_REJECT, 0
                        )
                    else:
                        direct_decision = 0xff
                if collect_generation_telemetry:
                    reason16 = postfilter_reason_view[cursor]
                    _count_postfilter_reason(
                        postfilter_reason_counts,
                        profile_index * GENERATION_POSTFILTER_REASON_COUNT,
                        reason16,
                    )
                    sequence_length = self._lengths[record.sequence_index]
                    if (
                        view.profiles[profile_index].model_length > 0
                        and sequence_length > (<uint64_t> -1) // (
                            <uint64_t> view.profiles[profile_index].model_length
                        )
                    ):
                        raise OverflowError(
                            "post-filter logical cell count overflows uint64"
                        )
                    postfilter_base_cells = sequence_length * (
                        <uint64_t> view.profiles[profile_index].model_length
                    )
                    postfilter_execution_count = _postfilter_execution_count(
                        reason16
                    )
                    if (
                        postfilter_execution_count != 0
                        and postfilter_base_cells > (<uint64_t> -1) // (
                            postfilter_execution_count
                        )
                    ):
                        raise OverflowError(
                            "post-filter execution cell product overflows uint64"
                        )
                    logical_cells = (
                        postfilter_base_cells * postfilter_execution_count
                    )
                    if logical_cells > (<uint64_t> -1) - generation_metrics[
                        reason_base + GENERATION_POSTFILTER_LOGICAL_CELLS
                    ]:
                        raise OverflowError(
                            "post-filter logical cell total overflows uint64"
                        )
                    generation_metrics[
                        reason_base + GENERATION_POSTFILTER_LOGICAL_CELLS
                    ] += logical_cells
                    _count_reason_cells16(
                        postfilter_reason_cells,
                        profile_index * GENERATION_POSTFILTER_REASON_COUNT,
                        GENERATION_POSTFILTER_REASON_COUNT,
                        reason16 & <uint16_t> 0x3fff,
                        logical_cells,
                    )
                    if reason16 & PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED:
                        postfilter_reason_cells[
                            profile_index * GENERATION_POSTFILTER_REASON_COUNT + 14
                        ] += postfilter_base_cells
                    if reason16 & PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED:
                        postfilter_reason_cells[
                            profile_index * GENERATION_POSTFILTER_REASON_COUNT + 15
                        ] += postfilter_base_cells
                if not host_attested or record.action != PLAN7_BIAS_DEFINITE_PASS:
                    if resident_f2_pass:
                        raise RuntimeError(
                            "resident F2 selected an ineligible post-filter row"
                        )
                    if collect_generation_telemetry:
                        f2_reason_counts[
                            profile_index * GENERATION_F2_REASON_COUNT
                        ] += 1
                    if direct_sparse_v3:
                        if direct_decision == 0xff:
                            direct_decision = _direct_v3_decision(
                                PLAN7_CONTINUATION_V3_F2_SURVIVOR,
                                PLAN7_CONTINUATION_V3_FILTER_SCORES,
                            )
                        _direct_v3_plan_initial(
                            direct_decisions,
                            cursor,
                            direct_decision,
                            &direct_decision_terms,
                            &direct_exception_count,
                        )
                    continue
                vfsc_bits.value = record.vfsc
                if (
                    record.msv_status != PLAN7_SSV_OK
                    or not isfinite(record.filtersc)
                    or (
                        not isfinite(record.vfsc)
                        and vfsc_bits.bits != 0x7f800000
                    )
                    or not isfinite(view.profiles[profile_index].scale)
                    or view.profiles[profile_index].scale <= 0.0
                ):
                    if resident_f2_pass:
                        raise RuntimeError(
                            "resident F2 selected an invalid post-filter row"
                        )
                    if collect_generation_telemetry:
                        f2_reason_counts[
                            profile_index * GENERATION_F2_REASON_COUNT + 1
                        ] += 1
                    if direct_sparse_v3:
                        if direct_decision == 0xff:
                            direct_decision = _direct_v3_decision(
                                PLAN7_CONTINUATION_V3_F2_SURVIVOR,
                                PLAN7_CONTINUATION_V3_FILTER_SCORES,
                            )
                        _direct_v3_plan_initial(
                            direct_decisions,
                            cursor,
                            direct_decision,
                            &direct_decision_terms,
                            &direct_exception_count,
                        )
                    continue
                usc = <float> record.msv_numerator
                usc = usc / view.profiles[profile_index].scale
                usc = usc - <float> 3.0
                host_f2_pass = True
                if collect_generation_telemetry or not use_resident_f2:
                    bit_score = <float> (
                        (usc - record.filtersc) / eslCONST_LOG2
                    )
                    probability = esl_gumbel_surv(
                        bit_score,
                        view.m_mu[profile_index],
                        view.m_lambda[profile_index],
                    )
                    if probability > f2:
                        if collect_generation_telemetry:
                            f2_reason_counts[
                                profile_index * GENERATION_F2_REASON_COUNT + 2
                            ] += 1
                        bit_score = <float> (
                            (record.vfsc - record.filtersc) / eslCONST_LOG2
                        )
                        probability = esl_gumbel_surv(
                            bit_score,
                            view.v_mu[profile_index],
                            view.v_lambda[profile_index],
                        )
                        if probability > f2:
                            host_f2_pass = False
                            if collect_generation_telemetry:
                                f2_reason_counts[
                                    profile_index
                                    * GENERATION_F2_REASON_COUNT + 3
                                ] += 1
                if (
                    use_resident_f2
                    and collect_generation_telemetry
                    and resident_f2_pass != host_f2_pass
                ):
                    raise RuntimeError(
                        "resident F2 decision differs from the host oracle"
                    )
                if (
                    (use_resident_f2 and not resident_f2_pass)
                    or (not use_resident_f2 and not host_f2_pass)
                ):
                    if direct_sparse_v3:
                        _direct_v3_plan_initial(
                            direct_decisions,
                            cursor,
                            _direct_v3_decision(
                                PLAN7_CONTINUATION_V3_F2_REJECT, 0
                            ),
                            &direct_decision_terms,
                            &direct_exception_count,
                        )
                    continue
                if use_resident_f2:
                    resident_f2_cursor += 1
                if residue_offsets[record.sequence_index + 1] < (
                    residue_offsets[record.sequence_index]
                ):
                    raise ValueError("target residue prefix is not monotone")
                sequence_length = (
                    residue_offsets[record.sequence_index + 1]
                    - residue_offsets[record.sequence_index]
                )
                if sequence_length > 100000:
                    raise ValueError("Forward target exceeds HMMER's protein limit")
                if direct_sparse_v3:
                    _direct_v3_plan_initial(
                        direct_decisions,
                        cursor,
                        _direct_v3_decision(
                            PLAN7_CONTINUATION_V3_F2_SURVIVOR,
                            PLAN7_CONTINUATION_V3_FILTER_SCORES,
                        ),
                        &direct_decision_terms,
                        &direct_exception_count,
                    )
                if collect_generation_telemetry:
                    f2_reason_counts[
                        profile_index * GENERATION_F2_REASON_COUNT + 4
                    ] += 1
                    generation_metrics[
                        reason_base + GENERATION_F2_PASS_COUNT
                    ] += 1
                cpu_forward_selected = False
                forward_work_cells = 0
                if cpu_forward_route_mode != 0:
                    if (
                        view.profiles[profile_index].model_length > 0
                        and sequence_length > (<uint64_t> -1) // (
                            <uint64_t> view.profiles[profile_index].model_length
                        )
                    ):
                        raise OverflowError("Forward work cells overflow uint64")
                    forward_work_cells = sequence_length * (
                        <uint64_t> view.profiles[profile_index].model_length
                    )
                if cpu_forward_route_mode == 1:
                    cpu_forward_selected = True
                elif cpu_forward_route_mode in (2, 4):
                    cpu_forward_selected = (
                        forward_work_cells >= self._cpu_forward_min_cells
                        if cpu_forward_route_mode == 2
                        else forward_work_cells <= self._cpu_forward_max_cells
                    )
                elif cpu_forward_route_mode == 3:
                    cpu_forward_selected = (
                        sequence_length >= self._cpu_forward_min_length
                    )
                if cpu_forward_route_mode != 0:
                    if cpu_forward_selected:
                        if (
                            self._cpu_forward_selected_cells
                            > (<uint64_t> -1) - forward_work_cells
                        ):
                            raise OverflowError("CPU Forward cell census overflow")
                        self._cpu_forward_selected_count += 1
                        self._cpu_forward_selected_cells += forward_work_cells
                    else:
                        if (
                            self._gpu_forward_selected_cells
                            > (<uint64_t> -1) - forward_work_cells
                        ):
                            raise OverflowError("GPU Forward cell census overflow")
                        self._gpu_forward_selected_count += 1
                        self._gpu_forward_selected_cells += forward_work_cells
                if cpu_forward_selected:
                    continue
                candidate_indices.push_back(record.sequence_index)
                filter_scores.push_back(record.filtersc)
                uncorrected_scores.push_back(usc)
                candidate_records.push_back(record)
                candidate_profiles.push_back(<uint32_t> profile_index)
                if direct_sparse_v3:
                    candidate_postfilter_sources.push_back(cursor)
            candidate_offsets.push_back(candidate_indices.size())

        if (
            use_resident_f2
            and resident_f2_cursor != f2_resident_view.selected_count
        ):
            raise RuntimeError("resident F2 sources do not span the selection")
        # The resident F2 view authenticates the complete F2 survivor list.
        # Experimental CPU ownership deliberately removes a subset before
        # Forward, so the remaining GPU rows use the existing explicit,
        # authenticated candidate input instead of misrepresenting that view.
        if cpu_forward_route_mode != 0:
            use_resident_f2 = False

        candidate_count = candidate_indices.size()
        row_offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            row_offsets.data.as_ulonglongs[profile_index] = (
                candidate_offsets[profile_index]
            )
        expected_indices = clone(_UINT32_ARRAY_TEMPLATE, candidate_count, False)
        for cursor in range(candidate_count):
            expected_indices.data.as_uints[cursor] = candidate_indices[cursor]
        if self._generation_ledger_enabled:
            self._ledger_f2_control_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
            ledger_start_ns = _time.perf_counter_ns()
        error[0] = 0
        with nogil:
            status = plan7_profile_selection_stage_forward(
                selection._selection, &database, error, sizeof(error)
            )
        if self._generation_ledger_enabled:
            self._ledger_forward_stage_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
        if status == 0 and collect_generation_telemetry:
            for profile_index in range(profile_count):
                error[0] = 0
                with nogil:
                    status = plan7_forward_database_get_profile_snapshot(
                        database,
                        profile_index,
                        &forward_snapshots[profile_index],
                        error,
                        sizeof(error),
                    )
                if status != 0:
                    break
        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()
        if status == 0 and use_resident_f2:
            with nogil:
                status = plan7_forward_run_batch_workspace_postfilter_resident(
                    database,
                    self._batch,
                    view.identity_tokens,
                    profile_count,
                    candidate_offsets.data(),
                    candidate_indices.data(),
                    filter_scores.data(),
                    candidate_count,
                    &f2_resident_view,
                    f3,
                    gathered_byte_budget,
                    1 if collect_generation_telemetry else 0,
                    &output,
                    error,
                    sizeof(error),
                )
        elif status == 0 and collect_generation_telemetry:
            if sealed_domain_journal:
                with nogil:
                    status = (
                        plan7_forward_run_batch_workspace_resident_reason_facts(
                            database,
                            self._batch,
                            view.identity_tokens,
                            profile_count,
                            candidate_offsets.data(),
                            candidate_indices.data(),
                            filter_scores.data(),
                            candidate_count,
                            f3,
                            gathered_byte_budget,
                            &output,
                            error,
                            sizeof(error),
                        )
                    )
            else:
                with nogil:
                    status = plan7_forward_run_batch_workspace_reason_facts(
                        database,
                        self._batch,
                        view.identity_tokens,
                        profile_count,
                        candidate_offsets.data(),
                        candidate_indices.data(),
                        filter_scores.data(),
                        candidate_count,
                        f3,
                        gathered_byte_budget,
                        &output,
                        error,
                        sizeof(error),
                    )
        elif status == 0:
            if sealed_domain_journal:
                with nogil:
                    status = plan7_forward_run_batch_workspace_resident(
                        database,
                        self._batch,
                        view.identity_tokens,
                        profile_count,
                        candidate_offsets.data(),
                        candidate_indices.data(),
                        filter_scores.data(),
                        candidate_count,
                        f3,
                        gathered_byte_budget,
                        &output,
                        error,
                        sizeof(error),
                    )
            else:
                with nogil:
                    status = plan7_forward_run_batch_workspace(
                        database,
                        self._batch,
                        view.identity_tokens,
                        profile_count,
                        candidate_offsets.data(),
                        candidate_indices.data(),
                        filter_scores.data(),
                        candidate_count,
                        f3,
                        gathered_byte_budget,
                        &output,
                        error,
                        sizeof(error),
                    )
        if status != 0:
            if database != NULL:
                plan7_forward_database_destroy(&database, NULL, 0)
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)
            raise RuntimeError(error.decode("utf-8", "replace"))
        if self._generation_ledger_enabled:
            self._ledger_forward_native_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )

        if sealed_domain_journal:
            try:
                result_count = plan7_forward_output_result_count(output)
                native_results = plan7_forward_output_results(output)
                native_forward_reasons = (
                    plan7_forward_output_reason_facts(output)
                )
                native_offsets = plan7_forward_output_special_offsets(output)
                native_specials = plan7_forward_output_specials(output)
                native_provenance = plan7_forward_output_provenance(output)
                special_count = plan7_forward_output_special_count(output)
                if result_count != candidate_count:
                    raise RuntimeError("Forward result count changed")
                if (
                    native_offsets == NULL
                    or native_provenance == NULL
                    or (result_count and native_results == NULL)
                    or (special_count and native_specials == NULL)
                    or (
                        collect_generation_telemetry
                        and (
                            plan7_forward_output_reason_count(output)
                            != result_count
                            or (result_count and native_forward_reasons == NULL)
                        )
                    )
                ):
                    raise RuntimeError("Forward journal storage is incomplete")

                if collect_generation_telemetry:
                    if plan7_forward_output_contract_fallback(output) == 1:
                        forward_call_facts |= (
                            PLAN7_FORWARD_CALL_REASON_CONTRACT_FALLBACK
                        )
                    for profile_index in range(profile_count):
                        reason_base = profile_index * GENERATION_METRIC_COUNT
                        start = <size_t> candidate_offsets[profile_index]
                        stop = <size_t> candidate_offsets[profile_index + 1]
                        for source in range(start, stop):
                            sequence_length = self._lengths[
                                candidate_indices[source]
                            ]
                            logical_cells = 0
                            if forward_call_facts == 0:
                                logical_cells = sequence_length * (
                                    <uint64_t> forward_snapshots[
                                        profile_index
                                    ].model_length
                                )
                                if logical_cells > (<uint64_t> -1) - (
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_FORWARD_LOGICAL_CELLS
                                    ]
                                ):
                                    raise OverflowError(
                                        "Forward logical cell total overflows uint64"
                                    )
                                generation_metrics[
                                    reason_base
                                    + GENERATION_FORWARD_LOGICAL_CELLS
                                ] += logical_cells
                            forward_facts = 0
                            if collect_generation_telemetry:
                                forward_facts = native_forward_reasons[source]
                            # A call-wide contract fallback preinitializes
                            # conservative rows without launching Forward.
                            # Its exact source transition is the call fact;
                            # default row fields are not kernel facts.
                            if forward_call_facts == 0:
                                if native_results[source].status != PLAN7_FORWARD_OK:
                                    forward_facts |= (
                                        PLAN7_FORWARD_REASON_KERNEL_STATUS_NON_OK
                                    )
                                if sequence_length == 0:
                                    forward_facts |= PLAN7_FORWARD_REASON_TARGET_EMPTY
                                if not isfinite(native_results[source].fwdsc):
                                    forward_facts |= (
                                        PLAN7_FORWARD_REASON_FWDSC_NONFINITE
                                    )
                                if not isfinite(filter_scores[source]):
                                    forward_facts |= (
                                        PLAN7_FORWARD_REASON_FILTERSC_NONFINITE
                                    )
                                if not isfinite(
                                    forward_snapshots[profile_index].f_tau
                                ):
                                    forward_facts |= (
                                        PLAN7_FORWARD_REASON_TAU_NONFINITE
                                    )
                                if (
                                    not isfinite(
                                        forward_snapshots[profile_index].f_lambda
                                    )
                                    or forward_snapshots[
                                        profile_index
                                    ].f_lambda <= 0.0
                                ):
                                    forward_facts |= (
                                        PLAN7_FORWARD_REASON_LAMBDA_INVALID
                                    )
                            if native_results[source].action == (
                                PLAN7_FORWARD_DEFINITE_REJECT
                            ):
                                forward_facts |= PLAN7_FORWARD_REASON_F3_REJECT
                                generation_metrics[
                                    reason_base
                                    + GENERATION_FORWARD_REJECT_COUNT
                                ] += 1
                            elif native_results[source].action == (
                                PLAN7_FORWARD_DEFINITE_PASS
                            ):
                                forward_facts |= (
                                    PLAN7_FORWARD_REASON_SURVIVOR_GATHERED
                                )
                                generation_metrics[
                                    reason_base + GENERATION_FORWARD_PASS_COUNT
                                ] += 1
                            elif native_results[source].action == (
                                PLAN7_FORWARD_CPU_REQUIRED
                            ):
                                generation_metrics[
                                    reason_base + GENERATION_FORWARD_CPU_COUNT
                                ] += 1
                                if forward_call_facts == 0 and forward_facts == 0:
                                    raise RuntimeError(
                                        "Forward CPU route lacks a native source fact"
                                    )
                            else:
                                raise RuntimeError(
                                    "Forward result action is invalid"
                                )
                            _count_forward_reason(
                                forward_reason_counts,
                                profile_index
                                * GENERATION_FORWARD_REASON_COUNT,
                                forward_facts,
                            )
                            _count_reason_cells16(
                                forward_reason_cells,
                                profile_index
                                * GENERATION_FORWARD_REASON_COUNT,
                                GENERATION_FORWARD_REASON_COUNT,
                                forward_facts,
                                logical_cells,
                            )

                pass_special_offsets.push_back(0)
                journal_profile_offsets.push_back(0)
                pass_count = 0
                for profile_index in range(profile_count):
                    start = <size_t> candidate_offsets[profile_index]
                    stop = <size_t> candidate_offsets[profile_index + 1]
                    for source in range(start, stop):
                        if (
                            native_results[source].action
                            != PLAN7_FORWARD_DEFINITE_PASS
                        ):
                            continue
                        if native_offsets[source] != pass_special_offsets.back():
                            raise RuntimeError(
                                "Forward pass special rows are not contiguous"
                            )
                        domain_candidate.profile_index = candidate_profiles[source]
                        domain_candidate.sequence_index = candidate_indices[source]
                        domain_candidates.push_back(domain_candidate)
                        pass_sources.push_back(source)
                        pass_special_offsets.push_back(native_offsets[source + 1])
                        pass_count += 1
                    journal_profile_offsets.push_back(pass_count)
                if (
                    pass_count != native_provenance.pass_count
                    or pass_special_offsets[pass_count] != special_count
                ):
                    raise RuntimeError("Forward pass provenance count changed")
                if ga_pruning:
                    ga_whole_forward_scores.resize(pass_count)
                    for row in range(pass_count):
                        source = pass_sources[row]
                        if not isfinite(native_results[source].fwdsc):
                            raise RuntimeError(
                                "GA whole Forward score is not finite"
                            )
                        ga_whole_forward_scores[row] = (
                            native_results[source].fwdsc
                        )

                error[0] = 0
                with nogil:
                    resident_status = plan7_forward_output_get_resident_view(
                        output, &resident_view, error, sizeof(error)
                    )
                if resident_status < 0:
                    raise RuntimeError(error.decode("utf-8", "replace"))
                error[0] = 0
                if self._generation_ledger_enabled:
                    ledger_start_ns = _time.perf_counter_ns()
                if self._cpu_domain_route:
                    with nogil:
                        status = (
                            plan7_backward_domain_route_all_cpu_from_forward_output(
                                (
                                    domain_candidates.data()
                                    if pass_count
                                    else NULL
                                ),
                                pass_count,
                                pass_special_offsets.data(),
                                output,
                                rt1,
                                rt2,
                                rt3,
                                guard_band,
                                1 if collect_generation_telemetry else 0,
                                &domain_output,
                                error,
                                sizeof(error),
                            )
                        )
                elif collect_generation_telemetry and resident_status == 1:
                    with nogil:
                        status = (
                            plan7_backward_domain_run_from_forward_output_with_reason_facts(
                                database,
                                self._batch,
                                (
                                    domain_candidates.data()
                                    if pass_count
                                    else NULL
                                ),
                                pass_count,
                                pass_special_offsets.data(),
                                output,
                                rt1,
                                rt2,
                                rt3,
                                guard_band,
                                0,
                                &domain_output,
                                error,
                                sizeof(error),
                            )
                        )
                elif collect_generation_telemetry:
                    with nogil:
                        status = plan7_backward_domain_run_with_reason_facts(
                            database,
                            self._batch,
                            domain_candidates.data() if pass_count else NULL,
                            pass_count,
                            native_provenance,
                            pass_special_offsets.data(),
                            native_specials if special_count else NULL,
                            special_count,
                            rt1,
                            rt2,
                            rt3,
                            guard_band,
                            0,
                            &domain_output,
                            error,
                            sizeof(error),
                        )
                elif resident_status == 1:
                    with nogil:
                        status = plan7_backward_domain_run_from_forward_output(
                            database,
                            self._batch,
                            domain_candidates.data() if pass_count else NULL,
                            pass_count,
                            pass_special_offsets.data(),
                            output,
                            rt1,
                            rt2,
                            rt3,
                            guard_band,
                            0,
                            &domain_output,
                            error,
                            sizeof(error),
                        )
                else:
                    with nogil:
                        status = plan7_backward_domain_run(
                            database,
                            self._batch,
                            domain_candidates.data() if pass_count else NULL,
                            pass_count,
                            native_provenance,
                            pass_special_offsets.data(),
                            native_specials if special_count else NULL,
                            special_count,
                            rt1,
                            rt2,
                            rt3,
                            guard_band,
                            0,
                            &domain_output,
                            error,
                            sizeof(error),
                        )
                if status != 0:
                    raise RuntimeError(error.decode("utf-8", "replace"))
                if self._generation_ledger_enabled:
                    self._ledger_backward_native_ns += (
                        _time.perf_counter_ns() - ledger_start_ns
                    )
                if collect_generation_telemetry:
                    reason_count = (
                        plan7_backward_domain_output_reason_count(
                            domain_output
                        )
                    )
                    native_domain_reasons = (
                        plan7_backward_domain_output_reason_facts(
                            domain_output
                        )
                    )
                    native_domain_results = (
                        plan7_backward_domain_output_results(domain_output)
                    )
                    native_domain_region_offsets = (
                        plan7_backward_domain_output_region_offsets(
                            domain_output
                        )
                    )
                    native_domain_regions = (
                        plan7_backward_domain_output_regions(domain_output)
                    )
                    if (
                        reason_count != pass_count
                        or (pass_count and native_domain_reasons == NULL)
                        or (pass_count and native_domain_results == NULL)
                        or native_domain_region_offsets == NULL
                    ):
                        raise RuntimeError(
                            "Backward/domain reason storage is incomplete"
                        )
                    backward_preflight_mask = (
                        PLAN7_BACKWARD_DOMAIN_REASON_TARGET_EMPTY
                        | PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SPECIAL_NONFINITE
                        | PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SCALE_INVALID
                        | PLAN7_BACKWARD_DOMAIN_REASON_HOST_FLOAT_ENV_INVALID
                        | PLAN7_BACKWARD_DOMAIN_REASON_MODE_OR_NJ_UNSUPPORTED
                        | PLAN7_BACKWARD_DOMAIN_REASON_WORK_CAP
                        | PLAN7_BACKWARD_DOMAIN_REASON_WORKSPACE_CAP
                    )
                    for row in range(pass_count):
                        profile_index = domain_candidates[row].profile_index
                        reason_base = profile_index * GENERATION_METRIC_COUNT
                        reason32 = native_domain_reasons[row]
                        if reason32 & <uint32_t> 0xfffc0000:
                            raise RuntimeError(
                                "Backward/domain reason facts contain unknown bits"
                            )
                        logical_cells = 0
                        generation_metrics[
                            reason_base + GENERATION_JOURNAL_ROW_COUNT
                        ] += 1
                        region = (
                            native_domain_region_offsets[row + 1]
                            - native_domain_region_offsets[row]
                        )
                        generation_metrics[
                            reason_base + GENERATION_JOURNAL_REGION_COUNT
                        ] += region
                        if native_domain_results[row].route == (
                            PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED
                        ):
                            if (
                                not (
                                    reason32
                                    & PLAN7_BACKWARD_DOMAIN_REASON_FINAL_CPU_REQUIRED
                                )
                                or not (reason32 & <uint32_t> 0x0001ffff)
                            ):
                                raise RuntimeError(
                                    "Backward/domain CPU route lacks source facts"
                                )
                            generation_metrics[
                                reason_base + GENERATION_BACKWARD_CPU_COUNT
                            ] += 1
                        elif native_domain_results[row].route == (
                            PLAN7_BACKWARD_DOMAIN_NO_REGIONS
                        ):
                            if not (
                                reason32
                                & PLAN7_BACKWARD_DOMAIN_REASON_NO_REGIONS
                            ):
                                raise RuntimeError(
                                    "Backward/domain no-region route lacks source fact"
                                )
                            generation_metrics[
                                reason_base
                                + GENERATION_BACKWARD_NO_REGION_COUNT
                            ] += 1
                        elif native_domain_results[row].route == (
                            PLAN7_BACKWARD_DOMAIN_SIMPLE
                        ):
                            if not (
                                reason32 & PLAN7_BACKWARD_DOMAIN_REASON_SIMPLE
                            ):
                                raise RuntimeError(
                                    "Backward/domain SIMPLE route lacks source fact"
                                )
                            generation_metrics[
                                reason_base + GENERATION_BACKWARD_SIMPLE_COUNT
                            ] += 1
                        if not (reason32 & backward_preflight_mask):
                            sequence_length = self._lengths[
                                domain_candidates[row].sequence_index
                            ]
                            logical_cells = sequence_length * (
                                <uint64_t> forward_snapshots[
                                    profile_index
                                ].model_length
                            )
                            if logical_cells > (<uint64_t> -1) - (
                                generation_metrics[
                                    reason_base
                                    + GENERATION_BACKWARD_LOGICAL_CELLS
                                ]
                            ):
                                raise OverflowError(
                                    "Backward logical cell total overflows uint64"
                                )
                            generation_metrics[
                                reason_base
                                + GENERATION_BACKWARD_LOGICAL_CELLS
                            ] += logical_cells
                        _count_backward_reason(
                            backward_reason_counts,
                            profile_index
                            * GENERATION_BACKWARD_REASON_COUNT,
                            reason32,
                        )
                        _count_reason_cells32(
                            backward_reason_cells,
                            profile_index
                            * GENERATION_BACKWARD_REASON_COUNT,
                            GENERATION_BACKWARD_REASON_COUNT,
                            reason32,
                            logical_cells,
                        )
                if (
                    rescore_simple_diagnostic
                    or generation_tail_fingerprint != 0
                ) and not self._cpu_rescore_route:
                    if rescore_test_fault != 0:
                        error[0] = 0
                        with nogil:
                            status = (
                                plan7_backward_domain_output_apply_test_fault(
                                    domain_output,
                                    rescore_test_fault,
                                    error,
                                    sizeof(error),
                                )
                            )
                        if status != 0:
                            raise RuntimeError(
                                error.decode("utf-8", "replace")
                            )
                    error[0] = 0
                    if self._generation_ledger_enabled:
                        ledger_start_ns = _time.perf_counter_ns()
                    if ga_pruning and collect_generation_telemetry:
                        with nogil:
                            status = (
                                plan7_domain_rescore_run_ga_with_reason_facts(
                                    database,
                                    self._batch,
                                    domain_output,
                                    (
                                        ga_whole_forward_scores.data()
                                        if pass_count
                                        else NULL
                                    ),
                                    pass_count,
                                    (
                                        &ga_target_cutoff_view[0]
                                        if profile_count
                                        else NULL
                                    ),
                                    profile_count,
                                    rescore_compact_byte_budget,
                                    rescore_matrix_byte_budget,
                                    rescore_trace_byte_budget,
                                    &rescore_output,
                                    error,
                                    sizeof(error),
                                )
                            )
                    elif ga_pruning:
                        with nogil:
                            status = plan7_domain_rescore_run_ga(
                                database,
                                self._batch,
                                domain_output,
                                (
                                    ga_whole_forward_scores.data()
                                    if pass_count
                                    else NULL
                                ),
                                pass_count,
                                (
                                    &ga_target_cutoff_view[0]
                                    if profile_count
                                    else NULL
                                ),
                                profile_count,
                                rescore_compact_byte_budget,
                                rescore_matrix_byte_budget,
                                rescore_trace_byte_budget,
                                &rescore_output,
                                error,
                                sizeof(error),
                            )
                    elif collect_generation_telemetry:
                        with nogil:
                            status = (
                                plan7_domain_rescore_run_with_reason_facts(
                                    database,
                                    self._batch,
                                    domain_output,
                                    rescore_compact_byte_budget,
                                    rescore_matrix_byte_budget,
                                    rescore_trace_byte_budget,
                                    &rescore_output,
                                    error,
                                    sizeof(error),
                                )
                            )
                    else:
                        with nogil:
                            status = plan7_domain_rescore_run(
                                database,
                                self._batch,
                                domain_output,
                                rescore_compact_byte_budget,
                                rescore_matrix_byte_budget,
                                rescore_trace_byte_budget,
                                &rescore_output,
                                error,
                                sizeof(error),
                            )
                    if status != 0:
                        raise RuntimeError(error.decode("utf-8", "replace"))
                    if self._generation_ledger_enabled:
                        self._ledger_rescore_native_ns += (
                            _time.perf_counter_ns() - ledger_start_ns
                        )
                    if collect_generation_telemetry:
                        reason_count = (
                            plan7_domain_rescore_output_reason_count(
                                rescore_output
                            )
                        )
                        native_rescore_reasons = (
                            plan7_domain_rescore_output_reason_facts(
                                rescore_output
                            )
                        )
                        result_count = (
                            plan7_domain_rescore_output_result_count(
                                rescore_output
                            )
                        )
                        native_rescore_results = (
                            plan7_domain_rescore_output_results(rescore_output)
                        )
                        if (
                            (reason_count and native_rescore_reasons == NULL)
                            or (result_count and native_rescore_results == NULL)
                            or (
                                result_count != 0
                                and result_count != reason_count
                            )
                        ):
                            raise RuntimeError(
                                "isolated-domain reason storage is incomplete"
                            )
                        if result_count:
                            for region in range(result_count):
                                profile_index = (
                                    native_rescore_results[
                                        region
                                    ].profile_index
                                )
                                if profile_index >= profile_count:
                                    raise RuntimeError(
                                        "isolated-domain reason profile changed"
                                    )
                                reason_base = (
                                    profile_index * GENERATION_METRIC_COUNT
                                )
                                reason32 = native_rescore_reasons[region]
                                if reason32 & <uint32_t> 0xfc000000:
                                    raise RuntimeError(
                                        "rescore reason facts contain unknown bits"
                                    )
                                logical_cells = 0
                                _count_rescore_reason(
                                    rescore_reason_counts,
                                    profile_index
                                    * GENERATION_RESCORE_REASON_COUNT,
                                    reason32,
                                )
                                generation_metrics[
                                    reason_base
                                    + GENERATION_RESCORE_REGION_COUNT
                                ] += 1
                                if native_rescore_results[region].action == (
                                    PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                                ):
                                    if not (
                                        reason32
                                        & PLAN7_DOMAIN_RESCORE_REASON_DEVICE_RESULT
                                    ):
                                        raise RuntimeError(
                                            "rescore device route lacks source fact"
                                        )
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_RESCORE_DEVICE_COUNT
                                    ] += 1
                                elif native_rescore_results[region].action == (
                                    PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
                                ):
                                    if (
                                        not (
                                            reason32
                                            & PLAN7_DOMAIN_RESCORE_REASON_FINAL_CPU_REQUIRED
                                        )
                                        or not (reason32 & <uint32_t> 0x00ffffff)
                                    ):
                                        raise RuntimeError(
                                            "rescore CPU route lacks source facts"
                                        )
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_RESCORE_CPU_COUNT
                                    ] += 1
                                elif native_rescore_results[region].action == (
                                    PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT
                                ):
                                    if not (
                                        reason32
                                        & PLAN7_DOMAIN_RESCORE_REASON_CERTIFIED_GA_REJECT
                                    ):
                                        raise RuntimeError(
                                            "rescore GA reject lacks source fact"
                                        )
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_RESCORE_CERTIFIED_GA_COUNT
                                    ] += 1
                                else:
                                    raise RuntimeError(
                                        "rescore result action is invalid"
                                    )
                                if _rescore_reason_admitted_work(reason32):
                                    sequence_length = (
                                        native_rescore_results[
                                            region
                                        ].envelope_end
                                        - native_rescore_results[
                                            region
                                        ].envelope_begin
                                        + 1
                                    )
                                    logical_cells = sequence_length * (
                                        <uint64_t> forward_snapshots[
                                            profile_index
                                        ].model_length
                                    )
                                    if logical_cells > (<uint64_t> -1) - (
                                        generation_metrics[
                                            reason_base
                                            + GENERATION_RESCORE_LOGICAL_CELLS
                                        ]
                                    ):
                                        raise OverflowError(
                                            "rescore logical cell total overflows uint64"
                                        )
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_RESCORE_LOGICAL_CELLS
                                    ] += logical_cells
                                _count_reason_cells32(
                                    rescore_reason_cells,
                                    profile_index
                                    * GENERATION_RESCORE_REASON_COUNT,
                                    GENERATION_RESCORE_REASON_COUNT,
                                    reason32,
                                    logical_cells,
                                )
                        elif reason_count:
                            region = 0
                            for row in range(pass_count):
                                profile_index = domain_candidates[
                                    row
                                ].profile_index
                                start = <size_t> native_domain_region_offsets[row]
                                stop = <size_t> native_domain_region_offsets[
                                    row + 1
                                ]
                                while region < stop:
                                    reason32 = native_rescore_reasons[region]
                                    if reason32 & <uint32_t> 0xfc000000:
                                        raise RuntimeError(
                                            "rescore reason facts contain unknown bits"
                                        )
                                    _count_rescore_reason(
                                        rescore_reason_counts,
                                        profile_index
                                        * GENERATION_RESCORE_REASON_COUNT,
                                        reason32,
                                    )
                                    _count_reason_cells32(
                                        rescore_reason_cells,
                                        profile_index
                                        * GENERATION_RESCORE_REASON_COUNT,
                                        GENERATION_RESCORE_REASON_COUNT,
                                        reason32,
                                        0,
                                    )
                                    reason_base = (
                                        profile_index
                                        * GENERATION_METRIC_COUNT
                                    )
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_RESCORE_REGION_COUNT
                                    ] += 1
                                    generation_metrics[
                                        reason_base
                                        + GENERATION_RESCORE_CPU_COUNT
                                    ] += 1
                                    region += 1
                            if region != reason_count:
                                raise RuntimeError(
                                    "global rescore reason attribution changed"
                                )
                    if rescore_simple_diagnostic:
                        upstream_payload = (
                            _backward_domain_route_payload_from_output(
                                domain_output
                            )
                        )
                        rescore_payload = _domain_rescore_payload_from_output(
                            rescore_output
                        )
                        rescore_payload = rescore_payload + (upstream_payload,)
                if direct_sparse_v3:
                    if candidate_postfilter_sources.size() != candidate_count:
                        raise RuntimeError(
                            "direct v3 candidate source mapping changed"
                        )
                    for source in range(candidate_count):
                        direct_postfilter_source = (
                            candidate_postfilter_sources[source]
                        )
                        if native_results[source].action == (
                            PLAN7_FORWARD_DEFINITE_REJECT
                        ):
                            if direct_exception_count == 0:
                                raise RuntimeError(
                                    "direct v3 exception count underflow"
                                )
                            _direct_v3_plan_replace(
                                direct_decisions,
                                direct_postfilter_source,
                                _direct_v3_decision(
                                    PLAN7_CONTINUATION_V3_F3_REJECT, 0
                                ),
                                &direct_decision_terms,
                            )
                            direct_exception_count -= 1
                        elif native_results[source].action not in (
                            PLAN7_FORWARD_CPU_REQUIRED,
                            PLAN7_FORWARD_DEFINITE_PASS,
                        ):
                            raise RuntimeError(
                                "direct v3 Forward action is invalid"
                            )

                    native_domain_results = (
                        plan7_backward_domain_output_results(domain_output)
                    )
                    native_domain_region_offsets = (
                        plan7_backward_domain_output_region_offsets(
                            domain_output
                        )
                    )
                    if (
                        (pass_count and native_domain_results == NULL)
                        or native_domain_region_offsets == NULL
                    ):
                        raise RuntimeError(
                            "direct v3 domain decision source is incomplete"
                        )
                    direct_compact_source_count = 0
                    native_rescore_results = NULL
                    native_rescore_trace_offsets = NULL
                    native_rescore_statistics = NULL
                    if rescore_output != NULL:
                        direct_compact_source_count = (
                            plan7_domain_rescore_output_result_count(
                                rescore_output
                            )
                        )
                        native_rescore_results = (
                            plan7_domain_rescore_output_results(rescore_output)
                        )
                        native_rescore_trace_offsets = (
                            plan7_domain_rescore_output_trace_offsets(
                                rescore_output
                            )
                        )
                        native_rescore_statistics = (
                            plan7_domain_rescore_output_statistics(
                                rescore_output
                            )
                        )
                        if (
                            native_rescore_statistics == NULL
                            or native_rescore_trace_offsets == NULL
                            or (
                                direct_compact_source_count
                                and native_rescore_results == NULL
                            )
                        ):
                            raise RuntimeError(
                                "direct v3 rescore decision source is incomplete"
                            )

                    for row in range(pass_count):
                        source = pass_sources[row]
                        if source >= candidate_count:
                            raise RuntimeError(
                                "direct v3 pass source index changed"
                            )
                        direct_postfilter_source = (
                            candidate_postfilter_sources[source]
                        )
                        if native_results[source].action != (
                            PLAN7_FORWARD_DEFINITE_PASS
                        ):
                            raise RuntimeError(
                                "direct v3 domain source is not a Forward pass"
                            )
                        direct_region_begin = (
                            <size_t> native_domain_region_offsets[row]
                        )
                        direct_region_end = (
                            <size_t> native_domain_region_offsets[row + 1]
                        )
                        if direct_region_begin > direct_region_end:
                            raise RuntimeError(
                                "direct v3 domain region mapping changed"
                            )
                        if (
                            native_domain_results[row].route
                            == PLAN7_BACKWARD_DOMAIN_SIMPLE
                        ):
                            if native_domain_results[row].region_count != (
                                direct_region_end - direct_region_begin
                            ):
                                raise RuntimeError(
                                    "direct v3 SIMPLE region mapping changed"
                                )
                        elif direct_region_begin != direct_region_end:
                            raise RuntimeError(
                                "direct v3 non-SIMPLE region mapping changed"
                            )

                        direct_compact_route = (
                            PLAN7_CONTINUATION_COMPACT_NONE
                        )
                        if direct_compact_source_count:
                            if (
                                direct_region_end
                                > direct_compact_source_count
                            ):
                                raise RuntimeError(
                                    "direct v3 compact row mapping changed"
                                )
                            if direct_region_begin != direct_region_end:
                                if native_rescore_results[
                                    direct_region_begin
                                ].action == PLAN7_DOMAIN_RESCORE_DEVICE_RESULT:
                                    direct_compact_route = (
                                        PLAN7_CONTINUATION_COMPACT_DEVICE
                                    )
                                elif native_rescore_results[
                                    direct_region_begin
                                ].action == PLAN7_DOMAIN_RESCORE_CPU_REQUIRED:
                                    direct_compact_route = (
                                        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                                    )
                                elif native_rescore_results[
                                    direct_region_begin
                                ].action == (
                                    PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT
                                ):
                                    direct_compact_route = (
                                        PLAN7_CONTINUATION_COMPACT_NONE
                                    )
                                else:
                                    raise RuntimeError(
                                        "direct v3 compact action is invalid"
                                    )
                        elif (
                            rescore_output != NULL
                            and native_rescore_statistics.global_cpu_fallback_count
                            != 0
                            and direct_region_begin != direct_region_end
                        ):
                            direct_compact_route = (
                                PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                            )

                        direct_domain_safe = (
                            native_domain_results[row].status
                            == PLAN7_BACKWARD_DOMAIN_OK
                            and native_domain_results[row].route in (
                                PLAN7_BACKWARD_DOMAIN_NO_REGIONS,
                                PLAN7_BACKWARD_DOMAIN_SIMPLE,
                            )
                            and not native_domain_results[row].has_own_scales
                            and native_domain_results[row].uncertain_count == 0
                            and native_domain_results[row].multidomain_count == 0
                        )
                        direct_no_region = (
                            direct_domain_safe
                            and native_domain_results[row].route
                            == PLAN7_BACKWARD_DOMAIN_NO_REGIONS
                            and direct_region_begin == direct_region_end
                            and direct_compact_route
                            == PLAN7_CONTINUATION_COMPACT_NONE
                        )
                        direct_ga_reject = (
                            ga_pruning
                            and direct_domain_safe
                            and native_domain_results[row].route
                            == PLAN7_BACKWARD_DOMAIN_SIMPLE
                            and direct_region_begin < direct_region_end
                            and native_rescore_results != NULL
                            and native_rescore_results[
                                direct_region_begin
                            ].action
                            == PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT
                        )
                        if direct_no_region or direct_ga_reject:
                            if direct_exception_count == 0:
                                raise RuntimeError(
                                    "direct v3 exception count underflow"
                                )
                            direct_decision = _direct_v3_decision(
                                PLAN7_CONTINUATION_V3_DOMAIN_NO_REGIONS, 0
                            )
                            direct_exception_count -= 1
                        elif direct_domain_safe:
                            if (
                                native_domain_results[row].route
                                == PLAN7_BACKWARD_DOMAIN_SIMPLE
                                and direct_compact_route
                                == PLAN7_CONTINUATION_COMPACT_DEVICE
                                and direct_region_begin < direct_region_end
                                and generation_tail_fingerprint != 0
                            ):
                                direct_decision = _direct_v3_decision(
                                    PLAN7_CONTINUATION_V3_DOMAIN_COMPACT,
                                    PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                                )
                                direct_payload_delta = (
                                    direct_region_end - direct_region_begin
                                )
                                if direct_payload_delta > (
                                    (<uint64_t> -1)
                                    - direct_compact_result_count
                                ):
                                    raise OverflowError(
                                        "direct v3 compact count overflow"
                                    )
                                direct_compact_result_count += (
                                    direct_payload_delta
                                )
                                if direct_payload_delta > (
                                    (<uint64_t> -1)
                                    // PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                                ):
                                    raise OverflowError(
                                        "direct v3 compact null2 overflow"
                                    )
                                direct_payload_delta *= (
                                    PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                                )
                                if direct_payload_delta > (
                                    (<uint64_t> -1)
                                    - direct_compact_null2_count
                                ):
                                    raise OverflowError(
                                        "direct v3 compact null2 overflow"
                                    )
                                direct_compact_null2_count += (
                                    direct_payload_delta
                                )
                                direct_trace_begin = (
                                    <size_t> native_rescore_trace_offsets[
                                        direct_region_begin
                                    ]
                                )
                                direct_trace_end = (
                                    <size_t> native_rescore_trace_offsets[
                                        direct_region_end
                                    ]
                                )
                                if direct_trace_begin > direct_trace_end:
                                    raise RuntimeError(
                                        "direct v3 compact traces are not monotone"
                                    )
                                direct_payload_delta = (
                                    direct_trace_end - direct_trace_begin
                                )
                                if direct_payload_delta > (
                                    (<uint64_t> -1)
                                    - direct_compact_trace_count
                                ):
                                    raise OverflowError(
                                        "direct v3 compact trace overflow"
                                    )
                                direct_compact_trace_count += (
                                    direct_payload_delta
                                )
                            else:
                                direct_decision = _direct_v3_decision(
                                    PLAN7_CONTINUATION_V3_DOMAIN_SIMPLE,
                                    PLAN7_CONTINUATION_V3_SIMPLE_REGIONS,
                                )
                            direct_payload_delta = (
                                direct_region_end - direct_region_begin
                            )
                            if direct_payload_delta > (
                                (<uint64_t> -1) - direct_region_count
                            ):
                                raise OverflowError(
                                    "direct v3 region count overflow"
                                )
                            direct_region_count += direct_payload_delta
                        else:
                            direct_decision = _direct_v3_decision(
                                PLAN7_CONTINUATION_V3_DOMAIN_CPU_REQUIRED,
                                PLAN7_CONTINUATION_V3_FORWARD_SCORES,
                            )

                        if direct_decision >> 4 in (
                            PLAN7_CONTINUATION_V3_FORWARD_SCORES,
                            PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                        ):
                            if native_offsets[source] > native_offsets[source + 1]:
                                raise RuntimeError(
                                    "direct v3 special offsets are not monotone"
                                )
                            direct_payload_delta = (
                                native_offsets[source + 1]
                                - native_offsets[source]
                            )
                            if direct_payload_delta > (
                                (<uint64_t> -1) - direct_special_count
                            ):
                                raise OverflowError(
                                    "direct v3 special count overflow"
                                )
                            direct_special_count += direct_payload_delta
                        _direct_v3_plan_replace(
                            direct_decisions,
                            direct_postfilter_source,
                            direct_decision,
                            &direct_decision_terms,
                        )
                profile_fingerprint_storage = b"".join(selection._fingerprints)
                profile_fingerprint_view = profile_fingerprint_storage
                sequence_fingerprint_view = self._content_fingerprint
                journal_capsule = _build_continuation_journal_capsule(
                    &view,
                    (
                        &profile_fingerprint_view[0]
                        if profile_fingerprint_view.shape[0]
                        else NULL
                    ),
                    &sequence_fingerprint_view[0],
                    (
                        <const plan7_postfilter_result *> &postfilter_records[0]
                        if record_count
                        else NULL
                    ),
                    record_count,
                    &postfilter_offsets[0],
                    candidate_records.data() if candidate_count else NULL,
                    uncorrected_scores.data() if candidate_count else NULL,
                    candidate_profiles.data() if candidate_count else NULL,
                    pass_sources.data() if pass_count else NULL,
                    candidate_offsets.data(),
                    journal_profile_offsets.data(),
                    pass_count,
                    pass_special_offsets.data(),
                    output,
                    domain_output,
                    (
                        rescore_output
                        if generation_tail_fingerprint != 0
                        else NULL
                    ),
                    generation_tail_fingerprint,
                    generation_f1,
                    f2,
                    f3,
                    generation_bias_filter,
                    rt1,
                    rt2,
                    rt3,
                    guard_band,
                    &journal_total_bytes,
                    postfilter_owner,
                    direct_sparse_v3,
                    direct_decision_plan,
                    direct_decision_terms,
                    direct_exception_count,
                    direct_special_count,
                    direct_region_count,
                    direct_compact_result_count,
                    direct_compact_trace_count,
                    direct_compact_null2_count,
                )
                native_statistics = plan7_forward_output_statistics(output)
                native_residency_statistics = (
                    plan7_forward_output_residency_statistics(output)
                )
                native_input_residency_statistics = (
                    plan7_forward_output_input_residency_statistics(output)
                )
                native_domain_statistics = (
                    plan7_backward_domain_output_statistics(domain_output)
                )
                native_domain_residency_statistics = (
                    plan7_backward_domain_output_residency_statistics(
                        domain_output
                    )
                )
                native_rescore_statistics = (
                    plan7_domain_rescore_output_statistics(rescore_output)
                    if rescore_output != NULL
                    else NULL
                )
                native_rescore_residency_statistics = (
                    plan7_domain_rescore_output_residency_statistics(
                        rescore_output
                    )
                    if rescore_output != NULL
                    else NULL
                )
                if (
                    native_statistics == NULL
                    or native_residency_statistics == NULL
                    or native_input_residency_statistics == NULL
                    or native_domain_statistics == NULL
                    or native_domain_residency_statistics == NULL
                    or (
                        rescore_output != NULL
                        and (
                            native_rescore_statistics == NULL
                            or native_rescore_residency_statistics == NULL
                        )
                    )
                ):
                    raise RuntimeError(
                        "sealed native stage timing storage is incomplete"
                    )
                _accumulate_forward_backward_residency_statistics(
                    native_residency_statistics,
                    native_domain_residency_statistics,
                    native_rescore_residency_statistics,
                )
                _accumulate_forward_input_residency_statistics(
                    native_input_residency_statistics
                )
                _accumulate_forward_f3_device_statistics(
                    plan7_forward_output_f3_device_statistics(output)
                )
                _accumulate_forward_subwarp_statistics(
                    plan7_forward_output_subwarp_statistics(output)
                )
                sealed_stage_timings = (
                    SEALED_STAGE_TIMING_SCHEMA_VERSION,
                    (
                        plan7_forward_database_pack_milliseconds(database),
                        plan7_forward_database_upload_milliseconds(database),
                        plan7_forward_output_upload_milliseconds(output),
                        native_statistics.kernel_milliseconds,
                        native_statistics.classification_milliseconds,
                        native_statistics.gather_milliseconds,
                        native_statistics.download_milliseconds,
                        native_statistics.total_milliseconds,
                        plan7_forward_output_total_milliseconds(output),
                    ),
                    (
                        native_domain_statistics.kernel_milliseconds,
                        native_domain_statistics.upload_milliseconds,
                        native_domain_statistics.download_milliseconds,
                        native_domain_statistics.total_milliseconds,
                    ),
                    (
                        (
                            native_rescore_statistics.kernel_milliseconds,
                            native_rescore_statistics.upload_milliseconds,
                            native_rescore_statistics.download_milliseconds,
                            native_rescore_statistics.total_milliseconds,
                        )
                        if native_rescore_statistics != NULL
                        else None
                    ),
                )
                if collect_generation_telemetry:
                    profile_records = []
                    for profile_index in range(profile_count):
                        metric_values = []
                        reason_base = (
                            profile_index * GENERATION_METRIC_COUNT
                        )
                        for cursor in range(GENERATION_METRIC_COUNT):
                            metric_values.append(
                                generation_metrics[reason_base + cursor]
                            )
                        stage_reason_values = []
                        stage_reason_cell_values = []
                        reason_values = []
                        reason_cell_values = []
                        reason_base = (
                            profile_index
                            * GENERATION_POSTFILTER_REASON_COUNT
                        )
                        for cursor in range(
                            GENERATION_POSTFILTER_REASON_COUNT
                        ):
                            reason_values.append(
                                postfilter_reason_counts[
                                    reason_base + cursor
                                ]
                            )
                            reason_cell_values.append(
                                postfilter_reason_cells[
                                    reason_base + cursor
                                ]
                            )
                        stage_reason_values.append(tuple(reason_values))
                        stage_reason_cell_values.append(
                            tuple(reason_cell_values)
                        )
                        reason_values = []
                        reason_cell_values = []
                        reason_base = (
                            profile_index * GENERATION_F2_REASON_COUNT
                        )
                        for cursor in range(GENERATION_F2_REASON_COUNT):
                            reason_values.append(
                                f2_reason_counts[reason_base + cursor]
                            )
                            reason_cell_values.append(
                                f2_reason_cells[reason_base + cursor]
                            )
                        stage_reason_values.append(tuple(reason_values))
                        stage_reason_cell_values.append(
                            tuple(reason_cell_values)
                        )
                        reason_values = []
                        reason_cell_values = []
                        reason_base = (
                            profile_index * GENERATION_FORWARD_REASON_COUNT
                        )
                        for cursor in range(GENERATION_FORWARD_REASON_COUNT):
                            reason_values.append(
                                forward_reason_counts[reason_base + cursor]
                            )
                            reason_cell_values.append(
                                forward_reason_cells[reason_base + cursor]
                            )
                        stage_reason_values.append(tuple(reason_values))
                        stage_reason_cell_values.append(
                            tuple(reason_cell_values)
                        )
                        reason_values = []
                        reason_cell_values = []
                        reason_base = (
                            profile_index * GENERATION_BACKWARD_REASON_COUNT
                        )
                        for cursor in range(GENERATION_BACKWARD_REASON_COUNT):
                            reason_values.append(
                                backward_reason_counts[reason_base + cursor]
                            )
                            reason_cell_values.append(
                                backward_reason_cells[reason_base + cursor]
                            )
                        stage_reason_values.append(tuple(reason_values))
                        stage_reason_cell_values.append(
                            tuple(reason_cell_values)
                        )
                        reason_values = []
                        reason_cell_values = []
                        reason_base = (
                            profile_index * GENERATION_RESCORE_REASON_COUNT
                        )
                        for cursor in range(GENERATION_RESCORE_REASON_COUNT):
                            reason_values.append(
                                rescore_reason_counts[reason_base + cursor]
                            )
                            reason_cell_values.append(
                                rescore_reason_cells[reason_base + cursor]
                            )
                        stage_reason_values.append(tuple(reason_values))
                        stage_reason_cell_values.append(
                            tuple(reason_cell_values)
                        )
                        profile_records.append(
                            (
                                tuple(metric_values),
                                tuple(stage_reason_values),
                                tuple(stage_reason_cell_values),
                            )
                        )
                    native_totals = {
                        "postfilter": {
                            "candidate_count": generation_telemetry_seed[2][0],
                            "full_msv_execution_count": (
                                generation_telemetry_seed[2][1]
                            ),
                            "viterbi_execution_count": (
                                generation_telemetry_seed[2][2]
                            ),
                            "full_msv_work_cells": (
                                generation_telemetry_seed[2][3]
                            ),
                            "viterbi_work_cells": (
                                generation_telemetry_seed[2][4]
                            ),
                            "work_cells": generation_telemetry_seed[2][5],
                        },
                        "forward": {
                            "candidate_count": native_statistics.candidate_count,
                            "survivor_count": native_statistics.survivor_count,
                            "work_cells": native_statistics.work_cells,
                            "output_cap_fallback_count": (
                                native_statistics.output_cap_fallback_count
                            ),
                        },
                        "backward_domain": {
                            "candidate_count": (
                                native_domain_statistics.candidate_count
                            ),
                            "device_result_count": (
                                native_domain_statistics.device_result_count
                            ),
                            "cpu_required_count": (
                                native_domain_statistics.cpu_required_count
                            ),
                            "work_cells": native_domain_statistics.work_cells,
                        },
                        "rescore": (
                            {
                                "region_count": (
                                    native_rescore_statistics.region_count
                                ),
                                "device_result_count": (
                                    native_rescore_statistics.device_result_count
                                ),
                                "cpu_required_count": (
                                    native_rescore_statistics.cpu_required_count
                                ),
                                "certified_ga_region_count": (
                                    native_rescore_statistics
                                    .certified_ga_region_count
                                ),
                                "certified_ga_row_count": (
                                    native_rescore_statistics
                                    .certified_ga_row_count
                                ),
                                "certified_ga_skipped_work_cells": (
                                    native_rescore_statistics
                                    .certified_ga_skipped_work_cells
                                ),
                                "ga_classification_ms": (
                                    native_rescore_statistics
                                    .ga_classification_milliseconds
                                ),
                                "work_cells": (
                                    native_rescore_statistics.work_cells
                                ),
                            }
                            if native_rescore_statistics != NULL
                            else None
                        ),
                    }
                    generation_statistics = (
                        _telemetry_module.build_generation_statistics(
                            GENERATION_TELEMETRY_SCHEMA_VERSION,
                            profile_count,
                            self._sequence_count,
                            total_target_residues,
                            tuple(profile_records),
                            int(forward_call_facts),
                            int(journal_total_bytes),
                            native_totals,
                        )
                    )
            except:
                if rescore_output != NULL:
                    plan7_domain_rescore_output_destroy(
                        &rescore_output, NULL, 0
                    )
                if domain_output != NULL:
                    plan7_backward_domain_output_destroy(
                        &domain_output, NULL, 0
                    )
                if database != NULL:
                    plan7_forward_database_destroy(&database, NULL, 0)
                if output != NULL:
                    plan7_forward_output_destroy(&output, NULL, 0)
                raise

        destroy_error[0] = 0
        if database != NULL:
            with nogil:
                destroy_status = plan7_forward_database_destroy(
                    &database, destroy_error, sizeof(destroy_error)
                )
        if destroy_status != 0:
            if rescore_output != NULL:
                plan7_domain_rescore_output_destroy(
                    &rescore_output, NULL, 0
                )
            if domain_output != NULL:
                plan7_backward_domain_output_destroy(&domain_output, NULL, 0)
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)
            raise RuntimeError(destroy_error.decode("utf-8", "replace"))

        if sealed_domain_journal:
            if rescore_output != NULL:
                plan7_domain_rescore_output_destroy(
                    &rescore_output, NULL, 0
                )
            if domain_output != NULL:
                plan7_backward_domain_output_destroy(
                    &domain_output, NULL, 0
                )
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)
            if rescore_simple_diagnostic:
                return journal_capsule, rescore_payload
            if collect_generation_telemetry:
                return (
                    journal_capsule,
                    sealed_stage_timings,
                    generation_statistics,
                )
            return journal_capsule, sealed_stage_timings

        try:
            result_count = plan7_forward_output_result_count(output)
            if result_count != candidate_count:
                raise RuntimeError("Forward result count changed")
            if result_count > (<size_t> -1) // sizeof(plan7_forward_result):
                raise OverflowError("Forward result size overflows size_t")
            result_bytes = result_count * sizeof(plan7_forward_result)
            if result_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("Forward result size exceeds Python limits")
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            native_results = plan7_forward_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Forward result storage is null")
                memcpy(PyBytes_AS_STRING(records), native_results, result_bytes)

            native_offsets = plan7_forward_output_special_offsets(output)
            if native_offsets == NULL:
                raise RuntimeError("Forward special offsets are null")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
            ) - 1:
                raise OverflowError("Forward special offsets exceed Python limits")
            offset_bytes = (result_count + 1) * sizeof(uint64_t)
            offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
            memcpy(PyBytes_AS_STRING(offset_storage), native_offsets, offset_bytes)
            special_offsets = memoryview(offset_storage).cast("Q")

            special_count = plan7_forward_output_special_count(output)
            if special_count > (<size_t> -1) // sizeof(float):
                raise OverflowError("Forward special matrix size overflows size_t")
            special_bytes = special_count * sizeof(float)
            if special_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("Forward special matrix exceeds Python limits")
            special_storage = PyBytes_FromStringAndSize(NULL, special_bytes)
            native_specials = plan7_forward_output_specials(output)
            if special_bytes:
                if native_specials == NULL:
                    raise RuntimeError("Forward special matrix is null")
                memcpy(
                    PyBytes_AS_STRING(special_storage),
                    native_specials,
                    special_bytes,
                )
            specials = memoryview(special_storage).cast("f")

            native_statistics = plan7_forward_output_statistics(output)
            if native_statistics == NULL:
                raise RuntimeError("Forward statistics are null")
            statistics = {
                "generation_f3_bits": native_statistics.generation_f3_bits,
                "candidate_count": native_statistics.candidate_count,
                "survivor_count": native_statistics.survivor_count,
                "work_cells": native_statistics.work_cells,
                "dp_workspace_bytes": native_statistics.dp_workspace_bytes,
                "xmx_workspace_bytes": native_statistics.xmx_workspace_bytes,
                "gather_workspace_bytes": native_statistics.gather_workspace_bytes,
                "gathered_xmx_bytes": native_statistics.gathered_xmx_bytes,
                "output_byte_limit": native_statistics.output_byte_limit,
                "output_cap_fallback_count": (
                    native_statistics.output_cap_fallback_count
                ),
                "kernel_ms": native_statistics.kernel_milliseconds,
                "classification_ms": native_statistics.classification_milliseconds,
                "gather_ms": native_statistics.gather_milliseconds,
                "download_ms": native_statistics.download_milliseconds,
                "total_ms": native_statistics.total_milliseconds,
            }
            statistics.update(_forward_f3_device_statistics_from_output(output))
            provenance = _forward_provenance_from_output(output)
            statistics.update({
                "database_generation": provenance._value.database_generation,
                "batch_generation": provenance._value.batch_generation,
                "row_hash": provenance._value.row_hash,
                "special_hash": provenance._value.special_hash,
                "continuation_hash": provenance._value.continuation_hash,
                "pass_count": provenance._value.pass_count,
                "special_count": provenance._value.special_count,
                "_provenance": provenance,
            })
            result = (
                records,
                row_offsets,
                expected_indices,
                special_offsets,
                specials,
                statistics,
            )
            return result
        finally:
            if rescore_output != NULL:
                plan7_domain_rescore_output_destroy(
                    &rescore_output, NULL, 0
                )
            if domain_output != NULL:
                plan7_backward_domain_output_destroy(&domain_output, NULL, 0)
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)

    def forward_profile_selection_raw(
        self,
        ProfileSelection selection,
        const uint8_t[::1] postfilter_records,
        const uint64_t[::1] postfilter_offsets,
        const uint64_t[::1] residue_offsets,
        double f2,
        double f3,
        uint64_t gathered_byte_budget=PLAN7_FORWARD_MAX_GATHERED_BYTES,
    ):
        """Run diagnostic selection-aware F2/F3 without minting a seal."""
        return self._forward_profile_selection_raw(
            selection,
            postfilter_records,
            postfilter_offsets,
            residue_offsets,
            f2,
            f3,
            gathered_byte_budget,
            False,
            False,
            0,
            0,
            0,
            0,
            0,
            0.02,
            True,
            <float> 0.25,
            <float> 0.10,
            <float> 0.20,
            <float> 2.0e-4,
            None,
            None,
            False,
            None,
        )

    def _postfilter_forward_domain_selection_sealed(
        self,
        ProfileSelection selection,
        double f1,
        double f2,
        double f3,
        float guard_band=2.0e-4,
        uint64_t gathered_byte_budget=PLAN7_FORWARD_MAX_GATHERED_BYTES,
        bint rescore_simple_diagnostic=False,
        uint64_t rescore_matrix_byte_budget=0,
        uint64_t rescore_trace_byte_budget=0,
        uint64_t rescore_compact_byte_budget=0,
        int _rescore_test_fault=0,
        uint64_t generation_tail_fingerprint=0,
        bint _return_stage_timings=False,
        bint _return_generation_statistics=False,
        bint _direct_sparse_v3=False,
        object _ga_target_cutoffs=None,
    ):
        """Run the fused package-internal path and return one opaque seal.

        Underscore native entry points are trusted implementation details, not
        a security boundary against deliberate same-process private API use.
        """
        cdef object postfilter_records
        cdef object postfilter_offsets
        cdef object result
        cdef object generation_telemetry_seed = None
        cdef object postfilter_reason_facts = None
        cdef object postfilter_reason_statistics = None
        cdef carray residue_offsets
        cdef size_t index
        cdef uint64_t ledger_start_ns = 0
        cdef object sealed_bias_viterbi_policy = _os.environ.get(
            "PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP"
        )
        cdef bint sealed_bias_viterbi_skip = (
            _direct_sparse_v3
            and (
                sealed_bias_viterbi_policy == "1"
                or (
                    sealed_bias_viterbi_policy is None
                    and self._sequence_count > 65536
                )
            )
        )

        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()

        if rescore_simple_diagnostic and (
            _return_stage_timings or _return_generation_statistics
        ):
            raise ValueError(
                "sealed telemetry cannot be combined with rescore diagnostics"
            )

        if _return_generation_statistics:
            (
                postfilter_records,
                postfilter_offsets,
                postfilter_reason_facts,
                postfilter_reason_statistics,
            ) = self.postfilter_profile_selection_csr_raw(
                selection,
                f1,
                _return_reason_facts=True,
                _immutable_records=_direct_sparse_v3,
                _sealed_bias_viterbi_skip=sealed_bias_viterbi_skip,
            )
            generation_telemetry_seed = (
                GENERATION_TELEMETRY_SCHEMA_VERSION,
                postfilter_reason_facts,
                postfilter_reason_statistics,
            )
        else:
            postfilter_records, postfilter_offsets = (
                self.postfilter_profile_selection_csr_raw(
                    selection,
                    f1,
                    _immutable_records=_direct_sparse_v3,
                    _sealed_bias_viterbi_skip=sealed_bias_viterbi_skip,
                )
            )
        residue_offsets = clone(
            _UINT64_ARRAY_TEMPLATE, self._sequence_count + 1, False
        )
        residue_offsets.data.as_ulonglongs[0] = 0
        for index in range(self._sequence_count):
            if self._lengths[index] > (
                (<uint64_t> -1)
                - residue_offsets.data.as_ulonglongs[index]
            ):
                raise OverflowError("target residue prefix overflows uint64")
            residue_offsets.data.as_ulonglongs[index + 1] = (
                residue_offsets.data.as_ulonglongs[index]
                + self._lengths[index]
            )
        result = self._forward_profile_selection_raw(
            selection,
            memoryview(postfilter_records),
            memoryview(postfilter_offsets),
            memoryview(residue_offsets),
            f2,
            f3,
            gathered_byte_budget,
            True,
            rescore_simple_diagnostic,
            rescore_compact_byte_budget,
            rescore_matrix_byte_budget,
            rescore_trace_byte_budget,
            _rescore_test_fault,
            generation_tail_fingerprint,
            f1,
            True,
            <float> 0.25,
            <float> 0.10,
            <float> 0.20,
            guard_band,
            generation_telemetry_seed,
            postfilter_records,
            _direct_sparse_v3,
            _ga_target_cutoffs,
        )
        if self._generation_ledger_enabled:
            self._ledger_fused_call_count += 1
            self._ledger_fused_total_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
        if (
            rescore_simple_diagnostic
            or _return_stage_timings
            or _return_generation_statistics
        ):
            return result
        return result[0]

    cdef size_t _run_profile_selection_candidates(
        self,
        const plan7_profile_selection_view *view,
        double f1,
        bint host_candidate_expansion,
    ) except? 0:
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t words_per_profile
        cdef size_t candidate_word_count
        cdef size_t word_index
        cdef size_t sequence_index
        cdef size_t candidate_count = 0
        cdef size_t output_index
        cdef uint32_t word
        cdef unsigned bit
        cdef int status
        cdef char error[512]
        cdef plan7_ssv_f1_candidate_view compact_view
        cdef plan7_bias_candidate mapping

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        profile_count = view.profile_count
        if profile_count and (
            view.packed_scores == NULL
            or view.profiles == NULL
            or view.m_mu == NULL
            or view.m_lambda == NULL
            or view.v_mu == NULL
            or view.v_lambda == NULL
            or view.bias_templates == NULL
            or view.identity_tokens == NULL
        ):
            raise RuntimeError("profile selection storage is incomplete")
        if self._sequence_count > (<size_t> -1) - 31:
            raise OverflowError("candidate mask size overflows size_t")
        words_per_profile = (self._sequence_count + 31) // 32
        if words_per_profile and profile_count > (<size_t> -1) // words_per_profile:
            raise OverflowError("candidate mask size overflows size_t")
        candidate_word_count = profile_count * words_per_profile
        self._candidate_counts.resize(profile_count)
        if not host_candidate_expansion and candidate_word_count <= INT_MAX:
            self._candidate_words.clear()
            error[0] = 0
            if self._generation_ledger_enabled:
                ledger_start_ns = _time.perf_counter_ns()
            with nogil:
                status = plan7_ssv_sequence_batch_f1_compact_many(
                    self._batch,
                    view.packed_scores,
                    view.packed_score_count,
                    view.profiles,
                    profile_count,
                    view.m_mu,
                    view.m_lambda,
                    f1,
                    error,
                    sizeof(error),
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            if self._generation_ledger_enabled:
                self._ledger_f1_native_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
                ledger_start_ns = _time.perf_counter_ns()
            error[0] = 0
            status = plan7_ssv_sequence_batch_get_f1_candidate_view(
                self._batch, &compact_view, error, sizeof(error)
            )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            if compact_view.profile_count != profile_count:
                raise RuntimeError("device candidate profile count changed")
            candidate_count = compact_view.candidate_count
            self._candidate_offsets.resize(profile_count)
            self._candidate_indices.resize(candidate_count)
            for profile_index in range(profile_count):
                if (
                    compact_view.candidate_offsets[profile_index]
                    > compact_view.candidate_offsets[profile_index + 1]
                    or compact_view.candidate_offsets[profile_index + 1]
                    > candidate_count
                ):
                    raise RuntimeError("device candidate offsets are invalid")
                self._candidate_offsets[profile_index] = (
                    compact_view.candidate_offsets[profile_index]
                )
                self._candidate_counts[profile_index] = (
                    compact_view.candidate_offsets[profile_index + 1]
                    - compact_view.candidate_offsets[profile_index]
                )
                for output_index in range(
                    compact_view.candidate_offsets[profile_index],
                    compact_view.candidate_offsets[profile_index + 1],
                ):
                    mapping = compact_view.candidates[output_index]
                    if (
                        mapping.profile_index != profile_index
                        or mapping.sequence_index >= self._sequence_count
                    ):
                        raise RuntimeError("device candidate mapping is invalid")
                    self._candidate_indices[output_index] = mapping.sequence_index
            if (
                profile_count
                and compact_view.candidate_offsets[profile_count] != candidate_count
            ):
                raise RuntimeError("device candidate count changed")
            if self._generation_ledger_enabled:
                self._ledger_f1_candidate_mirror_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
            return profile_count

        self._candidate_words.resize(candidate_word_count)
        error[0] = 0
        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()
        with nogil:
            status = plan7_ssv_sequence_batch_f1_mask_many(
                self._batch,
                view.packed_scores,
                view.packed_score_count,
                view.profiles,
                profile_count,
                view.m_mu,
                view.m_lambda,
                f1,
                self._candidate_words.data() if candidate_word_count else NULL,
                candidate_word_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        if self._generation_ledger_enabled:
            self._ledger_f1_native_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
            ledger_start_ns = _time.perf_counter_ns()

        self._candidate_offsets.resize(profile_count)
        for profile_index in range(profile_count):
            self._candidate_counts[profile_index] = 0
            for word_index in range(words_per_profile):
                self._candidate_counts[profile_index] += plan7_popcount_u32(
                    self._candidate_words[
                        profile_index * words_per_profile + word_index
                    ]
                )
            self._candidate_offsets[profile_index] = candidate_count
            if self._candidate_counts[profile_index] > (
                (<size_t> -1) - candidate_count
            ):
                raise OverflowError("candidate count overflows size_t")
            candidate_count += self._candidate_counts[profile_index]

        self._candidate_indices.resize(candidate_count)
        for profile_index in range(profile_count):
            output_index = self._candidate_offsets[profile_index]
            for word_index in range(words_per_profile):
                word = self._candidate_words[
                    profile_index * words_per_profile + word_index
                ]
                while word:
                    bit = plan7_ctz_u32(word)
                    sequence_index = word_index * 32 + bit
                    if sequence_index >= self._sequence_count:
                        raise RuntimeError("candidate mask has trailing bits set")
                    self._candidate_indices[output_index] = <uint32_t> sequence_index
                    output_index += 1
                    word &= word - 1
            if output_index != (
                self._candidate_offsets[profile_index]
                + self._candidate_counts[profile_index]
            ):
                raise RuntimeError("candidate mask count changed")
        if self._generation_ledger_enabled:
            self._ledger_f1_candidate_mirror_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
        return profile_count

    def postfilter_profile_selection_csr_raw(
        self,
        ProfileSelection selection,
        double f1,
        bint _return_reason_facts=False,
        bint _immutable_records=False,
        bint _host_candidate_expansion=False,
        bint _sealed_bias_viterbi_skip=False,
    ):
        """Run a sealed selection without reading any live optimized profile."""
        cdef plan7_profile_selection_view view = selection._view()
        cdef plan7_viterbi_database *database = NULL
        cdef char error[512]
        cdef char destroy_error[512]
        cdef int status = 0
        cdef int destroy_status = 0
        cdef int cutoff_mode
        cdef float cutoff
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef size_t result_bytes
        cdef object records
        cdef uint8_t[::1] record_view
        cdef carray offsets
        cdef vector[uint16_t] reason_facts
        cdef plan7_postfilter_reason_statistics reason_statistics
        cdef uint16_t *reason_facts_ptr = NULL
        cdef plan7_postfilter_reason_statistics *reason_statistics_ptr = NULL
        cdef size_t reason_count = 0
        cdef bytes reason_storage
        cdef uint64_t ledger_start_ns = 0

        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if sizeof(plan7_postfilter_result) != PLAN7_POSTFILTER_RECORD_SIZE:
            raise RuntimeError("post-filter result ABI size mismatch")
        profile_count = self._run_profile_selection_candidates(
            &view, f1, _host_candidate_expansion
        )
        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()
        self._bias_profiles.resize(profile_count)
        self._bias_candidate_offsets.resize(profile_count + 1)
        if profile_count:
            memcpy(
                self._bias_profiles.data(),
                view.bias_templates,
                profile_count * sizeof(plan7_bias_profile),
            )
        for profile_index in range(profile_count):
            cutoff_mode = plan7_ssv_f1_cutoff(
                view.m_mu[profile_index],
                view.m_lambda[profile_index],
                f1,
                &cutoff,
            )
            self._bias_profiles[profile_index].cutoff_mode = cutoff_mode
            if cutoff_mode == PLAN7_F1_CUTOFF_SCORE:
                self._bias_profiles[profile_index].cutoff_bit_score = cutoff
            self._bias_candidate_offsets[profile_index] = (
                self._candidate_offsets[profile_index]
            )
        candidate_count = self._candidate_indices.size()
        self._bias_candidate_offsets[profile_count] = candidate_count
        self._postfilter_results.resize(candidate_count)
        if _return_reason_facts:
            reason_facts.resize(candidate_count)
            reason_facts_ptr = reason_facts.data()
            reason_statistics_ptr = &reason_statistics
            reason_count = candidate_count
            reason_statistics.candidate_count = 0
            reason_statistics.full_msv_execution_count = 0
            reason_statistics.viterbi_execution_count = 0
            reason_statistics.full_msv_work_cells = 0
            reason_statistics.viterbi_work_cells = 0
            reason_statistics.work_cells = 0

        if candidate_count:
            if self._generation_ledger_enabled:
                self._ledger_postfilter_host_prepare_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
                ledger_start_ns = _time.perf_counter_ns()
            error[0] = 0
            with nogil:
                status = plan7_profile_selection_stage_viterbi(
                    selection._selection, &database, error, sizeof(error)
                )
            if self._generation_ledger_enabled:
                self._ledger_viterbi_stage_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
            if status == 0:
                if self._generation_ledger_enabled:
                    ledger_start_ns = _time.perf_counter_ns()
                if _sealed_bias_viterbi_skip:
                    with nogil:
                        status = (
                            plan7_ssv_sequence_batch_postfilter_candidates_many_fixed_bias(
                                self._batch,
                                self._bias_profiles.data(),
                                profile_count,
                                self._bias_candidate_offsets.data(),
                                self._candidate_indices.data(),
                                candidate_count,
                                view.identity_tokens,
                                database,
                                self._postfilter_results.data(),
                                candidate_count,
                                reason_facts_ptr,
                                reason_count,
                                reason_statistics_ptr,
                                error,
                                sizeof(error),
                            )
                        )
                elif _return_reason_facts:
                    with nogil:
                        status = (
                            plan7_ssv_sequence_batch_postfilter_candidates_many_reason_facts(
                                self._batch,
                                self._bias_profiles.data(),
                                profile_count,
                                self._bias_candidate_offsets.data(),
                                self._candidate_indices.data(),
                                candidate_count,
                                view.identity_tokens,
                                database,
                                self._postfilter_results.data(),
                                candidate_count,
                                reason_facts.data(),
                                candidate_count,
                                &reason_statistics,
                                error,
                                sizeof(error),
                            )
                        )
                else:
                    with nogil:
                        status = plan7_ssv_sequence_batch_postfilter_candidates_many(
                            self._batch,
                            self._bias_profiles.data(),
                            profile_count,
                            self._bias_candidate_offsets.data(),
                            self._candidate_indices.data(),
                            candidate_count,
                            view.identity_tokens,
                            database,
                            self._postfilter_results.data(),
                            candidate_count,
                            error,
                            sizeof(error),
                        )
                if self._generation_ledger_enabled:
                    self._ledger_postfilter_native_ns += (
                        _time.perf_counter_ns() - ledger_start_ns
                    )
                destroy_error[0] = 0
                with nogil:
                    destroy_status = plan7_viterbi_database_destroy(
                        &database, destroy_error, sizeof(destroy_error)
                    )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            if destroy_status != 0:
                raise RuntimeError(destroy_error.decode("utf-8", "replace"))
        elif self._generation_ledger_enabled:
            self._ledger_postfilter_host_prepare_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )

        if self._generation_ledger_enabled:
            ledger_start_ns = _time.perf_counter_ns()
        if candidate_count > (<size_t> -1) // sizeof(plan7_postfilter_result):
            raise OverflowError("post-filter result size overflows size_t")
        result_bytes = candidate_count * sizeof(plan7_postfilter_result)
        if _immutable_records:
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            if result_bytes:
                memcpy(
                    PyBytes_AS_STRING(records),
                    self._postfilter_results.data(),
                    result_bytes,
                )
        else:
            records = bytearray(result_bytes)
            if result_bytes:
                record_view = records
                memcpy(
                    &record_view[0], self._postfilter_results.data(), result_bytes
                )
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._bias_candidate_offsets[profile_index]
            )
        if _return_reason_facts:
            if candidate_count > (<size_t> PY_SSIZE_T_MAX // sizeof(uint16_t)):
                raise OverflowError("post-filter reason facts exceed Python limits")
            reason_storage = PyBytes_FromStringAndSize(
                NULL, candidate_count * sizeof(uint16_t)
            )
            if candidate_count:
                memcpy(
                    PyBytes_AS_STRING(reason_storage),
                    reason_facts.data(),
                    candidate_count * sizeof(uint16_t),
                )
            if self._generation_ledger_enabled:
                self._ledger_postfilter_materialize_ns += (
                    _time.perf_counter_ns() - ledger_start_ns
                )
            return records, offsets, reason_storage, (
                reason_statistics.candidate_count,
                reason_statistics.full_msv_execution_count,
                reason_statistics.viterbi_execution_count,
                reason_statistics.full_msv_work_cells,
                reason_statistics.viterbi_work_cells,
                reason_statistics.work_cells,
            )
        if self._generation_ledger_enabled:
            self._ledger_postfilter_materialize_ns += (
                _time.perf_counter_ns() - ledger_start_ns
            )
        return records, offsets

    cdef size_t _run_postfilter_candidates_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
        source_profiles,
        ViterbiProfiles viterbi_profiles,
    ) except? 0:
        cdef char error[512]
        cdef size_t profile_count
        cdef size_t candidate_count
        cdef size_t profile_index
        cdef size_t expected_bias_bytes
        cdef int status
        cdef tuple owners = tuple(source_profiles)
        cdef vector[uintptr_t] source_pointers
        cdef OptimizedProfile source_profile
        cdef object value

        if viterbi_profiles._database == NULL:
            raise RuntimeError("Viterbi profiles are closed")
        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            False,
        )
        if plan7_viterbi_database_profile_count(
            viterbi_profiles._database
        ) != profile_count:
            raise ValueError("Viterbi and SSV profile counts differ")
        if len(owners) != profile_count:
            raise ValueError("source and SSV profile counts differ")
        source_pointers.reserve(profile_count)
        for profile_index in range(profile_count):
            value = owners[profile_index]
            if not isinstance(value, OptimizedProfile):
                raise TypeError(
                    "source profiles must be OptimizedProfile objects"
                )
            if value is not viterbi_profiles._owners[profile_index]:
                raise ValueError(
                    "source profile identity differs from Viterbi profile row"
                )
            source_profile = value
            source_pointers.push_back(<uintptr_t> source_profile._om)
        if profile_count == 0:
            if packed_bias_profiles.shape[0] != 0:
                raise ValueError("packed bias profiles have trailing bytes")
            self._bias_profiles.clear()
            self._postfilter_results.clear()
            self._bias_candidate_offsets.resize(1)
            self._bias_candidate_offsets[0] = 0
            return 0
        if profile_count > (<size_t> -1) // sizeof(plan7_bias_profile):
            raise OverflowError("packed bias profile size overflows size_t")
        expected_bias_bytes = profile_count * sizeof(plan7_bias_profile)
        if <size_t> packed_bias_profiles.shape[0] != expected_bias_bytes:
            raise ValueError("packed bias profile buffer has the wrong size")

        self._bias_profiles.resize(profile_count)
        memcpy(
            self._bias_profiles.data(),
            &packed_bias_profiles[0],
            expected_bias_bytes,
        )
        candidate_count = self._candidate_indices.size()
        self._postfilter_results.resize(candidate_count)
        self._bias_candidate_offsets.resize(profile_count + 1)
        for profile_index in range(profile_count):
            self._bias_candidate_offsets[profile_index] = (
                self._candidate_offsets[profile_index]
            )
        self._bias_candidate_offsets[profile_count] = candidate_count
        if candidate_count == 0:
            return profile_count

        error[0] = 0
        # Hold the GIL while native code validates live P7_OPROFILE snapshots;
        # PyHMMER mutation APIs must not race those private-array reads.
        status = plan7_ssv_sequence_batch_postfilter_candidates_many(
            self._batch,
            self._bias_profiles.data(),
            profile_count,
            self._bias_candidate_offsets.data(),
            self._candidate_indices.data(),
            candidate_count,
            source_pointers.data(),
            viterbi_profiles._database,
            self._postfilter_results.data(),
            candidate_count,
            error,
            sizeof(error),
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return profile_count

    def postfilter_candidates_many_csr_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
        source_profiles,
        ViterbiProfiles viterbi_profiles,
    ):
        """Return version-1 post-filter records and native uint64 row offsets."""
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef size_t result_bytes
        cdef bytearray records
        cdef uint8_t[::1] record_view
        cdef carray offsets

        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if sizeof(plan7_postfilter_result) != PLAN7_POSTFILTER_RECORD_SIZE:
            raise RuntimeError("post-filter result ABI size mismatch")
        profile_count = self._run_postfilter_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            packed_bias_profiles,
            source_profiles,
            viterbi_profiles,
        )
        candidate_count = self._postfilter_results.size()
        if candidate_count > (<size_t> -1) // sizeof(plan7_postfilter_result):
            raise OverflowError("post-filter result size overflows size_t")
        result_bytes = candidate_count * sizeof(plan7_postfilter_result)
        records = bytearray(result_bytes)
        if result_bytes:
            record_view = records
            memcpy(
                &record_view[0], self._postfilter_results.data(), result_bytes
            )
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._bias_candidate_offsets[profile_index]
            )
        return records, offsets


def filter_raw(
    const uint8_t[::1] striped_scores,
    int score_stride,
    int model_length,
    int alphabet_size,
    const uint8_t[::1] residues,
    const uint64_t[::1] offsets,
    int tbm,
    int tec,
    int base,
    int bias,
    float scale,
):
    cdef SequenceBatch batch = SequenceBatch(residues, offsets, alphabet_size)
    try:
        return batch.filter_raw(
            striped_scores,
            score_stride,
            model_length,
            alphabet_size,
            tbm,
            tec,
            base,
            bias,
            scale,
        )
    finally:
        batch.close()


def f1_decision(
    int status,
    int numerator,
    uint64_t length,
    float scale,
    float m_mu,
    float m_lambda,
    double f1,
):
    cdef double p
    cdef int action
    if not 0 <= status <= 255:
        raise ValueError("status must fit in uint8")
    if not -32768 <= numerator <= 32767:
        raise ValueError("numerator must fit in int16")
    action = plan7_ssv_f1_decision(
        <uint8_t> status,
        <int16_t> numerator,
        length,
        scale,
        m_mu,
        m_lambda,
        f1,
        &p,
    )
    return action, p


def f1_cutoff(float m_mu, float m_lambda, double f1):
    cdef float cutoff
    cdef int mode = plan7_ssv_f1_cutoff(m_mu, m_lambda, f1, &cutoff)
    return mode, cutoff if mode == PLAN7_F1_CUTOFF_SCORE else None


def f1_cutoff_decision(
    int status,
    int numerator,
    uint64_t length,
    float scale,
    int cutoff_mode,
    float cutoff_bit_score,
):
    if not 0 <= status <= 255:
        raise ValueError("status must fit in uint8")
    if not -32768 <= numerator <= 32767:
        raise ValueError("numerator must fit in int16")
    return plan7_ssv_f1_cutoff_decision(
        <uint8_t> status,
        <int16_t> numerator,
        length,
        scale,
        cutoff_mode,
        cutoff_bit_score,
    )


STATUS_OK = PLAN7_SSV_OK
STATUS_ERANGE = PLAN7_SSV_ERANGE
STATUS_ENORESULT = PLAN7_SSV_ENORESULT
STATUS_EMPTY = PLAN7_SSV_EMPTY
F1_CPU_REQUIRED = PLAN7_F1_CPU_REQUIRED
F1_DEFINITE_REJECT = PLAN7_F1_DEFINITE_REJECT
F1_CUTOFF_INVALID = PLAN7_F1_CUTOFF_INVALID
F1_CUTOFF_SCORE = PLAN7_F1_CUTOFF_SCORE
F1_CUTOFF_ALWAYS_REJECT = PLAN7_F1_CUTOFF_ALWAYS_REJECT
F1_CUTOFF_ALWAYS_CPU = PLAN7_F1_CUTOFF_ALWAYS_CPU
EXECUTION_POLICY_VERSION = PLAN7_GPU_EXECUTION_POLICY_VERSION
EXECUTION_POLICY_AUTO = PLAN7_GPU_EXECUTION_POLICY_AUTO
EXECUTION_POLICY_SIMPLE = PLAN7_GPU_EXECUTION_POLICY_SIMPLE
EXECUTION_POLICY_THROUGHPUT = PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT
EXECUTION_POLICY_FORWARD_CANDIDATES_PER_WARP = (
    PLAN7_GPU_EXECUTION_POLICY_FORWARD_CANDIDATES_PER_WARP
)
BIAS_CPU_REQUIRED = PLAN7_BIAS_CPU_REQUIRED
BIAS_DEFINITE_REJECT = PLAN7_BIAS_DEFINITE_REJECT
BIAS_DEFINITE_PASS = PLAN7_BIAS_DEFINITE_PASS
BIAS_CUTOFF_INVALID = PLAN7_BIAS_CUTOFF_INVALID
BIAS_CUTOFF_SCORE = PLAN7_BIAS_CUTOFF_SCORE
BIAS_CUTOFF_ALWAYS_REJECT = PLAN7_BIAS_CUTOFF_ALWAYS_REJECT
BIAS_CUTOFF_ALWAYS_PASS = PLAN7_BIAS_CUTOFF_ALWAYS_PASS
BIAS_CUDA_UNATTESTED = PLAN7_BIAS_CUDA_UNATTESTED
BIAS_CUDA_SM75_RTX2080_TI = PLAN7_BIAS_CUDA_SM75_RTX2080_TI
BIAS_CUDA_SM90_H200 = PLAN7_BIAS_CUDA_SM90_H200
BIAS_LIBM_BUILD_ID_SIZE = PLAN7_BIAS_LIBM_BUILD_ID_SIZE
BIAS_PROFILE_SIZE = sizeof(plan7_bias_profile)
BIAS_RESULT_SIZE = sizeof(plan7_bias_result)
POSTFILTER_RECORD_VERSION = PLAN7_POSTFILTER_RECORD_VERSION
POSTFILTER_RESULT_SIZE = sizeof(plan7_postfilter_result)
FORWARD_RECORD_VERSION = PLAN7_FORWARD_RECORD_VERSION
FORWARD_RESULT_SIZE = sizeof(plan7_forward_result)
FORWARD_MAX_GATHERED_BYTES = PLAN7_FORWARD_MAX_GATHERED_BYTES
FORWARD_CPU_REQUIRED = PLAN7_FORWARD_CPU_REQUIRED
FORWARD_DEFINITE_REJECT = PLAN7_FORWARD_DEFINITE_REJECT
FORWARD_DEFINITE_PASS = PLAN7_FORWARD_DEFINITE_PASS
FORWARD_STATUS_OK = PLAN7_FORWARD_OK
FORWARD_STATUS_ERANGE = PLAN7_FORWARD_ERANGE
FORWARD_STATUS_ENORESULT = PLAN7_FORWARD_ENORESULT
FORWARD_STATUS_EMPTY = PLAN7_FORWARD_EMPTY
FORWARD_SUBWARP_POLICY_NO_KERNEL = PLAN7_FORWARD_SUBWARP_POLICY_NO_KERNEL
FORWARD_SUBWARP_POLICY_FORCED = PLAN7_FORWARD_SUBWARP_POLICY_FORCED
FORWARD_SUBWARP_POLICY_SPARSE_WIDTH1 = (
    PLAN7_FORWARD_SUBWARP_POLICY_SPARSE_WIDTH1
)
FORWARD_SUBWARP_POLICY_SHORT_WIDTH4 = (
    PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH4
)
FORWARD_SUBWARP_POLICY_SHORT_WIDTH2 = (
    PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH2
)
FORWARD_SUBWARP_POLICY_LONG_WIDTH4 = (
    PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH4
)
FORWARD_SUBWARP_POLICY_LONG_WIDTH2 = (
    PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH2
)
FORWARD_SUBWARP_POLICY_LONG_SATURATED_WIDTH1 = (
    PLAN7_FORWARD_SUBWARP_POLICY_LONG_SATURATED_WIDTH1
)
FORWARD_SUBWARP_POLICY_DIVERGENT_WIDTH1 = (
    PLAN7_FORWARD_SUBWARP_POLICY_DIVERGENT_WIDTH1
)
BACKWARD_DOMAIN_RECORD_VERSION = PLAN7_BACKWARD_DOMAIN_RECORD_VERSION
BACKWARD_DOMAIN_RESULT_SIZE = sizeof(plan7_backward_domain_result)
BACKWARD_DOMAIN_POSTERIOR_SIZE = sizeof(plan7_domain_posterior)
BACKWARD_DOMAIN_REGION_SIZE = sizeof(plan7_simple_region)
BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES = (
    PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES
)
BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS = (
    PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS
)
BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS = (
    PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS
)
BACKWARD_DOMAIN_CPU_REQUIRED = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED
BACKWARD_DOMAIN_NO_REGIONS = PLAN7_BACKWARD_DOMAIN_NO_REGIONS
BACKWARD_DOMAIN_SIMPLE = PLAN7_BACKWARD_DOMAIN_SIMPLE
BACKWARD_DOMAIN_STATUS_OK = PLAN7_BACKWARD_DOMAIN_OK
BACKWARD_DOMAIN_STATUS_ERANGE = PLAN7_BACKWARD_DOMAIN_ERANGE
BACKWARD_DOMAIN_STATUS_ENORESULT = PLAN7_BACKWARD_DOMAIN_ENORESULT
BACKWARD_DOMAIN_STATUS_EMPTY = PLAN7_BACKWARD_DOMAIN_EMPTY
BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH = (
    PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH
)
BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH = (
    PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH
)
BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE = (
    PLAN7_BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE
)
DOMAIN_RESCORE_RECORD_VERSION = PLAN7_DOMAIN_RESCORE_RECORD_VERSION
DOMAIN_RESCORE_RESULT_SIZE = sizeof(plan7_domain_rescore_result)
DOMAIN_RESCORE_TRACE_STEP_SIZE = sizeof(plan7_domain_rescore_trace_step)
DOMAIN_RESCORE_NULL2_COUNT = PLAN7_DOMAIN_RESCORE_NULL2_COUNT
DOMAIN_RESCORE_MAX_COMPACT_BYTES = PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
DOMAIN_RESCORE_MAX_MATRIX_BYTES = PLAN7_DOMAIN_RESCORE_MAX_MATRIX_BYTES
DOMAIN_RESCORE_MAX_TRACE_BYTES = PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES
DOMAIN_RESCORE_CPU_REQUIRED = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
DOMAIN_RESCORE_DEVICE_RESULT = PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
DOMAIN_RESCORE_STATUS_OK = PLAN7_DOMAIN_RESCORE_OK
DOMAIN_RESCORE_STATUS_ERANGE = PLAN7_DOMAIN_RESCORE_ERANGE
DOMAIN_RESCORE_STATUS_ENORESULT = PLAN7_DOMAIN_RESCORE_ENORESULT
DOMAIN_RESCORE_STATUS_ECAP = PLAN7_DOMAIN_RESCORE_ECAP
DOMAIN_RESCORE_STATUS_EMPTY = PLAN7_DOMAIN_RESCORE_EMPTY
