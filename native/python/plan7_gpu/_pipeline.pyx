# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Candidate-aware companion for PyHMMER 0.12.0's comparison pipeline.

This module deliberately cimports PyHMMER private state.  It must be rebuilt
for any PyHMMER version change; the runtime guard below prevents accidentally
loading it against an unsupported private ABI.
"""

import os as _os

from libc.stddef cimport size_t
from libc.math cimport exp, isfinite, isnan, log
from libc.stdint cimport (
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libc.stdlib cimport calloc, free, getenv, malloc
from libc.string cimport memcmp, memcpy, memset, strcmp, strlen
from cpython.bytes cimport (
    PyBytes_AS_STRING,
    PyBytes_FromStringAndSize,
)
from cpython.buffer cimport Py_buffer, PyBuffer_FillInfo
from cpython.pycapsule cimport (
    PyCapsule_Destructor,
    PyCapsule_GetContext,
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
    PyCapsule_SetContext,
    PyCapsule_SetDestructor,
    PyCapsule_SetName,
    PyCapsule_SetPointer,
)
from cpython.pyport cimport PY_SSIZE_T_MAX
from cpython.unicode cimport PyUnicode_FromString

from libeasel cimport (
    eslCONST_LOG2,
    eslERRBUFSIZE,
    eslEINACCURATE,
    eslEINVAL,
    eslERANGE,
    eslOK,
)
from libeasel.alphabet cimport eslAMINO
from libeasel.random cimport (
    ESL_RANDOMNESS,
    eslRND_FAST,
    eslRND_MERSENNE,
)
from libeasel.sq cimport ESL_SQ
from libhmmer cimport (
    p7_CUTOFF_UNSET,
    p7_FLAMBDA,
    p7_FTAU,
    p7_GA1,
    p7_GA2,
    p7_MLAMBDA,
    p7_MMU,
    p7_VLAMBDA,
    p7_VMU,
)
from libhmmer.p7_hmm cimport p7H_GA
from libhmmer.impl.p7_oprofile cimport (
    P7_OPROFILE,
    p7_oprofile_Compare,
    p7_oprofile_ReconfigLength,
)
from libhmmer.impl.p7_omx cimport P7_OMX
from libhmmer.p7_bg cimport (
    P7_BG,
    p7_bg_FilterScore,
    p7_bg_SetFilter,
    p7_bg_SetLength,
)
from libhmmer.p7_alidisplay cimport P7_ALIDISPLAY
from libhmmer.p7_domain cimport P7_DOMAIN
from libhmmer.p7_domaindef cimport P7_DOMAINDEF
from libhmmer.p7_hit cimport P7_HIT, p7_IS_INCLUDED
from libhmmer.p7_pipeline cimport (
    P7_PIPELINE,
    p7_SEARCH_SEQS,
    p7_ZSETBY_NTARGETS,
    p7_Pipeline,
    p7_pipeline_Reuse,
    p7_pli_NewModel,
)
from libhmmer.p7_tophits cimport P7_TOPHITS

from astra_pyhmmer.easel cimport DigitalSequence, DigitalSequenceBlock
from astra_pyhmmer.plan7 cimport (
    HMM,
    OptimizedProfile,
    Pipeline,
    Profile,
    TopHits,
)

import astra_pyhmmer as _pyhmmer
from astra_pyhmmer.errors import AlphabetMismatch, UnexpectedError
from array import array as _array
import hashlib as _hashlib
import importlib.util as _importlib_util
import io as _io
import time as _time
from pathlib import Path as _Path
import struct as _struct
from threading import Lock as _Lock


cdef extern from "sys/mman.h" nogil:
    int MADV_DONTNEED
    int madvise(void *address, size_t length, int advice)


cdef extern from "unistd.h" nogil:
    int _SC_PAGESIZE
    long sysconf(int name)


cdef extern from "hmmer.h" nogil:
    int p7_pipeline_IntraRowPageReleaseConfigure(
        uint64_t min_allocation_bytes,
    ) noexcept
    void p7_pipeline_IntraRowPageReleaseResetStatistics() noexcept
    void p7_pipeline_IntraRowPageReleaseStatistics(
        uint64_t *call_count,
        uint64_t *released_bytes,
    ) noexcept


cdef extern from "impl_sse/impl_sse.h" nogil:
    int p7X_NSCELLS
    int p7X_NXCELLS

# Keep the extension's established direct-file loading boundary usable by
# provenance/concurrency tests: a relative import has no parent package when
# importlib loads this DSO under its bare ``_pipeline`` initialization name.
from plan7_gpu import _telemetry as _telemetry_module

DIRECT_V3_STAGING_SCHEMA_VERSION = 3

# Astra consumes this private renderer only when this exact contract is
# present.  Version 2 is the first contract that includes the one-position
# alignment fix and rejects TopHits shapes outside the ordinary sorted protein
# search path before touching their private HMMER representation.
_ASTRA_TSV_RENDERER_ABI = 2


class _AstraTSVRendererUnsupported(ValueError):
    """The private renderer cannot prove compatibility with this TopHits."""

# These are the exact interval constants used by the patched HMMER compact
# continuation seam.  The Phase 9 evaluator applies the same input and final
# rounding allowance before it certifies any GA rejection.
cdef float _GA_COMPACT_INPUT_EPSILON_NATS = <float> 0.004
cdef float _GA_COMPACT_ROUND_EPSILON_BITS = <float> 0.00001


cdef float _ga_null_score_for_length(int length) noexcept nogil:
    """Reproduce p7_bg_SetLength() followed by p7_bg_NullOne()."""
    cdef float p1 = <float> length / <float> (length + 1)
    return <float> (
        <float> length * log(<double> p1)
        + log(1.0 - <double> p1)
    )

cdef extern from "dlfcn.h" nogil:
    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase

    int RTLD_NOLOAD
    int RTLD_NOW
    int dladdr(const void* address, Dl_info* info)
    int dlclose(void* handle)
    void* dlopen(const char* filename, int flags)
    void* dlsym(void* handle, const char* symbol)


cdef extern from "avx512_tail.h" nogil:
    int PLAN7_AVX512_TAIL_LANES
    int plan7_avx512_tail_available() noexcept
    int plan7_avx512_forward4_varlen(
        const unsigned char *const sequences[4],
        const int lengths[4],
        const P7_OPROFILE *profile,
        const float *forward_xmx[4],
        uint64_t forward_xmx_counts[4],
        float forward_scores[4],
        float forward_totscales[4],
        int forward_statuses[4],
        uint64_t *elapsed_ns,
    ) noexcept
    int plan7_avx512_backward4_varlen(
        const unsigned char *const sequences[4],
        const int lengths[4],
        const P7_OPROFILE *profile,
        const float *const forward_xmx[4],
        const uint64_t forward_xmx_counts[4],
        const float *backward_xmx[4],
        uint64_t backward_xmx_counts[4],
        float backward_scores[4],
        float backward_totscales[4],
        int backward_has_own_scales[4],
        uint64_t *elapsed_ns,
    ) noexcept
    int plan7_avx512_tail_madvise_pages(
        uint64_t *released_bytes,
    ) noexcept
    void plan7_avx512_tail_madvise_statistics(
        uint64_t *call_count,
        uint64_t *released_bytes,
    ) noexcept
    void plan7_avx512_tail_madvise_statistics_reset() noexcept


cdef extern from * nogil:
    """
    #include <fenv.h>
    #include <stdint.h>
    #include <string.h>
#if defined(__x86_64__) && defined(__GLIBC__)
    #include <xmmintrin.h>
#endif

    static int
    plan7_gpu_pipeline_host_environment_attested(void)
    {
#if defined(__x86_64__) && defined(__GLIBC__)
      const unsigned int rounding_ftz_daz_mask = 0x0000e040u;
      return fegetround() == FE_TONEAREST &&
             (_mm_getcsr() & rounding_ftz_daz_mask) == 0;
#else
      return 0;
#endif
    }

    static uint32_t
    plan7_semantic_oprofile_xf_bits(const P7_OPROFILE *om, int x, int y)
    {
      uint32_t bits;
      memcpy(&bits, &om->xf[x][y], sizeof(bits));
      return bits;
    }
    """
    uint32_t plan7_semantic_oprofile_xf_bits(
        const P7_OPROFILE *om,
        int x,
        int y,
    )
    int plan7_gpu_pipeline_host_environment_attested()


cdef extern from "esl_gumbel.h" nogil:
    double esl_gumbel_surv(double, double, double)


cdef extern from "esl_exponential.h" nogil:
    double esl_exp_surv(double, double, double)


cdef extern from "continuation_journal.h":
    const char *PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    const char *PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME
    const char *PLAN7_CONTINUATION_JOURNAL_V3_CAPSULE_NAME
    const char *PLAN7_CONTINUATION_JOURNAL_V3_CONSUMED_NAME

    cdef enum plan7_continuation_journal_abi:
        PLAN7_CONTINUATION_JOURNAL_VERSION
        PLAN7_CONTINUATION_JOURNAL_MAGIC
        PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE

    cdef enum plan7_continuation_journal_v3_abi:
        PLAN7_CONTINUATION_JOURNAL_V3_VERSION
        PLAN7_CONTINUATION_JOURNAL_V3_MAGIC
        PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE

    cdef enum plan7_continuation_journal_v3_source_kind:
        PLAN7_CONTINUATION_V3_SOURCE_HOST_SEAL
        PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL
        PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT

    cdef enum plan7_continuation_journal_v3_source_stage:
        PLAN7_CONTINUATION_V3_BEFORE_F1
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

    cdef enum plan7_continuation_journal_v3_payload:
        PLAN7_CONTINUATION_V3_HAS_POSTFILTER
        PLAN7_CONTINUATION_V3_HAS_FORWARD
        PLAN7_CONTINUATION_V3_HAS_DOMAIN
        PLAN7_CONTINUATION_V3_HAS_SPECIALS
        PLAN7_CONTINUATION_V3_HAS_REGIONS
        PLAN7_CONTINUATION_V3_HAS_COMPACT

    cdef enum plan7_continuation_journal_v3_precondition:
        PLAN7_CONTINUATION_V3_PRE_F2_SURVIVOR
        PLAN7_CONTINUATION_V3_PRE_DIRECT_FORWARD
        PLAN7_CONTINUATION_V3_PRE_F3_SURVIVOR
        PLAN7_CONTINUATION_V3_PRE_DOMAIN_SAFE
        PLAN7_CONTINUATION_V3_PRE_COMPACT_DEVICE

    cdef enum plan7_continuation_journal_v3_profile_flag:
        PLAN7_CONTINUATION_V3_PROFILE_HAS_V2_IDENTITY
        PLAN7_CONTINUATION_V3_PROFILE_HAS_FINGERPRINT

    cdef enum plan7_domain_rescore_abi:
        PLAN7_DOMAIN_RESCORE_RECORD_VERSION
        PLAN7_DOMAIN_RESCORE_RECORD_SIZE
        PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE
        PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES

    cdef enum plan7_domain_rescore_action:
        PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT

    cdef enum plan7_domain_rescore_status:
        PLAN7_DOMAIN_RESCORE_OK
        PLAN7_DOMAIN_RESCORE_ERANGE
        PLAN7_DOMAIN_RESCORE_ENORESULT
        PLAN7_DOMAIN_RESCORE_ECAP
        PLAN7_DOMAIN_RESCORE_EMPTY

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

    ctypedef struct plan7_backward_domain_provenance:
        plan7_forward_provenance forward
        uint64_t threshold_hash
        uint64_t result_hash
        uint64_t region_hash
        uint64_t candidate_count
        uint64_t region_count

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

    cdef enum plan7_continuation_compact_route:
        PLAN7_CONTINUATION_COMPACT_NONE
        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
        PLAN7_CONTINUATION_COMPACT_DEVICE

    ctypedef struct plan7_simple_region:
        uint32_t begin
        uint32_t end

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

    ctypedef struct plan7_continuation_journal_v3_options:
        uint64_t f1_bits
        uint64_t f2_bits
        uint64_t f3_bits
        uint64_t E_bits
        uint64_t T_bits
        uint64_t domE_bits
        uint64_t domT_bits
        uint64_t incE_bits
        uint64_t incT_bits
        uint64_t incdomE_bits
        uint64_t incdomT_bits
        uint64_t Z_bits
        uint64_t domZ_bits
        uint32_t rt1_bits
        uint32_t rt2_bits
        uint32_t rt3_bits
        int32_t do_biasfilter
        int32_t do_null2
        int32_t do_alignment_score_calc
        int32_t by_E
        int32_t dom_by_E
        int32_t inc_by_E
        int32_t incdom_by_E
        int32_t use_bit_cutoffs
        int32_t Z_setby
        int32_t domZ_setby
        int32_t mode
        int32_t long_targets
        uint32_t complete
        uint32_t reserved

    ctypedef struct plan7_continuation_journal_v3_certificate:
        uint64_t target_begin
        uint64_t target_end
        uint64_t residue_prefix_begin
        uint64_t residue_prefix_end
        uint64_t target_delta
        uint64_t residue_delta
        uint64_t before_f1_count
        uint64_t raw_f1_reject_count
        uint64_t bias_reject_count
        uint64_t f2_reject_count
        uint64_t f3_reject_count
        uint64_t no_region_count
        uint64_t n_past_msv_delta
        uint64_t n_past_bias_delta
        uint64_t n_past_vit_delta
        uint64_t n_past_fwd_delta
        uint32_t profile_index
        uint32_t segment_index
        uint64_t segment_tag

    ctypedef struct plan7_continuation_journal_v3_profile:
        uint64_t certificate_begin
        uint64_t certificate_count
        uint64_t exception_begin
        uint64_t exception_count
        uint64_t target_count
        uint64_t total_residues
        uint64_t source_postfilter_begin
        uint64_t source_postfilter_count
        uint64_t source_forward_begin
        uint64_t source_forward_count
        uint64_t source_domain_begin
        uint64_t source_domain_count
        uint64_t identity_token
        uint32_t profile_index
        uint32_t flags
        uint8_t profile_fingerprint[32]
        uint64_t profile_tag

    ctypedef struct plan7_continuation_journal_v3_exception:
        uint64_t source_postfilter_index
        uint64_t source_forward_index
        uint64_t source_domain_index
        uint64_t residue_prefix_begin
        uint64_t residue_prefix_end
        uint64_t residue_delta
        uint64_t special_begin
        uint64_t special_count
        uint64_t region_begin
        uint64_t region_count
        uint64_t compact_result_begin
        uint64_t compact_result_count
        uint64_t compact_trace_begin
        uint64_t compact_trace_count
        uint64_t compact_null2_begin
        uint64_t compact_null2_count
        uint32_t profile_index
        uint32_t sequence_index
        uint32_t exception_index
        uint8_t source_stage
        uint8_t route
        uint8_t payload_flags
        uint8_t preconditions
        uint8_t postfilter_record[16]
        uint8_t forward_record[12]
        uint8_t domain_record[64]
        uint32_t reserved
        uint64_t exception_tag

    ctypedef struct plan7_continuation_journal_v3:
        uint32_t magic
        uint16_t version
        uint16_t header_size
        uint32_t profile_size
        uint32_t certificate_size
        uint32_t exception_size
        uint32_t region_size
        uint32_t compact_result_size
        uint32_t compact_trace_step_size
        uint32_t compact_null2_stride
        uint32_t source_kind
        uint32_t reserved0
        uint64_t total_bytes
        uint64_t source_seal_token
        uint64_t session_id
        uint64_t selection_id
        uint64_t batch_generation
        uint64_t profile_count
        uint64_t target_count
        uint64_t total_residues
        uint64_t source_postfilter_count
        uint64_t source_forward_count
        uint64_t source_domain_count
        uint64_t certificate_count
        uint64_t exception_count
        uint64_t special_count
        uint64_t region_count
        uint64_t compact_result_count
        uint64_t compact_trace_offset_count
        uint64_t compact_trace_count
        uint64_t compact_null2_count
        uint64_t source_v2_total_bytes
        uint64_t source_v2_integrity_tag
        uint64_t generation_tail_fingerprint
        uint64_t profiles_offset
        uint64_t certificates_offset
        uint64_t exceptions_offset
        uint64_t specials_offset
        uint64_t regions_offset
        uint64_t compact_results_offset
        uint64_t compact_trace_offsets_offset
        uint64_t compact_traces_offset
        uint64_t compact_null2_offset
        uint64_t background_fingerprint_offset
        uint64_t background_fingerprint_bytes
        uint8_t sequence_content_fingerprint[32]
        plan7_continuation_journal_v3_options options
        plan7_forward_provenance forward
        plan7_backward_domain_provenance backward
        plan7_domain_rescore_provenance rescore
        uint64_t integrity_tag

    ctypedef struct plan7_continuation_journal_v3_owner:
        uint64_t allocation_bytes
        uint64_t source_seal_token

    uint64_t plan7_continuation_journal_integrity(
        const plan7_continuation_journal *journal,
    ) nogil

    uint64_t plan7_continuation_journal_v3_integrity(
        const plan7_continuation_journal_v3 *journal,
    ) nogil

    uint64_t plan7_continuation_journal_v3_certificate_tag(
        const plan7_continuation_journal_v3_certificate *certificate,
    ) nogil

    uint64_t plan7_continuation_journal_v3_profile_tag(
        const plan7_continuation_journal_v3_profile *profile,
    ) nogil

    uint64_t plan7_continuation_journal_v3_exception_tag(
        const plan7_continuation_journal_v3_exception *exception,
    ) nogil

    int plan7_continuation_journal_v3_checked_add(
        uint64_t left,
        uint64_t right,
        uint64_t *sum,
    ) nogil

    int plan7_continuation_journal_v3_checked_multiply(
        uint64_t left,
        uint64_t right,
        uint64_t *product,
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

    int plan7_continuation_journal_rescore_hashes(
        const plan7_domain_rescore_result *results,
        uint64_t result_count,
        const uint64_t *trace_offsets,
        const plan7_domain_rescore_trace_step *traces,
        uint64_t trace_count,
        const float *null2,
        uint64_t null2_count,
        uint64_t *result_hash_out,
        uint64_t *trace_hash_out,
        uint64_t *null2_hash_out,
    ) nogil


_abi_spec = _importlib_util.spec_from_file_location(
    "_plan7_gpu_pyhmmer_abi", _Path(__file__).resolve().with_name("_abi.py")
)
if _abi_spec is None or _abi_spec.loader is None:
    raise ImportError("cannot load the plan7_gpu PyHMMER ABI verifier")
_abi_module = _importlib_util.module_from_spec(_abi_spec)
_abi_spec.loader.exec_module(_abi_module)

_fingerprint_spec = _importlib_util.spec_from_file_location(
    "_plan7_gpu_profile_fingerprint",
    _Path(__file__).resolve().with_name("_fingerprint.py"),
)
if _fingerprint_spec is None or _fingerprint_spec.loader is None:
    raise ImportError("cannot load the plan7_gpu optimized-profile fingerprint")
_fingerprint_module = _importlib_util.module_from_spec(_fingerprint_spec)
_fingerprint_spec.loader.exec_module(_fingerprint_module)


PYHMMER_PRIVATE_ABI = "0.12.0"
PYHMMER_PRIVATE_ABI_SHA256 = PYHMMER_ABI_SHA256
if _pyhmmer.__version__ != PYHMMER_PRIVATE_ABI:
    raise ImportError(
        f"plan7_gpu._pipeline requires PyHMMER {PYHMMER_PRIVATE_ABI}, "
        f"found {_pyhmmer.__version__}"
    )
_abi_module.validate_private_abi_platform()
_runtime_abi_sha256 = _abi_module.pyhmmer_abi_fingerprint()
if _runtime_abi_sha256 != PYHMMER_PRIVATE_ABI_SHA256:
    raise ImportError(
        "plan7_gpu._pipeline was built against a different PyHMMER private ABI "
        f"({PYHMMER_PRIVATE_ABI_SHA256} != {_runtime_abi_sha256})"
    )


cdef size_t HMMER_TARGET_LIMIT = 100000
SEALED_STAGE_TIMING_SCHEMA_VERSION = 1

cdef enum:
    BIAS_CPU_REQUIRED = 0
    BIAS_DEFINITE_REJECT = 1
    BIAS_DEFINITE_PASS = 2
    SSV_OK = 0
    SSV_ENORESULT = 19
    FORWARD_CPU_REQUIRED = 0
    FORWARD_DEFINITE_REJECT = 1
    FORWARD_DEFINITE_PASS = 2
    FORWARD_OK = 0
    FORWARD_ERANGE = 16
    FORWARD_ENORESULT = 19
    FORWARD_EMPTY = 255
    DOMAIN_CPU_REQUIRED = 0
    DOMAIN_NO_REGIONS = 1
    DOMAIN_SIMPLE = 2
    DOMAIN_OK = 0
    DOMAIN_ERANGE = 16
    DOMAIN_ENORESULT = 19
    DOMAIN_EMPTY = 255
    P7_VIT_EXTERNAL = 1
    P7_VIT_CPU = 2


cdef struct _bias_result:
    uint32_t sequence_index
    float filtersc
    int16_t ssv_numerator
    uint8_t ssv_status
    uint8_t action


cdef struct _postfilter_result:
    uint32_t sequence_index
    float filtersc
    int16_t msv_numerator
    uint8_t msv_status
    uint8_t action
    float vfsc


cdef struct _forward_result:
    uint32_t sequence_index
    float fwdsc
    uint8_t status
    uint8_t action
    uint16_t reserved


cdef struct _compact_consumption_statistics:
    uint64_t target_count
    uint64_t postfilter_record_count
    uint64_t f1_reject_count
    uint64_t cpu_pipeline_count
    uint64_t definite_reject_count
    uint64_t filter_continuation_count
    uint64_t forward_continuation_count
    uint64_t simple_continuation_count
    uint64_t journal_match_count
    uint64_t journal_cpu_required_count
    uint64_t journal_no_region_count
    uint64_t journal_simple_count
    uint64_t attempt_count
    uint64_t accepted_count
    uint64_t invalid_retry_count
    uint64_t threshold_retry_count
    uint64_t first_row_index
    uint32_t first_profile_index
    uint32_t first_sequence_index
    uint64_t first_domain_count
    uint64_t requested_profile_index
    uint64_t journal_row_start
    uint64_t journal_row_stop
    uint64_t source_postfilter_cpu_count
    uint64_t source_definite_reject_count
    uint64_t source_filter_count
    uint64_t source_forward_count
    uint64_t source_journal_eligible_count
    uint64_t source_simple_bypass_count
    uint64_t decision_forward_row_external_unavailable
    uint64_t decision_forward_seam_unavailable
    uint64_t decision_forward_f2_changed
    uint64_t decision_forward_f3_changed
    uint64_t decision_forward_bias_changed
    uint64_t decision_journal_storage_unavailable
    uint64_t decision_journal_simple_seam_unavailable
    uint64_t decision_journal_tail_changed
    uint64_t decision_compact_route_not_device
    uint64_t decision_compact_empty
    uint64_t decision_compact_tail_changed
    uint64_t decision_compact_rebase_unavailable
    uint64_t cpu_tail_simd_eligible_rows
    uint64_t cpu_tail_simd_groups
    uint64_t cpu_tail_simd_rows
    uint64_t cpu_tail_simd_scalar_tail_rows
    uint64_t cpu_tail_simd_fallback_groups
    uint64_t cpu_tail_simd_kernel_ns
    uint64_t cpu_tail_filter_simd_eligible_rows
    uint64_t cpu_tail_filter_simd_groups
    uint64_t cpu_tail_filter_simd_rows
    uint64_t cpu_tail_filter_simd_scalar_tail_rows
    uint64_t cpu_tail_filter_simd_fallback_groups
    uint64_t cpu_tail_filter_simd_fallback_rows
    uint64_t cpu_tail_filter_simd_f3_survivor_rows
    uint64_t cpu_tail_filter_simd_backward_groups
    uint64_t cpu_tail_filter_simd_forward_kernel_ns
    uint64_t cpu_tail_filter_simd_backward_kernel_ns


ctypedef int (*_pipeline_from_filter_scores_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    int,
    float,
) noexcept nogil

ctypedef int (*_pipeline_from_filter_and_forward_scores_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    const float*,
    uint64_t,
) noexcept nogil

ctypedef int (*_pipeline_from_filter_forward_backward_scores_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    const float*,
    uint64_t,
    float,
    const float*,
    uint64_t,
    float,
    int,
) noexcept nogil

ctypedef int (*_pipeline_from_filter_and_forward_simple_regions_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    uint64_t,
    uint64_t,
    uint64_t,
    int,
    int,
    float,
    const plan7_simple_region*,
    uint64_t,
) noexcept nogil

ctypedef uint64_t (*_pipeline_compact_tail_fingerprint_f)(
    const P7_PIPELINE*,
) noexcept nogil

ctypedef int (*_pipeline_from_filter_forward_compact_domains_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    uint64_t,
    uint64_t,
    uint32_t,
    uint32_t,
    uint32_t,
    float,
    const plan7_domain_rescore_result*,
    uint64_t,
    const uint64_t*,
    uint64_t,
    const plan7_domain_rescore_trace_step*,
    uint64_t,
    const float*,
    uint64_t,
) noexcept nogil


cdef union _float_bits:
    float value
    uint32_t bits


cdef union _double_bits:
    double value
    uint64_t bits


cdef struct _pipeline_tail_snapshot:
    uint64_t f1_bits
    uint64_t f2_bits
    uint64_t f3_bits
    uint64_t E_bits
    uint64_t T_bits
    uint64_t domE_bits
    uint64_t domT_bits
    uint64_t incE_bits
    uint64_t incT_bits
    uint64_t incdomE_bits
    uint64_t incdomT_bits
    uint64_t Z_bits
    uint64_t domZ_bits
    uint32_t rt1_bits
    uint32_t rt2_bits
    uint32_t rt3_bits
    int32_t do_biasfilter
    int32_t do_null2
    int32_t do_alignment_score_calc
    int32_t by_E
    int32_t dom_by_E
    int32_t inc_by_E
    int32_t incdom_by_E
    int32_t use_bit_cutoffs
    int32_t Z_setby
    int32_t domZ_setby
    int32_t mode
    int32_t long_targets


cdef class _SealedPostfilterBatch:
    """Opaque, lifetime-pinned continuation data validated as one batch."""

    cdef bint _ready
    cdef tuple _queries
    cdef tuple _optimized_profiles
    cdef DigitalSequenceBlock _sequences
    cdef const uint8_t[::1] _postfilter_records
    cdef const uint64_t[::1] _postfilter_offsets
    cdef const uint64_t[::1] _residue_offsets
    cdef const uint8_t[::1] _forward_records
    cdef const uint64_t[::1] _forward_offsets
    cdef const uint64_t[::1] _special_offsets
    cdef const float[::1] _specials
    cdef const uint8_t[::1] _row_has_external
    cdef const uint8_t[::1] _background_fingerprint
    cdef const uint8_t[::1] _journal_storage
    cdef const uint64_t[::1] _journal_profile_offsets
    cdef const uint8_t[::1] _journal_rows
    cdef const uint64_t[::1] _journal_region_offsets
    cdef const uint8_t[::1] _journal_regions
    cdef const uint64_t[::1] _journal_compact_row_offsets
    cdef const uint8_t[::1] _journal_compact_results
    cdef const uint64_t[::1] _journal_compact_trace_offsets
    cdef const uint8_t[::1] _journal_compact_traces
    cdef const float[::1] _journal_compact_null2
    cdef double _f1
    cdef uint64_t _generation_f2_bits
    cdef uint64_t _generation_f3_bits
    cdef bint _generation_bias_filter
    cdef uint32_t _journal_guard_bits
    cdef uint64_t _generation_tail_fingerprint
    cdef uint64_t _rescore_simple_row_count
    cdef uint64_t _rescore_device_result_count
    cdef uint64_t _rescore_cpu_required_count
    cdef uint64_t _rescore_certified_ga_count
    cdef uint64_t _rescore_numeric_fallback_count
    cdef uint64_t _rescore_cap_fallback_count
    cdef uint64_t _rescore_global_cpu_fallback_count
    cdef object _owned_residue_offsets_bytes
    cdef object _owned_background_fingerprint_bytes
    cdef object _excluded_residue_offsets_bytes
    cdef object _excluded_background_fingerprint_bytes
    cdef object _native_stage_timings
    cdef object _generation_statistics
    cdef uint64_t _telemetry_session_id
    cdef uint64_t _telemetry_selection_id
    cdef uint64_t _telemetry_batch_generation
    cdef plan7_continuation_journal_v3 *_journal_v3
    cdef uint64_t _journal_v3_bytes
    cdef uint64_t _journal_v3_planning_ns
    cdef uint64_t _journal_v3_validation_ns
    cdef bint _direct_v3_source
    cdef uint64_t _direct_v3_eliminated_v2_bytes
    cdef uint64_t _direct_v3_staging_bytes
    cdef uint64_t _direct_v3_staging_build_ns
    cdef uint64_t _direct_v3_source_validation_ns
    cdef const uint8_t[::1] _direct_v3_decisions
    cdef uint64_t _direct_v3_expected_exception_count
    cdef uint64_t _direct_v3_expected_special_count
    cdef uint64_t _direct_v3_expected_region_count
    cdef uint64_t _direct_v3_expected_compact_result_count
    cdef uint64_t _direct_v3_expected_compact_trace_count
    cdef uint64_t _direct_v3_expected_compact_null2_count
    cdef uint64_t _direct_v3_expected_decision_tag
    cdef uint64_t _journal_v3_source_scan_count
    cdef uint64_t _journal_v3_decision_scan_count
    cdef uint64_t _source_consumer_validation_ns
    cdef const uint64_t[::1] _source_identity_tokens
    cdef const uint8_t[::1] _source_profile_fingerprints
    cdef const uint8_t[::1] _source_sequence_fingerprint
    cdef plan7_forward_provenance _source_forward_provenance
    cdef plan7_backward_domain_provenance _source_backward_provenance
    cdef plan7_domain_rescore_provenance _source_rescore_provenance
    cdef _pipeline_tail_snapshot _pipeline_options
    cdef _pipeline_from_filter_scores_f _filter_scores_seam
    cdef _pipeline_from_filter_and_forward_scores_f _forward_scores_seam
    cdef _pipeline_from_filter_forward_backward_scores_f _forward_backward_scores_seam
    cdef _pipeline_from_filter_and_forward_simple_regions_f _simple_regions_seam
    cdef _pipeline_compact_tail_fingerprint_f _compact_tail_fingerprint
    cdef _pipeline_from_filter_forward_compact_domains_f _compact_domains_seam

    def __cinit__(self):
        self._ready = False
        self._owned_residue_offsets_bytes = 0
        self._owned_background_fingerprint_bytes = 0
        self._excluded_residue_offsets_bytes = 0
        self._excluded_background_fingerprint_bytes = 0
        self._native_stage_timings = None
        self._generation_statistics = None
        self._telemetry_session_id = 0
        self._telemetry_selection_id = 0
        self._telemetry_batch_generation = 0
        self._journal_v3 = NULL
        self._journal_v3_bytes = 0
        self._journal_v3_planning_ns = 0
        self._journal_v3_validation_ns = 0
        self._direct_v3_source = False
        self._direct_v3_eliminated_v2_bytes = 0
        self._direct_v3_staging_bytes = 0
        self._direct_v3_staging_build_ns = 0
        self._direct_v3_source_validation_ns = 0
        self._direct_v3_expected_exception_count = 0
        self._direct_v3_expected_special_count = 0
        self._direct_v3_expected_region_count = 0
        self._direct_v3_expected_compact_result_count = 0
        self._direct_v3_expected_compact_trace_count = 0
        self._direct_v3_expected_compact_null2_count = 0
        self._direct_v3_expected_decision_tag = 0
        self._journal_v3_source_scan_count = 0
        self._journal_v3_decision_scan_count = 0
        self._source_consumer_validation_ns = 0

    def __dealloc__(self):
        if self._journal_v3 != NULL:
            free(self._journal_v3)
            self._journal_v3 = NULL

    def __repr__(self):
        return "<opaque sealed post-filter batch>"

    def __reduce__(self):
        raise TypeError("sealed post-filter batches cannot be pickled")


cdef class _ContinuationJournalStorage:
    """Read-only buffer exporting one transferred native journal allocation."""

    cdef uint8_t *_data
    cdef Py_ssize_t _size
    cdef bint _owns

    def __cinit__(self):
        self._data = NULL
        self._size = 0
        self._owns = False

    def __dealloc__(self):
        if self._owns and self._data != NULL:
            free(self._data)

    def __getbuffer__(self, Py_buffer *view, int flags):
        if self._data == NULL:
            raise BufferError("continuation journal storage is unavailable")
        PyBuffer_FillInfo(
            view, self, self._data, self._size, 1, flags
        )

    def __releasebuffer__(self, Py_buffer *view):
        pass


cdef _pipeline_from_filter_scores_f _filter_scores_seam_cache = NULL
cdef _pipeline_from_filter_and_forward_scores_f _forward_scores_seam_cache = NULL
cdef _pipeline_from_filter_forward_backward_scores_f _forward_backward_scores_seam_cache = NULL
cdef _pipeline_from_filter_and_forward_simple_regions_f _simple_regions_seam_cache = NULL
cdef _pipeline_compact_tail_fingerprint_f _compact_tail_fingerprint_cache = NULL
cdef _pipeline_from_filter_forward_compact_domains_f _compact_domains_seam_cache = NULL
cdef bint _filter_scores_seam_resolved = False
cdef bint _forward_scores_seam_resolved = False
cdef bint _forward_backward_scores_seam_resolved = False
cdef bint _simple_regions_seam_resolved = False
cdef bint _compact_domains_seam_resolved = False
cdef bint _filter_scores_same_dso = False
cdef bint _forward_scores_same_dso = False
cdef bint _forward_backward_scores_same_dso = False
cdef bint _simple_regions_same_dso = False
cdef bint _compact_domains_same_dso = False
cdef uint64_t _filter_scores_resolutions = 0
cdef uint64_t _forward_scores_resolutions = 0
cdef uint64_t _forward_backward_scores_resolutions = 0
cdef uint64_t _filter_scores_dlopen_calls = 0
cdef uint64_t _forward_scores_dlopen_calls = 0
cdef uint64_t _forward_backward_scores_dlopen_calls = 0
cdef uint64_t _filter_scores_dlclose_calls = 0
cdef uint64_t _forward_scores_dlclose_calls = 0
cdef uint64_t _forward_backward_scores_dlclose_calls = 0
cdef uint64_t _simple_regions_resolutions = 0
cdef uint64_t _simple_regions_dlopen_calls = 0
cdef uint64_t _simple_regions_dlclose_calls = 0
cdef uint64_t _compact_domains_resolutions = 0
cdef uint64_t _compact_domains_dlopen_calls = 0
cdef uint64_t _compact_domains_dlclose_calls = 0
cdef uint64_t _v3_consumer_call_count = 0
cdef uint64_t _v3_consumer_preflight_ns = 0
cdef uint64_t _v3_consumer_core_ns = 0
cdef uint64_t _v3_consumer_statistics_ns = 0
cdef uint64_t _v3_consumer_certificate_visits = 0
cdef uint64_t _v3_consumer_exception_visits = 0
cdef uint64_t _filter_tail_audit_eligible_rows = 0
cdef uint64_t _filter_tail_audit_groups = 0
cdef uint64_t _filter_tail_audit_rows = 0
cdef uint64_t _filter_tail_audit_scalar_tail_rows = 0
cdef uint64_t _filter_tail_audit_fallback_groups = 0
cdef uint64_t _filter_tail_audit_fallback_rows = 0
cdef uint64_t _filter_tail_audit_f3_survivor_rows = 0
cdef uint64_t _filter_tail_audit_backward_groups = 0
cdef uint64_t _filter_tail_audit_forward_kernel_ns = 0
cdef uint64_t _filter_tail_audit_backward_kernel_ns = 0
cdef uint64_t _shard_z_audit_call_count = 0
cdef uint64_t _shard_z_audit_first_shard_count = 0
cdef uint64_t _shard_z_audit_terminal_shard_count = 0
cdef uint64_t _shard_z_audit_exception_count = 0
cdef uint64_t _shard_z_audit_local_target_visits = 0
cdef uint64_t _shard_z_audit_dynamic_applicable_count = 0
cdef uint64_t _shard_z_audit_nonzero_offset_count = 0
cdef uint64_t _shard_z_audit_restored_count = 0
cdef uint64_t _shard_z_audit_offset_sum = 0
cdef uint64_t _shard_z_audit_offset_max = 0
cdef int _avx512_tail_madvise_override = -1
cdef int _filter_tail_simd_override = -1

_continuation_seam_resolve_lock = _Lock()
cdef uint8_t _consumed_journal_sentinel = 0


def _bias_filter_score_bits(
    Pipeline pipeline,
    OptimizedProfile optimized_profile,
    DigitalSequence sequence,
):
    """Return HMMER 3.4 ``p7_bg_FilterScore`` as raw binary32 bits."""
    cdef float filtersc
    cdef _float_bits encoded
    cdef int status
    if not pipeline.alphabet._eq(optimized_profile.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, optimized_profile.alphabet)
    if not pipeline.alphabet._eq(sequence.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sequence.alphabet)
    if sequence._sq.n == 0 or sequence._sq.n > HMMER_TARGET_LIMIT:
        raise ValueError("bias oracle target length must be in [1, 100000]")
    with nogil:
        status = p7_bg_SetFilter(
            pipeline.background._bg,
            optimized_profile._om.M,
            optimized_profile._om.compo,
        )
        if status == eslOK:
            status = p7_bg_SetLength(pipeline.background._bg, sequence._sq.n)
        if status == eslOK:
            status = p7_bg_FilterScore(
                pipeline.background._bg,
                sequence._sq.dsq,
                sequence._sq.n,
                &filtersc,
            )
    if status != eslOK:
        raise UnexpectedError(status, "p7_bg_FilterScore")
    encoded.value = filtersc
    return encoded.bits


def _oprofiles_equal_hmmer(
    OptimizedProfile expected,
    OptimizedProfile observed,
):
    """Run HMMER's own optimized-profile comparator at zero tolerance."""
    cdef char error[eslERRBUFSIZE]
    cdef int status

    error[0] = 0
    with nogil:
        status = p7_oprofile_Compare(expected._om, observed._om, 0.0, error)
    return status == eslOK


cdef _pipeline_from_filter_scores_f _resolve_filter_scores_seam() noexcept nogil:
    global _filter_scores_dlopen_calls
    global _filter_scores_dlclose_calls
    global _filter_scores_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _filter_scores_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(handle, "p7_PipelineFromFilterScores")
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _filter_scores_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _filter_scores_same_dso = True
    _filter_scores_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_scores_f> symbol


cdef _pipeline_from_filter_scores_f _cached_filter_scores_seam():
    global _filter_scores_resolutions
    global _filter_scores_seam_cache
    global _filter_scores_seam_resolved

    if not _filter_scores_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _filter_scores_seam_resolved:
                _filter_scores_resolutions += 1
                _filter_scores_seam_cache = _resolve_filter_scores_seam()
                _filter_scores_seam_resolved = True
    return _filter_scores_seam_cache


cdef size_t _madvise_full_pages(
    void *allocation,
    size_t allocation_bytes,
    size_t page_size,
    int *failed,
) noexcept nogil:
    """Discard only pages wholly contained in one live allocation."""
    cdef uintptr_t start
    cdef uintptr_t first
    cdef uintptr_t end

    if allocation == NULL or allocation_bytes < page_size:
        return 0
    start = <uintptr_t> allocation
    first = ((start + page_size - 1) // page_size) * page_size
    end = ((start + allocation_bytes) // page_size) * page_size
    if end <= first:
        return 0
    if madvise(<void *> first, <size_t> (end - first), MADV_DONTNEED) != 0:
        failed[0] = 1
        return 0
    return <size_t> (end - first)


cdef size_t _madvise_omx_pages(
    P7_OMX *matrix,
    size_t page_size,
    int *failed,
) noexcept nogil:
    cdef size_t released = 0
    cdef size_t dp_bytes
    cdef size_t x_bytes

    if matrix == NULL:
        return 0
    # p7_omx_GrowTo() records scalar DP capacity in ncells.  The SSE
    # allocation contains three 16-byte striped vectors per four scalar cells.
    dp_bytes = matrix.ncells * 12 + 15
    x_bytes = <size_t> matrix.allocXR * 4 * p7X_NXCELLS + 15
    released += _madvise_full_pages(
        matrix.dp_mem, dp_bytes, page_size, failed
    )
    released += _madvise_full_pages(
        matrix.x_mem, x_bytes, page_size, failed
    )
    return released


def _madvise_pipeline_pages_bound(Pipeline pipeline):
    """Release invalid reusable DP pages without changing Pipeline capacity."""
    cdef P7_PIPELINE *pli = pipeline._pli
    cdef long observed_page_size
    cdef size_t page_size
    cdef size_t released = 0
    cdef int failed = 0

    if pli == NULL:
        raise RuntimeError("pipeline state is unavailable")
    observed_page_size = sysconf(_SC_PAGESIZE)
    if observed_page_size <= 0:
        raise RuntimeError("could not resolve the system page size")
    page_size = <size_t> observed_page_size
    with nogil:
        released += _madvise_omx_pages(pli.oxf, page_size, &failed)
        released += _madvise_omx_pages(pli.oxb, page_size, &failed)
        released += _madvise_omx_pages(pli.fwd, page_size, &failed)
        released += _madvise_omx_pages(pli.bck, page_size, &failed)
    if failed:
        raise OSError("madvise(MADV_DONTNEED) failed for Pipeline DP pages")
    return int(released)


def _avx512_tail_madvise_statistics_bound():
    """Return process-wide counters for the private AVX result release."""
    cdef uint64_t call_count = 0
    cdef uint64_t released_bytes = 0
    with nogil:
        plan7_avx512_tail_madvise_statistics(
            &call_count, &released_bytes
        )
    return {
        "call_count": int(call_count),
        "released_bytes": int(released_bytes),
    }


def _reset_avx512_tail_madvise_statistics_bound():
    """Reset private AVX result-release counters for exact host gates."""
    with nogil:
        plan7_avx512_tail_madvise_statistics_reset()


def _configure_avx512_tail_madvise_bound(enabled):
    """Override AVX result-page release; None restores environment policy."""
    _swap_avx512_tail_madvise_bound(enabled)


def _swap_avx512_tail_madvise_bound(enabled):
    """Set AVX result-page release and return the prior scoped override."""
    global _avx512_tail_madvise_override
    cdef int previous = _avx512_tail_madvise_override
    if enabled is not None and type(enabled) is not bool:
        raise TypeError("enabled must be bool or None")
    _avx512_tail_madvise_override = (
        -1 if enabled is None else int(enabled)
    )
    return None if previous < 0 else bool(previous)


def _configure_filter_tail_simd_bound(enabled):
    """Override filter-tail SIMD; None restores environment policy."""
    _swap_filter_tail_simd_bound(enabled)


def _swap_filter_tail_simd_bound(enabled):
    """Set filter-tail SIMD and return the prior scoped override."""
    global _filter_tail_simd_override
    cdef int previous = _filter_tail_simd_override
    if enabled is not None and type(enabled) is not bool:
        raise TypeError("enabled must be bool or None")
    _filter_tail_simd_override = (
        -1 if enabled is None else int(enabled)
    )
    return None if previous < 0 else bool(previous)


def _configure_intrarow_page_release_bound(min_allocation_bytes):
    """Configure HMMER intra-row page release; zero disables it."""
    _swap_intrarow_page_release_bound(min_allocation_bytes)


cdef uint64_t _intrarow_page_release_min_bytes_bound = 0


def _intrarow_page_release_configuration_bound():
    """Return the threshold last installed through this private binding."""
    return int(_intrarow_page_release_min_bytes_bound)


def _swap_intrarow_page_release_bound(min_allocation_bytes):
    """Install a HMMER release threshold and return the prior threshold."""
    global _intrarow_page_release_min_bytes_bound
    cdef uint64_t threshold
    cdef uint64_t previous = _intrarow_page_release_min_bytes_bound
    cdef int status

    if type(min_allocation_bytes) is not int:
        raise TypeError("min_allocation_bytes must be an integer")
    if min_allocation_bytes < 0 or min_allocation_bytes > 0xffffffffffffffff:
        raise ValueError("min_allocation_bytes is outside uint64 range")
    threshold = <uint64_t> min_allocation_bytes
    with nogil:
        status = p7_pipeline_IntraRowPageReleaseConfigure(threshold)
    if status != eslOK:
        raise RuntimeError(
            "p7_pipeline_IntraRowPageReleaseConfigure failed "
            f"with status {status}"
        )
    _intrarow_page_release_min_bytes_bound = threshold
    return int(previous)


def _intrarow_page_release_statistics_bound():
    """Return process-wide counters for HMMER intra-row page release."""
    cdef uint64_t call_count = 0
    cdef uint64_t released_bytes = 0
    with nogil:
        p7_pipeline_IntraRowPageReleaseStatistics(
            &call_count, &released_bytes
        )
    return {
        "call_count": int(call_count),
        "released_bytes": int(released_bytes),
    }


def _reset_intrarow_page_release_statistics_bound():
    """Reset HMMER intra-row page-release counters."""
    with nogil:
        p7_pipeline_IntraRowPageReleaseResetStatistics()


def _filter_scores_seam_available():
    """Return whether the project-private HMMER continuation seam is loaded."""
    return _cached_filter_scores_seam() != NULL


cdef _pipeline_from_filter_and_forward_scores_f _resolve_filter_and_forward_scores_seam() noexcept nogil:
    global _forward_scores_dlopen_calls
    global _forward_scores_dlclose_calls
    global _forward_scores_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _forward_scores_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(handle, "p7_PipelineFromFilterAndForwardScores")
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _forward_scores_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _forward_scores_same_dso = True
    _forward_scores_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_and_forward_scores_f> symbol


cdef _pipeline_from_filter_and_forward_scores_f _cached_filter_and_forward_scores_seam():
    global _forward_scores_resolutions
    global _forward_scores_seam_cache
    global _forward_scores_seam_resolved

    if not _forward_scores_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _forward_scores_seam_resolved:
                _forward_scores_resolutions += 1
                _forward_scores_seam_cache = (
                    _resolve_filter_and_forward_scores_seam()
                )
                _forward_scores_seam_resolved = True
    return _forward_scores_seam_cache


def _filter_and_forward_scores_seam_available():
    """Return whether the exact external-Forward seam is loaded."""
    return _cached_filter_and_forward_scores_seam() != NULL


cdef _pipeline_from_filter_forward_backward_scores_f _resolve_filter_forward_backward_scores_seam() noexcept nogil:
    global _forward_backward_scores_dlopen_calls
    global _forward_backward_scores_dlclose_calls
    global _forward_backward_scores_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _forward_backward_scores_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(
        handle, "p7_PipelineFromFilterForwardAndBackwardScores"
    )
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _forward_backward_scores_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _forward_backward_scores_same_dso = True
    _forward_backward_scores_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_forward_backward_scores_f> symbol


cdef _pipeline_from_filter_forward_backward_scores_f _cached_filter_forward_backward_scores_seam():
    global _forward_backward_scores_resolutions
    global _forward_backward_scores_seam_cache
    global _forward_backward_scores_seam_resolved

    if not _forward_backward_scores_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _forward_backward_scores_seam_resolved:
                _forward_backward_scores_resolutions += 1
                _forward_backward_scores_seam_cache = (
                    _resolve_filter_forward_backward_scores_seam()
                )
                _forward_backward_scores_seam_resolved = True
    return _forward_backward_scores_seam_cache


def _filter_forward_backward_scores_seam_available():
    """Return whether the exact external-Backward seam is loaded."""
    return _cached_filter_forward_backward_scores_seam() != NULL


cdef _pipeline_from_filter_and_forward_simple_regions_f _resolve_simple_regions_seam() noexcept nogil:
    global _simple_regions_dlopen_calls
    global _simple_regions_dlclose_calls
    global _simple_regions_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _simple_regions_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(
        handle, "p7_PipelineFromFilterAndForwardSimpleRegions"
    )
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _simple_regions_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _simple_regions_same_dso = True
    _simple_regions_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_and_forward_simple_regions_f> symbol


cdef _pipeline_from_filter_and_forward_simple_regions_f _cached_simple_regions_seam():
    global _simple_regions_resolutions
    global _simple_regions_seam_cache
    global _simple_regions_seam_resolved

    if not _simple_regions_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _simple_regions_seam_resolved:
                _simple_regions_resolutions += 1
                _simple_regions_seam_cache = _resolve_simple_regions_seam()
                _simple_regions_seam_resolved = True
    return _simple_regions_seam_cache


def _simple_regions_seam_available():
    """Return whether the guarded simple-region continuation seam is loaded."""
    return _cached_simple_regions_seam() != NULL


cdef void _resolve_compact_domain_seams() noexcept nogil:
    global _compact_domains_dlopen_calls
    global _compact_domains_dlclose_calls
    global _compact_domains_same_dso
    global _compact_domains_seam_cache
    global _compact_tail_fingerprint_cache
    cdef Dl_info info
    cdef Dl_info domain_info
    cdef Dl_info fingerprint_info
    cdef void* handle
    cdef void* domain_symbol
    cdef void* fingerprint_symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return
    _compact_domains_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return
    domain_symbol = dlsym(
        handle, "p7_PipelineFromFilterForwardAndCompactDomainsV2"
    )
    fingerprint_symbol = dlsym(
        handle, "p7_pipeline_CompactTailFingerprintV2"
    )
    if (
        domain_symbol == NULL
        or fingerprint_symbol == NULL
        or dladdr(domain_symbol, &domain_info) == 0
        or dladdr(fingerprint_symbol, &fingerprint_info) == 0
        or domain_info.dli_fbase != info.dli_fbase
        or fingerprint_info.dli_fbase != info.dli_fbase
    ):
        _compact_domains_dlclose_calls += 1
        dlclose(handle)
        return
    _compact_domains_same_dso = True
    _compact_domains_seam_cache = (
        <_pipeline_from_filter_forward_compact_domains_f> domain_symbol
    )
    _compact_tail_fingerprint_cache = (
        <_pipeline_compact_tail_fingerprint_f> fingerprint_symbol
    )
    _compact_domains_dlclose_calls += 1
    dlclose(handle)


cdef _pipeline_from_filter_forward_compact_domains_f _cached_compact_domains_seam():
    global _compact_domains_resolutions
    global _compact_domains_seam_resolved

    if not _compact_domains_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _compact_domains_seam_resolved:
                _compact_domains_resolutions += 1
                with nogil:
                    _resolve_compact_domain_seams()
                _compact_domains_seam_resolved = True
    return _compact_domains_seam_cache


def _compact_domains_seam_available():
    """Return whether the same-DSO compact-domain seam pair is loaded."""
    return (
        _cached_compact_domains_seam() != NULL
        and _compact_tail_fingerprint_cache != NULL
    )


def _compact_tail_fingerprint_bound(Pipeline pipeline):
    """Capture the exact same-DSO compact-tail option fingerprint."""
    cdef uint64_t fingerprint
    if _cached_compact_domains_seam() == NULL:
        raise RuntimeError("the private compact-domain seam is unavailable")
    if _compact_tail_fingerprint_cache == NULL:
        raise RuntimeError("the compact-tail fingerprint seam is unavailable")
    with nogil:
        fingerprint = _compact_tail_fingerprint_cache(pipeline._pli)
    if fingerprint == 0:
        raise ValueError("pipeline compact-tail options are invalid")
    return fingerprint


def _continuation_seam_cache_info():
    """Return private resolver state for concurrency and lifetime tests."""
    return {
        "filter": {
            "resolved": bool(_filter_scores_seam_resolved),
            "available": _filter_scores_seam_cache != NULL,
            "same_dso": bool(_filter_scores_same_dso),
            "resolutions": _filter_scores_resolutions,
            "dlopen_calls": _filter_scores_dlopen_calls,
            "dlclose_calls": _filter_scores_dlclose_calls,
        },
        "forward": {
            "resolved": bool(_forward_scores_seam_resolved),
            "available": _forward_scores_seam_cache != NULL,
            "same_dso": bool(_forward_scores_same_dso),
            "resolutions": _forward_scores_resolutions,
            "dlopen_calls": _forward_scores_dlopen_calls,
            "dlclose_calls": _forward_scores_dlclose_calls,
        },
        "simple_regions": {
            "resolved": bool(_simple_regions_seam_resolved),
            "available": _simple_regions_seam_cache != NULL,
            "same_dso": bool(_simple_regions_same_dso),
            "resolutions": _simple_regions_resolutions,
            "dlopen_calls": _simple_regions_dlopen_calls,
            "dlclose_calls": _simple_regions_dlclose_calls,
        },
        "compact_domains": {
            "resolved": bool(_compact_domains_seam_resolved),
            "available": (
                _compact_domains_seam_cache != NULL
                and _compact_tail_fingerprint_cache != NULL
            ),
            "same_dso": bool(_compact_domains_same_dso),
            "resolutions": _compact_domains_resolutions,
            "dlopen_calls": _compact_domains_dlopen_calls,
            "dlclose_calls": _compact_domains_dlclose_calls,
        },
    }


cdef void _capture_pipeline_tail_options(
    Pipeline pipeline,
    _pipeline_tail_snapshot *snapshot,
) except *:
    cdef _double_bits encoded_double
    cdef _float_bits encoded_float
    cdef P7_PIPELINE *pli = pipeline._pli

    if pli == NULL or pli.ddef == NULL:
        raise RuntimeError("pipeline domain state is unavailable")
    encoded_double.value = pli.F1
    snapshot.f1_bits = encoded_double.bits
    encoded_double.value = pli.F2
    snapshot.f2_bits = encoded_double.bits
    encoded_double.value = pli.F3
    snapshot.f3_bits = encoded_double.bits
    encoded_double.value = pli.E
    snapshot.E_bits = encoded_double.bits
    encoded_double.value = pli.T
    snapshot.T_bits = encoded_double.bits
    encoded_double.value = pli.domE
    snapshot.domE_bits = encoded_double.bits
    encoded_double.value = pli.domT
    snapshot.domT_bits = encoded_double.bits
    encoded_double.value = pli.incE
    snapshot.incE_bits = encoded_double.bits
    encoded_double.value = pli.incT
    snapshot.incT_bits = encoded_double.bits
    encoded_double.value = pli.incdomE
    snapshot.incdomE_bits = encoded_double.bits
    encoded_double.value = pli.incdomT
    snapshot.incdomT_bits = encoded_double.bits
    encoded_double.value = pli.Z
    snapshot.Z_bits = encoded_double.bits
    encoded_double.value = pli.domZ
    snapshot.domZ_bits = encoded_double.bits
    encoded_float.value = pli.ddef.rt1
    snapshot.rt1_bits = encoded_float.bits
    encoded_float.value = pli.ddef.rt2
    snapshot.rt2_bits = encoded_float.bits
    encoded_float.value = pli.ddef.rt3
    snapshot.rt3_bits = encoded_float.bits
    snapshot.do_biasfilter = pli.do_biasfilter
    snapshot.do_null2 = pli.do_null2
    snapshot.do_alignment_score_calc = pli.do_alignment_score_calc
    snapshot.by_E = pli.by_E
    snapshot.dom_by_E = pli.dom_by_E
    snapshot.inc_by_E = pli.inc_by_E
    snapshot.incdom_by_E = pli.incdom_by_E
    snapshot.use_bit_cutoffs = pli.use_bit_cutoffs
    snapshot.Z_setby = pli.Z_setby
    snapshot.domZ_setby = pli.domZ_setby
    snapshot.mode = pli.mode
    snapshot.long_targets = pli.long_targets


cdef bint _pipeline_tail_options_match(
    const _pipeline_tail_snapshot *expected,
    Pipeline pipeline,
) except -1:
    cdef _pipeline_tail_snapshot observed
    _capture_pipeline_tail_options(pipeline, &observed)
    if (
        observed.f1_bits != expected.f1_bits
        or observed.f2_bits != expected.f2_bits
        or observed.f3_bits != expected.f3_bits
        or observed.E_bits != expected.E_bits
        or observed.domE_bits != expected.domE_bits
        or observed.incE_bits != expected.incE_bits
        or observed.incdomE_bits != expected.incdomE_bits
        or observed.rt1_bits != expected.rt1_bits
        or observed.rt2_bits != expected.rt2_bits
        or observed.rt3_bits != expected.rt3_bits
        or observed.do_biasfilter != expected.do_biasfilter
        or observed.do_null2 != expected.do_null2
        or observed.do_alignment_score_calc
        != expected.do_alignment_score_calc
        or observed.by_E != expected.by_E
        or observed.dom_by_E != expected.dom_by_E
        or observed.inc_by_E != expected.inc_by_E
        or observed.incdom_by_E != expected.incdom_by_E
        or observed.use_bit_cutoffs != expected.use_bit_cutoffs
        or observed.Z_setby != expected.Z_setby
        or observed.domZ_setby != expected.domZ_setby
        or observed.mode != expected.mode
        or observed.long_targets != expected.long_targets
    ):
        return False
    if expected.use_bit_cutoffs == 0 and (
        observed.T_bits != expected.T_bits
        or observed.domT_bits != expected.domT_bits
        or observed.incT_bits != expected.incT_bits
        or observed.incdomT_bits != expected.incdomT_bits
    ):
        return False
    if expected.Z_setby != p7_ZSETBY_NTARGETS and (
        observed.Z_bits != expected.Z_bits
    ):
        return False
    if expected.domZ_setby != p7_ZSETBY_NTARGETS and (
        observed.domZ_bits != expected.domZ_bits
    ):
        return False
    return True


def _validate_simple_region_generation_bound(
    Pipeline pipeline,
    double f1,
    double f2,
    double f3,
    bias_filter,
    double guard_band,
):
    """Validate product-domain options without mutating the pipeline."""
    cdef _pipeline_tail_snapshot snapshot
    cdef _double_bits encoded
    cdef _float_bits observed_float
    cdef _float_bits expected_float

    if type(bias_filter) is not bool or not bias_filter:
        raise ValueError("simple-region generation requires bias filtering")
    if (
        not isfinite(f1) or f1 < 0.0 or f1 > 1.0
        or not isfinite(f2) or f2 < 0.0 or f2 > 1.0
        or not isfinite(f3) or f3 < 0.0 or f3 > 1.0
        or not isfinite(guard_band)
        or guard_band < 2.0e-4 or guard_band > 1.0
    ):
        raise ValueError("invalid simple-region generation thresholds")
    _capture_pipeline_tail_options(pipeline, &snapshot)
    encoded.value = f1
    if encoded.bits != snapshot.f1_bits:
        raise ValueError("pipeline F1 differs from journal generation F1")
    encoded.value = f2
    if encoded.bits != snapshot.f2_bits:
        raise ValueError("pipeline F2 differs from journal generation F2")
    encoded.value = f3
    if encoded.bits != snapshot.f3_bits:
        raise ValueError("pipeline F3 differs from journal generation F3")
    if not snapshot.do_biasfilter:
        raise ValueError("pipeline bias filter is disabled")
    if snapshot.mode != p7_SEARCH_SEQS or snapshot.long_targets:
        raise ValueError("simple-region generation requires sequence-search mode")
    expected_float.value = <float> 0.25
    if snapshot.rt1_bits != expected_float.bits:
        raise ValueError("pipeline rt1 is not the calibrated default")
    expected_float.value = <float> 0.10
    if snapshot.rt2_bits != expected_float.bits:
        raise ValueError("pipeline rt2 is not the calibrated default")
    expected_float.value = <float> 0.20
    if snapshot.rt3_bits != expected_float.bits:
        raise ValueError("pipeline rt3 is not the calibrated default")
    observed_float.value = <float> guard_band
    return (0.25, 0.10, 0.20, observed_float.value)


cdef int _hmmer_f2_decision(
    const P7_OPROFILE *profile,
    const _postfilter_result *record,
    double f2,
) noexcept nogil:
    """Replay HMMER's exact post-bias/F2 predicates.

    Return 1 for an F2 survivor, 0 for an exact F2 reject, and -1 when the
    supplied source row cannot enter the exact external-score seam.  Keeping
    this helper shared by Forward selection and journal-v3 planning prevents a
    second, approximate accounting classifier from drifting away from HMMER.
    """
    cdef _float_bits vfsc_bits
    cdef float usc
    cdef float bit_score
    cdef double probability

    if (
        profile == NULL
        or record == NULL
        or record.action != BIAS_DEFINITE_PASS
        or record.msv_status != SSV_OK
        or not isfinite(record.filtersc)
        or not isfinite(profile.scale_b)
        or profile.scale_b <= 0.0
    ):
        return -1
    vfsc_bits.value = record.vfsc
    if not isfinite(record.vfsc) and vfsc_bits.bits != 0x7f800000:
        return -1

    usc = <float> record.msv_numerator
    usc = usc / profile.scale_b
    usc = usc - <float> 3.0
    bit_score = <float> ((usc - record.filtersc) / eslCONST_LOG2)
    probability = esl_gumbel_surv(
        bit_score,
        profile.evparam[<int> p7_MMU],
        profile.evparam[<int> p7_MLAMBDA],
    )
    if probability > f2:
        bit_score = <float> (
            (record.vfsc - record.filtersc) / eslCONST_LOG2
        )
        probability = esl_gumbel_surv(
            bit_score,
            profile.evparam[<int> p7_VMU],
            profile.evparam[<int> p7_VLAMBDA],
        )
        if probability > f2:
            return 0
    return 1


cdef int _hmmer_f3_decision(
    const P7_OPROFILE *profile,
    float filtersc,
    float fwdsc,
    double f3,
) noexcept nogil:
    """Replay the exact host predicate used by Forward result classification."""
    cdef float difference
    cdef float bit_score
    cdef double probability

    if (
        profile == NULL
        or not isfinite(filtersc)
        or not isfinite(fwdsc)
        or not isfinite(profile.evparam[<int> p7_FTAU])
        or not isfinite(profile.evparam[<int> p7_FLAMBDA])
        or profile.evparam[<int> p7_FLAMBDA] <= 0.0
    ):
        return -1
    difference = fwdsc - filtersc
    bit_score = <float> (<double> difference / eslCONST_LOG2)
    probability = esl_exp_surv(
        bit_score,
        profile.evparam[<int> p7_FTAU],
        profile.evparam[<int> p7_FLAMBDA],
    )
    return 0 if probability > f3 else 1


def _select_forward_inputs_bound(
    profiles,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] postfilter_offsets,
    const uint64_t[::1] residue_offsets,
    double f2,
):
    """Select exact F2 survivors from trusted profile-major post-filter rows.

    Rows that fail F2 remain in the original post-filter stream and therefore
    take the ordinary CPU Forward continuation. The native Forward runner caps
    actual gathered F3-pass matrices; charging all F2 survivors here would
    discard work that ultimately needs no matrix.
    """
    cdef size_t profile_count = len(profiles)
    cdef size_t profile_index
    cdef size_t cursor
    cdef size_t start
    cdef size_t stop
    cdef size_t record_count
    cdef size_t sequence_count
    cdef uint32_t previous
    cdef bint have_previous
    cdef uint64_t sequence_length
    cdef _postfilter_result record
    cdef OptimizedProfile profile
    cdef object value
    cdef object candidate_offsets = _array("Q", [0])
    cdef object candidate_indices = _array("I")
    cdef object filter_scores = _array("f")

    if sizeof(_postfilter_result) != 16:
        raise RuntimeError("native post-filter result ABI is not 16 bytes")
    if postfilter_offsets.shape[0] != profile_count + 1:
        raise ValueError("post-filter row-offset count differs from profiles")
    if postfilter_offsets[0] != 0:
        raise ValueError("post-filter row offsets must start at zero")
    if <size_t> postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result storage has trailing bytes")
    record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    if postfilter_offsets[profile_count] != record_count:
        raise ValueError("post-filter row offsets do not span result storage")
    if residue_offsets.shape[0] == 0 or residue_offsets[0] != 0:
        raise ValueError("target residue prefix must contain an initial zero")
    sequence_count = <size_t> residue_offsets.shape[0] - 1

    for profile_index in range(profile_count):
        start = <size_t> postfilter_offsets[profile_index]
        stop = <size_t> postfilter_offsets[profile_index + 1]
        if start > stop or stop > record_count:
            raise ValueError("post-filter row offsets are not monotone")
        value = profiles[profile_index]
        if not isinstance(value, _pyhmmer.plan7.OptimizedProfile):
            raise TypeError("Forward source profiles must be OptimizedProfile objects")
        profile = value
        previous = 0
        have_previous = False
        for cursor in range(start, stop):
            memcpy(
                &record,
                &postfilter_records[cursor * sizeof(_postfilter_result)],
                sizeof(_postfilter_result),
            )
            if record.sequence_index >= sequence_count:
                raise IndexError(
                    f"post-filter result sequence index out of range: "
                    f"{record.sequence_index}"
                )
            if have_previous and record.sequence_index <= previous:
                raise ValueError(
                    "post-filter result sequence indexes must be strictly "
                    "increasing and unique within each row"
                )
            previous = record.sequence_index
            have_previous = True

            if record.action != BIAS_DEFINITE_PASS:
                continue
            if _hmmer_f2_decision(profile._om, &record, f2) != 1:
                continue

            if residue_offsets[record.sequence_index + 1] < (
                residue_offsets[record.sequence_index]
            ):
                raise ValueError("target residue prefix is not monotone")
            sequence_length = (
                residue_offsets[record.sequence_index + 1]
                - residue_offsets[record.sequence_index]
            )
            if sequence_length > HMMER_TARGET_LIMIT:
                raise ValueError("Forward target exceeds HMMER's protein limit")
            candidate_indices.append(record.sequence_index)
            filter_scores.append(record.filtersc)
        candidate_offsets.append(len(candidate_indices))

    return candidate_offsets, candidate_indices, filter_scores


cdef int _search_loop_candidates(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint32_t* candidate_indexes,
    size_t candidate_count,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
) except 1 nogil:
    """Mirror ``Pipeline._search_loop`` while skipping definite rejects."""
    cdef int status
    cdef size_t candidate_cursor
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for candidate_cursor in range(candidate_count):
        t = candidate_indexes[candidate_cursor]
        target_end = t + 1

        # Advance exactly the p7_pli_NewSeq accounting through this target.
        # Prefix residue offsets make gaps between candidates O(1), while Z
        # still has the value HMMER would see at this target's ordinal.
        if not pli.long_targets:
            pli.nseqs += target_end - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[target_end] - residue_offsets[previous_end]

        # A rejected prefix would have reused any residual workspaces before
        # this first candidate in HMMER's ordinary target loop.
        if candidate_cursor == 0 and t != 0:
            p7_pipeline_Reuse(pli)

        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
        status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        if status == eslEINVAL:
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    # Account the rejected tail in one step.
    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

    # HMMER leaves both models configured for the final target, even when it
    # fails F1. Preserve that state with one configuration for a rejected tail.
    if (
        n_targets != 0
        and (
            candidate_count == 0
            or candidate_indexes[candidate_count - 1] != n_targets - 1
        )
    ):
        status = p7_bg_SetLength(bg, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
        p7_pipeline_Reuse(pli)

    return 0


cdef int _search_loop_bias(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint8_t* record_bytes,
    size_t record_count,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
    _pipeline_from_filter_scores_f filter_scores_seam,
) except 1 nogil:
    """Run one validated sparse bias row through HMMER's exact pipeline."""
    cdef _bias_result record
    cdef float usc
    cdef int status
    cdef size_t cursor
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for cursor in range(record_count):
        memcpy(
            &record,
            record_bytes + cursor * sizeof(_bias_result),
            sizeof(_bias_result),
        )
        t = record.sequence_index
        target_end = t + 1

        if not pli.long_targets:
            pli.nseqs += target_end - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[target_end] - residue_offsets[previous_end]

        if cursor == 0 and t != 0:
            p7_pipeline_Reuse(pli)

        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")

        if record.action == BIAS_CPU_REQUIRED:
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        else:
            usc = <float> record.ssv_numerator
            usc = usc / om.scale_b
            usc = usc - <float> 3.0
            status = filter_scores_seam(
                pli,
                om,
                bg,
                sq[t],
                NULL,
                th,
                usc,
                record.filtersc,
                P7_VIT_CPU,
                <float> 0.0,
            )
        if status == eslEINVAL:
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

    if (
        n_targets != 0
        and (
            record_count == 0
            or record.sequence_index != n_targets - 1
        )
    ):
        status = p7_bg_SetLength(bg, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
        p7_pipeline_Reuse(pli)

    return 0


cdef int _search_loop_postfilter(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint8_t* record_bytes,
    size_t record_count,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
    _pipeline_from_filter_scores_f filter_scores_seam,
) except 1 nogil:
    cdef _postfilter_result record
    cdef float usc
    cdef int status
    cdef size_t cursor
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for cursor in range(record_count):
        memcpy(
            &record,
            record_bytes + cursor * sizeof(_postfilter_result),
            sizeof(_postfilter_result),
        )
        t = record.sequence_index
        target_end = t + 1

        if not pli.long_targets:
            pli.nseqs += target_end - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[target_end] - residue_offsets[previous_end]

        if cursor == 0 and t != 0:
            p7_pipeline_Reuse(pli)

        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")

        if record.action == BIAS_CPU_REQUIRED:
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        elif isnan(record.filtersc):
            # A fallback full-MSV score can reject at F1 after the public SSV
            # path conservatively retained it. Its row is accounted but does
            # not enter the pipeline continuation.
            status = eslOK
        else:
            usc = <float> record.msv_numerator
            usc = usc / om.scale_b
            usc = usc - <float> 3.0
            status = filter_scores_seam(
                pli,
                om,
                bg,
                sq[t],
                NULL,
                th,
                usc,
                record.filtersc,
                P7_VIT_EXTERNAL,
                record.vfsc,
            )
        if status == eslEINVAL:
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

    if (
        n_targets != 0
        and (
            record_count == 0
            or record.sequence_index != n_targets - 1
        )
    ):
        status = p7_bg_SetLength(bg, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
        p7_pipeline_Reuse(pli)

    return 0


cdef int _search_loop_postfilter_forward(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint8_t* postfilter_bytes,
    size_t postfilter_count,
    const uint8_t* forward_bytes,
    size_t forward_count,
    const uint64_t* special_offsets,
    const float* specials,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
    const uint8_t* journal_bytes,
    size_t journal_count,
    const uint64_t* journal_region_offsets,
    const uint8_t* journal_region_bytes,
    uint64_t generation_f1_bits,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    int generation_bias_filter,
    _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam,
    const uint64_t* journal_compact_row_offsets,
    const uint8_t* journal_compact_result_bytes,
    const uint64_t* compact_trace_offsets,
    const uint8_t* compact_trace_bytes,
    const float* compact_null2,
    uint64_t journal_row_base,
    uint64_t generation_tail_fingerprint,
    _pipeline_compact_tail_fingerprint_f compact_tail_fingerprint,
    _pipeline_from_filter_forward_compact_domains_f compact_domains_seam,
    uint64_t* compact_rebased_offsets,
    _compact_consumption_statistics* compact_statistics,
) except 1 nogil:
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef const float* xmx = NULL
    cdef uint64_t xmx_count = 0
    cdef float usc
    cdef int status
    cdef size_t cursor
    cdef size_t forward_cursor = 0
    cdef size_t journal_cursor = 0
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t
    cdef bint has_forward
    cdef bint used_forward_seam
    cdef bint used_simple_regions_seam
    cdef bint used_compact_domains_seam
    cdef bint compact_generation_matches = False
    cdef bint has_journal
    cdef plan7_continuation_journal_row journal
    cdef const plan7_simple_region* regions = NULL
    cdef uint64_t region_start
    cdef uint64_t region_stop
    cdef uint64_t compact_start
    cdef uint64_t compact_stop
    cdef uint64_t compact_count
    cdef uint64_t compact_index
    cdef uint64_t compact_trace_base
    cdef uint64_t compact_trace_stop
    cdef const plan7_domain_rescore_result* compact_domains = NULL
    cdef const plan7_domain_rescore_trace_step* compact_traces = NULL
    cdef const float* compact_row_null2 = NULL

    if (
        compact_tail_fingerprint != NULL
        and compact_domains_seam != NULL
        and generation_tail_fingerprint != 0
        and compact_tail_fingerprint(pli) == generation_tail_fingerprint
    ):
        compact_generation_matches = True

    if compact_statistics != NULL:
        if postfilter_count > n_targets:
            raise RuntimeError("continuation post-filter rows exceed targets")
        compact_statistics.target_count = n_targets
        compact_statistics.postfilter_record_count = postfilter_count
        compact_statistics.f1_reject_count = n_targets - postfilter_count

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for cursor in range(postfilter_count):
        memcpy(
            &postfilter,
            postfilter_bytes + cursor * sizeof(_postfilter_result),
            sizeof(_postfilter_result),
        )
        t = postfilter.sequence_index
        target_end = t + 1

        if not pli.long_targets:
            pli.nseqs += target_end - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[target_end] - residue_offsets[previous_end]

        if cursor == 0 and t != 0:
            p7_pipeline_Reuse(pli)

        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")

        has_forward = False
        if forward_cursor < forward_count:
            memcpy(
                &forward,
                forward_bytes + forward_cursor * sizeof(_forward_result),
                sizeof(_forward_result),
            )
            has_forward = forward.sequence_index == postfilter.sequence_index

        has_journal = False
        if journal_cursor < journal_count:
            memcpy(
                &journal,
                journal_bytes
                + journal_cursor * sizeof(plan7_continuation_journal_row),
                sizeof(plan7_continuation_journal_row),
            )
            has_journal = journal.sequence_index == postfilter.sequence_index
        if has_journal and compact_statistics != NULL:
            compact_statistics.journal_match_count += 1
            if (
                journal.domain_status != DOMAIN_OK
                or journal.domain_route == DOMAIN_CPU_REQUIRED
                or journal.has_own_scales
                or journal.uncertain_count != 0
                or journal.multidomain_count != 0
            ):
                compact_statistics.journal_cpu_required_count += 1
            elif journal.domain_route == DOMAIN_NO_REGIONS:
                compact_statistics.journal_no_region_count += 1
            elif journal.domain_route == DOMAIN_SIMPLE:
                compact_statistics.journal_simple_count += 1

        used_forward_seam = False
        used_simple_regions_seam = False
        used_compact_domains_seam = False
        if postfilter.action == BIAS_CPU_REQUIRED:
            if compact_statistics != NULL:
                compact_statistics.cpu_pipeline_count += 1
                compact_statistics.source_postfilter_cpu_count += 1
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        elif isnan(postfilter.filtersc):
            if compact_statistics != NULL:
                compact_statistics.definite_reject_count += 1
                compact_statistics.source_definite_reject_count += 1
            status = eslOK
        else:
            usc = <float> postfilter.msv_numerator
            usc = usc / om.scale_b
            usc = usc - <float> 3.0
            if (
                has_forward
                and has_journal
                and forward.action == FORWARD_DEFINITE_PASS
                and simple_regions_seam != NULL
                and journal.domain_status == DOMAIN_OK
                and (
                    journal.domain_route == DOMAIN_NO_REGIONS
                    or journal.domain_route == DOMAIN_SIMPLE
                )
                and not journal.has_own_scales
                and journal.uncertain_count == 0
                and journal.multidomain_count == 0
            ):
                if compact_statistics != NULL:
                    compact_statistics.source_journal_eligible_count += 1
                region_start = journal_region_offsets[journal_cursor]
                region_stop = journal_region_offsets[journal_cursor + 1]
                if region_start == region_stop:
                    regions = NULL
                else:
                    regions = <const plan7_simple_region *> (
                        journal_region_bytes
                        + region_start * sizeof(plan7_simple_region)
                    )
                compact_start = journal_compact_row_offsets[journal_cursor]
                compact_stop = journal_compact_row_offsets[journal_cursor + 1]
                compact_count = compact_stop - compact_start
                if (
                    journal.domain_route == DOMAIN_SIMPLE
                    and journal.compact_route
                    == PLAN7_CONTINUATION_COMPACT_DEVICE
                    and compact_count != 0
                    and compact_generation_matches
                    and compact_rebased_offsets != NULL
                ):
                    if compact_statistics != NULL:
                        if compact_statistics.attempt_count == 0:
                            compact_statistics.first_row_index = (
                                journal_row_base + journal_cursor
                            )
                            compact_statistics.first_profile_index = (
                                journal.profile_index
                            )
                            compact_statistics.first_sequence_index = (
                                journal.sequence_index
                            )
                            compact_statistics.first_domain_count = (
                                compact_count
                            )
                        compact_statistics.attempt_count += 1
                    compact_trace_base = compact_trace_offsets[compact_start]
                    compact_trace_stop = compact_trace_offsets[compact_stop]
                    for compact_index in range(compact_count + 1):
                        compact_rebased_offsets[compact_index] = (
                            compact_trace_offsets[
                                compact_start + compact_index
                            ]
                            - compact_trace_base
                        )
                    compact_domains = (
                        <const plan7_domain_rescore_result *> (
                            journal_compact_result_bytes
                            + compact_start
                            * sizeof(plan7_domain_rescore_result)
                        )
                    )
                    compact_traces = (
                        <const plan7_domain_rescore_trace_step *> (
                            compact_trace_bytes
                            + compact_trace_base
                            * sizeof(plan7_domain_rescore_trace_step)
                        )
                    )
                    compact_row_null2 = (
                        compact_null2
                        + compact_start * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                    )
                    status = compact_domains_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        journal.usc,
                        journal.filtersc,
                        journal.vfsc,
                        journal.fwdsc,
                        generation_tail_fingerprint,
                        n_targets,
                        <uint32_t> (journal_row_base + journal_cursor),
                        journal.profile_index,
                        journal.sequence_index,
                        journal.nexpected,
                        compact_domains,
                        compact_count,
                        compact_rebased_offsets,
                        compact_count + 1,
                        compact_traces,
                        compact_trace_stop - compact_trace_base,
                        compact_row_null2,
                        compact_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
                    )
                    used_compact_domains_seam = True
                    if status == eslEINACCURATE:
                        if compact_statistics != NULL:
                            compact_statistics.threshold_retry_count += 1
                            compact_statistics.cpu_pipeline_count += 1
                        # The guard covers uncertainty in both the compact
                        # domains and the upstream Forward score. Recompute
                        # the entire native pipeline so the retry is exact.
                        status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
                        used_compact_domains_seam = False
                    elif status == eslEINVAL:
                        if compact_statistics != NULL:
                            compact_statistics.invalid_retry_count += 1
                            compact_statistics.forward_continuation_count += 1
                        xmx_count = (
                            special_offsets[forward_cursor + 1]
                            - special_offsets[forward_cursor]
                        )
                        if xmx_count == 0:
                            xmx = NULL
                        else:
                            xmx = specials + special_offsets[forward_cursor]
                        status = forward_scores_seam(
                            pli,
                            om,
                            bg,
                            sq[t],
                            NULL,
                            th,
                            usc,
                            postfilter.filtersc,
                            postfilter.vfsc,
                            forward.fwdsc,
                            xmx,
                            xmx_count,
                        )
                        used_compact_domains_seam = False
                        used_forward_seam = True
                    elif status == eslOK and compact_statistics != NULL:
                        compact_statistics.accepted_count += 1
                else:
                    if compact_statistics != NULL:
                        compact_statistics.simple_continuation_count += 1
                        compact_statistics.source_simple_bypass_count += 1
                        if journal.domain_route != DOMAIN_SIMPLE or (
                            journal.compact_route
                            != PLAN7_CONTINUATION_COMPACT_DEVICE
                        ):
                            compact_statistics.decision_compact_route_not_device += 1
                        if compact_count == 0:
                            compact_statistics.decision_compact_empty += 1
                        if not compact_generation_matches:
                            compact_statistics.decision_compact_tail_changed += 1
                        if compact_rebased_offsets == NULL:
                            compact_statistics.decision_compact_rebase_unavailable += 1
                    status = simple_regions_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        journal.usc,
                        journal.filtersc,
                        journal.vfsc,
                        journal.fwdsc,
                        generation_f1_bits,
                        generation_f2_bits,
                        generation_f3_bits,
                        generation_bias_filter,
                        journal.domain_route,
                        journal.nexpected,
                        regions,
                        region_stop - region_start,
                    )
                    used_simple_regions_seam = True
            elif has_forward and forward.action != FORWARD_CPU_REQUIRED:
                if compact_statistics != NULL:
                    compact_statistics.forward_continuation_count += 1
                    compact_statistics.source_forward_count += 1
                xmx_count = (
                    special_offsets[forward_cursor + 1]
                    - special_offsets[forward_cursor]
                )
                if xmx_count == 0:
                    xmx = NULL
                else:
                    xmx = specials + special_offsets[forward_cursor]
                status = forward_scores_seam(
                    pli,
                    om,
                    bg,
                    sq[t],
                    NULL,
                    th,
                    usc,
                    postfilter.filtersc,
                    postfilter.vfsc,
                    forward.fwdsc,
                    xmx,
                    xmx_count,
                )
                used_forward_seam = True
            else:
                if compact_statistics != NULL:
                    compact_statistics.filter_continuation_count += 1
                    compact_statistics.source_filter_count += 1
                status = filter_scores_seam(
                    pli,
                    om,
                    bg,
                    sq[t],
                    NULL,
                    th,
                    usc,
                    postfilter.filtersc,
                    P7_VIT_EXTERNAL,
                    postfilter.vfsc,
                )

        if has_forward:
            forward_cursor += 1
        if has_journal:
            journal_cursor += 1
        if status == eslEINVAL:
            if used_simple_regions_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterAndForwardSimpleRegions",
                )
            elif used_compact_domains_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterForwardAndCompactDomainsV2",
                )
            elif used_forward_seam:
                raise UnexpectedError(
                    status, "p7_PipelineFromFilterAndForwardScores"
                )
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    if journal_cursor != journal_count:
        raise ValueError("continuation journal row was not consumed")

    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

    if (
        n_targets != 0
        and (
            postfilter_count == 0
            or postfilter.sequence_index != n_targets - 1
        )
    ):
        status = p7_bg_SetLength(bg, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
        p7_pipeline_Reuse(pli)

    return 0


cdef void _validate_inputs(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    bint validate_target_lengths,
) except *:
    """Validate every input without mutating the supplied pipeline."""
    if not pipeline.alphabet._eq(query.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, query.alphabet)
    if not pipeline.alphabet._eq(optimized_profile.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, optimized_profile.alphabet)
    if not pipeline.alphabet._eq(sequences.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sequences.alphabet)

    if query._hmm.M != optimized_profile._om.M:
        raise ValueError(
            "HMM and optimized profile model lengths differ: "
            f"{query._hmm.M} != {optimized_profile._om.M}"
        )
    if (
        query.name != optimized_profile.name
        or query.accession != optimized_profile.accession
    ):
        raise ValueError("HMM and optimized profile metadata differ")

    if (
        validate_target_lengths
        and sequences._length != 0
        and len(sequences.largest()) > HMMER_TARGET_LIMIT
    ):
        raise ValueError(
            f"sequence length over comparison pipeline limit "
            f"({HMMER_TARGET_LIMIT})"
        )


cdef TopHits _search_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
    const uint64_t* residue_offsets,
):
    cdef const uint32_t* candidate_ptr = NULL
    cdef TopHits hits = TopHits(query)

    if candidate_indexes.shape[0] != 0:
        candidate_ptr = &candidate_indexes[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_candidates(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            candidate_ptr,
            <size_t> candidate_indexes.shape[0],
            residue_offsets,
            hits._th,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    # Preserve Astra/PyHMMER query metadata and object identity.  In
    # particular, never expose the lockstep OptimizedProfile as hits.query.
    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _search_bias_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
    const uint64_t* residue_offsets,
    _pipeline_from_filter_scores_f filter_scores_seam,
):
    cdef const uint8_t* record_ptr = NULL
    cdef size_t record_count = (
        <size_t> bias_records.shape[0] // sizeof(_bias_result)
    )
    cdef TopHits hits = TopHits(query)

    if bias_records.shape[0] != 0:
        record_ptr = &bias_records[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_bias(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            record_ptr,
            record_count,
            residue_offsets,
            hits._th,
            filter_scores_seam,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _search_postfilter_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t* residue_offsets,
    _pipeline_from_filter_scores_f filter_scores_seam,
):
    cdef const uint8_t* record_ptr = NULL
    cdef size_t record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef TopHits hits = TopHits(query)

    if postfilter_records.shape[0] != 0:
        record_ptr = &postfilter_records[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_postfilter(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            record_ptr,
            record_count,
            residue_offsets,
            hits._th,
            filter_scores_seam,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _search_postfilter_forward_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint64_t* residue_offsets,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
):
    cdef const uint8_t* postfilter_ptr = NULL
    cdef const uint8_t* forward_ptr = NULL
    cdef const float* special_ptr = NULL
    cdef size_t postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef size_t forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    cdef TopHits hits = TopHits(query)

    if postfilter_records.shape[0] != 0:
        postfilter_ptr = &postfilter_records[0]
    if forward_records.shape[0] != 0:
        forward_ptr = &forward_records[0]
    if specials.shape[0] != 0:
        special_ptr = &specials[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_postfilter_forward(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            postfilter_ptr,
            postfilter_count,
            forward_ptr,
            forward_count,
            &special_offsets[0],
            special_ptr,
            residue_offsets,
            hits._th,
            filter_scores_seam,
            forward_scores_seam,
            NULL,
            0,
            NULL,
            NULL,
            0,
            0,
            0,
            0,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            0,
            0,
            NULL,
            NULL,
            NULL,
            NULL,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _search_postfilter_forward_journal_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint8_t[::1] journal_rows,
    const uint64_t[::1] journal_region_offsets,
    const uint8_t[::1] journal_regions,
    const uint64_t[::1] journal_compact_row_offsets,
    const uint8_t[::1] journal_compact_results,
    const uint64_t[::1] journal_compact_trace_offsets,
    const uint8_t[::1] journal_compact_traces,
    const float[::1] journal_compact_null2,
    uint64_t journal_row_base,
    uint64_t generation_tail_fingerprint,
    const uint64_t* residue_offsets,
    uint64_t generation_f1_bits,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    int generation_bias_filter,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
    _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam,
    _pipeline_compact_tail_fingerprint_f compact_tail_fingerprint,
    _pipeline_from_filter_forward_compact_domains_f compact_domains_seam,
    _compact_consumption_statistics* compact_statistics,
):
    cdef const uint8_t* postfilter_ptr = NULL
    cdef const uint8_t* forward_ptr = NULL
    cdef const float* special_ptr = NULL
    cdef const uint8_t* journal_ptr = NULL
    cdef const uint8_t* region_ptr = NULL
    cdef const uint8_t* compact_result_ptr = NULL
    cdef const uint8_t* compact_trace_ptr = NULL
    cdef const float* compact_null2_ptr = NULL
    cdef uint64_t* compact_rebased_offsets = NULL
    cdef size_t compact_profile_count
    cdef size_t postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef size_t forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    cdef size_t journal_count = (
        <size_t> journal_rows.shape[0]
        // sizeof(plan7_continuation_journal_row)
    )
    cdef TopHits hits = TopHits(query)

    if postfilter_records.shape[0]:
        postfilter_ptr = &postfilter_records[0]
    if forward_records.shape[0]:
        forward_ptr = &forward_records[0]
    if specials.shape[0]:
        special_ptr = &specials[0]
    if journal_rows.shape[0]:
        journal_ptr = &journal_rows[0]
    if journal_regions.shape[0]:
        region_ptr = &journal_regions[0]
    if journal_compact_results.shape[0]:
        compact_result_ptr = &journal_compact_results[0]
    if journal_compact_traces.shape[0]:
        compact_trace_ptr = &journal_compact_traces[0]
    if journal_compact_null2.shape[0]:
        compact_null2_ptr = &journal_compact_null2[0]
    compact_profile_count = <size_t> (
        journal_compact_row_offsets[journal_count]
        - journal_compact_row_offsets[0]
    )
    if compact_profile_count > (<size_t> -1) // sizeof(uint64_t) - 1:
        raise OverflowError("compact trace-offset scratch size overflows")
    compact_rebased_offsets = <uint64_t*> malloc(
        (compact_profile_count + 1) * sizeof(uint64_t)
    )
    if compact_rebased_offsets == NULL:
        raise MemoryError("compact trace-offset scratch allocation failed")

    try:
        with nogil:
            pipeline._pli.mode = p7_SEARCH_SEQS
            pipeline._pli.nseqs = 0
            _search_loop_postfilter_forward(
                pipeline._pli,
                optimized_profile._om,
                pipeline.background._bg,
                <const ESL_SQ**> sequences._refs,
                sequences._length,
                postfilter_ptr,
                postfilter_count,
                forward_ptr,
                forward_count,
                &special_offsets[0],
                special_ptr,
                residue_offsets,
                hits._th,
                filter_scores_seam,
                forward_scores_seam,
                journal_ptr,
                journal_count,
                &journal_region_offsets[0],
                region_ptr,
                generation_f1_bits,
                generation_f2_bits,
                generation_f3_bits,
                generation_bias_filter,
                simple_regions_seam,
                &journal_compact_row_offsets[0],
                compact_result_ptr,
                &journal_compact_trace_offsets[0],
                compact_trace_ptr,
                compact_null2_ptr,
                journal_row_base,
                generation_tail_fingerprint,
                compact_tail_fingerprint,
                compact_domains_seam,
                compact_rebased_offsets,
                compact_statistics,
            )
    finally:
        free(compact_rebased_offsets)
        compact_rebased_offsets = NULL
    with nogil:
        hits._sort_by_key()
        hits._threshold(pipeline)

    hits._query = query
    hits._empty = False
    return hits


cdef void _validate_candidate_indexes(
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
) except *:
    cdef size_t cursor
    cdef uint32_t index
    cdef uint32_t previous = 0

    for cursor in range(<size_t> candidate_indexes.shape[0]):
        index = candidate_indexes[cursor]
        if index >= sequences._length:
            raise IndexError(f"candidate index out of range: {index}")
        if cursor != 0 and index <= previous:
            raise ValueError(
                "candidate indexes must be strictly increasing and unique"
            )
        previous = index


cdef bint _validate_bias_records(
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
) except -1:
    cdef _bias_result record
    cdef bint has_direct = False
    cdef size_t cursor
    cdef size_t record_count
    cdef uint32_t previous = 0

    if sizeof(_bias_result) != 12:
        raise RuntimeError("native bias result ABI is not 12 bytes")
    if <size_t> bias_records.shape[0] % sizeof(_bias_result) != 0:
        raise ValueError("bias result row has trailing bytes")
    record_count = <size_t> bias_records.shape[0] // sizeof(_bias_result)

    for cursor in range(record_count):
        memcpy(
            &record,
            &bias_records[cursor * sizeof(_bias_result)],
            sizeof(_bias_result),
        )
        if record.sequence_index >= sequences._length:
            raise IndexError(
                f"bias result sequence index out of range: "
                f"{record.sequence_index}"
            )
        if cursor != 0 and record.sequence_index <= previous:
            raise ValueError(
                "bias result sequence indexes must be strictly increasing "
                "and unique"
            )
        if record.action == BIAS_CPU_REQUIRED:
            pass
        elif (
            record.action == BIAS_DEFINITE_REJECT
            or record.action == BIAS_DEFINITE_PASS
        ):
            if record.ssv_status != SSV_OK:
                raise ValueError("direct bias result requires eslOK SSV status")
            if not isfinite(record.filtersc):
                raise ValueError("direct bias result requires finite filter score")
            if sequences._refs[record.sequence_index].n == 0:
                raise ValueError("empty target requires CPU bias fallback")
            has_direct = True
        else:
            raise ValueError(f"unknown bias result action: {record.action}")
        previous = record.sequence_index

    if has_direct and (
        not isfinite(optimized_profile._om.scale_b)
        or optimized_profile._om.scale_b <= 0.0
    ):
        raise ValueError("direct bias results require a positive finite SSV scale")
    return has_direct


cdef void _validate_bias_residue_offsets(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
    const uint64_t[::1] residue_offsets,
) except *:
    cdef _bias_result record
    cdef size_t cursor
    cdef size_t record_count = (
        <size_t> bias_records.shape[0] // sizeof(_bias_result)
    )
    cdef size_t target
    cdef size_t target_end
    cdef size_t previous_end = 0

    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")

    for cursor in range(record_count):
        memcpy(
            &record,
            &bias_records[cursor * sizeof(_bias_result)],
            sizeof(_bias_result),
        )
        target = record.sequence_index
        target_end = target + 1
        if residue_offsets[target_end] < residue_offsets[previous_end]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target_end] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")
        previous_end = target_end
    if residue_offsets[sequences._length] < residue_offsets[previous_end]:
        raise ValueError("target residue prefix is not monotone")
    if sequences._length != 0 and (
        residue_offsets[sequences._length]
        - residue_offsets[sequences._length - 1]
        != <uint64_t> sequences._refs[sequences._length - 1].n
    ):
        raise ValueError("target residue prefix differs from final target length")


cdef bint _validate_postfilter_records(
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
) except -1:
    cdef _postfilter_result record
    cdef _float_bits vfsc_bits
    cdef bint has_direct = False
    cdef size_t cursor
    cdef size_t record_count
    cdef uint32_t previous = 0

    if sizeof(_postfilter_result) != 16:
        raise RuntimeError("native post-filter result ABI is not 16 bytes")
    if <size_t> postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result row has trailing bytes")
    record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )

    for cursor in range(record_count):
        memcpy(
            &record,
            &postfilter_records[cursor * sizeof(_postfilter_result)],
            sizeof(_postfilter_result),
        )
        if record.sequence_index >= sequences._length:
            raise IndexError(
                f"post-filter result sequence index out of range: "
                f"{record.sequence_index}"
            )
        if cursor != 0 and record.sequence_index <= previous:
            raise ValueError(
                "post-filter result sequence indexes must be strictly "
                "increasing and unique"
            )
        if record.action == BIAS_CPU_REQUIRED:
            pass
        elif (
            record.action == BIAS_DEFINITE_REJECT
            and record.msv_status == SSV_OK
            and isnan(record.filtersc)
            and isnan(record.vfsc)
        ):
            pass
        elif (
            record.action == BIAS_DEFINITE_REJECT
            or record.action == BIAS_DEFINITE_PASS
        ):
            if record.msv_status != SSV_OK:
                raise ValueError(
                    "direct post-filter result requires eslOK MSV status"
                )
            if not isfinite(record.filtersc):
                raise ValueError(
                    "direct post-filter result requires finite filter score"
                )
            vfsc_bits.value = record.vfsc
            if not isfinite(record.vfsc) and vfsc_bits.bits != 0x7f800000:
                raise ValueError(
                    "direct post-filter result requires finite or +infinity "
                    "Viterbi score"
                )
            if sequences._refs[record.sequence_index].n == 0:
                raise ValueError("empty target requires CPU post-filter fallback")
            has_direct = True
        else:
            raise ValueError(f"unknown post-filter result action: {record.action}")
        previous = record.sequence_index

    if has_direct and (
        not isfinite(optimized_profile._om.scale_b)
        or optimized_profile._om.scale_b <= 0.0
    ):
        raise ValueError(
            "direct post-filter results require a positive finite SSV scale"
        )
    return has_direct


cdef void _validate_postfilter_residue_offsets(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] residue_offsets,
) except *:
    cdef _postfilter_result record
    cdef size_t cursor
    cdef size_t record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef size_t target
    cdef size_t target_end
    cdef size_t previous_end = 0

    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")

    for cursor in range(record_count):
        memcpy(
            &record,
            &postfilter_records[cursor * sizeof(_postfilter_result)],
            sizeof(_postfilter_result),
        )
        target = record.sequence_index
        target_end = target + 1
        if residue_offsets[target_end] < residue_offsets[previous_end]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target_end] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")
        previous_end = target_end
    if residue_offsets[sequences._length] < residue_offsets[previous_end]:
        raise ValueError("target residue prefix is not monotone")
    if sequences._length != 0 and (
        residue_offsets[sequences._length]
        - residue_offsets[sequences._length - 1]
        != <uint64_t> sequences._refs[sequences._length - 1].n
    ):
        raise ValueError("target residue prefix differs from final target length")


cdef bint _validate_forward_augmentation(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
) except -1:
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t postfilter_cursor = 0
    cdef size_t forward_cursor
    cdef size_t special_cursor
    cdef uint32_t previous = 0
    cdef uint64_t special_start
    cdef uint64_t special_stop
    cdef uint64_t expected_count
    cdef bint has_external = False

    if sizeof(_forward_result) != 12:
        raise RuntimeError("native Forward result ABI is not 12 bytes")
    if <size_t> forward_records.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("Forward result row has trailing bytes")
    forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    if special_offsets.shape[0] != forward_count + 1:
        raise ValueError("Forward special-offset count differs from result count")
    if special_offsets[0] > <uint64_t> specials.shape[0]:
        raise ValueError("Forward special offsets start outside storage")
    postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )

    for forward_cursor in range(forward_count):
        memcpy(
            &forward,
            &forward_records[forward_cursor * sizeof(_forward_result)],
            sizeof(_forward_result),
        )
        if forward.sequence_index >= sequences._length:
            raise IndexError(
                f"Forward result sequence index out of range: "
                f"{forward.sequence_index}"
            )
        if forward_cursor != 0 and forward.sequence_index <= previous:
            raise ValueError(
                "Forward result sequence indexes must be strictly increasing "
                "and unique"
            )
        previous = forward.sequence_index
        if forward.reserved != 0:
            raise ValueError("Forward result reserved field is nonzero")

        while postfilter_cursor < postfilter_count:
            memcpy(
                &postfilter,
                &postfilter_records[
                    postfilter_cursor * sizeof(_postfilter_result)
                ],
                sizeof(_postfilter_result),
            )
            if postfilter.sequence_index >= forward.sequence_index:
                break
            postfilter_cursor += 1
        if (
            postfilter_cursor == postfilter_count
            or postfilter.sequence_index != forward.sequence_index
            or postfilter.action != BIAS_DEFINITE_PASS
        ):
            raise ValueError(
                "Forward results must be a subset of direct post-filter passes"
            )

        special_start = special_offsets[forward_cursor]
        special_stop = special_offsets[forward_cursor + 1]
        if special_start > special_stop:
            raise ValueError("Forward special offsets are not monotone")
        if special_stop > <uint64_t> specials.shape[0]:
            raise ValueError("Forward special offsets exceed storage")

        if forward.action == FORWARD_CPU_REQUIRED:
            if forward.status not in (
                FORWARD_OK,
                FORWARD_ERANGE,
                FORWARD_ENORESULT,
                FORWARD_EMPTY,
            ):
                raise ValueError("unknown CPU-required Forward status")
            if forward.status == FORWARD_OK:
                if not isfinite(forward.fwdsc):
                    raise ValueError(
                        "eslOK CPU-required Forward result needs a finite score"
                    )
            elif not isnan(forward.fwdsc):
                raise ValueError(
                    "failed CPU-required Forward result must have NaN score"
                )
            if special_start != special_stop:
                raise ValueError(
                    "CPU-required Forward result must not have special states"
                )
        elif (
            forward.action == FORWARD_DEFINITE_REJECT
            or forward.action == FORWARD_DEFINITE_PASS
        ):
            if forward.status != FORWARD_OK:
                raise ValueError("direct Forward result requires eslOK status")
            if not isfinite(forward.fwdsc):
                raise ValueError("direct Forward result requires finite score")
            if forward.action == FORWARD_DEFINITE_REJECT:
                if special_start != special_stop:
                    raise ValueError(
                        "rejected Forward result must not have special states"
                    )
            else:
                expected_count = 6 * (
                    <uint64_t> sequences._refs[forward.sequence_index].n + 1
                )
                if special_stop - special_start != expected_count:
                    raise ValueError(
                        "passing Forward result has wrong special-state span"
                    )
                for special_cursor in range(
                    <size_t> special_start, <size_t> special_stop
                ):
                    if (
                        not isfinite(specials[special_cursor])
                        or specials[special_cursor] < 0.0
                    ):
                        raise ValueError(
                            "Forward special states must be finite and nonnegative"
                        )
            has_external = True
        else:
            raise ValueError(f"unknown Forward result action: {forward.action}")

    return has_external


def _validate_forward_batch_bound(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] postfilter_offsets,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] forward_offsets,
    const uint32_t[::1] expected_indices,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
):
    """Validate one complete native Forward result before it is retained."""
    cdef _forward_result forward
    cdef size_t profile_count
    cdef size_t profile_index
    cdef size_t cursor
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop

    if postfilter_offsets.shape[0] == 0:
        raise ValueError("post-filter row offsets need an initial zero")
    if postfilter_offsets.shape[0] != forward_offsets.shape[0]:
        raise ValueError("Forward row-offset count differs from post-filter rows")
    profile_count = <size_t> postfilter_offsets.shape[0] - 1
    if postfilter_offsets[0] != 0 or forward_offsets[0] != 0:
        raise ValueError("Forward batch row offsets must start at zero")
    if <size_t> postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result storage has trailing bytes")
    if <size_t> forward_records.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("Forward result storage has trailing bytes")
    postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    if postfilter_offsets[profile_count] != postfilter_count:
        raise ValueError("post-filter row offsets do not span result storage")
    if forward_offsets[profile_count] != forward_count:
        raise ValueError("Forward row offsets do not span result storage")
    if expected_indices.shape[0] != forward_count:
        raise ValueError("Forward result count differs from selected inputs")
    if special_offsets.shape[0] != forward_count + 1:
        raise ValueError("Forward special-offset count differs from result count")
    if special_offsets[0] != 0:
        raise ValueError("Forward global special offsets must start at zero")
    if special_offsets[forward_count] != <uint64_t> specials.shape[0]:
        raise ValueError("Forward special offsets do not span matrix storage")

    for cursor in range(forward_count):
        memcpy(
            &forward,
            &forward_records[cursor * sizeof(_forward_result)],
            sizeof(_forward_result),
        )
        if forward.sequence_index != expected_indices[cursor]:
            raise ValueError("Forward result order differs from selected inputs")

    for profile_index in range(profile_count):
        postfilter_start = <size_t> postfilter_offsets[profile_index]
        postfilter_stop = <size_t> postfilter_offsets[profile_index + 1]
        forward_start = <size_t> forward_offsets[profile_index]
        forward_stop = <size_t> forward_offsets[profile_index + 1]
        if postfilter_start > postfilter_stop or postfilter_stop > postfilter_count:
            raise ValueError("post-filter row offsets are not monotone")
        if forward_start > forward_stop or forward_stop > forward_count:
            raise ValueError("Forward row offsets are not monotone")
        _validate_forward_augmentation(
            sequences,
            postfilter_records[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            forward_records[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            special_offsets[forward_start:forward_stop + 1],
            specials,
        )


cdef void _validate_sealed_pair(
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
) except *:
    if not query.alphabet._eq(optimized_profile.alphabet):
        raise AlphabetMismatch(query.alphabet, optimized_profile.alphabet)
    if not query.alphabet._eq(sequences.alphabet):
        raise AlphabetMismatch(query.alphabet, sequences.alphabet)
    if query._hmm.M != optimized_profile._om.M:
        raise ValueError(
            "HMM and optimized profile model lengths differ: "
            f"{query._hmm.M} != {optimized_profile._om.M}"
        )
    if (
        query.name != optimized_profile.name
        or query.accession != optimized_profile.accession
    ):
        raise ValueError("HMM and optimized profile metadata differ")


cdef void _validate_sealed_residue_offsets(
    DigitalSequenceBlock sequences,
    const uint64_t[::1] residue_offsets,
) except *:
    cdef size_t target

    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")
    for target in range(sequences._length):
        if sequences._refs[target].n > HMMER_TARGET_LIMIT:
            raise ValueError("target exceeds HMMER's protein limit")
        if residue_offsets[target + 1] < residue_offsets[target]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target + 1] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")


cdef tuple _immutable_owned_view_with_copy(object value, str expected_format):
    """Return a full bytes-backed native view and whether it was copied."""
    cdef object view = memoryview(value)
    cdef object owner
    cdef object frozen

    if view.ndim != 1 or not view.c_contiguous:
        raise ValueError("sealed buffers must be one-dimensional and contiguous")
    if view.format != expected_format:
        raise TypeError(
            f"sealed buffer format {view.format!r} is not {expected_format!r}"
        )
    owner = view.obj
    while type(owner) is memoryview:
        owner = owner.obj
    # A readonly slice still pins the owner's entire allocation. Copy slices
    # so the charged retained payload is exactly the backing allocation size.
    if type(owner) is bytes and view.readonly and view.nbytes == len(owner):
        return view, False
    frozen = memoryview(view.cast("B").tobytes()).cast(expected_format)
    return frozen, True


cdef object _immutable_owned_view(object value, str expected_format):
    """Return a one-dimensional full-owner immutable native view."""
    return _immutable_owned_view_with_copy(value, expected_format)[0]


cdef tuple _validate_native_timing_stage(
    object values,
    size_t expected_count,
    str stage,
):
    """Freeze one exact native timing tuple after a strict scalar check."""
    cdef tuple frozen
    cdef object value
    cdef size_t index

    if type(values) is not tuple:
        raise TypeError(f"{stage} native timings must be exactly tuple")
    frozen = <tuple> values
    if len(frozen) != expected_count:
        raise ValueError(
            f"{stage} native timing field count differs from schema"
        )
    for index in range(expected_count):
        value = frozen[index]
        if type(value) is not float:
            raise TypeError(
                f"{stage} native timing field {index} must be exactly float"
            )
        if not isfinite(<double> value) or <double> value < 0.0:
            raise ValueError(
                f"{stage} native timing field {index} is not finite nonnegative"
            )
    return frozen


cdef tuple _validate_native_stage_timings(
    object values,
    bint rescore_expected,
):
    """Validate the sidecar paired with one consumed continuation journal."""
    cdef tuple frozen
    cdef tuple forward
    cdef tuple backward_domain
    cdef object domain_rescore

    if type(values) is not tuple:
        raise TypeError("native stage timings must be exactly tuple")
    frozen = <tuple> values
    if len(frozen) != 4:
        raise ValueError("native stage timing tuple differs from schema")
    if (
        type(frozen[0]) is not int
        or frozen[0] != SEALED_STAGE_TIMING_SCHEMA_VERSION
    ):
        raise ValueError("native stage timing schema version differs")
    forward = _validate_native_timing_stage(frozen[1], 9, "Forward")
    backward_domain = _validate_native_timing_stage(
        frozen[2], 4, "Backward/domain"
    )
    domain_rescore = frozen[3]
    if rescore_expected:
        domain_rescore = _validate_native_timing_stage(
            domain_rescore, 4, "domain rescore"
        )
    elif domain_rescore is not None:
        raise ValueError("domain-rescore timings exist without compact output")
    return (
        SEALED_STAGE_TIMING_SCHEMA_VERSION,
        forward,
        backward_domain,
        domain_rescore,
    )


cdef dict _native_stage_timing_evidence(tuple values):
    """Copy a validated immutable timing tuple into JSON-safe evidence."""
    cdef tuple forward = <tuple> values[1]
    cdef tuple backward_domain = <tuple> values[2]
    cdef object domain_rescore = values[3]
    cdef object rescore_evidence = None

    if domain_rescore is not None:
        rescore_evidence = {
            "kernel_ms": domain_rescore[0],
            "upload_ms": domain_rescore[1],
            "download_ms": domain_rescore[2],
            "total_ms": domain_rescore[3],
            "upload_scope": (
                "device allocations plus host-to-device work, offset, result, "
                "and null2 buffers; queued trace and trace-count memset "
                "completion is not "
                "isolated by this host timer"
            ),
            "kernel_scope": "three isolated-domain CUDA kernels",
            "download_scope": (
                "four synchronous device-to-host result, null2, trace-count, "
                "and trace copies"
            ),
            "total_scope": (
                "successful domain_rescore_run_impl entry through host "
                "validation, trace compaction, and statistics; excludes the "
                "later provenance seal"
            ),
        }
    return {
        "schema_version": values[0],
        "units": "milliseconds",
        "source": "native CUDA/steady-clock statistics",
        "forward": {
            "profile_staging_scope": "per-selection database",
            "profile_pack_scope": (
                "host descriptor and identity snapshot allocation and "
                "validation; ProfileSelection emissions and transitions were "
                "already packed"
            ),
            "profile_upload_scope": (
                "database device allocations, descriptor/emission/transition "
                "host-to-device copies, and device synchronization"
            ),
            "run_upload_scope": (
                "synchronous initial candidate, length-transition, and offset "
                "host-to-device copies plus survivor index/offset copies"
            ),
            "kernel_scope": "main Forward CUDA kernels only",
            "classification_scope": "per-tile CPU result classification",
            "gather_scope": "surviving special-row gather CUDA kernels only",
            "download_scope": (
                "synchronous Forward result and gathered-special "
                "device-to-host copies"
            ),
            "timed_loop_scope": (
                "tile loop after host candidate preparation, workspace growth, "
                "initial upload, event setup, and host result allocation; "
                "includes main kernels, result download, CPU classification, "
                "survivor upload, gather kernels, and gathered download; ends "
                "before provenance sealing"
            ),
            "call_total_scope": (
                "successful plan7_forward_run_with_workspace entry through "
                "provenance sealing; excludes per-selection database staging"
            ),
            "profile_pack_ms": forward[0],
            "profile_upload_ms": forward[1],
            "run_upload_ms": forward[2],
            "kernel_ms": forward[3],
            "classification_ms": forward[4],
            "gather_ms": forward[5],
            "download_ms": forward[6],
            "timed_loop_ms": forward[7],
            "call_total_ms": forward[8],
        },
        "backward_domain": {
            "kernel_ms": backward_domain[0],
            "upload_ms": backward_domain[1],
            "post_primary_materialization_ms": backward_domain[2],
            "total_ms": backward_domain[3],
            "upload_scope": "device allocations plus initial host-to-device copies",
            "kernel_scope": "primary backward_domain_kernel only",
            "post_primary_materialization_scope": (
                "native download_milliseconds field: initial result "
                "device-to-host copy, CPU routing/output sizing and allocation, "
                "region-offset device allocation/upload, simple-region gather "
                "kernel, region/posterior downloads, and output CSR remap"
            ),
            "post_primary_materialization_native_field": (
                "plan7_backward_domain_statistics.download_milliseconds"
            ),
            "total_scope": (
                "successful backward_domain_run_impl entry through route "
                "statistics aggregation; excludes the later provenance seal"
            ),
        },
        "domain_rescore": rescore_evidence,
    }


def _validate_native_stage_timings_bound(values, bint rescore_expected):
    """Host-test boundary for the exact sealed timing sidecar schema."""
    return _native_stage_timing_evidence(
        _validate_native_stage_timings(values, rescore_expected)
    )


cdef bint _journal_expected_segment(
    size_t *cursor,
    uint64_t observed_offset,
    uint64_t count,
    size_t item_size,
) noexcept:
    cdef size_t aligned
    cdef size_t native_count
    cdef size_t byte_count
    if count > <uint64_t> (<size_t> -1):
        return False
    native_count = <size_t> count
    if cursor[0] > (<size_t> -1) - 7:
        return False
    aligned = (cursor[0] + 7) & ~<size_t> 7
    if observed_offset != <uint64_t> aligned:
        return False
    if native_count != 0 and item_size > (<size_t> -1) // native_count:
        return False
    byte_count = native_count * item_size
    if aligned > (<size_t> -1) - byte_count:
        return False
    cursor[0] = aligned + byte_count
    return True


cdef bint _v3_advance_segment(
    size_t *cursor,
    uint64_t count,
    size_t item_size,
    uint64_t *offset,
) noexcept:
    cdef size_t aligned
    cdef size_t native_count
    cdef size_t byte_count
    if count > <uint64_t> (<size_t> -1):
        return False
    native_count = <size_t> count
    if cursor[0] > (<size_t> -1) - 7:
        return False
    aligned = (cursor[0] + 7) & ~<size_t> 7
    if native_count != 0 and item_size > (<size_t> -1) // native_count:
        return False
    byte_count = native_count * item_size
    if aligned > (<size_t> -1) - byte_count:
        return False
    offset[0] = <uint64_t> aligned
    cursor[0] = aligned + byte_count
    return True


cdef bint _v3_checked_increment(uint64_t *value, uint64_t delta) noexcept:
    cdef uint64_t updated
    if not plan7_continuation_journal_v3_checked_add(value[0], delta, &updated):
        return False
    value[0] = updated
    return True


cdef bint _v3_domain_continuation_safe(
    const plan7_continuation_journal_row *row,
    uint64_t region_begin,
    uint64_t region_end,
) noexcept nogil:
    return (
        row != NULL
        and row.domain_status == DOMAIN_OK
        and row.domain_route in (DOMAIN_NO_REGIONS, DOMAIN_SIMPLE)
        and not row.has_own_scales
        and row.uncertain_count == 0
        and row.multidomain_count == 0
        and region_begin <= region_end
        and row.region_count == region_end - region_begin
    )


cdef bint _v3_no_region_certifiable(
    const plan7_continuation_journal_row *row,
    uint64_t region_begin,
    uint64_t region_end,
    uint64_t compact_begin,
    uint64_t compact_end,
) noexcept nogil:
    return (
        _v3_domain_continuation_safe(row, region_begin, region_end)
        and row.domain_route == DOMAIN_NO_REGIONS
        and row.region_count == 0
        and region_begin == region_end
        and row.compact_route == PLAN7_CONTINUATION_COMPACT_NONE
        and row.compact_result_count == 0
        and compact_begin == compact_end
    )


cdef void _v3_capsule_destroy(object capsule) noexcept:
    cdef plan7_continuation_journal_v3 *journal
    cdef plan7_continuation_journal_v3_owner *owner
    journal = <plan7_continuation_journal_v3 *> PyCapsule_GetPointer(
        capsule, PLAN7_CONTINUATION_JOURNAL_V3_CAPSULE_NAME
    )
    owner = <plan7_continuation_journal_v3_owner *> PyCapsule_GetContext(
        capsule
    )
    if journal != NULL:
        free(journal)
    if owner != NULL:
        free(owner)


cdef void _v2_test_fixture_capsule_destroy(object capsule) noexcept:
    """Own a synthetic v2 allocation until the production seal consumes it."""
    cdef plan7_continuation_journal *journal
    journal = <plan7_continuation_journal *> PyCapsule_GetPointer(
        capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    )
    if journal != NULL:
        free(journal)


cdef void _v3_fill_options(
    plan7_continuation_journal_v3_options *destination,
    _SealedPostfilterBatch sealed,
    bint complete,
) noexcept:
    cdef _double_bits f1
    f1.value = sealed._f1
    destination.f1_bits = f1.bits
    destination.f2_bits = sealed._generation_f2_bits
    destination.f3_bits = sealed._generation_f3_bits
    destination.do_biasfilter = sealed._generation_bias_filter
    destination.complete = <uint32_t> complete
    if not complete:
        return
    destination.E_bits = sealed._pipeline_options.E_bits
    destination.T_bits = sealed._pipeline_options.T_bits
    destination.domE_bits = sealed._pipeline_options.domE_bits
    destination.domT_bits = sealed._pipeline_options.domT_bits
    destination.incE_bits = sealed._pipeline_options.incE_bits
    destination.incT_bits = sealed._pipeline_options.incT_bits
    destination.incdomE_bits = sealed._pipeline_options.incdomE_bits
    destination.incdomT_bits = sealed._pipeline_options.incdomT_bits
    destination.Z_bits = sealed._pipeline_options.Z_bits
    destination.domZ_bits = sealed._pipeline_options.domZ_bits
    destination.rt1_bits = sealed._pipeline_options.rt1_bits
    destination.rt2_bits = sealed._pipeline_options.rt2_bits
    destination.rt3_bits = sealed._pipeline_options.rt3_bits
    destination.do_null2 = sealed._pipeline_options.do_null2
    destination.do_alignment_score_calc = (
        sealed._pipeline_options.do_alignment_score_calc
    )
    destination.by_E = sealed._pipeline_options.by_E
    destination.dom_by_E = sealed._pipeline_options.dom_by_E
    destination.inc_by_E = sealed._pipeline_options.inc_by_E
    destination.incdom_by_E = sealed._pipeline_options.incdom_by_E
    destination.use_bit_cutoffs = sealed._pipeline_options.use_bit_cutoffs
    destination.Z_setby = sealed._pipeline_options.Z_setby
    destination.domZ_setby = sealed._pipeline_options.domZ_setby
    destination.mode = sealed._pipeline_options.mode
    destination.long_targets = sealed._pipeline_options.long_targets


cdef bint _v3_fill_certificate(
    plan7_continuation_journal_v3_certificate *certificate,
    _SealedPostfilterBatch sealed,
    uint32_t profile_index,
    uint32_t segment_index,
    uint64_t target_begin,
    uint64_t target_end,
    uint64_t raw_f1,
    uint64_t bias_reject,
    uint64_t f2_reject,
    uint64_t f3_reject,
    uint64_t no_region,
) noexcept:
    cdef uint64_t terminal_count = 0
    cdef uint64_t promotion = 0
    if target_begin > target_end or target_end > sealed._sequences._length:
        return False
    certificate.target_begin = target_begin
    certificate.target_end = target_end
    certificate.residue_prefix_begin = sealed._residue_offsets[target_begin]
    certificate.residue_prefix_end = sealed._residue_offsets[target_end]
    certificate.target_delta = target_end - target_begin
    if certificate.residue_prefix_end < certificate.residue_prefix_begin:
        return False
    certificate.residue_delta = (
        certificate.residue_prefix_end - certificate.residue_prefix_begin
    )
    if (
        not plan7_continuation_journal_v3_checked_add(
            raw_f1, bias_reject, &terminal_count
        )
        or not _v3_checked_increment(&terminal_count, f2_reject)
        or not _v3_checked_increment(&terminal_count, f3_reject)
        or not _v3_checked_increment(&terminal_count, no_region)
        or terminal_count > certificate.target_delta
    ):
        return False
    certificate.before_f1_count = certificate.target_delta - terminal_count
    certificate.raw_f1_reject_count = raw_f1
    certificate.bias_reject_count = bias_reject
    certificate.f2_reject_count = f2_reject
    certificate.f3_reject_count = f3_reject
    certificate.no_region_count = no_region
    if (
        not plan7_continuation_journal_v3_checked_add(
            bias_reject, f2_reject, &promotion
        )
        or not _v3_checked_increment(&promotion, f3_reject)
        or not _v3_checked_increment(&promotion, no_region)
    ):
        return False
    certificate.n_past_msv_delta = promotion
    if (
        not plan7_continuation_journal_v3_checked_add(
            f2_reject, f3_reject, &promotion
        )
        or not _v3_checked_increment(&promotion, no_region)
    ):
        return False
    certificate.n_past_bias_delta = promotion
    if not plan7_continuation_journal_v3_checked_add(
        f3_reject, no_region, &promotion
    ):
        return False
    certificate.n_past_vit_delta = promotion
    certificate.n_past_fwd_delta = no_region
    certificate.profile_index = profile_index
    certificate.segment_index = segment_index
    certificate.segment_tag = plan7_continuation_journal_v3_certificate_tag(
        certificate
    )
    return certificate.segment_tag != 0


cdef uint8_t _v3_decide_row(
    const P7_OPROFILE *profile,
    const _postfilter_result *postfilter,
    const _forward_result *forward,
    bint has_forward,
    const plan7_continuation_journal_row *domain,
    bint has_domain,
    uint64_t region_begin,
    uint64_t region_end,
    uint64_t compact_begin,
    uint64_t compact_end,
    bint compact_available,
    double f2,
    double f3,
) noexcept nogil:
    cdef int f2_decision
    cdef int f3_decision
    cdef uint8_t stage
    cdef uint8_t route

    if postfilter.action == BIAS_CPU_REQUIRED:
        stage = PLAN7_CONTINUATION_V3_CPU_REQUIRED
        route = PLAN7_CONTINUATION_V3_FULL_PIPELINE
        return <uint8_t> stage | (<uint8_t> route << 4)
    if isnan(postfilter.filtersc):
        return <uint8_t> PLAN7_CONTINUATION_V3_RAW_F1_REJECT
    if postfilter.action == BIAS_DEFINITE_REJECT:
        return <uint8_t> PLAN7_CONTINUATION_V3_BIAS_REJECT

    f2_decision = _hmmer_f2_decision(profile, postfilter, f2)
    if has_forward and f2_decision != 1:
        # A Forward row is authenticated only for an exact F2 survivor.  A
        # contradictory host fixture must fail closed; sending it through the
        # Forward-score seam would violate that seam's counter precondition.
        return 0xff
    if not has_forward:
        if f2_decision == 0:
            return <uint8_t> PLAN7_CONTINUATION_V3_F2_REJECT
        stage = PLAN7_CONTINUATION_V3_F2_SURVIVOR
        route = PLAN7_CONTINUATION_V3_FILTER_SCORES
        return <uint8_t> stage | (<uint8_t> route << 4)

    if forward.action == FORWARD_CPU_REQUIRED:
        stage = PLAN7_CONTINUATION_V3_F2_SURVIVOR
        route = PLAN7_CONTINUATION_V3_FILTER_SCORES
        return <uint8_t> stage | (<uint8_t> route << 4)

    # Native generation admits Forward rows only for exact F2 survivors.  A
    # contradictory private fixture stays an exception so the later consumer
    # can preserve the dense seam's validation/failure behavior.
    if f2_decision != 1:
        stage = PLAN7_CONTINUATION_V3_F3_SURVIVOR
        route = PLAN7_CONTINUATION_V3_FORWARD_SCORES
        return <uint8_t> stage | (<uint8_t> route << 4)

    f3_decision = _hmmer_f3_decision(
        profile, postfilter.filtersc, forward.fwdsc, f3
    )
    if forward.action == FORWARD_DEFINITE_REJECT and f3_decision == 0:
        return <uint8_t> PLAN7_CONTINUATION_V3_F3_REJECT
    if forward.action != FORWARD_DEFINITE_PASS or f3_decision != 1:
        stage = PLAN7_CONTINUATION_V3_F3_SURVIVOR
        route = PLAN7_CONTINUATION_V3_FORWARD_SCORES
        return <uint8_t> stage | (<uint8_t> route << 4)

    if has_domain and _v3_no_region_certifiable(
        domain, region_begin, region_end, compact_begin, compact_end
    ):
        return <uint8_t> PLAN7_CONTINUATION_V3_DOMAIN_NO_REGIONS

    if has_domain and _v3_domain_continuation_safe(
        domain, region_begin, region_end
    ):
        if (
            domain.domain_route == DOMAIN_SIMPLE
            and domain.compact_route == PLAN7_CONTINUATION_COMPACT_DEVICE
            and compact_begin < compact_end
            and compact_available
        ):
            stage = PLAN7_CONTINUATION_V3_DOMAIN_COMPACT
            route = PLAN7_CONTINUATION_V3_COMPACT_DOMAINS
        else:
            stage = PLAN7_CONTINUATION_V3_DOMAIN_SIMPLE
            route = PLAN7_CONTINUATION_V3_SIMPLE_REGIONS
        return <uint8_t> stage | (<uint8_t> route << 4)

    stage = (
        PLAN7_CONTINUATION_V3_DOMAIN_CPU_REQUIRED
        if has_domain
        else PLAN7_CONTINUATION_V3_F3_SURVIVOR
    )
    route = PLAN7_CONTINUATION_V3_FORWARD_SCORES
    return <uint8_t> stage | (<uint8_t> route << 4)


cdef plan7_continuation_journal_v3 *_v3_allocate_from_seal(
    _SealedPostfilterBatch sealed,
) except NULL:
    cdef plan7_continuation_journal_v3 *journal = NULL
    cdef plan7_continuation_journal *source_v2 = NULL
    cdef plan7_continuation_journal_v3_profile *profiles
    cdef plan7_continuation_journal_v3_certificate *certificates
    cdef plan7_continuation_journal_v3_exception *exceptions
    cdef plan7_continuation_journal_v3_profile *profile_record
    cdef plan7_continuation_journal_v3_certificate *certificate
    cdef plan7_continuation_journal_v3_exception *exception
    cdef plan7_continuation_journal_row domain
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef OptimizedProfile optimized_profile
    cdef uint8_t *decisions = NULL
    cdef uint8_t decision
    cdef uint8_t stage
    cdef uint8_t route
    cdef size_t decision_bytes
    cdef size_t cursor
    cdef size_t profile_count = len(sealed._optimized_profiles)
    cdef size_t planning_profile_count
    cdef size_t target_count = sealed._sequences._length
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t domain_count
    cdef size_t profile
    cdef size_t postfilter_cursor
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_cursor
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t domain_cursor
    cdef size_t domain_start
    cdef size_t domain_stop
    cdef size_t compact_index
    cdef size_t compact_source
    cdef uint64_t certificate_count = 0
    cdef uint64_t exception_count = 0
    cdef uint64_t special_count = 0
    cdef uint64_t region_count = 0
    cdef uint64_t compact_result_count = 0
    cdef uint64_t compact_trace_offset_count = 0
    cdef uint64_t compact_trace_count = 0
    cdef uint64_t compact_null2_count = 0
    cdef uint64_t profiles_offset = 0
    cdef uint64_t certificates_offset = 0
    cdef uint64_t exceptions_offset = 0
    cdef uint64_t specials_offset = 0
    cdef uint64_t regions_offset = 0
    cdef uint64_t compact_results_offset = 0
    cdef uint64_t compact_trace_offsets_offset = 0
    cdef uint64_t compact_traces_offset = 0
    cdef uint64_t compact_null2_offset = 0
    cdef uint64_t background_fingerprint_offset = 0
    cdef uint64_t special_begin
    cdef uint64_t special_end
    cdef uint64_t region_begin
    cdef uint64_t region_end
    cdef uint64_t compact_begin
    cdef uint64_t compact_end
    cdef uint64_t trace_begin
    cdef uint64_t trace_end
    cdef uint64_t payload_count
    cdef uint64_t certificate_cursor = 0
    cdef uint64_t exception_cursor = 0
    cdef uint64_t special_cursor = 0
    cdef uint64_t region_cursor = 0
    cdef uint64_t compact_cursor = 0
    cdef uint64_t trace_cursor = 0
    cdef uint64_t null2_cursor = 0
    cdef uint64_t segment_begin
    cdef uint64_t raw_f1
    cdef uint64_t bias_reject
    cdef uint64_t f2_reject
    cdef uint64_t f3_reject
    cdef uint64_t no_region
    cdef uint64_t profile_certificate_begin
    cdef uint64_t profile_exception_begin
    cdef uint64_t source_trace_begin
    cdef uint64_t source_trace_end
    cdef uint32_t segment_index
    cdef int f2_decision
    cdef int f3_decision
    cdef bint has_forward
    cdef bint has_domain
    cdef bint has_source_v2
    cdef bint has_direct_source
    cdef bint has_authenticated_source
    cdef bint owns_decisions = False
    cdef bint domain_safe
    cdef bint compact_available
    cdef _double_bits f2_bits
    cdef _double_bits f3_bits
    cdef const uint64_t *source_identity_tokens = NULL
    cdef const uint8_t *source_profile_fingerprints = NULL
    cdef uint64_t *compact_trace_offsets
    cdef uint8_t *destination
    cdef uint64_t direct_decision_state = 0
    cdef uint64_t direct_decision_tag = 0
    cdef _double_bits f1_bits

    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if profile_count > <size_t> 0xffffffff or target_count > <size_t> 0xffffffff:
        raise OverflowError("journal v3 profile or target index exceeds uint32")
    postfilter_count = (
        <size_t> sealed._postfilter_records.shape[0]
        // sizeof(_postfilter_result)
    )
    forward_count = (
        <size_t> sealed._forward_records.shape[0] // sizeof(_forward_result)
    )
    domain_count = (
        <size_t> sealed._journal_rows.shape[0]
        // sizeof(plan7_continuation_journal_row)
    )
    has_source_v2 = sealed._journal_storage.shape[0] != 0
    has_direct_source = sealed._direct_v3_source
    has_authenticated_source = has_source_v2 or has_direct_source
    if not sealed._generation_bias_filter:
        raise ValueError(
            "journal v3 semantic planning requires bias-enabled generation"
        )
    decision_bytes = postfilter_count
    planning_profile_count = profile_count
    if has_direct_source:
        planning_profile_count = 0
        sealed._journal_v3_source_scan_count = 1
        sealed._journal_v3_decision_scan_count = 0
        if sealed._direct_v3_decisions.shape[0] != postfilter_count:
            raise ValueError("direct v3 decision plan size differs from source")
        if decision_bytes:
            decisions = <uint8_t *> &sealed._direct_v3_decisions[0]
    elif decision_bytes:
        decisions = <uint8_t *> malloc(decision_bytes)
        if decisions == NULL:
            raise MemoryError("journal v3 decision workspace allocation failed")
        owns_decisions = True
    if not has_direct_source:
        sealed._journal_v3_source_scan_count = 2
        sealed._journal_v3_decision_scan_count = 1

    f2_bits.bits = sealed._generation_f2_bits
    f3_bits.bits = sealed._generation_f3_bits
    compact_available = (
        sealed._compact_domains_seam != NULL
        and sealed._generation_tail_fingerprint != 0
    )
    if has_source_v2:
        if sealed._journal_storage.shape[0] < sizeof(plan7_continuation_journal):
            if owns_decisions:
                free(decisions)
            raise ValueError("sealed v2 journal storage is truncated")
        source_v2 = <plan7_continuation_journal *> &sealed._journal_storage[0]
        source_identity_tokens = <const uint64_t *> (
            <const uint8_t *> source_v2 + source_v2.identity_tokens_offset
        )
        source_profile_fingerprints = (
            <const uint8_t *> source_v2
            + source_v2.profile_fingerprints_offset
        )

    try:
        # First pass derives one semantic decision per dense post-filter row and
        # exact sparse payload sizes.  No Python object is created per row.
        for profile in range(planning_profile_count):
            optimized_profile = <OptimizedProfile> sealed._optimized_profiles[profile]
            postfilter_start = <size_t> sealed._postfilter_offsets[profile]
            postfilter_stop = <size_t> sealed._postfilter_offsets[profile + 1]
            forward_start = <size_t> sealed._forward_offsets[profile]
            forward_stop = <size_t> sealed._forward_offsets[profile + 1]
            forward_cursor = forward_start
            if has_authenticated_source:
                domain_start = <size_t> sealed._journal_profile_offsets[profile]
                domain_stop = <size_t> sealed._journal_profile_offsets[profile + 1]
            else:
                domain_start = 0
                domain_stop = 0
            domain_cursor = domain_start

            for postfilter_cursor in range(postfilter_start, postfilter_stop):
                memcpy(
                    &postfilter,
                    &sealed._postfilter_records[
                        postfilter_cursor * sizeof(_postfilter_result)
                    ],
                    sizeof(_postfilter_result),
                )
                has_forward = False
                if forward_cursor < forward_stop:
                    memcpy(
                        &forward,
                        &sealed._forward_records[
                            forward_cursor * sizeof(_forward_result)
                        ],
                        sizeof(_forward_result),
                    )
                    has_forward = forward.sequence_index == postfilter.sequence_index
                has_domain = False
                region_begin = 0
                region_end = 0
                compact_begin = 0
                compact_end = 0
                if domain_cursor < domain_stop:
                    memcpy(
                        &domain,
                        &sealed._journal_rows[
                            domain_cursor * sizeof(plan7_continuation_journal_row)
                        ],
                        sizeof(plan7_continuation_journal_row),
                    )
                    has_domain = domain.sequence_index == postfilter.sequence_index
                    if has_domain:
                        region_begin = sealed._journal_region_offsets[domain_cursor]
                        region_end = sealed._journal_region_offsets[domain_cursor + 1]
                        compact_begin = (
                            sealed._journal_compact_row_offsets[domain_cursor]
                        )
                        compact_end = (
                            sealed._journal_compact_row_offsets[domain_cursor + 1]
                        )
                decision = _v3_decide_row(
                    optimized_profile._om,
                    &postfilter,
                    &forward,
                    has_forward,
                    &domain,
                    has_domain,
                    region_begin,
                    region_end,
                    compact_begin,
                    compact_end,
                    compact_available,
                    f2_bits.value,
                    f3_bits.value,
                )
                if decision == 0xff:
                    raise ValueError(
                        "journal v3 Forward source is not an exact F2 survivor"
                    )
                decisions[postfilter_cursor] = decision
                route = <uint8_t> (decision >> 4)
                if route != 0:
                    if not _v3_checked_increment(&exception_count, 1):
                        raise OverflowError("journal v3 exception count overflow")
                    if route in (
                        PLAN7_CONTINUATION_V3_FORWARD_SCORES,
                        PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                    ):
                        if not has_forward:
                            raise ValueError("journal v3 Forward route lacks source row")
                        special_begin = sealed._special_offsets[forward_cursor]
                        special_end = sealed._special_offsets[forward_cursor + 1]
                        if special_begin > special_end:
                            raise ValueError("journal v3 source specials are not monotone")
                        if not _v3_checked_increment(
                            &special_count, special_end - special_begin
                        ):
                            raise OverflowError("journal v3 special count overflow")
                    if route in (
                        PLAN7_CONTINUATION_V3_SIMPLE_REGIONS,
                        PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                    ):
                        if not has_domain or region_begin > region_end:
                            raise ValueError("journal v3 domain route lacks source row")
                        if not _v3_checked_increment(
                            &region_count, region_end - region_begin
                        ):
                            raise OverflowError("journal v3 region count overflow")
                    if route == PLAN7_CONTINUATION_V3_COMPACT_DOMAINS:
                        if compact_begin > compact_end:
                            raise ValueError("journal v3 compact rows are not monotone")
                        payload_count = compact_end - compact_begin
                        if (
                            not _v3_checked_increment(
                                &compact_result_count, payload_count
                            )
                            or not plan7_continuation_journal_v3_checked_multiply(
                                payload_count,
                                PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
                                &payload_count,
                            )
                            or not _v3_checked_increment(
                                &compact_null2_count, payload_count
                            )
                        ):
                            raise OverflowError("journal v3 compact count overflow")
                        source_trace_begin = (
                            sealed._journal_compact_trace_offsets[compact_begin]
                        )
                        source_trace_end = (
                            sealed._journal_compact_trace_offsets[compact_end]
                        )
                        if (
                            source_trace_begin > source_trace_end
                            or not _v3_checked_increment(
                                &compact_trace_count,
                                source_trace_end - source_trace_begin,
                            )
                        ):
                            raise OverflowError("journal v3 compact trace overflow")
                if has_forward:
                    forward_cursor += 1
                if has_domain:
                    domain_cursor += 1
            if forward_cursor != forward_stop or domain_cursor != domain_stop:
                raise ValueError("journal v3 source row mapping is incomplete")

        if has_direct_source:
            exception_count = sealed._direct_v3_expected_exception_count
            special_count = sealed._direct_v3_expected_special_count
            region_count = sealed._direct_v3_expected_region_count
            compact_result_count = (
                sealed._direct_v3_expected_compact_result_count
            )
            compact_trace_count = sealed._direct_v3_expected_compact_trace_count
            compact_null2_count = sealed._direct_v3_expected_compact_null2_count

        if not plan7_continuation_journal_v3_checked_add(
            exception_count, profile_count, &certificate_count
        ):
            raise OverflowError("journal v3 certificate count overflow")
        if not plan7_continuation_journal_v3_checked_add(
            compact_result_count, 1, &compact_trace_offset_count
        ):
            raise OverflowError("journal v3 compact offset count overflow")

        cursor = sizeof(plan7_continuation_journal_v3)
        if not (
            _v3_advance_segment(
                &cursor, profile_count,
                sizeof(plan7_continuation_journal_v3_profile),
                &profiles_offset,
            )
            and _v3_advance_segment(
                &cursor, certificate_count,
                sizeof(plan7_continuation_journal_v3_certificate),
                &certificates_offset,
            )
            and _v3_advance_segment(
                &cursor, exception_count,
                sizeof(plan7_continuation_journal_v3_exception),
                &exceptions_offset,
            )
            and _v3_advance_segment(
                &cursor, special_count, sizeof(float), &specials_offset
            )
            and _v3_advance_segment(
                &cursor, region_count, sizeof(plan7_simple_region),
                &regions_offset,
            )
            and _v3_advance_segment(
                &cursor, compact_result_count,
                sizeof(plan7_domain_rescore_result),
                &compact_results_offset,
            )
            and _v3_advance_segment(
                &cursor, compact_trace_offset_count, sizeof(uint64_t),
                &compact_trace_offsets_offset,
            )
            and _v3_advance_segment(
                &cursor, compact_trace_count,
                sizeof(plan7_domain_rescore_trace_step),
                &compact_traces_offset,
            )
            and _v3_advance_segment(
                &cursor, compact_null2_count, sizeof(float),
                &compact_null2_offset,
            )
            and _v3_advance_segment(
                &cursor, sealed._background_fingerprint.shape[0], 1,
                &background_fingerprint_offset,
            )
            and cursor <= <size_t> PY_SSIZE_T_MAX
        ):
            raise OverflowError("journal v3 storage layout overflow")

        journal = <plan7_continuation_journal_v3 *> calloc(1, cursor)
        if journal == NULL:
            raise MemoryError("journal v3 allocation failed")
        journal.magic = PLAN7_CONTINUATION_JOURNAL_V3_MAGIC
        journal.version = PLAN7_CONTINUATION_JOURNAL_V3_VERSION
        journal.header_size = sizeof(plan7_continuation_journal_v3)
        journal.profile_size = sizeof(plan7_continuation_journal_v3_profile)
        journal.certificate_size = sizeof(
            plan7_continuation_journal_v3_certificate
        )
        journal.exception_size = sizeof(plan7_continuation_journal_v3_exception)
        journal.region_size = sizeof(plan7_simple_region)
        journal.compact_result_size = sizeof(plan7_domain_rescore_result)
        journal.compact_trace_step_size = sizeof(
            plan7_domain_rescore_trace_step
        )
        journal.compact_null2_stride = PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        journal.source_kind = (
            PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT
            if has_direct_source
            else (
                PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL
                if has_source_v2
                else PLAN7_CONTINUATION_V3_SOURCE_HOST_SEAL
            )
        )
        journal.total_bytes = cursor
        journal.source_seal_token = <uint64_t> id(sealed)
        journal.profile_count = profile_count
        journal.target_count = target_count
        journal.total_residues = sealed._residue_offsets[target_count]
        journal.source_postfilter_count = postfilter_count
        journal.source_forward_count = forward_count
        journal.source_domain_count = domain_count
        journal.certificate_count = certificate_count
        journal.exception_count = exception_count
        journal.special_count = special_count
        journal.region_count = region_count
        journal.compact_result_count = compact_result_count
        journal.compact_trace_offset_count = compact_trace_offset_count
        journal.compact_trace_count = compact_trace_count
        journal.compact_null2_count = compact_null2_count
        journal.generation_tail_fingerprint = sealed._generation_tail_fingerprint
        journal.profiles_offset = profiles_offset
        journal.certificates_offset = certificates_offset
        journal.exceptions_offset = exceptions_offset
        journal.specials_offset = specials_offset
        journal.regions_offset = regions_offset
        journal.compact_results_offset = compact_results_offset
        journal.compact_trace_offsets_offset = compact_trace_offsets_offset
        journal.compact_traces_offset = compact_traces_offset
        journal.compact_null2_offset = compact_null2_offset
        journal.background_fingerprint_offset = background_fingerprint_offset
        journal.background_fingerprint_bytes = (
            sealed._background_fingerprint.shape[0]
        )
        _v3_fill_options(&journal.options, sealed, has_authenticated_source)
        if has_source_v2:
            journal.session_id = source_v2.session_id
            journal.selection_id = source_v2.selection_id
            journal.batch_generation = source_v2.forward.batch_generation
            journal.source_v2_total_bytes = source_v2.total_bytes
            journal.source_v2_integrity_tag = source_v2.integrity_tag
            memcpy(
                journal.sequence_content_fingerprint,
                source_v2.sequence_content_fingerprint,
                PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE,
            )
            memcpy(
                &journal.forward,
                &source_v2.forward,
                sizeof(plan7_forward_provenance),
            )
            memcpy(
                &journal.backward,
                &source_v2.backward,
                sizeof(plan7_backward_domain_provenance),
            )
            memcpy(
                &journal.rescore,
                &source_v2.rescore,
                sizeof(plan7_domain_rescore_provenance),
            )
        elif has_direct_source:
            journal.session_id = sealed._telemetry_session_id
            journal.selection_id = sealed._telemetry_selection_id
            journal.batch_generation = sealed._telemetry_batch_generation
            memcpy(
                journal.sequence_content_fingerprint,
                &sealed._source_sequence_fingerprint[0],
                PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE,
            )
            memcpy(
                &journal.forward,
                &sealed._source_forward_provenance,
                sizeof(plan7_forward_provenance),
            )
            memcpy(
                &journal.backward,
                &sealed._source_backward_provenance,
                sizeof(plan7_backward_domain_provenance),
            )
            memcpy(
                &journal.rescore,
                &sealed._source_rescore_provenance,
                sizeof(plan7_domain_rescore_provenance),
            )
        destination = <uint8_t *> journal
        if sealed._background_fingerprint.shape[0]:
            memcpy(
                destination + background_fingerprint_offset,
                &sealed._background_fingerprint[0],
                sealed._background_fingerprint.shape[0],
            )

        profiles = <plan7_continuation_journal_v3_profile *> (
            destination + profiles_offset
        )
        certificates = <plan7_continuation_journal_v3_certificate *> (
            destination + certificates_offset
        )
        exceptions = <plan7_continuation_journal_v3_exception *> (
            destination + exceptions_offset
        )
        compact_trace_offsets = <uint64_t *> (
            destination + compact_trace_offsets_offset
        )
        compact_trace_offsets[0] = 0
        if has_direct_source:
            f1_bits.value = sealed._f1
            direct_decision_state = (
                plan7_continuation_journal_v3_decision_seed(
                    postfilter_count,
                    profile_count,
                    journal.session_id,
                    journal.selection_id,
                    journal.batch_generation,
                    f1_bits.bits,
                    sealed._generation_f2_bits,
                    sealed._generation_f3_bits,
                    sealed._generation_tail_fingerprint,
                )
            )

        # Emit ordered certificates and copy only exception payloads.  This is
        # the second dense-source pass for v2/audit, but the only planner scan
        # for native-direct.  Certificate targets never include the following
        # exception.
        for profile in range(profile_count):
            optimized_profile = <OptimizedProfile> sealed._optimized_profiles[profile]
            postfilter_start = <size_t> sealed._postfilter_offsets[profile]
            postfilter_stop = <size_t> sealed._postfilter_offsets[profile + 1]
            forward_start = <size_t> sealed._forward_offsets[profile]
            forward_stop = <size_t> sealed._forward_offsets[profile + 1]
            forward_cursor = forward_start
            if has_authenticated_source:
                domain_start = <size_t> sealed._journal_profile_offsets[profile]
                domain_stop = <size_t> sealed._journal_profile_offsets[profile + 1]
            else:
                domain_start = 0
                domain_stop = 0
            domain_cursor = domain_start
            profile_certificate_begin = certificate_cursor
            profile_exception_begin = exception_cursor
            segment_begin = 0
            segment_index = 0
            raw_f1 = 0
            bias_reject = 0
            f2_reject = 0
            f3_reject = 0
            no_region = 0

            for postfilter_cursor in range(postfilter_start, postfilter_stop):
                memcpy(
                    &postfilter,
                    &sealed._postfilter_records[
                        postfilter_cursor * sizeof(_postfilter_result)
                    ],
                    sizeof(_postfilter_result),
                )
                has_forward = False
                if forward_cursor < forward_stop:
                    memcpy(
                        &forward,
                        &sealed._forward_records[
                            forward_cursor * sizeof(_forward_result)
                        ],
                        sizeof(_forward_result),
                    )
                    has_forward = forward.sequence_index == postfilter.sequence_index
                has_domain = False
                region_begin = 0
                region_end = 0
                compact_begin = 0
                compact_end = 0
                if domain_cursor < domain_stop:
                    memcpy(
                        &domain,
                        &sealed._journal_rows[
                            domain_cursor * sizeof(plan7_continuation_journal_row)
                        ],
                        sizeof(plan7_continuation_journal_row),
                    )
                    has_domain = domain.sequence_index == postfilter.sequence_index
                    if has_domain:
                        region_begin = sealed._journal_region_offsets[domain_cursor]
                        region_end = sealed._journal_region_offsets[domain_cursor + 1]
                        compact_begin = (
                            sealed._journal_compact_row_offsets[domain_cursor]
                        )
                        compact_end = (
                            sealed._journal_compact_row_offsets[domain_cursor + 1]
                        )
                decision = decisions[postfilter_cursor]
                if has_direct_source:
                    direct_decision_state ^= (
                        plan7_continuation_journal_v3_decision_term(
                            postfilter_cursor, decision
                        )
                    )
                stage = <uint8_t> (decision & 0x0f)
                route = <uint8_t> (decision >> 4)
                if route == 0:
                    if stage == PLAN7_CONTINUATION_V3_RAW_F1_REJECT:
                        if not _v3_checked_increment(&raw_f1, 1):
                            raise OverflowError("journal v3 raw-F1 count overflow")
                    elif stage == PLAN7_CONTINUATION_V3_BIAS_REJECT:
                        if not _v3_checked_increment(&bias_reject, 1):
                            raise OverflowError("journal v3 bias count overflow")
                    elif stage == PLAN7_CONTINUATION_V3_F2_REJECT:
                        if not _v3_checked_increment(&f2_reject, 1):
                            raise OverflowError("journal v3 F2 count overflow")
                    elif stage == PLAN7_CONTINUATION_V3_F3_REJECT:
                        if not _v3_checked_increment(&f3_reject, 1):
                            raise OverflowError("journal v3 F3 count overflow")
                    elif stage == PLAN7_CONTINUATION_V3_DOMAIN_NO_REGIONS:
                        if not _v3_checked_increment(&no_region, 1):
                            raise OverflowError("journal v3 no-region count overflow")
                    else:
                        raise ValueError("journal v3 terminal source stage is invalid")
                else:
                    if (
                        certificate_cursor >= certificate_count
                        or exception_cursor >= exception_count
                    ):
                        raise ValueError(
                            "journal v3 decision counts understate exceptions"
                        )
                    certificate = &certificates[certificate_cursor]
                    if not _v3_fill_certificate(
                        certificate,
                        sealed,
                        <uint32_t> profile,
                        segment_index,
                        segment_begin,
                        postfilter.sequence_index,
                        raw_f1,
                        bias_reject,
                        f2_reject,
                        f3_reject,
                        no_region,
                    ):
                        raise OverflowError("journal v3 certificate is inconsistent")
                    certificate_cursor += 1
                    segment_index += 1
                    segment_begin = <uint64_t> postfilter.sequence_index + 1
                    raw_f1 = 0
                    bias_reject = 0
                    f2_reject = 0
                    f3_reject = 0
                    no_region = 0

                    exception = &exceptions[exception_cursor]
                    exception.source_postfilter_index = postfilter_cursor
                    exception.source_forward_index = (
                        forward_cursor
                        if has_forward
                        else <uint64_t> -1
                    )
                    exception.source_domain_index = (
                        domain_cursor
                        if has_domain
                        else <uint64_t> -1
                    )
                    exception.residue_prefix_begin = (
                        sealed._residue_offsets[postfilter.sequence_index]
                    )
                    exception.residue_prefix_end = (
                        sealed._residue_offsets[postfilter.sequence_index + 1]
                    )
                    exception.residue_delta = (
                        exception.residue_prefix_end
                        - exception.residue_prefix_begin
                    )
                    exception.profile_index = <uint32_t> profile
                    exception.sequence_index = postfilter.sequence_index
                    exception.exception_index = <uint32_t> (
                        exception_cursor - profile_exception_begin
                    )
                    exception.source_stage = stage
                    exception.route = route
                    exception.payload_flags = PLAN7_CONTINUATION_V3_HAS_POSTFILTER
                    memcpy(
                        exception.postfilter_record,
                        &postfilter,
                        sizeof(_postfilter_result),
                    )
                    if has_forward:
                        exception.payload_flags |= PLAN7_CONTINUATION_V3_HAS_FORWARD
                        memcpy(
                            exception.forward_record,
                            &forward,
                            sizeof(_forward_result),
                        )
                    if has_domain:
                        exception.payload_flags |= PLAN7_CONTINUATION_V3_HAS_DOMAIN
                        memcpy(
                            exception.domain_record,
                            &domain,
                            sizeof(plan7_continuation_journal_row),
                        )
                    f2_decision = _hmmer_f2_decision(
                        optimized_profile._om, &postfilter, f2_bits.value
                    )
                    if f2_decision == 1:
                        exception.preconditions |= (
                            PLAN7_CONTINUATION_V3_PRE_F2_SURVIVOR
                        )
                    if has_forward and forward.action != FORWARD_CPU_REQUIRED:
                        exception.preconditions |= (
                            PLAN7_CONTINUATION_V3_PRE_DIRECT_FORWARD
                        )
                        f3_decision = _hmmer_f3_decision(
                            optimized_profile._om,
                            postfilter.filtersc,
                            forward.fwdsc,
                            f3_bits.value,
                        )
                        if f3_decision == 1:
                            exception.preconditions |= (
                                PLAN7_CONTINUATION_V3_PRE_F3_SURVIVOR
                            )
                    domain_safe = has_domain and _v3_domain_continuation_safe(
                        &domain, region_begin, region_end
                    )
                    if domain_safe:
                        exception.preconditions |= (
                            PLAN7_CONTINUATION_V3_PRE_DOMAIN_SAFE
                        )
                    if (
                        has_domain
                        and domain.compact_route
                        == PLAN7_CONTINUATION_COMPACT_DEVICE
                    ):
                        exception.preconditions |= (
                            PLAN7_CONTINUATION_V3_PRE_COMPACT_DEVICE
                        )

                    if route in (
                        PLAN7_CONTINUATION_V3_FORWARD_SCORES,
                        PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                    ):
                        special_begin = sealed._special_offsets[forward_cursor]
                        special_end = sealed._special_offsets[forward_cursor + 1]
                        exception.special_begin = special_cursor
                        exception.special_count = special_end - special_begin
                        if exception.special_count > special_count - special_cursor:
                            raise ValueError(
                                "journal v3 decision counts understate specials"
                            )
                        if exception.special_count:
                            exception.payload_flags |= (
                                PLAN7_CONTINUATION_V3_HAS_SPECIALS
                            )
                            memcpy(
                                destination + specials_offset
                                + special_cursor * sizeof(float),
                                &sealed._specials[special_begin],
                                exception.special_count * sizeof(float),
                            )
                        special_cursor += exception.special_count
                    if route in (
                        PLAN7_CONTINUATION_V3_SIMPLE_REGIONS,
                        PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                    ):
                        exception.region_begin = region_cursor
                        exception.region_count = region_end - region_begin
                        if exception.region_count > region_count - region_cursor:
                            raise ValueError(
                                "journal v3 decision counts understate regions"
                            )
                        if exception.region_count:
                            exception.payload_flags |= (
                                PLAN7_CONTINUATION_V3_HAS_REGIONS
                            )
                            memcpy(
                                destination + regions_offset
                                + region_cursor * sizeof(plan7_simple_region),
                                &sealed._journal_regions[
                                    region_begin * sizeof(plan7_simple_region)
                                ],
                                exception.region_count
                                * sizeof(plan7_simple_region),
                            )
                        region_cursor += exception.region_count
                    if route == PLAN7_CONTINUATION_V3_COMPACT_DOMAINS:
                        exception.payload_flags |= PLAN7_CONTINUATION_V3_HAS_COMPACT
                        exception.compact_result_begin = compact_cursor
                        exception.compact_result_count = compact_end - compact_begin
                        if exception.compact_result_count > (
                            compact_result_count - compact_cursor
                        ):
                            raise ValueError(
                                "journal v3 decision counts understate compact rows"
                            )
                        exception.compact_trace_begin = trace_cursor
                        exception.compact_null2_begin = null2_cursor
                        if exception.compact_result_count:
                            memcpy(
                                destination + compact_results_offset
                                + compact_cursor
                                * sizeof(plan7_domain_rescore_result),
                                &sealed._journal_compact_results[
                                    compact_begin
                                    * sizeof(plan7_domain_rescore_result)
                                ],
                                exception.compact_result_count
                                * sizeof(plan7_domain_rescore_result),
                            )
                        for compact_index in range(
                            <size_t> exception.compact_result_count
                        ):
                            compact_source = <size_t> compact_begin + compact_index
                            source_trace_begin = (
                                sealed._journal_compact_trace_offsets[
                                    compact_source
                                ]
                            )
                            source_trace_end = (
                                sealed._journal_compact_trace_offsets[
                                    compact_source + 1
                                ]
                            )
                            if (
                                source_trace_begin > source_trace_end
                                or source_trace_end - source_trace_begin
                                > compact_trace_count - trace_cursor
                            ):
                                raise ValueError(
                                    "journal v3 decision counts understate traces"
                                )
                            compact_trace_offsets[
                                compact_cursor + compact_index
                            ] = trace_cursor
                            if source_trace_end > source_trace_begin:
                                memcpy(
                                    destination + compact_traces_offset
                                    + trace_cursor
                                    * sizeof(plan7_domain_rescore_trace_step),
                                    &sealed._journal_compact_traces[
                                        source_trace_begin
                                        * sizeof(
                                            plan7_domain_rescore_trace_step
                                        )
                                    ],
                                    (source_trace_end - source_trace_begin)
                                    * sizeof(plan7_domain_rescore_trace_step),
                                )
                            trace_cursor += source_trace_end - source_trace_begin
                        compact_trace_offsets[
                            compact_cursor + exception.compact_result_count
                        ] = trace_cursor
                        exception.compact_trace_count = (
                            trace_cursor - exception.compact_trace_begin
                        )
                        exception.compact_null2_count = (
                            exception.compact_result_count
                            * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                        )
                        if exception.compact_null2_count > (
                            compact_null2_count - null2_cursor
                        ):
                            raise ValueError(
                                "journal v3 decision counts understate compact null2"
                            )
                        if exception.compact_null2_count:
                            memcpy(
                                destination + compact_null2_offset
                                + null2_cursor * sizeof(float),
                                &sealed._journal_compact_null2[
                                    compact_begin
                                    * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                                ],
                                exception.compact_null2_count * sizeof(float),
                            )
                        null2_cursor += exception.compact_null2_count
                        compact_cursor += exception.compact_result_count
                    exception.exception_tag = (
                        plan7_continuation_journal_v3_exception_tag(exception)
                    )
                    if exception.exception_tag == 0:
                        raise ValueError("journal v3 exception tag is zero")
                    exception_cursor += 1

                if has_forward:
                    forward_cursor += 1
                if has_domain:
                    domain_cursor += 1

            if certificate_cursor >= certificate_count:
                raise ValueError(
                    "journal v3 decision counts understate certificates"
                )
            certificate = &certificates[certificate_cursor]
            if not _v3_fill_certificate(
                certificate,
                sealed,
                <uint32_t> profile,
                segment_index,
                segment_begin,
                target_count,
                raw_f1,
                bias_reject,
                f2_reject,
                f3_reject,
                no_region,
            ):
                raise OverflowError("journal v3 tail certificate is inconsistent")
            certificate_cursor += 1

            profile_record = &profiles[profile]
            profile_record.certificate_begin = profile_certificate_begin
            profile_record.certificate_count = (
                certificate_cursor - profile_certificate_begin
            )
            profile_record.exception_begin = profile_exception_begin
            profile_record.exception_count = (
                exception_cursor - profile_exception_begin
            )
            profile_record.target_count = target_count
            profile_record.total_residues = sealed._residue_offsets[target_count]
            profile_record.source_postfilter_begin = postfilter_start
            profile_record.source_postfilter_count = postfilter_stop - postfilter_start
            profile_record.source_forward_begin = forward_start
            profile_record.source_forward_count = forward_stop - forward_start
            profile_record.source_domain_begin = domain_start
            profile_record.source_domain_count = domain_stop - domain_start
            profile_record.profile_index = <uint32_t> profile
            if has_authenticated_source:
                profile_record.identity_token = (
                    source_identity_tokens[profile]
                    if has_source_v2
                    else sealed._source_identity_tokens[profile]
                )
                profile_record.flags = (
                    PLAN7_CONTINUATION_V3_PROFILE_HAS_V2_IDENTITY
                    | PLAN7_CONTINUATION_V3_PROFILE_HAS_FINGERPRINT
                )
                if has_source_v2:
                    memcpy(
                        profile_record.profile_fingerprint,
                        source_profile_fingerprints
                        + profile
                        * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
                        PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
                    )
                else:
                    memcpy(
                        profile_record.profile_fingerprint,
                        &sealed._source_profile_fingerprints[
                            profile
                            * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
                        ],
                        PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
                    )
            else:
                profile_record.identity_token = (
                    <uint64_t> <size_t> optimized_profile._om
                )
            profile_record.profile_tag = (
                plan7_continuation_journal_v3_profile_tag(profile_record)
            )
            if profile_record.profile_tag == 0:
                raise ValueError("journal v3 profile tag is zero")

        if (
            certificate_cursor != certificate_count
            or exception_cursor != exception_count
            or special_cursor != special_count
            or region_cursor != region_count
            or compact_cursor != compact_result_count
            or trace_cursor != compact_trace_count
            or null2_cursor != compact_null2_count
            or compact_trace_offsets[compact_result_count]
            != compact_trace_count
        ):
            raise ValueError("journal v3 emitted payload counts differ")
        if has_direct_source:
            direct_decision_tag = (
                plan7_continuation_journal_v3_decision_finish(
                    direct_decision_state,
                    exception_count,
                    special_count,
                    region_count,
                    compact_result_count,
                    compact_trace_count,
                    compact_null2_count,
                )
            )
            if direct_decision_tag != sealed._direct_v3_expected_decision_tag:
                raise ValueError(
                    "direct v3 decision plan authentication differs"
                )
        journal.integrity_tag = plan7_continuation_journal_v3_integrity(journal)
        if journal.integrity_tag == 0:
            raise ValueError("journal v3 integrity tag is zero")
        return journal
    except:
        if journal != NULL:
            free(journal)
        raise
    finally:
        if owns_decisions and decisions != NULL:
            free(decisions)


cdef plan7_continuation_journal_v3 *_v3_validate_packet(
    plan7_continuation_journal_v3 *journal,
    uint64_t allocation_bytes,
    uint64_t source_seal_token,
    _SealedPostfilterBatch sealed,
    bint rebuild_source,
) except NULL:
    cdef plan7_continuation_journal_v3 *expected = NULL
    cdef plan7_continuation_journal *source_v2 = NULL
    cdef plan7_continuation_journal_v3_profile *profiles
    cdef plan7_continuation_journal_v3_certificate *certificates
    cdef plan7_continuation_journal_v3_exception *exceptions
    cdef plan7_continuation_journal_v3_profile *profile_record
    cdef plan7_continuation_journal_v3_certificate *certificate
    cdef plan7_continuation_journal_v3_exception *exception
    cdef uint8_t *base
    cdef size_t cursor
    cdef size_t profile
    cdef size_t local_index
    cdef uint64_t target_cursor
    cdef uint64_t stage_total
    cdef uint64_t expected_value
    cdef uint64_t special_cursor = 0
    cdef uint64_t region_cursor = 0
    cdef uint64_t compact_cursor = 0
    cdef uint64_t trace_cursor = 0
    cdef uint64_t null2_cursor = 0
    cdef uint64_t cert_begin
    cdef uint64_t cert_end
    cdef uint64_t exception_begin
    cdef uint64_t exception_end
    cdef uint64_t integrity
    cdef uint64_t compact_null2_expected
    cdef uint64_t trace_offset_index
    cdef uint64_t expected_source_domain_begin
    cdef uint64_t expected_source_domain_count
    cdef const uint64_t *source_identity_tokens = NULL
    cdef const uint8_t *source_profile_fingerprints = NULL
    cdef const uint64_t *compact_trace_offsets = NULL
    cdef plan7_continuation_journal_v3_options expected_options

    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if journal == NULL:
        raise TypeError("continuation journal v3 storage is unavailable")
    if (
        source_seal_token != <uint64_t> id(sealed)
        or journal.source_seal_token != source_seal_token
    ):
        raise ValueError("continuation journal v3 source seal differs")
    if (
        journal.magic != PLAN7_CONTINUATION_JOURNAL_V3_MAGIC
        or journal.version != PLAN7_CONTINUATION_JOURNAL_V3_VERSION
        or journal.header_size != sizeof(plan7_continuation_journal_v3)
        or journal.profile_size
        != sizeof(plan7_continuation_journal_v3_profile)
        or journal.certificate_size
        != sizeof(plan7_continuation_journal_v3_certificate)
        or journal.exception_size
        != sizeof(plan7_continuation_journal_v3_exception)
        or journal.region_size != sizeof(plan7_simple_region)
        or journal.compact_result_size
        != sizeof(plan7_domain_rescore_result)
        or journal.compact_trace_step_size
        != sizeof(plan7_domain_rescore_trace_step)
        or journal.compact_null2_stride
        != PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.reserved0 != 0
        or journal.total_bytes != allocation_bytes
        or journal.total_bytes < sizeof(plan7_continuation_journal_v3)
        or journal.total_bytes > <uint64_t> (<size_t> -1)
        or journal.total_bytes > <uint64_t> PY_SSIZE_T_MAX
        or journal.profile_count != len(sealed._optimized_profiles)
        or journal.target_count != sealed._sequences._length
        or journal.total_residues
        != sealed._residue_offsets[sealed._sequences._length]
        or journal.compact_result_count == <uint64_t> -1
        or journal.compact_trace_offset_count == 0
        or journal.compact_trace_offset_count
        != journal.compact_result_count + 1
        or journal.source_kind not in (
            PLAN7_CONTINUATION_V3_SOURCE_HOST_SEAL,
            PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL,
            PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT,
        )
    ):
        raise ValueError("continuation journal v3 ABI header is invalid")
    cursor = sizeof(plan7_continuation_journal_v3)
    if not (
        _journal_expected_segment(
            &cursor, journal.profiles_offset, journal.profile_count,
            sizeof(plan7_continuation_journal_v3_profile),
        )
        and _journal_expected_segment(
            &cursor, journal.certificates_offset, journal.certificate_count,
            sizeof(plan7_continuation_journal_v3_certificate),
        )
        and _journal_expected_segment(
            &cursor, journal.exceptions_offset, journal.exception_count,
            sizeof(plan7_continuation_journal_v3_exception),
        )
        and _journal_expected_segment(
            &cursor, journal.specials_offset, journal.special_count,
            sizeof(float),
        )
        and _journal_expected_segment(
            &cursor, journal.regions_offset, journal.region_count,
            sizeof(plan7_simple_region),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_results_offset,
            journal.compact_result_count,
            sizeof(plan7_domain_rescore_result),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_trace_offsets_offset,
            journal.compact_trace_offset_count, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_traces_offset,
            journal.compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_null2_offset,
            journal.compact_null2_count, sizeof(float),
        )
        and _journal_expected_segment(
            &cursor, journal.background_fingerprint_offset,
            journal.background_fingerprint_bytes, 1,
        )
        and cursor == <size_t> journal.total_bytes
    ):
        raise ValueError("continuation journal v3 storage layout is invalid")
    integrity = plan7_continuation_journal_v3_integrity(journal)
    if journal.integrity_tag == 0 or journal.integrity_tag != integrity:
        raise ValueError("continuation journal v3 integrity check failed")
    if (
        journal.background_fingerprint_bytes
        != sealed._background_fingerprint.shape[0]
        or (
            journal.background_fingerprint_bytes != 0
            and memcmp(
                <uint8_t *> journal + journal.background_fingerprint_offset,
                &sealed._background_fingerprint[0],
                journal.background_fingerprint_bytes,
            ) != 0
        )
    ):
        raise ValueError("continuation journal v3 background identity differs")
    if (
        bool(sealed._journal_storage.shape[0])
        != (journal.source_kind == PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL)
        or sealed._direct_v3_source
        != (journal.source_kind == PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT)
    ):
        raise ValueError("continuation journal v3 source provenance differs")
    memset(&expected_options, 0, sizeof(expected_options))
    _v3_fill_options(
        &expected_options,
        sealed,
        journal.source_kind != PLAN7_CONTINUATION_V3_SOURCE_HOST_SEAL,
    )
    if (
        journal.source_postfilter_count
        != sealed._postfilter_records.shape[0] // sizeof(_postfilter_result)
        or journal.source_forward_count
        != sealed._forward_records.shape[0] // sizeof(_forward_result)
        or journal.source_domain_count
        != sealed._journal_rows.shape[0]
        // sizeof(plan7_continuation_journal_row)
        or journal.generation_tail_fingerprint
        != sealed._generation_tail_fingerprint
        or memcmp(
            &journal.options,
            &expected_options,
            sizeof(plan7_continuation_journal_v3_options),
        ) != 0
    ):
        raise ValueError("continuation journal v3 source metadata differs")
    if journal.source_kind == PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL:
        if sealed._journal_storage.shape[0] < sizeof(plan7_continuation_journal):
            raise ValueError("continuation journal v3 dense source is truncated")
        source_v2 = <plan7_continuation_journal *> &sealed._journal_storage[0]
        if (
            journal.session_id != source_v2.session_id
            or journal.selection_id != source_v2.selection_id
            or journal.batch_generation != source_v2.forward.batch_generation
            or journal.source_v2_total_bytes != source_v2.total_bytes
            or journal.source_v2_total_bytes
            != <uint64_t> sealed._journal_storage.shape[0]
            or journal.source_v2_integrity_tag != source_v2.integrity_tag
            or memcmp(
                journal.sequence_content_fingerprint,
                source_v2.sequence_content_fingerprint,
                PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE,
            ) != 0
            or memcmp(
                &journal.forward,
                &source_v2.forward,
                sizeof(plan7_forward_provenance),
            ) != 0
            or memcmp(
                &journal.backward,
                &source_v2.backward,
                sizeof(plan7_backward_domain_provenance),
            ) != 0
            or memcmp(
                &journal.rescore,
                &source_v2.rescore,
                sizeof(plan7_domain_rescore_provenance),
            ) != 0
        ):
            raise ValueError("continuation journal v3 dense source identity differs")
        source_identity_tokens = <const uint64_t *> (
            <const uint8_t *> source_v2 + source_v2.identity_tokens_offset
        )
        source_profile_fingerprints = (
            <const uint8_t *> source_v2
            + source_v2.profile_fingerprints_offset
        )
    elif journal.source_kind == PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT:
        if (
            journal.session_id != sealed._telemetry_session_id
            or journal.selection_id != sealed._telemetry_selection_id
            or journal.batch_generation != sealed._telemetry_batch_generation
            or journal.source_v2_total_bytes != 0
            or journal.source_v2_integrity_tag != 0
            or memcmp(
                journal.sequence_content_fingerprint,
                &sealed._source_sequence_fingerprint[0],
                PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE,
            ) != 0
            or memcmp(
                &journal.forward,
                &sealed._source_forward_provenance,
                sizeof(plan7_forward_provenance),
            ) != 0
            or memcmp(
                &journal.backward,
                &sealed._source_backward_provenance,
                sizeof(plan7_backward_domain_provenance),
            ) != 0
            or memcmp(
                &journal.rescore,
                &sealed._source_rescore_provenance,
                sizeof(plan7_domain_rescore_provenance),
            ) != 0
        ):
            raise ValueError("continuation journal v3 direct source identity differs")
        source_identity_tokens = &sealed._source_identity_tokens[0]
        source_profile_fingerprints = &sealed._source_profile_fingerprints[0]
    elif (
        journal.session_id != 0
        or journal.selection_id != 0
        or journal.batch_generation != 0
        or journal.source_v2_total_bytes != 0
        or journal.source_v2_integrity_tag != 0
    ):
        raise ValueError("continuation journal v3 host source identity differs")

    base = <uint8_t *> journal
    profiles = <plan7_continuation_journal_v3_profile *> (
        base + journal.profiles_offset
    )
    certificates = <plan7_continuation_journal_v3_certificate *> (
        base + journal.certificates_offset
    )
    exceptions = <plan7_continuation_journal_v3_exception *> (
        base + journal.exceptions_offset
    )
    compact_trace_offsets = <const uint64_t *> (
        base + journal.compact_trace_offsets_offset
    )
    if (
        not plan7_continuation_journal_v3_checked_multiply(
            journal.compact_result_count,
            PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
            &compact_null2_expected,
        )
        or journal.compact_null2_count != compact_null2_expected
        or compact_trace_offsets[0] != 0
    ):
        raise ValueError("continuation journal v3 compact payload differs")
    for trace_offset_index in range(1, journal.compact_trace_offset_count):
        if (
            compact_trace_offsets[trace_offset_index]
            < compact_trace_offsets[trace_offset_index - 1]
            or compact_trace_offsets[trace_offset_index]
            > journal.compact_trace_count
        ):
            raise ValueError("continuation journal v3 trace offsets differ")
    if (
        compact_trace_offsets[journal.compact_trace_offset_count - 1]
        != journal.compact_trace_count
    ):
        raise ValueError("continuation journal v3 trace tail differs")
    cert_end = 0
    exception_end = 0
    for profile in range(<size_t> journal.profile_count):
        profile_record = &profiles[profile]
        if journal.source_kind == PLAN7_CONTINUATION_V3_SOURCE_HOST_SEAL:
            expected_source_domain_begin = 0
            expected_source_domain_count = 0
        else:
            expected_source_domain_begin = sealed._journal_profile_offsets[profile]
            expected_source_domain_count = (
                sealed._journal_profile_offsets[profile + 1]
                - sealed._journal_profile_offsets[profile]
            )
        if (
            profile_record.profile_index != profile
            or profile_record.profile_tag == 0
            or profile_record.profile_tag
            != plan7_continuation_journal_v3_profile_tag(profile_record)
            or profile_record.target_count != journal.target_count
            or profile_record.total_residues != journal.total_residues
            or profile_record.certificate_begin != cert_end
            or profile_record.exception_begin != exception_end
            or profile_record.certificate_count
            != profile_record.exception_count + 1
            or profile_record.certificate_begin > journal.certificate_count
            or profile_record.exception_begin > journal.exception_count
            or profile_record.certificate_count
            > journal.certificate_count - profile_record.certificate_begin
            or profile_record.exception_count
            > journal.exception_count - profile_record.exception_begin
            or profile_record.source_postfilter_begin
            != sealed._postfilter_offsets[profile]
            or profile_record.source_postfilter_count
            != sealed._postfilter_offsets[profile + 1]
            - sealed._postfilter_offsets[profile]
            or profile_record.source_forward_begin
            != sealed._forward_offsets[profile]
            or profile_record.source_forward_count
            != sealed._forward_offsets[profile + 1]
            - sealed._forward_offsets[profile]
            or profile_record.source_domain_begin
            != expected_source_domain_begin
            or profile_record.source_domain_count
            != expected_source_domain_count
        ):
            raise ValueError("continuation journal v3 profile certificate is invalid")
        if journal.source_kind in (
            PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL,
            PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT,
        ):
            if (
                profile_record.flags
                != (
                    PLAN7_CONTINUATION_V3_PROFILE_HAS_V2_IDENTITY
                    | PLAN7_CONTINUATION_V3_PROFILE_HAS_FINGERPRINT
                )
                or profile_record.identity_token
                != source_identity_tokens[profile]
                or memcmp(
                    profile_record.profile_fingerprint,
                    source_profile_fingerprints
                    + profile
                    * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
                    PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
                ) != 0
            ):
                raise ValueError(
                    "continuation journal v3 profile source identity differs"
                )
        elif (
            profile_record.flags != 0
            or profile_record.identity_token
            != <uint64_t> <size_t> (
                <OptimizedProfile> sealed._optimized_profiles[profile]
            )._om
        ):
            raise ValueError("continuation journal v3 host profile identity differs")
        cert_begin = profile_record.certificate_begin
        cert_end = cert_begin + profile_record.certificate_count
        exception_begin = profile_record.exception_begin
        exception_end = exception_begin + profile_record.exception_count
        target_cursor = 0
        for local_index in range(<size_t> profile_record.certificate_count):
            certificate = &certificates[cert_begin + local_index]
            if (
                certificate.profile_index != profile
                or certificate.segment_index != local_index
                or certificate.segment_tag == 0
                or certificate.segment_tag
                != plan7_continuation_journal_v3_certificate_tag(certificate)
                or certificate.target_begin != target_cursor
                or certificate.target_begin > certificate.target_end
                or certificate.target_end > journal.target_count
                or certificate.target_delta
                != certificate.target_end - certificate.target_begin
                or certificate.residue_prefix_begin
                != sealed._residue_offsets[certificate.target_begin]
                or certificate.residue_prefix_end
                != sealed._residue_offsets[certificate.target_end]
                or certificate.residue_prefix_begin
                > certificate.residue_prefix_end
                or certificate.residue_delta
                != certificate.residue_prefix_end
                - certificate.residue_prefix_begin
            ):
                raise ValueError("continuation journal v3 segment is invalid")
            stage_total = certificate.raw_f1_reject_count
            if (
                not _v3_checked_increment(
                    &stage_total, certificate.bias_reject_count
                )
                or not _v3_checked_increment(
                    &stage_total, certificate.f2_reject_count
                )
                or not _v3_checked_increment(
                    &stage_total, certificate.f3_reject_count
                )
                or not _v3_checked_increment(
                    &stage_total, certificate.no_region_count
                )
                or not _v3_checked_increment(
                    &stage_total, certificate.before_f1_count
                )
                or stage_total != certificate.target_delta
            ):
                raise ValueError("continuation journal v3 stage accounting differs")
            expected_value = certificate.bias_reject_count
            if (
                not _v3_checked_increment(
                    &expected_value, certificate.f2_reject_count
                )
                or not _v3_checked_increment(
                    &expected_value, certificate.f3_reject_count
                )
                or not _v3_checked_increment(
                    &expected_value, certificate.no_region_count
                )
                or certificate.n_past_msv_delta != expected_value
            ):
                raise ValueError("continuation journal v3 MSV delta differs")
            expected_value = certificate.f2_reject_count
            if (
                not _v3_checked_increment(
                    &expected_value, certificate.f3_reject_count
                )
                or not _v3_checked_increment(
                    &expected_value, certificate.no_region_count
                )
                or certificate.n_past_bias_delta != expected_value
            ):
                raise ValueError("continuation journal v3 bias delta differs")
            if (
                not plan7_continuation_journal_v3_checked_add(
                    certificate.f3_reject_count,
                    certificate.no_region_count,
                    &expected_value,
                )
                or certificate.n_past_vit_delta != expected_value
                or certificate.n_past_fwd_delta
                != certificate.no_region_count
            ):
                raise ValueError("continuation journal v3 late-filter delta differs")

            if local_index < profile_record.exception_count:
                exception = &exceptions[exception_begin + local_index]
                if (
                    exception.profile_index != profile
                    or exception.exception_index != local_index
                    or exception.sequence_index != certificate.target_end
                    or exception.sequence_index >= journal.target_count
                    or exception.residue_prefix_begin
                    != sealed._residue_offsets[exception.sequence_index]
                    or exception.residue_prefix_end
                    != sealed._residue_offsets[exception.sequence_index + 1]
                    or exception.residue_prefix_begin
                    > exception.residue_prefix_end
                    or exception.residue_delta
                    != exception.residue_prefix_end
                    - exception.residue_prefix_begin
                    or exception.route
                    not in (
                        PLAN7_CONTINUATION_V3_FULL_PIPELINE,
                        PLAN7_CONTINUATION_V3_FILTER_SCORES,
                        PLAN7_CONTINUATION_V3_FORWARD_SCORES,
                        PLAN7_CONTINUATION_V3_SIMPLE_REGIONS,
                        PLAN7_CONTINUATION_V3_COMPACT_DOMAINS,
                    )
                    or exception.source_stage
                    not in (
                        PLAN7_CONTINUATION_V3_CPU_REQUIRED,
                        PLAN7_CONTINUATION_V3_F2_SURVIVOR,
                        PLAN7_CONTINUATION_V3_F3_SURVIVOR,
                        PLAN7_CONTINUATION_V3_DOMAIN_CPU_REQUIRED,
                        PLAN7_CONTINUATION_V3_DOMAIN_SIMPLE,
                        PLAN7_CONTINUATION_V3_DOMAIN_COMPACT,
                    )
                    or (
                        exception.route == PLAN7_CONTINUATION_V3_FULL_PIPELINE
                        and exception.source_stage
                        != PLAN7_CONTINUATION_V3_CPU_REQUIRED
                    )
                    or (
                        exception.route == PLAN7_CONTINUATION_V3_FILTER_SCORES
                        and exception.source_stage
                        != PLAN7_CONTINUATION_V3_F2_SURVIVOR
                    )
                    or (
                        exception.route == PLAN7_CONTINUATION_V3_FORWARD_SCORES
                        and exception.source_stage not in (
                            PLAN7_CONTINUATION_V3_F3_SURVIVOR,
                            PLAN7_CONTINUATION_V3_DOMAIN_CPU_REQUIRED,
                        )
                    )
                    or (
                        exception.route == PLAN7_CONTINUATION_V3_SIMPLE_REGIONS
                        and exception.source_stage
                        != PLAN7_CONTINUATION_V3_DOMAIN_SIMPLE
                    )
                    or (
                        exception.route == PLAN7_CONTINUATION_V3_COMPACT_DOMAINS
                        and exception.source_stage
                        != PLAN7_CONTINUATION_V3_DOMAIN_COMPACT
                    )
                    or exception.payload_flags & <uint8_t> ~(
                        PLAN7_CONTINUATION_V3_HAS_POSTFILTER
                        | PLAN7_CONTINUATION_V3_HAS_FORWARD
                        | PLAN7_CONTINUATION_V3_HAS_DOMAIN
                        | PLAN7_CONTINUATION_V3_HAS_SPECIALS
                        | PLAN7_CONTINUATION_V3_HAS_REGIONS
                        | PLAN7_CONTINUATION_V3_HAS_COMPACT
                    )
                    or exception.preconditions & <uint8_t> ~(
                        PLAN7_CONTINUATION_V3_PRE_F2_SURVIVOR
                        | PLAN7_CONTINUATION_V3_PRE_DIRECT_FORWARD
                        | PLAN7_CONTINUATION_V3_PRE_F3_SURVIVOR
                        | PLAN7_CONTINUATION_V3_PRE_DOMAIN_SAFE
                        | PLAN7_CONTINUATION_V3_PRE_COMPACT_DEVICE
                    )
                    or exception.exception_tag == 0
                    or exception.exception_tag
                    != plan7_continuation_journal_v3_exception_tag(exception)
                    or exception.reserved != 0
                    or exception.source_postfilter_index
                    < profile_record.source_postfilter_begin
                    or exception.source_postfilter_index
                    >= profile_record.source_postfilter_begin
                    + profile_record.source_postfilter_count
                    or not (
                        exception.payload_flags
                        & PLAN7_CONTINUATION_V3_HAS_POSTFILTER
                    )
                    or bool(
                        exception.payload_flags
                        & PLAN7_CONTINUATION_V3_HAS_FORWARD
                    )
                    != (
                        exception.source_forward_index != <uint64_t> -1
                    )
                    or (
                        exception.source_forward_index != <uint64_t> -1
                        and not (
                            exception.preconditions
                            & PLAN7_CONTINUATION_V3_PRE_F2_SURVIVOR
                        )
                    )
                    or bool(
                        exception.payload_flags
                        & PLAN7_CONTINUATION_V3_HAS_DOMAIN
                    )
                    != (exception.source_domain_index != <uint64_t> -1)
                    or (
                        exception.source_forward_index != <uint64_t> -1
                        and (
                            exception.source_forward_index
                            < profile_record.source_forward_begin
                            or exception.source_forward_index
                            >= profile_record.source_forward_begin
                            + profile_record.source_forward_count
                        )
                    )
                    or (
                        exception.source_domain_index != <uint64_t> -1
                        and (
                            exception.source_domain_index
                            < profile_record.source_domain_begin
                            or exception.source_domain_index
                            >= profile_record.source_domain_begin
                            + profile_record.source_domain_count
                        )
                    )
                    or bool(
                        exception.payload_flags
                        & PLAN7_CONTINUATION_V3_HAS_SPECIALS
                    ) != bool(exception.special_count)
                    or bool(
                        exception.payload_flags
                        & PLAN7_CONTINUATION_V3_HAS_REGIONS
                    ) != bool(exception.region_count)
                    or bool(
                        exception.payload_flags
                        & PLAN7_CONTINUATION_V3_HAS_COMPACT
                    ) != (
                        exception.route == PLAN7_CONTINUATION_V3_COMPACT_DOMAINS
                    )
                ):
                    raise ValueError("continuation journal v3 exception is invalid")
                if memcmp(
                    exception.postfilter_record,
                    &sealed._postfilter_records[
                        exception.source_postfilter_index
                        * sizeof(_postfilter_result)
                    ],
                    sizeof(_postfilter_result),
                ) != 0:
                    raise ValueError(
                        "continuation journal v3 post-filter source differs"
                    )
                if (
                    exception.source_forward_index != <uint64_t> -1
                    and memcmp(
                        exception.forward_record,
                        &sealed._forward_records[
                            exception.source_forward_index
                            * sizeof(_forward_result)
                        ],
                        sizeof(_forward_result),
                    ) != 0
                ):
                    raise ValueError(
                        "continuation journal v3 Forward source differs"
                    )
                if (
                    exception.source_domain_index != <uint64_t> -1
                    and memcmp(
                        exception.domain_record,
                        &sealed._journal_rows[
                            exception.source_domain_index
                            * sizeof(plan7_continuation_journal_row)
                        ],
                        sizeof(plan7_continuation_journal_row),
                    ) != 0
                ):
                    raise ValueError(
                        "continuation journal v3 domain source differs"
                    )
                if exception.special_count:
                    if exception.special_begin != special_cursor:
                        raise ValueError("continuation journal v3 special span differs")
                    special_cursor += exception.special_count
                elif exception.special_begin != 0:
                    raise ValueError("empty journal v3 special span has an offset")
                if exception.region_count:
                    if exception.region_begin != region_cursor:
                        raise ValueError("continuation journal v3 region span differs")
                    region_cursor += exception.region_count
                elif exception.region_begin != 0:
                    raise ValueError("empty journal v3 region span has an offset")
                if exception.compact_result_count:
                    if (
                        exception.compact_result_begin != compact_cursor
                        or exception.compact_trace_begin != trace_cursor
                        or exception.compact_null2_begin != null2_cursor
                        or exception.compact_null2_count
                        != exception.compact_result_count
                        * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                        or compact_trace_offsets[
                            exception.compact_result_begin
                        ] != exception.compact_trace_begin
                        or compact_trace_offsets[
                            exception.compact_result_begin
                            + exception.compact_result_count
                        ]
                        != exception.compact_trace_begin
                        + exception.compact_trace_count
                    ):
                        raise ValueError("continuation journal v3 compact span differs")
                    compact_cursor += exception.compact_result_count
                    trace_cursor += exception.compact_trace_count
                    null2_cursor += exception.compact_null2_count
                elif (
                    exception.compact_result_begin != 0
                    or exception.compact_trace_begin != 0
                    or exception.compact_trace_count != 0
                    or exception.compact_null2_begin != 0
                    or exception.compact_null2_count != 0
                ):
                    raise ValueError("empty journal v3 compact span has an offset")
                target_cursor = <uint64_t> exception.sequence_index + 1
            else:
                if certificate.target_end != journal.target_count:
                    raise ValueError("journal v3 tail certificate is incomplete")
                target_cursor = certificate.target_end
        if target_cursor != journal.target_count:
            raise ValueError("journal v3 profile partition is incomplete")
    if (
        cert_end != journal.certificate_count
        or exception_end != journal.exception_count
        or special_cursor != journal.special_count
        or region_cursor != journal.region_count
        or compact_cursor != journal.compact_result_count
        or trace_cursor != journal.compact_trace_count
        or null2_cursor != journal.compact_null2_count
    ):
        raise ValueError("continuation journal v3 payload partition differs")

    if rebuild_source:
        # The dual/debug oracle independently rebuilds every certificate,
        # route, identity, and copied payload byte from the dense source.
        expected = _v3_allocate_from_seal(sealed)
        try:
            if (
                expected.total_bytes != journal.total_bytes
                or memcmp(journal, expected, <size_t> journal.total_bytes) != 0
            ):
                raise ValueError(
                    "continuation journal v3 differs from its dense source"
                )
        finally:
            free(expected)
    return journal


cdef void _v3_drop_direct_staging(_SealedPostfilterBatch sealed) except *:
    """Release every dense planning view after direct v3 is authenticated."""
    cdef object empty_bytes = b""
    cdef object empty_q = memoryview(b"").cast("Q")
    cdef object zero_q = memoryview(bytes(sizeof(uint64_t))).cast("Q")
    cdef object empty_f = memoryview(b"").cast("f")
    if not sealed._direct_v3_source:
        return
    if sealed._journal_v3 == NULL:
        raise RuntimeError("direct v3 staging cannot be dropped before sealing")
    sealed._postfilter_records = empty_bytes
    sealed._postfilter_offsets = zero_q
    sealed._forward_records = empty_bytes
    sealed._forward_offsets = zero_q
    sealed._special_offsets = zero_q
    sealed._specials = empty_f
    sealed._row_has_external = empty_bytes
    sealed._journal_storage = empty_bytes
    sealed._journal_profile_offsets = zero_q
    sealed._journal_rows = empty_bytes
    sealed._journal_region_offsets = zero_q
    sealed._journal_regions = empty_bytes
    sealed._journal_compact_row_offsets = zero_q
    sealed._journal_compact_results = empty_bytes
    sealed._journal_compact_trace_offsets = zero_q
    sealed._journal_compact_traces = empty_bytes
    sealed._journal_compact_null2 = empty_f
    sealed._source_identity_tokens = empty_q
    sealed._source_profile_fingerprints = empty_bytes
    sealed._source_sequence_fingerprint = empty_bytes
    sealed._direct_v3_decisions = empty_bytes


cdef plan7_continuation_journal_v3 *_v3_validate_capsule(
    object capsule,
    _SealedPostfilterBatch sealed,
    plan7_continuation_journal_v3_owner **owner_out,
    bint rebuild_source,
) except NULL:
    cdef plan7_continuation_journal_v3 *journal
    cdef plan7_continuation_journal_v3_owner *owner

    if not PyCapsule_IsValid(
        capsule, PLAN7_CONTINUATION_JOURNAL_V3_CAPSULE_NAME
    ):
        raise TypeError("continuation journal v3 capsule is invalid or consumed")
    journal = <plan7_continuation_journal_v3 *> PyCapsule_GetPointer(
        capsule, PLAN7_CONTINUATION_JOURNAL_V3_CAPSULE_NAME
    )
    owner = <plan7_continuation_journal_v3_owner *> PyCapsule_GetContext(
        capsule
    )
    if journal == NULL or owner == NULL:
        raise TypeError("continuation journal v3 capsule has no storage owner")
    journal = _v3_validate_packet(
        journal,
        owner.allocation_bytes,
        owner.source_seal_token,
        sealed,
        rebuild_source,
    )
    owner_out[0] = owner
    return journal


cdef dict _v3_debug_summary(
    const plan7_continuation_journal_v3 *journal,
    bint include_details,
):
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    cdef const plan7_continuation_journal_v3_certificate *certificates = (
        <const plan7_continuation_journal_v3_certificate *> (
            base + journal.certificates_offset
        )
    )
    cdef const plan7_continuation_journal_v3_exception *exceptions = (
        <const plan7_continuation_journal_v3_exception *> (
            base + journal.exceptions_offset
        )
    )
    cdef const uint64_t *compact_trace_offsets = <const uint64_t *> (
        base + journal.compact_trace_offsets_offset
    )
    cdef const plan7_continuation_journal_v3_profile *profile_record
    cdef const plan7_continuation_journal_v3_certificate *certificate
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef size_t profile
    cdef size_t index
    cdef uint64_t before_f1 = 0
    cdef uint64_t raw_f1 = 0
    cdef uint64_t bias_reject = 0
    cdef uint64_t f2_reject = 0
    cdef uint64_t f3_reject = 0
    cdef uint64_t no_region = 0
    cdef uint64_t past_msv = 0
    cdef uint64_t past_bias = 0
    cdef uint64_t past_vit = 0
    cdef uint64_t past_fwd = 0
    cdef uint64_t full_pipeline = 0
    cdef uint64_t filter_scores = 0
    cdef uint64_t forward_scores = 0
    cdef uint64_t simple_regions = 0
    cdef uint64_t compact_domains = 0
    cdef list profile_details = []
    cdef list certificate_details
    cdef list exception_details

    for index in range(<size_t> journal.certificate_count):
        certificate = &certificates[index]
        before_f1 += certificate.before_f1_count
        raw_f1 += certificate.raw_f1_reject_count
        bias_reject += certificate.bias_reject_count
        f2_reject += certificate.f2_reject_count
        f3_reject += certificate.f3_reject_count
        no_region += certificate.no_region_count
        past_msv += certificate.n_past_msv_delta
        past_bias += certificate.n_past_bias_delta
        past_vit += certificate.n_past_vit_delta
        past_fwd += certificate.n_past_fwd_delta
    for index in range(<size_t> journal.exception_count):
        exception = &exceptions[index]
        if exception.route == PLAN7_CONTINUATION_V3_FULL_PIPELINE:
            full_pipeline += 1
        elif exception.route == PLAN7_CONTINUATION_V3_FILTER_SCORES:
            filter_scores += 1
        elif exception.route == PLAN7_CONTINUATION_V3_FORWARD_SCORES:
            forward_scores += 1
        elif exception.route == PLAN7_CONTINUATION_V3_SIMPLE_REGIONS:
            simple_regions += 1
        elif exception.route == PLAN7_CONTINUATION_V3_COMPACT_DOMAINS:
            compact_domains += 1

    if include_details:
        for profile in range(<size_t> journal.profile_count):
            profile_record = &profiles[profile]
            certificate_details = []
            exception_details = []
            for index in range(<size_t> profile_record.certificate_count):
                certificate = &certificates[
                    profile_record.certificate_begin + index
                ]
                certificate_details.append({
                    "begin": certificate.target_begin,
                    "end": certificate.target_end,
                    "residue_prefix_begin": certificate.residue_prefix_begin,
                    "residue_prefix_end": certificate.residue_prefix_end,
                    "target_delta": certificate.target_delta,
                    "residue_delta": certificate.residue_delta,
                    "before_f1": certificate.before_f1_count,
                    "raw_f1_reject": certificate.raw_f1_reject_count,
                    "bias_reject": certificate.bias_reject_count,
                    "f2_reject": certificate.f2_reject_count,
                    "f3_reject": certificate.f3_reject_count,
                    "no_region": certificate.no_region_count,
                    "promotions": (
                        certificate.n_past_msv_delta,
                        certificate.n_past_bias_delta,
                        certificate.n_past_vit_delta,
                        certificate.n_past_fwd_delta,
                    ),
                })
            for index in range(<size_t> profile_record.exception_count):
                exception = &exceptions[
                    profile_record.exception_begin + index
                ]
                exception_details.append({
                    "sequence_index": exception.sequence_index,
                    "source_stage": exception.source_stage,
                    "route": exception.route,
                    "payload_flags": exception.payload_flags,
                    "preconditions": exception.preconditions,
                    "residue_prefix_begin": exception.residue_prefix_begin,
                    "residue_prefix_end": exception.residue_prefix_end,
                    "source_postfilter_index": (
                        exception.source_postfilter_index
                    ),
                    "source_forward_index": exception.source_forward_index,
                    "source_domain_index": exception.source_domain_index,
                    "special_begin": exception.special_begin,
                    "special_count": exception.special_count,
                    "region_begin": exception.region_begin,
                    "region_count": exception.region_count,
                    "compact_result_begin": exception.compact_result_begin,
                    "compact_result_count": exception.compact_result_count,
                    "compact_trace_begin": exception.compact_trace_begin,
                    "compact_trace_count": exception.compact_trace_count,
                    "compact_null2_begin": exception.compact_null2_begin,
                    "compact_null2_count": exception.compact_null2_count,
                })
            profile_details.append({
                "profile_index": profile,
                "target_count": profile_record.target_count,
                "total_residues": profile_record.total_residues,
                "identity_token": profile_record.identity_token,
                "flags": profile_record.flags,
                "source_postfilter_span": (
                    profile_record.source_postfilter_begin,
                    profile_record.source_postfilter_begin
                    + profile_record.source_postfilter_count,
                ),
                "source_forward_span": (
                    profile_record.source_forward_begin,
                    profile_record.source_forward_begin
                    + profile_record.source_forward_count,
                ),
                "source_domain_span": (
                    profile_record.source_domain_begin,
                    profile_record.source_domain_begin
                    + profile_record.source_domain_count,
                ),
                "certificates": tuple(certificate_details),
                "exceptions": tuple(exception_details),
            })

    return {
        "schema_version": journal.version,
        "source_kind": (
            "native_direct"
            if journal.source_kind == PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT
            else (
                "v2_journal"
                if journal.source_kind
                == PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL
                else "host_seal"
            )
        ),
        "profile_count": journal.profile_count,
        "target_count": journal.target_count,
        "total_residues": journal.total_residues,
        "dense_postfilter_count": journal.source_postfilter_count,
        "dense_forward_count": journal.source_forward_count,
        "dense_domain_count": journal.source_domain_count,
        "certificate_count": journal.certificate_count,
        "exception_count": journal.exception_count,
        "stage_counts": {
            "before_f1": before_f1,
            "raw_f1_reject": raw_f1,
            "bias_reject": bias_reject,
            "f2_reject": f2_reject,
            "f3_reject": f3_reject,
            "domain_no_regions": no_region,
        },
        "promotion_deltas": {
            "n_past_msv": past_msv,
            "n_past_bias": past_bias,
            "n_past_vit": past_vit,
            "n_past_fwd": past_fwd,
        },
        "exception_routes": {
            "full_pipeline": full_pipeline,
            "filter_scores": filter_scores,
            "forward_scores": forward_scores,
            "simple_regions": simple_regions,
            "compact_domains": compact_domains,
        },
        "payload_counts": {
            "specials": journal.special_count,
            "regions": journal.region_count,
            "compact_results": journal.compact_result_count,
            "compact_traces": journal.compact_trace_count,
            "compact_null2": journal.compact_null2_count,
        },
        "packet_bytes": journal.total_bytes,
        "source_v2_bytes": journal.source_v2_total_bytes,
        "source_v2_integrity_tag": journal.source_v2_integrity_tag,
        "session_id": journal.session_id,
        "selection_id": journal.selection_id,
        "batch_generation": journal.batch_generation,
        "generation_tail_fingerprint": journal.generation_tail_fingerprint,
        "options_complete": bool(journal.options.complete),
        "header_size": journal.header_size,
        "total_bytes_offset": (
            <size_t> &journal.total_bytes - <size_t> journal
        ),
        "integrity_offset": (
            <size_t> &journal.integrity_tag - <size_t> journal
        ),
        "profiles_offset": journal.profiles_offset,
        "certificates_offset": journal.certificates_offset,
        "exceptions_offset": journal.exceptions_offset,
        "profile_record_size": journal.profile_size,
        "certificate_record_size": journal.certificate_size,
        "exception_record_size": journal.exception_size,
        "compact_trace_offsets": (
            tuple(
                compact_trace_offsets[index]
                for index in range(
                    <size_t> journal.compact_trace_offset_count
                )
            )
            if include_details
            else None
        ),
        "profiles": tuple(profile_details) if include_details else None,
    }


cdef void _v3_retire_debug_capsule(
    object capsule,
    plan7_continuation_journal_v3 *journal,
    plan7_continuation_journal_v3_owner *owner,
) except *:
    cdef bint ownership_transferred = False
    try:
        # Cython declares every PyCapsule setter ``except -1``.  Once the
        # destructor is cleared, this frame owns both allocations on success
        # and on every later exceptional edge.
        PyCapsule_SetDestructor(capsule, <PyCapsule_Destructor> NULL)
        ownership_transferred = True
        try:
            PyCapsule_SetPointer(capsule, &_consumed_journal_sentinel)
            PyCapsule_SetContext(capsule, NULL)
            PyCapsule_SetName(
                capsule, PLAN7_CONTINUATION_JOURNAL_V3_CONSUMED_NAME
            )
        except:
            # Make a partially retired capsule fail its public name check
            # before the finally block releases its former storage.
            try:
                PyCapsule_SetName(
                    capsule, PLAN7_CONTINUATION_JOURNAL_V3_CONSUMED_NAME
                )
            except:
                pass
            raise
    finally:
        if ownership_transferred:
            free(journal)
            free(owner)


def _plan_continuation_journal_v3_bound(sealed_object):
    """Compact one validated dense seal into an opaque debug journal-v3 packet.

    The production opt-in retains its packet inside the seal. This capsule
    boundary remains available for one-shot validation and dual-oracle work.
    """
    cdef _SealedPostfilterBatch sealed
    cdef plan7_continuation_journal_v3 *journal = NULL
    cdef plan7_continuation_journal_v3_owner *owner = NULL
    cdef object capsule

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    journal = _v3_allocate_from_seal(sealed)
    owner = <plan7_continuation_journal_v3_owner *> calloc(
        1, sizeof(plan7_continuation_journal_v3_owner)
    )
    if owner == NULL:
        free(journal)
        raise MemoryError("journal v3 capsule owner allocation failed")
    owner.allocation_bytes = journal.total_bytes
    owner.source_seal_token = journal.source_seal_token
    try:
        capsule = PyCapsule_New(
            journal,
            PLAN7_CONTINUATION_JOURNAL_V3_CAPSULE_NAME,
            _v3_capsule_destroy,
        )
    except:
        free(journal)
        free(owner)
        raise
    try:
        PyCapsule_SetContext(capsule, owner)
    except:
        # The capsule destructor still owns ``journal``.  Releasing only the
        # unattached context avoids both the Cython auto-raise leak and a
        # destructor double-free while the local capsule unwinds.
        free(owner)
        raise
    return capsule


def _validate_continuation_journal_v3_bound(
    capsule,
    sealed_object,
    bint consume=False,
    bint include_details=False,
):
    """Validate a v3 packet against its dense seal and return a debug summary.

    ``consume=True`` is a one-shot debug sink used before the sparse consumer
    exists.  Failed validation never retires the capsule or mutates any HMMER
    pipeline state.
    """
    cdef _SealedPostfilterBatch sealed
    cdef plan7_continuation_journal_v3 *journal
    cdef plan7_continuation_journal_v3_owner *owner = NULL
    cdef dict summary

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    journal = _v3_validate_capsule(capsule, sealed, &owner, True)
    summary = _v3_debug_summary(journal, include_details)
    if consume:
        _v3_retire_debug_capsule(capsule, journal, owner)
    return summary


cdef tuple _consume_validate_continuation_journal(
    object capsule,
    tuple profiles,
    DigitalSequenceBlock sequences,
    uint64_t expected_session_id,
    uint64_t expected_selection_id,
    const uint64_t[::1] expected_identity_tokens,
    const uint8_t[::1] expected_profile_fingerprints,
    uint64_t expected_batch_generation,
    const uint8_t[::1] expected_sequence_fingerprint,
    double f1,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    bint generation_bias_filter,
    uint64_t expected_tail_fingerprint,
):
    cdef plan7_continuation_journal *journal
    cdef size_t cursor
    cdef size_t profile_count = len(profiles)
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t profile
    cdef size_t row
    cdef size_t row_start
    cdef size_t row_stop
    cdef size_t postfilter_cursor
    cdef size_t postfilter_stop
    cdef size_t forward_cursor
    cdef size_t forward_stop
    cdef size_t region
    cdef size_t region_start
    cdef size_t region_stop
    cdef size_t compact_start
    cdef size_t compact_stop
    cdef size_t compact_index
    cdef size_t trace_start
    cdef size_t trace_stop
    cdef size_t simple_row_count = 0
    cdef size_t device_result_count = 0
    cdef size_t cpu_required_count = 0
    cdef size_t compact_bytes
    cdef size_t residue
    cdef uint8_t compact_row_action
    cdef uint8_t expected_compact_route
    cdef float null2_value
    cdef uint32_t previous_sequence
    cdef uint32_t previous_end
    cdef uint32_t guard_bits
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef plan7_continuation_journal_row journal_row
    cdef plan7_simple_region interval
    cdef plan7_domain_rescore_result compact_result
    cdef OptimizedProfile optimized_profile
    cdef float computed_usc
    cdef _float_bits expected_float
    cdef _float_bits observed_float
    cdef _double_bits expected_double
    cdef object tokens = set()
    cdef _ContinuationJournalStorage storage
    cdef object storage_view
    cdef const uint8_t[::1] postfilter_view
    cdef const uint64_t[::1] postfilter_offset_view
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint64_t[::1] profile_offset_view
    cdef const uint64_t[::1] identity_token_view
    cdef const uint8_t[::1] profile_fingerprint_view
    cdef const uint8_t[::1] row_view
    cdef const uint64_t[::1] journal_special_offset_view
    cdef const uint64_t[::1] region_offset_view
    cdef const uint8_t[::1] region_view
    cdef const uint64_t[::1] compact_row_offset_view
    cdef const uint8_t[::1] compact_result_view
    cdef const uint64_t[::1] compact_trace_offset_view
    cdef const uint8_t[::1] compact_trace_view
    cdef const float[::1] compact_null2_view
    cdef uint64_t observed_result_hash = 0
    cdef uint64_t observed_trace_hash = 0
    cdef uint64_t observed_null2_hash = 0

    if not PyCapsule_IsValid(
        capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    ):
        raise TypeError("continuation journal capsule is invalid or consumed")
    journal = <plan7_continuation_journal *> PyCapsule_GetPointer(
        capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    )
    if journal == NULL:
        raise TypeError("continuation journal capsule has no storage")
    if (
        journal.magic != PLAN7_CONTINUATION_JOURNAL_MAGIC
        or journal.version != PLAN7_CONTINUATION_JOURNAL_VERSION
        or journal.header_size != sizeof(plan7_continuation_journal)
        or journal.row_size != sizeof(plan7_continuation_journal_row)
        or journal.region_size != sizeof(plan7_simple_region)
        or journal.compact_result_size
        != sizeof(plan7_domain_rescore_result)
        or journal.compact_trace_step_size
        != sizeof(plan7_domain_rescore_trace_step)
        or journal.compact_null2_stride
        != PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.total_bytes < sizeof(plan7_continuation_journal)
        or journal.total_bytes > <uint64_t> (<size_t> -1)
        or journal.total_bytes > <uint64_t> PY_SSIZE_T_MAX
        or journal.profile_count != profile_count
        or journal.profile_count > <uint64_t> (<size_t> -1) - 1
        or journal.postfilter_count > <uint64_t> (<size_t> -1)
        or journal.forward_count > <uint64_t> (<size_t> -1) - 1
        or journal.row_count > <uint64_t> (<size_t> -1) - 1
        or journal.special_count > <uint64_t> (<size_t> -1)
        or journal.region_count > <uint64_t> (<size_t> -1)
        or journal.compact_result_count > <uint64_t> (<size_t> -1) - 1
        or journal.compact_trace_offset_count
        > <uint64_t> (<size_t> -1)
        or journal.compact_trace_count > <uint64_t> (<size_t> -1)
        or journal.compact_null2_count > <uint64_t> (<size_t> -1)
    ):
        raise ValueError("continuation journal ABI header is invalid")
    cursor = sizeof(plan7_continuation_journal)
    if not (
        _journal_expected_segment(
            &cursor, journal.postfilter_offsets_offset,
            journal.profile_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.postfilter_records_offset,
            journal.postfilter_count, sizeof(_postfilter_result),
        )
        and _journal_expected_segment(
            &cursor, journal.forward_offsets_offset,
            journal.profile_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.forward_records_offset,
            journal.forward_count, sizeof(_forward_result),
        )
        and _journal_expected_segment(
            &cursor, journal.forward_special_offsets_offset,
            journal.forward_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.profile_offsets_offset,
            journal.profile_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.identity_tokens_offset,
            journal.profile_count, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.profile_fingerprints_offset,
            journal.profile_count,
            PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
        )
        and _journal_expected_segment(
            &cursor, journal.rows_offset,
            journal.row_count, sizeof(plan7_continuation_journal_row),
        )
        and _journal_expected_segment(
            &cursor, journal.special_offsets_offset,
            journal.row_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.specials_offset,
            journal.special_count, sizeof(float),
        )
        and _journal_expected_segment(
            &cursor, journal.region_offsets_offset,
            journal.row_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.regions_offset,
            journal.region_count, sizeof(plan7_simple_region),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_row_offsets_offset,
            journal.row_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_results_offset,
            journal.compact_result_count,
            sizeof(plan7_domain_rescore_result),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_trace_offsets_offset,
            journal.compact_trace_offset_count, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_traces_offset,
            journal.compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_null2_offset,
            journal.compact_null2_count, sizeof(float),
        )
        and cursor == <size_t> journal.total_bytes
    ):
        raise ValueError("continuation journal storage layout is invalid")
    if (
        journal.integrity_tag == 0
        or journal.integrity_tag
        != plan7_continuation_journal_integrity(journal)
    ):
        raise ValueError("continuation journal integrity check failed")
    if (
        journal.session_id == 0
        or journal.selection_id == 0
        or journal.session_id != expected_session_id
        or journal.selection_id != expected_selection_id
        or journal.profile_count != profile_count
    ):
        raise ValueError("continuation journal selection identity differs")
    if expected_identity_tokens.shape[0] != profile_count:
        raise ValueError("continuation journal profile identity count differs")
    if expected_profile_fingerprints.shape[0] != (
        profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
    ):
        raise ValueError("continuation journal profile fingerprint count differs")
    if expected_sequence_fingerprint.shape[0] != 32:
        raise ValueError("sequence content fingerprint has the wrong size")
    if memcmp(
        journal.sequence_content_fingerprint,
        &expected_sequence_fingerprint[0],
        32,
    ) != 0:
        raise ValueError("continuation journal target content differs")
    expected_double.value = f1
    if journal.generation_f1_bits != expected_double.bits:
        raise ValueError("continuation journal F1 provenance differs")
    if (
        journal.generation_f2_bits != generation_f2_bits
        or journal.generation_f3_bits != generation_f3_bits
        or journal.generation_bias_filter != <uint8_t> generation_bias_filter
        or not generation_bias_filter
    ):
        raise ValueError("continuation journal filter provenance differs")
    expected_float.value = <float> 0.25
    if journal.rt1_bits != expected_float.bits:
        raise ValueError("continuation journal rt1 differs")
    expected_float.value = <float> 0.10
    if journal.rt2_bits != expected_float.bits:
        raise ValueError("continuation journal rt2 differs")
    expected_float.value = <float> 0.20
    if journal.rt3_bits != expected_float.bits:
        raise ValueError("continuation journal rt3 differs")
    observed_float.bits = journal.guard_band_bits
    guard_bits = journal.guard_band_bits
    if (
        not isfinite(observed_float.value)
        or observed_float.value < <float> 2.0e-4
        or observed_float.value > <float> 1.0
    ):
        raise ValueError("continuation journal guard band is invalid")
    if (
        journal.forward.database_generation == 0
        or journal.forward.batch_generation == 0
        or journal.forward.batch_generation != expected_batch_generation
        or journal.forward.pass_count != journal.row_count
        or journal.forward.special_count != journal.special_count
        or journal.forward.generation_f3_bits != generation_f3_bits
        or journal.backward.candidate_count != journal.row_count
        or journal.backward.region_count != journal.region_count
        or memcmp(
            &journal.forward,
            &journal.backward.forward,
            sizeof(plan7_forward_provenance),
        ) != 0
    ):
        raise ValueError("continuation journal native provenance differs")
    if journal.generation_compact_domains > 1:
        raise ValueError("continuation journal compact-domain flag is invalid")
    if not journal.generation_compact_domains:
        if (
            journal.generation_tail_fingerprint != 0
            or journal.compact_result_count != 0
            or journal.compact_trace_offset_count != 1
            or journal.compact_trace_count != 0
            or journal.compact_null2_count != 0
            or journal.rescore_simple_row_count != 0
            or journal.rescore_device_result_count != 0
            or journal.rescore_cpu_required_count != 0
            or journal.rescore_numeric_fallback_count != 0
            or journal.rescore_cap_fallback_count != 0
            or journal.rescore_global_cpu_fallback_count != 0
            or journal.rescore_compact_output_byte_limit != 0
            or journal.rescore_compact_output_bytes != 0
            or journal.compact_global_fallback != 0
            or journal.rescore.result_hash != 0
            or journal.rescore.trace_hash != 0
            or journal.rescore.null2_hash != 0
            or journal.rescore.result_count != 0
            or journal.rescore.trace_count != 0
            or journal.rescore.null2_count != 0
        ):
            raise ValueError("disabled compact-domain journal carries output")
    elif (
        journal.generation_tail_fingerprint == 0
        or expected_tail_fingerprint == 0
        or journal.generation_tail_fingerprint != expected_tail_fingerprint
        or journal.compact_trace_offset_count
        != journal.compact_result_count + 1
        or journal.compact_global_fallback > 1
        or memcmp(
            &journal.rescore.backward,
            &journal.backward,
            sizeof(plan7_backward_domain_provenance),
        ) != 0
        or journal.rescore.result_count != journal.compact_result_count
        or journal.rescore.trace_count != journal.compact_trace_count
        or journal.rescore.null2_count != journal.compact_null2_count
        or journal.rescore_simple_row_count > journal.row_count
        or journal.rescore_device_result_count
        + journal.rescore_cpu_required_count != journal.region_count
        or journal.rescore_numeric_fallback_count
        + journal.rescore_cap_fallback_count
        != journal.rescore_cpu_required_count
        or journal.rescore_global_cpu_fallback_count
        not in (0, journal.region_count)
        or bool(journal.compact_global_fallback)
        != (journal.rescore_global_cpu_fallback_count != 0)
        or journal.rescore_compact_output_byte_limit < sizeof(uint64_t)
        or journal.rescore_compact_output_byte_limit
        > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        or journal.rescore_compact_output_bytes
        > journal.rescore_compact_output_byte_limit
        or journal.compact_result_count
        > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        // sizeof(plan7_domain_rescore_result)
        or journal.compact_trace_count
        > PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES
        // sizeof(plan7_domain_rescore_trace_step)
        or journal.compact_result_count
        > (<uint64_t> -1) // PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.compact_null2_count
        != journal.compact_result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.compact_result_count not in (0, journal.region_count)
    ):
        raise ValueError("continuation journal compact provenance differs")

    storage = _ContinuationJournalStorage.__new__(_ContinuationJournalStorage)
    storage._data = <uint8_t *> journal
    storage._size = <Py_ssize_t> journal.total_bytes
    storage_view = memoryview(storage).cast("B")
    postfilter_offset_view = storage_view[
        journal.postfilter_offsets_offset:
        journal.postfilter_offsets_offset
        + (profile_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    postfilter_view = storage_view[
        journal.postfilter_records_offset:
        journal.postfilter_records_offset
        + <size_t> journal.postfilter_count * sizeof(_postfilter_result)
    ]
    forward_offset_view = storage_view[
        journal.forward_offsets_offset:
        journal.forward_offsets_offset
        + (profile_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    forward_view = storage_view[
        journal.forward_records_offset:
        journal.forward_records_offset
        + <size_t> journal.forward_count * sizeof(_forward_result)
    ]
    special_offset_view = storage_view[
        journal.forward_special_offsets_offset:
        journal.forward_special_offsets_offset
        + (<size_t> journal.forward_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    profile_offset_view = storage_view[
        journal.profile_offsets_offset:
        journal.profile_offsets_offset + (profile_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    identity_token_view = storage_view[
        journal.identity_tokens_offset:
        journal.identity_tokens_offset + profile_count * sizeof(uint64_t)
    ].cast("Q")
    profile_fingerprint_view = storage_view[
        journal.profile_fingerprints_offset:
        journal.profile_fingerprints_offset
        + profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
    ]
    row_view = storage_view[
        journal.rows_offset:
        journal.rows_offset
        + <size_t> journal.row_count * sizeof(plan7_continuation_journal_row)
    ]
    journal_special_offset_view = storage_view[
        journal.special_offsets_offset:
        journal.special_offsets_offset
        + (<size_t> journal.row_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    special_view = storage_view[
        journal.specials_offset:
        journal.specials_offset
        + <size_t> journal.special_count * sizeof(float)
    ].cast("f")
    region_offset_view = storage_view[
        journal.region_offsets_offset:
        journal.region_offsets_offset
        + (<size_t> journal.row_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    region_view = storage_view[
        journal.regions_offset:
        journal.regions_offset
        + <size_t> journal.region_count * sizeof(plan7_simple_region)
    ]
    compact_row_offset_view = storage_view[
        journal.compact_row_offsets_offset:
        journal.compact_row_offsets_offset
        + (<size_t> journal.row_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    compact_result_view = storage_view[
        journal.compact_results_offset:
        journal.compact_results_offset
        + <size_t> journal.compact_result_count
        * sizeof(plan7_domain_rescore_result)
    ]
    compact_trace_offset_view = storage_view[
        journal.compact_trace_offsets_offset:
        journal.compact_trace_offsets_offset
        + <size_t> journal.compact_trace_offset_count * sizeof(uint64_t)
    ].cast("Q")
    compact_trace_view = storage_view[
        journal.compact_traces_offset:
        journal.compact_traces_offset
        + <size_t> journal.compact_trace_count
        * sizeof(plan7_domain_rescore_trace_step)
    ]
    compact_null2_view = storage_view[
        journal.compact_null2_offset:
        journal.compact_null2_offset
        + <size_t> journal.compact_null2_count * sizeof(float)
    ].cast("f")

    if (
        postfilter_offset_view[0] != 0
        or postfilter_offset_view[profile_count] != journal.postfilter_count
    ):
        raise ValueError("continuation journal post-filter offsets do not span rows")
    if (
        forward_offset_view[0] != 0
        or forward_offset_view[profile_count] != journal.forward_count
    ):
        raise ValueError("continuation journal Forward offsets do not span rows")
    if (
        special_offset_view[0] != 0
        or special_offset_view[journal.forward_count] != journal.special_count
    ):
        raise ValueError("continuation journal Forward specials do not span storage")
    if profile_offset_view[0] != 0 or (
        profile_offset_view[profile_count] != journal.row_count
    ):
        raise ValueError("continuation journal profile offsets do not span rows")
    if journal_special_offset_view[0] != 0 or (
        journal_special_offset_view[journal.row_count] != journal.special_count
    ):
        raise ValueError("continuation journal special offsets do not span storage")
    if region_offset_view[0] != 0 or (
        region_offset_view[journal.row_count] != journal.region_count
    ):
        raise ValueError("continuation journal region offsets do not span storage")
    if compact_row_offset_view[0] != 0 or (
        compact_row_offset_view[journal.row_count]
        != journal.compact_result_count
    ):
        raise ValueError("continuation journal compact rows do not span storage")
    if compact_trace_offset_view[0] != 0 or (
        compact_trace_offset_view[journal.compact_trace_offset_count - 1]
        != journal.compact_trace_count
    ):
        raise ValueError("continuation journal compact traces do not span storage")
    if journal.generation_compact_domains:
        compact_bytes = (
            <size_t> journal.compact_result_count
            * sizeof(plan7_domain_rescore_result)
            + (<size_t> journal.compact_result_count + 1) * sizeof(uint64_t)
            + <size_t> journal.compact_trace_count
            * sizeof(plan7_domain_rescore_trace_step)
            + <size_t> journal.compact_null2_count * sizeof(float)
        )
        if compact_bytes != journal.rescore_compact_output_bytes:
            raise ValueError("continuation journal compact byte count differs")
        if not plan7_continuation_journal_rescore_hashes(
            (
                <const plan7_domain_rescore_result *> &compact_result_view[0]
                if journal.compact_result_count
                else NULL
            ),
            journal.compact_result_count,
            &compact_trace_offset_view[0],
            (
                <const plan7_domain_rescore_trace_step *> &compact_trace_view[0]
                if journal.compact_trace_count
                else NULL
            ),
            journal.compact_trace_count,
            &compact_null2_view[0] if journal.compact_null2_count else NULL,
            journal.compact_null2_count,
            &observed_result_hash,
            &observed_trace_hash,
            &observed_null2_hash,
        ):
            raise ValueError("continuation journal compact hashes are invalid")
        if (
            observed_result_hash != journal.rescore.result_hash
            or observed_trace_hash != journal.rescore.trace_hash
            or observed_null2_hash != journal.rescore.null2_hash
        ):
            raise ValueError("continuation journal compact hashes differ")
        if journal.compact_result_count == 0:
            if (
                journal.compact_trace_count != 0
                or journal.rescore_device_result_count != 0
                or journal.rescore_cpu_required_count != journal.region_count
                or journal.rescore_global_cpu_fallback_count
                != journal.region_count
            ):
                raise ValueError(
                    "continuation journal global compact fallback differs"
                )
        elif journal.rescore_global_cpu_fallback_count != 0:
            raise ValueError("continuation journal compact fallback differs")
    else:
        for row in range(<size_t> journal.row_count + 1):
            if compact_row_offset_view[row] != 0:
                raise ValueError("disabled compact journal has row offsets")
        if compact_trace_offset_view[0] != 0:
            raise ValueError("disabled compact journal has trace offsets")
    for profile in range(profile_count):
        if (
            identity_token_view[profile] == 0
            or identity_token_view[profile] in tokens
        ):
            raise ValueError("continuation journal profile identity is invalid")
        if identity_token_view[profile] != expected_identity_tokens[profile]:
            raise ValueError("continuation journal profile identity differs")
        tokens.add(identity_token_view[profile])
    if profile_count and memcmp(
        &profile_fingerprint_view[0],
        &expected_profile_fingerprints[0],
        profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
    ) != 0:
        raise ValueError("continuation journal optimized-profile snapshot differs")

    postfilter_count = <size_t> journal.postfilter_count
    forward_count = <size_t> journal.forward_count

    for profile in range(profile_count):
        row_start = <size_t> profile_offset_view[profile]
        row_stop = <size_t> profile_offset_view[profile + 1]
        if row_start > row_stop or row_stop > journal.row_count:
            raise ValueError("continuation journal profile offsets are not monotone")
        postfilter_cursor = <size_t> postfilter_offset_view[profile]
        postfilter_stop = <size_t> postfilter_offset_view[profile + 1]
        forward_cursor = <size_t> forward_offset_view[profile]
        forward_stop = <size_t> forward_offset_view[profile + 1]
        if (
            postfilter_cursor > postfilter_stop
            or postfilter_stop > postfilter_count
            or forward_cursor > forward_stop
            or forward_stop > forward_count
        ):
            raise ValueError("continuation journal row offsets are not monotone")
        row = row_start
        while forward_cursor < forward_stop:
            memcpy(
                &forward,
                &forward_view[forward_cursor * sizeof(_forward_result)],
                sizeof(_forward_result),
            )
            if forward.action != FORWARD_DEFINITE_PASS:
                forward_cursor += 1
                continue
            if row >= row_stop:
                raise ValueError("continuation journal omits a Forward pass")
            memcpy(
                &journal_row,
                &row_view[row * sizeof(plan7_continuation_journal_row)],
                sizeof(plan7_continuation_journal_row),
            )
            if (
                journal_row.profile_index != profile
                or journal_row.sequence_index >= sequences._length
                or journal_row.sequence_index != forward.sequence_index
                or journal_row.forward_status != forward.status
                or journal_row.forward_action != forward.action
            ):
                raise ValueError("continuation journal Forward identity differs")
            expected_float.value = forward.fwdsc
            observed_float.value = journal_row.fwdsc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal Forward score differs")
            while postfilter_cursor < postfilter_stop:
                memcpy(
                    &postfilter,
                    &postfilter_view[
                        postfilter_cursor * sizeof(_postfilter_result)
                    ],
                    sizeof(_postfilter_result),
                )
                if postfilter.sequence_index >= journal_row.sequence_index:
                    break
                postfilter_cursor += 1
            if (
                postfilter_cursor == postfilter_stop
                or postfilter.sequence_index != journal_row.sequence_index
                or postfilter.msv_status != journal_row.postfilter_status
                or postfilter.action != journal_row.postfilter_action
                or postfilter.action != BIAS_DEFINITE_PASS
            ):
                raise ValueError("continuation journal post-filter identity differs")
            expected_float.value = postfilter.filtersc
            observed_float.value = journal_row.filtersc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal bias score differs")
            expected_float.value = postfilter.vfsc
            observed_float.value = journal_row.vfsc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal Viterbi score differs")
            optimized_profile = <OptimizedProfile> profiles[profile]
            computed_usc = <float> postfilter.msv_numerator
            computed_usc = computed_usc / optimized_profile._om.scale_b
            computed_usc = computed_usc - <float> 3.0
            expected_float.value = computed_usc
            observed_float.value = journal_row.usc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal MSV score differs")
            if (
                journal_special_offset_view[row]
                != special_offset_view[forward_cursor]
                or journal_special_offset_view[row + 1]
                != special_offset_view[forward_cursor + 1]
            ):
                raise ValueError("continuation journal Forward span differs")

            region_start = <size_t> region_offset_view[row]
            region_stop = <size_t> region_offset_view[row + 1]
            if (
                region_start > region_stop
                or region_stop > journal.region_count
                or journal_row.reserved != 0
                or journal_row.domain_status not in (
                    DOMAIN_OK, DOMAIN_ERANGE, DOMAIN_ENORESULT, DOMAIN_EMPTY
                )
                or journal_row.domain_route not in (
                    DOMAIN_CPU_REQUIRED, DOMAIN_NO_REGIONS, DOMAIN_SIMPLE
                )
                or journal_row.has_own_scales > 1
                or journal_row.compact_route not in (
                    PLAN7_CONTINUATION_COMPACT_NONE,
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED,
                    PLAN7_CONTINUATION_COMPACT_DEVICE,
                )
                or journal_row.reserved2[0] != 0
                or journal_row.reserved2[1] != 0
                or journal_row.reserved2[2] != 0
                or journal_row.reserved3 != 0
            ):
                raise ValueError("continuation journal domain record is invalid")
            if journal_row.domain_route == DOMAIN_CPU_REQUIRED:
                if region_start != region_stop:
                    raise ValueError("CPU domain route carries simple regions")
            else:
                if (
                    journal_row.domain_status != DOMAIN_OK
                    or journal_row.uncertain_count != 0
                    or journal_row.multidomain_count != 0
                    or not isfinite(journal_row.backward_score)
                    or not isfinite(journal_row.nexpected)
                    or journal_row.nexpected < 0.0
                    or journal_row.nexpected
                    > <float> sequences._refs[journal_row.sequence_index].n
                ):
                    raise ValueError("domain route is not continuation-safe")
                if journal_row.domain_route == DOMAIN_NO_REGIONS:
                    if journal_row.region_count != 0 or region_start != region_stop:
                        raise ValueError("no-region route carries intervals")
                elif (
                    region_start == region_stop
                    or journal_row.region_count != region_stop - region_start
                ):
                    raise ValueError("simple domain route has invalid intervals")

            previous_end = 0
            for region in range(region_start, region_stop):
                memcpy(
                    &interval,
                    &region_view[region * sizeof(plan7_simple_region)],
                    sizeof(plan7_simple_region),
                )
                if (
                    interval.begin < 1
                    or interval.begin > interval.end
                    or interval.end
                    > <uint32_t> sequences._refs[journal_row.sequence_index].n
                    or (region != region_start and interval.begin <= previous_end)
                ):
                    raise ValueError("continuation journal intervals are invalid")
                previous_end = interval.end

            compact_start = <size_t> compact_row_offset_view[row]
            compact_stop = <size_t> compact_row_offset_view[row + 1]
            if (
                compact_start > compact_stop
                or compact_stop > journal.compact_result_count
                or journal_row.compact_result_count
                != compact_stop - compact_start
            ):
                raise ValueError(
                    "continuation journal compact row offsets are not monotone"
                )
            if journal_row.domain_route == DOMAIN_SIMPLE:
                simple_row_count += 1
            if journal.compact_result_count == 0:
                if compact_start != 0 or compact_stop != 0:
                    raise ValueError("empty compact output carries row results")
            elif (
                compact_start != region_start
                or compact_stop != region_stop
            ):
                raise ValueError("compact results differ from simple regions")

            expected_compact_route = PLAN7_CONTINUATION_COMPACT_NONE
            if (
                journal.compact_global_fallback
                and journal_row.domain_route == DOMAIN_SIMPLE
                and region_start != region_stop
            ):
                expected_compact_route = (
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                )
            if compact_start != compact_stop:
                compact_row_action = 255
            for compact_index in range(compact_start, compact_stop):
                memcpy(
                    &compact_result,
                    &compact_result_view[
                        compact_index * sizeof(plan7_domain_rescore_result)
                    ],
                    sizeof(plan7_domain_rescore_result),
                )
                memcpy(
                    &interval,
                    &region_view[
                        compact_index * sizeof(plan7_simple_region)
                    ],
                    sizeof(plan7_simple_region),
                )
                trace_start = <size_t> compact_trace_offset_view[compact_index]
                trace_stop = <size_t> compact_trace_offset_view[
                    compact_index + 1
                ]
                if (
                    trace_start > trace_stop
                    or trace_stop > journal.compact_trace_count
                    or compact_result.row_index != row
                    or compact_result.profile_index != profile
                    or compact_result.sequence_index
                    != journal_row.sequence_index
                    or compact_result.envelope_begin != interval.begin
                    or compact_result.envelope_end != interval.end
                    or compact_result.action not in (
                        PLAN7_DOMAIN_RESCORE_CPU_REQUIRED,
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT,
                    )
                    or compact_result.status not in (
                        PLAN7_DOMAIN_RESCORE_OK,
                        PLAN7_DOMAIN_RESCORE_ERANGE,
                        PLAN7_DOMAIN_RESCORE_ENORESULT,
                        PLAN7_DOMAIN_RESCORE_ECAP,
                        PLAN7_DOMAIN_RESCORE_EMPTY,
                    )
                    or compact_result.has_own_scales > 1
                    or compact_result.reserved != 0
                    or compact_result.reserved2 != 0
                ):
                    raise ValueError("continuation compact result is invalid")
                if compact_index == compact_start:
                    compact_row_action = compact_result.action
                elif compact_result.action != compact_row_action:
                    raise ValueError("continuation compact row is not atomic")

                if compact_result.action == (
                    PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                ):
                    if (
                        compact_result.status != PLAN7_DOMAIN_RESCORE_OK
                        or compact_result.has_own_scales
                        or not isfinite(compact_result.forward_score)
                        or not isfinite(compact_result.backward_score)
                        or not isfinite(compact_result.oa_score)
                        or not isfinite(compact_result.domain_correction)
                        or not isfinite(compact_result.score_consistency)
                        or compact_result.score_consistency < 0.0
                        or compact_result.score_consistency > 2.0e-3
                        or compact_result.alignment_begin
                        < compact_result.envelope_begin
                        or compact_result.alignment_begin
                        > compact_result.alignment_end
                        or compact_result.alignment_end
                        > compact_result.envelope_end
                        or compact_result.model_begin < 1
                        or compact_result.model_begin
                        > compact_result.model_end
                        or compact_result.model_end
                        > <uint32_t> optimized_profile._om.M
                        or trace_start == trace_stop
                    ):
                        raise ValueError(
                            "continuation device compact result is invalid"
                        )
                    for residue in range(PLAN7_DOMAIN_RESCORE_NULL2_COUNT):
                        null2_value = compact_null2_view[
                            compact_index
                            * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                            + residue
                        ]
                        if not isfinite(null2_value) or null2_value <= 0.0:
                            raise ValueError(
                                "continuation compact null2 is invalid"
                            )
                    device_result_count += 1
                else:
                    if trace_start != trace_stop:
                        raise ValueError(
                            "CPU compact fallback carries a trace"
                        )
                    for residue in range(PLAN7_DOMAIN_RESCORE_NULL2_COUNT):
                        if not isnan(compact_null2_view[
                            compact_index
                            * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                            + residue
                        ]):
                            raise ValueError(
                                "CPU compact fallback carries null2"
                            )
                    cpu_required_count += 1
            if compact_start != compact_stop:
                if compact_row_action == PLAN7_DOMAIN_RESCORE_DEVICE_RESULT:
                    expected_compact_route = PLAN7_CONTINUATION_COMPACT_DEVICE
                else:
                    expected_compact_route = (
                        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                    )
            if journal_row.compact_route != expected_compact_route:
                raise ValueError("continuation compact row route differs")
            row += 1
            forward_cursor += 1
        if row != row_stop:
            raise ValueError("continuation journal has an extra Forward row")

    if journal.generation_compact_domains and (
        simple_row_count != journal.rescore_simple_row_count
        or (
            journal.compact_result_count != 0
            and (
                device_result_count != journal.rescore_device_result_count
                or cpu_required_count != journal.rescore_cpu_required_count
            )
        )
    ):
        raise ValueError("continuation compact route accounting differs")

    if PyCapsule_SetDestructor(
        capsule, <PyCapsule_Destructor> NULL
    ) != 0:
        raise RuntimeError("cannot consume continuation journal capsule")
    if PyCapsule_SetPointer(capsule, &_consumed_journal_sentinel) != 0:
        storage._data = NULL
        storage._size = 0
        free(journal)
        PyCapsule_SetName(capsule, PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME)
        raise RuntimeError("cannot retire continuation journal capsule")
    storage._owns = True
    if PyCapsule_SetName(
        capsule, PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME
    ) != 0:
        raise RuntimeError("cannot mark continuation journal consumed")
    return (
        storage_view,
        postfilter_view,
        postfilter_offset_view,
        forward_view,
        forward_offset_view,
        special_offset_view,
        special_view,
        profile_offset_view,
        row_view,
        region_offset_view,
        region_view,
        compact_row_offset_view,
        compact_result_view,
        compact_trace_offset_view,
        compact_trace_view,
        compact_null2_view,
        guard_bits,
        journal.generation_tail_fingerprint,
        journal.generation_compact_domains,
        journal.rescore_simple_row_count,
        journal.rescore_device_result_count,
        journal.rescore_cpu_required_count,
        journal.rescore_numeric_fallback_count,
        journal.rescore_cap_fallback_count,
        journal.rescore_global_cpu_fallback_count,
    )


cdef tuple _consume_validate_direct_v3_staging(
    object source_object,
    size_t profile_count,
    size_t target_count,
    uint64_t expected_session_id,
    uint64_t expected_selection_id,
    const uint64_t[::1] expected_identity_tokens,
    const uint8_t[::1] expected_profile_fingerprints,
    uint64_t expected_batch_generation,
    const uint8_t[::1] expected_sequence_fingerprint,
    double expected_f1,
    uint64_t expected_f2_bits,
    uint64_t expected_f3_bits,
    uint64_t expected_tail_fingerprint,
):
    """Validate transient native facts without allocating a dense v2 packet."""
    cdef tuple source
    cdef object postfilter_owner
    cdef object postfilter_offset_owner
    cdef object forward_owner
    cdef object forward_offset_owner
    cdef object special_offset_owner
    cdef object special_owner
    cdef object profile_offset_owner
    cdef object row_owner
    cdef object region_offset_owner
    cdef object region_owner
    cdef object compact_row_offset_owner
    cdef object compact_result_owner
    cdef object compact_trace_offset_owner
    cdef object compact_trace_owner
    cdef object compact_null2_owner
    cdef object decision_owner
    cdef tuple decision_counts
    cdef const uint8_t[::1] postfilter_view
    cdef const uint64_t[::1] postfilter_offset_view
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint64_t[::1] profile_offset_view
    cdef const uint8_t[::1] row_view
    cdef const uint64_t[::1] region_offset_view
    cdef const uint8_t[::1] region_view
    cdef const uint64_t[::1] compact_row_offset_view
    cdef const uint8_t[::1] compact_result_view
    cdef const uint64_t[::1] compact_trace_offset_view
    cdef const uint8_t[::1] compact_trace_view
    cdef const float[::1] compact_null2_view
    cdef const uint8_t[::1] decision_view
    cdef const uint8_t[::1] identity_view
    cdef const uint8_t[::1] profile_fingerprint_view
    cdef const uint8_t[::1] sequence_fingerprint_view
    cdef const uint8_t[::1] forward_provenance_view
    cdef const uint8_t[::1] backward_provenance_view
    cdef const uint8_t[::1] rescore_provenance_view
    cdef plan7_forward_provenance forward_provenance
    cdef plan7_backward_domain_provenance backward_provenance
    cdef plan7_domain_rescore_provenance rescore_provenance
    cdef _double_bits encoded
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t domain_count
    cdef size_t region_count
    cdef size_t compact_result_count
    cdef size_t compact_trace_count
    cdef size_t compact_null2_count
    cdef size_t index

    if type(source_object) is not tuple:
        raise TypeError("direct v3 staging must be exactly tuple")
    source = <tuple> source_object
    if (
        len(source) != 45
        or type(source[0]) is not int
        or source[0] != DIRECT_V3_STAGING_SCHEMA_VERSION
    ):
        raise ValueError("direct v3 staging schema differs")
    for index in range(1, 16):
        if type(source[index]) is not bytes:
            raise TypeError("direct v3 staging payload must be bytes")
    for index in range(16, 31):
        if type(source[index]) not in (int, bool):
            raise TypeError("direct v3 staging metadata must be integral")
    for index in range(31, 34):
        if type(source[index]) is not bytes:
            raise TypeError("direct v3 staging identity must be bytes")
    if type(source[34]) is not int or source[34] < 0:
        raise TypeError("direct v3 staging F1 bits must be nonnegative int")
    if type(source[35]) is not float or type(source[36]) is not float:
        raise TypeError("direct v3 staging thresholds must be float")
    if type(source[37]) is not bool:
        raise TypeError("direct v3 staging bias flag must be bool")
    for index in range(38, 41):
        if type(source[index]) is not bytes:
            raise TypeError("direct v3 staging provenance must be bytes")
    if type(source[41]) is not int or source[41] < 0:
        raise TypeError("direct v3 source validation time must be nonnegative int")
    if type(source[42]) is not bytes:
        raise TypeError("direct v3 decision plan must be immutable bytes")
    if type(source[43]) is not tuple or len(source[43]) != 7:
        raise TypeError("direct v3 decision counts must be a seven-item tuple")
    if type(source[44]) is not int or not 0 <= source[44] <= <uint64_t> -1:
        raise TypeError("direct v3 GA count must be uint64")
    decision_counts = <tuple> source[43]
    for index in range(7):
        if (
            type(decision_counts[index]) is not int
            or decision_counts[index] < 0
            or decision_counts[index] > <uint64_t> -1
        ):
            raise TypeError("direct v3 decision metadata must be uint64")

    postfilter_owner = _immutable_owned_view(source[1], "B")
    postfilter_offset_owner = _immutable_owned_view(
        memoryview(source[2]).cast("Q"), "Q"
    )
    forward_owner = _immutable_owned_view(source[3], "B")
    forward_offset_owner = _immutable_owned_view(
        memoryview(source[4]).cast("Q"), "Q"
    )
    special_offset_owner = _immutable_owned_view(
        memoryview(source[5]).cast("Q"), "Q"
    )
    special_owner = _immutable_owned_view(
        memoryview(source[6]).cast("f"), "f"
    )
    profile_offset_owner = _immutable_owned_view(
        memoryview(source[7]).cast("Q"), "Q"
    )
    row_owner = _immutable_owned_view(source[8], "B")
    region_offset_owner = _immutable_owned_view(
        memoryview(source[9]).cast("Q"), "Q"
    )
    region_owner = _immutable_owned_view(source[10], "B")
    compact_row_offset_owner = _immutable_owned_view(
        memoryview(source[11]).cast("Q"), "Q"
    )
    compact_result_owner = _immutable_owned_view(source[12], "B")
    compact_trace_offset_owner = _immutable_owned_view(
        memoryview(source[13]).cast("Q"), "Q"
    )
    compact_trace_owner = _immutable_owned_view(source[14], "B")
    compact_null2_owner = _immutable_owned_view(
        memoryview(source[15]).cast("f"), "f"
    )
    decision_owner = _immutable_owned_view(source[42], "B")
    postfilter_view = postfilter_owner
    postfilter_offset_view = postfilter_offset_owner
    forward_view = forward_owner
    forward_offset_view = forward_offset_owner
    special_offset_view = special_offset_owner
    special_view = special_owner
    profile_offset_view = profile_offset_owner
    row_view = row_owner
    region_offset_view = region_offset_owner
    region_view = region_owner
    compact_row_offset_view = compact_row_offset_owner
    compact_result_view = compact_result_owner
    compact_trace_offset_view = compact_trace_offset_owner
    compact_trace_view = compact_trace_owner
    compact_null2_view = compact_null2_owner
    decision_view = decision_owner

    if (
        postfilter_view.shape[0] % sizeof(_postfilter_result)
        or forward_view.shape[0] % sizeof(_forward_result)
        or row_view.shape[0] % sizeof(plan7_continuation_journal_row)
        or region_view.shape[0] % sizeof(plan7_simple_region)
        or compact_result_view.shape[0]
        % sizeof(plan7_domain_rescore_result)
        or compact_trace_view.shape[0]
        % sizeof(plan7_domain_rescore_trace_step)
    ):
        raise ValueError("direct v3 staging record storage has trailing bytes")
    postfilter_count = postfilter_view.shape[0] // sizeof(_postfilter_result)
    forward_count = forward_view.shape[0] // sizeof(_forward_result)
    domain_count = row_view.shape[0] // sizeof(plan7_continuation_journal_row)
    region_count = region_view.shape[0] // sizeof(plan7_simple_region)
    compact_result_count = (
        compact_result_view.shape[0] // sizeof(plan7_domain_rescore_result)
    )
    compact_trace_count = (
        compact_trace_view.shape[0]
        // sizeof(plan7_domain_rescore_trace_step)
    )
    compact_null2_count = compact_null2_view.shape[0]
    if (
        decision_view.shape[0] != postfilter_count
        or decision_counts[0] > postfilter_count
        or decision_counts[1] > special_view.shape[0]
        or decision_counts[2] > region_count
        or decision_counts[3] > compact_result_count
        or decision_counts[4] > compact_trace_count
        or decision_counts[5] > compact_null2_count
        or decision_counts[6] == 0
        or
        postfilter_offset_view.shape[0] != profile_count + 1
        or forward_offset_view.shape[0] != profile_count + 1
        or special_offset_view.shape[0] != forward_count + 1
        or profile_offset_view.shape[0] != profile_count + 1
        or region_offset_view.shape[0] != domain_count + 1
        or compact_row_offset_view.shape[0] != domain_count + 1
        or compact_trace_offset_view.shape[0] != compact_result_count + 1
        or postfilter_offset_view[0] != 0
        or postfilter_offset_view[profile_count] != postfilter_count
        or forward_offset_view[0] != 0
        or forward_offset_view[profile_count] != forward_count
        or special_offset_view[0] != 0
        or special_offset_view[forward_count] != special_view.shape[0]
        or profile_offset_view[0] != 0
        or profile_offset_view[profile_count] != domain_count
        or region_offset_view[0] != 0
        or region_offset_view[domain_count] != region_count
        or compact_row_offset_view[0] != 0
        or compact_row_offset_view[domain_count] != compact_result_count
        or compact_trace_offset_view[0] != 0
        or compact_trace_offset_view[compact_result_count]
        != compact_trace_count
        or compact_result_count > (<size_t> -1)
        // PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or compact_null2_count
        != compact_result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
    ):
        raise ValueError("direct v3 staging offsets or counts differ")
    for index in range(profile_count):
        if (
            postfilter_offset_view[index] > postfilter_offset_view[index + 1]
            or forward_offset_view[index] > forward_offset_view[index + 1]
            or profile_offset_view[index] > profile_offset_view[index + 1]
        ):
            raise ValueError("direct v3 profile offsets are not monotone")
    for index in range(forward_count):
        if special_offset_view[index] > special_offset_view[index + 1]:
            raise ValueError("direct v3 special offsets are not monotone")
    for index in range(domain_count):
        if (
            region_offset_view[index] > region_offset_view[index + 1]
            or compact_row_offset_view[index]
            > compact_row_offset_view[index + 1]
        ):
            raise ValueError("direct v3 domain offsets are not monotone")
    for index in range(compact_result_count):
        if (
            compact_trace_offset_view[index]
            > compact_trace_offset_view[index + 1]
        ):
            raise ValueError("direct v3 trace offsets are not monotone")

    identity_view = source[31]
    profile_fingerprint_view = source[32]
    sequence_fingerprint_view = source[33]
    if (
        source[28] != expected_session_id
        or source[29] != expected_selection_id
        or source[30] != expected_batch_generation
        or identity_view.shape[0] != profile_count * sizeof(uint64_t)
        or (
            identity_view.shape[0] != 0
            and memcmp(
                &identity_view[0],
                &expected_identity_tokens[0],
                identity_view.shape[0],
            ) != 0
        )
        or profile_fingerprint_view.shape[0]
        != expected_profile_fingerprints.shape[0]
        or (
            profile_fingerprint_view.shape[0] != 0
            and memcmp(
                &profile_fingerprint_view[0],
                &expected_profile_fingerprints[0],
                profile_fingerprint_view.shape[0],
            ) != 0
        )
        or sequence_fingerprint_view.shape[0]
        != expected_sequence_fingerprint.shape[0]
        or memcmp(
            &sequence_fingerprint_view[0],
            &expected_sequence_fingerprint[0],
            sequence_fingerprint_view.shape[0],
        ) != 0
    ):
        raise ValueError("direct v3 staging identity differs")
    encoded.value = expected_f1
    if source[34] != encoded.bits or not source[37]:
        raise ValueError("direct v3 F1 or bias provenance differs")
    encoded.value = source[35]
    if encoded.bits != expected_f2_bits:
        raise ValueError("direct v3 F2 provenance differs")
    encoded.value = source[36]
    if encoded.bits != expected_f3_bits:
        raise ValueError("direct v3 F3 provenance differs")
    if (
        source[17] != expected_tail_fingerprint
        or bool(source[18]) != bool(expected_tail_fingerprint)
        or source[25] <= 0
        or source[26] < 0
        or source[27] < 0
    ):
        raise ValueError("direct v3 staging generation metadata differs")

    forward_provenance_view = source[38]
    backward_provenance_view = source[39]
    rescore_provenance_view = source[40]
    if (
        forward_provenance_view.shape[0] != sizeof(plan7_forward_provenance)
        or backward_provenance_view.shape[0]
        != sizeof(plan7_backward_domain_provenance)
        or rescore_provenance_view.shape[0]
        != sizeof(plan7_domain_rescore_provenance)
    ):
        raise ValueError("direct v3 staging provenance size differs")
    memcpy(
        &forward_provenance,
        &forward_provenance_view[0],
        sizeof(plan7_forward_provenance),
    )
    memcpy(
        &backward_provenance,
        &backward_provenance_view[0],
        sizeof(plan7_backward_domain_provenance),
    )
    memcpy(
        &rescore_provenance,
        &rescore_provenance_view[0],
        sizeof(plan7_domain_rescore_provenance),
    )
    if (
        forward_provenance.batch_generation != expected_batch_generation
        or forward_provenance.pass_count != domain_count
        or forward_provenance.special_count != special_view.shape[0]
        or memcmp(
            &forward_provenance,
            &backward_provenance.forward,
            sizeof(plan7_forward_provenance),
        ) != 0
        or backward_provenance.candidate_count != domain_count
        or backward_provenance.region_count != region_count
    ):
        raise ValueError("direct v3 Forward/domain provenance differs")
    if source[18]:
        if (
            memcmp(
                &rescore_provenance.backward,
                &backward_provenance,
                sizeof(plan7_backward_domain_provenance),
            ) != 0
            or rescore_provenance.result_count != compact_result_count
            or rescore_provenance.trace_count != compact_trace_count
            or rescore_provenance.null2_count != compact_null2_count
            or source[20] + source[21] + source[44] != region_count
        ):
            raise ValueError("direct v3 rescore provenance differs")
    elif (
        compact_result_count != 0
        or compact_trace_count != 0
        or compact_null2_count != 0
        or rescore_provenance.result_count != 0
        or rescore_provenance.trace_count != 0
        or rescore_provenance.null2_count != 0
    ):
        raise ValueError("disabled direct v3 rescore carries output")

    return (
        b"",
        postfilter_owner,
        postfilter_offset_owner,
        forward_owner,
        forward_offset_owner,
        special_offset_owner,
        special_owner,
        profile_offset_owner,
        row_owner,
        region_offset_owner,
        region_owner,
        compact_row_offset_owner,
        compact_result_owner,
        compact_trace_offset_owner,
        compact_trace_owner,
        compact_null2_owner,
        source[16],
        source[17],
        source[18],
        source[19],
        source[20],
        source[21],
        source[22],
        source[23],
        source[24],
        decision_owner,
        decision_counts,
        source[44],
    )


def _seal_postfilter_batch_bound(
    queries,
    optimized_profiles,
    DigitalSequenceBlock sequences,
    postfilter_records,
    postfilter_offsets,
    residue_offsets,
    double f1,
    background_fingerprint,
    forward_records=None,
    forward_offsets=None,
    special_offsets=None,
    specials=None,
    expected_forward_indices=None,
    uint64_t generation_f2_bits=0,
    uint64_t generation_f3_bits=0,
    generation_bias_filter=False,
    continuation_journal=None,
    selection_identity=None,
    selection_identity_tokens=None,
    domain_pipeline=None,
    _residue_offsets_shared=False,
    _background_fingerprint_shared=False,
):
    """Seal one already-owned post-filter batch after one complete validation.

    This private factory's HMM, profile, and target arguments must come from
    the adapter's hidden registries while its pair and sequence locks are
    held. It defensively freezes every caller buffer before validating it.
    The returned object exposes neither those buffers nor native pointers;
    the provenance adapter keeps the matching pair and pipeline locks around
    every search.
    """
    cdef tuple query_tuple = tuple(queries)
    cdef tuple profile_tuple = tuple(optimized_profiles)
    cdef object postfilter_owner = _immutable_owned_view(
        postfilter_records, "B"
    )
    cdef object postfilter_offset_owner = _immutable_owned_view(
        postfilter_offsets, "Q"
    )
    cdef tuple residue_owner_record = _immutable_owned_view_with_copy(
        residue_offsets, "Q"
    )
    cdef tuple background_owner_record = _immutable_owned_view_with_copy(
        background_fingerprint, "B"
    )
    cdef object residue_offset_owner = residue_owner_record[0]
    cdef object background_owner = background_owner_record[0]
    cdef bint residue_offsets_copied = residue_owner_record[1]
    cdef bint background_fingerprint_copied = background_owner_record[1]
    cdef const uint8_t[::1] postfilter_view = postfilter_owner
    cdef const uint64_t[::1] postfilter_offset_view = postfilter_offset_owner
    cdef const uint64_t[::1] residue_offset_view = residue_offset_owner
    cdef const uint8_t[::1] background_view = background_owner
    cdef size_t profile_count
    cdef size_t profile_index
    cdef size_t postfilter_count
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_count
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t forward_cursor
    cdef _forward_result forward_record
    cdef bint has_direct = False
    cdef bint has_forward_storage
    cdef bint row_has_external
    cdef bytearray external_rows
    cdef object empty_forward_records
    cdef object empty_forward_offsets
    cdef object empty_special_offsets
    cdef object empty_specials
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint32_t[::1] expected_forward_view
    cdef _SealedPostfilterBatch sealed
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL
    cdef _pipeline_from_filter_and_forward_scores_f forward_scores_seam = NULL
    cdef _pipeline_from_filter_forward_backward_scores_f forward_backward_scores_seam = NULL
    cdef _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam = NULL
    cdef _pipeline_tail_snapshot pipeline_options
    cdef object journal_values = None
    cdef object empty_journal_storage = b""
    cdef object empty_journal_profile_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_rows = b""
    cdef object empty_journal_region_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_regions = b""
    cdef object empty_journal_compact_row_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_compact_results = b""
    cdef object empty_journal_compact_trace_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_compact_traces = b""
    cdef object empty_journal_compact_null2 = b""
    cdef const uint8_t[::1] journal_storage_view = empty_journal_storage
    cdef const uint64_t[::1] journal_profile_offset_view = memoryview(
        empty_journal_profile_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_row_view = empty_journal_rows
    cdef const uint64_t[::1] journal_region_offset_view = memoryview(
        empty_journal_region_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_region_view = empty_journal_regions
    cdef const uint64_t[::1] journal_compact_row_offset_view = memoryview(
        empty_journal_compact_row_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_compact_result_view = (
        empty_journal_compact_results
    )
    cdef const uint64_t[::1] journal_compact_trace_offset_view = memoryview(
        empty_journal_compact_trace_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_compact_trace_view = (
        empty_journal_compact_traces
    )
    cdef const float[::1] journal_compact_null2_view = memoryview(
        empty_journal_compact_null2
    ).cast("f")
    cdef uint32_t journal_guard_bits = 0
    cdef uint64_t expected_session_id = 0
    cdef uint64_t expected_selection_id = 0
    cdef size_t frequency_bytes
    cdef _double_bits generation_f1
    cdef _float_bits expected_rt
    cdef Pipeline journal_pipeline
    cdef object expected_identity_owner
    cdef const uint64_t[::1] expected_identity_view

    if type(_residue_offsets_shared) is not bool:
        raise TypeError("residue-offset ownership provenance must be bool")
    if type(_background_fingerprint_shared) is not bool:
        raise TypeError("background ownership provenance must be bool")

    if postfilter_offset_view.shape[0] == 0:
        raise ValueError("post-filter row offsets need an initial zero")
    profile_count = <size_t> postfilter_offset_view.shape[0] - 1
    if len(query_tuple) != profile_count:
        raise ValueError("HMM count differs from post-filter row count")
    if len(profile_tuple) != profile_count:
        raise ValueError(
            "optimized-profile count differs from post-filter row count"
        )
    if postfilter_offset_view[0] != 0:
        raise ValueError("post-filter row offsets must start at zero")
    if <size_t> postfilter_view.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result storage has trailing bytes")
    postfilter_count = (
        <size_t> postfilter_view.shape[0] // sizeof(_postfilter_result)
    )
    if postfilter_offset_view[profile_count] != postfilter_count:
        raise ValueError("post-filter row offsets do not span result storage")
    if not isfinite(f1) or f1 < 0.0 or f1 > 1.0:
        raise ValueError("sealed F1 must be a finite number in [0, 1]")
    if background_view.shape[0] != (
        (<size_t> sequences.alphabet.K + 1) * sizeof(float)
    ):
        raise ValueError("canonical background fingerprint has the wrong size")
    if type(generation_bias_filter) is not bool:
        raise TypeError("generation bias filter must be bool")

    _validate_sealed_residue_offsets(sequences, residue_offset_view)

    has_forward_storage = any(
        value is not None
        for value in (forward_records, forward_offsets, special_offsets, specials)
    )
    if has_forward_storage and any(
        value is None
        for value in (forward_records, forward_offsets, special_offsets, specials)
    ):
        raise ValueError("Forward storage must be supplied as one complete batch")
    if has_forward_storage:
        forward_view = _immutable_owned_view(forward_records, "B")
        forward_offset_view = _immutable_owned_view(forward_offsets, "Q")
        special_offset_view = _immutable_owned_view(special_offsets, "Q")
        special_view = _immutable_owned_view(specials, "f")
    else:
        empty_forward_records = b""
        empty_forward_offsets = bytes((profile_count + 1) * sizeof(uint64_t))
        empty_special_offsets = bytes(sizeof(uint64_t))
        empty_specials = b""
        forward_view = empty_forward_records
        forward_offset_view = memoryview(empty_forward_offsets).cast("Q")
        special_offset_view = memoryview(empty_special_offsets).cast("Q")
        special_view = memoryview(empty_specials).cast("f")

    if expected_forward_indices is not None:
        if not has_forward_storage:
            raise ValueError(
                "expected Forward indexes require complete Forward storage"
            )
        expected_forward_view = expected_forward_indices

    if forward_offset_view.shape[0] != profile_count + 1:
        raise ValueError("Forward row-offset count differs from post-filter rows")
    if forward_offset_view[0] != 0:
        raise ValueError("Forward row offsets must start at zero")
    if <size_t> forward_view.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("Forward result storage has trailing bytes")
    forward_count = <size_t> forward_view.shape[0] // sizeof(_forward_result)
    if forward_offset_view[profile_count] != forward_count:
        raise ValueError("Forward row offsets do not span result storage")
    if special_offset_view.shape[0] != forward_count + 1:
        raise ValueError("Forward special-offset count differs from result count")
    if special_offset_view[0] != 0:
        raise ValueError("Forward global special offsets must start at zero")
    if special_offset_view[forward_count] != <uint64_t> special_view.shape[0]:
        raise ValueError("Forward special offsets do not span matrix storage")
    if expected_forward_indices is not None:
        if expected_forward_view.shape[0] != forward_count:
            raise ValueError("Forward result count differs from selected inputs")
        for forward_cursor in range(forward_count):
            memcpy(
                &forward_record,
                &forward_view[forward_cursor * sizeof(_forward_result)],
                sizeof(_forward_result),
            )
            if forward_record.sequence_index != expected_forward_view[forward_cursor]:
                raise ValueError("Forward result order differs from selected inputs")

    external_rows = bytearray(profile_count)
    for profile_index in range(profile_count):
        if type(query_tuple[profile_index]) is not _pyhmmer.plan7.HMM:
            raise TypeError("sealed queries must be exactly astra_pyhmmer.plan7.HMM")
        if (
            type(profile_tuple[profile_index])
            is not _pyhmmer.plan7.OptimizedProfile
        ):
            raise TypeError(
                "sealed profiles must be exactly astra_pyhmmer.plan7.OptimizedProfile"
            )
        _validate_sealed_pair(
            <HMM> query_tuple[profile_index],
            <OptimizedProfile> profile_tuple[profile_index],
            sequences,
        )

        postfilter_start = <size_t> postfilter_offset_view[profile_index]
        postfilter_stop = <size_t> postfilter_offset_view[profile_index + 1]
        if (
            postfilter_start > postfilter_stop
            or postfilter_stop > postfilter_count
        ):
            raise ValueError("post-filter row offsets are not monotone")
        if _validate_postfilter_records(
            <OptimizedProfile> profile_tuple[profile_index],
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
        ):
            has_direct = True

        forward_start = <size_t> forward_offset_view[profile_index]
        forward_stop = <size_t> forward_offset_view[profile_index + 1]
        if forward_start > forward_stop or forward_stop > forward_count:
            raise ValueError("Forward row offsets are not monotone")
        row_has_external = _validate_forward_augmentation(
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            forward_view[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            special_offset_view[forward_start:forward_stop + 1],
            special_view,
        )
        external_rows[profile_index] = <uint8_t> row_has_external

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct post-filter records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )
    if has_forward_storage:
        forward_scores_seam = _cached_filter_and_forward_scores_seam()
        forward_backward_scores_seam = (
            _cached_filter_forward_backward_scores_seam()
        )

    if (
        continuation_journal is not None
        or selection_identity is not None
        or selection_identity_tokens is not None
        or domain_pipeline is not None
    ):
        raise TypeError(
            "continuation journals require the fused profile-selection factory"
        )
    sealed = _SealedPostfilterBatch()
    sealed._queries = query_tuple
    sealed._optimized_profiles = profile_tuple
    sealed._sequences = sequences
    sealed._postfilter_records = postfilter_view
    sealed._postfilter_offsets = postfilter_offset_view
    sealed._residue_offsets = residue_offset_view
    sealed._forward_records = forward_view
    sealed._forward_offsets = forward_offset_view
    sealed._special_offsets = special_offset_view
    sealed._specials = special_view
    sealed._row_has_external = bytes(external_rows)
    sealed._background_fingerprint = background_view
    sealed._journal_storage = journal_storage_view
    sealed._journal_profile_offsets = journal_profile_offset_view
    sealed._journal_rows = journal_row_view
    sealed._journal_region_offsets = journal_region_offset_view
    sealed._journal_regions = journal_region_view
    sealed._journal_compact_row_offsets = journal_compact_row_offset_view
    sealed._journal_compact_results = journal_compact_result_view
    sealed._journal_compact_trace_offsets = (
        journal_compact_trace_offset_view
    )
    sealed._journal_compact_traces = journal_compact_trace_view
    sealed._journal_compact_null2 = journal_compact_null2_view
    sealed._f1 = f1
    sealed._generation_f2_bits = generation_f2_bits
    sealed._generation_f3_bits = generation_f3_bits
    sealed._generation_bias_filter = generation_bias_filter
    sealed._journal_guard_bits = journal_guard_bits
    sealed._generation_tail_fingerprint = 0
    sealed._rescore_simple_row_count = 0
    sealed._rescore_device_result_count = 0
    sealed._rescore_cpu_required_count = 0
    sealed._rescore_numeric_fallback_count = 0
    sealed._rescore_cap_fallback_count = 0
    sealed._rescore_global_cpu_fallback_count = 0
    if residue_offsets_copied or not _residue_offsets_shared:
        sealed._owned_residue_offsets_bytes = (
            int(residue_offset_view.shape[0]) * sizeof(uint64_t)
        )
    else:
        sealed._excluded_residue_offsets_bytes = (
            int(residue_offset_view.shape[0]) * sizeof(uint64_t)
        )
    if background_fingerprint_copied or not _background_fingerprint_shared:
        sealed._owned_background_fingerprint_bytes = int(
            background_view.shape[0]
        )
    else:
        sealed._excluded_background_fingerprint_bytes = int(
            background_view.shape[0]
        )
    if continuation_journal is not None:
        sealed._pipeline_options = pipeline_options
    sealed._filter_scores_seam = filter_scores_seam
    sealed._forward_scores_seam = forward_scores_seam
    sealed._forward_backward_scores_seam = forward_backward_scores_seam
    sealed._simple_regions_seam = simple_regions_seam
    sealed._compact_tail_fingerprint = NULL
    sealed._compact_domains_seam = NULL
    sealed._ready = True
    return sealed


def _seal_profile_selection_continuation_bound(
    queries,
    optimized_profiles,
    DigitalSequenceBlock sequences,
    residue_offsets,
    double f1,
    background_fingerprint,
    continuation_journal,
    selection_identity,
    selection_identity_tokens,
    profile_fingerprints,
    uint64_t batch_generation,
    sequence_content_fingerprint,
    Pipeline pipeline,
    double guard_band,
    native_stage_timings=None,
    generation_statistics=None,
    bint sparse_journal_v3=False,
):
    """Consume one package-internal journal into an opaque continuation batch.

    The capsule and underscore factory are trusted in-process transport. Their
    integrity checks reject accidental corruption and cross-binding, not a
    caller deliberately reaching private extension APIs or ctypes.
    """
    cdef tuple query_tuple = tuple(queries)
    cdef tuple profile_tuple = tuple(optimized_profiles)
    cdef size_t profile_count = len(profile_tuple)
    cdef tuple residue_owner_record = _immutable_owned_view_with_copy(
        residue_offsets, "Q"
    )
    cdef tuple background_owner_record = _immutable_owned_view_with_copy(
        background_fingerprint, "B"
    )
    cdef object residue_offset_owner = residue_owner_record[0]
    cdef object background_owner = background_owner_record[0]
    cdef bint residue_offsets_copied = residue_owner_record[1]
    cdef bint background_fingerprint_copied = background_owner_record[1]
    cdef object identity_owner = _immutable_owned_view(
        selection_identity_tokens, "Q"
    )
    cdef object profile_fingerprint_owner = _immutable_owned_view(
        profile_fingerprints, "B"
    )
    cdef object sequence_fingerprint_owner = _immutable_owned_view(
        sequence_content_fingerprint, "B"
    )
    cdef const uint64_t[::1] residue_offset_view = residue_offset_owner
    cdef const uint8_t[::1] background_view = background_owner
    cdef const uint64_t[::1] identity_view = identity_owner
    cdef const uint8_t[::1] profile_fingerprint_view = (
        profile_fingerprint_owner
    )
    cdef const uint8_t[::1] sequence_fingerprint_view = (
        sequence_fingerprint_owner
    )
    cdef object journal_values
    cdef tuple direct_source
    cdef bint direct_v3_source = False
    cdef const uint8_t[::1] journal_storage_view
    cdef const uint8_t[::1] postfilter_view
    cdef const uint64_t[::1] postfilter_offset_view
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint64_t[::1] journal_profile_offset_view
    cdef const uint8_t[::1] journal_row_view
    cdef const uint64_t[::1] journal_region_offset_view
    cdef const uint8_t[::1] journal_region_view
    cdef const uint64_t[::1] journal_compact_row_offset_view
    cdef const uint8_t[::1] journal_compact_result_view
    cdef const uint64_t[::1] journal_compact_trace_offset_view
    cdef const uint8_t[::1] journal_compact_trace_view
    cdef const float[::1] journal_compact_null2_view
    cdef _pipeline_tail_snapshot pipeline_options
    cdef _SealedPostfilterBatch sealed
    cdef _pipeline_from_filter_scores_f filter_scores_seam
    cdef _pipeline_from_filter_and_forward_scores_f forward_scores_seam
    cdef _pipeline_from_filter_forward_backward_scores_f forward_backward_scores_seam
    cdef _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam
    cdef _pipeline_from_filter_forward_compact_domains_f compact_domains_seam
    cdef _pipeline_compact_tail_fingerprint_f compact_tail_fingerprint
    cdef size_t profile
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t journal_start
    cdef size_t journal_stop
    cdef bint has_direct = False
    cdef bint row_has_external
    cdef bytearray external_rows = bytearray(profile_count)
    cdef uint64_t expected_session_id
    cdef uint64_t expected_selection_id
    cdef uint32_t journal_guard_bits
    cdef _float_bits expected_guard
    cdef size_t frequency_bytes
    cdef uint64_t expected_tail_fingerprint = 0
    cdef uint64_t generation_tail_fingerprint = 0
    cdef bint generation_compact_domains = False
    cdef uint64_t rescore_simple_row_count = 0
    cdef uint64_t rescore_device_result_count = 0
    cdef uint64_t rescore_cpu_required_count = 0
    cdef uint64_t rescore_certified_ga_count = 0
    cdef uint64_t rescore_numeric_fallback_count = 0
    cdef uint64_t rescore_cap_fallback_count = 0
    cdef uint64_t rescore_global_cpu_fallback_count = 0
    cdef object validated_stage_timings = None
    cdef object profile_statistics
    cdef object planning_start_ns
    cdef object validation_start_ns
    cdef object source_validation_start_ns
    cdef object source_validation_elapsed_ns
    cdef plan7_continuation_journal_v3 *journal_v3 = NULL
    cdef const uint8_t[::1] direct_forward_provenance_view
    cdef const uint8_t[::1] direct_backward_provenance_view
    cdef const uint8_t[::1] direct_rescore_provenance_view
    cdef const uint8_t[::1] direct_decision_view
    cdef tuple direct_decision_counts

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    if len(query_tuple) != profile_count:
        raise ValueError("HMM count differs from optimized-profile count")
    if (
        type(selection_identity) is not tuple
        or len(selection_identity) != 2
        or type(selection_identity[0]) is not int
        or type(selection_identity[1]) is not int
    ):
        raise TypeError("continuation selection identity is invalid")
    expected_session_id = selection_identity[0]
    expected_selection_id = selection_identity[1]
    if expected_session_id == 0 or expected_selection_id == 0:
        raise ValueError("continuation selection identity is zero")
    if batch_generation == 0:
        raise ValueError("continuation target generation is zero")
    if sequence_fingerprint_view.shape[0] != 32:
        raise ValueError("sequence content fingerprint has the wrong size")
    if profile_fingerprint_view.shape[0] != (
        profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
    ):
        raise ValueError("optimized-profile fingerprint storage has the wrong size")
    if background_view.shape[0] != (
        (<size_t> sequences.alphabet.K + 1) * sizeof(float)
    ):
        raise ValueError("canonical background fingerprint has the wrong size")
    _validate_sealed_residue_offsets(sequences, residue_offset_view)
    _validate_simple_region_generation_bound(
        pipeline,
        f1,
        pipeline._pli.F2,
        pipeline._pli.F3,
        True,
        guard_band,
    )
    _capture_pipeline_tail_options(pipeline, &pipeline_options)
    compact_domains_seam = _cached_compact_domains_seam()
    compact_tail_fingerprint = _compact_tail_fingerprint_cache
    if compact_domains_seam != NULL and compact_tail_fingerprint != NULL:
        with nogil:
            expected_tail_fingerprint = compact_tail_fingerprint(
                pipeline._pli
            )
    if _os.environ.get("PLAN7_GPU_DOMAIN_OWNERSHIP") == "cpu_rescore":
        expected_tail_fingerprint = 0
    frequency_bytes = <size_t> pipeline.background._bg.abc.K * sizeof(float)
    if (
        background_view.shape[0] != frequency_bytes + sizeof(float)
        or memcmp(
            &background_view[0],
            pipeline.background._bg.f,
            frequency_bytes,
        ) != 0
        or memcmp(
            &background_view[frequency_bytes],
            &pipeline.background._bg.omega,
            sizeof(float),
        ) != 0
    ):
        raise ValueError(
            "pipeline background differs from journal generation"
        )

    # Reject every live PyHMMER object before consuming the one-shot capsule.
    # The journal validator may safely dereference only after this completes.
    for profile in range(profile_count):
        if type(query_tuple[profile]) is not _pyhmmer.plan7.HMM:
            raise TypeError("sealed queries must be exactly astra_pyhmmer.plan7.HMM")
        if (
            type(profile_tuple[profile])
            is not _pyhmmer.plan7.OptimizedProfile
        ):
            raise TypeError(
                "sealed profiles must be exactly astra_pyhmmer.plan7.OptimizedProfile"
            )
        if (
            (<HMM> query_tuple[profile])._hmm == NULL
            or (<OptimizedProfile> profile_tuple[profile])._om == NULL
        ):
            raise ValueError("sealed HMM/profile storage is unavailable")
        if (
            not profile_tuple[profile].local
            or not profile_tuple[profile].multihit
        ):
            raise ValueError(
                "sealed optimized profiles must be local multihit"
            )
        _validate_sealed_pair(
            <HMM> query_tuple[profile],
            <OptimizedProfile> profile_tuple[profile],
            sequences,
        )

    direct_v3_source = (
        sparse_journal_v3
        and type(continuation_journal) is tuple
        and len(continuation_journal) != 0
        and continuation_journal[0] == DIRECT_V3_STAGING_SCHEMA_VERSION
    )
    source_validation_start_ns = _time.perf_counter_ns()
    if direct_v3_source:
        direct_source = <tuple> continuation_journal
        journal_values = _consume_validate_direct_v3_staging(
            direct_source,
            profile_count,
            len(sequences),
            expected_session_id,
            expected_selection_id,
            identity_view,
            profile_fingerprint_view,
            batch_generation,
            sequence_fingerprint_view,
            f1,
            pipeline_options.f2_bits,
            pipeline_options.f3_bits,
            expected_tail_fingerprint,
        )
    else:
        journal_values = _consume_validate_continuation_journal(
            continuation_journal,
            profile_tuple,
            sequences,
            expected_session_id,
            expected_selection_id,
            identity_view,
            profile_fingerprint_view,
            batch_generation,
            sequence_fingerprint_view,
            f1,
            pipeline_options.f2_bits,
            pipeline_options.f3_bits,
            True,
            expected_tail_fingerprint,
        )
    source_validation_elapsed_ns = (
        _time.perf_counter_ns() - source_validation_start_ns
    )
    journal_storage_view = journal_values[0]
    postfilter_view = journal_values[1]
    postfilter_offset_view = journal_values[2]
    forward_view = journal_values[3]
    forward_offset_view = journal_values[4]
    special_offset_view = journal_values[5]
    special_view = journal_values[6]
    journal_profile_offset_view = journal_values[7]
    journal_row_view = journal_values[8]
    journal_region_offset_view = journal_values[9]
    journal_region_view = journal_values[10]
    journal_compact_row_offset_view = journal_values[11]
    journal_compact_result_view = journal_values[12]
    journal_compact_trace_offset_view = journal_values[13]
    journal_compact_trace_view = journal_values[14]
    journal_compact_null2_view = journal_values[15]
    journal_guard_bits = journal_values[16]
    generation_tail_fingerprint = journal_values[17]
    generation_compact_domains = journal_values[18]
    rescore_simple_row_count = journal_values[19]
    rescore_device_result_count = journal_values[20]
    rescore_cpu_required_count = journal_values[21]
    rescore_numeric_fallback_count = journal_values[22]
    rescore_cap_fallback_count = journal_values[23]
    rescore_global_cpu_fallback_count = journal_values[24]
    if direct_v3_source:
        rescore_certified_ga_count = journal_values[27]
    if native_stage_timings is not None:
        validated_stage_timings = _validate_native_stage_timings(
            native_stage_timings,
            generation_compact_domains,
        )
    if generation_statistics is not None:
        generation_statistics = (
            _telemetry_module.validate_generation_statistics(
                generation_statistics
            )
        )
        if generation_statistics["journal"]["allocation_bytes"] != (
            direct_source[25]
            if direct_v3_source
            else journal_storage_view.shape[0]
        ):
            raise ValueError(
                "generation telemetry journal allocation differs"
            )
        if (
            generation_statistics["profile_count"] != profile_count
            or generation_statistics["target_count"] != len(sequences)
            or generation_statistics["total_target_residues"]
            != residue_offset_view[len(sequences)]
        ):
            raise ValueError(
                "generation telemetry profile or target identity differs"
            )
        for profile in range(profile_count):
            profile_statistics = generation_statistics["profiles"][profile]
            postfilter_start = <size_t> postfilter_offset_view[profile]
            postfilter_stop = <size_t> postfilter_offset_view[profile + 1]
            journal_start = <size_t> journal_profile_offset_view[profile]
            journal_stop = <size_t> journal_profile_offset_view[profile + 1]
            if (
                profile_statistics["model_length"]
                != (<OptimizedProfile> profile_tuple[profile])._om.M
                or profile_statistics["counts"]["f1_candidate_count"]
                != postfilter_stop - postfilter_start
                or profile_statistics["counts"]["forward_pass_count"]
                != journal_stop - journal_start
                or profile_statistics["journal"]["row_count"]
                != journal_stop - journal_start
                or profile_statistics["journal"]["region_count"]
                != (
                    journal_region_offset_view[journal_stop]
                    - journal_region_offset_view[journal_start]
                )
            ):
                raise ValueError(
                    "generation telemetry profile attribution differs"
                )
        generation_statistics = (
            _telemetry_module.bind_generation_statistics_identity(
                generation_statistics,
                (
                    int(expected_session_id),
                    int(expected_selection_id),
                    int(batch_generation),
                ),
            )
        )
    expected_guard.value = <float> guard_band
    if journal_guard_bits != expected_guard.bits:
        raise ValueError("continuation journal guard band differs")

    postfilter_count = (
        <size_t> postfilter_view.shape[0] // sizeof(_postfilter_result)
    )
    forward_count = <size_t> forward_view.shape[0] // sizeof(_forward_result)
    for profile in range(profile_count):
        postfilter_start = <size_t> postfilter_offset_view[profile]
        postfilter_stop = <size_t> postfilter_offset_view[profile + 1]
        forward_start = <size_t> forward_offset_view[profile]
        forward_stop = <size_t> forward_offset_view[profile + 1]
        if (
            postfilter_start > postfilter_stop
            or postfilter_stop > postfilter_count
            or forward_start > forward_stop
            or forward_stop > forward_count
        ):
            raise ValueError("continuation row offsets are not monotone")
        if _validate_postfilter_records(
            <OptimizedProfile> profile_tuple[profile],
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
        ):
            has_direct = True
        row_has_external = _validate_forward_augmentation(
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            forward_view[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            special_offset_view[forward_start:forward_stop + 1],
            special_view,
        )
        external_rows[profile] = <uint8_t> row_has_external

    filter_scores_seam = _cached_filter_scores_seam()
    forward_scores_seam = _cached_filter_and_forward_scores_seam()
    forward_backward_scores_seam = (
        _cached_filter_forward_backward_scores_seam()
    )
    simple_regions_seam = _cached_simple_regions_seam()
    if has_direct and filter_scores_seam == NULL:
        raise RuntimeError(
            "continuation batches require the private filter-score seam"
        )
    if forward_count and forward_scores_seam == NULL:
        raise RuntimeError(
            "continuation batches require the private Forward seam"
        )
    if journal_row_view.shape[0] and simple_regions_seam == NULL:
        raise RuntimeError(
            "continuation batches require the private simple-region seam"
        )

    sealed = _SealedPostfilterBatch()
    sealed._queries = query_tuple
    sealed._optimized_profiles = profile_tuple
    sealed._sequences = sequences
    sealed._postfilter_records = postfilter_view
    sealed._postfilter_offsets = postfilter_offset_view
    sealed._residue_offsets = residue_offset_view
    sealed._forward_records = forward_view
    sealed._forward_offsets = forward_offset_view
    sealed._special_offsets = special_offset_view
    sealed._specials = special_view
    sealed._row_has_external = bytes(external_rows)
    sealed._background_fingerprint = background_view
    sealed._journal_storage = journal_storage_view
    sealed._journal_profile_offsets = journal_profile_offset_view
    sealed._journal_rows = journal_row_view
    sealed._journal_region_offsets = journal_region_offset_view
    sealed._journal_regions = journal_region_view
    sealed._journal_compact_row_offsets = journal_compact_row_offset_view
    sealed._journal_compact_results = journal_compact_result_view
    sealed._journal_compact_trace_offsets = (
        journal_compact_trace_offset_view
    )
    sealed._journal_compact_traces = journal_compact_trace_view
    sealed._journal_compact_null2 = journal_compact_null2_view
    sealed._direct_v3_source = direct_v3_source
    sealed._source_consumer_validation_ns = source_validation_elapsed_ns
    if direct_v3_source:
        sealed._source_identity_tokens = identity_view
        sealed._source_profile_fingerprints = profile_fingerprint_view
        sealed._source_sequence_fingerprint = sequence_fingerprint_view
        sealed._direct_v3_eliminated_v2_bytes = direct_source[25]
        sealed._direct_v3_staging_build_ns = direct_source[26]
        sealed._direct_v3_staging_bytes = direct_source[27]
        sealed._direct_v3_source_validation_ns = direct_source[41]
        direct_decision_view = journal_values[25]
        direct_decision_counts = <tuple> journal_values[26]
        sealed._direct_v3_decisions = direct_decision_view
        sealed._direct_v3_expected_exception_count = direct_decision_counts[0]
        sealed._direct_v3_expected_special_count = direct_decision_counts[1]
        sealed._direct_v3_expected_region_count = direct_decision_counts[2]
        sealed._direct_v3_expected_compact_result_count = (
            direct_decision_counts[3]
        )
        sealed._direct_v3_expected_compact_trace_count = (
            direct_decision_counts[4]
        )
        sealed._direct_v3_expected_compact_null2_count = (
            direct_decision_counts[5]
        )
        sealed._direct_v3_expected_decision_tag = direct_decision_counts[6]
        direct_forward_provenance_view = direct_source[38]
        direct_backward_provenance_view = direct_source[39]
        direct_rescore_provenance_view = direct_source[40]
        memcpy(
            &sealed._source_forward_provenance,
            &direct_forward_provenance_view[0],
            sizeof(plan7_forward_provenance),
        )
        memcpy(
            &sealed._source_backward_provenance,
            &direct_backward_provenance_view[0],
            sizeof(plan7_backward_domain_provenance),
        )
        memcpy(
            &sealed._source_rescore_provenance,
            &direct_rescore_provenance_view[0],
            sizeof(plan7_domain_rescore_provenance),
        )
    sealed._f1 = f1
    sealed._generation_f2_bits = pipeline_options.f2_bits
    sealed._generation_f3_bits = pipeline_options.f3_bits
    sealed._generation_bias_filter = True
    sealed._journal_guard_bits = journal_guard_bits
    sealed._generation_tail_fingerprint = generation_tail_fingerprint
    sealed._rescore_simple_row_count = rescore_simple_row_count
    sealed._rescore_device_result_count = rescore_device_result_count
    sealed._rescore_cpu_required_count = rescore_cpu_required_count
    sealed._rescore_certified_ga_count = rescore_certified_ga_count
    sealed._rescore_numeric_fallback_count = rescore_numeric_fallback_count
    sealed._rescore_cap_fallback_count = rescore_cap_fallback_count
    sealed._rescore_global_cpu_fallback_count = (
        rescore_global_cpu_fallback_count
    )
    if residue_offsets_copied:
        sealed._owned_residue_offsets_bytes = (
            int(residue_offset_view.shape[0]) * sizeof(uint64_t)
        )
    else:
        sealed._excluded_residue_offsets_bytes = (
            int(residue_offset_view.shape[0]) * sizeof(uint64_t)
        )
    if background_fingerprint_copied:
        sealed._owned_background_fingerprint_bytes = int(
            background_view.shape[0]
        )
    else:
        sealed._excluded_background_fingerprint_bytes = int(
            background_view.shape[0]
        )
    sealed._native_stage_timings = validated_stage_timings
    sealed._generation_statistics = generation_statistics
    if generation_statistics is not None or direct_v3_source:
        sealed._telemetry_session_id = expected_session_id
        sealed._telemetry_selection_id = expected_selection_id
        sealed._telemetry_batch_generation = batch_generation
    sealed._pipeline_options = pipeline_options
    sealed._filter_scores_seam = filter_scores_seam
    sealed._forward_scores_seam = forward_scores_seam
    sealed._forward_backward_scores_seam = forward_backward_scores_seam
    sealed._simple_regions_seam = simple_regions_seam
    sealed._compact_tail_fingerprint = (
        compact_tail_fingerprint if generation_compact_domains else NULL
    )
    sealed._compact_domains_seam = (
        compact_domains_seam if generation_compact_domains else NULL
    )
    sealed._ready = True
    if sparse_journal_v3:
        try:
            planning_start_ns = _time.perf_counter_ns()
            journal_v3 = _v3_allocate_from_seal(sealed)
            sealed._journal_v3_bytes = journal_v3.total_bytes
            sealed._journal_v3_planning_ns = (
                _time.perf_counter_ns() - planning_start_ns
            )
            validation_start_ns = _time.perf_counter_ns()
            _v3_validate_packet(
                journal_v3,
                sealed._journal_v3_bytes,
                <uint64_t> id(sealed),
                sealed,
                False,
            )
            sealed._journal_v3_validation_ns = (
                _time.perf_counter_ns() - validation_start_ns
            )
            sealed._journal_v3 = journal_v3
            journal_v3 = NULL
        finally:
            free(journal_v3)
    if direct_v3_source:
        _v3_drop_direct_staging(sealed)
    return sealed


def _seal_continuation_journal_v2_test_fixture_bound(
    queries,
    optimized_profiles,
    DigitalSequenceBlock sequences,
    const uint64_t[::1] residue_offsets,
    double f1,
    background_fingerprint,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] postfilter_offsets,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] forward_offsets,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint64_t[::1] domain_profile_offsets,
    const uint8_t[::1] domain_rows,
    const uint64_t[::1] region_offsets,
    const uint8_t[::1] regions,
    const uint64_t[::1] compact_row_offsets,
    const uint8_t[::1] compact_results,
    const uint64_t[::1] compact_trace_offsets,
    const uint8_t[::1] compact_traces,
    const float[::1] compact_null2,
    Pipeline pipeline,
    double guard_band,
    bint sparse_journal_v3=False,
):
    """Mint and production-validate one synthetic binary v2 domain bundle.

    This deliberately narrow underscore hook is absent from the adapter and
    every public execution path.  CUDA-hidden tests use it to exercise the
    real v2 parser/seal and v3 serializer with actual ABI records; it is not a
    second Python classifier and is not reachable from production candidate
    construction.
    """
    cdef tuple query_tuple = tuple(queries)
    cdef tuple profile_tuple = tuple(optimized_profiles)
    cdef size_t profile_count = len(profile_tuple)
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t row_count
    cdef size_t region_count
    cdef size_t compact_result_count
    cdef size_t compact_trace_count
    cdef size_t compact_null2_count = compact_null2.shape[0]
    cdef size_t cursor
    cdef size_t row_index
    cdef size_t result_index
    cdef size_t simple_row_count = 0
    cdef size_t device_result_count = 0
    cdef size_t cpu_required_count = 0
    cdef uint64_t compact_output_bytes = 0
    cdef uint64_t compact_output_component = 0
    cdef uint64_t postfilter_offsets_offset = 0
    cdef uint64_t postfilter_records_offset = 0
    cdef uint64_t forward_offsets_offset = 0
    cdef uint64_t forward_records_offset = 0
    cdef uint64_t forward_special_offsets_offset = 0
    cdef uint64_t profile_offsets_offset = 0
    cdef uint64_t identity_tokens_offset = 0
    cdef uint64_t profile_fingerprints_offset = 0
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
    cdef uint64_t generation_tail_fingerprint = 0
    cdef uint64_t result_hash = 0
    cdef uint64_t trace_hash = 0
    cdef uint64_t null2_hash = 0
    cdef plan7_continuation_journal *journal = NULL
    cdef plan7_continuation_journal_row domain_row
    cdef plan7_domain_rescore_result compact_result
    cdef _pipeline_compact_tail_fingerprint_f tail_fingerprint = NULL
    cdef _double_bits f1_bits
    cdef _double_bits f2_bits
    cdef _double_bits f3_bits
    cdef _float_bits rt_bits
    cdef _float_bits guard_bits
    cdef object identity_owner
    cdef object profile_fingerprint_owner
    cdef object sequence_fingerprint_owner
    cdef const uint64_t[::1] identity_view
    cdef const uint8_t[::1] profile_fingerprint_view
    cdef const uint8_t[::1] sequence_fingerprint_view
    cdef object capsule
    cdef uint8_t *base

    if len(query_tuple) != profile_count:
        raise ValueError("test v2 fixture HMM/profile count differs")
    if postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("test v2 post-filter storage has trailing bytes")
    if forward_records.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("test v2 Forward storage has trailing bytes")
    if domain_rows.shape[0] % sizeof(plan7_continuation_journal_row) != 0:
        raise ValueError("test v2 domain storage has trailing bytes")
    if regions.shape[0] % sizeof(plan7_simple_region) != 0:
        raise ValueError("test v2 region storage has trailing bytes")
    if compact_results.shape[0] % sizeof(plan7_domain_rescore_result) != 0:
        raise ValueError("test v2 compact-result storage has trailing bytes")
    if compact_traces.shape[0] % sizeof(plan7_domain_rescore_trace_step) != 0:
        raise ValueError("test v2 compact-trace storage has trailing bytes")

    postfilter_count = postfilter_records.shape[0] // sizeof(_postfilter_result)
    forward_count = forward_records.shape[0] // sizeof(_forward_result)
    row_count = domain_rows.shape[0] // sizeof(plan7_continuation_journal_row)
    region_count = regions.shape[0] // sizeof(plan7_simple_region)
    compact_result_count = (
        compact_results.shape[0] // sizeof(plan7_domain_rescore_result)
    )
    compact_trace_count = (
        compact_traces.shape[0] // sizeof(plan7_domain_rescore_trace_step)
    )
    if compact_result_count > (<size_t> -1) // PLAN7_DOMAIN_RESCORE_NULL2_COUNT:
        raise OverflowError("test v2 compact null2 count overflows")
    if (
        postfilter_offsets.shape[0] != profile_count + 1
        or forward_offsets.shape[0] != profile_count + 1
        or special_offsets.shape[0] != forward_count + 1
        or domain_profile_offsets.shape[0] != profile_count + 1
        or region_offsets.shape[0] != row_count + 1
        or compact_row_offsets.shape[0] != row_count + 1
        or compact_trace_offsets.shape[0] != compact_result_count + 1
        or compact_null2_count
        != compact_result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
    ):
        raise ValueError("test v2 fixture offset/count shape differs")

    for row_index in range(row_count):
        memcpy(
            &domain_row,
            &domain_rows[row_index * sizeof(plan7_continuation_journal_row)],
            sizeof(plan7_continuation_journal_row),
        )
        if domain_row.domain_route == DOMAIN_SIMPLE:
            simple_row_count += 1
    for result_index in range(compact_result_count):
        memcpy(
            &compact_result,
            &compact_results[
                result_index * sizeof(plan7_domain_rescore_result)
            ],
            sizeof(plan7_domain_rescore_result),
        )
        if compact_result.action == PLAN7_DOMAIN_RESCORE_DEVICE_RESULT:
            device_result_count += 1
        elif compact_result.action == PLAN7_DOMAIN_RESCORE_CPU_REQUIRED:
            cpu_required_count += 1
        else:
            raise ValueError("test v2 compact action is invalid")

    if compact_result_count:
        if (
            _cached_compact_domains_seam() == NULL
            or _compact_tail_fingerprint_cache == NULL
        ):
            raise RuntimeError("test v2 fixture requires the compact tail seam")
        tail_fingerprint = _compact_tail_fingerprint_cache
        with nogil:
            generation_tail_fingerprint = tail_fingerprint(pipeline._pli)
        if generation_tail_fingerprint == 0:
            raise ValueError("test v2 compact tail fingerprint is zero")
        if (
            not plan7_continuation_journal_v3_checked_multiply(
                compact_result_count,
                sizeof(plan7_domain_rescore_result),
                &compact_output_bytes,
            )
            or not plan7_continuation_journal_v3_checked_multiply(
                compact_result_count + 1,
                sizeof(uint64_t),
                &compact_output_component,
            )
            or not _v3_checked_increment(
                &compact_output_bytes, compact_output_component
            )
            or not plan7_continuation_journal_v3_checked_multiply(
                compact_trace_count,
                sizeof(plan7_domain_rescore_trace_step),
                &compact_output_component,
            )
            or not _v3_checked_increment(
                &compact_output_bytes, compact_output_component
            )
            or not plan7_continuation_journal_v3_checked_multiply(
                compact_null2_count,
                sizeof(float),
                &compact_output_component,
            )
            or not _v3_checked_increment(
                &compact_output_bytes, compact_output_component
            )
        ):
            raise OverflowError("test v2 compact byte count overflows")
        if compact_output_bytes > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES:
            raise OverflowError("test v2 compact payload exceeds ABI limit")
        if not plan7_continuation_journal_rescore_hashes(
            <const plan7_domain_rescore_result *> &compact_results[0],
            compact_result_count,
            &compact_trace_offsets[0],
            (
                <const plan7_domain_rescore_trace_step *> &compact_traces[0]
                if compact_trace_count
                else NULL
            ),
            compact_trace_count,
            &compact_null2[0],
            compact_null2_count,
            &result_hash,
            &trace_hash,
            &null2_hash,
        ):
            raise ValueError("test v2 compact hashes cannot be computed")

    identity_owner = _array(
        "Q", (0x5632544553540001 + row for row in range(profile_count))
    )
    profile_fingerprint_owner = bytes(
        ((profile * 37 + byte_index + 1) & 0xff)
        for profile in range(profile_count)
        for byte_index in range(PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE)
    )
    sequence_fingerprint_owner = bytes(
        ((byte_index * 17 + 3) & 0xff)
        for byte_index in range(PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE)
    )
    identity_view = identity_owner
    profile_fingerprint_view = profile_fingerprint_owner
    sequence_fingerprint_view = sequence_fingerprint_owner

    cursor = sizeof(plan7_continuation_journal)
    if not (
        _v3_advance_segment(
            &cursor, profile_count + 1, sizeof(uint64_t),
            &postfilter_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, postfilter_count, sizeof(_postfilter_result),
            &postfilter_records_offset,
        )
        and _v3_advance_segment(
            &cursor, profile_count + 1, sizeof(uint64_t),
            &forward_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, forward_count, sizeof(_forward_result),
            &forward_records_offset,
        )
        and _v3_advance_segment(
            &cursor, forward_count + 1, sizeof(uint64_t),
            &forward_special_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, profile_count + 1, sizeof(uint64_t),
            &profile_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, profile_count, sizeof(uint64_t),
            &identity_tokens_offset,
        )
        and _v3_advance_segment(
            &cursor, profile_count,
            PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
            &profile_fingerprints_offset,
        )
        and _v3_advance_segment(
            &cursor, row_count, sizeof(plan7_continuation_journal_row),
            &rows_offset,
        )
        and _v3_advance_segment(
            &cursor, row_count + 1, sizeof(uint64_t),
            &special_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, specials.shape[0], sizeof(float), &specials_offset,
        )
        and _v3_advance_segment(
            &cursor, row_count + 1, sizeof(uint64_t), &region_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, region_count, sizeof(plan7_simple_region), &regions_offset,
        )
        and _v3_advance_segment(
            &cursor, row_count + 1, sizeof(uint64_t),
            &compact_row_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, compact_result_count,
            sizeof(plan7_domain_rescore_result), &compact_results_offset,
        )
        and _v3_advance_segment(
            &cursor, compact_result_count + 1, sizeof(uint64_t),
            &compact_trace_offsets_offset,
        )
        and _v3_advance_segment(
            &cursor, compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step), &compact_traces_offset,
        )
        and _v3_advance_segment(
            &cursor, compact_null2_count, sizeof(float), &compact_null2_offset,
        )
        and cursor <= <size_t> PY_SSIZE_T_MAX
    ):
        raise OverflowError("test v2 fixture storage layout overflow")

    journal = <plan7_continuation_journal *> calloc(1, cursor)
    if journal == NULL:
        raise MemoryError("test v2 fixture allocation failed")
    journal.magic = PLAN7_CONTINUATION_JOURNAL_MAGIC
    journal.version = PLAN7_CONTINUATION_JOURNAL_VERSION
    journal.header_size = sizeof(plan7_continuation_journal)
    journal.row_size = sizeof(plan7_continuation_journal_row)
    journal.region_size = sizeof(plan7_simple_region)
    journal.compact_result_size = sizeof(plan7_domain_rescore_result)
    journal.compact_trace_step_size = sizeof(plan7_domain_rescore_trace_step)
    journal.compact_null2_stride = PLAN7_DOMAIN_RESCORE_NULL2_COUNT
    journal.total_bytes = cursor
    journal.session_id = 0x5632544553540101
    journal.selection_id = 0x5632544553540102
    journal.profile_count = profile_count
    journal.postfilter_count = postfilter_count
    journal.forward_count = forward_count
    journal.row_count = row_count
    journal.special_count = specials.shape[0]
    journal.region_count = region_count
    journal.compact_result_count = compact_result_count
    journal.compact_trace_offset_count = compact_result_count + 1
    journal.compact_trace_count = compact_trace_count
    journal.compact_null2_count = compact_null2_count
    journal.generation_tail_fingerprint = generation_tail_fingerprint
    journal.rescore_simple_row_count = simple_row_count
    journal.rescore_device_result_count = device_result_count
    journal.rescore_cpu_required_count = cpu_required_count
    journal.rescore_cap_fallback_count = cpu_required_count
    journal.rescore_compact_output_byte_limit = (
        PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES if compact_result_count else 0
    )
    journal.rescore_compact_output_bytes = compact_output_bytes
    f1_bits.value = f1
    f2_bits.value = pipeline._pli.F2
    f3_bits.value = pipeline._pli.F3
    journal.generation_f1_bits = f1_bits.bits
    journal.generation_f2_bits = f2_bits.bits
    journal.generation_f3_bits = f3_bits.bits
    rt_bits.value = <float> 0.25
    journal.rt1_bits = rt_bits.bits
    rt_bits.value = <float> 0.10
    journal.rt2_bits = rt_bits.bits
    rt_bits.value = <float> 0.20
    journal.rt3_bits = rt_bits.bits
    guard_bits.value = <float> guard_band
    journal.guard_band_bits = guard_bits.bits
    journal.generation_bias_filter = 1
    journal.generation_compact_domains = <uint8_t> (compact_result_count != 0)
    memcpy(
        journal.sequence_content_fingerprint,
        &sequence_fingerprint_view[0],
        PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE,
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

    journal.forward.database_generation = 0x5632544553540201
    journal.forward.batch_generation = 0x5632544553540202
    journal.forward.row_hash = 0x5632544553540203
    journal.forward.special_hash = 0x5632544553540204
    journal.forward.continuation_hash = 0x5632544553540205
    journal.forward.pass_count = row_count
    journal.forward.special_count = specials.shape[0]
    journal.forward.generation_f3_bits = f3_bits.bits
    journal.forward.integrity_tag = 0x5632544553540206
    memcpy(
        &journal.backward.forward,
        &journal.forward,
        sizeof(plan7_forward_provenance),
    )
    journal.backward.threshold_hash = 0x5632544553540301
    journal.backward.result_hash = 0x5632544553540302
    journal.backward.region_hash = 0x5632544553540303
    journal.backward.candidate_count = row_count
    journal.backward.region_count = region_count
    if compact_result_count:
        memcpy(
            &journal.rescore.backward,
            &journal.backward,
            sizeof(plan7_backward_domain_provenance),
        )
        journal.rescore.result_hash = result_hash
        journal.rescore.trace_hash = trace_hash
        journal.rescore.null2_hash = null2_hash
        journal.rescore.result_count = compact_result_count
        journal.rescore.trace_count = compact_trace_count
        journal.rescore.null2_count = compact_null2_count

    base = <uint8_t *> journal
    memcpy(
        base + postfilter_offsets_offset, &postfilter_offsets[0],
        (profile_count + 1) * sizeof(uint64_t),
    )
    if postfilter_count:
        memcpy(
            base + postfilter_records_offset, &postfilter_records[0],
            postfilter_count * sizeof(_postfilter_result),
        )
    memcpy(
        base + forward_offsets_offset, &forward_offsets[0],
        (profile_count + 1) * sizeof(uint64_t),
    )
    if forward_count:
        memcpy(
            base + forward_records_offset, &forward_records[0],
            forward_count * sizeof(_forward_result),
        )
    memcpy(
        base + forward_special_offsets_offset, &special_offsets[0],
        (forward_count + 1) * sizeof(uint64_t),
    )
    memcpy(
        base + profile_offsets_offset, &domain_profile_offsets[0],
        (profile_count + 1) * sizeof(uint64_t),
    )
    if profile_count:
        memcpy(
            base + identity_tokens_offset, &identity_view[0],
            profile_count * sizeof(uint64_t),
        )
        memcpy(
            base + profile_fingerprints_offset, &profile_fingerprint_view[0],
            profile_count
            * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
        )
    if row_count:
        memcpy(
            base + rows_offset, &domain_rows[0],
            row_count * sizeof(plan7_continuation_journal_row),
        )
    memcpy(
        base + special_offsets_offset, &special_offsets[0],
        (row_count + 1) * sizeof(uint64_t),
    )
    if specials.shape[0]:
        memcpy(
            base + specials_offset, &specials[0],
            specials.shape[0] * sizeof(float),
        )
    memcpy(
        base + region_offsets_offset, &region_offsets[0],
        (row_count + 1) * sizeof(uint64_t),
    )
    if region_count:
        memcpy(
            base + regions_offset, &regions[0],
            region_count * sizeof(plan7_simple_region),
        )
    memcpy(
        base + compact_row_offsets_offset, &compact_row_offsets[0],
        (row_count + 1) * sizeof(uint64_t),
    )
    if compact_result_count:
        memcpy(
            base + compact_results_offset, &compact_results[0],
            compact_result_count * sizeof(plan7_domain_rescore_result),
        )
    memcpy(
        base + compact_trace_offsets_offset, &compact_trace_offsets[0],
        (compact_result_count + 1) * sizeof(uint64_t),
    )
    if compact_trace_count:
        memcpy(
            base + compact_traces_offset, &compact_traces[0],
            compact_trace_count * sizeof(plan7_domain_rescore_trace_step),
        )
    if compact_null2_count:
        memcpy(
            base + compact_null2_offset, &compact_null2[0],
            compact_null2_count * sizeof(float),
        )
    journal.integrity_tag = plan7_continuation_journal_integrity(journal)
    if journal.integrity_tag == 0:
        free(journal)
        raise ValueError("test v2 fixture integrity tag is zero")

    try:
        capsule = PyCapsule_New(
            journal,
            PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME,
            _v2_test_fixture_capsule_destroy,
        )
    except:
        free(journal)
        raise
    return _seal_profile_selection_continuation_bound(
        query_tuple,
        profile_tuple,
        sequences,
        residue_offsets,
        f1,
        background_fingerprint,
        capsule,
        (
            0x5632544553540101,
            0x5632544553540102,
        ),
        identity_owner,
        profile_fingerprint_owner,
        0x5632544553540202,
        sequence_fingerprint_owner,
        pipeline,
        guard_band,
        None,
        None,
        sparse_journal_v3,
    )


cdef bint _sealed_background_matches(
    _SealedPostfilterBatch sealed,
    Pipeline pipeline,
) noexcept nogil:
    cdef size_t frequency_bytes = (
        <size_t> pipeline.background._bg.abc.K * sizeof(float)
    )
    if sealed._background_fingerprint.shape[0] != frequency_bytes + sizeof(float):
        return False
    if memcmp(
        &sealed._background_fingerprint[0],
        pipeline.background._bg.f,
        frequency_bytes,
    ) != 0:
        return False
    return memcmp(
        &sealed._background_fingerprint[frequency_bytes],
        &pipeline.background._bg.omega,
        sizeof(float),
    ) == 0


cdef object _sealed_search_result(
    TopHits hits,
    const _compact_consumption_statistics* statistics,
    bint return_compact_statistics,
    bint return_route_statistics,
    object route_path,
    object start_ns,
    uint64_t telemetry_session_id,
    uint64_t telemetry_selection_id,
    uint64_t telemetry_batch_generation,
):
    cdef object elapsed_ns
    cdef object route_statistics
    if not return_compact_statistics and not return_route_statistics:
        return hits
    if return_route_statistics:
        elapsed_ns = _time.perf_counter_ns() - start_ns
        route_statistics = _telemetry_module.build_continuation_statistics(
            _telemetry_module.GENERATION_TELEMETRY_SCHEMA_VERSION,
            route_path,
            elapsed_ns,
            statistics.target_count,
            statistics.postfilter_record_count,
            (
                statistics.f1_reject_count,
                statistics.cpu_pipeline_count,
                statistics.definite_reject_count,
                statistics.filter_continuation_count,
                statistics.forward_continuation_count,
                statistics.simple_continuation_count,
                statistics.accepted_count,
            ),
            (
                statistics.journal_match_count,
                statistics.journal_cpu_required_count,
                statistics.journal_no_region_count,
                statistics.journal_simple_count,
            ),
            (
                statistics.attempt_count,
                statistics.accepted_count,
                statistics.invalid_retry_count,
                statistics.threshold_retry_count,
                statistics.first_row_index,
                statistics.first_profile_index,
                statistics.first_sequence_index,
                statistics.first_domain_count,
            ),
            (
                statistics.source_postfilter_cpu_count,
                statistics.source_definite_reject_count,
                statistics.source_filter_count,
                statistics.source_forward_count,
                statistics.source_journal_eligible_count,
                statistics.source_simple_bypass_count,
            ),
            (
                statistics.decision_forward_row_external_unavailable,
                statistics.decision_forward_seam_unavailable,
                statistics.decision_forward_f2_changed,
                statistics.decision_forward_f3_changed,
                statistics.decision_forward_bias_changed,
                statistics.decision_journal_storage_unavailable,
                statistics.decision_journal_simple_seam_unavailable,
                statistics.decision_journal_tail_changed,
                statistics.decision_compact_route_not_device,
                statistics.decision_compact_empty,
                statistics.decision_compact_tail_changed,
                statistics.decision_compact_rebase_unavailable,
            ),
            (
                statistics.requested_profile_index,
                statistics.journal_row_start,
                statistics.journal_row_stop,
            ),
            (
                None
                if telemetry_session_id == 0
                else (
                    int(telemetry_session_id),
                    int(telemetry_selection_id),
                    int(telemetry_batch_generation),
                )
            ),
        )
        route_statistics["cpu_tail_filter_simd"] = {
            "eligible_rows": statistics.cpu_tail_filter_simd_eligible_rows,
            "groups": statistics.cpu_tail_filter_simd_groups,
            "rows": statistics.cpu_tail_filter_simd_rows,
            "scalar_tail_rows": (
                statistics.cpu_tail_filter_simd_scalar_tail_rows
            ),
            "fallback_groups": (
                statistics.cpu_tail_filter_simd_fallback_groups
            ),
            "fallback_rows": (
                statistics.cpu_tail_filter_simd_fallback_rows
            ),
            "f3_survivor_rows": (
                statistics.cpu_tail_filter_simd_f3_survivor_rows
            ),
            "backward_groups": (
                statistics.cpu_tail_filter_simd_backward_groups
            ),
            "forward_kernel_ns": (
                statistics.cpu_tail_filter_simd_forward_kernel_ns
            ),
            "backward_kernel_ns": (
                statistics.cpu_tail_filter_simd_backward_kernel_ns
            ),
        }
        return hits, route_statistics
    return hits, {
        "attempt_count": statistics.attempt_count,
        "accepted_count": statistics.accepted_count,
        "invalid_retry_count": statistics.invalid_retry_count,
        "threshold_retry_count": statistics.threshold_retry_count,
        "first_attempt": (
            None
            if statistics.attempt_count == 0
            else (
                statistics.first_row_index,
                statistics.first_profile_index,
                statistics.first_sequence_index,
                statistics.first_domain_count,
            )
        ),
        "cpu_tail_simd": {
            "eligible_rows": statistics.cpu_tail_simd_eligible_rows,
            "groups": statistics.cpu_tail_simd_groups,
            "rows": statistics.cpu_tail_simd_rows,
            "scalar_tail_rows": statistics.cpu_tail_simd_scalar_tail_rows,
            "fallback_groups": statistics.cpu_tail_simd_fallback_groups,
            "kernel_ns": statistics.cpu_tail_simd_kernel_ns,
        },
        "cpu_tail_filter_simd": {
            "eligible_rows": statistics.cpu_tail_filter_simd_eligible_rows,
            "groups": statistics.cpu_tail_filter_simd_groups,
            "rows": statistics.cpu_tail_filter_simd_rows,
            "scalar_tail_rows": (
                statistics.cpu_tail_filter_simd_scalar_tail_rows
            ),
            "fallback_groups": (
                statistics.cpu_tail_filter_simd_fallback_groups
            ),
            "fallback_rows": (
                statistics.cpu_tail_filter_simd_fallback_rows
            ),
            "f3_survivor_rows": (
                statistics.cpu_tail_filter_simd_f3_survivor_rows
            ),
            "backward_groups": (
                statistics.cpu_tail_filter_simd_backward_groups
            ),
            "forward_kernel_ns": (
                statistics.cpu_tail_filter_simd_forward_kernel_ns
            ),
            "backward_kernel_ns": (
                statistics.cpu_tail_filter_simd_backward_kernel_ns
            ),
        },
    }


cdef void _count_nonjournal_routes(
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    uint64_t target_count,
    _compact_consumption_statistics* statistics,
) except *:
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef size_t postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef size_t forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    cdef size_t cursor
    cdef size_t forward_cursor = 0
    cdef bint has_forward
    if postfilter_count > target_count:
        raise RuntimeError("continuation post-filter rows exceed targets")
    statistics.target_count = target_count
    statistics.postfilter_record_count = postfilter_count
    statistics.f1_reject_count = target_count - postfilter_count
    for cursor in range(postfilter_count):
        memcpy(
            &postfilter,
            &postfilter_records[cursor * sizeof(_postfilter_result)],
            sizeof(_postfilter_result),
        )
        has_forward = False
        if forward_cursor < forward_count:
            memcpy(
                &forward,
                &forward_records[
                    forward_cursor * sizeof(_forward_result)
                ],
                sizeof(_forward_result),
            )
            has_forward = forward.sequence_index == postfilter.sequence_index
        if postfilter.action == BIAS_CPU_REQUIRED:
            statistics.cpu_pipeline_count += 1
            statistics.source_postfilter_cpu_count += 1
        elif isnan(postfilter.filtersc):
            statistics.definite_reject_count += 1
            statistics.source_definite_reject_count += 1
        elif has_forward and forward.action != FORWARD_CPU_REQUIRED:
            statistics.forward_continuation_count += 1
            statistics.source_forward_count += 1
        else:
            statistics.filter_continuation_count += 1
            statistics.source_filter_count += 1
        if has_forward:
            forward_cursor += 1
    if forward_cursor != forward_count:
        raise RuntimeError("continuation telemetry Forward rows changed")


def _search_hmm_sealed_postfilter_bound(
    sealed_object,
    Py_ssize_t row,
    Pipeline pipeline,
    bint _return_compact_statistics=False,
    bint _return_route_statistics=False,
):
    """Search one row whose complete immutable batch was already validated."""
    cdef _SealedPostfilterBatch sealed
    cdef HMM query
    cdef OptimizedProfile optimized_profile
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t journal_start
    cdef size_t journal_stop
    cdef _double_bits live_f2
    cdef _double_bits live_f3
    cdef _double_bits generation_f1
    cdef bint use_forward
    cdef bint use_journal
    cdef _compact_consumption_statistics statistics
    cdef _compact_consumption_statistics* compact_statistics = NULL
    cdef object start_ns = None

    if _return_compact_statistics or _return_route_statistics:
        memset(&statistics, 0, sizeof(_compact_consumption_statistics))
        compact_statistics = &statistics
    if _return_route_statistics:
        start_ns = _time.perf_counter_ns()

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if sealed._direct_v3_source:
        raise TypeError(
            "direct sparse-v3 batches cannot enter the dense replay consumer"
        )
    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")
    if not pipeline.alphabet._eq(sealed._sequences.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sealed._sequences.alphabet)
    if pipeline._pli.F1 != sealed._f1:
        raise ValueError(
            f"pipeline F1 {pipeline.F1!r} does not match "
            f"candidate F1 {sealed._f1!r}"
        )
    if not _sealed_background_matches(sealed, pipeline):
        raise ValueError(
            "pipeline background does not match the canonical hmmpress background"
        )

    # Make the query copy before any pipeline mutation. Each successful call
    # consequently owns an independent TopHits.query, even when a row is reused.
    query = (<HMM> sealed._queries[row]).copy()
    optimized_profile = <OptimizedProfile> sealed._optimized_profiles[row]
    postfilter_start = <size_t> sealed._postfilter_offsets[row]
    postfilter_stop = <size_t> sealed._postfilter_offsets[row + 1]
    forward_start = <size_t> sealed._forward_offsets[row]
    forward_stop = <size_t> sealed._forward_offsets[row + 1]
    if compact_statistics != NULL:
        statistics.requested_profile_index = row

    live_f2.value = pipeline._pli.F2
    live_f3.value = pipeline._pli.F3
    use_forward = (
        sealed._row_has_external[row]
        and sealed._forward_scores_seam != NULL
        and live_f2.bits == sealed._generation_f2_bits
        and live_f3.bits == sealed._generation_f3_bits
        and pipeline._pli.do_biasfilter == sealed._generation_bias_filter
    )
    if not use_forward:
        if _return_route_statistics:
            statistics.decision_forward_row_external_unavailable = (
                0 if sealed._row_has_external[row] else 1
            )
            statistics.decision_forward_seam_unavailable = (
                0 if sealed._forward_scores_seam != NULL else 1
            )
            statistics.decision_forward_f2_changed = (
                0 if live_f2.bits == sealed._generation_f2_bits else 1
            )
            statistics.decision_forward_f3_changed = (
                0 if live_f3.bits == sealed._generation_f3_bits else 1
            )
            statistics.decision_forward_bias_changed = (
                0
                if pipeline._pli.do_biasfilter
                == sealed._generation_bias_filter
                else 1
            )
            _count_nonjournal_routes(
                sealed._postfilter_records[
                    postfilter_start * sizeof(_postfilter_result):
                    postfilter_stop * sizeof(_postfilter_result)
                ],
                sealed._forward_records[0:0],
                len(sealed._sequences),
                &statistics,
            )
        return _sealed_search_result(
            _search_postfilter_validated(
                pipeline,
                query,
                optimized_profile,
                sealed._sequences,
                sealed._postfilter_records[
                    postfilter_start * sizeof(_postfilter_result):
                    postfilter_stop * sizeof(_postfilter_result)
                ],
                &sealed._residue_offsets[0],
                sealed._filter_scores_seam,
            ),
            &statistics,
            _return_compact_statistics,
            _return_route_statistics,
            "postfilter",
            start_ns,
            sealed._telemetry_session_id,
            sealed._telemetry_selection_id,
            sealed._telemetry_batch_generation,
        )
    use_journal = (
        sealed._journal_storage.shape[0] != 0
        and sealed._simple_regions_seam != NULL
        and _pipeline_tail_options_match(&sealed._pipeline_options, pipeline)
    )
    if use_journal:
        journal_start = <size_t> sealed._journal_profile_offsets[row]
        journal_stop = <size_t> sealed._journal_profile_offsets[row + 1]
        if compact_statistics != NULL:
            statistics.journal_row_start = journal_start
            statistics.journal_row_stop = journal_stop
        generation_f1.value = sealed._f1
        return _sealed_search_result(
            _search_postfilter_forward_journal_validated(
                pipeline,
                query,
                optimized_profile,
                sealed._sequences,
                sealed._postfilter_records[
                    postfilter_start * sizeof(_postfilter_result):
                    postfilter_stop * sizeof(_postfilter_result)
                ],
                sealed._forward_records[
                    forward_start * sizeof(_forward_result):
                    forward_stop * sizeof(_forward_result)
                ],
                sealed._special_offsets[forward_start:forward_stop + 1],
                sealed._specials,
                sealed._journal_rows[
                    journal_start * sizeof(plan7_continuation_journal_row):
                    journal_stop * sizeof(plan7_continuation_journal_row)
                ],
                sealed._journal_region_offsets[journal_start:journal_stop + 1],
                sealed._journal_regions,
                sealed._journal_compact_row_offsets[
                    journal_start:journal_stop + 1
                ],
                sealed._journal_compact_results,
                sealed._journal_compact_trace_offsets,
                sealed._journal_compact_traces,
                sealed._journal_compact_null2,
                journal_start,
                sealed._generation_tail_fingerprint,
                &sealed._residue_offsets[0],
                generation_f1.bits,
                sealed._generation_f2_bits,
                sealed._generation_f3_bits,
                sealed._generation_bias_filter,
                sealed._filter_scores_seam,
                sealed._forward_scores_seam,
                sealed._simple_regions_seam,
                sealed._compact_tail_fingerprint,
                sealed._compact_domains_seam,
                compact_statistics,
            ),
            &statistics,
            _return_compact_statistics,
            _return_route_statistics,
            "journal",
            start_ns,
            sealed._telemetry_session_id,
            sealed._telemetry_selection_id,
            sealed._telemetry_batch_generation,
        )
    if _return_route_statistics:
        statistics.decision_journal_storage_unavailable = (
            0 if sealed._journal_storage.shape[0] != 0 else 1
        )
        statistics.decision_journal_simple_seam_unavailable = (
            0 if sealed._simple_regions_seam != NULL else 1
        )
        statistics.decision_journal_tail_changed = (
            0
            if _pipeline_tail_options_match(
                &sealed._pipeline_options, pipeline
            )
            else 1
        )
        _count_nonjournal_routes(
            sealed._postfilter_records[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            sealed._forward_records[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            len(sealed._sequences),
            &statistics,
        )
    return _sealed_search_result(
        _search_postfilter_forward_validated(
            pipeline,
            query,
            optimized_profile,
            sealed._sequences,
            sealed._postfilter_records[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            sealed._forward_records[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            sealed._special_offsets[forward_start:forward_stop + 1],
            sealed._specials,
            &sealed._residue_offsets[0],
            sealed._filter_scores_seam,
            sealed._forward_scores_seam,
        ),
        &statistics,
        _return_compact_statistics,
        _return_route_statistics,
        "forward",
        start_ns,
        sealed._telemetry_session_id,
        sealed._telemetry_selection_id,
        sealed._telemetry_batch_generation,
    )


def _sealed_postfilter_candidate_count_bound(sealed_object, Py_ssize_t row):
    """Return one opaque batch's authentic post-filter row count."""
    cdef _SealedPostfilterBatch sealed
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_profile *profiles
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")
    if sealed._direct_v3_source:
        if sealed._journal_v3 == NULL:
            raise RuntimeError("direct v3 packet is unavailable")
        base = <const uint8_t *> sealed._journal_v3
        profiles = <const plan7_continuation_journal_v3_profile *> (
            base + sealed._journal_v3.profiles_offset
        )
        return profiles[row].source_postfilter_count
    return sealed._postfilter_offsets[row + 1] - sealed._postfilter_offsets[row]


def _sealed_continuation_work_hints_bound(sealed_object):
    """Return deterministic per-profile continuation work hints.

    The hints use only immutable, authenticated sparse-v3 metadata.  They are
    deliberately coarse: their purpose is to keep obviously heavy DP rows out
    of the same continuation task, not to predict elapsed time.
    """
    cdef _SealedPostfilterBatch sealed
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_profile *profiles
    cdef const plan7_continuation_journal_v3_exception *exceptions
    cdef const plan7_continuation_journal_v3_profile *profile
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef OptimizedProfile optimized_profile
    cdef uint64_t maximum = <uint64_t> -1
    cdef uint64_t cells
    cdef uint64_t weighted
    cdef uint64_t cost
    cdef uint64_t weight
    cdef uint64_t local_index
    cdef size_t profile_index
    cdef list hints

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if sealed._journal_v3 == NULL:
        raise TypeError("continuation work hints require sparse journal v3")

    base = <const uint8_t *> sealed._journal_v3
    profiles = <const plan7_continuation_journal_v3_profile *> (
        base + sealed._journal_v3.profiles_offset
    )
    exceptions = <const plan7_continuation_journal_v3_exception *> (
        base + sealed._journal_v3.exceptions_offset
    )
    hints = []
    for profile_index in range(<size_t> sealed._journal_v3.profile_count):
        profile = &profiles[profile_index]
        optimized_profile = <OptimizedProfile> sealed._optimized_profiles[
            profile_index
        ]
        # A nonzero base keeps cert-only profiles schedulable. Each exception
        # also contributes one unit so zero-cell output/control work remains
        # visible beside the DP-cell terms below.
        cost = 1
        for local_index in range(profile.exception_count):
            exception = &exceptions[profile.exception_begin + local_index]
            if cost != maximum and not plan7_continuation_journal_v3_checked_add(
                cost, 1, &cost
            ):
                cost = maximum
            weight = 0
            if exception.route == PLAN7_CONTINUATION_V3_FULL_PIPELINE:
                weight = 4
            elif exception.route == PLAN7_CONTINUATION_V3_FILTER_SCORES:
                weight = 3
            elif (
                exception.route == PLAN7_CONTINUATION_V3_FORWARD_SCORES
                and exception.payload_flags & PLAN7_CONTINUATION_V3_HAS_DOMAIN
            ):
                weight = 2
            if weight == 0 or cost == maximum:
                continue
            if not plan7_continuation_journal_v3_checked_multiply(
                <uint64_t> optimized_profile._om.M,
                exception.residue_delta,
                &cells,
            ) or not plan7_continuation_journal_v3_checked_multiply(
                cells, weight, &weighted
            ) or not plan7_continuation_journal_v3_checked_add(
                cost, weighted, &cost
            ):
                cost = maximum
        hints.append(cost)
    return tuple(hints)


def _sealed_continuation_shard_work_hints_bound(sealed_object):
    """Return deterministic work hints for each sparse-v3 exception.

    The nested tuples preserve profile and exception order exactly.  They use
    the same coarse route weights as the retained per-profile scheduler hint,
    but expose exception boundaries so an internal scheduler can split a
    pathological profile without splitting one HMMER target continuation.
    """
    cdef _SealedPostfilterBatch sealed
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_profile *profiles
    cdef const plan7_continuation_journal_v3_exception *exceptions
    cdef const plan7_continuation_journal_v3_profile *profile
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef OptimizedProfile optimized_profile
    cdef uint64_t maximum = <uint64_t> -1
    cdef uint64_t cells
    cdef uint64_t weighted
    cdef uint64_t cost
    cdef uint64_t weight
    cdef uint64_t local_index
    cdef size_t profile_index
    cdef list profile_hints
    cdef list exception_hints

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if sealed._journal_v3 == NULL:
        raise TypeError("continuation shard hints require sparse journal v3")

    base = <const uint8_t *> sealed._journal_v3
    profiles = <const plan7_continuation_journal_v3_profile *> (
        base + sealed._journal_v3.profiles_offset
    )
    exceptions = <const plan7_continuation_journal_v3_exception *> (
        base + sealed._journal_v3.exceptions_offset
    )
    profile_hints = []
    for profile_index in range(<size_t> sealed._journal_v3.profile_count):
        profile = &profiles[profile_index]
        optimized_profile = <OptimizedProfile> sealed._optimized_profiles[
            profile_index
        ]
        exception_hints = []
        for local_index in range(profile.exception_count):
            exception = &exceptions[profile.exception_begin + local_index]
            cost = 1
            weight = 0
            if exception.route == PLAN7_CONTINUATION_V3_FULL_PIPELINE:
                weight = 4
            elif exception.route == PLAN7_CONTINUATION_V3_FILTER_SCORES:
                weight = 3
            elif (
                exception.route == PLAN7_CONTINUATION_V3_FORWARD_SCORES
                and exception.payload_flags & PLAN7_CONTINUATION_V3_HAS_DOMAIN
            ):
                weight = 2
            if weight != 0:
                if not plan7_continuation_journal_v3_checked_multiply(
                    <uint64_t> optimized_profile._om.M,
                    exception.residue_delta,
                    &cells,
                ) or not plan7_continuation_journal_v3_checked_multiply(
                    cells, weight, &weighted
                ) or not plan7_continuation_journal_v3_checked_add(
                    cost, weighted, &cost
                ):
                    cost = maximum
            exception_hints.append(cost)
        profile_hints.append(tuple(exception_hints))
    return tuple(profile_hints)


def _sealed_sparse_journal_v3_enabled_bound(sealed_object):
    """Return whether one sealed batch opted into reusable sparse v3."""
    cdef _SealedPostfilterBatch sealed
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    return sealed._journal_v3 != NULL


def _sparse_journal_v3_consumer_statistics_bound():
    """Return cumulative successful sparse-consumer timing and visit counts."""
    return {
        "call_count": _v3_consumer_call_count,
        "preflight_ns": _v3_consumer_preflight_ns,
        "core_ns": _v3_consumer_core_ns,
        "statistics_ns": _v3_consumer_statistics_ns,
        "certificate_visits": _v3_consumer_certificate_visits,
        "exception_visits": _v3_consumer_exception_visits,
    }


cdef void _record_filter_tail_audit(
    const _compact_consumption_statistics *statistics,
) noexcept:
    """Accumulate successful filter-tail work while the calling thread has GIL."""
    global _filter_tail_audit_eligible_rows
    global _filter_tail_audit_groups
    global _filter_tail_audit_rows
    global _filter_tail_audit_scalar_tail_rows
    global _filter_tail_audit_fallback_groups
    global _filter_tail_audit_fallback_rows
    global _filter_tail_audit_f3_survivor_rows
    global _filter_tail_audit_backward_groups
    global _filter_tail_audit_forward_kernel_ns
    global _filter_tail_audit_backward_kernel_ns
    _filter_tail_audit_eligible_rows += (
        statistics.cpu_tail_filter_simd_eligible_rows
    )
    _filter_tail_audit_groups += statistics.cpu_tail_filter_simd_groups
    _filter_tail_audit_rows += statistics.cpu_tail_filter_simd_rows
    _filter_tail_audit_scalar_tail_rows += (
        statistics.cpu_tail_filter_simd_scalar_tail_rows
    )
    _filter_tail_audit_fallback_groups += (
        statistics.cpu_tail_filter_simd_fallback_groups
    )
    _filter_tail_audit_fallback_rows += (
        statistics.cpu_tail_filter_simd_fallback_rows
    )
    _filter_tail_audit_f3_survivor_rows += (
        statistics.cpu_tail_filter_simd_f3_survivor_rows
    )
    _filter_tail_audit_backward_groups += (
        statistics.cpu_tail_filter_simd_backward_groups
    )
    _filter_tail_audit_forward_kernel_ns += (
        statistics.cpu_tail_filter_simd_forward_kernel_ns
    )
    _filter_tail_audit_backward_kernel_ns += (
        statistics.cpu_tail_filter_simd_backward_kernel_ns
    )


cdef void _record_shard_z_audit(
    uint64_t exception_begin,
    uint64_t exception_end,
    bint terminal,
    uint64_t local_target_visits,
    bint dynamic_applicable,
    uint64_t dynamic_z_offset,
    bint dynamic_z_restored,
) noexcept:
    """Record exact shard/Z accounting after a successful shard returns."""
    global _shard_z_audit_call_count
    global _shard_z_audit_first_shard_count
    global _shard_z_audit_terminal_shard_count
    global _shard_z_audit_exception_count
    global _shard_z_audit_local_target_visits
    global _shard_z_audit_dynamic_applicable_count
    global _shard_z_audit_nonzero_offset_count
    global _shard_z_audit_restored_count
    global _shard_z_audit_offset_sum
    global _shard_z_audit_offset_max
    _shard_z_audit_call_count += 1
    _shard_z_audit_first_shard_count += exception_begin == 0
    _shard_z_audit_terminal_shard_count += terminal
    _shard_z_audit_exception_count += exception_end - exception_begin
    _shard_z_audit_local_target_visits += local_target_visits
    _shard_z_audit_dynamic_applicable_count += dynamic_applicable
    if dynamic_z_offset != 0:
        _shard_z_audit_nonzero_offset_count += 1
        _shard_z_audit_offset_sum += dynamic_z_offset
        if dynamic_z_offset > _shard_z_audit_offset_max:
            _shard_z_audit_offset_max = dynamic_z_offset
    _shard_z_audit_restored_count += dynamic_z_restored


def _reset_filter_tail_audit_statistics_bound():
    """Reset opt-in full-gate filter SIMD and exact shard/Z accounting."""
    global _filter_tail_audit_eligible_rows
    global _filter_tail_audit_groups
    global _filter_tail_audit_rows
    global _filter_tail_audit_scalar_tail_rows
    global _filter_tail_audit_fallback_groups
    global _filter_tail_audit_fallback_rows
    global _filter_tail_audit_f3_survivor_rows
    global _filter_tail_audit_backward_groups
    global _filter_tail_audit_forward_kernel_ns
    global _filter_tail_audit_backward_kernel_ns
    global _shard_z_audit_call_count
    global _shard_z_audit_first_shard_count
    global _shard_z_audit_terminal_shard_count
    global _shard_z_audit_exception_count
    global _shard_z_audit_local_target_visits
    global _shard_z_audit_dynamic_applicable_count
    global _shard_z_audit_nonzero_offset_count
    global _shard_z_audit_restored_count
    global _shard_z_audit_offset_sum
    global _shard_z_audit_offset_max
    _filter_tail_audit_eligible_rows = 0
    _filter_tail_audit_groups = 0
    _filter_tail_audit_rows = 0
    _filter_tail_audit_scalar_tail_rows = 0
    _filter_tail_audit_fallback_groups = 0
    _filter_tail_audit_fallback_rows = 0
    _filter_tail_audit_f3_survivor_rows = 0
    _filter_tail_audit_backward_groups = 0
    _filter_tail_audit_forward_kernel_ns = 0
    _filter_tail_audit_backward_kernel_ns = 0
    _shard_z_audit_call_count = 0
    _shard_z_audit_first_shard_count = 0
    _shard_z_audit_terminal_shard_count = 0
    _shard_z_audit_exception_count = 0
    _shard_z_audit_local_target_visits = 0
    _shard_z_audit_dynamic_applicable_count = 0
    _shard_z_audit_nonzero_offset_count = 0
    _shard_z_audit_restored_count = 0
    _shard_z_audit_offset_sum = 0
    _shard_z_audit_offset_max = 0


def _filter_tail_audit_statistics_bound():
    """Snapshot opt-in full-gate SIMD and exact shard/Z accounting."""
    return {
        "cpu_tail_filter_simd": {
            "eligible_rows": _filter_tail_audit_eligible_rows,
            "groups": _filter_tail_audit_groups,
            "rows": _filter_tail_audit_rows,
            "scalar_tail_rows": _filter_tail_audit_scalar_tail_rows,
            "fallback_groups": _filter_tail_audit_fallback_groups,
            "fallback_rows": _filter_tail_audit_fallback_rows,
            "f3_survivor_rows": _filter_tail_audit_f3_survivor_rows,
            "backward_groups": _filter_tail_audit_backward_groups,
            "forward_kernel_ns": _filter_tail_audit_forward_kernel_ns,
            "backward_kernel_ns": _filter_tail_audit_backward_kernel_ns,
        },
        "sharded_z": {
            "call_count": _shard_z_audit_call_count,
            "first_shard_count": _shard_z_audit_first_shard_count,
            "terminal_shard_count": _shard_z_audit_terminal_shard_count,
            "exception_count": _shard_z_audit_exception_count,
            "local_target_visits": _shard_z_audit_local_target_visits,
            "dynamic_applicable_count": (
                _shard_z_audit_dynamic_applicable_count
            ),
            "nonzero_offset_count": _shard_z_audit_nonzero_offset_count,
            "restored_count": _shard_z_audit_restored_count,
            "offset_sum": _shard_z_audit_offset_sum,
            "offset_max": _shard_z_audit_offset_max,
        },
    }


def _sealed_continuation_statistics_bound(sealed_object):
    """Return immutable route counts and native timing evidence."""
    cdef _SealedPostfilterBatch sealed
    cdef plan7_continuation_journal_row journal_row
    cdef size_t row
    cdef size_t row_count
    cdef size_t cpu_required = 0
    cdef size_t no_regions = 0
    cdef size_t simple = 0
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_certificate *certificates
    cdef const plan7_continuation_journal_v3_exception *exceptions
    cdef const plan7_continuation_journal_v3_certificate *certificate
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef size_t certificate_index
    cdef size_t exception_index
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready or (
        sealed._journal_storage.shape[0] == 0
        and not sealed._direct_v3_source
    ):
        raise TypeError("sealed batch has no continuation journal")
    if sealed._direct_v3_source:
        if sealed._journal_v3 == NULL:
            raise RuntimeError("direct v3 packet is unavailable")
        base = <const uint8_t *> sealed._journal_v3
        certificates = <const plan7_continuation_journal_v3_certificate *> (
            base + sealed._journal_v3.certificates_offset
        )
        exceptions = <const plan7_continuation_journal_v3_exception *> (
            base + sealed._journal_v3.exceptions_offset
        )
        row_count = <size_t> sealed._journal_v3.source_domain_count
        for certificate_index in range(
            <size_t> sealed._journal_v3.certificate_count
        ):
            certificate = &certificates[certificate_index]
            no_regions += <size_t> certificate.no_region_count
        for exception_index in range(
            <size_t> sealed._journal_v3.exception_count
        ):
            exception = &exceptions[exception_index]
            if not (
                exception.payload_flags & PLAN7_CONTINUATION_V3_HAS_DOMAIN
            ):
                continue
            memcpy(
                &journal_row,
                exception.domain_record,
                sizeof(plan7_continuation_journal_row),
            )
            if (
                journal_row.domain_status != DOMAIN_OK
                or journal_row.domain_route == DOMAIN_CPU_REQUIRED
                or journal_row.has_own_scales
                or journal_row.uncertain_count != 0
                or journal_row.multidomain_count != 0
            ):
                cpu_required += 1
            elif journal_row.domain_route == DOMAIN_NO_REGIONS:
                no_regions += 1
            elif journal_row.domain_route == DOMAIN_SIMPLE:
                simple += 1
            else:
                cpu_required += 1
        if cpu_required + no_regions + simple != row_count:
            raise RuntimeError("direct v3 domain census is incomplete")
    else:
        row_count = (
            <size_t> sealed._journal_rows.shape[0]
            // sizeof(plan7_continuation_journal_row)
        )
        for row in range(row_count):
            memcpy(
                &journal_row,
                &sealed._journal_rows[
                    row * sizeof(plan7_continuation_journal_row)
                ],
                sizeof(plan7_continuation_journal_row),
            )
            if (
                journal_row.domain_status != DOMAIN_OK
                or journal_row.domain_route == DOMAIN_CPU_REQUIRED
                or journal_row.has_own_scales
                or journal_row.uncertain_count != 0
                or journal_row.multidomain_count != 0
            ):
                cpu_required += 1
            elif journal_row.domain_route == DOMAIN_NO_REGIONS:
                no_regions += 1
            elif journal_row.domain_route == DOMAIN_SIMPLE:
                simple += 1
            else:
                cpu_required += 1
    return {
        "row_count": row_count,
        "cpu_required_count": cpu_required,
        "no_region_count": no_regions,
        "simple_count": simple,
        "journal_bytes": sealed._journal_storage.shape[0],
        "dense_v2_retained_bytes": sealed._journal_storage.shape[0],
        "eliminated_v2_bytes": sealed._direct_v3_eliminated_v2_bytes,
        "guard_band_bits": sealed._journal_guard_bits,
        "compact_enabled": sealed._compact_domains_seam != NULL,
        "compact_simple_row_count": sealed._rescore_simple_row_count,
        "compact_device_result_count": (
            sealed._rescore_device_result_count
        ),
        "compact_cpu_required_count": (
            sealed._rescore_cpu_required_count
        ),
        "compact_certified_ga_count": (
            sealed._rescore_certified_ga_count
        ),
        "compact_numeric_fallback_count": (
            sealed._rescore_numeric_fallback_count
        ),
        "compact_cap_fallback_count": sealed._rescore_cap_fallback_count,
        "compact_global_cpu_fallback_count": (
            sealed._rescore_global_cpu_fallback_count
        ),
        "native_stage_timings": (
            _native_stage_timing_evidence(<tuple> sealed._native_stage_timings)
            if sealed._native_stage_timings is not None
            else None
        ),
        "sparse_journal_v3": {
            "enabled": sealed._journal_v3 != NULL,
            "packet_bytes": sealed._journal_v3_bytes,
            "planning_ns": sealed._journal_v3_planning_ns,
            "validation_ns": sealed._journal_v3_validation_ns,
            "source_kind": (
                "native_direct" if sealed._direct_v3_source else "dense_v2"
            ),
            "direct_staging_build_ns": sealed._direct_v3_staging_build_ns,
            "direct_source_validation_ns": (
                sealed._direct_v3_source_validation_ns
            ),
            "source_consumer_validation_ns": (
                sealed._source_consumer_validation_ns
            ),
            "direct_staging_bytes": sealed._direct_v3_staging_bytes,
            "eliminated_v2_bytes": (
                sealed._direct_v3_eliminated_v2_bytes
            ),
            "certificate_count": (
                sealed._journal_v3.certificate_count
                if sealed._journal_v3 != NULL else 0
            ),
            "exception_count": (
                sealed._journal_v3.exception_count
                if sealed._journal_v3 != NULL else 0
            ),
            "planner_source_scan_count": (
                sealed._journal_v3_source_scan_count
            ),
            "separate_decision_scan_count": (
                sealed._journal_v3_decision_scan_count
            ),
            "planning_scope": "once per sealed batch",
            "validation_scope": "once per sealed batch",
            "consumer_timing": (
                "native row-consumer wall_ns returned by "
                "CandidateBatch.search(return_telemetry=True); excludes "
                "adapter lock wait"
            ),
        },
    }


def _astra_tsv_rows_bound(TopHits hits):
    """Render Astra's bulk TSV rows without constructing hit/domain wrappers.

    This is a private output seam for Astra's already-thresholded ``TopHits``
    objects.  It deliberately mirrors ``astra.search.process_hits_to_file``
    byte for byte while walking the native HMMER result structures directly.
    The ordinary PyHMMER and plan7_gpu result APIs remain unchanged.
    """
    cdef P7_HIT *hit
    cdef P7_DOMAIN *domain
    cdef P7_ALIDISPLAY *alignment
    cdef uint64_t hit_index
    cdef int domain_index
    cdef double full_evalue
    cdef double conditional_evalue
    cdef double independent_evalue
    cdef object cog
    cdef object hit_name
    cdef list rows = []

    if type(hits) is not _pyhmmer.plan7.TopHits:
        raise TypeError("hits must be exactly astra_pyhmmer.plan7.TopHits")
    if hits._th == NULL:
        raise ValueError("TopHits state is unavailable")
    if hits._query is None:
        raise ValueError("TopHits query is unavailable")
    if hits._pli.long_targets:
        raise _AstraTSVRendererUnsupported(
            "native Astra TSV rendering is limited to ordinary searches"
        )
    if hits._th.N != 0 and not hits._th.is_sorted_by_sortkey:
        raise _AstraTSVRendererUnsupported(
            "native Astra TSV rendering requires sort-key ordered TopHits"
        )
    cog = hits._query.name

    for hit_index in range(hits._th.N):
        hit = hits._th.hit[hit_index]
        if hit == NULL:
            raise _AstraTSVRendererUnsupported(
                "TopHits order contains a null hit"
            )
        if not (hit.flags & p7_IS_INCLUDED):
            continue
        if hit.name == NULL:
            raise _AstraTSVRendererUnsupported(
                "included TopHits row has no target name"
            )
        if hit.ndom < 0 or (hit.ndom != 0 and hit.dcl == NULL):
            raise _AstraTSVRendererUnsupported(
                "included TopHits row has invalid domains"
            )
        hit_name = PyUnicode_FromString(hit.name)
        full_evalue = exp(hit.lnP) * hits._pli.Z
        for domain_index in range(hit.ndom):
            domain = &hit.dcl[domain_index]
            if not domain.is_reported:
                continue
            if domain.jenv < domain.ienv:
                raise _AstraTSVRendererUnsupported(
                    "reported domain has reversed envelope coordinates"
                )
            conditional_evalue = exp(domain.lnP) * hits._pli.domZ
            independent_evalue = exp(domain.lnP) * hits._pli.Z
            alignment = domain.ad
            if alignment != NULL and (
                alignment.hmmto < alignment.hmmfrom
                or alignment.sqto < alignment.sqfrom
            ):
                raise _AstraTSVRendererUnsupported(
                    "reported domain has reversed alignment coordinates"
                )
            # ``Alignment.__len__`` is ``hmm_to - hmm_from`` in PyHMMER,
            # so a one-position alignment is false-valued even though its
            # P7_ALIDISPLAY pointer is present.  Mirror Astra's public
            # ``if alignment`` formatting branch exactly: use the envelope
            # coordinates and leave the HMM coordinates empty in that case.
            if alignment == NULL or alignment.hmmto == alignment.hmmfrom:
                rows.append(
                    f"{hit_name}\t{cog}\t{hit.score:.2f}\t{full_evalue:.2e}\t"
                    f"{conditional_evalue:.2e}\t{independent_evalue:.2e}\t"
                    f"{domain.ienv}\t{domain.jenv}\t{domain.bitscore:.2f}\t"
                    f"{domain.ienv}\t{domain.jenv}\t\t\n"
                )
            else:
                rows.append(
                    f"{hit_name}\t{cog}\t{hit.score:.2f}\t{full_evalue:.2e}\t"
                    f"{conditional_evalue:.2e}\t{independent_evalue:.2e}\t"
                    f"{domain.ienv}\t{domain.jenv}\t{domain.bitscore:.2f}\t"
                    f"{alignment.sqfrom}\t{alignment.sqto}\t"
                    f"{alignment.hmmfrom}\t{alignment.hmmto}\n"
                )
    return "".join(rows)


def _sealed_generation_statistics_bound(sealed_object):
    """Return a defensive copy of optional versioned generation telemetry."""
    cdef _SealedPostfilterBatch sealed
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    return _telemetry_module.defensive_generation_statistics(
        sealed._generation_statistics
    )


def _sealed_resident_memory_bound(sealed_object):
    """Return exact owned buffer payload bytes for one opaque sealed batch.

    Journal subviews are reported but charged only through their single native
    allocation. Proven-shared source identity buffers and generation workspaces
    are excluded; any defensive identity-buffer copy is charged to the seal.
    """
    cdef _SealedPostfilterBatch sealed
    cdef object journal_bytes
    cdef object journal_subview_bytes
    cdef object postfilter_bytes = 0
    cdef object postfilter_offset_bytes = 0
    cdef object forward_bytes = 0
    cdef object forward_offset_bytes = 0
    cdef object special_offset_bytes = 0
    cdef object special_bytes = 0
    cdef object row_marker_bytes
    cdef object journal_sentinel_bytes = 0
    cdef object owned_host_bytes
    cdef object owned_residue_offsets_bytes
    cdef object owned_background_fingerprint_bytes
    cdef object excluded_residue_offsets_bytes
    cdef object excluded_background_fingerprint_bytes
    cdef object shared_identity_bytes
    cdef object sparse_journal_v3_bytes
    cdef object dense_replay_retained_bytes
    cdef object direct_v3_staging_retained_bytes = 0
    cdef dict evidence

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")

    # Convert every shape to Python int before arithmetic so even a corrupt or
    # future expanded layout cannot silently wrap a native aggregate counter.
    journal_bytes = int(sealed._journal_storage.shape[0])
    journal_subview_bytes = (
        int(sealed._postfilter_records.shape[0])
        + int(sealed._postfilter_offsets.shape[0]) * sizeof(uint64_t)
        + int(sealed._forward_records.shape[0])
        + int(sealed._forward_offsets.shape[0]) * sizeof(uint64_t)
        + int(sealed._special_offsets.shape[0]) * sizeof(uint64_t)
        + int(sealed._specials.shape[0]) * sizeof(float)
        + int(sealed._journal_profile_offsets.shape[0]) * sizeof(uint64_t)
        + int(sealed._journal_rows.shape[0])
        + int(sealed._journal_region_offsets.shape[0]) * sizeof(uint64_t)
        + int(sealed._journal_regions.shape[0])
        + int(sealed._journal_compact_row_offsets.shape[0])
            * sizeof(uint64_t)
        + int(sealed._journal_compact_results.shape[0])
        + int(sealed._journal_compact_trace_offsets.shape[0])
            * sizeof(uint64_t)
        + int(sealed._journal_compact_traces.shape[0])
        + int(sealed._journal_compact_null2.shape[0]) * sizeof(float)
    )
    if journal_bytes == 0:
        postfilter_bytes = int(sealed._postfilter_records.shape[0])
        postfilter_offset_bytes = (
            int(sealed._postfilter_offsets.shape[0]) * sizeof(uint64_t)
        )
        forward_bytes = int(sealed._forward_records.shape[0])
        forward_offset_bytes = (
            int(sealed._forward_offsets.shape[0]) * sizeof(uint64_t)
        )
        special_offset_bytes = (
            int(sealed._special_offsets.shape[0]) * sizeof(uint64_t)
        )
        special_bytes = (
            int(sealed._specials.shape[0]) * sizeof(float)
        )
        journal_sentinel_bytes = (
            int(sealed._journal_profile_offsets.shape[0])
                * sizeof(uint64_t)
            + int(sealed._journal_region_offsets.shape[0])
                * sizeof(uint64_t)
            + int(sealed._journal_compact_row_offsets.shape[0])
                * sizeof(uint64_t)
            + int(sealed._journal_compact_trace_offsets.shape[0])
                * sizeof(uint64_t)
        )
    row_marker_bytes = int(sealed._row_has_external.shape[0])
    owned_residue_offsets_bytes = sealed._owned_residue_offsets_bytes
    owned_background_fingerprint_bytes = (
        sealed._owned_background_fingerprint_bytes
    )
    excluded_residue_offsets_bytes = sealed._excluded_residue_offsets_bytes
    excluded_background_fingerprint_bytes = (
        sealed._excluded_background_fingerprint_bytes
    )
    for value in (
        owned_residue_offsets_bytes,
        owned_background_fingerprint_bytes,
        excluded_residue_offsets_bytes,
        excluded_background_fingerprint_bytes,
    ):
        if type(value) is not int or value < 0:
            raise RuntimeError("sealed identity-buffer accounting is invalid")
    owned_host_bytes = (
        journal_bytes
        + int(sealed._journal_v3_bytes)
        + postfilter_bytes
        + postfilter_offset_bytes
        + forward_bytes
        + forward_offset_bytes
        + special_offset_bytes
        + special_bytes
        + row_marker_bytes
        + journal_sentinel_bytes
        + owned_residue_offsets_bytes
        + owned_background_fingerprint_bytes
    )
    shared_identity_bytes = (
        excluded_residue_offsets_bytes
        + excluded_background_fingerprint_bytes
    )
    sparse_journal_v3_bytes = int(sealed._journal_v3_bytes)
    dense_replay_retained_bytes = (
        int(sealed._journal_storage.shape[0])
        + int(sealed._postfilter_records.shape[0])
        + int(sealed._forward_records.shape[0])
        + int(sealed._specials.shape[0]) * sizeof(float)
        + int(sealed._journal_rows.shape[0])
        + int(sealed._journal_regions.shape[0])
        + int(sealed._journal_compact_results.shape[0])
        + int(sealed._journal_compact_traces.shape[0])
        + int(sealed._journal_compact_null2.shape[0]) * sizeof(float)
    )
    if sealed._direct_v3_source:
        direct_v3_staging_retained_bytes = (
            int(sealed._source_identity_tokens.shape[0]) * sizeof(uint64_t)
            + int(sealed._source_profile_fingerprints.shape[0])
            + int(sealed._source_sequence_fingerprint.shape[0])
            + int(sealed._direct_v3_decisions.shape[0])
        )
        if direct_v3_staging_retained_bytes:
            raise RuntimeError("direct v3 staging identity remains retained")
    evidence = {
        "schema_version": 1,
        "owned_host_bytes": owned_host_bytes,
        "owned_device_bytes": 0,
        "journal_allocation_bytes": journal_bytes,
        "journal_subview_bytes": journal_subview_bytes,
        "postfilter_records_bytes": postfilter_bytes,
        "postfilter_offsets_bytes": postfilter_offset_bytes,
        "forward_records_bytes": forward_bytes,
        "forward_offsets_bytes": forward_offset_bytes,
        "special_offsets_bytes": special_offset_bytes,
        "specials_bytes": special_bytes,
        "row_markers_bytes": row_marker_bytes,
        "journal_sentinel_bytes": journal_sentinel_bytes,
        "owned_residue_offsets_bytes": owned_residue_offsets_bytes,
        "owned_background_fingerprint_bytes": (
            owned_background_fingerprint_bytes
        ),
        "excluded_residue_offsets_bytes": excluded_residue_offsets_bytes,
        "excluded_background_fingerprint_bytes": (
            excluded_background_fingerprint_bytes
        ),
        "excluded_shared_identity_bytes": shared_identity_bytes,
        "direct_v3_source": bool(sealed._direct_v3_source),
        "dense_replay_retained_bytes": dense_replay_retained_bytes,
        "direct_v3_staging_retained_bytes": (
            direct_v3_staging_retained_bytes
        ),
        "eliminated_v2_bytes": sealed._direct_v3_eliminated_v2_bytes,
    }
    if sparse_journal_v3_bytes:
        evidence["sparse_journal_v3_bytes"] = sparse_journal_v3_bytes
    return evidence


def _search_hmm_candidates(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
):
    """Internal primitive consuming one sorted native-endian uint32 CSR row.

    Structural validation cannot establish candidate provenance.  Callers
    outside the package must not use this function directly.  The candidate
    buffer and all PyHMMER inputs must remain alive and unmodified until the
    call returns; the provenance-bound adapter is responsible for enforcing
    that ownership and synchronization contract.
    """
    cdef uint64_t* residue_offsets = NULL
    cdef size_t target

    _validate_inputs(pipeline, query, optimized_profile, sequences, True)
    _validate_candidate_indexes(sequences, candidate_indexes)

    residue_offsets = <uint64_t*> malloc(
        (sequences._length + 1) * sizeof(uint64_t)
    )
    if residue_offsets == NULL:
        raise MemoryError("target residue-prefix allocation failed")
    residue_offsets[0] = 0
    for target in range(sequences._length):
        residue_offsets[target + 1] = (
            residue_offsets[target] + sequences._refs[target].n
        )
    try:
        return _search_validated(
            pipeline,
            query,
            optimized_profile,
            sequences,
            candidate_indexes,
            residue_offsets,
        )
    finally:
        free(residue_offsets)


def _search_hmm_candidates_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
    const uint64_t[::1] residue_offsets,
):
    """Consume one provenance-bound CSR row and cumulative residue offsets.

    This is a package-private fast path. ``SequenceBatch`` creates both
    buffers from the same concealed target snapshot and keeps them immutable;
    callers outside the package cannot establish that provenance safely.
    """
    cdef size_t cursor
    cdef size_t target
    cdef size_t target_end
    cdef size_t previous_end = 0

    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    _validate_candidate_indexes(sequences, candidate_indexes)
    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")

    # Validate every prefix entry the sparse loop will dereference. Full
    # target-by-target validation belongs to SequenceBatch construction.
    for cursor in range(<size_t> candidate_indexes.shape[0]):
        target = candidate_indexes[cursor]
        target_end = target + 1
        if residue_offsets[target_end] < residue_offsets[previous_end]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target_end] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")
        previous_end = target_end
    if residue_offsets[sequences._length] < residue_offsets[previous_end]:
        raise ValueError("target residue prefix is not monotone")
    if sequences._length != 0 and (
        residue_offsets[sequences._length]
        - residue_offsets[sequences._length - 1]
        != <uint64_t> sequences._refs[sequences._length - 1].n
    ):
        raise ValueError("target residue prefix differs from final target length")

    return _search_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        candidate_indexes,
        &residue_offsets[0],
    )


def _search_hmm_bias_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
    const uint64_t[::1] residue_offsets,
):
    """Consume one provenance-bound row of exact CUDA bias records.

    The caller must hold the same exclusive ``Pipeline`` and immutable target
    ownership used by ``_search_hmm_candidates_bound``. Records absent from
    the row are definite F1 rejects. CPU-required records run the ordinary
    HMMER pipeline; direct records enter the patched HMMER continuation seam,
    which rechecks F1, the current bias setting, and F2 before CPU Viterbi.
    """
    cdef bint has_direct
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    has_direct = _validate_bias_records(
        optimized_profile, sequences, bias_records
    )
    _validate_bias_residue_offsets(sequences, bias_records, residue_offsets)

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct bias records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

    return _search_bias_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        bias_records,
        &residue_offsets[0],
        filter_scores_seam,
    )


def _search_hmm_postfilter_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] residue_offsets,
):
    """Consume one provenance-bound row of exact CUDA post-filter records."""
    cdef bint has_direct
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    has_direct = _validate_postfilter_records(
        optimized_profile, sequences, postfilter_records
    )
    _validate_postfilter_residue_offsets(
        sequences, postfilter_records, residue_offsets
    )

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct post-filter records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

    return _search_postfilter_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        postfilter_records,
        &residue_offsets[0],
        filter_scores_seam,
    )


def _search_hmm_postfilter_forward_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint64_t[::1] residue_offsets,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    bint generation_bias_filter,
):
    """Consume one provenance-bound post-filter row augmented by Forward.

    The original post-filter row remains authoritative. If the live pipeline
    options differ bit-for-bit from generation, the complete row takes the
    existing exact continuation and computes Forward on the CPU.
    """
    cdef bint has_direct
    cdef bint has_external
    cdef _double_bits live_f2
    cdef _double_bits live_f3
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL
    cdef _pipeline_from_filter_and_forward_scores_f forward_scores_seam = NULL

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    has_direct = _validate_postfilter_records(
        optimized_profile, sequences, postfilter_records
    )
    _validate_postfilter_residue_offsets(
        sequences, postfilter_records, residue_offsets
    )
    has_external = _validate_forward_augmentation(
        sequences,
        postfilter_records,
        forward_records,
        special_offsets,
        specials,
    )

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct post-filter records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

    live_f2.value = pipeline._pli.F2
    live_f3.value = pipeline._pli.F3
    if (
        not has_external
        or live_f2.bits != generation_f2_bits
        or live_f3.bits != generation_f3_bits
        or pipeline._pli.do_biasfilter != generation_bias_filter
    ):
        return _search_postfilter_validated(
            pipeline,
            query,
            optimized_profile,
            sequences,
            postfilter_records,
            &residue_offsets[0],
            filter_scores_seam,
        )

    forward_scores_seam = _cached_filter_and_forward_scores_seam()
    if forward_scores_seam == NULL:
        return _search_postfilter_validated(
            pipeline,
            query,
            optimized_profile,
            sequences,
            postfilter_records,
            &residue_offsets[0],
            filter_scores_seam,
        )

    return _search_postfilter_forward_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        postfilter_records,
        forward_records,
        special_offsets,
        specials,
        &residue_offsets[0],
        filter_scores_seam,
        forward_scores_seam,
    )


# Phase 1A semantic-state oracle.  This deliberately lives in the private
# extension: PyHMMER's public pickle state serializes dormant short-mode
# fields, while the C structs contain pointers, capacity, padding, and
# reusable workspaces that are not semantic state.
SEMANTIC_STATE_SCHEMA_VERSION = 1

cdef enum _semantic_field_tag:
    _SEM_BOOL = 1
    _SEM_I32 = 2
    _SEM_I64 = 3
    _SEM_U8 = 4
    _SEM_U32 = 5
    _SEM_U64 = 6
    _SEM_F32_BITS = 7
    _SEM_F64_BITS = 8
    _SEM_BYTES = 9
    _SEM_NULLABLE_BYTES = 10


cdef inline uint32_t _semantic_float_bits(float value) noexcept:
    cdef uint32_t bits
    memcpy(&bits, &value, sizeof(bits))
    return bits


cdef inline uint64_t _semantic_double_bits(double value) noexcept:
    cdef uint64_t bits
    memcpy(&bits, &value, sizeof(bits))
    return bits


cdef void _semantic_field(
    bytearray output,
    bytes name,
    uint8_t tag,
    bytes payload,
) except *:
    if len(name) > 65535:
        raise ValueError("semantic field name is too long")
    output.extend(_struct.pack("<HBQ", len(name), tag, len(payload)))
    output.extend(name)
    output.extend(payload)


cdef inline void _semantic_bool(
    bytearray output,
    bytes name,
    int value,
) except *:
    if value != 0 and value != 1:
        raise ValueError(f"{name.decode('ascii')} is not boolean")
    _semantic_field(output, name, _SEM_BOOL, _struct.pack("<B", value))


cdef inline void _semantic_i32(
    bytearray output,
    bytes name,
    int value,
) except *:
    _semantic_field(output, name, _SEM_I32, _struct.pack("<i", value))


cdef inline void _semantic_i64(
    bytearray output,
    bytes name,
    int64_t value,
) except *:
    _semantic_field(output, name, _SEM_I64, _struct.pack("<q", value))


cdef inline void _semantic_u8(
    bytearray output,
    bytes name,
    uint8_t value,
) except *:
    _semantic_field(output, name, _SEM_U8, _struct.pack("<B", value))


cdef inline void _semantic_u32(
    bytearray output,
    bytes name,
    uint32_t value,
) except *:
    _semantic_field(output, name, _SEM_U32, _struct.pack("<I", value))


cdef inline void _semantic_u64(
    bytearray output,
    bytes name,
    uint64_t value,
) except *:
    _semantic_field(output, name, _SEM_U64, _struct.pack("<Q", value))


cdef inline void _semantic_f32(
    bytearray output,
    bytes name,
    float value,
) except *:
    _semantic_field(
        output,
        name,
        _SEM_F32_BITS,
        _struct.pack("<I", _semantic_float_bits(value)),
    )


cdef inline void _semantic_f64(
    bytearray output,
    bytes name,
    double value,
) except *:
    _semantic_field(
        output,
        name,
        _SEM_F64_BITS,
        _struct.pack("<Q", _semantic_double_bits(value)),
    )


cdef inline void _semantic_bytes(
    bytearray output,
    bytes name,
    bytes value,
) except *:
    _semantic_field(output, name, _SEM_BYTES, value)


cdef inline void _semantic_nullable_bytes(
    bytearray output,
    bytes name,
    object value,
) except *:
    cdef bytes data
    if value is None:
        _semantic_field(output, name, _SEM_NULLABLE_BYTES, b"\x00")
        return
    if not isinstance(value, bytes):
        raise TypeError(f"{name.decode('ascii')} must be bytes or None")
    data = value
    _semantic_field(output, name, _SEM_NULLABLE_BYTES, b"\x01" + data)


cdef inline object _semantic_cstring(const char *value):
    if value == NULL:
        return None
    return PyBytes_FromStringAndSize(value, strlen(value))


cdef inline object _semantic_fixed_string(const char *value, int length):
    if value == NULL:
        return None
    if length < 0:
        raise ValueError("negative semantic string length")
    return PyBytes_FromStringAndSize(value, length)


cdef void _semantic_encode_pipeline_scalars(
    bytearray output,
    const P7_PIPELINE *pli,
    bytes prefix,
) except *:
    if pli == NULL:
        raise ValueError("pipeline state is unavailable")
    if pli.mode != p7_SEARCH_SEQS or pli.long_targets:
        raise ValueError(
            "semantic fingerprint requires a short protein sequence-search pipeline"
        )

    _semantic_bool(output, prefix + b".by_E", pli.by_E)
    _semantic_f64(output, prefix + b".E", pli.E)
    _semantic_f64(output, prefix + b".T", pli.T)
    _semantic_bool(output, prefix + b".dom_by_E", pli.dom_by_E)
    _semantic_f64(output, prefix + b".domE", pli.domE)
    _semantic_f64(output, prefix + b".domT", pli.domT)
    _semantic_i32(output, prefix + b".use_bit_cutoffs", pli.use_bit_cutoffs)
    _semantic_bool(output, prefix + b".inc_by_E", pli.inc_by_E)
    _semantic_f64(output, prefix + b".incE", pli.incE)
    _semantic_f64(output, prefix + b".incT", pli.incT)
    _semantic_bool(output, prefix + b".incdom_by_E", pli.incdom_by_E)
    _semantic_f64(output, prefix + b".incdomE", pli.incdomE)
    _semantic_f64(output, prefix + b".incdomT", pli.incdomT)

    _semantic_f64(output, prefix + b".Z", pli.Z)
    _semantic_f64(output, prefix + b".domZ", pli.domZ)
    _semantic_i32(output, prefix + b".Z_setby", pli.Z_setby)
    _semantic_i32(output, prefix + b".domZ_setby", pli.domZ_setby)

    _semantic_bool(output, prefix + b".do_max", pli.do_max)
    _semantic_f64(output, prefix + b".F1", pli.F1)
    _semantic_f64(output, prefix + b".F2", pli.F2)
    _semantic_f64(output, prefix + b".F3", pli.F3)
    _semantic_i32(output, prefix + b".B1", pli.B1)
    _semantic_i32(output, prefix + b".B2", pli.B2)
    _semantic_i32(output, prefix + b".B3", pli.B3)
    _semantic_bool(output, prefix + b".do_biasfilter", pli.do_biasfilter)
    _semantic_bool(output, prefix + b".do_null2", pli.do_null2)
    _semantic_bool(output, prefix + b".do_reseeding", pli.do_reseeding)
    _semantic_bool(
        output,
        prefix + b".do_alignment_score_calc",
        pli.do_alignment_score_calc,
    )

    _semantic_u64(output, prefix + b".nmodels", pli.nmodels)
    _semantic_u64(output, prefix + b".nseqs", pli.nseqs)
    _semantic_u64(output, prefix + b".nres", pli.nres)
    _semantic_u64(output, prefix + b".nnodes", pli.nnodes)
    _semantic_u64(output, prefix + b".n_past_msv", pli.n_past_msv)
    _semantic_u64(output, prefix + b".n_past_bias", pli.n_past_bias)
    _semantic_u64(output, prefix + b".n_past_vit", pli.n_past_vit)
    _semantic_u64(output, prefix + b".n_past_fwd", pli.n_past_fwd)
    _semantic_u64(output, prefix + b".pos_past_msv", pli.pos_past_msv)
    _semantic_u64(output, prefix + b".pos_past_bias", pli.pos_past_bias)
    _semantic_u64(output, prefix + b".pos_past_vit", pli.pos_past_vit)
    _semantic_u64(output, prefix + b".pos_past_fwd", pli.pos_past_fwd)

    _semantic_i32(output, prefix + b".mode", pli.mode)
    _semantic_bool(output, prefix + b".long_targets", pli.long_targets)
    # W is uninitialized in a newly allocated upstream P7_PIPELINE. NewModel
    # defines it before any search and clear() defines it as zero. Before the
    # first model it cannot affect execution because NewModel overwrites it.
    _semantic_bool(output, prefix + b".W_present", pli.nmodels != 0)
    if pli.nmodels != 0:
        _semantic_i32(output, prefix + b".W", pli.W)
    _semantic_bool(output, prefix + b".show_accessions", pli.show_accessions)
    _semantic_bool(output, prefix + b".show_alignments", pli.show_alignments)

    # Deliberately never read n_output, pos_output, strands, or block_length:
    # upstream allocation leaves these dormant short-mode fields undefined.


cdef void _semantic_encode_rng(
    bytearray output,
    const ESL_RANDOMNESS *rng,
    bytes prefix,
) except *:
    cdef int i
    if rng == NULL:
        raise ValueError("pipeline RNG is unavailable")
    _semantic_i32(output, prefix + b".type", rng.type)
    _semantic_u32(output, prefix + b".seed", rng.seed)
    if rng.type == eslRND_FAST:
        # mt[] is uninitialized for the fast LCG and must never be read.
        _semantic_u32(output, prefix + b".x", rng.x)
    elif rng.type == eslRND_MERSENNE:
        # x is irrelevant to MT state; mti and the initialized table are exact.
        _semantic_i32(output, prefix + b".mti", rng.mti)
        for i in range(624):
            _semantic_u32(
                output,
                prefix + b".mt[" + str(i).encode("ascii") + b"]",
                rng.mt[i],
            )
    else:
        raise ValueError("pipeline RNG has an unsupported type")


cdef void _semantic_encode_domaindef(
    bytearray output,
    const P7_DOMAINDEF *ddef,
    const ESL_RANDOMNESS *rng,
    bytes prefix,
) except *:
    if ddef == NULL:
        raise ValueError("pipeline domain-definition state is unavailable")
    if ddef.r != rng:
        raise ValueError("pipeline and domain-definition RNG ownership differs")
    if (
        ddef.mocc == NULL
        or ddef.btot == NULL
        or ddef.etot == NULL
        or ddef.n2sc == NULL
        or ddef.sp == NULL
        or ddef.tr == NULL
        or ddef.gtr == NULL
        or ddef.dcl == NULL
    ):
        raise ValueError("pipeline domain-definition workspace is incomplete")
    if (
        ddef.L != 0
        or ddef.ndom != 0
        or ddef.nexpected != 0.0
        or ddef.nregions != 0
        or ddef.nclustered != 0
        or ddef.noverlaps != 0
        or ddef.nenvelopes != 0
    ):
        raise ValueError(
            "semantic fingerprint requires successful reusable domain state"
        )
    if (
        ddef.sp.nsamples != 0
        or ddef.sp.n != 0
        or ddef.sp.nc != 0
        or ddef.sp.nsigc != 0
        or ddef.tr.N != 0
        or ddef.tr.M != 0
        or ddef.tr.L != 0
        or ddef.tr.ndom != 0
        or ddef.gtr.N != 0
        or ddef.gtr.M != 0
        or ddef.gtr.L != 0
        or ddef.gtr.ndom != 0
    ):
        raise ValueError(
            "semantic fingerprint requires reusable domain child state"
        )

    _semantic_f32(output, prefix + b".rt1", ddef.rt1)
    _semantic_f32(output, prefix + b".rt2", ddef.rt2)
    _semantic_f32(output, prefix + b".rt3", ddef.rt3)
    _semantic_i32(output, prefix + b".nsamples", ddef.nsamples)
    _semantic_f32(output, prefix + b".min_overlap", ddef.min_overlap)
    _semantic_bool(output, prefix + b".of_smaller", ddef.of_smaller)
    _semantic_i32(output, prefix + b".max_diagdiff", ddef.max_diagdiff)
    _semantic_f32(output, prefix + b".min_posterior", ddef.min_posterior)
    _semantic_f32(output, prefix + b".min_endpointp", ddef.min_endpointp)
    _semantic_bool(output, prefix + b".do_reseeding", ddef.do_reseeding)
    _semantic_i32(output, prefix + b".reusable.L", ddef.L)
    _semantic_i32(output, prefix + b".reusable.ndom", ddef.ndom)
    _semantic_f32(output, prefix + b".reusable.nexpected", ddef.nexpected)
    _semantic_i32(output, prefix + b".reusable.nregions", ddef.nregions)
    _semantic_i32(output, prefix + b".reusable.nclustered", ddef.nclustered)
    _semantic_i32(output, prefix + b".reusable.noverlaps", ddef.noverlaps)
    _semantic_i32(output, prefix + b".reusable.nenvelopes", ddef.nenvelopes)
    _semantic_i32(output, prefix + b".reusable.sp.nsamples", ddef.sp.nsamples)
    _semantic_i32(output, prefix + b".reusable.sp.n", ddef.sp.n)
    _semantic_i32(output, prefix + b".reusable.sp.nc", ddef.sp.nc)
    _semantic_i32(output, prefix + b".reusable.sp.nsigc", ddef.sp.nsigc)
    _semantic_i32(output, prefix + b".reusable.tr.N", ddef.tr.N)
    _semantic_i32(output, prefix + b".reusable.tr.M", ddef.tr.M)
    _semantic_i32(output, prefix + b".reusable.tr.L", ddef.tr.L)
    _semantic_i32(output, prefix + b".reusable.tr.ndom", ddef.tr.ndom)
    _semantic_i32(output, prefix + b".reusable.gtr.N", ddef.gtr.N)
    _semantic_i32(output, prefix + b".reusable.gtr.M", ddef.gtr.M)
    _semantic_i32(output, prefix + b".reusable.gtr.L", ddef.gtr.L)
    _semantic_i32(output, prefix + b".reusable.gtr.ndom", ddef.gtr.ndom)


cdef void _semantic_encode_reusable_omx(
    bytearray output,
    const P7_OMX *matrix,
    bytes prefix,
) except *:
    if matrix == NULL:
        raise ValueError("pipeline DP workspace is unavailable")
    if (
        matrix.dpf == NULL
        or matrix.dpw == NULL
        or matrix.dpb == NULL
        or matrix.dp_mem == NULL
        or matrix.xmx == NULL
        or matrix.x_mem == NULL
    ):
        raise ValueError("pipeline DP backing storage is incomplete")
    if (
        matrix.M != 0
        or matrix.L != 0
        or matrix.totscale != 0.0
        or matrix.has_own_scales != 1
    ):
        raise ValueError("semantic fingerprint requires reusable DP state")
    _semantic_i32(output, prefix + b".M", matrix.M)
    _semantic_i32(output, prefix + b".L", matrix.L)
    _semantic_f32(output, prefix + b".totscale", matrix.totscale)
    _semantic_bool(
        output,
        prefix + b".has_own_scales",
        matrix.has_own_scales,
    )


cdef void _semantic_encode_background(
    bytearray output,
    const P7_BG *bg,
    bytes prefix,
) except *:
    cdef int i
    if bg == NULL or bg.abc == NULL or bg.f == NULL:
        raise ValueError("pipeline background state is unavailable")
    if bg.abc.type != eslAMINO:
        raise ValueError("semantic fingerprint requires a protein alphabet")
    _semantic_i32(output, prefix + b".alphabet.type", bg.abc.type)
    _semantic_i32(output, prefix + b".alphabet.K", bg.abc.K)
    _semantic_i32(output, prefix + b".alphabet.Kp", bg.abc.Kp)
    _semantic_f32(output, prefix + b".p1", bg.p1)
    _semantic_f32(output, prefix + b".omega", bg.omega)
    for i in range(bg.abc.K):
        _semantic_f32(
            output,
            prefix + b".f[" + str(i).encode("ascii") + b"]",
            bg.f[i],
        )

    # P7_PIPELINE has no provenance bit recording whether the most recent
    # p7_pli_NewModel() call initialized fhmm.  The live do_biasfilter option
    # is mutable independently, so neither it nor nmodels proves that these
    # numeric buffers are initialized.  Do not inspect any fhmm storage until
    # the audit executor can supply explicit NewModel provenance.
    _semantic_bytes(
        output,
        prefix + b".fhmm.state",
        b"excluded-unproven",
    )


cdef void _semantic_encode_oprofile(
    bytearray output,
    const P7_OPROFILE *om,
    bytes prefix,
) except *:
    cdef int x
    cdef int y
    if om == NULL or om.abc == NULL:
        raise ValueError("optimized-profile state is unavailable")
    if om.abc.type != eslAMINO:
        raise ValueError("semantic fingerprint requires a protein profile")

    _semantic_i32(output, prefix + b".alphabet.type", om.abc.type)
    _semantic_i32(output, prefix + b".alphabet.K", om.abc.K)
    _semantic_i32(output, prefix + b".alphabet.Kp", om.abc.Kp)
    _semantic_nullable_bytes(output, prefix + b".name", _semantic_cstring(om.name))
    _semantic_nullable_bytes(output, prefix + b".acc", _semantic_cstring(om.acc))
    _semantic_i32(output, prefix + b".M", om.M)
    _semantic_i32(output, prefix + b".L", om.L)
    _semantic_i32(output, prefix + b".max_length", om.max_length)
    _semantic_i32(output, prefix + b".mode", om.mode)
    _semantic_f32(output, prefix + b".nj", om.nj)

    _semantic_u8(output, prefix + b".tbm_b", om.tbm_b)
    _semantic_u8(output, prefix + b".tec_b", om.tec_b)
    _semantic_u8(output, prefix + b".tjb_b", om.tjb_b)
    _semantic_f32(output, prefix + b".scale_b", om.scale_b)
    _semantic_u8(output, prefix + b".base_b", om.base_b)
    _semantic_u8(output, prefix + b".bias_b", om.bias_b)
    _semantic_f32(output, prefix + b".scale_w", om.scale_w)
    _semantic_i32(output, prefix + b".base_w", om.base_w)
    _semantic_i32(output, prefix + b".ddbound_w", om.ddbound_w)
    _semantic_f32(output, prefix + b".ncj_roundoff", om.ncj_roundoff)
    for x in range(4):
        for y in range(2):
            _semantic_i32(
                output,
                prefix
                + b".xw["
                + str(x).encode("ascii")
                + b"]["
                + str(y).encode("ascii")
                + b"]",
                om.xw[x][y],
            )
            _semantic_field(
                output,
                prefix
                + b".xf["
                + str(x).encode("ascii")
                + b"]["
                + str(y).encode("ascii")
                + b"]",
                _SEM_F32_BITS,
                _struct.pack(
                    "<I", plan7_semantic_oprofile_xf_bits(om, x, y)
                ),
            )


cdef void _semantic_encode_query_identity(
    bytearray output,
    object query,
    bytes prefix,
) except *:
    cdef bytes kind
    cdef object name
    cdef object accession
    cdef int model_length
    cdef HMM hmm
    cdef Profile profile
    cdef OptimizedProfile optimized
    if isinstance(query, _pyhmmer.plan7.HMM):
        kind = b"HMM"
        hmm = query
        name = _semantic_cstring(hmm._hmm.name)
        accession = _semantic_cstring(hmm._hmm.acc)
        model_length = hmm._hmm.M
    elif isinstance(query, _pyhmmer.plan7.Profile):
        kind = b"Profile"
        profile = query
        name = _semantic_cstring(profile._gm.name)
        accession = _semantic_cstring(profile._gm.acc)
        model_length = profile._gm.M
    elif isinstance(query, _pyhmmer.plan7.OptimizedProfile):
        kind = b"OptimizedProfile"
        optimized = query
        name = _semantic_cstring(optimized._om.name)
        accession = _semantic_cstring(optimized._om.acc)
        model_length = optimized._om.M
    else:
        raise TypeError(
            "semantic TopHits fingerprint requires an HMM/profile query"
        )
    _semantic_bytes(output, prefix + b".kind", kind)
    _semantic_nullable_bytes(output, prefix + b".name", name)
    _semantic_nullable_bytes(output, prefix + b".accession", accession)
    _semantic_i32(output, prefix + b".M", model_length)


cdef bytes _semantic_checked_profile_identity_fingerprint(
    OptimizedProfile optimized_profile,
    str label,
):
    cdef object fingerprint
    if optimized_profile._om == NULL or optimized_profile._om.abc == NULL:
        raise ValueError(f"{label} optimized profile state is unavailable")
    fingerprint = _fingerprint_module.optimized_profile_fingerprint(
        optimized_profile
    )
    if type(fingerprint) is not bytes or len(fingerprint) != 32:
        raise ValueError(
            f"{label} optimized-profile identity fingerprint is invalid"
        )
    return fingerprint


cdef void _semantic_validate_tophits_profile_binding(
    TopHits hits,
    OptimizedProfile optimized_profile,
    bytes expected_fingerprint,
    str label,
) except *:
    cdef object query = hits._query
    cdef object query_name
    cdef object query_accession
    cdef object query_description
    cdef bytes query_fingerprint
    cdef int query_model_length
    cdef int query_alphabet_type
    cdef int query_alphabet_k
    cdef int query_alphabet_kp
    cdef HMM hmm
    cdef Profile profile
    cdef OptimizedProfile optimized

    if isinstance(query, _pyhmmer.plan7.OptimizedProfile):
        optimized = query
        query_fingerprint = _semantic_checked_profile_identity_fingerprint(
            optimized,
            label + " TopHits query",
        )
        if query_fingerprint != expected_fingerprint:
            raise ValueError(
                f"{label} TopHits query identity differs from the supplied "
                "optimized profile"
            )
        return

    if isinstance(query, _pyhmmer.plan7.HMM):
        hmm = query
        if hmm._hmm == NULL or hmm._hmm.abc == NULL:
            raise ValueError(f"{label} TopHits HMM query state is unavailable")
        query_name = _semantic_cstring(hmm._hmm.name)
        query_accession = _semantic_cstring(hmm._hmm.acc)
        query_description = _semantic_cstring(hmm._hmm.desc)
        query_model_length = hmm._hmm.M
        query_alphabet_type = hmm._hmm.abc.type
        query_alphabet_k = hmm._hmm.abc.K
        query_alphabet_kp = hmm._hmm.abc.Kp
    elif isinstance(query, _pyhmmer.plan7.Profile):
        profile = query
        if profile._gm == NULL or profile._gm.abc == NULL:
            raise ValueError(
                f"{label} TopHits profile query state is unavailable"
            )
        query_name = _semantic_cstring(profile._gm.name)
        query_accession = _semantic_cstring(profile._gm.acc)
        query_description = _semantic_cstring(profile._gm.desc)
        query_model_length = profile._gm.M
        query_alphabet_type = profile._gm.abc.type
        query_alphabet_k = profile._gm.abc.K
        query_alphabet_kp = profile._gm.abc.Kp
    else:
        raise TypeError(
            f"{label} TopHits query must be an HMM, Profile, or "
            "OptimizedProfile"
        )

    if (
        query_name != _semantic_cstring(optimized_profile._om.name)
        or query_accession != _semantic_cstring(optimized_profile._om.acc)
        or query_description != _semantic_cstring(optimized_profile._om.desc)
        or query_model_length != optimized_profile._om.M
        or query_alphabet_type != optimized_profile._om.abc.type
        or query_alphabet_k != optimized_profile._om.abc.K
        or query_alphabet_kp != optimized_profile._om.abc.Kp
    ):
        raise ValueError(
            f"{label} TopHits query metadata differs from the supplied "
            "optimized profile"
        )


cdef void _semantic_encode_alidisplay(
    bytearray output,
    const P7_ALIDISPLAY *ad,
    bytes prefix,
) except *:
    if ad == NULL:
        _semantic_bool(output, prefix + b".present", False)
        return
    if ad.N < 0:
        raise ValueError("alignment display has a negative length")
    if ad.ntseq != NULL:
        raise ValueError(
            "short protein alignment display unexpectedly contains ntseq"
        )
    _semantic_bool(output, prefix + b".present", True)
    _semantic_i32(output, prefix + b".N", ad.N)
    _semantic_nullable_bytes(
        output, prefix + b".rfline", _semantic_fixed_string(ad.rfline, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".mmline", _semantic_fixed_string(ad.mmline, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".csline", _semantic_fixed_string(ad.csline, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".model", _semantic_fixed_string(ad.model, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".mline", _semantic_fixed_string(ad.mline, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".aseq", _semantic_fixed_string(ad.aseq, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".ppline", _semantic_fixed_string(ad.ppline, ad.N)
    )
    _semantic_nullable_bytes(
        output, prefix + b".hmmname", _semantic_cstring(ad.hmmname)
    )
    _semantic_nullable_bytes(
        output, prefix + b".hmmacc", _semantic_cstring(ad.hmmacc)
    )
    _semantic_nullable_bytes(
        output, prefix + b".hmmdesc", _semantic_cstring(ad.hmmdesc)
    )
    _semantic_i32(output, prefix + b".hmmfrom", ad.hmmfrom)
    _semantic_i32(output, prefix + b".hmmto", ad.hmmto)
    _semantic_i32(output, prefix + b".M", ad.M)
    _semantic_nullable_bytes(
        output, prefix + b".sqname", _semantic_cstring(ad.sqname)
    )
    _semantic_nullable_bytes(
        output, prefix + b".sqacc", _semantic_cstring(ad.sqacc)
    )
    _semantic_nullable_bytes(
        output, prefix + b".sqdesc", _semantic_cstring(ad.sqdesc)
    )
    _semantic_i64(output, prefix + b".sqfrom", ad.sqfrom)
    _semantic_i64(output, prefix + b".sqto", ad.sqto)
    _semantic_i64(output, prefix + b".L", ad.L)
    # mem/memsize are allocation representation, not alignment semantics.


cdef void _semantic_encode_domain(
    bytearray output,
    const P7_DOMAIN *domain,
    bytes prefix,
) except *:
    cdef int i
    if domain == NULL:
        raise ValueError("domain state is unavailable")
    _semantic_i64(output, prefix + b".ienv", domain.ienv)
    _semantic_i64(output, prefix + b".jenv", domain.jenv)
    _semantic_i64(output, prefix + b".iali", domain.iali)
    _semantic_i64(output, prefix + b".jali", domain.jali)
    # iorf/jorf are never initialized by the short protein-domain path.
    _semantic_f32(output, prefix + b".envsc", domain.envsc)
    _semantic_f32(output, prefix + b".domcorrection", domain.domcorrection)
    _semantic_f32(output, prefix + b".dombias", domain.dombias)
    _semantic_f32(output, prefix + b".oasc", domain.oasc)
    _semantic_f32(output, prefix + b".bitscore", domain.bitscore)
    _semantic_f64(output, prefix + b".lnP", domain.lnP)
    _semantic_bool(output, prefix + b".is_reported", domain.is_reported)
    _semantic_bool(output, prefix + b".is_included", domain.is_included)
    _semantic_encode_alidisplay(output, domain.ad, prefix + b".ad")
    _semantic_bool(
        output,
        prefix + b".scores_per_pos.present",
        domain.scores_per_pos != NULL,
    )
    if domain.scores_per_pos != NULL:
        if domain.ad == NULL:
            raise ValueError("domain score vector has no alignment display")
        for i in range(domain.ad.N):
            _semantic_f32(
                output,
                prefix
                + b".scores_per_pos["
                + str(i).encode("ascii")
                + b"]",
                domain.scores_per_pos[i],
            )


cdef void _semantic_encode_hit(
    bytearray output,
    const P7_HIT *hit,
    bytes prefix,
) except *:
    cdef int d
    if hit == NULL:
        raise ValueError("hit state is unavailable")
    if hit.ndom < 0 or (hit.ndom != 0 and hit.dcl == NULL):
        raise ValueError("hit domain storage is inconsistent")
    _semantic_nullable_bytes(output, prefix + b".name", _semantic_cstring(hit.name))
    _semantic_nullable_bytes(output, prefix + b".acc", _semantic_cstring(hit.acc))
    _semantic_nullable_bytes(output, prefix + b".desc", _semantic_cstring(hit.desc))
    # window_length, seqidx, and subseq_start are uninitialized in short mode.
    _semantic_f64(output, prefix + b".sortkey", hit.sortkey)
    _semantic_f32(output, prefix + b".score", hit.score)
    _semantic_f32(output, prefix + b".pre_score", hit.pre_score)
    _semantic_f32(output, prefix + b".sum_score", hit.sum_score)
    _semantic_f64(output, prefix + b".lnP", hit.lnP)
    _semantic_f64(output, prefix + b".pre_lnP", hit.pre_lnP)
    _semantic_f64(output, prefix + b".sum_lnP", hit.sum_lnP)
    _semantic_f32(output, prefix + b".nexpected", hit.nexpected)
    _semantic_i32(output, prefix + b".nregions", hit.nregions)
    _semantic_i32(output, prefix + b".nclustered", hit.nclustered)
    _semantic_i32(output, prefix + b".noverlaps", hit.noverlaps)
    _semantic_i32(output, prefix + b".nenvelopes", hit.nenvelopes)
    _semantic_i32(output, prefix + b".ndom", hit.ndom)
    _semantic_u32(output, prefix + b".flags", hit.flags)
    _semantic_i32(output, prefix + b".nreported", hit.nreported)
    _semantic_i32(output, prefix + b".nincluded", hit.nincluded)
    _semantic_i32(output, prefix + b".best_domain", hit.best_domain)
    _semantic_i64(output, prefix + b".offset", hit.offset)
    for d in range(hit.ndom):
        _semantic_encode_domain(
            output,
            &hit.dcl[d],
            prefix + b".domain[" + str(d).encode("ascii") + b"]",
        )


cdef bytes _semantic_pipeline_state_encoding(
    Pipeline pipeline,
    OptimizedProfile optimized_profile,
    bint include_optimized_profile,
):
    cdef bytearray output = bytearray(b"plan7-gpu-semantic-pipeline-v1\0")
    cdef P7_PIPELINE *pli = pipeline._pli
    cdef P7_BG *bg
    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    if pli == NULL:
        raise ValueError("pipeline state is unavailable")
    if (
        pli.oxf == NULL
        or pli.oxb == NULL
        or pli.fwd == NULL
        or pli.bck == NULL
        or pli.r == NULL
        or pli.ddef == NULL
    ):
        raise ValueError("pipeline reusable workspace is incomplete")
    if pli.hfp != NULL:
        raise ValueError("short protein search pipeline unexpectedly owns an HMM file")
    if pli.errbuf[0] != 0:
        raise ValueError("semantic fingerprint requires a successful clean pipeline")

    _semantic_encode_pipeline_scalars(output, pli, b"pipeline")
    _semantic_u32(output, b"pipeline.python_seed", pipeline._seed)
    _semantic_encode_rng(output, pli.r, b"pipeline.rng")
    _semantic_encode_domaindef(output, pli.ddef, pli.r, b"pipeline.ddef")
    _semantic_encode_reusable_omx(output, pli.oxf, b"pipeline.oxf")
    _semantic_encode_reusable_omx(output, pli.oxb, b"pipeline.oxb")
    _semantic_encode_reusable_omx(output, pli.fwd, b"pipeline.fwd")
    _semantic_encode_reusable_omx(output, pli.bck, b"pipeline.bck")

    if pipeline.background is None:
        raise ValueError("pipeline background is unavailable")
    bg = pipeline.background._bg
    _semantic_encode_background(
        output,
        bg,
        b"background",
    )
    _semantic_bool(output, b"optimized_profile.present", include_optimized_profile)
    if include_optimized_profile:
        if (
            optimized_profile._om == NULL
            or optimized_profile._om.abc == NULL
            or optimized_profile._om.abc.type != bg.abc.type
            or optimized_profile._om.abc.K != bg.abc.K
            or optimized_profile._om.abc.Kp != bg.abc.Kp
        ):
            raise ValueError("optimized profile and background alphabets differ")
        _semantic_encode_oprofile(
            output,
            optimized_profile._om,
            b"optimized_profile",
        )
    return bytes(output)


def _semantic_pipeline_state_encoding_bound(
    Pipeline pipeline,
    optimized_profile=None,
):
    """Return canonical typed short-search state; private Phase 1A oracle."""
    cdef OptimizedProfile profile
    if optimized_profile is None:
        profile = None
        return _semantic_pipeline_state_encoding(pipeline, profile, False)
    if type(optimized_profile) is not _pyhmmer.plan7.OptimizedProfile:
        raise TypeError(
            "optimized_profile must be exactly astra_pyhmmer.plan7.OptimizedProfile"
        )
    profile = optimized_profile
    return _semantic_pipeline_state_encoding(pipeline, profile, True)


def _semantic_pipeline_state_fingerprint_bound(
    Pipeline pipeline,
    optimized_profile=None,
):
    """Return SHA-256 of the canonical Phase 1A pipeline-state encoding."""
    return _hashlib.sha256(
        _semantic_pipeline_state_encoding_bound(pipeline, optimized_profile)
    ).digest()


cdef bytes _semantic_tophits_encoding(TopHits hits):
    cdef bytearray output = bytearray(b"plan7-gpu-semantic-tophits-v1\0")
    cdef P7_TOPHITS *th = hits._th
    cdef uintptr_t base
    cdef uintptr_t limit
    cdef uintptr_t pointer
    cdef uintptr_t distance
    cdef uintptr_t storage_bytes
    cdef uint64_t i
    cdef uint64_t order_index
    cdef bytes prefix
    cdef object seen_order = set()
    if type(hits) is not _pyhmmer.plan7.TopHits:
        raise TypeError("hits must be exactly astra_pyhmmer.plan7.TopHits")
    if th == NULL:
        raise ValueError("TopHits state is unavailable")
    if th.N > th.Nalloc:
        raise ValueError("TopHits count exceeds capacity")
    if th.N != 0 and (th.unsrt == NULL or th.hit == NULL):
        raise ValueError("TopHits storage is unavailable")
    _semantic_bool(output, b"tophits.empty", hits._empty)
    _semantic_encode_query_identity(output, hits._query, b"tophits.query")
    _semantic_u64(output, b"tophits.N", th.N)
    _semantic_u64(output, b"tophits.nreported", th.nreported)
    _semantic_u64(output, b"tophits.nincluded", th.nincluded)
    _semantic_bool(
        output,
        b"tophits.is_sorted_by_sortkey",
        th.is_sorted_by_sortkey,
    )
    _semantic_bool(
        output,
        b"tophits.is_sorted_by_seqidx",
        th.is_sorted_by_seqidx,
    )
    _semantic_encode_pipeline_scalars(output, &hits._pli, b"tophits.pipeline")

    if th.N != 0:
        base = <uintptr_t> th.unsrt
        if th.N > (<uintptr_t> -1) // sizeof(P7_HIT):
            raise OverflowError("TopHits storage size overflow")
        storage_bytes = <uintptr_t> th.N * sizeof(P7_HIT)
        if base > (<uintptr_t> -1) - storage_bytes:
            raise OverflowError("TopHits storage address overflow")
        limit = base + storage_bytes
        for i in range(th.N):
            pointer = <uintptr_t> th.hit[i]
            if pointer < base or pointer >= limit:
                raise ValueError("TopHits order pointer is outside hit storage")
            distance = pointer - base
            if distance % sizeof(P7_HIT) != 0:
                raise ValueError("TopHits order pointer is misaligned")
            order_index = distance // sizeof(P7_HIT)
            if order_index in seen_order:
                raise ValueError("TopHits order is not a unique permutation")
            seen_order.add(order_index)
            _semantic_u64(
                output,
                b"tophits.order[" + str(i).encode("ascii") + b"]",
                order_index,
            )
        if len(seen_order) != th.N:
            raise ValueError("TopHits order is not a complete permutation")
        for i in range(th.N):
            prefix = b"tophits.unsrt[" + str(i).encode("ascii") + b"]"
            _semantic_encode_hit(output, &th.unsrt[i], prefix)
    return bytes(output)


def _semantic_tophits_encoding_bound(TopHits hits):
    """Return canonical semantic TopHits bytes without raw serialization."""
    return _semantic_tophits_encoding(hits)


def _semantic_tophits_fingerprint_bound(TopHits hits):
    """Return SHA-256 of canonical semantic TopHits state."""
    return _hashlib.sha256(_semantic_tophits_encoding(hits)).digest()


cdef object _semantic_first_difference(bytes left, bytes right):
    cdef Py_ssize_t i
    cdef Py_ssize_t common = min(len(left), len(right))
    for i in range(common):
        if left[i] != right[i]:
            return i
    if len(left) != len(right):
        return common
    return None


def _semantic_dual_state_compare_bound(
    Pipeline left_pipeline,
    Pipeline right_pipeline,
    left_hits=None,
    right_hits=None,
    left_optimized_profile=None,
    right_optimized_profile=None,
):
    """Compare independently owned dense/sparse audit states without mutation."""
    cdef bytes left_pipeline_encoding
    cdef bytes right_pipeline_encoding
    cdef bytes left_hits_encoding
    cdef bytes right_hits_encoding
    cdef bytes left_profile_fingerprint
    cdef bytes right_profile_fingerprint
    if left_optimized_profile is None or right_optimized_profile is None:
        raise ValueError("both optimized profiles are required")
    if type(left_optimized_profile) is not _pyhmmer.plan7.OptimizedProfile:
        raise TypeError(
            "left_optimized_profile must be exactly "
            "astra_pyhmmer.plan7.OptimizedProfile"
        )
    if type(right_optimized_profile) is not _pyhmmer.plan7.OptimizedProfile:
        raise TypeError(
            "right_optimized_profile must be exactly "
            "astra_pyhmmer.plan7.OptimizedProfile"
        )
    if (left_hits is None) != (right_hits is None):
        raise ValueError("both TopHits values must be supplied together")
    left_profile_fingerprint = (
        _semantic_checked_profile_identity_fingerprint(
            left_optimized_profile,
            "left",
        )
    )
    right_profile_fingerprint = (
        _semantic_checked_profile_identity_fingerprint(
            right_optimized_profile,
            "right",
        )
    )
    if left_profile_fingerprint != right_profile_fingerprint:
        raise ValueError("optimized-profile identities differ")

    if left_hits is not None:
        if type(left_hits) is not _pyhmmer.plan7.TopHits:
            raise TypeError("left_hits must be exactly astra_pyhmmer.plan7.TopHits")
        if type(right_hits) is not _pyhmmer.plan7.TopHits:
            raise TypeError("right_hits must be exactly astra_pyhmmer.plan7.TopHits")
        _semantic_validate_tophits_profile_binding(
            left_hits,
            left_optimized_profile,
            left_profile_fingerprint,
            "left",
        )
        _semantic_validate_tophits_profile_binding(
            right_hits,
            right_optimized_profile,
            right_profile_fingerprint,
            "right",
        )

    left_pipeline_encoding = _semantic_pipeline_state_encoding_bound(
        left_pipeline, left_optimized_profile
    )
    right_pipeline_encoding = _semantic_pipeline_state_encoding_bound(
        right_pipeline, right_optimized_profile
    )
    result = {
        "schema_version": SEMANTIC_STATE_SCHEMA_VERSION,
        "pipeline": {
            "equal": left_pipeline_encoding == right_pipeline_encoding,
            "left_sha256": _hashlib.sha256(left_pipeline_encoding).hexdigest(),
            "right_sha256": _hashlib.sha256(right_pipeline_encoding).hexdigest(),
            "left_size": len(left_pipeline_encoding),
            "right_size": len(right_pipeline_encoding),
            "first_difference": _semantic_first_difference(
                left_pipeline_encoding, right_pipeline_encoding
            ),
        },
        "tophits": None,
    }
    if left_hits is not None:
        left_hits_encoding = _semantic_tophits_encoding(left_hits)
        right_hits_encoding = _semantic_tophits_encoding(right_hits)
        result["tophits"] = {
            "equal": left_hits_encoding == right_hits_encoding,
            "left_sha256": _hashlib.sha256(left_hits_encoding).hexdigest(),
            "right_sha256": _hashlib.sha256(right_hits_encoding).hexdigest(),
            "left_size": len(left_hits_encoding),
            "right_size": len(right_hits_encoding),
            "first_difference": _semantic_first_difference(
                left_hits_encoding, right_hits_encoding
            ),
        }
    return result


cdef void _v3_zero_statistics(
    _compact_consumption_statistics *statistics,
) noexcept nogil:
    memset(statistics, 0, sizeof(_compact_consumption_statistics))


cdef void _v3_apply_certificate_accounting(
    P7_PIPELINE *pli,
    const plan7_continuation_journal_v3_certificate *certificate,
) noexcept nogil:
    """Apply one already validated omitted-row certificate exactly once."""
    pli.nseqs += certificate.target_delta
    if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
        pli.Z = pli.nseqs
    pli.nres += certificate.residue_delta
    pli.n_past_msv += certificate.n_past_msv_delta
    pli.n_past_bias += certificate.n_past_bias_delta
    pli.n_past_vit += certificate.n_past_vit_delta
    pli.n_past_fwd += certificate.n_past_fwd_delta


cdef int _search_loop_continuation_journal_v3(
    P7_PIPELINE *pli,
    P7_OPROFILE *om,
    P7_BG *bg,
    const ESL_SQ **sq,
    size_t n_targets,
    const plan7_continuation_journal_v3 *journal,
    const plan7_continuation_journal_v3_profile *profile_record,
    P7_TOPHITS *th,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
    _pipeline_from_filter_forward_backward_scores_f forward_backward_scores_seam,
    _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam,
    _pipeline_from_filter_forward_compact_domains_f compact_domains_seam,
    uint64_t *compact_rebased_offsets,
    _compact_consumption_statistics *statistics,
    uint64_t local_begin,
    uint64_t local_stop,
    bint apply_terminal_certificate,
) except 1 nogil:
    """Consume one validated v3 profile exception range without dense replay."""
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_certificate *certificates = (
        <const plan7_continuation_journal_v3_certificate *> (
            base + journal.certificates_offset
        )
    )
    cdef const plan7_continuation_journal_v3_exception *exceptions = (
        <const plan7_continuation_journal_v3_exception *> (
            base + journal.exceptions_offset
        )
    )
    cdef const float *specials = <const float *> (
        base + journal.specials_offset
    )
    cdef const plan7_simple_region *all_regions = (
        <const plan7_simple_region *> (base + journal.regions_offset)
    )
    cdef const plan7_domain_rescore_result *all_compact_results = (
        <const plan7_domain_rescore_result *> (
            base + journal.compact_results_offset
        )
    )
    cdef const uint64_t *all_trace_offsets = <const uint64_t *> (
        base + journal.compact_trace_offsets_offset
    )
    cdef const plan7_domain_rescore_trace_step *all_traces = (
        <const plan7_domain_rescore_trace_step *> (
            base + journal.compact_traces_offset
        )
    )
    cdef const float *all_null2 = <const float *> (
        base + journal.compact_null2_offset
    )
    cdef const plan7_continuation_journal_v3_certificate *certificate
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef const plan7_simple_region *regions = NULL
    cdef const plan7_domain_rescore_result *compact_results = NULL
    cdef const plan7_domain_rescore_trace_step *compact_traces = NULL
    cdef const float *compact_null2 = NULL
    cdef const float *xmx = NULL
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef plan7_continuation_journal_row domain
    cdef uint64_t local_index
    cdef uint64_t compact_index
    cdef uint64_t trace_base
    cdef uint64_t trace_stop
    cdef uint64_t xmx_count
    cdef uint64_t t
    cdef float usc
    cdef int status
    cdef bint used_forward_seam
    cdef bint used_forward_backward_seam
    cdef bint used_simple_regions_seam
    cdef bint used_compact_domains_seam
    cdef const unsigned char *simd_sequences[4]
    cdef int simd_lengths[4]
    cdef const float *simd_forward_xmx[4]
    cdef uint64_t simd_forward_xmx_counts[4]
    cdef const float *simd_backward_xmx[4]
    cdef uint64_t simd_backward_xmx_counts[4]
    cdef float simd_backward_scores[4]
    cdef float simd_backward_totscales[4]
    cdef int simd_backward_has_own_scales[4]
    cdef uint64_t simd_exception_indices[4]
    cdef uint64_t simd_kernel_ns = 0
    cdef const unsigned char *filter_simd_sequences[4]
    cdef int filter_simd_lengths[4]
    cdef const float *filter_simd_forward_xmx[4]
    cdef uint64_t filter_simd_forward_xmx_counts[4]
    cdef float filter_simd_forward_scores[4]
    cdef float filter_simd_forward_totscales[4]
    cdef int filter_simd_forward_statuses[4]
    cdef const float *filter_simd_backward_xmx[4]
    cdef uint64_t filter_simd_backward_xmx_counts[4]
    cdef float filter_simd_backward_scores[4]
    cdef float filter_simd_backward_totscales[4]
    cdef int filter_simd_backward_has_own_scales[4]
    cdef float filter_simd_filtersc[4]
    cdef bint filter_simd_f3_pass[4]
    cdef uint64_t filter_simd_exception_indices[4]
    cdef uint64_t filter_simd_forward_kernel_ns = 0
    cdef uint64_t filter_simd_backward_kernel_ns = 0
    cdef uint64_t scan_index
    cdef uint64_t scan_sequence_index
    cdef uint64_t dynamic_z_offset = 0
    cdef const plan7_continuation_journal_v3_exception *scan_exception
    cdef _postfilter_result scan_postfilter
    cdef int simd_count
    cdef int simd_lane = -1
    cdef int simd_status
    cdef int filter_simd_count
    cdef int filter_simd_lane = -1
    cdef int filter_simd_consumed_lane = -1
    cdef int filter_simd_index
    cdef int filter_simd_status
    cdef int filter_simd_f3_decision
    cdef int filter_simd_survivor_count
    cdef bint filter_simd_backward_ready = False
    cdef bint filter_simd_group_has_fallback = False
    cdef bint filter_simd_prevalidation_unchanged
    cdef uint64_t filter_simd_hits_before
    cdef uint64_t filter_simd_n_past_msv_before
    cdef uint64_t filter_simd_n_past_bias_before
    cdef uint64_t filter_simd_n_past_vit_before
    cdef uint64_t filter_simd_n_past_fwd_before
    cdef uint64_t filter_simd_pos_past_msv_before
    cdef uint64_t filter_simd_pos_past_bias_before
    cdef uint64_t filter_simd_pos_past_vit_before
    cdef uint64_t filter_simd_pos_past_fwd_before
    cdef const char *filter_simd_env = getenv(
        b"PLAN7_GPU_FILTER_TAIL_SIMD"
    )
    cdef const char *filter_simd_test_fallback_env = getenv(
        b"PLAN7_GPU_FILTER_TAIL_SIMD_TEST_FALLBACK"
    )
    cdef const char *tail_madvise_env = getenv(
        b"PLAN7_GPU_AVX512_TAIL_MADVISE"
    )
    cdef bint filter_simd_test_fallback = (
        filter_simd_test_fallback_env != NULL
        and strcmp(filter_simd_test_fallback_env, b"1") == 0
    )
    cdef bint tail_madvise = (
        _avx512_tail_madvise_override == 1
        or (
            _avx512_tail_madvise_override < 0
            and tail_madvise_env != NULL
            and strcmp(tail_madvise_env, b"1") == 0
        )
    )
    cdef uint64_t tail_madvise_released_bytes = 0
    cdef int tail_madvise_status
    cdef bint simd_available = (
        forward_backward_scores_seam != NULL
        and plan7_avx512_tail_available() != 0
    )
    cdef bint filter_simd_available = (
        (
            _filter_tail_simd_override == 1
            or (
                _filter_tail_simd_override < 0
                and filter_simd_env != NULL
                and strcmp(filter_simd_env, b"1") == 0
            )
        )
        and forward_scores_seam != NULL
        and forward_backward_scores_seam != NULL
        and plan7_avx512_tail_available() != 0
    )

    if local_begin > local_stop or local_stop > profile_record.exception_count:
        raise IndexError("journal v3 exception range is invalid")

    # A shard owns local accounting so that TopHits.merge() can add each
    # counter exactly once.  HMMER's online target-reportability decision,
    # however, depends on the absolute number of targets seen so far when Z
    # is dynamic.  Recover that absolute prefix from the preceding exception;
    # restore local Z before returning so pipeline merging remains exact.
    if local_begin != 0:
        dynamic_z_offset = (
            exceptions[
                profile_record.exception_begin + local_begin - 1
            ].sequence_index
            + 1
        )

    if statistics != NULL:
        statistics.target_count = n_targets
        statistics.postfilter_record_count = (
            profile_record.source_postfilter_count
        )
        statistics.f1_reject_count = (
            n_targets - profile_record.source_postfilter_count
        )

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for local_index in range(local_begin, local_stop):
        certificate = &certificates[
            profile_record.certificate_begin + local_index
        ]
        _v3_apply_certificate_accounting(pli, certificate)
        if (
            dynamic_z_offset != 0
            and pli.Z_setby == p7_ZSETBY_NTARGETS
            and pli.mode == p7_SEARCH_SEQS
        ):
            pli.Z = pli.nseqs + dynamic_z_offset
        if statistics != NULL:
            statistics.definite_reject_count += (
                certificate.raw_f1_reject_count
            )

        # The dense loop reuses once for an omitted prefix before its first
        # retained row. Later gaps follow a reuse of the preceding exception.
        if local_index == local_begin and certificate.target_delta != 0:
            p7_pipeline_Reuse(pli)

        exception = &exceptions[
            profile_record.exception_begin + local_index
        ]
        t = exception.sequence_index

        # The exception target belongs to its seam. Only NewSeq accounting is
        # applied here; none of its promotion counters are pre-accounted.
        pli.nseqs += 1
        if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
            pli.Z = pli.nseqs + dynamic_z_offset
        pli.nres += exception.residue_delta

        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")

        memcpy(
            &postfilter,
            exception.postfilter_record,
            sizeof(_postfilter_result),
        )
        used_forward_seam = False
        used_forward_backward_seam = False
        used_simple_regions_seam = False
        used_compact_domains_seam = False

        if exception.route == PLAN7_CONTINUATION_V3_FULL_PIPELINE:
            if statistics != NULL:
                statistics.cpu_pipeline_count += 1
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        else:
            usc = <float> postfilter.msv_numerator
            usc = usc / om.scale_b
            usc = usc - <float> 3.0
            if exception.route == PLAN7_CONTINUATION_V3_FILTER_SCORES:
                if statistics != NULL:
                    statistics.filter_continuation_count += 1
                    statistics.cpu_tail_filter_simd_eligible_rows += 1
                if (
                    filter_simd_lane < 0
                    and simd_lane < 0
                    and filter_simd_available
                ):
                    filter_simd_count = 0
                    for scan_index in range(local_index, local_stop):
                        scan_exception = &exceptions[
                            profile_record.exception_begin + scan_index
                        ]
                        if (
                            scan_exception.route
                            != PLAN7_CONTINUATION_V3_FILTER_SCORES
                        ):
                            continue
                        scan_sequence_index = scan_exception.sequence_index
                        if (
                            sq[scan_sequence_index].n <= 0
                            or sq[scan_sequence_index].n > 100000
                        ):
                            continue
                        memcpy(
                            &scan_postfilter,
                            scan_exception.postfilter_record,
                            sizeof(_postfilter_result),
                        )
                        filter_simd_exception_indices[
                            filter_simd_count
                        ] = scan_index
                        filter_simd_sequences[filter_simd_count] = (
                            <const unsigned char *>
                            sq[scan_sequence_index].dsq
                        )
                        filter_simd_lengths[filter_simd_count] = (
                            sq[scan_sequence_index].n
                        )
                        filter_simd_filtersc[filter_simd_count] = (
                            scan_postfilter.filtersc
                        )
                        filter_simd_count += 1
                        if (
                            filter_simd_count
                            == PLAN7_AVX512_TAIL_LANES
                        ):
                            break
                    if (
                        filter_simd_count
                        == PLAN7_AVX512_TAIL_LANES
                    ):
                        filter_simd_group_has_fallback = False
                        filter_simd_status = plan7_avx512_forward4_varlen(
                            filter_simd_sequences,
                            filter_simd_lengths,
                            om,
                            filter_simd_forward_xmx,
                            filter_simd_forward_xmx_counts,
                            filter_simd_forward_scores,
                            filter_simd_forward_totscales,
                            filter_simd_forward_statuses,
                            &filter_simd_forward_kernel_ns,
                        )
                        filter_simd_survivor_count = 0
                        if filter_simd_status == eslOK:
                            for filter_simd_index in range(
                                PLAN7_AVX512_TAIL_LANES
                            ):
                                if (
                                    filter_simd_forward_statuses[
                                        filter_simd_index
                                    ] != eslOK
                                    or not isfinite(
                                        filter_simd_forward_scores[
                                            filter_simd_index
                                        ]
                                    )
                                ):
                                    filter_simd_status = eslERANGE
                                    break
                                filter_simd_f3_decision = (
                                    _hmmer_f3_decision(
                                        om,
                                        filter_simd_filtersc[
                                            filter_simd_index
                                        ],
                                        filter_simd_forward_scores[
                                            filter_simd_index
                                        ],
                                        pli.F3,
                                    )
                                )
                                if filter_simd_f3_decision < 0:
                                    filter_simd_status = eslEINVAL
                                    break
                                filter_simd_f3_pass[
                                    filter_simd_index
                                ] = filter_simd_f3_decision != 0
                                if filter_simd_f3_pass[
                                    filter_simd_index
                                ]:
                                    filter_simd_survivor_count += 1

                        filter_simd_backward_ready = False
                        if (
                            filter_simd_status == eslOK
                            and filter_simd_survivor_count != 0
                        ):
                            simd_status = plan7_avx512_backward4_varlen(
                                filter_simd_sequences,
                                filter_simd_lengths,
                                om,
                                filter_simd_forward_xmx,
                                filter_simd_forward_xmx_counts,
                                filter_simd_backward_xmx,
                                filter_simd_backward_xmx_counts,
                                filter_simd_backward_scores,
                                filter_simd_backward_totscales,
                                filter_simd_backward_has_own_scales,
                                &filter_simd_backward_kernel_ns,
                            )
                            if simd_status == eslOK:
                                filter_simd_backward_ready = True
                                if statistics != NULL:
                                    statistics.cpu_tail_filter_simd_backward_groups += 1
                                    statistics.cpu_tail_filter_simd_backward_kernel_ns += (
                                        filter_simd_backward_kernel_ns
                                    )
                            elif statistics != NULL:
                                statistics.cpu_tail_filter_simd_fallback_groups += 1

                        if filter_simd_status == eslOK:
                            filter_simd_lane = 0
                            if statistics != NULL:
                                statistics.cpu_tail_filter_simd_groups += 1
                                statistics.cpu_tail_filter_simd_rows += 4
                                statistics.cpu_tail_filter_simd_f3_survivor_rows += (
                                    filter_simd_survivor_count
                                )
                                statistics.cpu_tail_filter_simd_forward_kernel_ns += (
                                    filter_simd_forward_kernel_ns
                                )
                        else:
                            filter_simd_available = False
                            if statistics != NULL:
                                statistics.cpu_tail_filter_simd_fallback_groups += 1

                if (
                    filter_simd_lane >= 0
                    and filter_simd_lane < PLAN7_AVX512_TAIL_LANES
                    and filter_simd_exception_indices[
                        filter_simd_lane
                    ] == local_index
                ):
                    if (
                        filter_simd_backward_ready
                        and filter_simd_f3_pass[filter_simd_lane]
                    ):
                        filter_simd_consumed_lane = filter_simd_lane
                        filter_simd_hits_before = th.N
                        filter_simd_n_past_msv_before = pli.n_past_msv
                        filter_simd_n_past_bias_before = pli.n_past_bias
                        filter_simd_n_past_vit_before = pli.n_past_vit
                        filter_simd_n_past_fwd_before = pli.n_past_fwd
                        filter_simd_pos_past_msv_before = pli.pos_past_msv
                        filter_simd_pos_past_bias_before = pli.pos_past_bias
                        filter_simd_pos_past_vit_before = pli.pos_past_vit
                        filter_simd_pos_past_fwd_before = pli.pos_past_fwd
                        if filter_simd_test_fallback:
                            status = eslEINVAL
                        else:
                            status = forward_backward_scores_seam(
                                pli,
                                om,
                                bg,
                                sq[t],
                                NULL,
                                th,
                                usc,
                                postfilter.filtersc,
                                postfilter.vfsc,
                                filter_simd_forward_scores[filter_simd_lane],
                                filter_simd_forward_xmx[filter_simd_lane],
                                filter_simd_forward_xmx_counts[filter_simd_lane],
                                filter_simd_backward_scores[filter_simd_lane],
                                filter_simd_backward_xmx[filter_simd_lane],
                                filter_simd_backward_xmx_counts[filter_simd_lane],
                                filter_simd_backward_totscales[
                                    filter_simd_lane
                                ],
                                filter_simd_backward_has_own_scales[
                                    filter_simd_lane
                                ],
                            )
                        filter_simd_prevalidation_unchanged = (
                            th.N == filter_simd_hits_before
                            and pli.n_past_msv == filter_simd_n_past_msv_before
                            and pli.n_past_bias == filter_simd_n_past_bias_before
                            and pli.n_past_vit == filter_simd_n_past_vit_before
                            and pli.n_past_fwd == filter_simd_n_past_fwd_before
                            and pli.pos_past_msv == filter_simd_pos_past_msv_before
                            and pli.pos_past_bias == filter_simd_pos_past_bias_before
                            and pli.pos_past_vit == filter_simd_pos_past_vit_before
                            and pli.pos_past_fwd == filter_simd_pos_past_fwd_before
                        )
                        if status == eslEINVAL and filter_simd_prevalidation_unchanged:
                            # The private seam validates every external score
                            # and special before changing any of these counters.
                            # Replay only that proven pre-mutation rejection;
                            # any later domain-workflow EINVAL remains fatal.
                            p7_pipeline_Reuse(pli)
                            pli.errbuf[0] = 0
                            status = p7_bg_SetLength(bg, sq[t].n)
                            if status == eslOK:
                                status = p7_oprofile_ReconfigLength(
                                    om, sq[t].n
                                )
                            if status == eslOK:
                                status = filter_scores_seam(
                                    pli,
                                    om,
                                    bg,
                                    sq[t],
                                    NULL,
                                    th,
                                    usc,
                                    postfilter.filtersc,
                                    P7_VIT_EXTERNAL,
                                    postfilter.vfsc,
                                )
                            used_forward_seam = True
                            if status == eslOK and statistics != NULL:
                                statistics.cpu_tail_filter_simd_fallback_rows += 1
                                if not filter_simd_group_has_fallback:
                                    statistics.cpu_tail_filter_simd_fallback_groups += 1
                                    filter_simd_group_has_fallback = True
                        else:
                            used_forward_backward_seam = True
                    else:
                        status = forward_scores_seam(
                            pli,
                            om,
                            bg,
                            sq[t],
                            NULL,
                            th,
                            usc,
                            postfilter.filtersc,
                            postfilter.vfsc,
                            filter_simd_forward_scores[filter_simd_lane],
                            filter_simd_forward_xmx[filter_simd_lane],
                            filter_simd_forward_xmx_counts[filter_simd_lane],
                        )
                        used_forward_seam = True
                    filter_simd_lane += 1
                    if filter_simd_lane == PLAN7_AVX512_TAIL_LANES:
                        filter_simd_lane = -1
                        filter_simd_backward_ready = False
                        if tail_madvise:
                            tail_madvise_status = (
                                plan7_avx512_tail_madvise_pages(
                                    &tail_madvise_released_bytes
                                )
                            )
                            if (
                                status == eslOK
                                and tail_madvise_status != eslOK
                            ):
                                status = tail_madvise_status
                else:
                    status = filter_scores_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        usc,
                        postfilter.filtersc,
                        P7_VIT_EXTERNAL,
                        postfilter.vfsc,
                    )
            elif exception.route == PLAN7_CONTINUATION_V3_FORWARD_SCORES:
                if statistics != NULL:
                    statistics.forward_continuation_count += 1
                    statistics.cpu_tail_simd_eligible_rows += 1
                memcpy(
                    &forward,
                    exception.forward_record,
                    sizeof(_forward_result),
                )
                xmx_count = exception.special_count
                xmx = (
                    specials + exception.special_begin
                    if xmx_count != 0
                    else NULL
                )
                if (
                    simd_lane < 0
                    and filter_simd_lane < 0
                    and simd_available
                ):
                    simd_count = 0
                    for scan_index in range(local_index, local_stop):
                        scan_exception = &exceptions[
                            profile_record.exception_begin + scan_index
                        ]
                        if (
                            scan_exception.route
                            != PLAN7_CONTINUATION_V3_FORWARD_SCORES
                        ):
                            continue
                        scan_sequence_index = scan_exception.sequence_index
                        if (
                            scan_exception.special_count == 0
                            or sq[scan_sequence_index].n <= 0
                            or sq[scan_sequence_index].n > 100000
                        ):
                            continue
                        simd_exception_indices[simd_count] = scan_index
                        simd_sequences[simd_count] = (
                            <const unsigned char *>
                            sq[scan_sequence_index].dsq
                        )
                        simd_lengths[simd_count] = sq[scan_sequence_index].n
                        simd_forward_xmx[simd_count] = (
                            specials + scan_exception.special_begin
                        )
                        simd_forward_xmx_counts[simd_count] = (
                            scan_exception.special_count
                        )
                        simd_count += 1
                        if simd_count == PLAN7_AVX512_TAIL_LANES:
                            break
                    if simd_count == PLAN7_AVX512_TAIL_LANES:
                        simd_status = plan7_avx512_backward4_varlen(
                            simd_sequences,
                            simd_lengths,
                            om,
                            simd_forward_xmx,
                            simd_forward_xmx_counts,
                            simd_backward_xmx,
                            simd_backward_xmx_counts,
                            simd_backward_scores,
                            simd_backward_totscales,
                            simd_backward_has_own_scales,
                            &simd_kernel_ns,
                        )
                        if simd_status == eslOK:
                            simd_lane = 0
                            if statistics != NULL:
                                statistics.cpu_tail_simd_groups += 1
                                statistics.cpu_tail_simd_rows += 4
                                statistics.cpu_tail_simd_kernel_ns += (
                                    simd_kernel_ns
                                )
                        else:
                            simd_available = False
                            if statistics != NULL:
                                statistics.cpu_tail_simd_fallback_groups += 1
                if (
                    simd_lane >= 0
                    and simd_lane < PLAN7_AVX512_TAIL_LANES
                    and simd_exception_indices[simd_lane] == local_index
                ):
                    status = forward_backward_scores_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        usc,
                        postfilter.filtersc,
                        postfilter.vfsc,
                        forward.fwdsc,
                        xmx,
                        xmx_count,
                        simd_backward_scores[simd_lane],
                        simd_backward_xmx[simd_lane],
                        simd_backward_xmx_counts[simd_lane],
                        simd_backward_totscales[simd_lane],
                        simd_backward_has_own_scales[simd_lane],
                    )
                    simd_lane += 1
                    if simd_lane == PLAN7_AVX512_TAIL_LANES:
                        simd_lane = -1
                        if tail_madvise:
                            tail_madvise_status = (
                                plan7_avx512_tail_madvise_pages(
                                    &tail_madvise_released_bytes
                                )
                            )
                            if (
                                status == eslOK
                                and tail_madvise_status != eslOK
                            ):
                                status = tail_madvise_status
                    used_forward_backward_seam = True
                else:
                    status = forward_scores_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        usc,
                        postfilter.filtersc,
                        postfilter.vfsc,
                        forward.fwdsc,
                        xmx,
                        xmx_count,
                    )
                    used_forward_seam = True
            elif exception.route == PLAN7_CONTINUATION_V3_SIMPLE_REGIONS:
                if statistics != NULL:
                    statistics.simple_continuation_count += 1
                memcpy(
                    &domain,
                    exception.domain_record,
                    sizeof(plan7_continuation_journal_row),
                )
                regions = (
                    all_regions + exception.region_begin
                    if exception.region_count != 0
                    else NULL
                )
                status = simple_regions_seam(
                    pli,
                    om,
                    bg,
                    sq[t],
                    NULL,
                    th,
                    domain.usc,
                    domain.filtersc,
                    domain.vfsc,
                    domain.fwdsc,
                    journal.options.f1_bits,
                    journal.options.f2_bits,
                    journal.options.f3_bits,
                    journal.options.do_biasfilter,
                    domain.domain_route,
                    domain.nexpected,
                    regions,
                    exception.region_count,
                )
                used_simple_regions_seam = True
            else:
                # Validation and preflight restrict the remaining route to the
                # compact-domain seam, with enough scratch for every offset.
                memcpy(
                    &forward,
                    exception.forward_record,
                    sizeof(_forward_result),
                )
                memcpy(
                    &domain,
                    exception.domain_record,
                    sizeof(plan7_continuation_journal_row),
                )
                trace_base = exception.compact_trace_begin
                trace_stop = trace_base + exception.compact_trace_count
                for compact_index in range(exception.compact_result_count + 1):
                    compact_rebased_offsets[compact_index] = (
                        all_trace_offsets[
                            exception.compact_result_begin + compact_index
                        ]
                        - trace_base
                    )
                compact_results = (
                    all_compact_results + exception.compact_result_begin
                )
                compact_traces = all_traces + trace_base
                compact_null2 = all_null2 + exception.compact_null2_begin
                if statistics != NULL:
                    if statistics.attempt_count == 0:
                        statistics.first_row_index = (
                            exception.source_domain_index
                        )
                        statistics.first_profile_index = exception.profile_index
                        statistics.first_sequence_index = exception.sequence_index
                        statistics.first_domain_count = (
                            exception.compact_result_count
                        )
                    statistics.attempt_count += 1
                status = compact_domains_seam(
                    pli,
                    om,
                    bg,
                    sq[t],
                    NULL,
                    th,
                    domain.usc,
                    domain.filtersc,
                    domain.vfsc,
                    domain.fwdsc,
                    journal.generation_tail_fingerprint,
                    n_targets,
                    <uint32_t> exception.source_domain_index,
                    exception.profile_index,
                    exception.sequence_index,
                    domain.nexpected,
                    compact_results,
                    exception.compact_result_count,
                    compact_rebased_offsets,
                    exception.compact_result_count + 1,
                    compact_traces,
                    trace_stop - trace_base,
                    compact_null2,
                    exception.compact_null2_count,
                )
                used_compact_domains_seam = True
                if status == eslEINACCURATE:
                    if statistics != NULL:
                        statistics.threshold_retry_count += 1
                        statistics.cpu_pipeline_count += 1
                    status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
                    used_compact_domains_seam = False
                elif status == eslEINVAL:
                    if statistics != NULL:
                        statistics.invalid_retry_count += 1
                        statistics.forward_continuation_count += 1
                    xmx_count = exception.special_count
                    xmx = (
                        specials + exception.special_begin
                        if xmx_count != 0
                        else NULL
                    )
                    status = forward_scores_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        usc,
                        postfilter.filtersc,
                        postfilter.vfsc,
                        forward.fwdsc,
                        xmx,
                        xmx_count,
                    )
                    used_compact_domains_seam = False
                    used_forward_seam = True
                elif status == eslOK and statistics != NULL:
                    statistics.accepted_count += 1

        if status == eslEINVAL:
            if used_simple_regions_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterAndForwardSimpleRegions",
                )
            elif used_compact_domains_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterForwardAndCompactDomainsV2",
                )
            elif used_forward_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterAndForwardScores",
                )
            elif used_forward_backward_seam:
                raise UnexpectedError(
                    status,
                    (
                        "p7_PipelineFromFilterForwardAndBackwardScores"
                        f"[profile={profile_record.profile_index},"
                        f"sequence={t},length={sq[t].n},"
                        f"local_exception={local_index},"
                        f"lane={filter_simd_consumed_lane},"
                        f"group_exceptions={filter_simd_exception_indices[0]}/"
                        f"{filter_simd_exception_indices[1]}/"
                        f"{filter_simd_exception_indices[2]}/"
                        f"{filter_simd_exception_indices[3]},"
                        f"fwd={filter_simd_forward_scores[filter_simd_consumed_lane]},"
                        f"bck={filter_simd_backward_scores[filter_simd_consumed_lane]},"
                        f"fwd_xmx={filter_simd_forward_xmx_counts[filter_simd_consumed_lane]},"
                        f"bck_xmx={filter_simd_backward_xmx_counts[filter_simd_consumed_lane]},"
                        f"bck_scale={filter_simd_backward_totscales[filter_simd_consumed_lane]},"
                        f"bck_own={filter_simd_backward_has_own_scales[filter_simd_consumed_lane]}]"
                    ),
                )
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")
        p7_pipeline_Reuse(pli)

    if statistics != NULL:
        statistics.cpu_tail_simd_scalar_tail_rows = (
            statistics.cpu_tail_simd_eligible_rows
            - statistics.cpu_tail_simd_rows
        )
        statistics.cpu_tail_filter_simd_scalar_tail_rows = (
            statistics.cpu_tail_filter_simd_eligible_rows
            - statistics.cpu_tail_filter_simd_rows
        )

    if apply_terminal_certificate:
        certificate = &certificates[
            profile_record.certificate_begin + profile_record.exception_count
        ]
        _v3_apply_certificate_accounting(pli, certificate)
        if (
            dynamic_z_offset != 0
            and pli.Z_setby == p7_ZSETBY_NTARGETS
            and pli.mode == p7_SEARCH_SEQS
        ):
            pli.Z = pli.nseqs + dynamic_z_offset
        if statistics != NULL:
            statistics.definite_reject_count += certificate.raw_f1_reject_count

        if (
            n_targets != 0
            and (
                profile_record.exception_count == 0
                or exceptions[
                    profile_record.exception_begin
                    + profile_record.exception_count - 1
                ].sequence_index != n_targets - 1
            )
        ):
            status = p7_bg_SetLength(bg, sq[n_targets - 1].n)
            if status != eslOK:
                raise UnexpectedError(status, "p7_bg_SetLength")
            status = p7_oprofile_ReconfigLength(om, sq[n_targets - 1].n)
            if status != eslOK:
                raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
            p7_pipeline_Reuse(pli)
    if (
        dynamic_z_offset != 0
        and pli.Z_setby == p7_ZSETBY_NTARGETS
        and pli.mode == p7_SEARCH_SEQS
    ):
        pli.Z = pli.nseqs
    return 0


cdef void _v3_claim_capsule(
    object capsule,
    plan7_continuation_journal_v3 *journal,
    plan7_continuation_journal_v3_owner *owner,
    bint *claimed,
) except *:
    """Transfer one validated packet to the executor before any mutation."""
    cdef bint transferred = False
    claimed[0] = False
    try:
        PyCapsule_SetDestructor(capsule, <PyCapsule_Destructor> NULL)
        transferred = True
        PyCapsule_SetPointer(capsule, &_consumed_journal_sentinel)
        PyCapsule_SetContext(capsule, NULL)
        PyCapsule_SetName(
            capsule, PLAN7_CONTINUATION_JOURNAL_V3_CONSUMED_NAME
        )
        claimed[0] = True
    except:
        if transferred:
            try:
                PyCapsule_SetName(
                    capsule, PLAN7_CONTINUATION_JOURNAL_V3_CONSUMED_NAME
                )
            except:
                pass
            free(journal)
            free(owner)
        raise


cdef void _v3_preflight_counter_capacity_range(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    Pipeline pipeline,
    size_t profile_begin,
    size_t profile_end,
) except *:
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    cdef const plan7_continuation_journal_v3_certificate *certificates = (
        <const plan7_continuation_journal_v3_certificate *> (
            base + journal.certificates_offset
        )
    )
    cdef uint64_t nres_delta
    cdef uint64_t nnodes_delta = 0
    cdef uint64_t msv_delta = 0
    cdef uint64_t bias_delta = 0
    cdef uint64_t vit_delta = 0
    cdef uint64_t fwd_delta = 0
    cdef uint64_t scratch
    cdef uint64_t profile_count
    cdef uint64_t certificate_begin
    cdef uint64_t certificate_end
    cdef uint64_t index
    cdef uint64_t profile_index
    cdef OptimizedProfile profile

    if profile_begin > profile_end or profile_end > journal.profile_count:
        raise IndexError("journal v3 preflight profile range is invalid")
    profile_count = profile_end - profile_begin
    if not plan7_continuation_journal_v3_checked_multiply(
        journal.total_residues, profile_count, &nres_delta
    ):
        raise OverflowError("journal v3 cumulative residue delta overflows")
    for profile_index in range(profile_begin, profile_end):
        profile = <OptimizedProfile> sealed._optimized_profiles[profile_index]
        if not _v3_checked_increment(&nnodes_delta, <uint64_t> profile._om.M):
            raise OverflowError("journal v3 cumulative model nodes overflow")
        if (
            not _v3_checked_increment(
                &msv_delta, profiles[profile_index].exception_count
            )
            or not _v3_checked_increment(
                &bias_delta, profiles[profile_index].exception_count
            )
            or not _v3_checked_increment(
                &vit_delta, profiles[profile_index].exception_count
            )
            or not _v3_checked_increment(
                &fwd_delta, profiles[profile_index].exception_count
            )
        ):
            raise OverflowError("journal v3 cumulative exception delta overflows")
        certificate_begin = profiles[profile_index].certificate_begin
        certificate_end = (
            certificate_begin + profiles[profile_index].certificate_count
        )
        for index in range(certificate_begin, certificate_end):
            if (
                not _v3_checked_increment(
                    &msv_delta, certificates[index].n_past_msv_delta
                )
                or not _v3_checked_increment(
                    &bias_delta, certificates[index].n_past_bias_delta
                )
                or not _v3_checked_increment(
                    &vit_delta, certificates[index].n_past_vit_delta
                )
                or not _v3_checked_increment(
                    &fwd_delta, certificates[index].n_past_fwd_delta
                )
            ):
                raise OverflowError(
                    "journal v3 cumulative promotion delta overflows"
                )
    if (
        not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.nmodels, profile_count, &scratch
        )
        or not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.nnodes, nnodes_delta, &scratch
        )
        or not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.nres, nres_delta, &scratch
        )
        or not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.n_past_msv, msv_delta, &scratch
        )
        or not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.n_past_bias, bias_delta, &scratch
        )
        or not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.n_past_vit, vit_delta, &scratch
        )
        or not plan7_continuation_journal_v3_checked_add(
            pipeline._pli.n_past_fwd, fwd_delta, &scratch
        )
    ):
        raise OverflowError("journal v3 pipeline counter prestate overflows")


cdef void _v3_preflight_live_pipeline_range(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    Pipeline pipeline,
    size_t profile_begin,
    size_t profile_end,
) except *:
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    cdef const plan7_continuation_journal_v3_exception *exceptions = (
        <const plan7_continuation_journal_v3_exception *> (
            base + journal.exceptions_offset
        )
    )
    cdef uint64_t index
    cdef uint64_t tail_fingerprint
    cdef bint needs_filter = False
    cdef bint needs_forward = False
    cdef bint needs_simple = False
    cdef bint needs_compact = False
    cdef bint environment_attested
    cdef size_t profile_index
    cdef uint64_t exception_begin
    cdef uint64_t exception_end
    cdef P7_OPROFILE *profile

    if profile_begin > profile_end or profile_end > journal.profile_count:
        raise IndexError("journal v3 preflight profile range is invalid")

    with nogil:
        environment_attested = (
            plan7_gpu_pipeline_host_environment_attested() == 1
        )
    if not environment_attested:
        raise RuntimeError("journal v3 host floating-point environment differs")

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    if not pipeline.alphabet._eq(sealed._sequences.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sealed._sequences.alphabet)
    if not journal.options.complete:
        raise ValueError("journal v3 execution requires complete options")
    if journal.source_kind not in (
        PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL,
        PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT,
    ):
        raise ValueError(
            "journal v3 execution requires an authenticated native source"
        )
    if not _pipeline_tail_options_match(&sealed._pipeline_options, pipeline):
        raise ValueError("journal v3 pipeline options differ from generation")
    if not _sealed_background_matches(sealed, pipeline):
        raise ValueError("journal v3 pipeline background differs from generation")
    if (
        not isfinite(pipeline.background._bg.omega)
        or pipeline.background._bg.omega <= 0.0
        or pipeline.background._bg.omega >= 1.0
    ):
        raise ValueError("journal v3 background omega is invalid")
    for profile_index in range(profile_begin, profile_end):
        profile = (<OptimizedProfile> sealed._optimized_profiles[profile_index])._om
        if (
            profile == NULL
            or not isfinite(profile.scale_b)
            or profile.scale_b <= 0.0
            or not isfinite(profile.evparam[<int> p7_MMU])
            or not isfinite(profile.evparam[<int> p7_MLAMBDA])
            or profile.evparam[<int> p7_MLAMBDA] <= 0.0
            or not isfinite(profile.evparam[<int> p7_VMU])
            or not isfinite(profile.evparam[<int> p7_VLAMBDA])
            or profile.evparam[<int> p7_VLAMBDA] <= 0.0
            or not isfinite(profile.evparam[<int> p7_FTAU])
            or not isfinite(profile.evparam[<int> p7_FLAMBDA])
            or profile.evparam[<int> p7_FLAMBDA] <= 0.0
        ):
            raise ValueError("journal v3 optimized-profile statistics are invalid")

    for profile_index in range(profile_begin, profile_end):
        exception_begin = profiles[profile_index].exception_begin
        exception_end = exception_begin + profiles[profile_index].exception_count
        for index in range(exception_begin, exception_end):
            if exceptions[index].route == PLAN7_CONTINUATION_V3_FILTER_SCORES:
                needs_filter = True
            elif exceptions[index].route == PLAN7_CONTINUATION_V3_FORWARD_SCORES:
                needs_forward = True
            elif exceptions[index].route == PLAN7_CONTINUATION_V3_SIMPLE_REGIONS:
                needs_simple = True
            elif exceptions[index].route == PLAN7_CONTINUATION_V3_COMPACT_DOMAINS:
                needs_compact = True
                needs_forward = True
                if exceptions[index].source_domain_index > <uint64_t> 0xffffffff:
                    raise OverflowError(
                        "journal v3 compact source row exceeds uint32"
                    )
    if needs_filter and sealed._filter_scores_seam == NULL:
        raise RuntimeError("journal v3 requires the private filter-score seam")
    if needs_forward and sealed._forward_scores_seam == NULL:
        raise RuntimeError("journal v3 requires the private Forward seam")
    if needs_simple and sealed._simple_regions_seam == NULL:
        raise RuntimeError("journal v3 requires the private simple-region seam")
    if needs_compact:
        if (
            sealed._compact_domains_seam == NULL
            or sealed._compact_tail_fingerprint == NULL
            or journal.generation_tail_fingerprint == 0
        ):
            raise RuntimeError("journal v3 requires the private compact-domain seam")
        with nogil:
            tail_fingerprint = sealed._compact_tail_fingerprint(pipeline._pli)
        if tail_fingerprint != journal.generation_tail_fingerprint:
            raise ValueError("journal v3 compact tail fingerprint differs")
    _v3_preflight_counter_capacity_range(
        journal, sealed, pipeline, profile_begin, profile_end
    )


cdef void _v3_preflight_live_pipeline(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    Pipeline pipeline,
) except *:
    _v3_preflight_live_pipeline_range(
        journal, sealed, pipeline, 0, journal.profile_count
    )


cdef void _v3_preflight_live_pipeline_row(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    Pipeline pipeline,
    size_t profile_index,
) except *:
    if profile_index >= journal.profile_count:
        raise IndexError("journal v3 profile row is out of range")
    _v3_preflight_live_pipeline_range(
        journal, sealed, pipeline, profile_index, profile_index + 1
    )


cdef TopHits _v3_sparse_profile_range_preallocated(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    size_t profile_index,
    uint64_t local_begin,
    uint64_t local_stop,
    bint apply_terminal_certificate,
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    TopHits hits,
    uint64_t *compact_rebased_offsets,
    _compact_consumption_statistics *statistics,
):
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_continuation_journal_v3(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ **> sealed._sequences._refs,
            sealed._sequences._length,
            journal,
            &profiles[profile_index],
            hits._th,
            sealed._filter_scores_seam,
            sealed._forward_scores_seam,
            sealed._forward_backward_scores_seam,
            sealed._simple_regions_seam,
            sealed._compact_domains_seam,
            compact_rebased_offsets,
            statistics,
            local_begin,
            local_stop,
            apply_terminal_certificate,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)
    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _v3_sparse_profile_preallocated(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    size_t profile_index,
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    TopHits hits,
    uint64_t *compact_rebased_offsets,
    _compact_consumption_statistics *statistics,
):
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    return _v3_sparse_profile_range_preallocated(
        journal,
        sealed,
        profile_index,
        0,
        profiles[profile_index].exception_count,
        True,
        pipeline,
        query,
        optimized_profile,
        hits,
        compact_rebased_offsets,
        statistics,
    )


cdef void _v3_complete_row_route_statistics(
    const plan7_continuation_journal_v3 *journal,
    _SealedPostfilterBatch sealed,
    size_t profile_index,
    Pipeline pipeline,
    _compact_consumption_statistics *statistics,
) except *:
    """Add omitted-stage and dense-domain attribution after sparse execution."""
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    cdef const plan7_continuation_journal_v3_certificate *certificates = (
        <const plan7_continuation_journal_v3_certificate *> (
            base + journal.certificates_offset
        )
    )
    cdef const plan7_continuation_journal_v3_exception *exceptions = (
        <const plan7_continuation_journal_v3_exception *> (
            base + journal.exceptions_offset
        )
    )
    cdef const plan7_continuation_journal_v3_profile *profile = (
        &profiles[profile_index]
    )
    cdef const plan7_continuation_journal_v3_certificate *certificate
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef plan7_continuation_journal_row domain
    cdef uint64_t certificate_index
    cdef uint64_t exception_index
    cdef uint64_t domain_index
    cdef uint64_t compact_begin
    cdef uint64_t compact_end
    cdef bint compact_generation_matches = False

    if (
        sealed._compact_tail_fingerprint != NULL
        and sealed._compact_domains_seam != NULL
        and journal.generation_tail_fingerprint != 0
    ):
        with nogil:
            compact_generation_matches = (
                sealed._compact_tail_fingerprint(pipeline._pli)
                == journal.generation_tail_fingerprint
            )

    statistics.requested_profile_index = profile_index
    statistics.journal_row_start = profile.source_domain_begin
    statistics.journal_row_stop = (
        profile.source_domain_begin + profile.source_domain_count
    )
    for certificate_index in range(profile.certificate_count):
        certificate = &certificates[
            profile.certificate_begin + certificate_index
        ]
        statistics.filter_continuation_count += (
            certificate.bias_reject_count + certificate.f2_reject_count
        )
        statistics.forward_continuation_count += certificate.f3_reject_count
        statistics.simple_continuation_count += certificate.no_region_count
        statistics.source_definite_reject_count += (
            certificate.raw_f1_reject_count
        )
        statistics.source_filter_count += (
            certificate.bias_reject_count + certificate.f2_reject_count
        )
        statistics.source_forward_count += certificate.f3_reject_count
        statistics.journal_match_count += certificate.no_region_count
        statistics.journal_no_region_count += certificate.no_region_count
        statistics.source_journal_eligible_count += (
            certificate.no_region_count
        )
        statistics.source_simple_bypass_count += certificate.no_region_count
        statistics.decision_compact_route_not_device += (
            certificate.no_region_count
        )
        statistics.decision_compact_empty += certificate.no_region_count
        if not compact_generation_matches:
            statistics.decision_compact_tail_changed += (
                certificate.no_region_count
            )

    for exception_index in range(profile.exception_count):
        exception = &exceptions[profile.exception_begin + exception_index]
        if exception.route == PLAN7_CONTINUATION_V3_FULL_PIPELINE:
            statistics.source_postfilter_cpu_count += 1
        elif exception.route == PLAN7_CONTINUATION_V3_FILTER_SCORES:
            statistics.source_filter_count += 1
        elif exception.route == PLAN7_CONTINUATION_V3_FORWARD_SCORES:
            statistics.source_forward_count += 1
        if not (
            exception.payload_flags & PLAN7_CONTINUATION_V3_HAS_DOMAIN
        ):
            continue
        memcpy(
            &domain,
            exception.domain_record,
            sizeof(plan7_continuation_journal_row),
        )
        statistics.journal_match_count += 1
        if (
            domain.domain_status != DOMAIN_OK
            or domain.domain_route == DOMAIN_CPU_REQUIRED
            or domain.has_own_scales
            or domain.uncertain_count != 0
            or domain.multidomain_count != 0
        ):
            statistics.journal_cpu_required_count += 1
            continue
        if domain.domain_route == DOMAIN_NO_REGIONS:
            statistics.journal_no_region_count += 1
        elif domain.domain_route == DOMAIN_SIMPLE:
            statistics.journal_simple_count += 1
        else:
            statistics.journal_cpu_required_count += 1
            continue
        statistics.source_journal_eligible_count += 1
        if (
            domain.domain_route == DOMAIN_SIMPLE
            and domain.compact_route == PLAN7_CONTINUATION_COMPACT_DEVICE
            and domain.compact_result_count != 0
            and compact_generation_matches
        ):
            continue
        statistics.source_simple_bypass_count += 1
        if (
            domain.domain_route != DOMAIN_SIMPLE
            or domain.compact_route != PLAN7_CONTINUATION_COMPACT_DEVICE
        ):
            statistics.decision_compact_route_not_device += 1
        if domain.compact_result_count == 0:
            statistics.decision_compact_empty += 1
        if not compact_generation_matches:
            statistics.decision_compact_tail_changed += 1
    if statistics.journal_match_count != profile.source_domain_count:
        raise RuntimeError("journal v3 packet domain census is incomplete")


def _search_hmm_sealed_sparse_journal_v3_bound(
    sealed_object,
    Py_ssize_t row,
    Pipeline pipeline,
    bint _return_route_statistics=False,
):
    """Search one row through a reusable, seal-owned sparse v3 packet."""
    cdef _SealedPostfilterBatch sealed
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_profile *profiles
    cdef const plan7_continuation_journal_v3_exception *exceptions
    cdef const plan7_continuation_journal_v3_profile *profile
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef HMM query
    cdef OptimizedProfile optimized_profile
    cdef TopHits hits
    cdef uint64_t *compact_rebased_offsets = NULL
    cdef uint64_t scratch_count = 1
    cdef uint64_t local_index
    cdef _compact_consumption_statistics statistics
    cdef object start_ns = None
    cdef object preflight_start_ns
    cdef object preflight_elapsed_ns
    cdef object core_start_ns
    cdef object core_elapsed_ns
    cdef object statistics_start_ns
    cdef object statistics_elapsed_ns = 0
    global _v3_consumer_call_count
    global _v3_consumer_preflight_ns
    global _v3_consumer_core_ns
    global _v3_consumer_statistics_ns
    global _v3_consumer_certificate_visits
    global _v3_consumer_exception_visits

    if _return_route_statistics:
        start_ns = _time.perf_counter_ns()
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if sealed._journal_v3 == NULL:
        raise TypeError("sealed batch has no sparse journal v3")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")

    preflight_start_ns = _time.perf_counter_ns()
    _v3_preflight_live_pipeline_row(
        sealed._journal_v3, sealed, pipeline, <size_t> row
    )
    preflight_elapsed_ns = _time.perf_counter_ns() - preflight_start_ns
    core_start_ns = _time.perf_counter_ns()
    query = (<HMM> sealed._queries[row]).copy()
    optimized_profile = <OptimizedProfile> sealed._optimized_profiles[row]
    hits = TopHits(query)
    base = <const uint8_t *> sealed._journal_v3
    profiles = <const plan7_continuation_journal_v3_profile *> (
        base + sealed._journal_v3.profiles_offset
    )
    exceptions = <const plan7_continuation_journal_v3_exception *> (
        base + sealed._journal_v3.exceptions_offset
    )
    profile = &profiles[row]
    for local_index in range(profile.exception_count):
        exception = &exceptions[profile.exception_begin + local_index]
        if exception.compact_result_count >= scratch_count:
            if exception.compact_result_count == <uint64_t> (<size_t> -1):
                raise OverflowError("journal v3 compact scratch size overflows")
            scratch_count = exception.compact_result_count + 1
    if scratch_count > <uint64_t> ((<size_t> -1) // sizeof(uint64_t)):
        raise OverflowError("journal v3 compact scratch size overflows")
    compact_rebased_offsets = <uint64_t *> malloc(
        <size_t> scratch_count * sizeof(uint64_t)
    )
    if compact_rebased_offsets == NULL:
        raise MemoryError("journal v3 compact scratch allocation failed")
    _v3_zero_statistics(&statistics)
    try:
        hits = _v3_sparse_profile_preallocated(
            sealed._journal_v3,
            sealed,
            <size_t> row,
            pipeline,
            query,
            optimized_profile,
            hits,
            compact_rebased_offsets,
            &statistics,
        )
        core_elapsed_ns = _time.perf_counter_ns() - core_start_ns
        if _return_route_statistics:
            statistics_start_ns = _time.perf_counter_ns()
            _v3_complete_row_route_statistics(
                sealed._journal_v3,
                sealed,
                <size_t> row,
                pipeline,
                &statistics,
            )
            statistics_elapsed_ns = (
                _time.perf_counter_ns() - statistics_start_ns
            )
        _v3_consumer_call_count += 1
        _v3_consumer_preflight_ns += <uint64_t> preflight_elapsed_ns
        _v3_consumer_core_ns += <uint64_t> core_elapsed_ns
        _v3_consumer_statistics_ns += <uint64_t> statistics_elapsed_ns
        _v3_consumer_certificate_visits += profile.certificate_count
        _v3_consumer_exception_visits += profile.exception_count
        _record_filter_tail_audit(&statistics)
        return _sealed_search_result(
            hits,
            &statistics,
            False,
            _return_route_statistics,
            "journal",
            start_ns,
            sealed._telemetry_session_id,
            sealed._telemetry_selection_id,
            sealed._telemetry_batch_generation,
        )
    finally:
        free(compact_rebased_offsets)


def _search_hmm_sealed_sparse_journal_v3_shard_bound(
    sealed_object,
    Py_ssize_t row,
    Py_ssize_t exception_begin,
    Py_ssize_t exception_end,
    Pipeline pipeline,
):
    """Search one exception-boundary shard for private parallel merging.

    Each shard owns complete certificate/exception units.  The final shard
    additionally owns the terminal certificate and final target-length state.
    This is deliberately restricted to a fresh, deterministically reseeded
    pipeline so HMMER's stochastic domain traces remain target-independent.
    """
    cdef _SealedPostfilterBatch sealed
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_profile *profiles
    cdef const plan7_continuation_journal_v3_exception *exceptions
    cdef const plan7_continuation_journal_v3_profile *profile
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef HMM query
    cdef OptimizedProfile optimized_profile
    cdef TopHits hits
    cdef uint64_t *compact_rebased_offsets = NULL
    cdef uint64_t scratch_count = 1
    cdef uint64_t local_index
    cdef uint64_t dynamic_z_offset = 0
    cdef bint apply_terminal_certificate
    cdef bint dynamic_z_applicable
    cdef bint dynamic_z_restored
    cdef _compact_consumption_statistics statistics

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if sealed._journal_v3 == NULL:
        raise TypeError("sealed batch has no sparse journal v3")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")
    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    if not pipeline._pli.do_reseeding:
        raise ValueError("journal v3 sharding requires deterministic reseeding")
    if (
        pipeline._pli.nmodels != 0
        or pipeline._pli.nseqs != 0
        or pipeline._pli.nres != 0
        or pipeline._pli.nnodes != 0
        or pipeline._pli.n_past_msv != 0
        or pipeline._pli.n_past_bias != 0
        or pipeline._pli.n_past_vit != 0
        or pipeline._pli.n_past_fwd != 0
    ):
        raise ValueError("journal v3 sharding requires a fresh pipeline")

    base = <const uint8_t *> sealed._journal_v3
    profiles = <const plan7_continuation_journal_v3_profile *> (
        base + sealed._journal_v3.profiles_offset
    )
    exceptions = <const plan7_continuation_journal_v3_exception *> (
        base + sealed._journal_v3.exceptions_offset
    )
    profile = &profiles[row]
    if (
        exception_begin < 0
        or exception_end <= exception_begin
        or <uint64_t> exception_end > profile.exception_count
    ):
        raise IndexError("journal v3 shard exception range is invalid")

    _v3_preflight_live_pipeline_row(
        sealed._journal_v3, sealed, pipeline, <size_t> row
    )
    query = (<HMM> sealed._queries[row]).copy()
    # Shards of one profile may execute concurrently. Reconfiguration mutates
    # an optimized profile's target-length transitions, so each shard must own
    # an independent copy rather than racing on the seal's shared profile.
    optimized_profile = (
        <OptimizedProfile> sealed._optimized_profiles[row]
    ).copy()
    hits = TopHits(query)
    for local_index in range(
        <uint64_t> exception_begin, <uint64_t> exception_end
    ):
        exception = &exceptions[profile.exception_begin + local_index]
        if exception.compact_result_count >= scratch_count:
            if exception.compact_result_count == <uint64_t> (<size_t> -1):
                raise OverflowError("journal v3 compact scratch size overflows")
            scratch_count = exception.compact_result_count + 1
    if scratch_count > <uint64_t> ((<size_t> -1) // sizeof(uint64_t)):
        raise OverflowError("journal v3 compact scratch size overflows")
    compact_rebased_offsets = <uint64_t *> malloc(
        <size_t> scratch_count * sizeof(uint64_t)
    )
    if compact_rebased_offsets == NULL:
        raise MemoryError("journal v3 compact scratch allocation failed")
    apply_terminal_certificate = (
        <uint64_t> exception_end == profile.exception_count
    )
    if exception_begin != 0:
        dynamic_z_offset = exceptions[
            profile.exception_begin + <uint64_t> exception_begin - 1
        ].sequence_index + 1
    _v3_zero_statistics(&statistics)
    try:
        hits = _v3_sparse_profile_range_preallocated(
            sealed._journal_v3,
            sealed,
            <size_t> row,
            <uint64_t> exception_begin,
            <uint64_t> exception_end,
            apply_terminal_certificate,
            pipeline,
            query,
            optimized_profile,
            hits,
            compact_rebased_offsets,
            &statistics,
        )
        dynamic_z_applicable = (
            pipeline._pli.Z_setby == p7_ZSETBY_NTARGETS
            and pipeline._pli.mode == p7_SEARCH_SEQS
        )
        dynamic_z_restored = (
            dynamic_z_applicable
            and pipeline._pli.Z == <double> pipeline._pli.nseqs
        )
        _record_filter_tail_audit(&statistics)
        _record_shard_z_audit(
            <uint64_t> exception_begin,
            <uint64_t> exception_end,
            apply_terminal_certificate,
            pipeline._pli.nseqs,
            dynamic_z_applicable,
            dynamic_z_offset,
            dynamic_z_restored,
        )
        return hits
    finally:
        free(compact_rebased_offsets)


def _sealed_ga_cutoff_census_bound(
    sealed_object,
    Py_ssize_t row,
    Pipeline pipeline,
    bint include_indices=False,
):
    """Census certified GA rejections without changing continuation state.

    The evaluator is deliberately restricted to authenticated compact-domain
    rows.  Those rows contain every isolated-domain Forward score needed to
    upper-bound both alternatives used by HMMER's final target score.  It does
    not claim that a whole-target Forward score alone bounds reconstruction.
    """
    cdef _SealedPostfilterBatch sealed
    cdef const uint8_t *base
    cdef const plan7_continuation_journal_v3_profile *profiles
    cdef const plan7_continuation_journal_v3_exception *exceptions
    cdef const plan7_continuation_journal_v3_profile *profile
    cdef const plan7_continuation_journal_v3_exception *exception
    cdef const plan7_domain_rescore_result *all_results
    cdef const plan7_domain_rescore_result *result
    cdef const ESL_SQ **targets
    cdef const ESL_SQ *target
    cdef OptimizedProfile optimized_profile
    cdef P7_OPROFILE *om
    cdef plan7_continuation_journal_row domain
    cdef uint64_t local_index
    cdef uint64_t compact_index
    cdef uint64_t evaluable_row_count = 0
    cdef uint64_t region_count = 0
    cdef uint64_t region_cells = 0
    cdef uint64_t domain_below_count = 0
    cdef uint64_t domain_below_cells = 0
    cdef uint64_t target_below_count = 0
    cdef uint64_t target_below_region_count = 0
    cdef uint64_t target_below_region_cells = 0
    cdef uint64_t whole_below_count = 0
    cdef uint64_t reconstruction_below_count = 0
    cdef uint64_t cell_count
    cdef uint64_t scratch
    cdef int length
    cdef float null_score
    cdef float target_cutoff
    cdef float domain_cutoff
    cdef double input_error = <double> _GA_COMPACT_INPUT_EPSILON_NATS
    cdef double round_error = <double> _GA_COMPACT_ROUND_EPSILON_BITS
    cdef double whole_upper
    cdef double reconstruction_nats
    cdef double reconstruction_upper
    cdef double target_upper
    cdef double domain_upper
    cdef double upper_envsc
    cdef object preflight_start_ns
    cdef object preflight_ns
    cdef object scan_start_ns
    cdef object scan_ns
    cdef list certified_indices = []

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if sealed._journal_v3 == NULL:
        raise TypeError("GA census requires a sparse journal v3 packet")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")

    preflight_start_ns = _time.perf_counter_ns()
    _v3_preflight_live_pipeline_row(
        sealed._journal_v3, sealed, pipeline, <size_t> row
    )
    preflight_ns = _time.perf_counter_ns() - preflight_start_ns

    optimized_profile = <OptimizedProfile> sealed._optimized_profiles[row]
    om = optimized_profile._om
    if pipeline._pli.use_bit_cutoffs != p7H_GA:
        raise ValueError("GA census requires gathering bit cutoffs")
    if om == NULL:
        raise ValueError("GA census optimized profile is unavailable")
    target_cutoff = om.cutoff[<int> p7_GA1]
    domain_cutoff = om.cutoff[<int> p7_GA2]
    if (
        not isfinite(target_cutoff)
        or not isfinite(domain_cutoff)
        or target_cutoff == p7_CUTOFF_UNSET
        or domain_cutoff == p7_CUTOFF_UNSET
    ):
        raise ValueError("GA census profile has no finite gathering cutoffs")

    base = <const uint8_t *> sealed._journal_v3
    profiles = <const plan7_continuation_journal_v3_profile *> (
        base + sealed._journal_v3.profiles_offset
    )
    exceptions = <const plan7_continuation_journal_v3_exception *> (
        base + sealed._journal_v3.exceptions_offset
    )
    all_results = <const plan7_domain_rescore_result *> (
        base + sealed._journal_v3.compact_results_offset
    )
    targets = <const ESL_SQ **> sealed._sequences._refs
    profile = &profiles[row]

    scan_start_ns = _time.perf_counter_ns()
    for local_index in range(profile.exception_count):
        exception = &exceptions[profile.exception_begin + local_index]
        if exception.route != PLAN7_CONTINUATION_V3_COMPACT_DOMAINS:
            continue
        if (
            not (exception.payload_flags & PLAN7_CONTINUATION_V3_HAS_COMPACT)
            or exception.compact_result_count == 0
            or exception.sequence_index >= sealed._sequences._length
        ):
            raise RuntimeError("GA census compact exception is incomplete")
        evaluable_row_count += 1
        target = targets[exception.sequence_index]
        if target == NULL or target.n <= 0 or target.n > 100000:
            raise RuntimeError("GA census target length is invalid")
        null_score = _ga_null_score_for_length(target.n)
        if not isfinite(null_score):
            raise RuntimeError("GA census null score is invalid")
        memcpy(
            &domain,
            exception.domain_record,
            sizeof(plan7_continuation_journal_row),
        )
        if not isfinite(domain.fwdsc):
            raise RuntimeError("GA census target Forward score is invalid")

        # HMMER's whole-target final score subtracts a nonnegative null2 bias.
        # The compact seam permits at most input_error nats in fwdsc and then
        # budgets round_error bits for the final score expression.
        whole_upper = (
            (<double> domain.fwdsc + input_error - <double> null_score)
            / <double> eslCONST_LOG2
            + round_error
        )
        reconstruction_nats = 0.0
        cell_count = 0
        for compact_index in range(exception.compact_result_count):
            result = &all_results[
                exception.compact_result_begin + compact_index
            ]
            if (
                result.status != PLAN7_DOMAIN_RESCORE_OK
                or result.action != PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                or result.has_own_scales
                or result.profile_index != <uint32_t> row
                or result.sequence_index != exception.sequence_index
                or result.envelope_begin == 0
                or result.envelope_begin > result.envelope_end
                or result.envelope_end > <uint32_t> target.n
                or not isfinite(result.forward_score)
            ):
                raise RuntimeError("GA census compact result is invalid")
            length = <int> (result.envelope_end - result.envelope_begin + 1)
            if not plan7_continuation_journal_v3_checked_multiply(
                <uint64_t> om.M, <uint64_t> length, &scratch
            ):
                raise OverflowError("GA census region cells overflow")
            if not plan7_continuation_journal_v3_checked_add(
                cell_count, scratch, &cell_count
            ):
                raise OverflowError("GA census row cells overflow")

            # The exact per-domain score additionally applies a nonpositive
            # length adjustment and a nonnegative null2 correction.  Omitting
            # both yields a conservative upper bound.
            domain_upper = (
                (<double> result.forward_score + input_error
                 - <double> null_score)
                / <double> eslCONST_LOG2
                + round_error
            )
            if domain_upper < <double> domain_cutoff:
                domain_below_count += 1
                if not plan7_continuation_journal_v3_checked_add(
                    domain_below_cells, scratch, &domain_below_cells
                ):
                    raise OverflowError("GA census domain cells overflow")

            # HMMER reconstruction selects only a subset of domains and adds a
            # nonpositive length term before subtracting nonnegative null2.
            # Summing every positive per-domain input upper bound is therefore
            # conservative even when a selected exact envsc is negative.
            upper_envsc = <double> result.forward_score + input_error
            if upper_envsc > 0.0:
                reconstruction_nats += upper_envsc

        if not plan7_continuation_journal_v3_checked_add(
            region_count, exception.compact_result_count, &region_count
        ) or not plan7_continuation_journal_v3_checked_add(
            region_cells, cell_count, &region_cells
        ):
            raise OverflowError("GA census aggregate cells overflow")
        reconstruction_upper = (
            (reconstruction_nats - <double> null_score)
            / <double> eslCONST_LOG2
            + round_error
        )
        target_upper = (
            whole_upper
            if whole_upper >= reconstruction_upper
            else reconstruction_upper
        )
        if whole_upper < <double> target_cutoff:
            whole_below_count += 1
        if reconstruction_upper < <double> target_cutoff:
            reconstruction_below_count += 1
        if target_upper < <double> target_cutoff:
            target_below_count += 1
            target_below_region_count += exception.compact_result_count
            if not plan7_continuation_journal_v3_checked_add(
                target_below_region_cells,
                cell_count,
                &target_below_region_cells,
            ):
                raise OverflowError("GA census target-reject cells overflow")
            if include_indices:
                certified_indices.append(exception.sequence_index)
    scan_ns = _time.perf_counter_ns() - scan_start_ns

    if target_below_region_count > region_count:
        raise RuntimeError("GA census target region count is inconsistent")
    return {
        "schema_version": 1,
        "mode": "gathering",
        "profile_index": row,
        "target_cutoff_bits": target_cutoff,
        "domain_cutoff_bits": domain_cutoff,
        "input_error_nats": _GA_COMPACT_INPUT_EPSILON_NATS,
        "round_error_bits": _GA_COMPACT_ROUND_EPSILON_BITS,
        "source_domain_rows": profile.source_domain_count,
        "evaluable_target_rows": evaluable_row_count,
        "compact_region_count": region_count,
        "compact_region_cells_per_downstream_pass": region_cells,
        "whole_forward_upper_below_ga_count": whole_below_count,
        "reconstruction_upper_below_ga_count": reconstruction_below_count,
        "certified_target_reject_count": target_below_count,
        "certified_target_reject_region_count": target_below_region_count,
        "certified_target_reject_region_cells_per_downstream_pass": (
            target_below_region_cells
        ),
        "domain_upper_below_ga_count": domain_below_count,
        "domain_upper_below_ga_region_cells_per_downstream_pass": (
            domain_below_cells
        ),
        "certified_target_indices": (
            tuple(certified_indices) if include_indices else None
        ),
        "certified_target_index_payload_bytes": (
            len(certified_indices) * sizeof(uint32_t) if include_indices else 0
        ),
        "native_temporary_bytes": 0,
        "preflight_ns": preflight_ns,
        "scan_ns": scan_ns,
    }


cdef TopHits _v3_dense_reference_profile_preallocated(
    _SealedPostfilterBatch sealed,
    size_t row,
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    TopHits hits,
    uint64_t *compact_rebased_offsets,
    _compact_consumption_statistics *statistics,
):
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t journal_start
    cdef size_t journal_stop
    cdef const uint8_t *postfilter_ptr = NULL
    cdef const uint8_t *forward_ptr = NULL
    cdef const float *special_ptr = NULL
    cdef const uint8_t *journal_ptr = NULL
    cdef const uint8_t *region_ptr = NULL
    cdef const uint8_t *compact_result_ptr = NULL
    cdef const uint8_t *compact_trace_ptr = NULL
    cdef const float *compact_null2_ptr = NULL
    cdef _double_bits generation_f1

    if sealed._direct_v3_source:
        raise TypeError(
            "direct sparse-v3 batches have no dense audit source"
        )
    postfilter_start = <size_t> sealed._postfilter_offsets[row]
    postfilter_stop = <size_t> sealed._postfilter_offsets[row + 1]
    forward_start = <size_t> sealed._forward_offsets[row]
    forward_stop = <size_t> sealed._forward_offsets[row + 1]
    journal_start = <size_t> sealed._journal_profile_offsets[row]
    journal_stop = <size_t> sealed._journal_profile_offsets[row + 1]

    if postfilter_stop != postfilter_start:
        postfilter_ptr = &sealed._postfilter_records[
            postfilter_start * sizeof(_postfilter_result)
        ]
    if forward_stop != forward_start:
        forward_ptr = &sealed._forward_records[
            forward_start * sizeof(_forward_result)
        ]
    if sealed._specials.shape[0]:
        special_ptr = &sealed._specials[0]
    if journal_stop != journal_start:
        journal_ptr = &sealed._journal_rows[
            journal_start * sizeof(plan7_continuation_journal_row)
        ]
    if sealed._journal_regions.shape[0]:
        region_ptr = &sealed._journal_regions[0]
    if sealed._journal_compact_results.shape[0]:
        compact_result_ptr = &sealed._journal_compact_results[0]
    if sealed._journal_compact_traces.shape[0]:
        compact_trace_ptr = &sealed._journal_compact_traces[0]
    if sealed._journal_compact_null2.shape[0]:
        compact_null2_ptr = &sealed._journal_compact_null2[0]
    generation_f1.value = sealed._f1

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_postfilter_forward(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ **> sealed._sequences._refs,
            sealed._sequences._length,
            postfilter_ptr,
            postfilter_stop - postfilter_start,
            forward_ptr,
            forward_stop - forward_start,
            &sealed._special_offsets[forward_start],
            special_ptr,
            &sealed._residue_offsets[0],
            hits._th,
            sealed._filter_scores_seam,
            sealed._forward_scores_seam,
            journal_ptr,
            journal_stop - journal_start,
            &sealed._journal_region_offsets[journal_start],
            region_ptr,
            generation_f1.bits,
            sealed._generation_f2_bits,
            sealed._generation_f3_bits,
            sealed._generation_bias_filter,
            sealed._simple_regions_seam,
            &sealed._journal_compact_row_offsets[journal_start],
            compact_result_ptr,
            &sealed._journal_compact_trace_offsets[0],
            compact_trace_ptr,
            compact_null2_ptr,
            journal_start,
            sealed._generation_tail_fingerprint,
            sealed._compact_tail_fingerprint,
            sealed._compact_domains_seam,
            compact_rebased_offsets,
            statistics,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)
    hits._query = query
    hits._empty = False
    return hits


cdef tuple _v3_make_execution_objects(
    _SealedPostfilterBatch sealed,
):
    cdef list queries = []
    cdef list profiles = []
    cdef list hits = []
    cdef size_t index
    cdef HMM query
    cdef OptimizedProfile profile
    cdef bytes expected_identity
    cdef bytes copied_identity
    for index in range(len(sealed._queries)):
        query = (<HMM> sealed._queries[index]).copy()
        profile = (<OptimizedProfile> sealed._optimized_profiles[index]).copy()
        expected_identity = _semantic_checked_profile_identity_fingerprint(
            <OptimizedProfile> sealed._optimized_profiles[index],
            "sealed",
        )
        copied_identity = _semantic_checked_profile_identity_fingerprint(
            profile,
            "execution",
        )
        if copied_identity != expected_identity:
            raise ValueError("journal v3 optimized-profile copy identity differs")
        queries.append(query)
        profiles.append(profile)
        hits.append(TopHits(query))
    return tuple(queries), tuple(profiles), tuple(hits)


cdef dict _v3_statistics_evidence(
    const _compact_consumption_statistics *statistics,
):
    return {
        "target_count": statistics.target_count,
        "postfilter_record_count": statistics.postfilter_record_count,
        "f1_reject_count": statistics.f1_reject_count,
        "cpu_pipeline_count": statistics.cpu_pipeline_count,
        "definite_reject_count": statistics.definite_reject_count,
        "filter_continuation_count": statistics.filter_continuation_count,
        "forward_continuation_count": statistics.forward_continuation_count,
        "simple_continuation_count": statistics.simple_continuation_count,
        "compact": {
            "attempt_count": statistics.attempt_count,
            "accepted_count": statistics.accepted_count,
            "invalid_retry_count": statistics.invalid_retry_count,
            "threshold_retry_count": statistics.threshold_retry_count,
            "first_attempt": (
                None
                if statistics.attempt_count == 0
                else (
                    statistics.first_row_index,
                    statistics.first_profile_index,
                    statistics.first_sequence_index,
                    statistics.first_domain_count,
                )
            ),
        },
    }


cdef dict _v3_profile_certificate_evidence(
    const plan7_continuation_journal_v3 *journal,
    size_t profile_index,
):
    cdef const uint8_t *base = <const uint8_t *> journal
    cdef const plan7_continuation_journal_v3_profile *profiles = (
        <const plan7_continuation_journal_v3_profile *> (
            base + journal.profiles_offset
        )
    )
    cdef const plan7_continuation_journal_v3_certificate *certificates = (
        <const plan7_continuation_journal_v3_certificate *> (
            base + journal.certificates_offset
        )
    )
    cdef const plan7_continuation_journal_v3_exception *exceptions = (
        <const plan7_continuation_journal_v3_exception *> (
            base + journal.exceptions_offset
        )
    )
    cdef const plan7_continuation_journal_v3_profile *profile = (
        &profiles[profile_index]
    )
    cdef const plan7_continuation_journal_v3_certificate *certificate
    cdef uint64_t target_delta = 0
    cdef uint64_t residue_delta = 0
    cdef uint64_t exception_residues = 0
    cdef uint64_t before_f1 = 0
    cdef uint64_t raw_f1 = 0
    cdef uint64_t bias = 0
    cdef uint64_t f2 = 0
    cdef uint64_t f3 = 0
    cdef uint64_t no_region = 0
    cdef uint64_t past_msv = 0
    cdef uint64_t past_bias = 0
    cdef uint64_t past_vit = 0
    cdef uint64_t past_fwd = 0
    cdef uint64_t index

    for index in range(profile.certificate_count):
        certificate = &certificates[profile.certificate_begin + index]
        target_delta += certificate.target_delta
        residue_delta += certificate.residue_delta
        before_f1 += certificate.before_f1_count
        raw_f1 += certificate.raw_f1_reject_count
        bias += certificate.bias_reject_count
        f2 += certificate.f2_reject_count
        f3 += certificate.f3_reject_count
        no_region += certificate.no_region_count
        past_msv += certificate.n_past_msv_delta
        past_bias += certificate.n_past_bias_delta
        past_vit += certificate.n_past_vit_delta
        past_fwd += certificate.n_past_fwd_delta
    for index in range(profile.exception_count):
        exception_residues += exceptions[
            profile.exception_begin + index
        ].residue_delta
    if (
        target_delta + profile.exception_count != profile.target_count
        or residue_delta + exception_residues != profile.total_residues
        or before_f1 + raw_f1 + bias + f2 + f3 + no_region != target_delta
    ):
        raise AssertionError("journal v3 certificate partition does not reconcile")
    return {
        "omitted_targets": target_delta,
        "omitted_residues": residue_delta,
        "exception_count": profile.exception_count,
        "exception_residues": exception_residues,
        "stages": {
            "before_f1": before_f1,
            "raw_f1": raw_f1,
            "bias": bias,
            "f2": f2,
            "f3": f3,
            "no_region": no_region,
        },
        "promotions": {
            "n_past_msv": past_msv,
            "n_past_bias": past_bias,
            "n_past_vit": past_vit,
            "n_past_fwd": past_fwd,
        },
    }


cdef dict _v3_reconcile_route_statistics(
    const _compact_consumption_statistics *dense,
    const _compact_consumption_statistics *sparse,
    dict certificate,
):
    cdef dict stages = certificate["stages"]
    cdef dict dense_evidence = _v3_statistics_evidence(dense)
    cdef dict sparse_evidence = _v3_statistics_evidence(sparse)
    cdef dict expected = {
        "target_count": sparse.target_count,
        "postfilter_record_count": sparse.postfilter_record_count,
        "f1_reject_count": sparse.f1_reject_count,
        "cpu_pipeline_count": sparse.cpu_pipeline_count,
        "definite_reject_count": sparse.definite_reject_count,
        "filter_continuation_count": (
            sparse.filter_continuation_count
            + stages["bias"] + stages["f2"]
        ),
        "forward_continuation_count": (
            sparse.forward_continuation_count + stages["f3"]
        ),
        "simple_continuation_count": (
            sparse.simple_continuation_count + stages["no_region"]
        ),
        "compact": sparse_evidence["compact"],
    }
    if dense_evidence != expected:
        raise AssertionError("dense v2 and sparse v3 routes do not reconcile")
    return {"equal": True, "dense": dense_evidence, "expected": expected}


cdef bytes _v3_table_bytes(TopHits hits, str table_format):
    cdef object stream = _io.BytesIO()
    hits.write(stream, format=table_format, header=True)
    return stream.getvalue()


def _consume_continuation_journal_v3_bound(
    capsule,
    sealed_object,
    Pipeline pipeline,
):
    """Consume one validated batch-wide v3 debug capsule through the sparse path.

    Production row searches use the reusable packet owned by their sealed
    batch; this one-shot entry remains the batch-wide proof/debug boundary.
    """
    cdef _SealedPostfilterBatch sealed
    cdef plan7_continuation_journal_v3 *journal
    cdef plan7_continuation_journal_v3_owner *owner = NULL
    cdef uint64_t *compact_rebased_offsets = NULL
    cdef tuple execution
    cdef tuple queries
    cdef tuple profiles
    cdef tuple hits
    cdef list evidence = []
    cdef _compact_consumption_statistics statistics
    cdef size_t profile_index
    cdef uint64_t packet_bytes
    cdef object start_ns
    cdef bytes prestate
    cdef dict row_statistics
    cdef bint claimed = False

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    journal = _v3_validate_capsule(capsule, sealed, &owner, False)
    _v3_preflight_live_pipeline(journal, sealed, pipeline)
    execution = _v3_make_execution_objects(sealed)
    queries, profiles, hits = execution
    if journal.profile_count:
        prestate = _semantic_pipeline_state_encoding(
            pipeline, <OptimizedProfile> profiles[0], True
        )
    else:
        prestate = _semantic_pipeline_state_encoding(pipeline, None, False)
    start_ns = _time.perf_counter_ns()
    if (
        journal.compact_result_count
        > <uint64_t> ((<size_t> -1) // sizeof(uint64_t)) - 1
    ):
        raise OverflowError("journal v3 compact scratch size overflows")
    compact_rebased_offsets = <uint64_t *> malloc(
        (<size_t> journal.compact_result_count + 1) * sizeof(uint64_t)
    )
    if compact_rebased_offsets == NULL:
        raise MemoryError("journal v3 compact scratch allocation failed")
    packet_bytes = journal.total_bytes
    try:
        _v3_claim_capsule(capsule, journal, owner, &claimed)
        for profile_index in range(journal.profile_count):
            _v3_zero_statistics(&statistics)
            _v3_sparse_profile_preallocated(
                journal,
                sealed,
                profile_index,
                pipeline,
                <HMM> queries[profile_index],
                <OptimizedProfile> profiles[profile_index],
                <TopHits> hits[profile_index],
                compact_rebased_offsets,
                &statistics,
            )
            row_statistics = _v3_statistics_evidence(&statistics)
            row_statistics["certificate"] = (
                _v3_profile_certificate_evidence(journal, profile_index)
            )
            evidence.append(row_statistics)
        elapsed_ns = _time.perf_counter_ns() - start_ns
        return {
            "schema_version": PLAN7_CONTINUATION_JOURNAL_V3_VERSION,
            "packet_bytes": packet_bytes,
            "elapsed_ns": elapsed_ns,
            "prestate_sha256": _hashlib.sha256(prestate).hexdigest(),
            "hits": hits,
            "profiles": profiles,
            "rows": tuple(evidence),
        }
    finally:
        free(compact_rebased_offsets)
        if claimed:
            free(journal)
            free(owner)


def _audit_continuation_journal_v3_bound(
    capsule,
    sealed_object,
    Pipeline dense_pipeline,
    Pipeline sparse_pipeline,
):
    """Run dense v2 and sparse v3 on independent state and assert equality."""
    cdef _SealedPostfilterBatch sealed
    cdef plan7_continuation_journal_v3 *journal
    cdef plan7_continuation_journal_v3_owner *owner = NULL
    cdef uint64_t *dense_scratch = NULL
    cdef uint64_t *sparse_scratch = NULL
    cdef tuple dense_execution
    cdef tuple sparse_execution
    cdef tuple dense_queries
    cdef tuple dense_profiles
    cdef tuple dense_hits
    cdef tuple sparse_queries
    cdef tuple sparse_profiles
    cdef tuple sparse_hits
    cdef _compact_consumption_statistics dense_statistics
    cdef _compact_consumption_statistics sparse_statistics
    cdef list row_evidence = []
    cdef dict comparison
    cdef dict dense_stats
    cdef dict sparse_stats
    cdef dict certificate
    cdef dict route_reconciliation
    cdef bytes dense_targets
    cdef bytes sparse_targets
    cdef bytes dense_domains
    cdef bytes sparse_domains
    cdef bytes left_prestate
    cdef bytes right_prestate
    cdef size_t profile_index
    cdef uint64_t packet_bytes
    cdef object start_ns
    cdef bint claimed = False

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    if (
        dense_pipeline is sparse_pipeline
        or dense_pipeline._pli == sparse_pipeline._pli
        or dense_pipeline.background is sparse_pipeline.background
        or dense_pipeline.background._bg == sparse_pipeline.background._bg
    ):
        raise ValueError("dense and sparse audit pipelines must be independent")
    sealed = <_SealedPostfilterBatch> sealed_object
    journal = _v3_validate_capsule(capsule, sealed, &owner, True)
    _v3_preflight_live_pipeline(journal, sealed, dense_pipeline)
    _v3_preflight_live_pipeline(journal, sealed, sparse_pipeline)

    dense_execution = _v3_make_execution_objects(sealed)
    sparse_execution = _v3_make_execution_objects(sealed)
    dense_queries, dense_profiles, dense_hits = dense_execution
    sparse_queries, sparse_profiles, sparse_hits = sparse_execution
    if journal.profile_count:
        left_prestate = _semantic_pipeline_state_encoding(
            dense_pipeline, <OptimizedProfile> dense_profiles[0], True
        )
        right_prestate = _semantic_pipeline_state_encoding(
            sparse_pipeline, <OptimizedProfile> sparse_profiles[0], True
        )
    else:
        left_prestate = _semantic_pipeline_state_encoding(
            dense_pipeline, None, False
        )
        right_prestate = _semantic_pipeline_state_encoding(
            sparse_pipeline, None, False
        )
    if left_prestate != right_prestate:
        raise ValueError("dense and sparse journal v3 prestates differ")
    start_ns = _time.perf_counter_ns()
    if (
        journal.compact_result_count
        > <uint64_t> ((<size_t> -1) // sizeof(uint64_t)) - 1
    ):
        raise OverflowError("journal v3 compact scratch size overflows")
    dense_scratch = <uint64_t *> malloc(
        (<size_t> journal.compact_result_count + 1) * sizeof(uint64_t)
    )
    sparse_scratch = <uint64_t *> malloc(
        (<size_t> journal.compact_result_count + 1) * sizeof(uint64_t)
    )
    if dense_scratch == NULL or sparse_scratch == NULL:
        free(dense_scratch)
        free(sparse_scratch)
        raise MemoryError("journal v3 dual compact scratch allocation failed")

    packet_bytes = journal.total_bytes
    try:
        _v3_claim_capsule(capsule, journal, owner, &claimed)
        for profile_index in range(journal.profile_count):
            _v3_zero_statistics(&dense_statistics)
            _v3_zero_statistics(&sparse_statistics)
            _v3_dense_reference_profile_preallocated(
                sealed,
                profile_index,
                dense_pipeline,
                <HMM> dense_queries[profile_index],
                <OptimizedProfile> dense_profiles[profile_index],
                <TopHits> dense_hits[profile_index],
                dense_scratch,
                &dense_statistics,
            )
            _v3_sparse_profile_preallocated(
                journal,
                sealed,
                profile_index,
                sparse_pipeline,
                <HMM> sparse_queries[profile_index],
                <OptimizedProfile> sparse_profiles[profile_index],
                <TopHits> sparse_hits[profile_index],
                sparse_scratch,
                &sparse_statistics,
            )
            comparison = _semantic_dual_state_compare_bound(
                dense_pipeline,
                sparse_pipeline,
                dense_hits[profile_index],
                sparse_hits[profile_index],
                dense_profiles[profile_index],
                sparse_profiles[profile_index],
            )
            dense_targets = _v3_table_bytes(
                <TopHits> dense_hits[profile_index], "targets"
            )
            sparse_targets = _v3_table_bytes(
                <TopHits> sparse_hits[profile_index], "targets"
            )
            dense_domains = _v3_table_bytes(
                <TopHits> dense_hits[profile_index], "domains"
            )
            sparse_domains = _v3_table_bytes(
                <TopHits> sparse_hits[profile_index], "domains"
            )
            dense_stats = _v3_statistics_evidence(&dense_statistics)
            sparse_stats = _v3_statistics_evidence(&sparse_statistics)
            certificate = _v3_profile_certificate_evidence(
                journal, profile_index
            )
            route_reconciliation = _v3_reconcile_route_statistics(
                &dense_statistics, &sparse_statistics, certificate
            )
            if (
                not comparison["pipeline"]["equal"]
                or not comparison["tophits"]["equal"]
                or dense_targets != sparse_targets
                or dense_domains != sparse_domains
                or dense_stats["compact"] != sparse_stats["compact"]
            ):
                raise AssertionError(
                    f"dense v2 and sparse v3 differ at profile {profile_index}"
                )
            row_evidence.append({
                "profile_index": profile_index,
                "semantic": comparison,
                "targets_sha256": _hashlib.sha256(dense_targets).hexdigest(),
                "domains_sha256": _hashlib.sha256(dense_domains).hexdigest(),
                "targets_bytes": len(dense_targets),
                "domains_bytes": len(dense_domains),
                "dense": dense_stats,
                "sparse": sparse_stats,
                "certificate": certificate,
                "route_reconciliation": route_reconciliation,
                "cpu_tail_simd": {
                    "eligible_rows": sparse_statistics.cpu_tail_simd_eligible_rows,
                    "groups": sparse_statistics.cpu_tail_simd_groups,
                    "rows": sparse_statistics.cpu_tail_simd_rows,
                    "scalar_tail_rows": sparse_statistics.cpu_tail_simd_scalar_tail_rows,
                    "fallback_groups": sparse_statistics.cpu_tail_simd_fallback_groups,
                    "kernel_ns": sparse_statistics.cpu_tail_simd_kernel_ns,
                },
                "cpu_tail_filter_simd": {
                    "eligible_rows": sparse_statistics.cpu_tail_filter_simd_eligible_rows,
                    "groups": sparse_statistics.cpu_tail_filter_simd_groups,
                    "rows": sparse_statistics.cpu_tail_filter_simd_rows,
                    "scalar_tail_rows": sparse_statistics.cpu_tail_filter_simd_scalar_tail_rows,
                    "fallback_groups": sparse_statistics.cpu_tail_filter_simd_fallback_groups,
                    "fallback_rows": sparse_statistics.cpu_tail_filter_simd_fallback_rows,
                    "f3_survivor_rows": sparse_statistics.cpu_tail_filter_simd_f3_survivor_rows,
                    "backward_groups": sparse_statistics.cpu_tail_filter_simd_backward_groups,
                    "forward_kernel_ns": sparse_statistics.cpu_tail_filter_simd_forward_kernel_ns,
                    "backward_kernel_ns": sparse_statistics.cpu_tail_filter_simd_backward_kernel_ns,
                },
            })
        elapsed_ns = _time.perf_counter_ns() - start_ns
        return {
            "schema_version": PLAN7_CONTINUATION_JOURNAL_V3_VERSION,
            "equal": True,
            "packet_bytes": packet_bytes,
            "elapsed_ns": elapsed_ns,
            "prestate_sha256": _hashlib.sha256(left_prestate).hexdigest(),
            "dense_hits": dense_hits,
            "sparse_hits": sparse_hits,
            "dense_profiles": dense_profiles,
            "sparse_profiles": sparse_profiles,
            "rows": tuple(row_evidence),
        }
    finally:
        free(dense_scratch)
        free(sparse_scratch)
        if claimed:
            free(journal)
            free(owner)


def _v3_test_set_pipeline_counter_bound(
    Pipeline pipeline,
    str field,
    uint64_t value,
):
    """Set one accounting field for private overflow-preflight tests."""
    cdef uint64_t previous
    if field == "nmodels":
        previous = pipeline._pli.nmodels
        pipeline._pli.nmodels = value
    elif field == "nres":
        previous = pipeline._pli.nres
        pipeline._pli.nres = value
    elif field == "nnodes":
        previous = pipeline._pli.nnodes
        pipeline._pli.nnodes = value
    elif field == "n_past_msv":
        previous = pipeline._pli.n_past_msv
        pipeline._pli.n_past_msv = value
    elif field == "n_past_bias":
        previous = pipeline._pli.n_past_bias
        pipeline._pli.n_past_bias = value
    elif field == "n_past_vit":
        previous = pipeline._pli.n_past_vit
        pipeline._pli.n_past_vit = value
    elif field == "n_past_fwd":
        previous = pipeline._pli.n_past_fwd
        pipeline._pli.n_past_fwd = value
    else:
        raise ValueError("unknown journal v3 test counter")
    return previous


def _v3_host_float_environment_attested_bound():
    """Expose the sparse-consumer host FP gate for private tests."""
    cdef bint attested
    with nogil:
        attested = plan7_gpu_pipeline_host_environment_attested() == 1
    return bool(attested)


def _semantic_test_swapped_tophits_order_encoding_bound(TopHits hits):
    """Return a temporary swapped-order encoding and restore the input."""
    cdef P7_HIT *temporary
    if type(hits) is not _pyhmmer.plan7.TopHits:
        raise TypeError("hits must be exactly astra_pyhmmer.plan7.TopHits")
    if hits._th == NULL or hits._th.N < 2 or hits._th.hit == NULL:
        raise ValueError("TopHits needs at least two ordered hits")
    temporary = hits._th.hit[0]
    hits._th.hit[0] = hits._th.hit[1]
    hits._th.hit[1] = temporary
    try:
        return _semantic_tophits_encoding(hits)
    finally:
        temporary = hits._th.hit[0]
        hits._th.hit[0] = hits._th.hit[1]
        hits._th.hit[1] = temporary


def _semantic_test_dirty_reusable_state_rejected_bound(
    Pipeline pipeline,
    str field,
):
    """Temporarily dirty one reuse cursor and prove the oracle rejects it."""
    cdef int *value = NULL
    cdef int previous
    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
    if pipeline._pli == NULL or pipeline._pli.ddef == NULL:
        raise ValueError("pipeline domain-definition state is unavailable")
    if field == "sp.nsamples":
        value = &pipeline._pli.ddef.sp.nsamples
    elif field == "sp.n":
        value = &pipeline._pli.ddef.sp.n
    elif field == "sp.nc":
        value = &pipeline._pli.ddef.sp.nc
    elif field == "sp.nsigc":
        value = &pipeline._pli.ddef.sp.nsigc
    elif field == "tr.N":
        value = &pipeline._pli.ddef.tr.N
    elif field == "tr.M":
        value = &pipeline._pli.ddef.tr.M
    elif field == "tr.L":
        value = &pipeline._pli.ddef.tr.L
    elif field == "tr.ndom":
        value = &pipeline._pli.ddef.tr.ndom
    elif field == "gtr.N":
        value = &pipeline._pli.ddef.gtr.N
    elif field == "gtr.M":
        value = &pipeline._pli.ddef.gtr.M
    elif field == "gtr.L":
        value = &pipeline._pli.ddef.gtr.L
    elif field == "gtr.ndom":
        value = &pipeline._pli.ddef.gtr.ndom
    else:
        raise ValueError("unknown reusable-state field")
    previous = value[0]
    value[0] = 1
    try:
        try:
            _semantic_pipeline_state_encoding(pipeline, None, False)
        except ValueError:
            return True
        return False
    finally:
        value[0] = previous
