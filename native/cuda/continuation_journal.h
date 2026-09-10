#ifndef PLAN7_GPU_CONTINUATION_JOURNAL_H
#define PLAN7_GPU_CONTINUATION_JOURNAL_H

#include <stddef.h>
#include <stdint.h>

#include "domain_rescore_cuda.h"
#include "postfilter_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This is an in-process, one-shot transport between plan7_gpu._native and
 * plan7_gpu._pipeline.  Every address is represented as an offset from the
 * beginning of one allocation; the capsule that owns the allocation is the
 * only Python-visible object.  The integrity tag detects accidental damage
 * or object mixing; it is not a cryptographic boundary against deliberate
 * ctypes use or calls into package-private underscore APIs. */
enum plan7_continuation_journal_abi {
  PLAN7_CONTINUATION_JOURNAL_VERSION = 2,
  PLAN7_CONTINUATION_JOURNAL_MAGIC = 0x504a4e4c,
  PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE = 32
};

enum plan7_continuation_compact_route {
  PLAN7_CONTINUATION_COMPACT_NONE = 0,
  PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED = 1,
  PLAN7_CONTINUATION_COMPACT_DEVICE = 2
};

#define PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME \
  "plan7_gpu._native._continuation_journal_v2"
#define PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME \
  "plan7_gpu._pipeline._consumed_journal_v2"

typedef struct plan7_continuation_journal_row {
  uint32_t profile_index;
  uint32_t sequence_index;
  float usc;
  float filtersc;
  float vfsc;
  float fwdsc;
  float backward_score;
  float nexpected;
  uint32_t uncertain_count;
  uint32_t region_count;
  uint32_t multidomain_count;
  uint8_t postfilter_status;
  uint8_t postfilter_action;
  uint8_t forward_status;
  uint8_t forward_action;
  uint8_t domain_status;
  uint8_t domain_route;
  uint8_t has_own_scales;
  uint8_t reserved;
  uint32_t compact_result_count;
  uint8_t compact_route;
  uint8_t reserved2[3];
  uint32_t reserved3;
} plan7_continuation_journal_row;

/* Version 3 is an additive sparse-accounting ABI.  Version 2 above remains
 * the audit/default transport and is intentionally unchanged.  A production
 * sparse request may construct v3 from a transient native staging bundle;
 * that path never allocates or retains a dense v2 packet. */
enum plan7_continuation_journal_v3_abi {
  PLAN7_CONTINUATION_JOURNAL_V3_VERSION = 3,
  PLAN7_CONTINUATION_JOURNAL_V3_MAGIC = 0x504a4e33,
  PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE = 32,
  PLAN7_CONTINUATION_JOURNAL_V3_BACKGROUND_ALIGNMENT = 8
};

#define PLAN7_CONTINUATION_JOURNAL_V3_CAPSULE_NAME \
  "plan7_gpu._pipeline._continuation_journal_v3"
#define PLAN7_CONTINUATION_JOURNAL_V3_CONSUMED_NAME \
  "plan7_gpu._pipeline._consumed_journal_v3"
#define PLAN7_CONTINUATION_JOURNAL_V3_NO_SOURCE_INDEX UINT64_MAX

enum plan7_continuation_journal_v3_source_kind {
  PLAN7_CONTINUATION_V3_SOURCE_HOST_SEAL = 1,
  PLAN7_CONTINUATION_V3_SOURCE_V2_JOURNAL = 2,
  PLAN7_CONTINUATION_V3_SOURCE_NATIVE_DIRECT = 3
};

/* These are semantic source stages, not planner policy guesses.  Terminal
 * stages are emitted only after replaying the same HMMER/Easel predicates (or
 * consuming a producer action already authenticated by v2). */
enum plan7_continuation_journal_v3_source_stage {
  PLAN7_CONTINUATION_V3_BEFORE_F1 = 0,
  PLAN7_CONTINUATION_V3_RAW_F1_REJECT = 1,
  PLAN7_CONTINUATION_V3_BIAS_REJECT = 2,
  PLAN7_CONTINUATION_V3_F2_REJECT = 3,
  PLAN7_CONTINUATION_V3_F3_REJECT = 4,
  PLAN7_CONTINUATION_V3_DOMAIN_NO_REGIONS = 5,
  PLAN7_CONTINUATION_V3_CPU_REQUIRED = 6,
  PLAN7_CONTINUATION_V3_F2_SURVIVOR = 7,
  PLAN7_CONTINUATION_V3_F3_SURVIVOR = 8,
  PLAN7_CONTINUATION_V3_DOMAIN_CPU_REQUIRED = 9,
  PLAN7_CONTINUATION_V3_DOMAIN_SIMPLE = 10,
  PLAN7_CONTINUATION_V3_DOMAIN_COMPACT = 11
};

enum plan7_continuation_journal_v3_exception_route {
  PLAN7_CONTINUATION_V3_FULL_PIPELINE = 1,
  PLAN7_CONTINUATION_V3_FILTER_SCORES = 2,
  PLAN7_CONTINUATION_V3_FORWARD_SCORES = 3,
  PLAN7_CONTINUATION_V3_SIMPLE_REGIONS = 4,
  PLAN7_CONTINUATION_V3_COMPACT_DOMAINS = 5
};

enum plan7_continuation_journal_v3_payload {
  PLAN7_CONTINUATION_V3_HAS_POSTFILTER = 0x01,
  PLAN7_CONTINUATION_V3_HAS_FORWARD = 0x02,
  PLAN7_CONTINUATION_V3_HAS_DOMAIN = 0x04,
  PLAN7_CONTINUATION_V3_HAS_SPECIALS = 0x08,
  PLAN7_CONTINUATION_V3_HAS_REGIONS = 0x10,
  PLAN7_CONTINUATION_V3_HAS_COMPACT = 0x20
};

enum plan7_continuation_journal_v3_precondition {
  PLAN7_CONTINUATION_V3_PRE_F2_SURVIVOR = 0x01,
  PLAN7_CONTINUATION_V3_PRE_DIRECT_FORWARD = 0x02,
  PLAN7_CONTINUATION_V3_PRE_F3_SURVIVOR = 0x04,
  PLAN7_CONTINUATION_V3_PRE_DOMAIN_SAFE = 0x08,
  PLAN7_CONTINUATION_V3_PRE_COMPACT_DEVICE = 0x10
};

enum plan7_continuation_journal_v3_profile_flag {
  PLAN7_CONTINUATION_V3_PROFILE_HAS_V2_IDENTITY = 0x01,
  PLAN7_CONTINUATION_V3_PROFILE_HAS_FINGERPRINT = 0x02
};

/* The native-direct producer builds one final semantic decision byte while it
 * is already visiting each post-filter row for F2.  Later Forward/domain
 * results replace only their sparse source positions.  This order-independent
 * tag lets the v3 emitter authenticate that immutable plan during its single
 * copy pass, without adding a separate dense validation scan. */
static inline uint64_t plan7_continuation_journal_v3_decision_mix(
    uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

static inline uint64_t plan7_continuation_journal_v3_decision_term(
    uint64_t source_index, uint8_t decision) {
  return plan7_continuation_journal_v3_decision_mix(
      source_index ^ (UINT64_C(0xd6e8feb86659fd93) *
                      ((uint64_t)decision + UINT64_C(1))));
}

static inline uint64_t plan7_continuation_journal_v3_decision_seed(
    uint64_t postfilter_count, uint64_t profile_count, uint64_t session_id,
    uint64_t selection_id, uint64_t batch_generation, uint64_t f1_bits,
    uint64_t f2_bits, uint64_t f3_bits,
    uint64_t generation_tail_fingerprint) {
  uint64_t state = UINT64_C(0x563344454349534e);
  state ^= plan7_continuation_journal_v3_decision_mix(postfilter_count);
  state ^= plan7_continuation_journal_v3_decision_mix(profile_count);
  state ^= plan7_continuation_journal_v3_decision_mix(session_id);
  state ^= plan7_continuation_journal_v3_decision_mix(selection_id);
  state ^= plan7_continuation_journal_v3_decision_mix(batch_generation);
  state ^= plan7_continuation_journal_v3_decision_mix(f1_bits);
  state ^= plan7_continuation_journal_v3_decision_mix(f2_bits);
  state ^= plan7_continuation_journal_v3_decision_mix(f3_bits);
  state ^= plan7_continuation_journal_v3_decision_mix(
      generation_tail_fingerprint);
  return state;
}

static inline uint64_t plan7_continuation_journal_v3_decision_finish(
    uint64_t state, uint64_t exception_count, uint64_t special_count,
    uint64_t region_count, uint64_t compact_result_count,
    uint64_t compact_trace_count, uint64_t compact_null2_count) {
  state ^= plan7_continuation_journal_v3_decision_mix(exception_count);
  state ^= plan7_continuation_journal_v3_decision_mix(special_count);
  state ^= plan7_continuation_journal_v3_decision_mix(region_count);
  state ^= plan7_continuation_journal_v3_decision_mix(compact_result_count);
  state ^= plan7_continuation_journal_v3_decision_mix(compact_trace_count);
  state ^= plan7_continuation_journal_v3_decision_mix(compact_null2_count);
  state = plan7_continuation_journal_v3_decision_mix(state);
  return state != 0 ? state : UINT64_C(0x5633504c414e5447);
}

typedef struct plan7_continuation_journal_v3_options {
  uint64_t f1_bits;
  uint64_t f2_bits;
  uint64_t f3_bits;
  uint64_t E_bits;
  uint64_t T_bits;
  uint64_t domE_bits;
  uint64_t domT_bits;
  uint64_t incE_bits;
  uint64_t incT_bits;
  uint64_t incdomE_bits;
  uint64_t incdomT_bits;
  uint64_t Z_bits;
  uint64_t domZ_bits;
  uint32_t rt1_bits;
  uint32_t rt2_bits;
  uint32_t rt3_bits;
  int32_t do_biasfilter;
  int32_t do_null2;
  int32_t do_alignment_score_calc;
  int32_t by_E;
  int32_t dom_by_E;
  int32_t inc_by_E;
  int32_t incdom_by_E;
  int32_t use_bit_cutoffs;
  int32_t Z_setby;
  int32_t domZ_setby;
  int32_t mode;
  int32_t long_targets;
  uint32_t complete;
  uint32_t reserved;
} plan7_continuation_journal_v3_options;

typedef struct plan7_continuation_journal_v3_certificate {
  uint64_t target_begin;
  uint64_t target_end;
  uint64_t residue_prefix_begin;
  uint64_t residue_prefix_end;
  uint64_t target_delta;
  uint64_t residue_delta;
  uint64_t before_f1_count;
  uint64_t raw_f1_reject_count;
  uint64_t bias_reject_count;
  uint64_t f2_reject_count;
  uint64_t f3_reject_count;
  uint64_t no_region_count;
  uint64_t n_past_msv_delta;
  uint64_t n_past_bias_delta;
  uint64_t n_past_vit_delta;
  uint64_t n_past_fwd_delta;
  uint32_t profile_index;
  uint32_t segment_index;
  uint64_t segment_tag;
} plan7_continuation_journal_v3_certificate;

typedef struct plan7_continuation_journal_v3_profile {
  uint64_t certificate_begin;
  uint64_t certificate_count;
  uint64_t exception_begin;
  uint64_t exception_count;
  uint64_t target_count;
  uint64_t total_residues;
  uint64_t source_postfilter_begin;
  uint64_t source_postfilter_count;
  uint64_t source_forward_begin;
  uint64_t source_forward_count;
  uint64_t source_domain_begin;
  uint64_t source_domain_count;
  uint64_t identity_token;
  uint32_t profile_index;
  uint32_t flags;
  uint8_t profile_fingerprint[
      PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE];
  uint64_t profile_tag;
} plan7_continuation_journal_v3_profile;

/* The three source records are copied byte-for-byte.  This keeps v3 additive:
 * their original public ABIs remain owned by their stage headers, while the
 * sparse packet has no pointers into the dense v2 allocation. */
typedef struct plan7_continuation_journal_v3_exception {
  uint64_t source_postfilter_index;
  uint64_t source_forward_index;
  uint64_t source_domain_index;
  uint64_t residue_prefix_begin;
  uint64_t residue_prefix_end;
  uint64_t residue_delta;
  uint64_t special_begin;
  uint64_t special_count;
  uint64_t region_begin;
  uint64_t region_count;
  uint64_t compact_result_begin;
  uint64_t compact_result_count;
  uint64_t compact_trace_begin;
  uint64_t compact_trace_count;
  uint64_t compact_null2_begin;
  uint64_t compact_null2_count;
  uint32_t profile_index;
  uint32_t sequence_index;
  uint32_t exception_index;
  uint8_t source_stage;
  uint8_t route;
  uint8_t payload_flags;
  uint8_t preconditions;
  uint8_t postfilter_record[PLAN7_POSTFILTER_RECORD_SIZE];
  uint8_t forward_record[PLAN7_FORWARD_RECORD_SIZE];
  uint8_t domain_record[sizeof(plan7_continuation_journal_row)];
  uint32_t reserved;
  uint64_t exception_tag;
} plan7_continuation_journal_v3_exception;

typedef struct plan7_continuation_journal_v3 {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint32_t profile_size;
  uint32_t certificate_size;
  uint32_t exception_size;
  uint32_t region_size;
  uint32_t compact_result_size;
  uint32_t compact_trace_step_size;
  uint32_t compact_null2_stride;
  uint32_t source_kind;
  uint32_t reserved0;
  uint64_t total_bytes;
  uint64_t source_seal_token;
  uint64_t session_id;
  uint64_t selection_id;
  uint64_t batch_generation;
  uint64_t profile_count;
  uint64_t target_count;
  uint64_t total_residues;
  uint64_t source_postfilter_count;
  uint64_t source_forward_count;
  uint64_t source_domain_count;
  uint64_t certificate_count;
  uint64_t exception_count;
  uint64_t special_count;
  uint64_t region_count;
  uint64_t compact_result_count;
  uint64_t compact_trace_offset_count;
  uint64_t compact_trace_count;
  uint64_t compact_null2_count;
  uint64_t source_v2_total_bytes;
  uint64_t source_v2_integrity_tag;
  uint64_t generation_tail_fingerprint;
  uint64_t profiles_offset;
  uint64_t certificates_offset;
  uint64_t exceptions_offset;
  uint64_t specials_offset;
  uint64_t regions_offset;
  uint64_t compact_results_offset;
  uint64_t compact_trace_offsets_offset;
  uint64_t compact_traces_offset;
  uint64_t compact_null2_offset;
  uint64_t background_fingerprint_offset;
  uint64_t background_fingerprint_bytes;
  uint8_t sequence_content_fingerprint[
      PLAN7_CONTINUATION_JOURNAL_V3_SEQUENCE_FINGERPRINT_SIZE];
  plan7_continuation_journal_v3_options options;
  plan7_forward_provenance forward;
  plan7_backward_domain_provenance backward;
  plan7_domain_rescore_provenance rescore;
  uint64_t integrity_tag;
} plan7_continuation_journal_v3;

/* Capsule context records the true allocation boundary.  Unlike trusting a
 * mutable in-packet total_bytes field, this lets the validator reject damage
 * before hashing outside the allocation. */
typedef struct plan7_continuation_journal_v3_owner {
  uint64_t allocation_bytes;
  uint64_t source_seal_token;
} plan7_continuation_journal_v3_owner;

typedef struct plan7_continuation_journal {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint32_t row_size;
  uint32_t region_size;
  uint32_t compact_result_size;
  uint32_t compact_trace_step_size;
  uint32_t compact_null2_stride;
  uint64_t total_bytes;

  uint64_t session_id;
  uint64_t selection_id;
  uint64_t profile_count;
  uint64_t postfilter_count;
  uint64_t forward_count;
  uint64_t row_count;
  uint64_t special_count;
  uint64_t region_count;

  uint64_t compact_result_count;
  uint64_t compact_trace_offset_count;
  uint64_t compact_trace_count;
  uint64_t compact_null2_count;
  uint64_t generation_tail_fingerprint;
  uint64_t rescore_simple_row_count;
  uint64_t rescore_device_result_count;
  uint64_t rescore_cpu_required_count;
  uint64_t rescore_numeric_fallback_count;
  uint64_t rescore_cap_fallback_count;
  uint64_t rescore_global_cpu_fallback_count;
  uint64_t rescore_compact_output_byte_limit;
  uint64_t rescore_compact_output_bytes;

  uint64_t generation_f1_bits;
  uint64_t generation_f2_bits;
  uint64_t generation_f3_bits;
  uint32_t rt1_bits;
  uint32_t rt2_bits;
  uint32_t rt3_bits;
  uint32_t guard_band_bits;
  uint8_t generation_bias_filter;
  uint8_t generation_compact_domains;
  uint8_t compact_global_fallback;
  uint8_t reserved0[5];
  uint8_t sequence_content_fingerprint[32];

  uint64_t postfilter_offsets_offset;
  uint64_t postfilter_records_offset;
  uint64_t forward_offsets_offset;
  uint64_t forward_records_offset;
  uint64_t forward_special_offsets_offset;
  uint64_t profile_offsets_offset;
  uint64_t identity_tokens_offset;
  uint64_t profile_fingerprints_offset;
  uint64_t rows_offset;
  uint64_t special_offsets_offset;
  uint64_t specials_offset;
  uint64_t region_offsets_offset;
  uint64_t regions_offset;
  uint64_t compact_row_offsets_offset;
  uint64_t compact_results_offset;
  uint64_t compact_trace_offsets_offset;
  uint64_t compact_traces_offset;
  uint64_t compact_null2_offset;

  plan7_forward_provenance forward;
  plan7_backward_domain_provenance backward;
  plan7_domain_rescore_provenance rescore;
  uint64_t integrity_tag;
} plan7_continuation_journal;

#ifdef __cplusplus
static_assert(sizeof(plan7_continuation_journal_row) == 64,
              "continuation journal row ABI changed");
static_assert(offsetof(plan7_continuation_journal, integrity_tag) == 776,
              "continuation journal integrity offset changed");
static_assert(sizeof(plan7_continuation_journal) == 784,
              "continuation journal header ABI changed");
static_assert(sizeof(plan7_continuation_journal_v3_options) == 176,
              "continuation journal v3 option ABI changed");
static_assert(sizeof(plan7_continuation_journal_v3_certificate) == 144,
              "continuation journal v3 certificate ABI changed");
static_assert(sizeof(plan7_continuation_journal_v3_profile) == 152,
              "continuation journal v3 profile ABI changed");
static_assert(sizeof(plan7_continuation_journal_v3_exception) == 248,
              "continuation journal v3 exception ABI changed");
static_assert(offsetof(plan7_continuation_journal_v3, integrity_tag) == 864,
              "continuation journal v3 integrity offset changed");
static_assert(sizeof(plan7_continuation_journal_v3) == 872,
              "continuation journal v3 header ABI changed");
#else
_Static_assert(sizeof(plan7_continuation_journal_row) == 64,
               "continuation journal row ABI changed");
_Static_assert(offsetof(plan7_continuation_journal, integrity_tag) == 776,
               "continuation journal integrity offset changed");
_Static_assert(sizeof(plan7_continuation_journal) == 784,
               "continuation journal header ABI changed");
_Static_assert(sizeof(plan7_continuation_journal_v3_options) == 176,
               "continuation journal v3 option ABI changed");
_Static_assert(sizeof(plan7_continuation_journal_v3_certificate) == 144,
               "continuation journal v3 certificate ABI changed");
_Static_assert(sizeof(plan7_continuation_journal_v3_profile) == 152,
               "continuation journal v3 profile ABI changed");
_Static_assert(sizeof(plan7_continuation_journal_v3_exception) == 248,
               "continuation journal v3 exception ABI changed");
_Static_assert(offsetof(plan7_continuation_journal_v3, integrity_tag) == 864,
               "continuation journal v3 integrity offset changed");
_Static_assert(sizeof(plan7_continuation_journal_v3) == 872,
               "continuation journal v3 header ABI changed");
#endif

static inline uint64_t plan7_continuation_journal_hash_bytes(
    uint64_t hash, const void *data, size_t size) {
  const uint8_t *bytes = (const uint8_t *)data;
  for (size_t i = 0; i < size; ++i) {
    hash ^= (uint64_t)bytes[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline uint64_t plan7_continuation_journal_hash_u32(
    uint64_t hash, uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8) {
    hash ^= (uint64_t)((uint8_t)(value >> shift));
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline uint64_t plan7_continuation_journal_hash_u64(
    uint64_t hash, uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    hash ^= (uint64_t)((uint8_t)(value >> shift));
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline int plan7_continuation_journal_v3_checked_add(
    uint64_t left, uint64_t right, uint64_t *sum) {
  if (sum == NULL || right > UINT64_MAX - left) return 0;
  *sum = left + right;
  return 1;
}

static inline int plan7_continuation_journal_v3_checked_multiply(
    uint64_t left, uint64_t right, uint64_t *product) {
  if (product == NULL || (left != 0 && right > UINT64_MAX / left)) return 0;
  *product = left * right;
  return 1;
}

static inline uint64_t plan7_continuation_journal_v3_certificate_tag(
    const plan7_continuation_journal_v3_certificate *certificate) {
  if (certificate == NULL) return 0;
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = plan7_continuation_journal_hash_u64(
      hash, UINT64_C(0x5633434552540001));
  return plan7_continuation_journal_hash_bytes(
      hash, certificate,
      offsetof(plan7_continuation_journal_v3_certificate, segment_tag));
}

static inline uint64_t plan7_continuation_journal_v3_profile_tag(
    const plan7_continuation_journal_v3_profile *profile) {
  if (profile == NULL) return 0;
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = plan7_continuation_journal_hash_u64(
      hash, UINT64_C(0x563350524f460001));
  return plan7_continuation_journal_hash_bytes(
      hash, profile,
      offsetof(plan7_continuation_journal_v3_profile, profile_tag));
}

static inline uint64_t plan7_continuation_journal_v3_exception_tag(
    const plan7_continuation_journal_v3_exception *exception) {
  if (exception == NULL) return 0;
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = plan7_continuation_journal_hash_u64(
      hash, UINT64_C(0x5633455843500001));
  return plan7_continuation_journal_hash_bytes(
      hash, exception,
      offsetof(plan7_continuation_journal_v3_exception, exception_tag));
}

/* Recompute the three canonical hashes produced by domain_rescore_cuda.cu.
 * Keeping this verifier in the transport header lets the CPU consumer
 * authenticate a copied journal without linking the PyHMMER companion
 * extension against CUDA. */
static inline int plan7_continuation_journal_rescore_hashes(
    const plan7_domain_rescore_result *results, uint64_t result_count,
    const uint64_t *trace_offsets,
    const plan7_domain_rescore_trace_step *traces, uint64_t trace_count,
    const float *null2, uint64_t null2_count,
    uint64_t *result_hash_out, uint64_t *trace_hash_out,
    uint64_t *null2_hash_out) {
  union { float value; uint32_t bits; } encoded;
  uint64_t result_hash = UINT64_C(1469598103934665603);
  uint64_t trace_hash = UINT64_C(1469598103934665603);
  uint64_t null2_hash = UINT64_C(1469598103934665603);
  uint64_t index;

  if (trace_offsets == NULL || result_hash_out == NULL ||
      trace_hash_out == NULL || null2_hash_out == NULL ||
      (result_count != 0 && results == NULL) ||
      (trace_count != 0 && traces == NULL) ||
      (null2_count != 0 && null2 == NULL) || trace_offsets[0] != 0 ||
      trace_offsets[result_count] != trace_count ||
      result_count > UINT64_MAX / PLAN7_DOMAIN_RESCORE_NULL2_COUNT ||
      null2_count != result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT)
    return 0;

  result_hash = plan7_continuation_journal_hash_u64(
      result_hash, UINT64_C(0x44524553));
  trace_hash = plan7_continuation_journal_hash_u64(
      trace_hash, UINT64_C(0x54524345));
  null2_hash = plan7_continuation_journal_hash_u64(
      null2_hash, UINT64_C(0x4e554c32));
  for (index = 0; index < result_count; ++index) {
    const plan7_domain_rescore_result *result = results + index;
    const uint64_t begin = trace_offsets[index];
    const uint64_t end = trace_offsets[index + 1];
    uint64_t step;
    uint64_t residue;
    if (begin > end || end > trace_count) return 0;
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->row_index);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->profile_index);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->sequence_index);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->envelope_begin);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->envelope_end);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->alignment_begin);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->alignment_end);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->model_begin);
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->model_end);
#define PLAN7_CONTINUATION_HASH_FLOAT(field) do {                             \
      encoded.value = result->field;                                         \
      result_hash = plan7_continuation_journal_hash_u32(                     \
          result_hash, encoded.bits);                                        \
    } while (0)
    PLAN7_CONTINUATION_HASH_FLOAT(forward_score);
    PLAN7_CONTINUATION_HASH_FLOAT(backward_score);
    PLAN7_CONTINUATION_HASH_FLOAT(oa_score);
    PLAN7_CONTINUATION_HASH_FLOAT(domain_correction);
    PLAN7_CONTINUATION_HASH_FLOAT(score_consistency);
#undef PLAN7_CONTINUATION_HASH_FLOAT
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, (uint32_t)result->status |
                         ((uint32_t)result->action << 8) |
                         ((uint32_t)result->has_own_scales << 16) |
                         ((uint32_t)result->reserved << 24));
    result_hash = plan7_continuation_journal_hash_u32(
        result_hash, result->reserved2);

    trace_hash = plan7_continuation_journal_hash_u64(
        trace_hash, end - begin);
    for (step = begin; step < end; ++step) {
      encoded.value = traces[step].posterior;
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, traces[step].sequence_position);
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, traces[step].model_position);
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, encoded.bits);
      trace_hash = plan7_continuation_journal_hash_u32(
          trace_hash, traces[step].state);
    }
    for (residue = 0; residue < PLAN7_DOMAIN_RESCORE_NULL2_COUNT; ++residue) {
      encoded.value = null2[
          index * PLAN7_DOMAIN_RESCORE_NULL2_COUNT + residue];
      null2_hash = plan7_continuation_journal_hash_u32(
          null2_hash, encoded.bits);
    }
  }
  *result_hash_out = plan7_continuation_journal_hash_u64(
      result_hash, PLAN7_DOMAIN_RESCORE_RECORD_VERSION);
  *trace_hash_out = plan7_continuation_journal_hash_u64(
      trace_hash, trace_count);
  *null2_hash_out = plan7_continuation_journal_hash_u64(
      null2_hash, null2_count);
  return 1;
}

static inline uint64_t plan7_continuation_journal_integrity(
    const plan7_continuation_journal *journal) {
  if (journal == NULL || journal->total_bytes < sizeof(*journal)) return 0;
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = plan7_continuation_journal_hash_bytes(
      hash, journal, offsetof(plan7_continuation_journal, integrity_tag));
  return plan7_continuation_journal_hash_bytes(
      hash, (const uint8_t *)journal + sizeof(*journal),
      (size_t)(journal->total_bytes - sizeof(*journal)));
}

static inline uint64_t plan7_continuation_journal_v3_integrity(
    const plan7_continuation_journal_v3 *journal) {
  if (journal == NULL || journal->total_bytes < sizeof(*journal)) return 0;
  uint64_t hash = UINT64_C(1469598103934665603);
  hash = plan7_continuation_journal_hash_u64(
      hash, UINT64_C(0x56334a4e4c000001));
  hash = plan7_continuation_journal_hash_bytes(
      hash, journal, offsetof(plan7_continuation_journal_v3, integrity_tag));
  return plan7_continuation_journal_hash_bytes(
      hash, (const uint8_t *)journal + sizeof(*journal),
      (size_t)(journal->total_bytes - sizeof(*journal)));
}

#ifdef __cplusplus
}
#endif

#endif
