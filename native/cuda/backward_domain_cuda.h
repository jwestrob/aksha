#ifndef PLAN7_GPU_BACKWARD_DOMAIN_CUDA_H
#define PLAN7_GPU_BACKWARD_DOMAIN_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "forward_cuda.h"
#include "ssv_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_backward_domain_abi {
  PLAN7_BACKWARD_DOMAIN_RECORD_VERSION = 2,
  PLAN7_BACKWARD_DOMAIN_RECORD_SIZE = 32,
  PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE = 12,
  PLAN7_BACKWARD_DOMAIN_REGION_SIZE = 8,
  PLAN7_BACKWARD_DOMAIN_RESIDENT_REGION_SIZE = 24,
  PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES = 384 * 1024 * 1024,
  PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS = 10000000,
  PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS = 256 * 1024 * 1024
};

enum plan7_backward_domain_route {
  PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED = 0,
  PLAN7_BACKWARD_DOMAIN_NO_REGIONS = 1,
  PLAN7_BACKWARD_DOMAIN_SIMPLE = 2
};

enum plan7_backward_domain_status {
  PLAN7_BACKWARD_DOMAIN_OK = 0,
  PLAN7_BACKWARD_DOMAIN_ERANGE = 16,
  PLAN7_BACKWARD_DOMAIN_ENORESULT = 19,
  PLAN7_BACKWARD_DOMAIN_EMPTY = 255
};

/* Opt-in source-transition facts, kept out of the version-2 row and sealed
 * provenance. Multiple facts may coexist on a conservative route. */
enum plan7_backward_domain_reason_fact {
  PLAN7_BACKWARD_DOMAIN_REASON_TARGET_EMPTY = UINT32_C(0x0001),
  PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SPECIAL_NONFINITE = UINT32_C(0x0002),
  PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SCALE_INVALID = UINT32_C(0x0004),
  PLAN7_BACKWARD_DOMAIN_REASON_HOST_FLOAT_ENV_INVALID = UINT32_C(0x0008),
  PLAN7_BACKWARD_DOMAIN_REASON_MODE_OR_NJ_UNSUPPORTED = UINT32_C(0x0010),
  PLAN7_BACKWARD_DOMAIN_REASON_WORK_CAP = UINT32_C(0x0020),
  PLAN7_BACKWARD_DOMAIN_REASON_WORKSPACE_CAP = UINT32_C(0x0040),
  PLAN7_BACKWARD_DOMAIN_REASON_TERMINAL_SCORE_INVALID = UINT32_C(0x0080),
  PLAN7_BACKWARD_DOMAIN_REASON_POSTERIOR_OR_BACKWARD_SCORE_NONFINITE =
      UINT32_C(0x0100),
  PLAN7_BACKWARD_DOMAIN_REASON_NEXPECTED_INVALID = UINT32_C(0x0200),
  PLAN7_BACKWARD_DOMAIN_REASON_HAS_OWN_SCALES = UINT32_C(0x0400),
  PLAN7_BACKWARD_DOMAIN_REASON_THRESHOLD_UNCERTAIN = UINT32_C(0x0800),
  PLAN7_BACKWARD_DOMAIN_REASON_MULTIDOMAIN = UINT32_C(0x1000),
  PLAN7_BACKWARD_DOMAIN_REASON_NO_REGIONS = UINT32_C(0x2000),
  PLAN7_BACKWARD_DOMAIN_REASON_SIMPLE = UINT32_C(0x4000),
  PLAN7_BACKWARD_DOMAIN_REASON_REGION_OUTPUT_CAP = UINT32_C(0x8000),
  PLAN7_BACKWARD_DOMAIN_REASON_OTHER_CPU_REQUIRED = UINT32_C(0x00010000),
  PLAN7_BACKWARD_DOMAIN_REASON_FINAL_CPU_REQUIRED = UINT32_C(0x00020000)
};

/* Deliberately inaccessible from production wrappers. These mutations let
 * tests prove that the next opaque stage rejects corrupt seals and retains a
 * valid, sealed own-scale row on CPU. */
enum plan7_backward_domain_test_fault {
  PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH = 1,
  PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH = 2,
  PLAN7_BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE = 3
};

/* The row record is pointer-free: profiles and sequences are identified by
 * index, and Forward parser state is an ordinary packed array. */
typedef struct plan7_backward_domain_candidate {
  uint32_t profile_index;
  uint32_t sequence_index;
} plan7_backward_domain_candidate;

typedef struct plan7_backward_domain_result {
  uint32_t profile_index;
  uint32_t sequence_index;
  float backward_score;
  float nexpected;
  uint32_t uncertain_count;
  uint32_t region_count;
  uint32_t multidomain_count;
  uint8_t status;
  uint8_t route;
  uint8_t has_own_scales;
  uint8_t reserved;
} plan7_backward_domain_result;

typedef struct plan7_domain_posterior {
  float btot;
  float etot;
  float mocc;
} plan7_domain_posterior;

typedef struct plan7_simple_region {
  uint32_t begin;
  uint32_t end;
} plan7_simple_region;

/* Device-resident form of one sealed SIMPLE interval.  It carries exactly
 * the immutable source identity needed to compile domain-rescore work without
 * replaying the host Backward journal to the device. */
typedef struct plan7_backward_domain_resident_region {
  uint32_t profile_index;
  uint32_t sequence_index;
  uint32_t envelope_begin;
  uint32_t envelope_end;
  uint32_t target_length;
  uint8_t has_own_scales;
  uint8_t reserved[3];
} plan7_backward_domain_resident_region;

/* Seals the exact Forward generation, decision thresholds, ordered result
 * records, and compact interval journal consumed by the continuation seam. */
typedef struct plan7_backward_domain_provenance {
  plan7_forward_provenance forward;
  uint64_t threshold_hash;
  uint64_t result_hash;
  uint64_t region_hash;
  uint64_t candidate_count;
  uint64_t region_count;
} plan7_backward_domain_provenance;

typedef struct plan7_backward_domain_statistics {
  uint64_t candidate_count;
  uint64_t device_result_count;
  uint64_t cpu_required_count;
  uint64_t work_cells;
  uint64_t dp_workspace_bytes;
  uint64_t backward_special_workspace_bytes;
  uint64_t forward_special_workspace_bytes;
  uint64_t posterior_bytes;
  uint64_t simple_region_bytes;
  uint64_t output_byte_limit;
  uint64_t output_cap_fallback_count;
  uint64_t work_cap_fallback_count;
  uint64_t posterior_omitted_count;
  uint64_t own_scale_count;
  uint64_t threshold_uncertain_count;
  uint64_t no_region_count;
  uint64_t simple_count;
  uint64_t multidomain_fallback_count;
  float kernel_milliseconds;
  float upload_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
} plan7_backward_domain_statistics;

/* Additive transfer accounting for the resident Forward handoff. */
typedef struct plan7_backward_domain_residency_statistics {
  uint64_t forward_special_h2d_bytes;
  uint64_t eliminated_forward_special_h2d_bytes;
  uint64_t resident_input_count;
  float forward_special_upload_milliseconds;
  uint64_t resident_region_requested_bytes;
  uint64_t resident_region_allocated_bytes;
  uint64_t resident_region_materialized_bytes;
  uint64_t resident_region_allocation_fallback_count;
  float resident_region_allocation_milliseconds;
  float resident_region_materialization_milliseconds;
} plan7_backward_domain_residency_statistics;

/* Read-only view consumed synchronously by isolated-domain rescore.  Storage
 * is owned by plan7_backward_domain_output and expires when that handle is
 * destroyed.  Hashes and generations bind the resident descriptors to the
 * unchanged host journal and its provenance seal. */
typedef struct plan7_backward_domain_resident_view {
  uint64_t database_generation;
  uint64_t batch_generation;
  uint64_t result_hash;
  uint64_t region_hash;
  uint64_t row_count;
  uint64_t region_count;
  int32_t device_ordinal;
  uint32_t reserved;
  const plan7_backward_domain_resident_region *regions;
} plan7_backward_domain_resident_view;

typedef struct plan7_backward_domain_output plan7_backward_domain_output;

/* Low-level diagnostic seam. Calls must be serialized with all other uses and
 * destruction of database and batch. The implementation snapshots all raw
 * caller arrays before validation. Forward offsets contain float offsets and
 * must span exactly 6*(L+1) values for each row. Rows must be profile-major
 * and target indexes must be strictly increasing within a profile. A future
 * production wrapper must consume sealed CandidateBatch/Forward handles and
 * enforce the calibrated minimum guard; raw guard_band=0 is oracle-only. */
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
  size_t error_size);

/* Additive diagnostic twin. The ordinary entry point does not allocate,
 * upload, write, or download a reason sidecar. */
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
  size_t error_size);

/* Opaque production seam.  Host validation and final output semantics are
 * identical to plan7_backward_domain_run(), but the recurrence consumes the
 * generation-owned resident Forward trajectories directly. */
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
  size_t error_size);

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
  size_t error_size);

/* Experimental reverse-ownership seam.  It authenticates the sealed Forward
 * generation and emits an ordered CPU_REQUIRED domain shell without scanning
 * or executing Backward trajectories.  The ordinary GPU route remains the
 * default and this seam is selected only by the private performance policy. */
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
  size_t error_size);

/* Test-only recurrence seam for synthetic Forward scale fixtures. It never
 * validates provenance or seals output and forces every returned route to
 * CPU_REQUIRED, so its journal cannot be consumed by production code. */
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
  size_t error_size);

int plan7_backward_domain_output_destroy(
  plan7_backward_domain_output **output,
  char *error,
  size_t error_size);

size_t plan7_backward_domain_output_result_count(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_result *plan7_backward_domain_output_results(
  const plan7_backward_domain_output *output);
const uint64_t *plan7_backward_domain_output_posterior_offsets(
  const plan7_backward_domain_output *output);
size_t plan7_backward_domain_output_posterior_count(
  const plan7_backward_domain_output *output);
const plan7_domain_posterior *plan7_backward_domain_output_posteriors(
  const plan7_backward_domain_output *output);
const uint64_t *plan7_backward_domain_output_region_offsets(
  const plan7_backward_domain_output *output);
size_t plan7_backward_domain_output_region_count(
  const plan7_backward_domain_output *output);
const plan7_simple_region *plan7_backward_domain_output_regions(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_provenance *
plan7_backward_domain_output_provenance(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_statistics *
plan7_backward_domain_output_statistics(
  const plan7_backward_domain_output *output);
const plan7_backward_domain_residency_statistics *
plan7_backward_domain_output_residency_statistics(
  const plan7_backward_domain_output *output);
/* Returns 1 when the sealed device generation is available, 0 when bounded
 * allocation failed soft to the legacy host replay, and -1 for an invalid or
 * incomplete output. */
int plan7_backward_domain_output_get_resident_view(
  const plan7_backward_domain_output *output,
  plan7_backward_domain_resident_view *view,
  char *error,
  size_t error_size);
size_t plan7_backward_domain_output_reason_count(
  const plan7_backward_domain_output *output);
const uint32_t *plan7_backward_domain_output_reason_facts(
  const plan7_backward_domain_output *output);

/* Host-only test boundary for the exact compact-active-row -> source-row
 * reason merge used by production. Existing destination facts are preserved
 * for inactive source rows. */
int plan7_backward_domain_merge_reason_facts_for_test(
  const uint64_t *active_sources,
  const uint32_t *active_facts,
  size_t active_count,
  uint32_t *source_facts,
  size_t source_count);

/* Returns 1 only for a sealed production output generated with canonical
 * domain thresholds and the calibrated >=2e-4 ambiguity guard. */
int plan7_backward_domain_output_is_production_calibrated(
  const plan7_backward_domain_output *output);

/* Test-only opaque-output mutation seam. FORCE_SIMPLE_OWN_SCALE reseals the
 * journal after changing the first SIMPLE row; the tamper operations do not,
 * so a production consumer must reject them. */
int plan7_backward_domain_output_apply_test_fault(
  plan7_backward_domain_output *output,
  int fault,
  char *error,
  size_t error_size);

/* Pristine HMMER 3.4 SSE oracle for focused tests. The only pointer-valued
 * argument is deliberately confined to this test seam; production work uses
 * the pointer-free indexed ABI above. Input residues are 0-based digital
 * codes, while this helper constructs HMMER's sentinel-bearing ESL_DSQ. */
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
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
