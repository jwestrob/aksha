#ifndef PLAN7_GPU_FORWARD_CUDA_H
#define PLAN7_GPU_FORWARD_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_forward_abi {
  PLAN7_FORWARD_RECORD_VERSION = 1,
  PLAN7_FORWARD_RECORD_SIZE = 12,
  PLAN7_FORWARD_MAX_GATHERED_BYTES = 384 * 1024 * 1024
};

enum plan7_forward_action {
  PLAN7_FORWARD_CPU_REQUIRED = 0,
  PLAN7_FORWARD_DEFINITE_REJECT = 1,
  PLAN7_FORWARD_DEFINITE_PASS = 2
};

enum plan7_forward_status {
  PLAN7_FORWARD_OK = 0,
  PLAN7_FORWARD_ERANGE = 16,
  PLAN7_FORWARD_ENORESULT = 19,
  PLAN7_FORWARD_EMPTY = 255
};

/* Version-1 diagnostic facts are derived from the exact native classification
 * predicate and are transported outside the result/provenance ABI. */
enum plan7_forward_reason_fact {
  PLAN7_FORWARD_REASON_KERNEL_STATUS_NON_OK = UINT16_C(0x0001),
  PLAN7_FORWARD_REASON_TARGET_EMPTY = UINT16_C(0x0002),
  PLAN7_FORWARD_REASON_FWDSC_NONFINITE = UINT16_C(0x0004),
  PLAN7_FORWARD_REASON_FILTERSC_NONFINITE = UINT16_C(0x0008),
  PLAN7_FORWARD_REASON_TAU_NONFINITE = UINT16_C(0x0010),
  PLAN7_FORWARD_REASON_LAMBDA_INVALID = UINT16_C(0x0020),
  PLAN7_FORWARD_REASON_F3_REJECT = UINT16_C(0x0040),
  PLAN7_FORWARD_REASON_OUTPUT_CAP = UINT16_C(0x0080),
  PLAN7_FORWARD_REASON_SURVIVOR_GATHERED = UINT16_C(0x0100),
  PLAN7_FORWARD_REASON_OTHER_CPU_REQUIRED = UINT16_C(0x0200)
};

enum plan7_forward_call_reason_fact {
  PLAN7_FORWARD_CALL_REASON_CONTRACT_FALLBACK = UINT8_C(0x01)
};

enum plan7_forward_subwarp_policy_reason {
  PLAN7_FORWARD_SUBWARP_POLICY_NO_KERNEL = 0,
  PLAN7_FORWARD_SUBWARP_POLICY_FORCED = 1,
  PLAN7_FORWARD_SUBWARP_POLICY_SPARSE_WIDTH1 = 2,
  PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH4 = 3,
  PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH2 = 4,
  PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH4 = 5,
  PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH2 = 6,
  PLAN7_FORWARD_SUBWARP_POLICY_LONG_SATURATED_WIDTH1 = 7,
  PLAN7_FORWARD_SUBWARP_POLICY_DIVERGENT_WIDTH1 = 8
};

/* One record is returned for every input F2 survivor, in input order.
 * Special-state rows are returned only for DEFINITE_PASS records. */
typedef struct plan7_forward_result {
  uint32_t sequence_index;
  float fwdsc;
  uint8_t status;
  uint8_t action;
  uint16_t reserved;
} plan7_forward_result;

typedef struct plan7_forward_statistics {
  uint64_t generation_f3_bits;
  uint64_t candidate_count;
  uint64_t survivor_count;
  uint64_t work_cells;
  uint64_t dp_workspace_bytes;
  uint64_t xmx_workspace_bytes;
  uint64_t gather_workspace_bytes;
  uint64_t gathered_xmx_bytes;
  uint64_t output_byte_limit;
  uint64_t output_cap_fallback_count;
  float kernel_milliseconds;
  float classification_milliseconds;
  float gather_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
} plan7_forward_statistics;

/* Additive audit counters for host-compiled F3 thresholds consumed by CUDA.
 * This remains outside plan7_forward_statistics so the existing timing and
 * continuation ABIs do not change. */
typedef struct plan7_forward_f3_device_statistics {
  uint64_t compiled_profile_count;
  uint64_t unsupported_profile_count;
  uint64_t host_audit_count;
  uint64_t host_decision_avoided_count;
  uint64_t device_decision_count;
  uint64_t device_reject_count;
  uint64_t device_pass_count;
  uint64_t host_fallback_count;
  uint64_t decision_mismatch_count;
  uint64_t device_compaction_run_count;
  uint64_t device_compaction_candidate_count;
  uint64_t device_compacted_survivor_count;
  uint64_t survivor_upload_avoided_bytes;
} plan7_forward_f3_device_statistics;

/* Sealed identity of the exact resident profile database, sequence batch,
 * ordered F3-pass rows, and gathered Forward special-state bits. Later
 * stages must validate all fields before doing work. */
typedef struct plan7_forward_provenance {
  uint64_t database_generation;
  uint64_t batch_generation;
  uint64_t row_hash;
  uint64_t special_hash;
  uint64_t continuation_hash;
  uint64_t pass_count;
  uint64_t special_count;
  uint64_t generation_f3_bits;
  uint64_t integrity_tag;
} plan7_forward_provenance;

typedef struct plan7_forward_database plan7_forward_database;
typedef struct plan7_ssv_sequence_batch plan7_ssv_sequence_batch;
typedef struct plan7_forward_output plan7_forward_output;
typedef struct plan7_forward_workspace plan7_forward_workspace;
struct plan7_postfilter_f2_resident_view;

/* Pointer-free immutable host representation used by ProfileSelection. */
typedef struct plan7_forward_snapshot_profile {
  uint64_t emission_offset;
  uint64_t transition_offset;
  uint32_t q;
  uint32_t model_length;
  float e_move;
  float e_loop;
  float f_tau;
  float f_lambda;
  float nj;
  int32_t mode;
} plan7_forward_snapshot_profile;

/* Read-only internal view shared with later CUDA pipeline stages. Public
 * callers should treat every pointer as opaque device storage. */
typedef struct plan7_forward_device_profile {
  uint64_t emission_offset;
  uint64_t transition_offset;
  uint32_t q;
  uint32_t model_length;
  float e_move;
  float e_loop;
} plan7_forward_device_profile;

typedef struct plan7_forward_device_view {
  uint64_t generation_id;
  int32_t device_ordinal;
  int32_t alphabet_size;
  size_t profile_count;
  const plan7_forward_device_profile *profiles;
  const float *emissions;
  const float *transitions;
} plan7_forward_device_view;

enum plan7_forward_workspace_capacity {
  PLAN7_FORWARD_CAPACITY_CANDIDATE_PROFILES = 0,
  PLAN7_FORWARD_CAPACITY_CANDIDATE_SEQUENCES = 1,
  PLAN7_FORWARD_CAPACITY_FILTER_SCORES = 2,
  PLAN7_FORWARD_CAPACITY_F3_THRESHOLDS = 3,
  PLAN7_FORWARD_CAPACITY_LENGTH_TRANSITIONS = 4,
  PLAN7_FORWARD_CAPACITY_DP_OFFSETS = 5,
  PLAN7_FORWARD_CAPACITY_X_OFFSETS = 6,
  PLAN7_FORWARD_CAPACITY_DP = 7,
  PLAN7_FORWARD_CAPACITY_XMX = 8,
  PLAN7_FORWARD_CAPACITY_RESULTS = 9,
  PLAN7_FORWARD_CAPACITY_SURVIVOR_CANDIDATES = 10,
  PLAN7_FORWARD_CAPACITY_SURVIVOR_OFFSETS = 11,
  PLAN7_FORWARD_CAPACITY_GATHERED = 12,
  PLAN7_FORWARD_CAPACITY_COUNT = 13
};

typedef struct plan7_forward_workspace_statistics {
  uint64_t device_bytes;
  uint64_t dp_capacity_bytes;
  uint64_t xmx_capacity_bytes;
  uint64_t gather_capacity_bytes;
  uint64_t growth_count;
  uint64_t event_create_count;
  uint64_t run_count;
  uint64_t capacity_bytes[PLAN7_FORWARD_CAPACITY_COUNT];
} plan7_forward_workspace_statistics;

/* Additive instrumentation for the opt-in resident continuation path.  This
 * stays outside plan7_forward_statistics so the existing result/statistics
 * ABI remains unchanged. */
typedef struct plan7_forward_residency_statistics {
  uint64_t requested_bytes;
  uint64_t allocated_bytes;
  uint64_t materialized_bytes;
  uint64_t allocation_fallback_count;
  float allocation_milliseconds;
  float materialization_milliseconds;
} plan7_forward_residency_statistics;

typedef struct plan7_forward_input_residency_statistics {
  uint64_t resident_f2_call_count;
  uint64_t resident_f2_candidate_count;
  uint64_t eliminated_candidate_h2d_bytes;
  float gather_milliseconds;
} plan7_forward_input_residency_statistics;

/* Additive instrumentation for the isolated multi-candidate subwarp
 * experiment.  It deliberately stays outside plan7_forward_statistics and
 * the sealed result/provenance ABIs. */
typedef struct plan7_forward_subwarp_statistics {
  uint32_t policy_version;
  uint32_t requested_candidates_per_warp;
  uint32_t candidates_per_warp;
  uint32_t policy_reason;
  uint32_t multiprocessor_count;
  uint32_t reserved;
  uint64_t l2_cache_bytes;
  uint64_t policy_tile_candidate_count;
  uint64_t model_length_sum;
  uint64_t target_length_sum;
  uint64_t average_model_length;
  uint64_t average_target_length;
  uint64_t maximum_model_length;
  uint64_t maximum_target_length;
  uint64_t maximum_candidate_work_cells;
  uint64_t average_work_cells;
  uint64_t short_width4_workspace_limit_bytes;
  uint64_t long_packed_workspace_limit_bytes;
  uint64_t policy_xmx_workspace_bytes;
  uint64_t minimum_cta_count;
  uint64_t width1_cta_count;
  uint64_t width2_cta_count;
  uint64_t width4_cta_count;
  uint64_t kernel_launch_count;
  uint64_t scheduled_warp_count;
  uint64_t candidate_subwarp_count;
  uint64_t active_lane_slots;
  uint64_t issued_lane_slots;
} plan7_forward_subwarp_statistics;

/* Read-only internal view consumed by the next native CUDA stage.  The
 * pointer is generation-owned by plan7_forward_output and is valid only until
 * that output is destroyed.  Public callers must treat it as opaque device
 * storage. */
typedef struct plan7_forward_resident_view {
  uint64_t database_generation;
  uint64_t batch_generation;
  uint64_t pass_count;
  uint64_t special_count;
  int32_t device_ordinal;
  uint32_t reserved;
  const float *specials;
} plan7_forward_resident_view;

/* Profiles are retained by the Python owner for this object's lifetime. */
int plan7_forward_database_create(const uintptr_t *profile_pointers,
                                  size_t profile_count,
                                  plan7_forward_database **database,
                                  char *error,
                                  size_t error_size);

/* Stage one trusted immutable ProfileSelection snapshot on the current
 * device. All host arrays are consumed synchronously and need not outlive the
 * call. identity_tokens are opaque values and are never dereferenced. */
int plan7_forward_database_create_snapshot(
  int alphabet_size,
  const plan7_forward_snapshot_profile *profiles,
  size_t profile_count,
  const float *emissions,
  size_t emission_count,
  const float *transitions,
  size_t transition_count,
  const uintptr_t *identity_tokens,
  plan7_forward_database **database,
  char *error,
  size_t error_size);

int plan7_forward_database_destroy(plan7_forward_database **database,
                                   char *error,
                                   size_t error_size);

size_t plan7_forward_database_profile_count(
  const plan7_forward_database *database);

uint64_t plan7_forward_database_device_bytes(
  const plan7_forward_database *database);

float plan7_forward_database_pack_milliseconds(
  const plan7_forward_database *database);

float plan7_forward_database_upload_milliseconds(
  const plan7_forward_database *database);

int plan7_forward_database_get_device_view(
  const plan7_forward_database *database,
  plan7_forward_device_view *view,
  char *error,
  size_t error_size);

int plan7_forward_database_get_profile_snapshot(
  const plan7_forward_database *database,
  size_t profile_index,
  plan7_forward_snapshot_profile *profile,
  char *error,
  size_t error_size);

int plan7_forward_workspace_create(plan7_forward_workspace **workspace,
                                   char *error,
                                   size_t error_size);

int plan7_forward_workspace_destroy(plan7_forward_workspace **workspace,
                                    char *error,
                                    size_t error_size);

int plan7_forward_workspace_get_statistics(
  const plan7_forward_workspace *workspace,
  plan7_forward_workspace_statistics *statistics,
  char *error,
  size_t error_size);

/* candidate_offsets is a profile-major CSR indptr with profile_count+1
 * entries. special_offsets has candidate_count+1 float offsets. Rejected and
 * CPU_REQUIRED records have empty special-state spans. */
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
  size_t error_size);

/* Calls sharing a workspace must be serialized. */
int plan7_forward_run_with_workspace(
  plan7_forward_workspace *workspace,
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
  size_t error_size);

/* Internal SequenceBatch entry point. Calls that share a batch must be
 * serialized, including calls to batch destruction or workspace statistics. */
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
  size_t error_size);

/* Diagnostic-only forced-kernel twin used by the Phase 4 exact oracle and
 * microbenchmark.  The production entry point above remains fixed at one
 * four-lane candidate per warp until a policy is validated. */
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
  size_t error_size);

/* Opt-in diagnostic twin. The ordinary result/statistics/provenance ABIs and
 * the default entry point above are unchanged. */
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
  size_t error_size);

/* Production continuation twins.  In addition to the unchanged host output,
 * these retain the gathered pass trajectories on the selected device so an
 * immediately following Backward/domain call can consume them without an
 * H2D replay.  Allocation pressure fails soft to the legacy host-only output;
 * callers must query plan7_forward_output_get_resident_view(). */
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
  size_t error_size);

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
  size_t error_size);

/* Production fused handoff: consume the exact stable F2 selection directly
 * from the postfilter workspace while retaining host arrays solely for
 * provenance/journal validation. */
int plan7_forward_run_batch_workspace_postfilter_resident(
  const plan7_forward_database *database,
  plan7_ssv_sequence_batch *batch,
  const uintptr_t *source_profile_pointers,
  size_t profile_count,
  const uint64_t *candidate_offsets,
  const uint32_t *candidate_indices,
  const float *filter_scores,
  size_t candidate_count,
  const struct plan7_postfilter_f2_resident_view *postfilter_view,
  double f3,
  uint64_t gathered_byte_budget,
  int collect_reason_facts,
  plan7_forward_output **output,
  char *error,
  size_t error_size);

/* Exact diagnostic twin of the production decision path. Supported profiles
 * still consume the authenticated device decision, but the linked host HMMER
 * predicate is also evaluated and any disagreement fails the call. Unsupported
 * profiles use the host predicate in both modes. */
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
  size_t error_size);

int plan7_forward_output_destroy(plan7_forward_output **output,
                                 char *error,
                                 size_t error_size);

size_t plan7_forward_output_result_count(const plan7_forward_output *output);
const plan7_forward_result *plan7_forward_output_results(
  const plan7_forward_output *output);
size_t plan7_forward_output_reason_count(const plan7_forward_output *output);
const uint16_t *plan7_forward_output_reason_facts(
  const plan7_forward_output *output);
const uint64_t *plan7_forward_output_special_offsets(
  const plan7_forward_output *output);
size_t plan7_forward_output_special_count(const plan7_forward_output *output);
const float *plan7_forward_output_specials(
  const plan7_forward_output *output);
const plan7_forward_statistics *plan7_forward_output_statistics(
  const plan7_forward_output *output);
const plan7_forward_residency_statistics *
plan7_forward_output_residency_statistics(
  const plan7_forward_output *output);
const plan7_forward_input_residency_statistics *
plan7_forward_output_input_residency_statistics(
  const plan7_forward_output *output);
const plan7_forward_subwarp_statistics *
plan7_forward_output_subwarp_statistics(
  const plan7_forward_output *output);

/* Returns 1 when a complete resident view is available, 0 for the ordinary
 * host-only/fail-soft path, and -1 for invalid arguments or an inconsistent
 * generation-owned view. */
int plan7_forward_output_get_resident_view(
  const plan7_forward_output *output,
  plan7_forward_resident_view *view,
  char *error,
  size_t error_size);

const plan7_forward_f3_device_statistics *
plan7_forward_output_f3_device_statistics(
  const plan7_forward_output *output);
/* Additive instrumentation accessors on the opaque output. They do not alter
 * plan7_forward_statistics, the continuation journal, or provenance hashes. */
float plan7_forward_output_upload_milliseconds(
  const plan7_forward_output *output);
float plan7_forward_output_total_milliseconds(
  const plan7_forward_output *output);

/* True only for the exact call-wide conservative fallback at the native
 * contract/environment gate. It is not inferred from row actions. */
int plan7_forward_output_contract_fallback(
  const plan7_forward_output *output);
const plan7_forward_provenance *plan7_forward_output_provenance(
  const plan7_forward_output *output);

/* Validates the complete opaque token, including continuation_hash and the
 * generation threshold. The integrity tag detects accidental mixing and
 * mutation inside a trusted process; it is not a cryptographic MAC. */
int plan7_forward_database_validate_provenance(
  const plan7_forward_database *database,
  const plan7_forward_provenance *provenance);

#ifdef __cplusplus
}
#endif

#endif
