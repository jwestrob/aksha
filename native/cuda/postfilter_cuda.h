#ifndef PLAN7_GPU_POSTFILTER_CUDA_H
#define PLAN7_GPU_POSTFILTER_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "ssv_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_postfilter_abi {
  PLAN7_POSTFILTER_RECORD_VERSION = 1,
  PLAN7_POSTFILTER_RECORD_SIZE = 16
};

/* Opt-in, nonsemantic source-transition facts. These bits live in a separate
 * diagnostic sidecar and never enter the version-1 result record or its
 * provenance. Multiple facts may apply to one retained F1 row. */
enum plan7_postfilter_reason_fact {
  PLAN7_POSTFILTER_REASON_RAW_F1_REJECT = UINT16_C(0x0001),
  /* Final candidate state is MSV-range.  This does not imply that the full
   * MSV kernel ran: an incoming sparse-SSV ERANGE has the same final state. */
  PLAN7_POSTFILTER_REASON_MSV_RANGE_STATE = UINT16_C(0x0002),
  PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU = UINT16_C(0x0004),
  PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO = UINT16_C(0x0008),
  PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED = UINT16_C(0x0010),
  PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE = UINT16_C(0x0020),
  PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED = UINT16_C(0x0040),
  PLAN7_POSTFILTER_REASON_VITERBI_ERANGE = UINT16_C(0x0080),
  PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS = UINT16_C(0x0100),
  PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED = UINT16_C(0x0200),
  PLAN7_POSTFILTER_REASON_FINAL_REJECT = UINT16_C(0x0400),
  PLAN7_POSTFILTER_REASON_FINAL_PASS = UINT16_C(0x0800),
  PLAN7_POSTFILTER_REASON_OTHER_CPU_REQUIRED = UINT16_C(0x1000),
  PLAN7_POSTFILTER_REASON_CONTRACT_FALLBACK = UINT16_C(0x2000),
  PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED = UINT16_C(0x4000),
  PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED = UINT16_C(0x8000)
};

/* Opt-in source-execution census.  This sidecar is not part of any result,
 * workspace-statistics, provenance, or journal ABI. */
typedef struct plan7_postfilter_reason_statistics {
  uint64_t candidate_count;
  uint64_t full_msv_execution_count;
  uint64_t viterbi_execution_count;
  uint64_t full_msv_work_cells;
  uint64_t viterbi_work_cells;
  uint64_t work_cells;
} plan7_postfilter_reason_statistics;

/*
 * Version 1 deliberately preserves plan7_bias_result as its 12-byte prefix.
 * A finite vfsc or +INFINITY is an exact external Viterbi score. A NaN vfsc
 * means that the caller must run the CPU Viterbi filter. -INFINITY is never a
 * valid encoding. Bias-rejected finite MSV rows still carry an external vfsc,
 * so changing --nobias does not require recomputing their Viterbi DP.
 * MSV/SSV overflow rows use CPU_REQUIRED: a negative F2 threshold can make
 * even P=0 proceed to Viterbi, so +INFINITY is not a threshold-independent
 * substitute for the exact CPU Viterbi result.
 * A DEFINITE_REJECT row with NaN filtersc/vfsc is an exact raw-F1 reject and
 * may only be skipped when the consumer is bound to the identical F1 value
 * used to generate this CSR batch. It must never enter FilterScores directly.
 */
typedef struct plan7_postfilter_result {
  uint32_t sequence_index;
  float filtersc;
  int16_t msv_numerator;
  uint8_t msv_status;
  uint8_t action;
  float vfsc;
} plan7_postfilter_result;

enum plan7_postfilter_f2_fact {
  PLAN7_POSTFILTER_F2_NOT_PASS_OR_UNATTESTED = UINT8_C(0x01),
  PLAN7_POSTFILTER_F2_INPUT_INVALID = UINT8_C(0x02),
  PLAN7_POSTFILTER_F2_MSV_THRESHOLD_EXCEEDED = UINT8_C(0x04),
  PLAN7_POSTFILTER_F2_VITERBI_THRESHOLD_EXCEEDED = UINT8_C(0x08),
  PLAN7_POSTFILTER_F2_PASS = UINT8_C(0x10)
};

typedef struct plan7_postfilter_f2_statistics {
  uint64_t source_count;
  uint64_t selected_count;
  uint64_t compiled_profile_count;
  uint64_t unsupported_profile_count;
  uint64_t mask_word_count;
  uint64_t selected_d2h_bytes;
  uint64_t run_count;
  float compile_milliseconds;
  float upload_milliseconds;
  float kernel_milliseconds;
  float scan_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
} plan7_postfilter_f2_statistics;

/* Immutable read-only view over one exact, stable F2 compaction.  Device
 * pointers alias scratch owned by the postfilter workspace; host source
 * indexes are a byte-exact mirror used to reconstruct journal ordinals. */
typedef struct plan7_postfilter_f2_resident_view {
  uint64_t batch_generation;
  uint64_t workspace_generation;
  uint64_t selected_source_hash;
  int32_t device_ordinal;
  uint32_t supported;
  size_t profile_count;
  size_t source_count;
  size_t selected_count;
  const uint32_t *host_selected_sources;
  const plan7_bias_candidate *host_candidates;
  const plan7_postfilter_result *host_results;
  const plan7_bias_candidate *device_candidates;
  const plan7_postfilter_result *device_results;
  const uint32_t *device_selected_sources;
  const struct plan7_postfilter_workspace *owner;
  plan7_postfilter_f2_statistics statistics;
} plan7_postfilter_f2_resident_view;

typedef struct plan7_viterbi_database plan7_viterbi_database;
typedef struct plan7_postfilter_workspace plan7_postfilter_workspace;
typedef struct plan7_profile_session plan7_profile_session;
typedef struct plan7_profile_selection plan7_profile_selection;

/* Immutable, pointer-free host data for one ordered profile selection.  Every
 * pointer remains valid until the selection is destroyed.  The identity
 * tokens are opaque values: they are compared but never dereferenced. */
typedef struct plan7_profile_selection_view {
  uint64_t session_id;
  uint64_t selection_id;
  size_t profile_count;
  const uint8_t *packed_scores;
  size_t packed_score_count;
  const plan7_ssv_profile *profiles;
  const float *m_mu;
  const float *m_lambda;
  const float *v_mu;
  const float *v_lambda;
  const plan7_bias_profile *bias_templates;
  const uintptr_t *identity_tokens;
  uint64_t host_bytes;
} plan7_profile_selection_view;

typedef struct plan7_profile_session_statistics {
  uint64_t session_id;
  uint64_t profile_count;
  /* Legacy alias for selection_worker_count. */
  uint64_t worker_count;
  uint64_t build_worker_count;
  uint64_t selection_worker_count;
  uint64_t selection_count;
  uint64_t parallel_run_count;
  uint64_t build_parallel_run_count;
  uint64_t selection_parallel_run_count;
  /* Immutable payload bytes, excluding allocator slack and worker stacks. */
  uint64_t host_bytes;
  uint64_t ssv_score_bytes;
  uint64_t bias_profile_bytes;
  uint64_t viterbi_descriptor_bytes;
  uint64_t viterbi_emission_bytes;
  uint64_t viterbi_transition_bytes;
  uint64_t viterbi_exact_rbv_bytes;
  uint64_t forward_descriptor_bytes;
  uint64_t forward_emission_bytes;
  uint64_t forward_transition_bytes;
  /* A chunk-local session retains only live profile pointers, identity
   * tokens, and its copied background until a selection is requested. */
  uint64_t chunk_local_pack;
  uint64_t profile_pointer_bytes;
  uint64_t identity_token_bytes;
  uint64_t background_bytes;
} plan7_profile_session_statistics;

enum plan7_postfilter_workspace_capacity {
  PLAN7_POSTFILTER_CAPACITY_STATES = 0,
  PLAN7_POSTFILTER_CAPACITY_BIAS_INPUTS = 1,
  PLAN7_POSTFILTER_CAPACITY_BIAS_RESULTS = 2,
  PLAN7_POSTFILTER_CAPACITY_VITERBI_RESULTS = 3,
  PLAN7_POSTFILTER_CAPACITY_LENGTH_TRANSITIONS = 4,
  PLAN7_POSTFILTER_CAPACITY_MSV_OFFSETS = 5,
  PLAN7_POSTFILTER_CAPACITY_VITERBI_OFFSETS = 6,
  PLAN7_POSTFILTER_CAPACITY_DP = 7,
  PLAN7_POSTFILTER_CAPACITY_RESULTS = 8,
  PLAN7_POSTFILTER_CAPACITY_COUNT = 9
};

typedef struct plan7_postfilter_workspace_statistics {
  uint64_t device_bytes;
  uint64_t dp_capacity_bytes;
  uint64_t growth_count;
  uint64_t run_count;
  uint64_t full_msv_compaction_run_count;
  uint64_t full_msv_compaction_chunk_count;
  uint64_t full_msv_compaction_source_count;
  uint64_t full_msv_compaction_selected_count;
  uint64_t full_msv_legacy_run_count;
  uint64_t full_msv_launch_candidate_count;
  uint64_t full_msv_launch_candidate_avoided_count;
  uint64_t full_msv_index_d2h_bytes;
  uint64_t full_msv_packed_run_count;
  uint64_t full_msv_packed_group_count;
  uint64_t full_msv_packed_candidate_count;
  uint64_t full_msv_scalar_candidate_count;
  uint64_t vit_length_cache_run_count;
  uint64_t vit_length_cache_entry_count;
  uint64_t vit_length_cache_candidate_count;
  uint64_t vit_length_direct_candidate_count;
  uint64_t vit_length_cache_build_ns;
  uint64_t vit_length_candidate_plan_ns;
  uint64_t capacity_bytes[PLAN7_POSTFILTER_CAPACITY_COUNT];
} plan7_postfilter_workspace_statistics;

/* Profiles are retained by the Python owner for this object's lifetime. */
int plan7_viterbi_database_create(const uintptr_t *profile_pointers,
                                  size_t profile_count,
                                  plan7_viterbi_database **database,
                                  char *error,
                                  size_t error_size);

int plan7_viterbi_database_destroy(plan7_viterbi_database **database,
                                   char *error,
                                   size_t error_size);

size_t plan7_viterbi_database_profile_count(
  const plan7_viterbi_database *database);

/* Snapshot private optimized-profile arrays into immutable host storage.
 * With chunk_local_pack=1, retain only the supplied pointers and copy each
 * ordered selection directly into its own immutable host pack. The caller
 * must keep those source profiles alive and immutable until the session is
 * destroyed. No CUDA context or device allocation is used here. */
int plan7_profile_session_create(const uintptr_t *profile_pointers,
                                 size_t profile_count,
                                 const float *background,
                                 size_t background_count,
                                 size_t build_worker_count,
                                 size_t selection_worker_count,
                                 int chunk_local_pack,
                                 plan7_profile_session **session,
                                 char *error,
                                 size_t error_size);

int plan7_profile_session_destroy(plan7_profile_session **session,
                                  char *error,
                                  size_t error_size);

int plan7_profile_session_get_statistics(
  const plan7_profile_session *session,
  plan7_profile_session_statistics *statistics,
  char *error,
  size_t error_size);

/* Selection order is semantically significant.  Indexes must be unique and
 * in range.  Calls on a session, including destroy, must be serialized. */
int plan7_profile_session_select(plan7_profile_session *session,
                                 const size_t *profile_indices,
                                 size_t profile_count,
                                 plan7_profile_selection **selection,
                                 char *error,
                                 size_t error_size);

int plan7_profile_selection_destroy(plan7_profile_selection **selection,
                                    char *error,
                                    size_t error_size);

/* Destroy must be serialized against view and staging calls on the same
 * selection.  A selection remains valid after its source session closes. */
int plan7_profile_selection_get_view(
  const plan7_profile_selection *selection,
  plan7_profile_selection_view *view,
  char *error,
  size_t error_size);

/* Test-only exact comparison of immutable snapshot payloads. Identity tokens
 * are deliberately excluded because they identify a session, not profile
 * arithmetic. Returns 1 for equal, 0 for different, and -1 on invalid input. */
int plan7_profile_selection_snapshot_equal_for_test(
  const plan7_profile_selection *left,
  const plan7_profile_selection *right,
  char *error,
  size_t error_size);

/* Upload only this selection to the current device.  The resulting database
 * contains sealed host provenance and never reads a live P7_OPROFILE. */
int plan7_profile_selection_stage_viterbi(
  const plan7_profile_selection *selection,
  plan7_viterbi_database **database,
  char *error,
  size_t error_size);

struct plan7_forward_database;

/* Upload the immutable Forward pack for this selection to the current
 * device. The returned database owns everything needed after this call. */
int plan7_profile_selection_stage_forward(
  const plan7_profile_selection *selection,
  struct plan7_forward_database **database,
  char *error,
  size_t error_size);

/* Check exact source identity, packed SSV row affinity, and (for legacy
 * databases) the live optimized-profile snapshot. */
int plan7_viterbi_database_matches_ssv(
  const plan7_viterbi_database *database,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const uintptr_t *source_profile_pointers,
  const plan7_ssv_f1_profile *profiles,
  size_t profile_count,
  char *error,
  size_t error_size);

int plan7_postfilter_workspace_create(plan7_postfilter_workspace **workspace,
                                      char *error,
                                      size_t error_size);

int plan7_postfilter_workspace_destroy(plan7_postfilter_workspace **workspace,
                                       char *error,
                                       size_t error_size);

int plan7_postfilter_workspace_get_statistics(
  const plan7_postfilter_workspace *workspace,
  plan7_postfilter_workspace_statistics *statistics,
  char *error,
  size_t error_size);

int plan7_postfilter_workspace_compact_f2(
  plan7_postfilter_workspace *workspace,
  uint64_t batch_generation,
  const plan7_ssv_profile *profiles,
  const float *m_mu,
  const float *m_lambda,
  const float *v_mu,
  const float *v_lambda,
  size_t profile_count,
  double f2,
  int host_environment_attested,
  plan7_postfilter_f2_resident_view *view,
  char *error,
  size_t error_size);

int plan7_postfilter_f2_resident_view_validate(
  const plan7_postfilter_f2_resident_view *view,
  char *error,
  size_t error_size);

/* Internal resident-buffer entry point used by plan7_ssv_sequence_batch. */
int plan7_postfilter_candidates_device(
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  char *error,
  size_t error_size);

/* Serialized resident-workspace variant used by plan7_ssv_sequence_batch.
 * The workspace and database must belong to the current CUDA device. */
int plan7_postfilter_candidates_device_with_workspace(
  plan7_postfilter_workspace *workspace,
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  char *error,
  size_t error_size);

/* Opt-in diagnostic twin of the resident-workspace entry point. The ordinary
 * function above has no reason-sidecar allocation, kernel writes, or copies.
 * reason_facts has exactly candidate_count uint16_t entries. */
int plan7_postfilter_candidates_device_with_workspace_reason_facts(
  plan7_postfilter_workspace *workspace,
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  uint16_t *reason_facts,
  plan7_postfilter_reason_statistics *reason_statistics,
  char *error,
  size_t error_size);

/* Request-scoped policy twins used by SequenceBatch.  The legacy entry
 * points above remain AUTO for ABI compatibility. */
int plan7_postfilter_candidates_device_with_workspace_policy(
  plan7_postfilter_workspace *workspace,
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  int execution_policy,
  char *error,
  size_t error_size);

int plan7_postfilter_candidates_device_with_workspace_reason_facts_policy(
  plan7_postfilter_workspace *workspace,
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  uint16_t *reason_facts,
  plan7_postfilter_reason_statistics *reason_statistics,
  int execution_policy,
  char *error,
  size_t error_size);

/* Fixed-options sealed twins.  These are used only while constructing a
 * direct sparse packet whose authenticated generation contract requires the
 * bias filter.  General/reusable candidate batches retain Viterbi scores for
 * bias rejects through the ordinary functions above. */
int plan7_postfilter_candidates_device_with_workspace_fixed_bias_policy(
  plan7_postfilter_workspace *workspace,
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  int execution_policy,
  char *error,
  size_t error_size);

int
plan7_postfilter_candidates_device_with_workspace_fixed_bias_reason_facts_policy(
  plan7_postfilter_workspace *workspace,
  const plan7_viterbi_database *database,
  const uint8_t *device_residues,
  const uint64_t *device_sequence_offsets,
  const uint64_t *host_sequence_lengths,
  size_t sequence_count,
  const float *device_null_scores,
  const uint8_t *device_compact_scores,
  const plan7_ssv_f1_profile *device_f1_profiles,
  const uint8_t *device_tjb,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_bias_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_candidate *host_candidates,
  plan7_bias_ssv_input *device_msv_inputs,
  size_t candidate_count,
  plan7_postfilter_result *host_results,
  uint16_t *reason_facts,
  plan7_postfilter_reason_statistics *reason_statistics,
  int execution_policy,
  char *error,
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
