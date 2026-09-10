#ifndef PLAN7_GPU_SSV_CUDA_H
#define PLAN7_GPU_SSV_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "bias_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_ssv_status {
  PLAN7_SSV_OK = 0,
  PLAN7_SSV_ERANGE = 16,
  PLAN7_SSV_ENORESULT = 19,
  PLAN7_SSV_EMPTY = 255
};

enum plan7_f1_action {
  PLAN7_F1_CPU_REQUIRED = 0,
  PLAN7_F1_DEFINITE_REJECT = 1
};

enum plan7_f1_cutoff_mode {
  PLAN7_F1_CUTOFF_INVALID = 0,
  PLAN7_F1_CUTOFF_SCORE = 1,
  PLAN7_F1_CUTOFF_ALWAYS_REJECT = 2,
  PLAN7_F1_CUTOFF_ALWAYS_CPU = 3
};

typedef struct {
  uint8_t xE;
  uint8_t status;
  uint8_t tjb;
  uint8_t reserved;
  int16_t numerator;
} plan7_ssv_result;

typedef struct {
  uint64_t score_offset;
  uint64_t score_count;
  int32_t score_stride;
  int32_t model_length;
  uint8_t tbm;
  uint8_t tec;
  uint8_t base;
  uint8_t bias;
  float scale;
} plan7_ssv_profile;

typedef struct plan7_ssv_f1_profile {
  plan7_ssv_profile profile;
  int32_t cutoff_mode;
  float cutoff_bit_score;
  uint64_t tjb_offset;
} plan7_ssv_f1_profile;

struct plan7_viterbi_database;
struct plan7_postfilter_result;

typedef struct plan7_ssv_sequence_batch plan7_ssv_sequence_batch;
struct plan7_postfilter_reason_statistics;
struct plan7_postfilter_f2_resident_view;
struct plan7_forward_workspace;

/* Version-1 request-scoped GPU execution policy.  This selects only among
 * exact GPU implementations that have already passed their semantic gates;
 * it never routes work to the CPU or enables an experimental filter. */
enum plan7_gpu_execution_policy {
  PLAN7_GPU_EXECUTION_POLICY_AUTO = 0,
  PLAN7_GPU_EXECUTION_POLICY_SIMPLE = 1,
  PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT = 2
};

enum plan7_gpu_execution_policy_abi {
  PLAN7_GPU_EXECUTION_POLICY_VERSION = 1,
  PLAN7_GPU_EXECUTION_POLICY_FORWARD_CANDIDATES_PER_WARP = 1
};

typedef struct plan7_gpu_execution_policy_statistics {
  uint32_t version;
  uint32_t mode;
  uint64_t target_count;
  uint64_t length_class_count;
  uint64_t f1_run_count;
  uint64_t forward_candidates_per_warp;
} plan7_gpu_execution_policy_statistics;

typedef struct plan7_ssv_workspace_statistics {
  uint64_t f1_device_compaction_run_count;
  uint64_t f1_host_expansion_run_count;
  uint64_t f1_candidate_upload_count;
  uint64_t f1_candidate_upload_avoided_count;
  uint64_t f1_profile_packed_run_count;
  uint64_t f1_profile_packed_quartet_count;
  uint64_t f1_profile_packed_profile_count;
  uint64_t f1_profile_scalar_profile_count;
  uint64_t f1_profile_packed_score_bytes;
  uint64_t f1_identity_padding_run_count;
  uint64_t f1_identity_padding_quartet_count;
  uint64_t f1_identity_padding_profile_count;
  uint64_t f1_length_class_run_count;
  uint64_t f1_length_class_value_count;
  uint64_t f1_length_compact_h2d_bytes;
  uint64_t f1_length_dense_h2d_bytes_avoided;
  uint64_t f1_length_dense_materialized_bytes;
  uint64_t postfilter_device_bytes;
  uint64_t postfilter_dp_capacity_bytes;
  uint64_t postfilter_growth_count;
  uint64_t postfilter_run_count;
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
  uint64_t forward_device_bytes;
  uint64_t forward_dp_capacity_bytes;
  uint64_t forward_xmx_capacity_bytes;
  uint64_t forward_gather_capacity_bytes;
  uint64_t forward_growth_count;
  uint64_t forward_event_create_count;
  uint64_t forward_run_count;
  uint64_t f1_raw_xe_run_count;
  uint64_t f1_raw_xe_logical_pair_count;
  uint64_t f1_raw_xe_sidecar_bytes_written;
  uint64_t f1_raw_xe_candidate_gather_count;
  uint64_t f1_candidate_ssv_replay_count;
  uint64_t f1_candidate_ssv_replay_avoided_count;
  uint64_t f1_raw_xe_fallback_run_count;
} plan7_ssv_workspace_statistics;

/* Requested device bytes for amino-profile packers. The Viterbi exact-RBV
 * allocation depends on profile contents, so it is reported as an upper
 * increment alongside minimum/maximum totals. */
typedef struct plan7_profile_footprint {
  uint64_t profile_count;
  uint64_t ssv_device_bytes;
  uint64_t viterbi_device_bytes;
  uint64_t viterbi_exact_rbv_upper_bytes;
  uint64_t forward_device_bytes;
  uint64_t bias_device_bytes;
  uint64_t minimum_device_bytes;
  uint64_t maximum_device_bytes;
} plan7_profile_footprint;

typedef struct plan7_allocation_simulation {
  uint64_t peak_additional_bytes;
  uint64_t final_additional_bytes;
  uint64_t final_free_bytes;
  uint64_t growth_count;
  uint64_t first_unfit_index;
  int32_t fits;
  int32_t reserved;
} plan7_allocation_simulation;

enum plan7_ssv_device_capacity {
  PLAN7_SSV_CAPACITY_INPUT_RESIDUES = 0,
  PLAN7_SSV_CAPACITY_INPUT_OFFSETS = 1,
  PLAN7_SSV_CAPACITY_INPUT_NULL_SCORES = 2,
  PLAN7_SSV_CAPACITY_LENGTH_TJB = 3,
  PLAN7_SSV_CAPACITY_RESULTS = 4,
  PLAN7_SSV_CAPACITY_COMPACT_SCORES = 5,
  PLAN7_SSV_CAPACITY_PROFILES = 6,
  PLAN7_SSV_CAPACITY_F1_PROFILES = 7,
  PLAN7_SSV_CAPACITY_CANDIDATE_WORDS = 8,
  PLAN7_SSV_CAPACITY_BIAS_PROFILES = 9,
  PLAN7_SSV_CAPACITY_BIAS_CANDIDATES = 10,
  PLAN7_SSV_CAPACITY_BIAS_SSV_INPUTS = 11,
  PLAN7_SSV_CAPACITY_BIAS_RESULTS = 12,
  PLAN7_SSV_CAPACITY_BIAS_LOGP = 13,
  PLAN7_SSV_CAPACITY_BIAS_LOG1MP = 14,
  PLAN7_SSV_CAPACITY_POSTFILTER_STATES = 15,
  PLAN7_SSV_CAPACITY_POSTFILTER_BIAS_INPUTS = 16,
  PLAN7_SSV_CAPACITY_POSTFILTER_BIAS_RESULTS = 17,
  PLAN7_SSV_CAPACITY_POSTFILTER_VITERBI_RESULTS = 18,
  PLAN7_SSV_CAPACITY_POSTFILTER_LENGTH_TRANSITIONS = 19,
  PLAN7_SSV_CAPACITY_POSTFILTER_MSV_OFFSETS = 20,
  PLAN7_SSV_CAPACITY_POSTFILTER_VITERBI_OFFSETS = 21,
  PLAN7_SSV_CAPACITY_POSTFILTER_DP = 22,
  PLAN7_SSV_CAPACITY_POSTFILTER_RESULTS = 23,
  PLAN7_SSV_CAPACITY_FORWARD_CANDIDATE_PROFILES = 24,
  PLAN7_SSV_CAPACITY_FORWARD_CANDIDATE_SEQUENCES = 25,
  PLAN7_SSV_CAPACITY_FORWARD_FILTER_SCORES = 26,
  PLAN7_SSV_CAPACITY_FORWARD_F3_THRESHOLDS = 27,
  PLAN7_SSV_CAPACITY_FORWARD_LENGTH_TRANSITIONS = 28,
  PLAN7_SSV_CAPACITY_FORWARD_DP_OFFSETS = 29,
  PLAN7_SSV_CAPACITY_FORWARD_X_OFFSETS = 30,
  PLAN7_SSV_CAPACITY_FORWARD_DP = 31,
  PLAN7_SSV_CAPACITY_FORWARD_XMX = 32,
  PLAN7_SSV_CAPACITY_FORWARD_RESULTS = 33,
  PLAN7_SSV_CAPACITY_FORWARD_SURVIVOR_CANDIDATES = 34,
  PLAN7_SSV_CAPACITY_FORWARD_SURVIVOR_OFFSETS = 35,
  PLAN7_SSV_CAPACITY_FORWARD_GATHERED = 36,
  PLAN7_SSV_CAPACITY_CANDIDATE_WORD_COUNTS = 37,
  PLAN7_SSV_CAPACITY_CANDIDATE_WORD_OFFSETS = 38,
  PLAN7_SSV_CAPACITY_CANDIDATE_PROFILE_OFFSETS = 39,
  PLAN7_SSV_CAPACITY_CANDIDATE_SCAN_WORKSPACE = 40,
  PLAN7_SSV_CAPACITY_F1_PROFILE_PACKED_SCORES = 41,
  PLAN7_SSV_CAPACITY_F1_PROFILE_PACKED_QUARTETS = 42,
  PLAN7_SSV_CAPACITY_F1_SCALAR_PROFILE_INDICES = 43,
  PLAN7_SSV_CAPACITY_LENGTH_CLASS_INDICES = 44,
  PLAN7_SSV_CAPACITY_F1_COMPACT_TJB = 45,
  PLAN7_SSV_CAPACITY_F1_RAW_XE = 46,
  PLAN7_SSV_DEVICE_CAPACITY_COUNT = 47
};

typedef struct plan7_ssv_memory_snapshot {
  int32_t device_ordinal;
  int32_t reserved;
  uint64_t cuda_free_bytes;
  uint64_t cuda_total_bytes;
  uint64_t persistent_device_bytes;
  uint64_t device_capacity_bytes[PLAN7_SSV_DEVICE_CAPACITY_COUNT];
} plan7_ssv_memory_snapshot;

/* Immutable device-input view for sibling CUDA stages. The owning sequence
 * batch must outlive every use of this view. */
typedef struct plan7_ssv_sequence_batch_view {
  uint64_t generation_id;
  int device_ordinal;
  int alphabet_size;
  int host_float_environment_valid;
  size_t sequence_count;
  const uint64_t *host_lengths;
  const uint8_t *device_residues;
  const uint64_t *device_offsets;
  uint64_t input_device_bytes;
} plan7_ssv_sequence_batch_view;

/* Host mirror of the stable device-compacted F1 mapping. The batch owns the
 * pointed-to storage; it remains valid only until the next operation on that
 * serialized batch. */
typedef struct plan7_ssv_f1_candidate_view {
  size_t profile_count;
  size_t candidate_count;
  const size_t *candidate_offsets;
  const plan7_bias_candidate *candidates;
} plan7_ssv_f1_candidate_view;

/* Experimental Phase-8 census only. A coarse candidate is unresolved and
 * must still execute exact SSV; a certified reject may safely skip it. */
typedef struct plan7_f0_profile_statistics {
  uint64_t logical_pair_count;
  uint64_t exact_candidate_count;
  uint64_t coarse_candidate_count;
  uint64_t certified_reject_count;
  uint64_t false_reject_count;
  uint64_t logical_cell_count;
  uint64_t survivor_exact_cell_count;
} plan7_f0_profile_statistics;

typedef struct plan7_f0_evaluation_statistics {
  uint64_t profile_count;
  uint64_t sequence_count;
  uint64_t class_count;
  uint64_t logical_pair_count;
  uint64_t exact_candidate_count;
  uint64_t coarse_candidate_count;
  uint64_t certified_reject_count;
  uint64_t false_reject_count;
  uint64_t logical_cell_count;
  uint64_t survivor_exact_cell_count;
  uint64_t coarse_table_bytes;
  uint64_t temporary_device_bytes;
  double exact_generation_milliseconds;
  double coarse_table_build_milliseconds;
  double coarse_upload_milliseconds;
  double coarse_kernel_milliseconds;
  double analysis_milliseconds;
} plan7_f0_evaluation_statistics;

/* Experimental Phase-10 census only. A seed candidate is unresolved and
 * still executes exact SSV. A certified reject contains no mandatory word
 * under the bounded-window theorem. */
typedef struct plan7_seed_profile_statistics {
  uint64_t logical_pair_count;
  uint64_t exact_candidate_count;
  uint64_t seed_candidate_count;
  uint64_t certified_reject_count;
  uint64_t false_reject_count;
  uint64_t unsupported_pair_count;
  uint64_t logical_cell_count;
  uint64_t survivor_exact_cell_count;
} plan7_seed_profile_statistics;

typedef struct plan7_seed_evaluation_statistics {
  uint64_t profile_count;
  uint64_t sequence_count;
  uint64_t maximum_word_length;
  uint64_t logical_pair_count;
  uint64_t exact_candidate_count;
  uint64_t seed_candidate_count;
  uint64_t certified_reject_count;
  uint64_t false_reject_count;
  uint64_t unsupported_pair_count;
  uint64_t logical_cell_count;
  uint64_t survivor_exact_cell_count;
  uint64_t temporary_device_bytes;
  double exact_generation_milliseconds;
  double seed_kernel_milliseconds;
  double analysis_milliseconds;
} plan7_seed_evaluation_statistics;

int plan7_cuda_device_count(char *error, size_t error_size);
int plan7_cuda_memory_info(int *device_ordinal,
                           uint64_t *free_bytes,
                           uint64_t *total_bytes,
                           char *error,
                           size_t error_size);
int plan7_validate_device_ordinal(int owner_device,
                                  int current_device,
                                  char *error,
                                  size_t error_size);
int plan7_profile_footprint_compute(const uint32_t *model_lengths,
                                    size_t profile_count,
                                    plan7_profile_footprint *footprint,
                                    char *error,
                                    size_t error_size);
int plan7_profile_slice_cell_count(uint64_t profile_count,
                                   uint64_t target_count,
                                   uint64_t cell_limit,
                                   uint64_t *cell_count,
                                   char *error,
                                   size_t error_size);
/* Simulate buffers in the caller's real allocation order. Each growth first
 * allocates the full required capacity and only then releases the old one.
 * Counts are requested bytes; allocator fragmentation is outside this model. */
int plan7_simulate_allocate_before_free(
  const uint64_t *current_capacities,
  const uint64_t *required_capacities,
  size_t capacity_count,
  uint64_t free_bytes,
  uint64_t *final_capacities,
  plan7_allocation_simulation *simulation,
  char *error,
  size_t error_size);
int plan7_tjb_for_length(float scale, uint64_t length);
int plan7_ssv_f1_decision(uint8_t status,
                          int16_t numerator,
                          uint64_t length,
                          float scale,
                          float m_mu,
                          float m_lambda,
                          double f1,
                          double *ret_p);

int plan7_ssv_f1_cutoff(float m_mu,
                        float m_lambda,
                        double f1,
                        float *ret_bit_score);

int plan7_ssv_f1_cutoff_decision(uint8_t status,
                                 int16_t numerator,
                                 uint64_t length,
                                 float scale,
                                 int cutoff_mode,
                                 float cutoff_bit_score);

int plan7_ssv_sequence_batch_create(const uint8_t *residues,
                                    size_t residue_count,
                                    const uint64_t *offsets,
                                    size_t offset_count,
                                    int alphabet_size,
                                    plan7_ssv_sequence_batch **batch,
                                    char *error,
                                    size_t error_size);

int plan7_ssv_sequence_batch_destroy(plan7_ssv_sequence_batch **batch,
                                     char *error,
                                     size_t error_size);

/* Configure the immutable policy before the first fused F1 operation.  The
 * ordinary constructor leaves AUTO selected for ABI compatibility. */
int plan7_ssv_sequence_batch_set_execution_policy(
  plan7_ssv_sequence_batch *batch,
  int policy,
  char *error,
  size_t error_size);

int plan7_ssv_sequence_batch_get_execution_policy_statistics(
  const plan7_ssv_sequence_batch *batch,
  plan7_gpu_execution_policy_statistics *statistics,
  char *error,
  size_t error_size);

int plan7_ssv_sequence_batch_get_view(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_sequence_batch_view *view,
  char *error,
  size_t error_size);

int plan7_ssv_sequence_batch_get_workspace_statistics(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_workspace_statistics *statistics,
  char *error,
  size_t error_size);

/* Requested device capacities only; host allocations are intentionally out
 * of scope. Calls must be serialized with every operation on the batch. */
int plan7_ssv_sequence_batch_get_memory_snapshot(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_memory_snapshot *snapshot,
  char *error,
  size_t error_size);

/* Internal serialized accessor used by the Forward stage. */
int plan7_ssv_sequence_batch_get_forward_workspace(
  plan7_ssv_sequence_batch *batch,
  struct plan7_forward_workspace **workspace,
  char *error,
  size_t error_size);

int plan7_ssv_sequence_batch_filter(plan7_ssv_sequence_batch *batch,
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
                                    size_t error_size);

int plan7_ssv_sequence_batch_filter_many(
  plan7_ssv_sequence_batch *batch,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const plan7_ssv_profile *profiles,
  size_t profile_count,
  plan7_ssv_result *profile_major_results,
  size_t result_count,
  char *error,
  size_t error_size);

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
  size_t error_size);

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
  size_t error_size);

/* Run the exact fused F1 kernel, then stably compact its cached device mask.
 * CUB accepts an int item count, so callers must retain the mask API as the
 * explicit fallback when profile_count * ceil(sequence_count/32) > INT_MAX. */
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
  size_t error_size);

int plan7_ssv_sequence_batch_get_f1_candidate_view(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_f1_candidate_view *view,
  char *error,
  size_t error_size);

/* Offline evaluator for a certified reduced-alphabet F0 hypothesis. It first
 * produces the ordinary exact F1 mask, then evaluates the optimistic coarse
 * recurrence. This function is diagnostic-only and never participates in
 * production dispatch. */
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
  size_t error_size);

/* Offline evaluator for the certified mandatory-word hypothesis. It compares
 * a bounded-window seed mask with the ordinary exact F1 mask. This function
 * is diagnostic-only and never participates in production dispatch. */
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
  size_t error_size);

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
  size_t error_size);

int plan7_ssv_sequence_batch_postfilter_candidates_many(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const struct plan7_viterbi_database *viterbi_database,
  struct plan7_postfilter_result *results,
  size_t result_count,
  char *error,
  size_t error_size);

/* Additive diagnostic entry point. The result ABI is unchanged; one uint16_t
 * source-fact word is returned for each ordinary result row. */
int plan7_ssv_sequence_batch_postfilter_candidates_many_reason_facts(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const struct plan7_viterbi_database *viterbi_database,
  struct plan7_postfilter_result *results,
  size_t result_count,
  uint16_t *reason_facts,
  size_t reason_count,
  struct plan7_postfilter_reason_statistics *reason_statistics,
  char *error,
  size_t error_size);

/* Private sealed fixed-options path.  A non-null reason sidecar selects the
 * same diagnostic census as the ordinary entry point.  Bias-rejected rows
 * are terminal under the authenticated request and need no reusable Viterbi
 * score. */
int plan7_ssv_sequence_batch_postfilter_candidates_many_fixed_bias(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const struct plan7_viterbi_database *viterbi_database,
  struct plan7_postfilter_result *results,
  size_t result_count,
  uint16_t *reason_facts,
  size_t reason_count,
  struct plan7_postfilter_reason_statistics *reason_statistics,
  char *error,
  size_t error_size);

/* Compile exact F2 boundaries and stably compact the just-produced resident
 * postfilter rows.  The view remains valid only until the next serialized
 * operation on batch.  Unsupported profile parameters return a valid view
 * with supported=0 so callers can use the unchanged host path. */
int plan7_ssv_sequence_batch_compact_postfilter_f2(
  plan7_ssv_sequence_batch *batch,
  const plan7_ssv_profile *profiles,
  const float *m_mu,
  const float *m_lambda,
  const float *v_mu,
  const float *v_lambda,
  size_t profile_count,
  double f2,
  struct plan7_postfilter_f2_resident_view *view,
  char *error,
  size_t error_size);

int plan7_ssv_filter_cuda(const uint8_t *striped_scores,
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
                          size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
