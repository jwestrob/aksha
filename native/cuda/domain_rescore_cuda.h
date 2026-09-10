#ifndef PLAN7_GPU_DOMAIN_RESCORE_CUDA_H
#define PLAN7_GPU_DOMAIN_RESCORE_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "backward_domain_cuda.h"
#include "forward_cuda.h"
#include "ssv_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_domain_rescore_abi {
  PLAN7_DOMAIN_RESCORE_RECORD_VERSION = 1,
  PLAN7_DOMAIN_RESCORE_RECORD_SIZE = 64,
  PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE = 16,
  PLAN7_DOMAIN_RESCORE_NULL2_COUNT = 29,
  PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES = 128 * 1024 * 1024,
  PLAN7_DOMAIN_RESCORE_MAX_MATRIX_BYTES = 512 * 1024 * 1024,
  PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES = 128 * 1024 * 1024,
  PLAN7_DOMAIN_RESCORE_MAX_ROW_WORK_CELLS = 10000000,
  PLAN7_DOMAIN_RESCORE_MAX_RUN_WORK_CELLS = 64000000
};

enum plan7_domain_rescore_action {
  PLAN7_DOMAIN_RESCORE_CPU_REQUIRED = 0,
  PLAN7_DOMAIN_RESCORE_DEVICE_RESULT = 1,
  PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT = 2
};

enum plan7_domain_rescore_status {
  PLAN7_DOMAIN_RESCORE_OK = 0,
  PLAN7_DOMAIN_RESCORE_ERANGE = 16,
  PLAN7_DOMAIN_RESCORE_ENORESULT = 19,
  PLAN7_DOMAIN_RESCORE_ECAP = 75,
  PLAN7_DOMAIN_RESCORE_EMPTY = 255
};

/* Opt-in source-transition facts. They are intentionally not part of the
 * version-1 compact result or its provenance seal. */
enum plan7_domain_rescore_reason_fact {
  PLAN7_DOMAIN_RESCORE_REASON_GLOBAL_COMPACT_BUDGET = UINT32_C(0x00000001),
  PLAN7_DOMAIN_RESCORE_REASON_OWN_SCALES = UINT32_C(0x00000002),
  PLAN7_DOMAIN_RESCORE_REASON_REGION_WORK_CAP = UINT32_C(0x00000004),
  PLAN7_DOMAIN_RESCORE_REASON_ROW_WORK_CAP = UINT32_C(0x00000008),
  PLAN7_DOMAIN_RESCORE_REASON_MATRIX_CAP = UINT32_C(0x00000010),
  PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAP = UINT32_C(0x00000020),
  PLAN7_DOMAIN_RESCORE_REASON_RUN_WORK_CAP = UINT32_C(0x00000040),
  PLAN7_DOMAIN_RESCORE_REASON_FORWARD_SCORE_INVALID = UINT32_C(0x00000080),
  PLAN7_DOMAIN_RESCORE_REASON_BACKWARD_SCORE_INVALID = UINT32_C(0x00000100),
  PLAN7_DOMAIN_RESCORE_REASON_SCALEPRODUCT_INVALID = UINT32_C(0x00000200),
  PLAN7_DOMAIN_RESCORE_REASON_NULL2_OR_CORRECTION_INVALID = UINT32_C(0x00000400),
  PLAN7_DOMAIN_RESCORE_REASON_OA_SCORE_INVALID = UINT32_C(0x00000800),
  PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAPACITY_EXHAUSTED = UINT32_C(0x00001000),
  PLAN7_DOMAIN_RESCORE_REASON_TRACE_ITERATION_INVALID = UINT32_C(0x00002000),
  PLAN7_DOMAIN_RESCORE_REASON_TRACE_PREDECESSOR_INVALID = UINT32_C(0x00004000),
  PLAN7_DOMAIN_RESCORE_REASON_TRACE_COORDINATES_INVALID = UINT32_C(0x00008000),
  PLAN7_DOMAIN_RESCORE_REASON_IDENTITY_MISMATCH = UINT32_C(0x00010000),
  PLAN7_DOMAIN_RESCORE_REASON_HOST_RESULT_INVALID = UINT32_C(0x00020000),
  PLAN7_DOMAIN_RESCORE_REASON_HOST_TRACE_INVALID = UINT32_C(0x00040000),
  PLAN7_DOMAIN_RESCORE_REASON_HOST_NULL2_INVALID = UINT32_C(0x00080000),
  PLAN7_DOMAIN_RESCORE_REASON_ROW_ATOMIC_PROPAGATION = UINT32_C(0x00100000),
  PLAN7_DOMAIN_RESCORE_REASON_DEVICE_RESULT = UINT32_C(0x00200000),
  PLAN7_DOMAIN_RESCORE_REASON_OTHER_CPU_REQUIRED = UINT32_C(0x00400000),
  /* Distinct from OWN_SCALES discovered after isolated DP was admitted. */
  PLAN7_DOMAIN_RESCORE_REASON_UPSTREAM_OWN_SCALES = UINT32_C(0x00800000),
  PLAN7_DOMAIN_RESCORE_REASON_FINAL_CPU_REQUIRED = UINT32_C(0x01000000),
  PLAN7_DOMAIN_RESCORE_REASON_CERTIFIED_GA_REJECT = UINT32_C(0x02000000)
};

/* One record describes one simple region. All coordinates are one-based and
 * refer to the original target. Trace and null2 payloads live in separate
 * pointer-free arrays owned by the opaque output handle. */
typedef struct plan7_domain_rescore_result {
  uint32_t row_index;
  uint32_t profile_index;
  uint32_t sequence_index;
  uint32_t envelope_begin;
  uint32_t envelope_end;
  uint32_t alignment_begin;
  uint32_t alignment_end;
  uint32_t model_begin;
  uint32_t model_end;
  float forward_score;
  float backward_score;
  float oa_score;
  float domain_correction;
  float score_consistency;
  uint8_t status;
  uint8_t action;
  uint8_t has_own_scales;
  uint8_t reserved;
  uint32_t reserved2;
} plan7_domain_rescore_result;

typedef struct plan7_domain_rescore_trace_step {
  uint32_t sequence_position;
  uint32_t model_position;
  float posterior;
  uint8_t state;
  uint8_t reserved[3];
} plan7_domain_rescore_trace_step;

/* The seal binds the exact upstream Backward/domain generation and the
 * ordered region/result/trace/null2 payloads. */
typedef struct plan7_domain_rescore_provenance {
  plan7_backward_domain_provenance backward;
  uint64_t result_hash;
  uint64_t trace_hash;
  uint64_t null2_hash;
  uint64_t result_count;
  uint64_t trace_count;
  uint64_t null2_count;
} plan7_domain_rescore_provenance;

typedef struct plan7_domain_rescore_statistics {
  uint64_t upstream_row_count;
  uint64_t simple_row_count;
  uint64_t region_count;
  uint64_t device_result_count;
  uint64_t cpu_required_count;
  uint64_t numeric_fallback_count;
  uint64_t cap_fallback_count;
  uint64_t global_cpu_fallback_count;
  uint64_t work_cells;
  uint64_t forward_matrix_bytes;
  uint64_t posterior_matrix_bytes;
  uint64_t special_workspace_bytes;
  uint64_t trace_workspace_bytes;
  uint64_t compact_output_byte_limit;
  uint64_t compact_output_bytes;
  float kernel_milliseconds;
  float upload_milliseconds;
  float download_milliseconds;
  float total_milliseconds;
  uint64_t certified_ga_row_count;
  uint64_t certified_ga_region_count;
  uint64_t certified_ga_skipped_work_cells;
  float ga_classification_milliseconds;
} plan7_domain_rescore_statistics;

/* Additive transfer accounting for the resident Backward/domain handoff.
 * The legacy ABI and ordinary stage statistics remain unchanged. */
typedef struct plan7_domain_rescore_residency_statistics {
  uint64_t upstream_h2d_bytes;
  uint64_t eliminated_upstream_h2d_bytes;
  uint64_t resident_selection_h2d_bytes;
  uint64_t resident_input_count;
  float upstream_upload_milliseconds;
  float resident_prepare_milliseconds;
} plan7_domain_rescore_residency_statistics;

typedef struct plan7_domain_rescore_output plan7_domain_rescore_output;

/* Production entry point. The upstream output is opaque and immutable. Calls
 * must be serialized with use/destruction of database, batch, and upstream.
 * Only upstream SIMPLE rows are eligible; every other or capped row remains a
 * conservative CPU fallback. No profile or target storage is duplicated. If
 * the minimum result journal itself exceeds the hard compact-output cap, no
 * result records are returned and global_cpu_fallback_count identifies the
 * number of upstream regions retained on CPU. */
int plan7_domain_rescore_run(
  const plan7_forward_database *database,
  const plan7_ssv_sequence_batch *batch,
  const plan7_backward_domain_output *upstream,
  uint64_t compact_byte_budget,
  uint64_t matrix_byte_budget,
  uint64_t trace_byte_budget,
  plan7_domain_rescore_output **output,
  char *error,
  size_t error_size);

int plan7_domain_rescore_run_with_reason_facts(
  const plan7_forward_database *database,
  const plan7_ssv_sequence_batch *batch,
  const plan7_backward_domain_output *upstream,
  uint64_t compact_byte_budget,
  uint64_t matrix_byte_budget,
  uint64_t trace_byte_budget,
  plan7_domain_rescore_output **output,
  char *error,
  size_t error_size);

/* Exact --cut_ga specialization. The caller supplies one whole-target
 * Forward score per upstream row and one binary32 GA target cutoff per
 * selected profile. Rows certified below GA after isolated Forward retain
 * their exact Forward score but skip isolated Backward/OA/trace/null2. */
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
  size_t error_size);

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
  size_t error_size);

int plan7_domain_rescore_output_destroy(
  plan7_domain_rescore_output **output,
  char *error,
  size_t error_size);

size_t plan7_domain_rescore_output_result_count(
  const plan7_domain_rescore_output *output);
const plan7_domain_rescore_result *plan7_domain_rescore_output_results(
  const plan7_domain_rescore_output *output);
const uint64_t *plan7_domain_rescore_output_trace_offsets(
  const plan7_domain_rescore_output *output);
size_t plan7_domain_rescore_output_trace_count(
  const plan7_domain_rescore_output *output);
const plan7_domain_rescore_trace_step *plan7_domain_rescore_output_traces(
  const plan7_domain_rescore_output *output);
size_t plan7_domain_rescore_output_null2_count(
  const plan7_domain_rescore_output *output);
const float *plan7_domain_rescore_output_null2(
  const plan7_domain_rescore_output *output);
const plan7_domain_rescore_provenance *plan7_domain_rescore_output_provenance(
  const plan7_domain_rescore_output *output);
const plan7_domain_rescore_statistics *plan7_domain_rescore_output_statistics(
  const plan7_domain_rescore_output *output);
const plan7_domain_rescore_residency_statistics *
plan7_domain_rescore_output_residency_statistics(
  const plan7_domain_rescore_output *output);
size_t plan7_domain_rescore_output_reason_count(
  const plan7_domain_rescore_output *output);
const uint32_t *plan7_domain_rescore_output_reason_facts(
  const plan7_domain_rescore_output *output);

/* Host-only test boundary for the exact compact-active-region -> source-region
 * reason merge used by production. Existing facts for inactive regions remain
 * untouched. */
int plan7_domain_rescore_merge_reason_facts_for_test(
  const uint32_t *active_result_indices,
  const uint32_t *active_facts,
  size_t active_count,
  uint32_t *source_facts,
  size_t source_count);

/* Test-only boundary predicates shared with the device implementation. */
int plan7_domain_rescore_own_scale_required_for_test(float xB);
int plan7_domain_rescore_oatrace_j_predecessor_for_test(
  float jpath,
  float epath,
  int j_loop_enabled,
  int e_loop_enabled);

/* Pristine HMMER/PyHMMER oracle for focused validation only. Production code
 * never accepts source pointers or raw sequence/region arrays. */
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
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
