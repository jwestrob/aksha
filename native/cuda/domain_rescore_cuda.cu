#include "domain_rescore_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {
#include <easel.h>
#include <esl_alphabet.h>
#include <hmmer.h>
#include <impl_sse/impl_sse.h>
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

static_assert(sizeof(float) == 4,
              "isolated-domain CUDA requires binary32 float");
static_assert(sizeof(plan7_domain_rescore_result) ==
                  PLAN7_DOMAIN_RESCORE_RECORD_SIZE,
              "isolated-domain result ABI changed");
static_assert(sizeof(plan7_domain_rescore_trace_step) ==
                  PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE,
              "isolated-domain trace ABI changed");
static_assert(sizeof(plan7_backward_domain_provenance) == 112,
              "Backward/domain provenance ABI changed");
static_assert(sizeof(plan7_backward_domain_resident_region) ==
                  PLAN7_BACKWARD_DOMAIN_RESIDENT_REGION_SIZE,
              "Resident Backward region ABI changed");
static_assert(sizeof(plan7_forward_device_profile) == 32,
              "Forward device-profile ABI changed");
static_assert(p7X_NXCELLS == 6 && p7X_NSCELLS == 3 &&
                  p7O_NTRANS == 8,
              "HMMER optimized matrix layout changed");

namespace {

constexpr int kThreads = 256;
constexpr int kSubwarp = 4;
constexpr int kRegionsPerBlock = kThreads / 32;
constexpr uint64_t kMaximumTargetLength = 100000;
constexpr uint64_t kMaximumModelLength = 100000;
constexpr uint64_t kCompactOutputByteLimit =
    PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES;
constexpr uint64_t kMatrixByteLimit = PLAN7_DOMAIN_RESCORE_MAX_MATRIX_BYTES;
constexpr uint64_t kTraceByteLimit = PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES;
constexpr uint64_t kMaximumRowWorkCells =
    PLAN7_DOMAIN_RESCORE_MAX_ROW_WORK_CELLS;
constexpr uint64_t kMaximumRunWorkCells =
    PLAN7_DOMAIN_RESCORE_MAX_RUN_WORK_CELLS;
constexpr uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr uint64_t kHashPrime = UINT64_C(1099511628211);

union FloatBits {
  float value;
  uint32_t bits;
};

uint64_t hash_byte(uint64_t hash, uint8_t value) {
  return (hash ^ value) * kHashPrime;
}

uint64_t hash_u32(uint64_t hash, uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    hash = hash_byte(hash, static_cast<uint8_t>(value >> shift));
  return hash;
}

uint64_t hash_u64(uint64_t hash, uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8)
    hash = hash_byte(hash, static_cast<uint8_t>(value >> shift));
  return hash;
}

void set_error(char *error, size_t error_size, const char *message) {
  if (error != nullptr && error_size != 0)
    std::snprintf(error, error_size, "%s", message);
}

void set_cuda_error(char *error, size_t error_size, const char *operation,
                    cudaError_t status) {
  if (error != nullptr && error_size != 0)
    std::snprintf(error, error_size, "%s: %s", operation,
                  cudaGetErrorString(status));
}

bool checked_add(uint64_t left, uint64_t right, uint64_t *sum) {
  if (right > UINT64_MAX - left) return false;
  *sum = left + right;
  return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t *product) {
  if (left != 0 && right > UINT64_MAX / left) return false;
  *product = left * right;
  return true;
}

bool checked_bytes(size_t count, size_t item_size, size_t *bytes) {
  if (item_size != 0 && count > SIZE_MAX / item_size) return false;
  *bytes = count * item_size;
  return true;
}

__host__ __device__ __forceinline__ bool backward_requires_own_scales(
    float xB) {
  /* Stock fwdback.c uses an unsuffixed double literal. The nearest binary32
   * value to 1e16 lies just above the double threshold, so 1.0e16f would
   * move this branch by one float ULP. */
  return static_cast<double>(xB) > 1.0e16;
}

__host__ __device__ __forceinline__ bool oatrace_prefers_j(
    float jpath, float epath, bool j_loop_enabled, bool e_loop_enabled) {
  const float path0 = j_loop_enabled ? jpath : -HUGE_VALF;
  const float path1 = e_loop_enabled ? epath : -HUGE_VALF;
  /* Stock OATrace chooses path 0 only on strict greater-than; exact ties go
   * to path 1. */
  return path0 > path1;
}

bool device_allocation_on(const void *pointer, int device_ordinal) {
  if (pointer == nullptr) return false;
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

struct RegionWork {
  uint32_t result_index;
  uint32_t row_index;
  uint32_t profile_index;
  uint32_t sequence_index;
  uint32_t envelope_begin;
  uint32_t envelope_end;
  uint32_t target_length;
  uint32_t reserved;
};

static_assert(sizeof(RegionWork) == 32, "region work descriptor changed");

struct ResidentSelection {
  uint32_t result_index;
  uint32_t row_index;
};

static_assert(sizeof(ResidentSelection) == 8,
              "resident region selection changed");

__global__ void prepare_resident_rescore_inputs_kernel(
    const plan7_backward_domain_resident_region *resident_regions,
    const ResidentSelection *selections, size_t selection_count,
    RegionWork *work, plan7_domain_rescore_result *results) {
  const size_t active = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (active >= selection_count) return;
  const ResidentSelection selection = selections[active];
  const plan7_backward_domain_resident_region source =
      resident_regions[selection.result_index];
  work[active] = {
      selection.result_index,
      selection.row_index,
      source.profile_index,
      source.sequence_index,
      source.envelope_begin,
      source.envelope_end,
      source.target_length,
      0};
  plan7_domain_rescore_result result{};
  result.row_index = selection.row_index;
  result.profile_index = source.profile_index;
  result.sequence_index = source.sequence_index;
  result.envelope_begin = source.envelope_begin;
  result.envelope_end = source.envelope_end;
  const float canonical_nan = __uint_as_float(UINT32_C(0x7fc00000));
  result.forward_score = canonical_nan;
  result.backward_score = canonical_nan;
  result.oa_score = canonical_nan;
  result.domain_correction = canonical_nan;
  result.score_consistency = canonical_nan;
  result.status = PLAN7_DOMAIN_RESCORE_OK;
  result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
  result.has_own_scales = source.has_own_scales;
  results[active] = result;
}

__device__ __forceinline__ float add_rn(float left, float right) {
  return __fadd_rn(left, right);
}

__device__ __forceinline__ float mul_rn(float left, float right) {
  return __fmul_rn(left, right);
}

__device__ __forceinline__ float sub_rn(float left, float right) {
  return __fsub_rn(left, right);
}

__device__ __forceinline__ float shift_right(float value, unsigned mask,
                                              int sublane) {
  const float shifted = __shfl_up_sync(mask, value, 1, kSubwarp);
  return sublane == 0 ? 0.0f : shifted;
}

__device__ __forceinline__ float shift_right_inf(float value, unsigned mask,
                                                  int sublane) {
  const float shifted = __shfl_up_sync(mask, value, 1, kSubwarp);
  return sublane == 0 ? -CUDART_INF_F : shifted;
}

__device__ __forceinline__ float shift_left(float value, unsigned mask,
                                             int sublane) {
  const float shifted = __shfl_down_sync(mask, value, 1, kSubwarp);
  return sublane == kSubwarp - 1 ? 0.0f : shifted;
}

__device__ __forceinline__ float horizontal_sum_sse(float value,
                                                     unsigned mask,
                                                     int sublane) {
  float rotated = __shfl_sync(mask, value, (sublane + 1) & 3, kSubwarp);
  value = add_rn(value, rotated);
  rotated = __shfl_sync(mask, value, (sublane + 2) & 3, kSubwarp);
  value = add_rn(value, rotated);
  return __shfl_sync(mask, value, 0, kSubwarp);
}

__device__ __forceinline__ float horizontal_max(float value,
                                                 unsigned mask,
                                                 int sublane) {
  float rotated = __shfl_sync(mask, value, (sublane + 1) & 3, kSubwarp);
  value = fmaxf(value, rotated);
  rotated = __shfl_sync(mask, value, (sublane + 2) & 3, kSubwarp);
  value = fmaxf(value, rotated);
  return __shfl_sync(mask, value, 0, kSubwarp);
}

__device__ __forceinline__ float transition_value(
    const plan7_forward_device_profile &profile, const float *transitions,
    int q, int transition, int sublane) {
  return transitions[profile.transition_offset +
                     (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
                         kSubwarp + sublane];
}

__device__ __forceinline__ float emission_value(
    const plan7_forward_device_profile &profile, const float *emissions,
    unsigned residue, int q, int sublane) {
  return emissions[profile.emission_offset +
                   (static_cast<uint64_t>(residue) * profile.q + q) *
                       kSubwarp + sublane];
}

__device__ __forceinline__ uint64_t dp_cell(int row, int q, int state,
                                            int lane, int Q) {
  return (static_cast<uint64_t>(row) * Q * p7X_NSCELLS +
          static_cast<uint64_t>(q) * p7X_NSCELLS + state) *
             kSubwarp + lane;
}

template <bool CollectReasonFacts>
__device__ __forceinline__ void add_rescore_reason(
    uint32_t *reason_facts, size_t region, uint32_t reason) {
  if constexpr (CollectReasonFacts) reason_facts[region] |= reason;
}

template <bool CollectReasonFacts>
__global__ void isolated_forward_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const plan7_forward_device_profile *profiles, const float *emissions,
    const float *transitions, const RegionWork *work,
    const uint64_t *matrix_offsets, const uint64_t *special_offsets,
    size_t work_count, float *forward_matrix, float *forward_specials,
    plan7_domain_rescore_result *results, uint32_t *reason_facts) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  if (lane >= kSubwarp) return;
  const size_t region =
      static_cast<size_t>(blockIdx.x) * kRegionsPerBlock + warp_in_block;
  if (region >= work_count) return;
  const int sublane = lane;
  const unsigned mask = 0xFU;
  const RegionWork item = work[region];
  const plan7_forward_device_profile profile =
      profiles[item.profile_index];
  const int Q = static_cast<int>(profile.q);
  const int Ld = static_cast<int>(item.envelope_end -
                                  item.envelope_begin + 1);
  const uint64_t sequence_start =
      sequence_offsets[item.sequence_index] + item.envelope_begin - 1;
  const float move = __fdiv_rn(
      2.0f, add_rn(static_cast<float>(item.target_length), 2.0f));
  const float loop = sub_rn(1.0f, move);
  float *matrix = forward_matrix + matrix_offsets[region];
  float *special = forward_specials + special_offsets[region];

  for (int q = 0; q < Q; ++q) {
    matrix[dp_cell(0, q, p7X_M, sublane, Q)] = 0.0f;
    matrix[dp_cell(0, q, p7X_D, sublane, Q)] = 0.0f;
    matrix[dp_cell(0, q, p7X_I, sublane, Q)] = 0.0f;
  }
  float xN = 1.0f;
  float xJ = 0.0f;
  float xB = move;
  float xC = 0.0f;
  float totscale = 0.0f;
  if (sublane == 0) {
    special[p7X_E] = 0.0f;
    special[p7X_N] = 1.0f;
    special[p7X_J] = 0.0f;
    special[p7X_B] = move;
    special[p7X_C] = 0.0f;
    special[p7X_SCALE] = 1.0f;
  }

  for (int i = 1; i <= Ld; ++i) {
    const unsigned residue = residues[sequence_start + i - 1];
    float dcv = 0.0f;
    float xEv = 0.0f;
    float mpv = shift_right(
        matrix[dp_cell(i - 1, Q - 1, p7X_M, sublane, Q)], mask,
        sublane);
    float dpv = shift_right(
        matrix[dp_cell(i - 1, Q - 1, p7X_D, sublane, Q)], mask,
        sublane);
    float ipv = shift_right(
        matrix[dp_cell(i - 1, Q - 1, p7X_I, sublane, Q)], mask,
        sublane);
    for (int q = 0; q < Q; ++q) {
      float value = mul_rn(
          xB, transition_value(profile, transitions, q, p7O_BM, sublane));
      value = add_rn(value, mul_rn(
          mpv, transition_value(profile, transitions, q, p7O_MM, sublane)));
      value = add_rn(value, mul_rn(
          ipv, transition_value(profile, transitions, q, p7O_IM, sublane)));
      value = add_rn(value, mul_rn(
          dpv, transition_value(profile, transitions, q, p7O_DM, sublane)));
      value = mul_rn(
          value, emission_value(profile, emissions, residue, q, sublane));
      xEv = add_rn(xEv, value);

      const float old_m =
          matrix[dp_cell(i - 1, q, p7X_M, sublane, Q)];
      const float old_d =
          matrix[dp_cell(i - 1, q, p7X_D, sublane, Q)];
      const float old_i =
          matrix[dp_cell(i - 1, q, p7X_I, sublane, Q)];
      matrix[dp_cell(i, q, p7X_M, sublane, Q)] = value;
      matrix[dp_cell(i, q, p7X_D, sublane, Q)] = dcv;
      dcv = mul_rn(
          value, transition_value(profile, transitions, q, p7O_MD, sublane));
      float insert = mul_rn(
          old_m, transition_value(profile, transitions, q, p7O_MI, sublane));
      insert = add_rn(insert, mul_rn(
          old_i, transition_value(profile, transitions, q, p7O_II, sublane)));
      matrix[dp_cell(i, q, p7X_I, sublane, Q)] = insert;
      mpv = old_m;
      dpv = old_d;
      ipv = old_i;
    }

    dcv = shift_right(dcv, mask, sublane);
    matrix[dp_cell(i, 0, p7X_D, sublane, Q)] = 0.0f;
    for (int q = 0; q < Q; ++q) {
      const uint64_t index = dp_cell(i, q, p7X_D, sublane, Q);
      const float updated = add_rn(dcv, matrix[index]);
      matrix[index] = updated;
      dcv = mul_rn(
          updated, transition_value(profile, transitions, q, p7O_DD, sublane));
    }
    for (int pass = 1; pass < 4; ++pass) {
      dcv = shift_right(dcv, mask, sublane);
      bool any_change = false;
      for (int q = 0; q < Q; ++q) {
        const uint64_t index = dp_cell(i, q, p7X_D, sublane, Q);
        const float current = matrix[index];
        const float updated = add_rn(dcv, current);
        any_change = any_change || updated > current;
        matrix[index] = updated;
        dcv = mul_rn(
            dcv, transition_value(profile, transitions, q, p7O_DD, sublane));
      }
      if (profile.model_length >= 100 &&
          __any_sync(mask, any_change) == 0)
        break;
    }
    for (int q = 0; q < Q; ++q)
      xEv = add_rn(
          matrix[dp_cell(i, q, p7X_D, sublane, Q)], xEv);
    float xE = horizontal_sum_sse(xEv, mask, sublane);

    xN = mul_rn(xN, loop);
    xC = add_rn(mul_rn(xC, loop), xE);  // unihit E->C is exactly 1
    xJ = mul_rn(xJ, loop);              // unihit E->J is exactly 0
    xB = add_rn(mul_rn(xJ, move), mul_rn(xN, move));
    if (xE > 1.0e4f) {
      xN = __fdiv_rn(xN, xE);
      xC = __fdiv_rn(xC, xE);
      xJ = __fdiv_rn(xJ, xE);
      xB = __fdiv_rn(xB, xE);
      const float inverse = __fdiv_rn(1.0f, xE);
      for (int q = 0; q < Q; ++q) {
        matrix[dp_cell(i, q, p7X_M, sublane, Q)] = mul_rn(
            matrix[dp_cell(i, q, p7X_M, sublane, Q)], inverse);
        matrix[dp_cell(i, q, p7X_D, sublane, Q)] = mul_rn(
            matrix[dp_cell(i, q, p7X_D, sublane, Q)], inverse);
        matrix[dp_cell(i, q, p7X_I, sublane, Q)] = mul_rn(
            matrix[dp_cell(i, q, p7X_I, sublane, Q)], inverse);
      }
      if (sublane == 0) {
        totscale = static_cast<float>(
            static_cast<double>(totscale) + log(static_cast<double>(xE)));
        special[i * p7X_NXCELLS + p7X_SCALE] = xE;
      }
      xE = 1.0f;
    } else if (sublane == 0) {
      special[i * p7X_NXCELLS + p7X_SCALE] = 1.0f;
    }
    if (sublane == 0) {
      special[i * p7X_NXCELLS + p7X_E] = xE;
      special[i * p7X_NXCELLS + p7X_N] = xN;
      special[i * p7X_NXCELLS + p7X_J] = xJ;
      special[i * p7X_NXCELLS + p7X_B] = xB;
      special[i * p7X_NXCELLS + p7X_C] = xC;
    }
  }

  if (sublane == 0) {
    plan7_domain_rescore_result &result = results[region];
    result.status = PLAN7_DOMAIN_RESCORE_OK;
    if (isnan(xC) || xC == 0.0f || isinf(xC)) {
      result.forward_score = nanf("");
      result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region,
          PLAN7_DOMAIN_RESCORE_REASON_FORWARD_SCORE_INVALID);
    } else {
      result.forward_score = static_cast<float>(
          static_cast<double>(totscale) +
          log(static_cast<double>(mul_rn(xC, move))));
    }
  }
}

template <bool CollectReasonFacts>
__global__ void isolated_backward_decode_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const plan7_forward_device_profile *profiles, const float *emissions,
    const float *transitions, const RegionWork *work,
    const uint64_t *matrix_offsets, const uint64_t *special_offsets,
    size_t work_count, const float *forward_matrix,
    const float *forward_specials, float *posterior_matrix,
    float *posterior_specials, plan7_domain_rescore_result *results,
    uint32_t *reason_facts) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  if (lane >= kSubwarp) return;
  const size_t region =
      static_cast<size_t>(blockIdx.x) * kRegionsPerBlock + warp_in_block;
  if (region >= work_count) return;
  const int sublane = lane;
  const unsigned mask = 0xFU;
  const RegionWork item = work[region];
  const plan7_forward_device_profile profile =
      profiles[item.profile_index];
  const int Q = static_cast<int>(profile.q);
  const int Ld = static_cast<int>(item.envelope_end -
                                  item.envelope_begin + 1);
  const uint64_t sequence_start =
      sequence_offsets[item.sequence_index] + item.envelope_begin - 1;
  const float move = __fdiv_rn(
      2.0f, add_rn(static_cast<float>(item.target_length), 2.0f));
  const float loop = sub_rn(1.0f, move);
  const float *fwd = forward_matrix + matrix_offsets[region];
  const float *fx = forward_specials + special_offsets[region];
  float *bck = posterior_matrix + matrix_offsets[region];
  float *bx = posterior_specials + special_offsets[region];
  plan7_domain_rescore_result &result = results[region];
  if (result.action == PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT) return;
  if (result.status != PLAN7_DOMAIN_RESCORE_OK) return;

  float xJ = 0.0f;
  float xB = 0.0f;
  float xN = 0.0f;
  float xC = move;
  float xE = xC;  // unihit E->C is exactly 1
  float xEv = xE;
  float dcv = 0.0f;
  for (int q = 0; q < Q; ++q) {
    bck[dp_cell(Ld, q, p7X_M, sublane, Q)] = xEv;
    bck[dp_cell(Ld, q, p7X_D, sublane, Q)] = xEv;
    bck[dp_cell(Ld, q, p7X_I, sublane, Q)] = 0.0f;
  }

  float dpv = shift_left(
      bck[dp_cell(Ld, Q - 1, p7X_D, sublane, Q)], mask, sublane);
  for (int q = Q - 1; q >= 0; --q) {
    dcv = mul_rn(
        dpv, transition_value(profile, transitions, q, p7O_DD, sublane));
    const uint64_t index = dp_cell(Ld, q, p7X_D, sublane, Q);
    bck[index] = add_rn(bck[index], dcv);
    dpv = bck[index];
  }
  for (int pass = 1; pass < 4; ++pass) {
    dcv = shift_left(dcv, mask, sublane);
    for (int q = Q - 1; q >= 0; --q) {
      dcv = mul_rn(
          dcv, transition_value(profile, transitions, q, p7O_DD, sublane));
      const uint64_t index = dp_cell(Ld, q, p7X_D, sublane, Q);
      bck[index] = add_rn(bck[index], dcv);
    }
  }
  dcv = shift_left(
      bck[dp_cell(Ld, 0, p7X_D, sublane, Q)], mask, sublane);
  for (int q = Q - 1; q >= 0; --q) {
    const uint64_t mindex = dp_cell(Ld, q, p7X_M, sublane, Q);
    bck[mindex] = add_rn(
        bck[mindex],
        mul_rn(dcv, transition_value(
            profile, transitions, q, p7O_MD, sublane)));
    dcv = bck[dp_cell(Ld, q, p7X_D, sublane, Q)];
  }

  float scale = fx[Ld * p7X_NXCELLS + p7X_SCALE];
  float totscale = static_cast<float>(log(static_cast<double>(scale)));
  if (scale > 1.0f) {
    xE = __fdiv_rn(xE, scale);
    xN = __fdiv_rn(xN, scale);
    xC = __fdiv_rn(xC, scale);
    xJ = __fdiv_rn(xJ, scale);
    xB = __fdiv_rn(xB, scale);
    const float inverse = __fdiv_rn(1.0f, scale);
    for (int q = 0; q < Q; ++q) {
      bck[dp_cell(Ld, q, p7X_M, sublane, Q)] = mul_rn(
          bck[dp_cell(Ld, q, p7X_M, sublane, Q)], inverse);
      bck[dp_cell(Ld, q, p7X_D, sublane, Q)] = mul_rn(
          bck[dp_cell(Ld, q, p7X_D, sublane, Q)], inverse);
      bck[dp_cell(Ld, q, p7X_I, sublane, Q)] = mul_rn(
          bck[dp_cell(Ld, q, p7X_I, sublane, Q)], inverse);
    }
  }
  if (sublane == 0) {
    bx[Ld * p7X_NXCELLS + p7X_E] = xE;
    bx[Ld * p7X_NXCELLS + p7X_N] = xN;
    bx[Ld * p7X_NXCELLS + p7X_J] = xJ;
    bx[Ld * p7X_NXCELLS + p7X_B] = xB;
    bx[Ld * p7X_NXCELLS + p7X_C] = xC;
    bx[Ld * p7X_NXCELLS + p7X_SCALE] = scale;
  }

  bool own_scales = false;
  for (int i = Ld - 1; i >= 1; --i) {
    const unsigned residue = residues[sequence_start + i];
    float tmmv = shift_left(
        transition_value(profile, transitions, 0, p7O_MM, sublane),
        mask, sublane);
    float timv = shift_left(
        transition_value(profile, transitions, 0, p7O_IM, sublane),
        mask, sublane);
    float tdmv = shift_left(
        transition_value(profile, transitions, 0, p7O_DM, sublane),
        mask, sublane);
    float mpv = mul_rn(
        bck[dp_cell(i + 1, 0, p7X_M, sublane, Q)],
        emission_value(profile, emissions, residue, 0, sublane));
    mpv = shift_left(mpv, mask, sublane);
    float xBv = 0.0f;
    for (int q = Q - 1; q >= 0; --q) {
      const float ipv = bck[dp_cell(i + 1, q, p7X_I, sublane, Q)];
      const float new_i = add_rn(
          mul_rn(ipv, transition_value(
              profile, transitions, q, p7O_II, sublane)),
          mul_rn(mpv, timv));
      const float new_d = mul_rn(mpv, tdmv);
      const float new_m = add_rn(
          mul_rn(ipv, transition_value(
              profile, transitions, q, p7O_MI, sublane)),
          mul_rn(mpv, tmmv));
      mpv = mul_rn(
          bck[dp_cell(i + 1, q, p7X_M, sublane, Q)],
          emission_value(profile, emissions, residue, q, sublane));
      bck[dp_cell(i, q, p7X_I, sublane, Q)] = new_i;
      bck[dp_cell(i, q, p7X_D, sublane, Q)] = new_d;
      bck[dp_cell(i, q, p7X_M, sublane, Q)] = new_m;
      tdmv = transition_value(profile, transitions, q, p7O_DM, sublane);
      timv = transition_value(profile, transitions, q, p7O_IM, sublane);
      tmmv = transition_value(profile, transitions, q, p7O_MM, sublane);
      xBv = add_rn(xBv, mul_rn(
          mpv, transition_value(profile, transitions, q, p7O_BM, sublane)));
    }
    xB = horizontal_sum_sse(xBv, mask, sublane);
    xC = mul_rn(xC, loop);
    xJ = add_rn(mul_rn(xB, move), mul_rn(xJ, loop));
    xN = add_rn(mul_rn(xB, move), mul_rn(xN, loop));
    xE = xC;  // E->C=1 and E->J=0 in unihit mode
    xEv = xE;

    dpv = add_rn(bck[dp_cell(i, 0, p7X_D, sublane, Q)], xEv);
    dpv = shift_left(dpv, mask, sublane);
    for (int q = Q - 1; q >= 0; --q) {
      dcv = mul_rn(
          dpv, transition_value(profile, transitions, q, p7O_DD, sublane));
      const uint64_t dindex = dp_cell(i, q, p7X_D, sublane, Q);
      bck[dindex] = add_rn(bck[dindex], add_rn(dcv, xEv));
      dpv = bck[dindex];
      const uint64_t mindex = dp_cell(i, q, p7X_M, sublane, Q);
      bck[mindex] = add_rn(bck[mindex], xEv);
    }
    for (int pass = 1; pass < 4; ++pass) {
      dcv = shift_left(dcv, mask, sublane);
      for (int q = Q - 1; q >= 0; --q) {
        dcv = mul_rn(
            dcv, transition_value(profile, transitions, q, p7O_DD, sublane));
        const uint64_t index = dp_cell(i, q, p7X_D, sublane, Q);
        bck[index] = add_rn(bck[index], dcv);
      }
    }
    dcv = shift_left(
        bck[dp_cell(i, 0, p7X_D, sublane, Q)], mask, sublane);
    for (int q = Q - 1; q >= 0; --q) {
      const uint64_t mindex = dp_cell(i, q, p7X_M, sublane, Q);
      bck[mindex] = add_rn(
          bck[mindex], mul_rn(
              dcv, transition_value(
                  profile, transitions, q, p7O_MD, sublane)));
      dcv = bck[dp_cell(i, q, p7X_D, sublane, Q)];
    }

    if (backward_requires_own_scales(xB)) own_scales = true;
    scale = own_scales ? (xB > 1.0e4f ? xB : 1.0f)
                       : fx[i * p7X_NXCELLS + p7X_SCALE];
    if (scale > 1.0f) {
      xE = __fdiv_rn(xE, scale);
      xN = __fdiv_rn(xN, scale);
      xJ = __fdiv_rn(xJ, scale);
      xB = __fdiv_rn(xB, scale);
      xC = __fdiv_rn(xC, scale);
      const float inverse = __fdiv_rn(1.0f, scale);
      for (int q = 0; q < Q; ++q) {
        bck[dp_cell(i, q, p7X_M, sublane, Q)] = mul_rn(
            bck[dp_cell(i, q, p7X_M, sublane, Q)], inverse);
        bck[dp_cell(i, q, p7X_D, sublane, Q)] = mul_rn(
            bck[dp_cell(i, q, p7X_D, sublane, Q)], inverse);
        bck[dp_cell(i, q, p7X_I, sublane, Q)] = mul_rn(
            bck[dp_cell(i, q, p7X_I, sublane, Q)], inverse);
      }
      if (sublane == 0)
        totscale = static_cast<float>(
            static_cast<double>(totscale) + log(static_cast<double>(scale)));
    }
    if (sublane == 0) {
      bx[i * p7X_NXCELLS + p7X_E] = xE;
      bx[i * p7X_NXCELLS + p7X_N] = xN;
      bx[i * p7X_NXCELLS + p7X_J] = xJ;
      bx[i * p7X_NXCELLS + p7X_B] = xB;
      bx[i * p7X_NXCELLS + p7X_C] = xC;
      bx[i * p7X_NXCELLS + p7X_SCALE] = scale;
    }
  }

  const unsigned first_residue = residues[sequence_start];
  float xBv = 0.0f;
  for (int q = 0; q < Q; ++q) {
    float mpv = mul_rn(
        bck[dp_cell(1, q, p7X_M, sublane, Q)],
        emission_value(profile, emissions, first_residue, q, sublane));
    mpv = mul_rn(
        mpv, transition_value(profile, transitions, q, p7O_BM, sublane));
    xBv = add_rn(xBv, mpv);
  }
  xB = horizontal_sum_sse(xBv, mask, sublane);
  xN = add_rn(mul_rn(xB, move), mul_rn(xN, loop));
  for (int q = 0; q < Q; ++q) {
    bck[dp_cell(0, q, p7X_M, sublane, Q)] = 0.0f;
    bck[dp_cell(0, q, p7X_D, sublane, Q)] = 0.0f;
    bck[dp_cell(0, q, p7X_I, sublane, Q)] = 0.0f;
  }
  if (sublane == 0) {
    bx[p7X_B] = xB;
    bx[p7X_C] = 0.0f;
    bx[p7X_J] = 0.0f;
    bx[p7X_N] = xN;
    bx[p7X_E] = 0.0f;
    bx[p7X_SCALE] = 1.0f;
    result.has_own_scales = own_scales ? 1 : 0;
    if (own_scales)
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region, PLAN7_DOMAIN_RESCORE_REASON_OWN_SCALES);
    if (isnan(xN) || xN == 0.0f || isinf(xN)) {
      result.backward_score = nanf("");
      result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region,
          PLAN7_DOMAIN_RESCORE_REASON_BACKWARD_SCORE_INVALID);
    } else {
      result.backward_score = static_cast<float>(
          static_cast<double>(totscale) + log(static_cast<double>(xN)));
      result.score_consistency =
          fabsf(result.forward_score - result.backward_score);
    }
  }
  __syncwarp(mask);
  if (result.status != PLAN7_DOMAIN_RESCORE_OK) return;

  float scaleproduct = __fdiv_rn(1.0f, bx[p7X_N]);
  for (int q = 0; q < Q; ++q) {
    bck[dp_cell(0, q, p7X_M, sublane, Q)] = 0.0f;
    bck[dp_cell(0, q, p7X_D, sublane, Q)] = 0.0f;
    bck[dp_cell(0, q, p7X_I, sublane, Q)] = 0.0f;
  }
  if (sublane == 0) {
    bx[p7X_E] = 0.0f;
    bx[p7X_N] = 0.0f;
    bx[p7X_J] = 0.0f;
    bx[p7X_B] = 0.0f;
    bx[p7X_C] = 0.0f;
    bx[p7X_SCALE] = 0.0f;
  }
  for (int i = 1; i <= Ld; ++i) {
    const float backward_scale = bx[i * p7X_NXCELLS + p7X_SCALE];
    const float backward_n = bx[i * p7X_NXCELLS + p7X_N];
    const float backward_j = bx[i * p7X_NXCELLS + p7X_J];
    const float backward_c = bx[i * p7X_NXCELLS + p7X_C];
    const float total = mul_rn(
        scaleproduct, fx[i * p7X_NXCELLS + p7X_SCALE]);
    for (int q = 0; q < Q; ++q) {
      const uint64_t mindex = dp_cell(i, q, p7X_M, sublane, Q);
      const uint64_t dindex = dp_cell(i, q, p7X_D, sublane, Q);
      const uint64_t iindex = dp_cell(i, q, p7X_I, sublane, Q);
      bck[mindex] = mul_rn(mul_rn(fwd[mindex], bck[mindex]), total);
      bck[dindex] = 0.0f;
      bck[iindex] = mul_rn(mul_rn(fwd[iindex], bck[iindex]), total);
    }
    if (sublane == 0) {
      bx[i * p7X_NXCELLS + p7X_E] = 0.0f;
      bx[i * p7X_NXCELLS + p7X_N] = mul_rn(
          mul_rn(mul_rn(
              fx[(i - 1) * p7X_NXCELLS + p7X_N],
              backward_n), loop), scaleproduct);
      bx[i * p7X_NXCELLS + p7X_J] = mul_rn(
          mul_rn(mul_rn(
              fx[(i - 1) * p7X_NXCELLS + p7X_J],
              backward_j), loop), scaleproduct);
      bx[i * p7X_NXCELLS + p7X_C] = mul_rn(
          mul_rn(mul_rn(
              fx[(i - 1) * p7X_NXCELLS + p7X_C],
              backward_c), loop), scaleproduct);
      bx[i * p7X_NXCELLS + p7X_B] = 0.0f;
      bx[i * p7X_NXCELLS + p7X_SCALE] = 0.0f;
    }
    if (own_scales)
      scaleproduct = mul_rn(
          scaleproduct,
          __fdiv_rn(fx[i * p7X_NXCELLS + p7X_SCALE],
                    backward_scale));
  }
  if (sublane == 0 && isinf(scaleproduct)) {
    result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
    result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
    add_rescore_reason<CollectReasonFacts>(
        reason_facts, region,
        PLAN7_DOMAIN_RESCORE_REASON_SCALEPRODUCT_INVALID);
  }
}

__device__ __forceinline__ float matrix_value(
    const float *matrix, int row, int model, int state, int Q) {
  if (model <= 0) return -CUDART_INF_F;
  const int q = (model - 1) % Q;
  const int lane = (model - 1) / Q;
  if (lane < 0 || lane >= kSubwarp) return -CUDART_INF_F;
  return matrix[dp_cell(row, q, state, lane, Q)];
}

__device__ __forceinline__ float scalar_transition(
    const plan7_forward_device_profile &profile, const float *transitions,
    int model, int transition) {
  if (model <= 0) return 0.0f;
  const int Q = static_cast<int>(profile.q);
  const int q = (model - 1) % Q;
  const int lane = (model - 1) / Q;
  if (lane < 0 || lane >= kSubwarp) return 0.0f;
  return transition_value(profile, transitions, q, transition, lane);
}

__device__ bool append_trace_step(
    plan7_domain_rescore_trace_step *trace, uint32_t capacity,
    uint32_t *count, uint8_t state, int model, int sequence,
    float posterior) {
  if (*count >= capacity) return false;
  plan7_domain_rescore_trace_step step{};
  step.state = state;
  switch (state) {
    case p7T_N:
    case p7T_C:
    case p7T_J:
      if (*count != 0 && trace[*count - 1].state == state) {
        step.sequence_position = static_cast<uint32_t>(sequence);
        step.posterior = posterior;
      }
      break;
    case p7T_D:
      step.model_position = static_cast<uint32_t>(model);
      break;
    case p7T_M:
    case p7T_I:
      step.model_position = static_cast<uint32_t>(model);
      step.sequence_position = static_cast<uint32_t>(sequence);
      step.posterior = posterior;
      break;
    default:
      break;
  }
  trace[*count] = step;
  ++*count;
  return true;
}

__device__ int select_match_predecessor_with_b(
    const plan7_forward_device_profile &profile, const float *transitions,
    const float *oa, const float *ox, int i, int k, int Q) {
  float path[4];
  path[0] = scalar_transition(profile, transitions, k, p7O_MM) == 0.0f
                ? -CUDART_INF_F
                : matrix_value(oa, i - 1, k - 1, p7X_M, Q);
  path[1] = scalar_transition(profile, transitions, k, p7O_IM) == 0.0f
                ? -CUDART_INF_F
                : matrix_value(oa, i - 1, k - 1, p7X_I, Q);
  path[2] = scalar_transition(profile, transitions, k, p7O_DM) == 0.0f
                ? -CUDART_INF_F
                : matrix_value(oa, i - 1, k - 1, p7X_D, Q);
  path[3] = scalar_transition(profile, transitions, k, p7O_BM) == 0.0f
                ? -CUDART_INF_F
                : ox[(i - 1) * p7X_NXCELLS + p7X_B];
  int choice = 0;
  for (int index = 1; index < 4; ++index)
    if (path[index] > path[choice]) choice = index;
  const int states[4] = {p7T_M, p7T_I, p7T_D, p7T_B};
  return states[choice];
}

__device__ int select_delete_predecessor(
    const plan7_forward_device_profile &profile, const float *transitions,
    const float *oa, int i, int k, int Q) {
  const float m = matrix_value(oa, i, k - 1, p7X_M, Q);
  const float d = matrix_value(oa, i, k - 1, p7X_D, Q);
  const float md = scalar_transition(profile, transitions, k - 1, p7O_MD);
  const float dd = scalar_transition(profile, transitions, k - 1, p7O_DD);
  const float mpath = md == 0.0f ? -CUDART_INF_F : m;
  const float dpath = dd == 0.0f ? -CUDART_INF_F : d;
  return mpath >= dpath ? p7T_M : p7T_D;
}

__device__ int select_insert_predecessor(
    const plan7_forward_device_profile &profile, const float *transitions,
    const float *oa, int i, int k, int Q) {
  const float m = matrix_value(oa, i - 1, k, p7X_M, Q);
  const float ins = matrix_value(oa, i - 1, k, p7X_I, Q);
  const float mi = scalar_transition(profile, transitions, k, p7O_MI);
  const float ii = scalar_transition(profile, transitions, k, p7O_II);
  const float mpath = mi == 0.0f ? -CUDART_INF_F : m;
  const float ipath = ii == 0.0f ? -CUDART_INF_F : ins;
  return mpath >= ipath ? p7T_M : p7T_I;
}

__device__ int select_end_predecessor(const float *oa, int i, int Q,
                                      int *model) {
  float maximum = -CUDART_INF_F;
  int state = -1;
  int selected_model = -1;
  for (int q = 0; q < Q; ++q) {
    for (int lane = 0; lane < kSubwarp; ++lane) {
      const int k = lane * Q + q + 1;
      const float m = oa[dp_cell(i, q, p7X_M, lane, Q)];
      if (m >= maximum) {
        maximum = m;
        state = p7T_M;
        selected_model = k;
      }
    }
    for (int lane = 0; lane < kSubwarp; ++lane) {
      const int k = lane * Q + q + 1;
      const float d = oa[dp_cell(i, q, p7X_D, lane, Q)];
      if (d > maximum) {
        maximum = d;
        state = p7T_D;
        selected_model = k;
      }
    }
  }
  *model = selected_model;
  return state;
}

__device__ float trace_postprob(const float *pp, const float *px,
                                int current_state, int previous_state,
                                int model, int sequence, int Q) {
  if (current_state == p7T_M)
    return matrix_value(pp, sequence, model, p7X_M, Q);
  if (current_state == p7T_I)
    return matrix_value(pp, sequence, model, p7X_I, Q);
  if (current_state == p7T_N && previous_state == current_state)
    return px[sequence * p7X_NXCELLS + p7X_N];
  if (current_state == p7T_C && previous_state == current_state)
    return px[sequence * p7X_NXCELLS + p7X_C];
  if (current_state == p7T_J && previous_state == current_state)
    return px[sequence * p7X_NXCELLS + p7X_J];
  return 0.0f;
}

template <bool CollectReasonFacts>
__global__ void isolated_null2_oa_trace_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const plan7_forward_device_profile *profiles, const float *emissions,
    const float *transitions, const RegionWork *work,
    const uint64_t *matrix_offsets, const uint64_t *special_offsets,
    const uint64_t *trace_capacity_offsets, size_t work_count,
    float *oa_matrix, float *oa_specials, const float *posterior_matrix,
    const float *posterior_specials, float *null2,
    plan7_domain_rescore_trace_step *traces, uint32_t *trace_counts,
    plan7_domain_rescore_result *results, uint32_t *reason_facts) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  if (lane >= kSubwarp) return;
  const size_t region =
      static_cast<size_t>(blockIdx.x) * kRegionsPerBlock + warp_in_block;
  if (region >= work_count) return;
  const int sublane = lane;
  const unsigned mask = 0xFU;
  const RegionWork item = work[region];
  const plan7_forward_device_profile profile =
      profiles[item.profile_index];
  const int Q = static_cast<int>(profile.q);
  const int M = static_cast<int>(profile.model_length);
  const int Ld = static_cast<int>(item.envelope_end -
                                  item.envelope_begin + 1);
  const uint64_t sequence_start =
      sequence_offsets[item.sequence_index] + item.envelope_begin - 1;
  float *oa = oa_matrix + matrix_offsets[region];
  float *ox = oa_specials + special_offsets[region];
  const float *pp = posterior_matrix + matrix_offsets[region];
  const float *px = posterior_specials + special_offsets[region];
  float *row_null2 = null2 + region * PLAN7_DOMAIN_RESCORE_NULL2_COUNT;
  plan7_domain_rescore_result &result = results[region];
  if (result.action == PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT) return;
  if (result.status != PLAN7_DOMAIN_RESCORE_OK || result.has_own_scales) {
    if (sublane == 0) {
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      if (result.has_own_scales)
        add_rescore_reason<CollectReasonFacts>(
            reason_facts, region, PLAN7_DOMAIN_RESCORE_REASON_OWN_SCALES);
    }
    return;
  }

  /* Null2_ByExpectation, preserving SSE's per-lane accumulation order. */
  const float norm = __fdiv_rn(1.0f, static_cast<float>(Ld));
  for (int q = 0; q < Q; ++q) {
    float match = pp[dp_cell(1, q, p7X_M, sublane, Q)];
    float insert = pp[dp_cell(1, q, p7X_I, sublane, Q)];
    for (int i = 2; i <= Ld; ++i) {
      match = add_rn(pp[dp_cell(i, q, p7X_M, sublane, Q)], match);
      insert = add_rn(pp[dp_cell(i, q, p7X_I, sublane, Q)], insert);
    }
    oa[dp_cell(0, q, p7X_M, sublane, Q)] = mul_rn(match, norm);
    oa[dp_cell(0, q, p7X_I, sublane, Q)] = mul_rn(insert, norm);
  }
  float xN = px[p7X_N + p7X_NXCELLS];
  float xC = px[p7X_C + p7X_NXCELLS];
  float xJ = px[p7X_J + p7X_NXCELLS];
  for (int i = 2; i <= Ld; ++i) {
    xN = add_rn(xN, px[i * p7X_NXCELLS + p7X_N]);
    xC = add_rn(xC, px[i * p7X_NXCELLS + p7X_C]);
    xJ = add_rn(xJ, px[i * p7X_NXCELLS + p7X_J]);
  }
  xN = mul_rn(xN, norm);
  xC = mul_rn(xC, norm);
  xJ = mul_rn(xJ, norm);
  const float xfactor = add_rn(add_rn(xN, xC), xJ);
  for (int residue = 0; residue < 20; ++residue) {
    float value = 0.0f;
    for (int q = 0; q < Q; ++q) {
      value = add_rn(value, mul_rn(
          oa[dp_cell(0, q, p7X_M, sublane, Q)],
          emission_value(profile, emissions, residue, q, sublane)));
      value = add_rn(
          value, oa[dp_cell(0, q, p7X_I, sublane, Q)]);
    }
    value = horizontal_sum_sse(value, mask, sublane);
    if (sublane == 0) row_null2[residue] = add_rn(value, xfactor);
  }
  __syncwarp(mask);
  if (sublane == 0) {
    row_null2[20] = 1.0f;
    row_null2[21] = mul_rn(add_rn(row_null2[11], row_null2[2]), 0.5f);
    row_null2[22] = mul_rn(add_rn(row_null2[7], row_null2[9]), 0.5f);
    row_null2[23] = mul_rn(add_rn(row_null2[13], row_null2[3]), 0.5f);
    row_null2[24] = row_null2[8];
    row_null2[25] = row_null2[1];
    float any = 0.0f;
    for (int residue = 0; residue < 20; ++residue)
      any = add_rn(any, row_null2[residue]);
    row_null2[26] = __fdiv_rn(any, 20.0f);
    row_null2[27] = 1.0f;
    row_null2[28] = 1.0f;
    float correction = 0.0f;
    bool finite = true;
    for (int i = 0; i < Ld; ++i) {
      const float odds = row_null2[residues[sequence_start + i]];
      finite = finite && isfinite(odds) && odds > 0.0f;
      correction = add_rn(correction, logf(odds));
    }
    result.domain_correction = correction;
    if (!finite || !isfinite(correction)) {
      result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region,
          PLAN7_DOMAIN_RESCORE_REASON_NULL2_OR_CORRECTION_INVALID);
    }
  }
  __syncwarp(mask);
  if (result.status != PLAN7_DOMAIN_RESCORE_OK) return;

  /* OptimalAccuracy overwrites the no-longer-needed Forward matrix. */
  for (int q = 0; q < Q; ++q) {
    oa[dp_cell(0, q, p7X_M, sublane, Q)] = -CUDART_INF_F;
    oa[dp_cell(0, q, p7X_D, sublane, Q)] = -CUDART_INF_F;
    oa[dp_cell(0, q, p7X_I, sublane, Q)] = -CUDART_INF_F;
  }
  if (sublane == 0) {
    ox[p7X_E] = -CUDART_INF_F;
    ox[p7X_N] = 0.0f;
    ox[p7X_J] = -CUDART_INF_F;
    ox[p7X_B] = 0.0f;
    ox[p7X_C] = -CUDART_INF_F;
    ox[p7X_SCALE] = 0.0f;
  }
  __syncwarp(mask);
  for (int i = 1; i <= Ld; ++i) {
    float dcv = -CUDART_INF_F;
    float xEv = -CUDART_INF_F;
    const float xBv = ox[(i - 1) * p7X_NXCELLS + p7X_B];
    float mpv = shift_right_inf(
        oa[dp_cell(i - 1, Q - 1, p7X_M, sublane, Q)], mask, sublane);
    float dpv = shift_right_inf(
        oa[dp_cell(i - 1, Q - 1, p7X_D, sublane, Q)], mask, sublane);
    float ipv = shift_right_inf(
        oa[dp_cell(i - 1, Q - 1, p7X_I, sublane, Q)], mask, sublane);
    for (int q = 0; q < Q; ++q) {
      float value = transition_value(
                        profile, transitions, q, p7O_BM, sublane) > 0.0f
                        ? xBv : 0.0f;
      value = fmaxf(value,
          transition_value(profile, transitions, q, p7O_MM, sublane) > 0.0f
              ? mpv : 0.0f);
      value = fmaxf(value,
          transition_value(profile, transitions, q, p7O_IM, sublane) > 0.0f
              ? ipv : 0.0f);
      value = fmaxf(value,
          transition_value(profile, transitions, q, p7O_DM, sublane) > 0.0f
              ? dpv : 0.0f);
      value = add_rn(value, pp[dp_cell(i, q, p7X_M, sublane, Q)]);
      xEv = fmaxf(xEv, value);

      const float old_m = oa[dp_cell(i - 1, q, p7X_M, sublane, Q)];
      const float old_d = oa[dp_cell(i - 1, q, p7X_D, sublane, Q)];
      const float old_i = oa[dp_cell(i - 1, q, p7X_I, sublane, Q)];
      oa[dp_cell(i, q, p7X_M, sublane, Q)] = value;
      oa[dp_cell(i, q, p7X_D, sublane, Q)] = dcv;
      dcv = transition_value(profile, transitions, q, p7O_MD, sublane) > 0.0f
                ? value : 0.0f;
      float insert = transition_value(
                         profile, transitions, q, p7O_MI, sublane) > 0.0f
                         ? old_m : 0.0f;
      insert = fmaxf(insert,
          transition_value(profile, transitions, q, p7O_II, sublane) > 0.0f
              ? old_i : 0.0f);
      oa[dp_cell(i, q, p7X_I, sublane, Q)] = add_rn(
          insert, pp[dp_cell(i, q, p7X_I, sublane, Q)]);
      mpv = old_m;
      dpv = old_d;
      ipv = old_i;
    }

    dcv = shift_right_inf(dcv, mask, sublane);
    for (int q = 0; q < Q; ++q) {
      const uint64_t index = dp_cell(i, q, p7X_D, sublane, Q);
      oa[index] = fmaxf(dcv, oa[index]);
      dcv = transition_value(profile, transitions, q, p7O_DD, sublane) > 0.0f
                ? oa[index] : 0.0f;
    }
    for (int pass = 1; pass < 4; ++pass) {
      dcv = shift_right_inf(dcv, mask, sublane);
      for (int q = 0; q < Q; ++q) {
        const uint64_t index = dp_cell(i, q, p7X_D, sublane, Q);
        oa[index] = fmaxf(dcv, oa[index]);
        dcv = transition_value(
                  profile, transitions, q, p7O_DD, sublane) > 0.0f
                  ? dcv : 0.0f;
      }
    }
    for (int q = 0; q < Q; ++q)
      xEv = fmaxf(xEv, oa[dp_cell(i, q, p7X_D, sublane, Q)]);
    const float row_e = horizontal_max(xEv, mask, sublane);
    if (sublane == 0) {
      ox[i * p7X_NXCELLS + p7X_E] = row_e;
      const float j1 = add_rn(
          ox[(i - 1) * p7X_NXCELLS + p7X_J],
          px[i * p7X_NXCELLS + p7X_J]);
      ox[i * p7X_NXCELLS + p7X_J] = fmaxf(j1, 0.0f);
      const float c1 = add_rn(
          ox[(i - 1) * p7X_NXCELLS + p7X_C],
          px[i * p7X_NXCELLS + p7X_C]);
      ox[i * p7X_NXCELLS + p7X_C] = fmaxf(c1, row_e);
      ox[i * p7X_NXCELLS + p7X_N] = add_rn(
          ox[(i - 1) * p7X_NXCELLS + p7X_N],
          px[i * p7X_NXCELLS + p7X_N]);
      ox[i * p7X_NXCELLS + p7X_B] = fmaxf(
          ox[i * p7X_NXCELLS + p7X_N],
          ox[i * p7X_NXCELLS + p7X_J]);
      ox[i * p7X_NXCELLS + p7X_SCALE] = 0.0f;
    }
    __syncwarp(mask);
  }
  if (sublane != 0) return;
  result.oa_score = ox[Ld * p7X_NXCELLS + p7X_C];
  if (!isfinite(result.oa_score)) {
    result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
    result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
    add_rescore_reason<CollectReasonFacts>(
        reason_facts, region, PLAN7_DOMAIN_RESCORE_REASON_OA_SCORE_INVALID);
    return;
  }

  /* Exact stock OATrace tie rules; retain the whole compact trace so the CPU
   * seam can construct its ordinary P7_ALIDISPLAY without dense matrices. */
  const uint64_t trace_begin = trace_capacity_offsets[region];
  const uint64_t trace_end = trace_capacity_offsets[region + 1];
  if (trace_end < trace_begin || trace_end - trace_begin > UINT32_MAX) {
    result.status = PLAN7_DOMAIN_RESCORE_ECAP;
    result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
    add_rescore_reason<CollectReasonFacts>(
        reason_facts, region,
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAPACITY_EXHAUSTED);
    return;
  }
  plan7_domain_rescore_trace_step *trace = traces + trace_begin;
  const uint32_t capacity = static_cast<uint32_t>(trace_end - trace_begin);
  uint32_t count = 0;
  int i = Ld;
  int k = 0;
  if (!append_trace_step(trace, capacity, &count, p7T_T, k, i, 0.0f) ||
      !append_trace_step(trace, capacity, &count, p7T_C, k, i, 0.0f)) {
    result.status = PLAN7_DOMAIN_RESCORE_ECAP;
    result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
    add_rescore_reason<CollectReasonFacts>(
        reason_facts, region,
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAPACITY_EXHAUSTED);
    return;
  }
  int previous_state = p7T_C;
  uint64_t iterations = 0;
  while (previous_state != p7T_S) {
    if (++iterations > static_cast<uint64_t>(capacity)) {
      result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region,
          PLAN7_DOMAIN_RESCORE_REASON_TRACE_ITERATION_INVALID);
      return;
    }
    int state = -1;
    switch (previous_state) {
      case p7T_M:
        if (i <= 0 || k <= 0) break;
        state = select_match_predecessor_with_b(
            profile, transitions, oa, ox, i, k, Q);
        --k;
        --i;
        break;
      case p7T_D:
        if (k <= 0) break;
        state = select_delete_predecessor(
            profile, transitions, oa, i, k, Q);
        --k;
        break;
      case p7T_I:
        if (i <= 0 || k <= 0) break;
        state = select_insert_predecessor(
            profile, transitions, oa, i, k, Q);
        --i;
        break;
      case p7T_N:
        state = i == 0 ? p7T_S : p7T_N;
        break;
      case p7T_C: {
        const float cpath = i > 0
            ? add_rn(ox[(i - 1) * p7X_NXCELLS + p7X_C],
                     px[i * p7X_NXCELLS + p7X_C])
            : -CUDART_INF_F;
        const float epath = ox[i * p7X_NXCELLS + p7X_E];
        state = cpath > epath ? p7T_C : p7T_E;
        break;
      }
      case p7T_J: {
        const float jpath = i > 0
            ? add_rn(ox[(i - 1) * p7X_NXCELLS + p7X_J],
                     px[i * p7X_NXCELLS + p7X_J])
            : -CUDART_INF_F;
        /* ReconfigUnihit disables E->J, so stock OATrace compares the
         * enabled J-loop path against -infinity here. */
        const float epath = -CUDART_INF_F;
        state = oatrace_prefers_j(jpath, epath, true, false)
                    ? p7T_J : p7T_E;
        break;
      }
      case p7T_E:
        state = select_end_predecessor(oa, i, Q, &k);
        break;
      case p7T_B:
        state = ox[i * p7X_NXCELLS + p7X_N] >
                        ox[i * p7X_NXCELLS + p7X_J]
                    ? p7T_N : p7T_J;
        break;
      default:
        break;
    }
    if (state < 0 || k < 0 || i < 0) {
      result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region,
          PLAN7_DOMAIN_RESCORE_REASON_TRACE_PREDECESSOR_INVALID);
      return;
    }
    const float posterior = trace_postprob(
        pp, px, state, previous_state, k, i, Q);
    if (!append_trace_step(trace, capacity, &count,
                           static_cast<uint8_t>(state), k, i, posterior)) {
      result.status = PLAN7_DOMAIN_RESCORE_ECAP;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      add_rescore_reason<CollectReasonFacts>(
          reason_facts, region,
          PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAPACITY_EXHAUSTED);
      return;
    }
    if ((state == p7T_N || state == p7T_J || state == p7T_C) &&
        state == previous_state)
      --i;
    previous_state = state;
  }

  for (uint32_t index = 0; index + 1 < count; ++index) {
    const uint8_t state = trace[index].state;
    if ((state == p7T_N || state == p7T_C || state == p7T_J) &&
        trace[index + 1].state == state &&
        trace[index].sequence_position == 0 &&
        trace[index + 1].sequence_position > 0) {
      trace[index].sequence_position = trace[index + 1].sequence_position;
      trace[index].posterior = trace[index + 1].posterior;
      trace[index + 1].sequence_position = 0;
      trace[index + 1].posterior = 0.0f;
    }
  }
  for (uint32_t left = 0; left < count / 2; ++left) {
    const uint32_t right = count - left - 1;
    const plan7_domain_rescore_trace_step temporary = trace[left];
    trace[left] = trace[right];
    trace[right] = temporary;
  }
  uint32_t first_match = 0;
  uint32_t last_match = 0;
  uint32_t first_model = 0;
  uint32_t last_model = 0;
  for (uint32_t index = 0; index < count; ++index) {
    if (trace[index].sequence_position > 0)
      trace[index].sequence_position += item.envelope_begin - 1;
    if (trace[index].state == p7T_M) {
      if (first_match == 0) {
        first_match = trace[index].sequence_position;
        first_model = trace[index].model_position;
      }
      last_match = trace[index].sequence_position;
      last_model = trace[index].model_position;
    }
  }
  if (first_match == 0 || last_match == 0 || first_model == 0 ||
      last_model == 0 || last_model > static_cast<uint32_t>(M)) {
    result.status = PLAN7_DOMAIN_RESCORE_ERANGE;
    result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
    add_rescore_reason<CollectReasonFacts>(
        reason_facts, region,
        PLAN7_DOMAIN_RESCORE_REASON_TRACE_COORDINATES_INVALID);
    return;
  }
  trace_counts[region] = count;
  result.alignment_begin = first_match;
  result.alignment_end = last_match;
  result.model_begin = first_model;
  result.model_end = last_model;
  result.action = PLAN7_DOMAIN_RESCORE_DEVICE_RESULT;
  add_rescore_reason<CollectReasonFacts>(
      reason_facts, region, PLAN7_DOMAIN_RESCORE_REASON_DEVICE_RESULT);
}

struct DeviceBuffers {
  RegionWork *work = nullptr;
  ResidentSelection *resident_selections = nullptr;
  uint64_t *matrix_offsets = nullptr;
  uint64_t *special_offsets = nullptr;
  uint64_t *trace_offsets = nullptr;
  float *forward_matrix = nullptr;
  float *posterior_matrix = nullptr;
  float *forward_specials = nullptr;
  float *posterior_specials = nullptr;
  float *null2 = nullptr;
  plan7_domain_rescore_trace_step *traces = nullptr;
  uint32_t *trace_counts = nullptr;
  plan7_domain_rescore_result *results = nullptr;
  uint32_t *reason_facts = nullptr;
};

void free_device_buffers(DeviceBuffers *buffers) {
  if (buffers == nullptr) return;
  cudaFree(buffers->reason_facts);
  cudaFree(buffers->results);
  cudaFree(buffers->trace_counts);
  cudaFree(buffers->traces);
  cudaFree(buffers->null2);
  cudaFree(buffers->posterior_specials);
  cudaFree(buffers->forward_specials);
  cudaFree(buffers->posterior_matrix);
  cudaFree(buffers->forward_matrix);
  cudaFree(buffers->trace_offsets);
  cudaFree(buffers->special_offsets);
  cudaFree(buffers->matrix_offsets);
  cudaFree(buffers->resident_selections);
  cudaFree(buffers->work);
  *buffers = {};
}

bool valid_profile_trace_transition(uint8_t left, uint8_t right) {
  switch (left) {
    case p7T_S: return right == p7T_N;
    case p7T_N: return right == p7T_N || right == p7T_B;
    case p7T_B: return right == p7T_M;
    case p7T_M:
      return right == p7T_M || right == p7T_D || right == p7T_I ||
             right == p7T_E;
    case p7T_D:
      return right == p7T_M || right == p7T_D || right == p7T_E;
    case p7T_I: return right == p7T_M || right == p7T_I;
    /* This stage reconfigures every envelope unihit, so E->J/J are not a
     * valid successful trace even though the stock traceback helper retains
     * its transition-gated J tie rule. */
    case p7T_E: return right == p7T_C;
    case p7T_C: return right == p7T_C || right == p7T_T;
    default: return false;
  }
}

bool validate_compact_trace(
    const plan7_domain_rescore_trace_step *trace, size_t count,
    const plan7_domain_rescore_result &result, uint32_t model_length) {
  if (trace == nullptr || count < 7 || trace[0].state != p7T_S ||
      trace[count - 1].state != p7T_T)
    return false;
  uint64_t next_sequence = result.envelope_begin;
  uint32_t first_match = 0;
  uint32_t last_match = 0;
  uint32_t first_model = 0;
  uint32_t last_model = 0;
  for (size_t index = 0; index < count; ++index) {
    const auto &step = trace[index];
    const bool allowed_state =
        step.state == p7T_S || step.state == p7T_N ||
        step.state == p7T_B || step.state == p7T_M ||
        step.state == p7T_D || step.state == p7T_I ||
        step.state == p7T_E || step.state == p7T_C ||
        step.state == p7T_T;
    if (!allowed_state ||
        step.reserved[0] != 0 || step.reserved[1] != 0 ||
        step.reserved[2] != 0 || !std::isfinite(step.posterior) ||
        step.posterior < 0.0f || step.posterior > 1.0001f)
      return false;
    if (index + 1 < count &&
        !valid_profile_trace_transition(step.state, trace[index + 1].state))
      return false;

    const bool core = step.state == p7T_M || step.state == p7T_D ||
                      step.state == p7T_I;
    const bool special = step.state == p7T_N || step.state == p7T_C;
    const bool emits = step.state == p7T_M || step.state == p7T_I ||
                       (special && index != 0 &&
                        trace[index - 1].state == step.state);
    if (core) {
      if (step.model_position == 0 ||
          step.model_position > model_length)
        return false;
    } else if (step.model_position != 0) {
      return false;
    }
    if (emits) {
      if (next_sequence > result.envelope_end ||
          step.sequence_position != next_sequence)
        return false;
      ++next_sequence;
    } else if (step.sequence_position != 0 || step.posterior != 0.0f) {
      return false;
    }

    if (index != 0) {
      const auto &previous = trace[index - 1];
      if (previous.state == p7T_M) {
        if ((step.state == p7T_M || step.state == p7T_D) &&
            step.model_position != previous.model_position + 1)
          return false;
        if (step.state == p7T_I &&
            step.model_position != previous.model_position)
          return false;
      } else if (previous.state == p7T_D) {
        if ((step.state == p7T_M || step.state == p7T_D) &&
            step.model_position != previous.model_position + 1)
          return false;
      } else if (previous.state == p7T_I) {
        if (step.state == p7T_M &&
            step.model_position != previous.model_position + 1)
          return false;
        if (step.state == p7T_I &&
            step.model_position != previous.model_position)
          return false;
      }
    }
    if (step.state == p7T_M) {
      if (first_match == 0) {
        first_match = step.sequence_position;
        first_model = step.model_position;
      }
      last_match = step.sequence_position;
      last_model = step.model_position;
    }
  }
  return next_sequence == static_cast<uint64_t>(result.envelope_end) + 1 &&
         first_match == result.alignment_begin &&
         last_match == result.alignment_end &&
         first_model == result.model_begin &&
         last_model == result.model_end;
}

bool validate_upstream_seal(
    const plan7_backward_domain_result *results, size_t result_count,
    const uint64_t *region_offsets, const plan7_simple_region *regions,
    size_t region_count,
    const plan7_backward_domain_provenance &provenance) {
  if (provenance.candidate_count != result_count ||
      provenance.region_count != region_count || region_offsets == nullptr ||
      region_offsets[0] != 0 || region_offsets[result_count] != region_count ||
      (result_count != 0 && results == nullptr) ||
      (region_count != 0 && regions == nullptr))
    return false;
  uint64_t result_hash = hash_u64(kHashOffset, UINT64_C(0x52455355));
  uint64_t region_hash = hash_u64(kHashOffset, UINT64_C(0x5245474e));
  for (size_t row = 0; row < result_count; ++row) {
    const plan7_backward_domain_result &result = results[row];
    FloatBits backward_bits{};
    FloatBits nexpected_bits{};
    backward_bits.value = result.backward_score;
    nexpected_bits.value = result.nexpected;
    result_hash = hash_u32(result_hash, result.profile_index);
    result_hash = hash_u32(result_hash, result.sequence_index);
    result_hash = hash_u32(result_hash, backward_bits.bits);
    result_hash = hash_u32(result_hash, nexpected_bits.bits);
    result_hash = hash_u32(result_hash, result.uncertain_count);
    result_hash = hash_u32(result_hash, result.region_count);
    result_hash = hash_u32(result_hash, result.multidomain_count);
    result_hash = hash_u32(
        result_hash, static_cast<uint32_t>(result.status) |
                         (static_cast<uint32_t>(result.route) << 8) |
                         (static_cast<uint32_t>(result.has_own_scales) << 16) |
                         (static_cast<uint32_t>(result.reserved) << 24));
    const uint64_t begin = region_offsets[row];
    const uint64_t end = region_offsets[row + 1];
    if (begin > end || end > region_count) return false;
    region_hash = hash_u64(region_hash, end - begin);
    for (uint64_t index = begin; index < end; ++index) {
      region_hash = hash_u32(region_hash, regions[index].begin);
      region_hash = hash_u32(region_hash, regions[index].end);
    }
  }
  return provenance.result_hash == hash_u64(result_hash, result_count) &&
         provenance.region_hash == hash_u64(region_hash, region_count);
}

bool merge_rescore_reason_facts(
    const uint32_t *active_result_indices, const uint32_t *active_facts,
    size_t active_count, uint32_t *source_facts, size_t source_count) {
  if ((active_count != 0 &&
       (active_result_indices == nullptr || active_facts == nullptr)) ||
      (source_count != 0 && source_facts == nullptr))
    return false;
  uint32_t previous = 0;
  bool have_previous = false;
  for (size_t active = 0; active < active_count; ++active) {
    const uint32_t source = active_result_indices[active];
    if (source >= source_count || (have_previous && source <= previous))
      return false;
    source_facts[source] = active_facts[active];
    previous = source;
    have_previous = true;
  }
  return true;
}

bool seal_rescore_provenance(struct plan7_domain_rescore_output *output);
int domain_rescore_run_impl(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_output *upstream,
    uint64_t compact_byte_budget, uint64_t matrix_byte_budget,
    uint64_t trace_byte_budget, bool collect_reason_facts,
    const float *whole_forward_scores, size_t whole_forward_score_count,
    const float *target_ga_cutoffs, size_t target_ga_cutoff_count,
    plan7_domain_rescore_output **output,
    char *error, size_t error_size);

}  // namespace

struct plan7_domain_rescore_output {
  std::vector<plan7_domain_rescore_result> results;
  std::vector<uint64_t> trace_offsets;
  std::vector<plan7_domain_rescore_trace_step> traces;
  std::vector<float> null2;
  std::vector<uint32_t> reason_facts;
  plan7_domain_rescore_provenance provenance;
  plan7_domain_rescore_statistics statistics;
  plan7_domain_rescore_residency_statistics residency_statistics{};
};

extern "C" int plan7_domain_rescore_run(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_output *upstream,
    uint64_t compact_byte_budget, uint64_t matrix_byte_budget,
    uint64_t trace_byte_budget,
    plan7_domain_rescore_output **output,
    char *error, size_t error_size) {
  try {
    return domain_rescore_run_impl(
        database, batch, upstream, compact_byte_budget,
        matrix_byte_budget, trace_byte_budget, false,
        nullptr, 0, nullptr, 0,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "isolated-domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "isolated-domain unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_domain_rescore_run_with_reason_facts(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_output *upstream,
    uint64_t compact_byte_budget, uint64_t matrix_byte_budget,
    uint64_t trace_byte_budget,
    plan7_domain_rescore_output **output,
    char *error, size_t error_size) {
  try {
    return domain_rescore_run_impl(
        database, batch, upstream, compact_byte_budget,
        matrix_byte_budget, trace_byte_budget, true,
        nullptr, 0, nullptr, 0,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "isolated-domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "isolated-domain unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_domain_rescore_run_ga(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_output *upstream,
    const float *whole_forward_scores, size_t whole_forward_score_count,
    const float *target_ga_cutoffs, size_t target_ga_cutoff_count,
    uint64_t compact_byte_budget, uint64_t matrix_byte_budget,
    uint64_t trace_byte_budget, plan7_domain_rescore_output **output,
    char *error, size_t error_size) {
  try {
    return domain_rescore_run_impl(
        database, batch, upstream, compact_byte_budget,
        matrix_byte_budget, trace_byte_budget, false,
        whole_forward_scores, whole_forward_score_count,
        target_ga_cutoffs, target_ga_cutoff_count,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "isolated-domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "isolated-domain unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_domain_rescore_run_ga_with_reason_facts(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_output *upstream,
    const float *whole_forward_scores, size_t whole_forward_score_count,
    const float *target_ga_cutoffs, size_t target_ga_cutoff_count,
    uint64_t compact_byte_budget, uint64_t matrix_byte_budget,
    uint64_t trace_byte_budget, plan7_domain_rescore_output **output,
    char *error, size_t error_size) {
  try {
    return domain_rescore_run_impl(
        database, batch, upstream, compact_byte_budget,
        matrix_byte_budget, trace_byte_budget, true,
        whole_forward_scores, whole_forward_score_count,
        target_ga_cutoffs, target_ga_cutoff_count,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "isolated-domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "isolated-domain unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_domain_rescore_output_destroy(
    plan7_domain_rescore_output **output,
    char *error, size_t error_size) {
  if (output == nullptr) {
    set_error(error, error_size, "isolated-domain output handle is null");
    return -1;
  }
  delete *output;
  *output = nullptr;
  return 0;
}

extern "C" size_t plan7_domain_rescore_output_result_count(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? 0 : output->results.size();
}

extern "C" const plan7_domain_rescore_result *
plan7_domain_rescore_output_results(
    const plan7_domain_rescore_output *output) {
  return output == nullptr || output->results.empty()
             ? nullptr : output->results.data();
}

extern "C" size_t plan7_domain_rescore_output_reason_count(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? 0 : output->reason_facts.size();
}

extern "C" const uint32_t *plan7_domain_rescore_output_reason_facts(
    const plan7_domain_rescore_output *output) {
  return output == nullptr || output->reason_facts.empty()
             ? nullptr : output->reason_facts.data();
}

extern "C" int plan7_domain_rescore_merge_reason_facts_for_test(
    const uint32_t *active_result_indices, const uint32_t *active_facts,
    size_t active_count, uint32_t *source_facts, size_t source_count) {
  return merge_rescore_reason_facts(
             active_result_indices, active_facts, active_count,
             source_facts, source_count)
             ? 0
             : -1;
}

extern "C" const uint64_t *plan7_domain_rescore_output_trace_offsets(
    const plan7_domain_rescore_output *output) {
  return output == nullptr || output->trace_offsets.empty()
             ? nullptr : output->trace_offsets.data();
}

extern "C" size_t plan7_domain_rescore_output_trace_count(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? 0 : output->traces.size();
}

extern "C" const plan7_domain_rescore_trace_step *
plan7_domain_rescore_output_traces(
    const plan7_domain_rescore_output *output) {
  return output == nullptr || output->traces.empty()
             ? nullptr : output->traces.data();
}

extern "C" size_t plan7_domain_rescore_output_null2_count(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? 0 : output->null2.size();
}

extern "C" const float *plan7_domain_rescore_output_null2(
    const plan7_domain_rescore_output *output) {
  return output == nullptr || output->null2.empty()
             ? nullptr : output->null2.data();
}

extern "C" const plan7_domain_rescore_provenance *
plan7_domain_rescore_output_provenance(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? nullptr : &output->provenance;
}

extern "C" const plan7_domain_rescore_statistics *
plan7_domain_rescore_output_statistics(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? nullptr : &output->statistics;
}

extern "C" const plan7_domain_rescore_residency_statistics *
plan7_domain_rescore_output_residency_statistics(
    const plan7_domain_rescore_output *output) {
  return output == nullptr ? nullptr : &output->residency_statistics;
}

extern "C" int plan7_domain_rescore_own_scale_required_for_test(float xB) {
  return backward_requires_own_scales(xB) ? 1 : 0;
}

extern "C" int plan7_domain_rescore_oatrace_j_predecessor_for_test(
    float jpath, float epath, int j_loop_enabled, int e_loop_enabled) {
  return oatrace_prefers_j(
             jpath, epath, j_loop_enabled != 0, e_loop_enabled != 0)
             ? 1 : 0;
}

namespace {

int domain_rescore_cpu_oracle_impl(
    uintptr_t source_profile_pointer,
    const uint8_t *residues, size_t residue_count,
    uint32_t envelope_begin, uint32_t envelope_end,
    plan7_domain_rescore_result *result,
    float *null2, size_t null2_count,
    plan7_domain_rescore_trace_step *trace, size_t trace_capacity,
    size_t *trace_count, char *error, size_t error_size) {
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      source_profile_pointer);
  if (source == nullptr || residues == nullptr || residue_count == 0 ||
      residue_count > kMaximumTargetLength || result == nullptr ||
      null2 == nullptr ||
      null2_count != PLAN7_DOMAIN_RESCORE_NULL2_COUNT ||
      trace_count == nullptr || (trace_capacity != 0 && trace == nullptr) ||
      envelope_begin == 0 || envelope_begin > envelope_end ||
      envelope_end > residue_count || source->abc == nullptr ||
      source->abc->type != eslAMINO || source->abc->K != 20 ||
      source->abc->Kp != 29 || source->M < 1 ||
      source->M > static_cast<int>(kMaximumModelLength) ||
      !p7_oprofile_IsLocal(source)) {
    set_error(error, error_size,
              "invalid isolated-domain CPU oracle input");
    return -1;
  }
  for (size_t index = 0; index < residue_count; ++index)
    if (residues[index] >= 29) {
      set_error(error, error_size,
                "isolated-domain oracle residue is out of range");
      return -1;
    }

  const int Ld = static_cast<int>(envelope_end - envelope_begin + 1);
  std::vector<ESL_DSQ> dsq;
  try {
    dsq.resize(residue_count + 2, eslDSQ_SENTINEL);
  } catch (...) {
    set_error(error, error_size,
              "isolated-domain oracle sequence allocation failed");
    return -1;
  }
  std::copy(residues, residues + residue_count, dsq.begin() + 1);
  P7_OPROFILE *profile = p7_oprofile_Clone(source);
  P7_OMX *forward = nullptr;
  P7_OMX *posterior = nullptr;
  P7_TRACE *stock_trace = nullptr;
  int status = eslOK;
  if (profile == nullptr ||
      p7_oprofile_ReconfigLength(
          profile, static_cast<int>(residue_count)) != eslOK ||
      p7_oprofile_ReconfigUnihit(
          profile, static_cast<int>(residue_count)) != eslOK ||
      (forward = p7_omx_Create(profile->M, Ld, Ld)) == nullptr ||
      (posterior = p7_omx_Create(profile->M, Ld, Ld)) == nullptr ||
      (stock_trace = p7_trace_CreateWithPP()) == nullptr) {
    p7_trace_Destroy(stock_trace);
    p7_omx_Destroy(posterior);
    p7_omx_Destroy(forward);
    p7_oprofile_Destroy(profile);
    set_error(error, error_size,
              "isolated-domain oracle workspace allocation failed");
    return -1;
  }

  *result = {};
  result->envelope_begin = envelope_begin;
  result->envelope_end = envelope_end;
  result->forward_score = NAN;
  result->backward_score = NAN;
  result->oa_score = NAN;
  result->domain_correction = NAN;
  result->score_consistency = NAN;
  result->status = PLAN7_DOMAIN_RESCORE_ERANGE;
  result->action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
  *trace_count = 0;
  status = p7_Forward(
      dsq.data() + envelope_begin - 1, Ld, profile, forward,
      &result->forward_score);
  if (status == eslOK)
    status = p7_Backward(
        dsq.data() + envelope_begin - 1, Ld, profile, forward,
        posterior, &result->backward_score);
  result->has_own_scales = posterior->has_own_scales ? 1 : 0;
  if (status == eslOK)
    status = p7_Decoding(profile, forward, posterior, posterior);
  if (status == eslOK)
    status = p7_OptimalAccuracy(
        profile, posterior, forward, &result->oa_score);
  if (status == eslOK)
    status = p7_OATrace(profile, posterior, forward, stock_trace);
  if (status == eslOK)
    status = p7_Null2_ByExpectation(profile, posterior, null2);
  if (status != eslOK) {
    p7_trace_Destroy(stock_trace);
    p7_omx_Destroy(posterior);
    p7_omx_Destroy(forward);
    p7_oprofile_Destroy(profile);
    return status;
  }
  if (static_cast<size_t>(stock_trace->N) > trace_capacity) {
    p7_trace_Destroy(stock_trace);
    p7_omx_Destroy(posterior);
    p7_omx_Destroy(forward);
    p7_oprofile_Destroy(profile);
    set_error(error, error_size,
              "isolated-domain oracle trace exceeds capacity");
    return eslERANGE;
  }

  uint32_t first_match = 0;
  uint32_t last_match = 0;
  uint32_t first_model = 0;
  uint32_t last_model = 0;
  for (int index = 0; index < stock_trace->N; ++index) {
    plan7_domain_rescore_trace_step step{};
    step.state = static_cast<uint8_t>(stock_trace->st[index]);
    step.model_position = static_cast<uint32_t>(stock_trace->k[index]);
    if (stock_trace->i[index] > 0)
      step.sequence_position = static_cast<uint32_t>(
          stock_trace->i[index] + envelope_begin - 1);
    step.posterior = stock_trace->pp[index];
    trace[index] = step;
    if (step.state == p7T_M) {
      if (first_match == 0) {
        first_match = step.sequence_position;
        first_model = step.model_position;
      }
      last_match = step.sequence_position;
      last_model = step.model_position;
    }
  }
  float correction = 0.0f;
  for (uint32_t position = envelope_begin;
       position <= envelope_end; ++position)
    correction += logf(null2[residues[position - 1]]);
  result->alignment_begin = first_match;
  result->alignment_end = last_match;
  result->model_begin = first_model;
  result->model_end = last_model;
  result->domain_correction = correction;
  result->score_consistency =
      std::fabs(result->forward_score - result->backward_score);
  result->status = PLAN7_DOMAIN_RESCORE_OK;
  result->action = PLAN7_DOMAIN_RESCORE_DEVICE_RESULT;
  *trace_count = static_cast<size_t>(stock_trace->N);

  p7_trace_Destroy(stock_trace);
  p7_omx_Destroy(posterior);
  p7_omx_Destroy(forward);
  p7_oprofile_Destroy(profile);
  return 0;
}

}  // namespace

extern "C" int plan7_domain_rescore_cpu_oracle(
    uintptr_t source_profile_pointer,
    const uint8_t *residues, size_t residue_count,
    uint32_t envelope_begin, uint32_t envelope_end,
    plan7_domain_rescore_result *result,
    float *null2, size_t null2_count,
    plan7_domain_rescore_trace_step *trace, size_t trace_capacity,
    size_t *trace_count, char *error, size_t error_size) {
  try {
    return domain_rescore_cpu_oracle_impl(
        source_profile_pointer, residues, residue_count,
        envelope_begin, envelope_end, result, null2, null2_count,
        trace, trace_capacity, trace_count, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size,
              "isolated-domain oracle host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "isolated-domain oracle unexpected native failure");
    return -1;
  }
}

namespace {

bool seal_rescore_provenance(plan7_domain_rescore_output *output) {
  if (output == nullptr ||
      output->trace_offsets.size() != output->results.size() + 1 ||
      output->null2.size() !=
          output->results.size() * PLAN7_DOMAIN_RESCORE_NULL2_COUNT ||
      output->trace_offsets.back() != output->traces.size())
    return false;
  uint64_t result_hash = hash_u64(kHashOffset, UINT64_C(0x44524553));
  uint64_t trace_hash = hash_u64(kHashOffset, UINT64_C(0x54524345));
  uint64_t null2_hash = hash_u64(kHashOffset, UINT64_C(0x4e554c32));
  for (size_t index = 0; index < output->results.size(); ++index) {
    const auto &result = output->results[index];
    result_hash = hash_u32(result_hash, result.row_index);
    result_hash = hash_u32(result_hash, result.profile_index);
    result_hash = hash_u32(result_hash, result.sequence_index);
    result_hash = hash_u32(result_hash, result.envelope_begin);
    result_hash = hash_u32(result_hash, result.envelope_end);
    result_hash = hash_u32(result_hash, result.alignment_begin);
    result_hash = hash_u32(result_hash, result.alignment_end);
    result_hash = hash_u32(result_hash, result.model_begin);
    result_hash = hash_u32(result_hash, result.model_end);
    for (const float value : {
             result.forward_score, result.backward_score, result.oa_score,
             result.domain_correction, result.score_consistency}) {
      FloatBits bits{};
      bits.value = value;
      result_hash = hash_u32(result_hash, bits.bits);
    }
    result_hash = hash_u32(
        result_hash, static_cast<uint32_t>(result.status) |
                         (static_cast<uint32_t>(result.action) << 8) |
                         (static_cast<uint32_t>(result.has_own_scales) << 16) |
                         (static_cast<uint32_t>(result.reserved) << 24));
    result_hash = hash_u32(result_hash, result.reserved2);
    const uint64_t begin = output->trace_offsets[index];
    const uint64_t end = output->trace_offsets[index + 1];
    if (begin > end || end > output->traces.size()) return false;
    trace_hash = hash_u64(trace_hash, end - begin);
    for (uint64_t step_index = begin; step_index < end; ++step_index) {
      const auto &step = output->traces[step_index];
      FloatBits posterior{};
      posterior.value = step.posterior;
      trace_hash = hash_u32(trace_hash, step.sequence_position);
      trace_hash = hash_u32(trace_hash, step.model_position);
      trace_hash = hash_u32(trace_hash, posterior.bits);
      trace_hash = hash_u32(trace_hash, step.state);
    }
    for (size_t residue = 0; residue < PLAN7_DOMAIN_RESCORE_NULL2_COUNT;
         ++residue) {
      FloatBits bits{};
      bits.value = output->null2[
          index * PLAN7_DOMAIN_RESCORE_NULL2_COUNT + residue];
      null2_hash = hash_u32(null2_hash, bits.bits);
    }
  }
  output->provenance.result_count = output->results.size();
  output->provenance.trace_count = output->traces.size();
  output->provenance.null2_count = output->null2.size();
  output->provenance.result_hash = hash_u64(
      result_hash, PLAN7_DOMAIN_RESCORE_RECORD_VERSION);
  output->provenance.trace_hash = hash_u64(
      trace_hash, output->traces.size());
  output->provenance.null2_hash = hash_u64(
      null2_hash, output->null2.size());
  return true;
}

int domain_rescore_run_impl(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_output *upstream,
    uint64_t compact_byte_budget, uint64_t matrix_byte_budget,
    uint64_t trace_byte_budget, bool collect_reason_facts,
    const float *whole_forward_scores, size_t whole_forward_score_count,
    const float *target_ga_cutoffs, size_t target_ga_cutoff_count,
    plan7_domain_rescore_output **output,
    char *error, size_t error_size) {
  const auto total_begin = std::chrono::steady_clock::now();
  if (database == nullptr || batch == nullptr || upstream == nullptr ||
      output == nullptr || *output != nullptr) {
    set_error(error, error_size, "invalid isolated-domain run arguments");
    return -1;
  }
  const auto *upstream_provenance =
      plan7_backward_domain_output_provenance(upstream);
  const auto *upstream_results =
      plan7_backward_domain_output_results(upstream);
  const auto *upstream_region_offsets =
      plan7_backward_domain_output_region_offsets(upstream);
  const auto *upstream_regions =
      plan7_backward_domain_output_regions(upstream);
  const size_t upstream_row_count =
      plan7_backward_domain_output_result_count(upstream);
  const size_t region_count =
      plan7_backward_domain_output_region_count(upstream);
  if (upstream_row_count > UINT32_MAX || region_count > UINT32_MAX ||
      region_count > SIZE_MAX / PLAN7_DOMAIN_RESCORE_NULL2_COUNT) {
    set_error(error, error_size,
              "isolated-domain input exceeds compact index limits");
    return -1;
  }
  if (upstream_provenance == nullptr ||
      plan7_backward_domain_output_is_production_calibrated(upstream) != 1 ||
      !validate_upstream_seal(
          upstream_results, upstream_row_count, upstream_region_offsets,
          upstream_regions, region_count, *upstream_provenance) ||
      plan7_forward_database_validate_provenance(
          database, &upstream_provenance->forward) != 1) {
    set_error(error, error_size,
              "isolated-domain upstream provenance mismatch");
    return -1;
  }

  plan7_forward_device_view profile_view{};
  plan7_ssv_sequence_batch_view sequence_view{};
  if (plan7_forward_database_get_device_view(
          database, &profile_view, error, error_size) != 0 ||
      plan7_ssv_sequence_batch_get_view(
          batch, &sequence_view, error, error_size) != 0)
    return -1;
  const bool ga_pruning = whole_forward_scores != nullptr ||
                          target_ga_cutoffs != nullptr ||
                          whole_forward_score_count != 0 ||
                          target_ga_cutoff_count != 0;
  if (ga_pruning &&
      ((whole_forward_score_count != 0 && whole_forward_scores == nullptr) ||
       target_ga_cutoffs == nullptr ||
       whole_forward_score_count != upstream_row_count ||
       target_ga_cutoff_count != profile_view.profile_count)) {
    set_error(error, error_size,
              "isolated-domain GA metadata is incomplete");
    return -1;
  }
  if (ga_pruning) {
    for (size_t profile = 0; profile < target_ga_cutoff_count; ++profile)
      if (!std::isfinite(target_ga_cutoffs[profile])) {
        set_error(error, error_size,
                  "isolated-domain GA cutoff is not finite");
        return -1;
      }
    for (size_t row = 0; row < whole_forward_score_count; ++row)
      if (!std::isfinite(whole_forward_scores[row])) {
        set_error(error, error_size,
                  "isolated-domain whole Forward score is not finite");
        return -1;
      }
  }
  if (profile_view.generation_id !=
          upstream_provenance->forward.database_generation ||
      sequence_view.generation_id !=
          upstream_provenance->forward.batch_generation ||
      profile_view.alphabet_size != 29 || sequence_view.alphabet_size != 29 ||
      !sequence_view.host_float_environment_valid) {
    set_error(error, error_size,
              "isolated-domain resident generation or environment mismatch");
    return -1;
  }
  int current_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&current_device);
  if (cuda_status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", cuda_status);
    return -1;
  }
  if (profile_view.device_ordinal != current_device ||
      sequence_view.device_ordinal != current_device ||
      (region_count != 0 &&
       (!device_allocation_on(profile_view.profiles, current_device) ||
        !device_allocation_on(profile_view.emissions, current_device) ||
        !device_allocation_on(profile_view.transitions, current_device) ||
        !device_allocation_on(sequence_view.device_residues, current_device) ||
        !device_allocation_on(sequence_view.device_offsets, current_device)))) {
    set_error(error, error_size,
              "isolated-domain input belongs to a different CUDA device");
    return -1;
  }

  plan7_backward_domain_resident_view resident_view{};
  const int resident_status = plan7_backward_domain_output_get_resident_view(
      upstream, &resident_view, error, error_size);
  if (resident_status < 0) return -1;
  const bool use_resident_backward = resident_status == 1;
  if (use_resident_backward &&
      (resident_view.database_generation !=
           upstream_provenance->forward.database_generation ||
       resident_view.batch_generation !=
           upstream_provenance->forward.batch_generation ||
       resident_view.result_hash != upstream_provenance->result_hash ||
       resident_view.region_hash != upstream_provenance->region_hash ||
       resident_view.row_count != upstream_row_count ||
       resident_view.region_count != region_count ||
       resident_view.device_ordinal != current_device ||
       resident_view.regions == nullptr ||
       !device_allocation_on(resident_view.regions, current_device))) {
    set_error(error, error_size,
              "isolated-domain resident Backward generation mismatch");
    return -1;
  }

  std::unique_ptr<plan7_domain_rescore_output> created(
      new (std::nothrow) plan7_domain_rescore_output{});
  if (!created) {
    set_error(error, error_size, "isolated-domain output allocation failed");
    return -1;
  }
  created->provenance.backward = *upstream_provenance;
  created->statistics.upstream_row_count = upstream_row_count;
  created->statistics.region_count = region_count;
  created->residency_statistics.resident_input_count =
      use_resident_backward ? 1 : 0;
  if (collect_reason_facts) {
    try {
      created->reason_facts.assign(region_count, 0);
    } catch (...) {
      set_error(error, error_size,
                "isolated-domain reason allocation failed");
      return -1;
    }
  }
  const uint64_t requested_compact_limit = compact_byte_budget == 0
      ? kCompactOutputByteLimit
      : std::min(compact_byte_budget, kCompactOutputByteLimit);
  /* Even a global fallback needs its terminal trace-offset word. */
  const uint64_t compact_limit = std::max(
      requested_compact_limit, static_cast<uint64_t>(sizeof(uint64_t)));
  created->statistics.compact_output_byte_limit = compact_limit;
  uint64_t minimum_compact_bytes = sizeof(uint64_t);
  uint64_t per_region_compact_bytes;
  const bool compact_fits =
      checked_multiply(
          PLAN7_DOMAIN_RESCORE_NULL2_COUNT, sizeof(float),
          &per_region_compact_bytes) &&
      checked_add(
          per_region_compact_bytes, sizeof(plan7_domain_rescore_result),
          &per_region_compact_bytes) &&
      checked_add(
          per_region_compact_bytes, sizeof(uint64_t),
          &per_region_compact_bytes) &&
      checked_multiply(
          region_count, per_region_compact_bytes,
          &minimum_compact_bytes) &&
      checked_add(minimum_compact_bytes, sizeof(uint64_t),
                  &minimum_compact_bytes) &&
      minimum_compact_bytes <= compact_limit;
  if (!compact_fits) {
    for (size_t row = 0; row < upstream_row_count; ++row)
      if (upstream_results[row].route == PLAN7_BACKWARD_DOMAIN_SIMPLE)
        ++created->statistics.simple_row_count;
    created->trace_offsets.assign(1, 0);
    created->statistics.cpu_required_count = region_count;
    created->statistics.cap_fallback_count = region_count;
    created->statistics.global_cpu_fallback_count = region_count;
    if (collect_reason_facts)
      std::fill(created->reason_facts.begin(), created->reason_facts.end(),
                PLAN7_DOMAIN_RESCORE_REASON_GLOBAL_COMPACT_BUDGET);
    created->statistics.compact_output_bytes = sizeof(uint64_t);
    created->statistics.total_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - total_begin).count();
    if (!seal_rescore_provenance(created.get())) {
      set_error(error, error_size,
                "isolated-domain global fallback sealing failed");
      return -1;
    }
    *output = created.release();
    return 0;
  }
  try {
    created->results.resize(region_count);
    created->null2.assign(
        region_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT, NAN);
    created->trace_offsets.assign(region_count + 1, 0);
  } catch (...) {
    set_error(error, error_size, "isolated-domain output allocation failed");
    return -1;
  }

  std::vector<plan7_forward_snapshot_profile> snapshots;
  try {
    snapshots.resize(profile_view.profile_count);
  } catch (...) {
    set_error(error, error_size, "isolated-domain profile allocation failed");
    return -1;
  }
  for (size_t profile = 0; profile < profile_view.profile_count; ++profile)
    if (plan7_forward_database_get_profile_snapshot(
            database, profile, &snapshots[profile], error, error_size) != 0)
      return -1;

  const uint64_t matrix_limit = matrix_byte_budget == 0
      ? kMatrixByteLimit : std::min(matrix_byte_budget, kMatrixByteLimit);
  const uint64_t requested_trace_limit = trace_byte_budget == 0
      ? kTraceByteLimit : std::min(trace_byte_budget, kTraceByteLimit);
  const uint64_t trace_limit = std::min(
      requested_trace_limit,
      compact_limit - minimum_compact_bytes);
  std::vector<RegionWork> active_work;
  std::vector<uint64_t> matrix_offsets(1, 0);
  std::vector<uint64_t> special_offsets(1, 0);
  std::vector<uint64_t> trace_capacity_offsets(1, 0);
  std::vector<uint32_t> result_rows(region_count, UINT32_MAX);
  uint64_t dense_bytes = 0;
  uint64_t trace_bytes = 0;
  uint64_t work_cells = 0;
  try {
    active_work.reserve(region_count);
    matrix_offsets.reserve(region_count + 1);
    special_offsets.reserve(region_count + 1);
    trace_capacity_offsets.reserve(region_count + 1);
  } catch (...) {
    set_error(error, error_size, "isolated-domain work allocation failed");
    return -1;
  }

  for (size_t row = 0; row < upstream_row_count; ++row) {
    const auto &source = upstream_results[row];
    const uint64_t region_begin = upstream_region_offsets[row];
    const uint64_t region_end = upstream_region_offsets[row + 1];
    if (source.profile_index >= profile_view.profile_count ||
        source.sequence_index >= sequence_view.sequence_count) {
      set_error(error, error_size,
                "isolated-domain upstream index is out of range");
      return -1;
    }
    if (source.route != PLAN7_BACKWARD_DOMAIN_SIMPLE) {
      if (region_begin != region_end) {
        set_error(error, error_size,
                  "isolated-domain non-simple row owns regions");
        return -1;
      }
      continue;
    }
    ++created->statistics.simple_row_count;
    if (source.status != PLAN7_BACKWARD_DOMAIN_OK ||
        source.uncertain_count != 0 ||
        source.multidomain_count != 0 || region_begin == region_end ||
        source.region_count != region_end - region_begin) {
      set_error(error, error_size,
                "isolated-domain SIMPLE row violates its seal");
      return -1;
    }
    const auto &snapshot = snapshots[source.profile_index];
    const uint64_t target_length =
        sequence_view.host_lengths[source.sequence_index];
    if (snapshot.model_length < 1 ||
        snapshot.model_length > kMaximumModelLength ||
        snapshot.q != std::max<uint32_t>(
                          2, (snapshot.model_length + 3) / 4) ||
        snapshot.mode != p7_LOCAL || snapshot.nj != 1.0f ||
        target_length == 0 || target_length > kMaximumTargetLength) {
      set_error(error, error_size,
                "isolated-domain profile or target dimensions are invalid");
      return -1;
    }

    uint64_t row_dense_bytes = 0;
    uint64_t row_trace_bytes = 0;
    uint64_t row_work_cells = 0;
    /* Backward/domain can validly classify a row as SIMPLE after taking its
     * own scales. Keep that row in the sealed result journal, but conservatively
     * leave its isolated envelopes on the CPU. */
    bool row_fits = !source.has_own_scales;
    bool row_work_cap = false;
    bool matrix_cap = false;
    bool trace_cap = false;
    bool run_work_cap = false;
    for (uint64_t region = region_begin; region < region_end; ++region) {
      const auto interval = upstream_regions[region];
      plan7_domain_rescore_result &result = created->results[region];
      result = {};
      result.row_index = static_cast<uint32_t>(row);
      result.profile_index = source.profile_index;
      result.sequence_index = source.sequence_index;
      result.envelope_begin = interval.begin;
      result.envelope_end = interval.end;
      result.forward_score = NAN;
      result.backward_score = NAN;
      result.oa_score = NAN;
      result.domain_correction = NAN;
      result.score_consistency = NAN;
      result.status = source.has_own_scales
                          ? PLAN7_DOMAIN_RESCORE_ERANGE
                          : PLAN7_DOMAIN_RESCORE_ECAP;
      result.action = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      result.has_own_scales = source.has_own_scales;
      if (collect_reason_facts && source.has_own_scales)
        created->reason_facts[region] |=
            PLAN7_DOMAIN_RESCORE_REASON_UPSTREAM_OWN_SCALES;
      result_rows[region] = static_cast<uint32_t>(row);
      if (interval.begin == 0 || interval.begin > interval.end ||
          interval.end > target_length ||
          (region != region_begin &&
           interval.begin <= upstream_regions[region - 1].end)) {
        set_error(error, error_size,
                  "isolated-domain interval is invalid");
        return -1;
      }
      const uint64_t length = interval.end - interval.begin + 1;
      uint64_t cells;
      uint64_t region_dense;
      uint64_t special_cells;
      uint64_t trace_capacity;
      uint64_t region_trace;
      uint64_t region_work;
      if (!checked_add(length, 1, &cells) ||
          !checked_multiply(cells, snapshot.q, &cells) ||
          !checked_multiply(cells, p7X_NSCELLS * kSubwarp, &cells) ||
          !checked_multiply(cells, 2 * sizeof(float), &region_dense) ||
          !checked_add(length, 1, &special_cells) ||
          !checked_multiply(special_cells, p7X_NXCELLS, &special_cells) ||
          !checked_multiply(special_cells, 2 * sizeof(float), &special_cells) ||
          !checked_add(region_dense, special_cells, &region_dense) ||
          !checked_add(length, snapshot.model_length + 16,
                       &trace_capacity) ||
          !checked_multiply(trace_capacity,
                            sizeof(plan7_domain_rescore_trace_step),
                            &region_trace) ||
          !checked_multiply(length, snapshot.model_length, &region_work) ||
          !checked_add(row_dense_bytes, region_dense, &row_dense_bytes) ||
          !checked_add(row_trace_bytes, region_trace, &row_trace_bytes) ||
          !checked_add(row_work_cells, region_work, &row_work_cells)) {
        set_error(error, error_size, "isolated-domain work size overflow");
        return -1;
      }
      if (region_work > kMaximumRowWorkCells) {
        row_fits = false;
        if (collect_reason_facts)
          created->reason_facts[region] |=
              PLAN7_DOMAIN_RESCORE_REASON_REGION_WORK_CAP;
      }
    }
    row_work_cap = row_work_cells > kMaximumRowWorkCells;
    matrix_cap =
        dense_bytes > matrix_limit - std::min(matrix_limit, row_dense_bytes) ||
        row_dense_bytes > matrix_limit - dense_bytes;
    trace_cap =
        trace_bytes > trace_limit - std::min(trace_limit, row_trace_bytes) ||
        row_trace_bytes > trace_limit - trace_bytes;
    run_work_cap =
        work_cells > kMaximumRunWorkCells -
                         std::min(kMaximumRunWorkCells, row_work_cells) ||
        row_work_cells > kMaximumRunWorkCells - work_cells;
    if (row_work_cap || matrix_cap || trace_cap || run_work_cap)
      row_fits = false;
    if (collect_reason_facts &&
        (row_work_cap || matrix_cap || trace_cap || run_work_cap)) {
      uint32_t facts = 0;
      if (row_work_cap) facts |= PLAN7_DOMAIN_RESCORE_REASON_ROW_WORK_CAP;
      if (matrix_cap) facts |= PLAN7_DOMAIN_RESCORE_REASON_MATRIX_CAP;
      if (trace_cap) facts |= PLAN7_DOMAIN_RESCORE_REASON_TRACE_CAP;
      if (run_work_cap) facts |= PLAN7_DOMAIN_RESCORE_REASON_RUN_WORK_CAP;
      for (uint64_t region = region_begin; region < region_end; ++region)
        created->reason_facts[region] |= facts;
    }
    if (!row_fits) {
      continue;
    }

    for (uint64_t region = region_begin; region < region_end; ++region) {
      const auto interval = upstream_regions[region];
      const uint64_t length = interval.end - interval.begin + 1;
      uint64_t matrix_cells;
      uint64_t special_cells;
      uint64_t trace_capacity;
      checked_add(length, 1, &matrix_cells);
      checked_multiply(matrix_cells, snapshot.q, &matrix_cells);
      checked_multiply(matrix_cells, p7X_NSCELLS * kSubwarp,
                       &matrix_cells);
      checked_add(length, 1, &special_cells);
      checked_multiply(special_cells, p7X_NXCELLS, &special_cells);
      checked_add(length, snapshot.model_length + 16, &trace_capacity);
      active_work.push_back({
          static_cast<uint32_t>(region), static_cast<uint32_t>(row),
          source.profile_index, source.sequence_index, interval.begin,
          interval.end, static_cast<uint32_t>(target_length), 0});
      matrix_offsets.push_back(matrix_offsets.back() + matrix_cells);
      special_offsets.push_back(special_offsets.back() + special_cells);
      trace_capacity_offsets.push_back(
          trace_capacity_offsets.back() + trace_capacity);
      created->results[region].status = PLAN7_DOMAIN_RESCORE_OK;
    }
    dense_bytes += row_dense_bytes;
    trace_bytes += row_trace_bytes;
    work_cells += row_work_cells;
  }

  const size_t active_count = active_work.size();
  DeviceBuffers buffers{};
  std::vector<ResidentSelection> resident_selections;
  if (use_resident_backward) {
    resident_selections.resize(active_count);
    for (size_t active = 0; active < active_count; ++active) {
      resident_selections[active] = {
          active_work[active].result_index,
          active_work[active].row_index};
    }
  }
  std::vector<plan7_domain_rescore_result> device_results(active_count);
  for (size_t active = 0; active < active_count; ++active)
    device_results[active] =
        created->results[active_work[active].result_index];
  std::vector<float> active_null2(
      active_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT, NAN);
  std::vector<uint32_t> active_trace_counts(active_count, 0);
  std::vector<uint32_t> active_reason_facts;
  if (collect_reason_facts) {
    active_reason_facts.resize(active_count);
    for (size_t active = 0; active < active_count; ++active)
      active_reason_facts[active] =
          created->reason_facts[active_work[active].result_index];
  }
  std::vector<plan7_domain_rescore_trace_step> active_traces(
      static_cast<size_t>(trace_capacity_offsets.back()));
  if (active_count != 0) {
    size_t work_bytes;
    size_t offset_bytes;
    size_t matrix_bytes;
    size_t special_bytes;
    size_t result_bytes;
    size_t null2_bytes;
    size_t trace_storage_bytes;
    size_t trace_count_bytes;
    size_t resident_selection_bytes = 0;
    size_t reason_bytes = 0;
    if (!checked_bytes(active_count, sizeof(RegionWork), &work_bytes) ||
        !checked_bytes(active_count + 1, sizeof(uint64_t), &offset_bytes) ||
        !checked_bytes(static_cast<size_t>(matrix_offsets.back()),
                       sizeof(float), &matrix_bytes) ||
        !checked_bytes(static_cast<size_t>(special_offsets.back()),
                       sizeof(float), &special_bytes) ||
        !checked_bytes(active_count,
                       sizeof(plan7_domain_rescore_result), &result_bytes) ||
        !checked_bytes(active_null2.size(), sizeof(float), &null2_bytes) ||
        !checked_bytes(active_traces.size(),
                       sizeof(plan7_domain_rescore_trace_step),
                       &trace_storage_bytes) ||
        !checked_bytes(active_count, sizeof(uint32_t), &trace_count_bytes) ||
        (use_resident_backward &&
         !checked_bytes(active_count, sizeof(ResidentSelection),
                        &resident_selection_bytes)) ||
        (collect_reason_facts &&
         !checked_bytes(active_count, sizeof(uint32_t), &reason_bytes))) {
      set_error(error, error_size, "isolated-domain device size overflow");
      return -1;
    }
    const auto upload_begin = std::chrono::steady_clock::now();
    cudaEvent_t begin_event = nullptr;
    cudaEvent_t end_event = nullptr;
    cudaEvent_t resident_begin_event = nullptr;
    cudaEvent_t resident_end_event = nullptr;
    const auto cleanup_device_run = [&]() {
      if (resident_end_event != nullptr)
        cudaEventDestroy(resident_end_event);
      if (resident_begin_event != nullptr)
        cudaEventDestroy(resident_begin_event);
      if (end_event != nullptr) cudaEventDestroy(end_event);
      if (begin_event != nullptr) cudaEventDestroy(begin_event);
      resident_end_event = nullptr;
      resident_begin_event = nullptr;
      end_event = nullptr;
      begin_event = nullptr;
      free_device_buffers(&buffers);
    };
#define CUDA_RUN(call)                                                        \
    do {                                                                      \
      cuda_status = (call);                                                   \
      if (cuda_status != cudaSuccess) {                                       \
        set_cuda_error(error, error_size, #call, cuda_status);                \
        cleanup_device_run();                                                 \
        return -1;                                                            \
      }                                                                       \
    } while (0)
    CUDA_RUN(cudaMalloc(&buffers.work, work_bytes));
    if (use_resident_backward)
      CUDA_RUN(cudaMalloc(&buffers.resident_selections,
                          resident_selection_bytes));
    CUDA_RUN(cudaMalloc(&buffers.matrix_offsets, offset_bytes));
    CUDA_RUN(cudaMalloc(&buffers.special_offsets, offset_bytes));
    CUDA_RUN(cudaMalloc(&buffers.trace_offsets, offset_bytes));
    CUDA_RUN(cudaMalloc(&buffers.forward_matrix, matrix_bytes));
    CUDA_RUN(cudaMalloc(&buffers.posterior_matrix, matrix_bytes));
    CUDA_RUN(cudaMalloc(&buffers.forward_specials, special_bytes));
    CUDA_RUN(cudaMalloc(&buffers.posterior_specials, special_bytes));
    CUDA_RUN(cudaMalloc(&buffers.null2, null2_bytes));
    CUDA_RUN(cudaMalloc(&buffers.traces, trace_storage_bytes));
    CUDA_RUN(cudaMalloc(&buffers.trace_counts, trace_count_bytes));
    CUDA_RUN(cudaMalloc(&buffers.results, result_bytes));
    if (collect_reason_facts)
      CUDA_RUN(cudaMalloc(&buffers.reason_facts, reason_bytes));
    const auto upstream_upload_begin = std::chrono::steady_clock::now();
    if (use_resident_backward) {
      CUDA_RUN(cudaMemcpy(buffers.resident_selections,
                          resident_selections.data(), resident_selection_bytes,
                          cudaMemcpyHostToDevice));
      created->residency_statistics.eliminated_upstream_h2d_bytes =
          static_cast<uint64_t>(work_bytes) + result_bytes;
      created->residency_statistics.resident_selection_h2d_bytes =
          resident_selection_bytes;
    } else {
      CUDA_RUN(cudaMemcpy(buffers.work, active_work.data(), work_bytes,
                          cudaMemcpyHostToDevice));
      CUDA_RUN(cudaMemcpy(buffers.results, device_results.data(), result_bytes,
                          cudaMemcpyHostToDevice));
      created->residency_statistics.upstream_h2d_bytes =
          static_cast<uint64_t>(work_bytes) + result_bytes;
    }
    created->residency_statistics.upstream_upload_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - upstream_upload_begin).count();
    CUDA_RUN(cudaMemcpy(buffers.matrix_offsets, matrix_offsets.data(),
                        offset_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.special_offsets, special_offsets.data(),
                        offset_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.trace_offsets, trace_capacity_offsets.data(),
                        offset_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.null2, active_null2.data(), null2_bytes,
                        cudaMemcpyHostToDevice));
    if (collect_reason_facts)
      CUDA_RUN(cudaMemcpy(buffers.reason_facts, active_reason_facts.data(),
                          reason_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemset(buffers.traces, 0, trace_storage_bytes));
    CUDA_RUN(cudaMemset(buffers.trace_counts, 0, trace_count_bytes));
    if (use_resident_backward) {
      CUDA_RUN(cudaEventCreate(&resident_begin_event));
      CUDA_RUN(cudaEventCreate(&resident_end_event));
      CUDA_RUN(cudaEventRecord(resident_begin_event));
      const unsigned prepare_blocks = static_cast<unsigned>(
          (active_count + kThreads - 1) / kThreads);
      prepare_resident_rescore_inputs_kernel<<<prepare_blocks, kThreads>>>(
          resident_view.regions, buffers.resident_selections, active_count,
          buffers.work, buffers.results);
      CUDA_RUN(cudaGetLastError());
      CUDA_RUN(cudaEventRecord(resident_end_event));
    }
    created->statistics.upload_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - upload_begin).count();

    CUDA_RUN(cudaEventCreate(&begin_event));
    CUDA_RUN(cudaEventCreate(&end_event));
    CUDA_RUN(cudaEventRecord(begin_event));
    const unsigned blocks = static_cast<unsigned>(
        (active_count + kRegionsPerBlock - 1) / kRegionsPerBlock);
    if (collect_reason_facts)
      isolated_forward_kernel<true><<<blocks, kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.work, buffers.matrix_offsets,
          buffers.special_offsets, active_count, buffers.forward_matrix,
          buffers.forward_specials, buffers.results, buffers.reason_facts);
    else
      isolated_forward_kernel<false><<<blocks, kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.work, buffers.matrix_offsets,
          buffers.special_offsets, active_count, buffers.forward_matrix,
          buffers.forward_specials, buffers.results, nullptr);
    CUDA_RUN(cudaGetLastError());

    if (ga_pruning) {
      const auto classification_begin = std::chrono::steady_clock::now();
      CUDA_RUN(cudaMemcpy(device_results.data(), buffers.results, result_bytes,
                          cudaMemcpyDeviceToHost));
      size_t active = 0;
      while (active < active_count) {
        const size_t row_begin = active;
        const uint32_t source_row = active_work[active].row_index;
        const uint32_t profile_index = active_work[active].profile_index;
        const uint32_t target_length = active_work[active].target_length;
        while (active < active_count &&
               active_work[active].row_index == source_row)
          ++active;
        if (source_row >= upstream_row_count ||
            profile_index >= target_ga_cutoff_count || target_length == 0) {
          set_error(error, error_size,
                    "isolated-domain GA row identity changed");
          cleanup_device_run();
          return -1;
        }
        const float p1 = static_cast<float>(target_length) /
                         static_cast<float>(target_length + 1);
        const float null_score = static_cast<float>(
            static_cast<float>(target_length) *
                std::log(static_cast<double>(p1)) +
            std::log(1.0 - static_cast<double>(p1)));
        if (!std::isfinite(null_score)) {
          set_error(error, error_size,
                    "isolated-domain GA null score is invalid");
          cleanup_device_run();
          return -1;
        }
        const double input_error = static_cast<double>(
            static_cast<float>(0.004));
        const double round_error = static_cast<double>(
            static_cast<float>(0.00001));
        const double whole_upper =
            (static_cast<double>(whole_forward_scores[source_row]) +
             input_error - static_cast<double>(null_score)) /
                static_cast<double>(eslCONST_LOG2) +
            round_error;
        double reconstruction_nats = 0.0;
        uint64_t skipped_cells = 0;
        for (size_t item = row_begin; item < active; ++item) {
          const auto &result = device_results[item];
          const auto &work = active_work[item];
          if (work.profile_index != profile_index ||
              work.target_length != target_length ||
              result.status != PLAN7_DOMAIN_RESCORE_OK ||
              !std::isfinite(result.forward_score)) {
            reconstruction_nats = HUGE_VAL;
            break;
          }
          const double upper =
              static_cast<double>(result.forward_score) + input_error;
          if (upper > 0.0) reconstruction_nats += upper;
          const uint64_t length =
              static_cast<uint64_t>(work.envelope_end) -
              static_cast<uint64_t>(work.envelope_begin) + 1;
          const uint64_t model_length =
              snapshots[profile_index].model_length;
          if (model_length != 0 && length > UINT64_MAX / model_length) {
            set_error(error, error_size,
                      "isolated-domain GA skipped work overflows");
            cleanup_device_run();
            return -1;
          }
          const uint64_t cells = length * model_length;
          if (skipped_cells > UINT64_MAX - cells) {
            set_error(error, error_size,
                      "isolated-domain GA skipped work overflows");
            cleanup_device_run();
            return -1;
          }
          skipped_cells += cells;
        }
        const double reconstruction_upper =
            (reconstruction_nats - static_cast<double>(null_score)) /
                static_cast<double>(eslCONST_LOG2) +
            round_error;
        const double target_upper =
            std::max(whole_upper, reconstruction_upper);
        if (target_upper <
            static_cast<double>(target_ga_cutoffs[profile_index])) {
          ++created->statistics.certified_ga_row_count;
          created->statistics.certified_ga_region_count += active - row_begin;
          if (created->statistics.certified_ga_skipped_work_cells >
              UINT64_MAX - skipped_cells) {
            set_error(error, error_size,
                      "isolated-domain GA skipped work total overflows");
            cleanup_device_run();
            return -1;
          }
          created->statistics.certified_ga_skipped_work_cells += skipped_cells;
          for (size_t item = row_begin; item < active; ++item) {
            device_results[item].action =
                PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT;
            if (collect_reason_facts)
              active_reason_facts[item] |=
                  PLAN7_DOMAIN_RESCORE_REASON_CERTIFIED_GA_REJECT;
          }
        }
      }
      CUDA_RUN(cudaMemcpy(buffers.results, device_results.data(), result_bytes,
                          cudaMemcpyHostToDevice));
      if (collect_reason_facts)
        CUDA_RUN(cudaMemcpy(buffers.reason_facts,
                            active_reason_facts.data(), reason_bytes,
                            cudaMemcpyHostToDevice));
      created->statistics.ga_classification_milliseconds =
          std::chrono::duration<float, std::milli>(
              std::chrono::steady_clock::now() - classification_begin)
              .count();
    }
    if (collect_reason_facts)
      isolated_backward_decode_kernel<true><<<blocks, kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.work, buffers.matrix_offsets,
          buffers.special_offsets, active_count, buffers.forward_matrix,
          buffers.forward_specials, buffers.posterior_matrix,
          buffers.posterior_specials, buffers.results, buffers.reason_facts);
    else
      isolated_backward_decode_kernel<false><<<blocks, kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.work, buffers.matrix_offsets,
          buffers.special_offsets, active_count, buffers.forward_matrix,
          buffers.forward_specials, buffers.posterior_matrix,
          buffers.posterior_specials, buffers.results, nullptr);
    CUDA_RUN(cudaGetLastError());
    if (collect_reason_facts)
      isolated_null2_oa_trace_kernel<true><<<blocks, kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.work, buffers.matrix_offsets,
          buffers.special_offsets, buffers.trace_offsets, active_count,
          buffers.forward_matrix, buffers.forward_specials,
          buffers.posterior_matrix, buffers.posterior_specials,
          buffers.null2, buffers.traces, buffers.trace_counts, buffers.results,
          buffers.reason_facts);
    else
      isolated_null2_oa_trace_kernel<false><<<blocks, kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.work, buffers.matrix_offsets,
          buffers.special_offsets, buffers.trace_offsets, active_count,
          buffers.forward_matrix, buffers.forward_specials,
          buffers.posterior_matrix, buffers.posterior_specials,
          buffers.null2, buffers.traces, buffers.trace_counts, buffers.results,
          nullptr);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaEventRecord(end_event));
    CUDA_RUN(cudaEventSynchronize(end_event));
    CUDA_RUN(cudaEventElapsedTime(&created->statistics.kernel_milliseconds,
                                  begin_event, end_event));
    if (use_resident_backward) {
      CUDA_RUN(cudaEventElapsedTime(
          &created->residency_statistics.resident_prepare_milliseconds,
          resident_begin_event, resident_end_event));
      cudaEventDestroy(resident_end_event);
      cudaEventDestroy(resident_begin_event);
      resident_end_event = nullptr;
      resident_begin_event = nullptr;
    }
    cudaEventDestroy(end_event);
    cudaEventDestroy(begin_event);
    end_event = nullptr;
    begin_event = nullptr;

    const auto download_begin = std::chrono::steady_clock::now();
    CUDA_RUN(cudaMemcpy(device_results.data(), buffers.results, result_bytes,
                        cudaMemcpyDeviceToHost));
    CUDA_RUN(cudaMemcpy(active_null2.data(), buffers.null2, null2_bytes,
                        cudaMemcpyDeviceToHost));
    CUDA_RUN(cudaMemcpy(active_trace_counts.data(), buffers.trace_counts,
                        trace_count_bytes, cudaMemcpyDeviceToHost));
    CUDA_RUN(cudaMemcpy(active_traces.data(), buffers.traces,
                        trace_storage_bytes, cudaMemcpyDeviceToHost));
    if (collect_reason_facts)
      CUDA_RUN(cudaMemcpy(active_reason_facts.data(), buffers.reason_facts,
                          reason_bytes, cudaMemcpyDeviceToHost));
    created->statistics.download_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - download_begin).count();
#undef CUDA_RUN
    free_device_buffers(&buffers);
  }

  /* A source row is atomic: one failed/capped envelope sends every envelope
   * in that row through the ordinary CPU continuation. */
  std::vector<uint32_t> active_result_indices;
  if (collect_reason_facts) {
    active_result_indices.resize(active_count);
    for (size_t active = 0; active < active_count; ++active)
      active_result_indices[active] = active_work[active].result_index;
    if (!merge_rescore_reason_facts(
            active_result_indices.data(), active_reason_facts.data(),
            active_count, created->reason_facts.data(),
            created->reason_facts.size())) {
      set_error(error, error_size,
                "isolated-domain reason source remap changed");
      return -1;
    }
  }
  for (size_t active = 0; active < active_count; ++active) {
    const size_t result_index = active_work[active].result_index;
    const auto expected = created->results[result_index];
    const auto observed = device_results[active];
    if (observed.row_index == expected.row_index &&
        observed.profile_index == expected.profile_index &&
        observed.sequence_index == expected.sequence_index &&
        observed.envelope_begin == expected.envelope_begin &&
        observed.envelope_end == expected.envelope_end &&
        observed.reserved == expected.reserved &&
        observed.reserved2 == expected.reserved2) {
      created->results[result_index] = observed;
    } else {
      created->results[result_index].status = PLAN7_DOMAIN_RESCORE_ERANGE;
      created->results[result_index].action =
          PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      if (collect_reason_facts)
        created->reason_facts[result_index] |=
            PLAN7_DOMAIN_RESCORE_REASON_IDENTITY_MISMATCH;
    }
    std::copy_n(
        active_null2.data() +
            active * PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
        PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
        created->null2.data() +
            result_index * PLAN7_DOMAIN_RESCORE_NULL2_COUNT);
  }
  std::vector<size_t> active_for_result(region_count, SIZE_MAX);
  for (size_t active = 0; active < active_count; ++active)
    active_for_result[active_work[active].result_index] = active;
  std::vector<uint8_t> row_failed(upstream_row_count, 0);
  for (size_t result_index = 0; result_index < region_count; ++result_index) {
    auto &result = created->results[result_index];
    const uint32_t source_row = result_rows[result_index];
    if (source_row == UINT32_MAX || source_row >= upstream_row_count) {
      set_error(error, error_size,
                "isolated-domain host row mapping is invalid");
      return -1;
    }
    const auto &snapshot = snapshots[result.profile_index];
    if (result.action == PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT) {
      if (!ga_pruning || result.status != PLAN7_DOMAIN_RESCORE_OK ||
          result.has_own_scales || !std::isfinite(result.forward_score) ||
          result.alignment_begin != 0 || result.alignment_end != 0 ||
          result.model_begin != 0 || result.model_end != 0) {
        set_error(error, error_size,
                  "isolated-domain certified GA result is invalid");
        return -1;
      }
      continue;
    }
    if (result.action != PLAN7_DOMAIN_RESCORE_DEVICE_RESULT ||
        result.status != PLAN7_DOMAIN_RESCORE_OK ||
        result.has_own_scales ||
        !std::isfinite(result.forward_score) ||
        !std::isfinite(result.backward_score) ||
        !std::isfinite(result.oa_score) ||
        !std::isfinite(result.domain_correction) ||
        !std::isfinite(result.score_consistency) ||
        result.score_consistency < 0.0f ||
        result.score_consistency > 0.002f ||
        result.alignment_begin < result.envelope_begin ||
        result.alignment_begin > result.alignment_end ||
        result.alignment_end > result.envelope_end ||
        result.model_begin == 0 ||
        result.model_begin > result.model_end ||
        result.model_end > snapshot.model_length) {
      row_failed[source_row] = 1;
      if (collect_reason_facts && active_for_result[result_index] != SIZE_MAX)
        created->reason_facts[result_index] |=
            PLAN7_DOMAIN_RESCORE_REASON_HOST_RESULT_INVALID;
    }
  }

  /* Device traceback and null2 are compact final outputs, not diagnostics.
   * Validate every value before allowing any envelope in its source row to
   * leave the ordinary CPU continuation. */
  for (size_t active = 0; active < active_count; ++active) {
    const size_t result_index = active_work[active].result_index;
    const uint32_t source_row = result_rows[result_index];
    if (row_failed[source_row]) continue;
    if (created->results[result_index].action ==
        PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT)
      continue;
    const uint64_t capacity =
        trace_capacity_offsets[active + 1] -
        trace_capacity_offsets[active];
    const uint32_t count = active_trace_counts[active];
    bool trace_valid = count >= 2 && count <= capacity;
    const size_t trace_begin =
        static_cast<size_t>(trace_capacity_offsets[active]);
    const auto &work = active_work[active];
    const auto &snapshot = snapshots[work.profile_index];
    if (trace_valid)
      trace_valid = validate_compact_trace(
          active_traces.data() + trace_begin, count,
          created->results[result_index], snapshot.model_length);
    const float *row_null2 =
        active_null2.data() +
        active * PLAN7_DOMAIN_RESCORE_NULL2_COUNT;
    bool null2_valid = true;
    for (size_t residue = 0;
         null2_valid && residue < PLAN7_DOMAIN_RESCORE_NULL2_COUNT; ++residue)
      null2_valid =
          std::isfinite(row_null2[residue]) && row_null2[residue] > 0.0f;
    if (!trace_valid || !null2_valid) {
      row_failed[source_row] = 1;
      if (collect_reason_facts) {
        if (!trace_valid)
          created->reason_facts[result_index] |=
              PLAN7_DOMAIN_RESCORE_REASON_HOST_TRACE_INVALID;
        if (!null2_valid)
          created->reason_facts[result_index] |=
              PLAN7_DOMAIN_RESCORE_REASON_HOST_NULL2_INVALID;
      }
    }
  }

  for (size_t result_index = 0; result_index < region_count; ++result_index)
    if (row_failed[result_rows[result_index]]) {
      created->results[result_index].action =
          PLAN7_DOMAIN_RESCORE_CPU_REQUIRED;
      if (collect_reason_facts)
        created->reason_facts[result_index] |=
            PLAN7_DOMAIN_RESCORE_REASON_ROW_ATOMIC_PROPAGATION;
      std::fill_n(
          created->null2.data() +
              result_index * PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
          PLAN7_DOMAIN_RESCORE_NULL2_COUNT, NAN);
    }

  try {
    for (size_t result_index = 0; result_index < region_count; ++result_index) {
      created->trace_offsets[result_index] = created->traces.size();
      const auto &result = created->results[result_index];
      if (result.action != PLAN7_DOMAIN_RESCORE_DEVICE_RESULT) continue;
      const size_t active = active_for_result[result_index];
      if (active == SIZE_MAX ||
          active_trace_counts[active] >
              trace_capacity_offsets[active + 1] -
                  trace_capacity_offsets[active]) {
        set_error(error, error_size,
                  "isolated-domain trace compaction mismatch");
        return -1;
      }
      const size_t trace_begin =
          static_cast<size_t>(trace_capacity_offsets[active]);
      created->traces.insert(
          created->traces.end(), active_traces.begin() + trace_begin,
          active_traces.begin() + trace_begin + active_trace_counts[active]);
    }
    created->trace_offsets[region_count] = created->traces.size();
  } catch (...) {
    set_error(error, error_size, "isolated-domain trace allocation failed");
    return -1;
  }

  for (size_t result_index = 0; result_index < created->results.size();
       ++result_index) {
    const auto &result = created->results[result_index];
    if (result.action == PLAN7_DOMAIN_RESCORE_DEVICE_RESULT)
      ++created->statistics.device_result_count;
    else if (result.action == PLAN7_DOMAIN_RESCORE_CERTIFIED_GA_REJECT) {
      if (collect_reason_facts &&
          !(created->reason_facts[result_index] &
            PLAN7_DOMAIN_RESCORE_REASON_CERTIFIED_GA_REJECT)) {
        set_error(error, error_size,
                  "isolated-domain GA result lacks source fact");
        return -1;
      }
    }
    else {
      ++created->statistics.cpu_required_count;
      if (result.status == PLAN7_DOMAIN_RESCORE_ECAP)
        ++created->statistics.cap_fallback_count;
      else
        ++created->statistics.numeric_fallback_count;
      if (collect_reason_facts) {
        if (created->reason_facts[result_index] == 0)
          created->reason_facts[result_index] |=
              PLAN7_DOMAIN_RESCORE_REASON_OTHER_CPU_REQUIRED;
        created->reason_facts[result_index] |=
            PLAN7_DOMAIN_RESCORE_REASON_FINAL_CPU_REQUIRED;
      }
    }
  }
  created->statistics.work_cells = work_cells;
  created->statistics.forward_matrix_bytes =
      matrix_offsets.back() * sizeof(float);
  created->statistics.posterior_matrix_bytes =
      matrix_offsets.back() * sizeof(float);
  created->statistics.special_workspace_bytes =
      special_offsets.back() * sizeof(float) * 2;
  created->statistics.trace_workspace_bytes = trace_bytes;
  created->statistics.compact_output_bytes =
      created->results.size() * sizeof(plan7_domain_rescore_result) +
      created->traces.size() * sizeof(plan7_domain_rescore_trace_step) +
      created->null2.size() * sizeof(float) +
      created->trace_offsets.size() * sizeof(uint64_t);
  created->statistics.total_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  if (!seal_rescore_provenance(created.get())) {
    set_error(error, error_size,
              "isolated-domain output provenance sealing failed");
    return -1;
  }
  *output = created.release();
  return 0;
}

}  // namespace
