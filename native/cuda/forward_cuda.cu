#include "forward_cuda.h"

#include "bias_cuda.h"
#include "f3_threshold.h"
#include "postfilter_cuda.h"
#include "ssv_cuda.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

extern "C" {
#include <easel.h>
#include <esl_exponential.h>
#include <hmmer.h>
#include <impl_sse/impl_sse.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <vector>

static_assert(sizeof(float) == 4,
              "Forward CUDA requires binary32 float");
static_assert(sizeof(plan7_forward_result) == PLAN7_FORWARD_RECORD_SIZE,
              "Forward result ABI size changed");
static_assert(offsetof(plan7_forward_result, sequence_index) == 0 &&
              offsetof(plan7_forward_result, fwdsc) == 4 &&
              offsetof(plan7_forward_result, status) == 8 &&
              offsetof(plan7_forward_result, action) == 9 &&
              offsetof(plan7_forward_result, reserved) == 10,
              "Forward result ABI layout changed");
static_assert(sizeof(plan7_forward_snapshot_profile) == 48 &&
              offsetof(plan7_forward_snapshot_profile, emission_offset) == 0 &&
              offsetof(plan7_forward_snapshot_profile, transition_offset) == 8 &&
              offsetof(plan7_forward_snapshot_profile, q) == 16 &&
              offsetof(plan7_forward_snapshot_profile, model_length) == 20 &&
              offsetof(plan7_forward_snapshot_profile, e_move) == 24 &&
              offsetof(plan7_forward_snapshot_profile, mode) == 44,
              "Forward snapshot descriptor ABI changed");
static_assert(sizeof(plan7_forward_provenance) == 72,
              "Forward provenance ABI changed");

namespace {

constexpr int kThreads = 256;
constexpr int kSubwarp = 4;
constexpr int kDefaultCandidatesPerWarp = 1;
constexpr int kWarpsPerBlock = kThreads / 32;
constexpr uint32_t kSubwarpPolicyVersion = 1;
constexpr uint64_t kLongAverageWorkCells = UINT64_C(1) << 17;
constexpr uint64_t kMinimumCtaCoverageNumerator = 3;
constexpr uint64_t kMinimumCtaCoverageDenominator = 4;
constexpr int kGatherThreads = 256;
constexpr uint64_t kMaximumModelLength = 100000;
constexpr uint64_t kDpWorkspaceByteLimit = UINT64_C(256) << 20;
constexpr uint64_t kXmxWorkspaceByteLimit = UINT64_C(512) << 20;
constexpr uint64_t kGatheredOutputByteLimit =
    PLAN7_FORWARD_MAX_GATHERED_BYTES;
constexpr double kLog2 = 0.69314718055994529;
static_assert(kGatheredOutputByteLimit % sizeof(float) == 0);

struct ForwardProfile {
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
  uintptr_t alphabet_pointer;
};

static_assert(sizeof(plan7_forward_device_profile) == 32,
              "Forward device descriptor must stay cache-compact");
static_assert(p7O_NTRANS == 8,
              "Forward transition-row footprint changed");
static_assert((29 + p7O_NTRANS) * kSubwarp * sizeof(float) == 592,
              "Forward packed-row footprint changed");

struct ForwardLengthTransitions {
  float move;
  float loop;
};

struct ForwardKernelResult {
  uint32_t status_and_f3;
  uint32_t score_bits;
};

static_assert(sizeof(ForwardKernelResult) == 8,
              "Forward kernel result transport changed");

enum KernelF3Decision : uint32_t {
  kKernelF3Unavailable = 0,
  kKernelF3Reject = 1,
  kKernelF3Pass = 2
};

constexpr uint32_t kKernelStatusMask = UINT32_C(0xff);
constexpr unsigned kKernelF3Shift = 8;

union FloatBits {
  float value;
  uint32_t bits;
};

union DoubleBits {
  double value;
  uint64_t bits;
};

constexpr uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr uint64_t kHashPrime = UINT64_C(1099511628211);
std::atomic<uint64_t> next_forward_database_generation{1};

uint64_t allocate_forward_database_generation() {
  uint64_t generation = next_forward_database_generation.fetch_add(
      1, std::memory_order_relaxed);
  if (generation == 0)
    generation = next_forward_database_generation.fetch_add(
        1, std::memory_order_relaxed);
  return generation;
}

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

uint64_t ceiling_divide(uint64_t numerator, uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

int select_forward_candidates_per_warp(
    int requested_candidates_per_warp, size_t tile_candidate_count,
    uint64_t work_cells, uint64_t xmx_workspace_bytes,
    int multiprocessor_count, int l2_cache_bytes,
    plan7_forward_subwarp_statistics *statistics) {
  statistics->policy_version = kSubwarpPolicyVersion;
  statistics->requested_candidates_per_warp =
      static_cast<uint32_t>(requested_candidates_per_warp);
  statistics->multiprocessor_count =
      static_cast<uint32_t>(multiprocessor_count);
  statistics->l2_cache_bytes = static_cast<uint64_t>(l2_cache_bytes);
  statistics->policy_tile_candidate_count = tile_candidate_count;
  statistics->policy_xmx_workspace_bytes = xmx_workspace_bytes;
  statistics->average_work_cells = tile_candidate_count == 0
      ? 0 : work_cells / statistics->candidate_subwarp_count;
  statistics->average_model_length = tile_candidate_count == 0
      ? 0 : statistics->model_length_sum /
          statistics->candidate_subwarp_count;
  statistics->average_target_length = tile_candidate_count == 0
      ? 0 : statistics->target_length_sum /
          statistics->candidate_subwarp_count;
  statistics->short_width4_workspace_limit_bytes =
      statistics->l2_cache_bytes;
  statistics->long_packed_workspace_limit_bytes =
      statistics->l2_cache_bytes > UINT64_MAX / 2
          ? UINT64_MAX : 2 * statistics->l2_cache_bytes;
  statistics->minimum_cta_count = ceiling_divide(
      static_cast<uint64_t>(multiprocessor_count) *
          kMinimumCtaCoverageNumerator,
      kMinimumCtaCoverageDenominator);
  statistics->width1_cta_count = ceiling_divide(
      tile_candidate_count,
      static_cast<uint64_t>(kWarpsPerBlock));
  statistics->width2_cta_count = ceiling_divide(
      tile_candidate_count,
      static_cast<uint64_t>(2 * kWarpsPerBlock));
  statistics->width4_cta_count = ceiling_divide(
      tile_candidate_count,
      static_cast<uint64_t>(4 * kWarpsPerBlock));

  if (tile_candidate_count == 0) {
    statistics->candidates_per_warp = kDefaultCandidatesPerWarp;
    statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_NO_KERNEL;
    return kDefaultCandidatesPerWarp;
  }
  if (requested_candidates_per_warp != 0) {
    statistics->candidates_per_warp =
        static_cast<uint32_t>(requested_candidates_per_warp);
    statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_FORCED;
    return requested_candidates_per_warp;
  }

  const bool width2_has_coverage =
      statistics->width2_cta_count >= statistics->minimum_cta_count;
  const bool width4_has_coverage =
      statistics->width4_cta_count >= statistics->minimum_cta_count;
  const bool long_rows =
      statistics->average_work_cells >= kLongAverageWorkCells;
  const bool divergent_lengths =
      statistics->maximum_target_length >
          2 * statistics->average_target_length;
  if (!width2_has_coverage) {
    statistics->candidates_per_warp = 1;
    statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_SPARSE_WIDTH1;
    return 1;
  }
  if (!long_rows) {
    if (width4_has_coverage &&
        xmx_workspace_bytes <=
            statistics->short_width4_workspace_limit_bytes) {
      statistics->candidates_per_warp = 4;
      statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH4;
      return 4;
    }
    if (divergent_lengths && !width4_has_coverage) {
      statistics->candidates_per_warp = 1;
      statistics->policy_reason =
          PLAN7_FORWARD_SUBWARP_POLICY_DIVERGENT_WIDTH1;
      return 1;
    }
    statistics->candidates_per_warp = 2;
    statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_SHORT_WIDTH2;
    return 2;
  }
  if (xmx_workspace_bytes <=
      statistics->long_packed_workspace_limit_bytes) {
    if (width4_has_coverage) {
      statistics->candidates_per_warp = 4;
      statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH4;
      return 4;
    }
    statistics->candidates_per_warp = 2;
    statistics->policy_reason = PLAN7_FORWARD_SUBWARP_POLICY_LONG_WIDTH2;
    return 2;
  }
  statistics->candidates_per_warp = 1;
  statistics->policy_reason =
      PLAN7_FORWARD_SUBWARP_POLICY_LONG_SATURATED_WIDTH1;
  return 1;
}

bool aligned_vector_address(const void *allocation, uintptr_t *address) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation);
  if (raw == 0 || raw > UINTPTR_MAX - 15) return false;
  *address = (raw + 15) & ~static_cast<uintptr_t>(15);
  return true;
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

bool valid_profile_storage(const P7_OPROFILE *profile) {
  if (profile == nullptr || profile->abc == nullptr ||
      profile->abc->type != eslAMINO || profile->abc->K != 20 ||
      profile->abc->Kp != 29 || profile->M < 1 ||
      profile->M > static_cast<int>(kMaximumModelLength) ||
      profile->allocM < profile->M ||
      profile->allocM > static_cast<int>(kMaximumModelLength) ||
      profile->allocQ4 != p7O_NQF(profile->allocM) ||
      profile->allocQ4 < p7O_NQF(profile->M) ||
      profile->rfv == nullptr || profile->tfv == nullptr ||
      profile->rfv_mem == nullptr || profile->tfv_mem == nullptr)
    return false;

  uintptr_t rfv_base;
  uintptr_t tfv_base;
  if (!aligned_vector_address(profile->rfv_mem, &rfv_base) ||
      !aligned_vector_address(profile->tfv_mem, &tfv_base) ||
      reinterpret_cast<uintptr_t>(profile->tfv) != tfv_base)
    return false;
  const size_t row_stride = static_cast<size_t>(profile->allocQ4);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const uintptr_t expected = rfv_base +
        static_cast<size_t>(residue) * row_stride * sizeof(__m128);
    if (profile->rfv[residue] == nullptr ||
        reinterpret_cast<uintptr_t>(profile->rfv[residue]) != expected)
      return false;
  }
  return true;
}

bool valid_forward_specials(const P7_OPROFILE *profile) {
  if ((profile->mode != p7_LOCAL && profile->mode != p7_UNILOCAL) ||
      (profile->nj != 0.0f && profile->nj != 1.0f))
    return false;
  if (profile->nj == 0.0f)
    return profile->xf[p7O_E][p7O_MOVE] == 1.0f &&
           profile->xf[p7O_E][p7O_LOOP] == 0.0f;
  return profile->xf[p7O_E][p7O_MOVE] == 0.5f &&
         profile->xf[p7O_E][p7O_LOOP] == 0.5f;
}

bool valid_probability_rows(const P7_OPROFILE *profile) {
  const int q_count = p7O_NQF(profile->M);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const auto *values = reinterpret_cast<const float *>(profile->rfv[residue]);
    for (int cell = 0; cell < q_count * kSubwarp; ++cell)
      if (!std::isfinite(values[cell]) || values[cell] < 0.0f) return false;
  }
  const auto *transitions = reinterpret_cast<const float *>(profile->tfv);
  for (int cell = 0; cell < q_count * p7O_NTRANS * kSubwarp; ++cell)
    if (!std::isfinite(transitions[cell]) || transitions[cell] < 0.0f)
      return false;
  return true;
}

ForwardLengthTransitions length_transitions_for(
    const ForwardProfile &profile, int length) {
  const float numerator = 2.0f + profile.nj;
  const float denominator =
      static_cast<float>(length) + 2.0f + profile.nj;
  const float move = numerator / denominator;
  return {move, 1.0f - move};
}

__device__ __forceinline__ float add_rn(float left, float right) {
  return __fadd_rn(left, right);
}

__device__ __forceinline__ float mul_rn(float left, float right) {
  return __fmul_rn(left, right);
}

__device__ __forceinline__ float subwarp_shift_right(float value,
                                                     unsigned mask,
                                                     int sublane) {
  const float shifted = __shfl_up_sync(mask, value, 1, kSubwarp);
  return sublane == 0 ? 0.0f : shifted;
}

__device__ __forceinline__ float sse_horizontal_sum(float value,
                                                    unsigned mask,
                                                    int sublane) {
  float rotated = __shfl_sync(mask, value, (sublane + 1) & 3, kSubwarp);
  value = add_rn(value, rotated);
  rotated = __shfl_sync(mask, value, (sublane + 2) & 3, kSubwarp);
  value = add_rn(value, rotated);
  return __shfl_sync(mask, value, 0, kSubwarp);
}

template<int CandidatesPerWarp>
__global__ void forward_kernel(
    const uint8_t *__restrict__ residues,
    const uint64_t *__restrict__ sequence_offsets,
    const plan7_forward_device_profile *__restrict__ profiles,
    const float *__restrict__ emissions,
    const float *__restrict__ transitions,
    const uint32_t *__restrict__ candidate_profiles,
    const uint32_t *__restrict__ candidate_sequences,
    const float *__restrict__ filter_scores,
    const uint32_t *__restrict__ f3_threshold_bits,
    const ForwardLengthTransitions *__restrict__ length_transitions,
    const uint64_t *__restrict__ dp_offsets,
    const uint64_t *__restrict__ x_offsets,
    size_t candidate_begin, size_t tile_count, uint64_t tile_dp_begin,
    uint64_t tile_x_begin, float *__restrict__ dp_storage,
    float *__restrict__ xmx_storage,
    ForwardKernelResult *__restrict__ results) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int candidate_in_warp = lane >> 2;
  const int sublane = lane & 3;
  static_assert(CandidatesPerWarp == 1 || CandidatesPerWarp == 2 ||
                CandidatesPerWarp == 4 || CandidatesPerWarp == 8,
                "unsupported Forward candidates-per-warp specialization");
  if (candidate_in_warp >= CandidatesPerWarp) return;
  constexpr int candidates_per_block =
      CandidatesPerWarp * kWarpsPerBlock;
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * candidates_per_block +
      warp_in_block * CandidatesPerWarp + candidate_in_warp;
  if (tile_candidate >= tile_count) return;
  const size_t candidate = candidate_begin + tile_candidate;
  const unsigned subwarp_mask = 0xFU << (candidate_in_warp * kSubwarp);

  const uint32_t profile_index = candidate_profiles[candidate];
  const uint32_t sequence_index = candidate_sequences[candidate];
  const plan7_forward_device_profile profile = profiles[profile_index];
  const int q_count = static_cast<int>(profile.q);
  const uint64_t sequence_start = sequence_offsets[sequence_index];
  const int sequence_length = static_cast<int>(
      sequence_offsets[sequence_index + 1] - sequence_start);
  const ForwardLengthTransitions length = length_transitions[candidate];
  float *mmx = dp_storage + dp_offsets[candidate] - tile_dp_begin;
  float *imx = mmx + static_cast<uint64_t>(q_count) * kSubwarp;
  float *dmx = imx + static_cast<uint64_t>(q_count) * kSubwarp;

  for (int q = 0; q < q_count; ++q) {
    mmx[q * kSubwarp + sublane] = 0.0f;
    imx[q * kSubwarp + sublane] = 0.0f;
    dmx[q * kSubwarp + sublane] = 0.0f;
  }

  float xN = 1.0f;
  float xJ = 0.0f;
  float xB = length.move;
  float xC = 0.0f;
  float totscale = 0.0f;
  float *xmx = xmx_storage + x_offsets[candidate] - tile_x_begin;
  if (sublane == 0) {
    xmx[p7X_E] = 0.0f;
    xmx[p7X_N] = 1.0f;
    xmx[p7X_J] = 0.0f;
    xmx[p7X_B] = length.move;
    xmx[p7X_C] = 0.0f;
    xmx[p7X_SCALE] = 1.0f;
  }

  for (int i = 0; i < sequence_length; ++i) {
    const unsigned residue = residues[sequence_start + i];
    const uint64_t emission_base = profile.emission_offset +
        static_cast<uint64_t>(residue) * q_count * kSubwarp;
    float dcv = 0.0f;
    float xEv = 0.0f;
    float mpv = subwarp_shift_right(
        mmx[(q_count - 1) * kSubwarp + sublane], subwarp_mask, sublane);
    float dpv = subwarp_shift_right(
        dmx[(q_count - 1) * kSubwarp + sublane], subwarp_mask, sublane);
    float ipv = subwarp_shift_right(
        imx[(q_count - 1) * kSubwarp + sublane], subwarp_mask, sublane);

    for (int q = 0; q < q_count; ++q) {
      const uint64_t transition_base = profile.transition_offset +
          static_cast<uint64_t>(q) * p7O_NTRANS * kSubwarp + sublane;
      float value = mul_rn(
          xB, transitions[transition_base + p7O_BM * kSubwarp]);
      value = add_rn(value, mul_rn(
          mpv, transitions[transition_base + p7O_MM * kSubwarp]));
      value = add_rn(value, mul_rn(
          ipv, transitions[transition_base + p7O_IM * kSubwarp]));
      value = add_rn(value, mul_rn(
          dpv, transitions[transition_base + p7O_DM * kSubwarp]));
      value = mul_rn(
          value, emissions[emission_base + q * kSubwarp + sublane]);
      xEv = add_rn(xEv, value);

      const float old_m = mmx[q * kSubwarp + sublane];
      const float old_i = imx[q * kSubwarp + sublane];
      const float old_d = dmx[q * kSubwarp + sublane];
      mmx[q * kSubwarp + sublane] = value;
      dmx[q * kSubwarp + sublane] = dcv;
      dcv = mul_rn(
          value, transitions[transition_base + p7O_MD * kSubwarp]);
      float insert = mul_rn(
          old_m, transitions[transition_base + p7O_MI * kSubwarp]);
      insert = add_rn(insert, mul_rn(
          old_i, transitions[transition_base + p7O_II * kSubwarp]));
      imx[q * kSubwarp + sublane] = insert;
      mpv = old_m;
      ipv = old_i;
      dpv = old_d;
    }

    dcv = subwarp_shift_right(dcv, subwarp_mask, sublane);
    dmx[sublane] = 0.0f;
    for (int q = 0; q < q_count; ++q) {
      const uint64_t transition_base = profile.transition_offset +
          static_cast<uint64_t>(q) * p7O_NTRANS * kSubwarp + sublane;
      const float updated = add_rn(dcv, dmx[q * kSubwarp + sublane]);
      dmx[q * kSubwarp + sublane] = updated;
      dcv = mul_rn(
          updated, transitions[transition_base + p7O_DD * kSubwarp]);
    }
    for (int pass = 1; pass < 4; ++pass) {
      dcv = subwarp_shift_right(dcv, subwarp_mask, sublane);
      bool any_change = false;
      for (int q = 0; q < q_count; ++q) {
        const uint64_t transition_base = profile.transition_offset +
            static_cast<uint64_t>(q) * p7O_NTRANS * kSubwarp + sublane;
        const float current = dmx[q * kSubwarp + sublane];
        const float updated = add_rn(dcv, current);
        any_change = any_change || updated > current;
        dmx[q * kSubwarp + sublane] = updated;
        dcv = mul_rn(
            dcv, transitions[transition_base + p7O_DD * kSubwarp]);
      }
      if (profile.model_length >= 100 &&
          __any_sync(subwarp_mask, any_change) == 0)
        break;
    }

    for (int q = 0; q < q_count; ++q)
      xEv = add_rn(dmx[q * kSubwarp + sublane], xEv);
    float xE = sse_horizontal_sum(xEv, subwarp_mask, sublane);

    xN = mul_rn(xN, length.loop);
    xC = add_rn(mul_rn(xC, length.loop), mul_rn(xE, profile.e_move));
    xJ = add_rn(mul_rn(xJ, length.loop), mul_rn(xE, profile.e_loop));
    xB = add_rn(mul_rn(xJ, length.move), mul_rn(xN, length.move));

    if (xE > 1.0e4f) {
      xN = __fdiv_rn(xN, xE);
      xC = __fdiv_rn(xC, xE);
      xJ = __fdiv_rn(xJ, xE);
      xB = __fdiv_rn(xB, xE);
      const float inverse = __fdiv_rn(1.0f, xE);
      for (int q = 0; q < q_count; ++q) {
        mmx[q * kSubwarp + sublane] =
            mul_rn(mmx[q * kSubwarp + sublane], inverse);
        dmx[q * kSubwarp + sublane] =
            mul_rn(dmx[q * kSubwarp + sublane], inverse);
        imx[q * kSubwarp + sublane] =
            mul_rn(imx[q * kSubwarp + sublane], inverse);
      }
      if (sublane == 0) {
        totscale = static_cast<float>(
            static_cast<double>(totscale) + log(static_cast<double>(xE)));
        xmx[(i + 1) * p7X_NXCELLS + p7X_SCALE] = xE;
      }
      xE = 1.0f;
    } else if (sublane == 0) {
      xmx[(i + 1) * p7X_NXCELLS + p7X_SCALE] = 1.0f;
    }
    if (sublane == 0) {
      xmx[(i + 1) * p7X_NXCELLS + p7X_E] = xE;
      xmx[(i + 1) * p7X_NXCELLS + p7X_N] = xN;
      xmx[(i + 1) * p7X_NXCELLS + p7X_J] = xJ;
      xmx[(i + 1) * p7X_NXCELLS + p7X_B] = xB;
      xmx[(i + 1) * p7X_NXCELLS + p7X_C] = xC;
    }
  }

  if (sublane == 0) {
    ForwardKernelResult result{};
    uint32_t status = eslOK;
    KernelF3Decision f3_decision = kKernelF3Unavailable;
    FloatBits score{};
    if (isnan(xC) || (sequence_length > 0 && xC == 0.0f) || isinf(xC)) {
      status = eslERANGE;
      score.bits = UINT32_C(0x7fc00000);
    } else {
      const float terminal = mul_rn(xC, length.move);
      score.value = static_cast<float>(
          static_cast<double>(totscale) + log(static_cast<double>(terminal)));
    }
    if (status == eslOK && isfinite(score.value) &&
        isfinite(filter_scores[candidate])) {
      FloatBits threshold{};
      threshold.bits = f3_threshold_bits[profile_index];
      if (!isnan(threshold.value)) {
        const float difference = __fsub_rn(
            score.value, filter_scores[candidate]);
        const double quotient = __ddiv_rn(
            static_cast<double>(difference), kLog2);
        const float bit_score = __double2float_rn(quotient);
        if (!isnan(bit_score))
          f3_decision = bit_score >= threshold.value
              ? kKernelF3Pass : kKernelF3Reject;
      }
    }
    result.status_and_f3 = status |
        (static_cast<uint32_t>(f3_decision) << kKernelF3Shift);
    result.score_bits = score.bits;
    results[candidate] = result;
  }
}

__global__ void gather_specials_kernel(
    const float *tile_xmx, const uint64_t *global_x_offsets,
    const uint32_t *survivor_candidates,
    const uint64_t *survivor_output_offsets, size_t survivor_count,
    uint64_t tile_x_begin, float *gathered) {
  const size_t survivor = static_cast<size_t>(blockIdx.x);
  if (survivor >= survivor_count) return;
  const uint32_t candidate = survivor_candidates[survivor];
  const uint64_t input_begin =
      global_x_offsets[candidate] - tile_x_begin;
  const uint64_t input_end =
      global_x_offsets[static_cast<size_t>(candidate) + 1] - tile_x_begin;
  const uint64_t output_begin = survivor_output_offsets[survivor];
  const uint64_t cell_count = input_end - input_begin;
  for (uint64_t cell = threadIdx.x; cell < cell_count;
       cell += blockDim.x)
    gathered[output_begin + cell] = tile_xmx[input_begin + cell];
}

__global__ void apply_f3_overrides_kernel(
    ForwardKernelResult *results, const uint64_t *overrides,
    size_t override_count) {
  const size_t override_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (override_index >= override_count) return;
  const uint64_t encoded = overrides[override_index];
  const uint32_t candidate = static_cast<uint32_t>(encoded >> 32);
  const uint32_t decision = static_cast<uint32_t>(encoded);
  results[candidate].status_and_f3 =
      (results[candidate].status_and_f3 & kKernelStatusMask) |
      (decision << kKernelF3Shift);
}

__global__ void mark_f3_survivors_kernel(
    const ForwardKernelResult *results, size_t candidate_begin,
    size_t tile_count, size_t accepted_candidate_end,
    uint64_t *survivor_ranks) {
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile_candidate >= tile_count) return;
  const size_t candidate = candidate_begin + tile_candidate;
  const KernelF3Decision decision = static_cast<KernelF3Decision>(
      results[candidate].status_and_f3 >> kKernelF3Shift);
  survivor_ranks[tile_candidate] =
      candidate < accepted_candidate_end && decision == kKernelF3Pass ? 1 : 0;
}

__global__ void scatter_f3_survivors_kernel(
    const ForwardKernelResult *results, const uint64_t *survivor_ranks,
    size_t candidate_begin, size_t tile_count,
    size_t accepted_candidate_end, uint32_t *survivor_candidates) {
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tile_candidate >= tile_count) return;
  const size_t candidate = candidate_begin + tile_candidate;
  const KernelF3Decision decision = static_cast<KernelF3Decision>(
      results[candidate].status_and_f3 >> kKernelF3Shift);
  if (candidate < accepted_candidate_end && decision == kKernelF3Pass)
    survivor_candidates[survivor_ranks[tile_candidate]] =
        static_cast<uint32_t>(candidate);
}

__global__ void gather_resident_f2_inputs_kernel(
    const plan7_bias_candidate *source_candidates,
    const plan7_postfilter_result *source_results,
    const uint32_t *selected_sources,
    size_t selected_count,
    uint32_t *candidate_profiles,
    uint32_t *candidate_sequences,
    float *filter_scores) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= selected_count) return;
  const uint32_t source = selected_sources[candidate];
  const plan7_bias_candidate mapping = source_candidates[source];
  candidate_profiles[candidate] = mapping.profile_index;
  candidate_sequences[candidate] = mapping.sequence_index;
  filter_scores[candidate] = source_results[source].filtersc;
}

__global__ void survivor_cell_counts_kernel(
    const uint32_t *survivor_candidates, size_t survivor_count,
    const uint64_t *global_x_offsets, uint64_t *survivor_offsets) {
  const size_t survivor =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (survivor < survivor_count) {
    const uint32_t candidate = survivor_candidates[survivor];
    survivor_offsets[survivor] =
        global_x_offsets[static_cast<size_t>(candidate) + 1] -
        global_x_offsets[candidate];
  } else if (survivor == survivor_count) {
    survivor_offsets[survivor] = 0;
  }
}

struct RunBuffers {
  uint32_t *candidate_profiles = nullptr;
  uint32_t *candidate_sequences = nullptr;
  float *filter_scores = nullptr;
  uint32_t *f3_threshold_bits = nullptr;
  ForwardLengthTransitions *length_transitions = nullptr;
  uint64_t *dp_offsets = nullptr;
  uint64_t *x_offsets = nullptr;
  float *dp = nullptr;
  float *xmx = nullptr;
  ForwardKernelResult *results = nullptr;
  uint32_t *survivor_candidates = nullptr;
  uint64_t *survivor_offsets = nullptr;
  float *gathered = nullptr;
  cudaEvent_t begin_event = nullptr;
  cudaEvent_t end_event = nullptr;
};

}  // namespace

struct plan7_forward_database {
  uint64_t generation_id;
  uint64_t provenance_salt;
  int device_ordinal;
  int alphabet_size;
  bool sealed_source;
  std::vector<ForwardProfile> host_profiles;
  std::vector<plan7_forward_device_profile> host_device_profiles;
  std::vector<uintptr_t> source_profile_pointers;
  std::vector<float> host_emissions;
  std::vector<float> host_transitions;
  plan7_forward_device_profile *device_profiles;
  float *device_emissions;
  float *device_transitions;
  uint64_t device_bytes;
  float pack_milliseconds;
  float upload_milliseconds;
};

struct plan7_forward_workspace {
  int device_ordinal;
  RunBuffers buffers;
  size_t candidate_profiles_capacity;
  size_t candidate_sequences_capacity;
  size_t filter_scores_capacity;
  size_t f3_threshold_bits_capacity;
  size_t length_transitions_capacity;
  size_t dp_offsets_capacity;
  size_t x_offsets_capacity;
  size_t dp_capacity;
  size_t xmx_capacity;
  size_t results_capacity;
  size_t survivor_candidates_capacity;
  size_t survivor_offsets_capacity;
  size_t gathered_capacity;
  std::vector<uint32_t> host_candidate_profiles;
  std::vector<uint32_t> host_candidate_sequences;
  std::vector<uint32_t> host_f3_threshold_bits;
  std::vector<ForwardLengthTransitions> host_length_transitions;
  std::vector<uint64_t> host_dp_offsets;
  std::vector<uint64_t> host_x_offsets;
  std::vector<size_t> tile_boundaries;
  std::vector<ForwardKernelResult> host_kernel_results;
  std::vector<uint64_t> host_f3_overrides;
  uint64_t growth_count;
  uint64_t event_create_count;
  uint64_t run_count;
};

struct ResidentForwardSpecials {
  float *pointer = nullptr;
  int device_ordinal = -1;

  ~ResidentForwardSpecials() {
    if (pointer == nullptr || device_ordinal < 0) return;
    int original_device = -1;
    const cudaError_t get_status = cudaGetDevice(&original_device);
    bool restore_device = false;
    if (get_status == cudaSuccess && original_device != device_ordinal) {
      if (cudaSetDevice(device_ordinal) != cudaSuccess) return;
      restore_device = true;
    } else if (get_status != cudaSuccess &&
               cudaSetDevice(device_ordinal) != cudaSuccess) {
      return;
    }
    cudaFree(pointer);
    if (restore_device) cudaSetDevice(original_device);
  }

  ResidentForwardSpecials() = default;
  ResidentForwardSpecials(const ResidentForwardSpecials &) = delete;
  ResidentForwardSpecials &operator=(const ResidentForwardSpecials &) = delete;
};

struct plan7_forward_output {
  std::vector<plan7_forward_result> results;
  std::vector<uint16_t> reason_facts;
  std::vector<uint64_t> special_offsets;
  std::vector<float> specials;
  plan7_forward_statistics statistics;
  plan7_forward_f3_device_statistics f3_device_statistics;
  plan7_forward_provenance provenance;
  ResidentForwardSpecials resident_specials;
  plan7_forward_residency_statistics residency_statistics{};
  plan7_forward_input_residency_statistics input_residency_statistics{};
  plan7_forward_subwarp_statistics subwarp_statistics{};
  float upload_milliseconds;
  float total_milliseconds;
  bool contract_fallback = false;
};

namespace {

uint64_t provenance_integrity_tag(
    const plan7_forward_database *database,
    const plan7_forward_provenance &provenance) {
  uint64_t hash = hash_u64(database->provenance_salt,
                           UINT64_C(0x46574450));
  hash = hash_u64(hash, provenance.database_generation);
  hash = hash_u64(hash, provenance.batch_generation);
  hash = hash_u64(hash, provenance.row_hash);
  hash = hash_u64(hash, provenance.special_hash);
  hash = hash_u64(hash, provenance.continuation_hash);
  hash = hash_u64(hash, provenance.pass_count);
  hash = hash_u64(hash, provenance.special_count);
  return hash_u64(hash, provenance.generation_f3_bits);
}

bool seal_forward_provenance(
    plan7_forward_output *output,
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch_view &batch,
    const std::vector<uint32_t> *candidate_profiles,
    const float *filter_scores, uint64_t generation_f3_bits) {
  if (output == nullptr || database == nullptr) return false;
  plan7_forward_provenance sealed{};
  sealed.database_generation = database->generation_id;
  sealed.batch_generation = batch.generation_id;
  uint64_t row_hash = hash_u64(kHashOffset, UINT64_C(0x524f5753));
  uint64_t special_hash = hash_u64(kHashOffset, UINT64_C(0x584d5821));
  uint64_t continuation_hash =
      hash_u64(kHashOffset, UINT64_C(0x434f4e54));
  if (output->special_offsets.size() != output->results.size() + 1)
    return false;
  for (size_t candidate = 0; candidate < output->results.size(); ++candidate) {
    if (output->results[candidate].action != PLAN7_FORWARD_DEFINITE_PASS)
      continue;
    if (candidate_profiles == nullptr ||
        candidate >= candidate_profiles->size() ||
        output->special_offsets[candidate] >
            output->special_offsets[candidate + 1] ||
        output->special_offsets[candidate + 1] > output->specials.size())
      return false;
    row_hash = hash_u32(row_hash, (*candidate_profiles)[candidate]);
    row_hash = hash_u32(row_hash,
                        output->results[candidate].sequence_index);
    if (filter_scores == nullptr) return false;
    FloatBits fwdsc_bits{};
    FloatBits filtersc_bits{};
    fwdsc_bits.value = output->results[candidate].fwdsc;
    filtersc_bits.value = filter_scores[candidate];
    continuation_hash = hash_u32(continuation_hash, fwdsc_bits.bits);
    continuation_hash = hash_u32(continuation_hash, filtersc_bits.bits);
    continuation_hash = hash_u32(
        continuation_hash,
        static_cast<uint32_t>(output->results[candidate].status) |
            (static_cast<uint32_t>(output->results[candidate].action) << 8) |
            (static_cast<uint32_t>(output->results[candidate].reserved) << 16));
    const uint64_t begin = output->special_offsets[candidate];
    const uint64_t end = output->special_offsets[candidate + 1];
    special_hash = hash_u64(special_hash, end - begin);
    for (uint64_t cell = begin; cell < end; ++cell) {
      FloatBits bits{};
      bits.value = output->specials[static_cast<size_t>(cell)];
      special_hash = hash_u32(special_hash, bits.bits);
    }
    ++sealed.pass_count;
  }
  sealed.special_count = output->specials.size();
  sealed.generation_f3_bits = generation_f3_bits;
  sealed.row_hash = hash_u64(row_hash, sealed.pass_count);
  sealed.special_hash = hash_u64(special_hash, sealed.special_count);
  continuation_hash = hash_u64(continuation_hash, sealed.pass_count);
  sealed.continuation_hash = hash_u64(
      continuation_hash, sealed.generation_f3_bits);
  sealed.integrity_tag = provenance_integrity_tag(database, sealed);
  output->provenance = sealed;
  return true;
}

template <typename T>
int grow_forward_workspace_buffer(T **buffer, size_t *capacity,
                                  size_t required_bytes,
                                  uint64_t *growth_count, const char *name,
                                  char *error, size_t error_size) {
  if (required_bytes <= *capacity) return 0;
  T *replacement = nullptr;
  cudaError_t status = cudaMalloc(&replacement, required_bytes);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, name, status);
    return -1;
  }
  status = cudaFree(*buffer);
  if (status != cudaSuccess) {
    cudaFree(replacement);
    set_cuda_error(error, error_size, "cudaFree(Forward workspace)", status);
    return -1;
  }
  *buffer = replacement;
  *capacity = required_bytes;
  ++*growth_count;
  return 0;
}

uint64_t forward_workspace_device_bytes(
    const plan7_forward_workspace *workspace) {
  if (workspace == nullptr) return 0;
  const size_t capacities[] = {
      workspace->candidate_profiles_capacity,
      workspace->candidate_sequences_capacity,
      workspace->filter_scores_capacity,
      workspace->f3_threshold_bits_capacity,
      workspace->length_transitions_capacity,
      workspace->dp_offsets_capacity,
      workspace->x_offsets_capacity,
      workspace->dp_capacity,
      workspace->xmx_capacity,
      workspace->results_capacity,
      workspace->survivor_candidates_capacity,
      workspace->survivor_offsets_capacity,
      workspace->gathered_capacity};
  uint64_t total = 0;
  for (const size_t capacity : capacities) {
    if (capacity > UINT64_MAX - total) return UINT64_MAX;
    total += static_cast<uint64_t>(capacity);
  }
  return total;
}

int destroy_forward_workspace_device(plan7_forward_workspace *workspace,
                                     char *error, size_t error_size) {
  if (workspace == nullptr) return 0;
  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool restore_device = false;
  bool device_ready = true;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess && original_device != workspace->device_ordinal) {
    status = cudaSetDevice(workspace->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice(workspace->device_ordinal);
    if (status != cudaSuccess) {
      first_error = status;
      device_ready = false;
    }
  }
  RunBuffers &buffers = workspace->buffers;
#define CUDA_DESTROY_FORWARD(pointer)                                          \
  do {                                                                         \
    if (device_ready) {                                                        \
      status = cudaFree(pointer);                                              \
      if (status != cudaSuccess && first_error == cudaSuccess)                \
        first_error = status;                                                  \
    }                                                                          \
  } while (0)
#define CUDA_DESTROY_FORWARD_EVENT(event)                                      \
  do {                                                                         \
    if (device_ready && event != nullptr) {                                    \
      status = cudaEventDestroy(event);                                        \
      if (status != cudaSuccess && first_error == cudaSuccess)                \
        first_error = status;                                                  \
    }                                                                          \
  } while (0)
  CUDA_DESTROY_FORWARD_EVENT(buffers.end_event);
  CUDA_DESTROY_FORWARD_EVENT(buffers.begin_event);
  CUDA_DESTROY_FORWARD(buffers.gathered);
  CUDA_DESTROY_FORWARD(buffers.survivor_offsets);
  CUDA_DESTROY_FORWARD(buffers.survivor_candidates);
  CUDA_DESTROY_FORWARD(buffers.results);
  CUDA_DESTROY_FORWARD(buffers.xmx);
  CUDA_DESTROY_FORWARD(buffers.dp);
  CUDA_DESTROY_FORWARD(buffers.x_offsets);
  CUDA_DESTROY_FORWARD(buffers.dp_offsets);
  CUDA_DESTROY_FORWARD(buffers.length_transitions);
  CUDA_DESTROY_FORWARD(buffers.f3_threshold_bits);
  CUDA_DESTROY_FORWARD(buffers.filter_scores);
  CUDA_DESTROY_FORWARD(buffers.candidate_sequences);
  CUDA_DESTROY_FORWARD(buffers.candidate_profiles);
#undef CUDA_DESTROY_FORWARD_EVENT
#undef CUDA_DESTROY_FORWARD
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (status != cudaSuccess && first_error == cudaSuccess)
      first_error = status;
  }
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy Forward workspace", first_error);
    return -1;
  }
  return 0;
}

bool live_profile_matches_snapshot(const plan7_forward_database *database,
                                   size_t profile_index) {
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      database->source_profile_pointers[profile_index]);
  const ForwardProfile &descriptor = database->host_profiles[profile_index];
  if (!valid_profile_storage(source) || !valid_forward_specials(source) ||
      !valid_probability_rows(source) ||
      source->abc->Kp != database->alphabet_size ||
      reinterpret_cast<uintptr_t>(source->abc) !=
          descriptor.alphabet_pointer ||
      source->M != static_cast<int>(descriptor.model_length) ||
      source->mode != descriptor.mode || source->nj != descriptor.nj ||
      source->xf[p7O_E][p7O_MOVE] != descriptor.e_move ||
      source->xf[p7O_E][p7O_LOOP] != descriptor.e_loop ||
      source->evparam[p7_FTAU] != descriptor.f_tau ||
      source->evparam[p7_FLAMBDA] != descriptor.f_lambda)
    return false;

  const size_t q_count = descriptor.q;
  const size_t row_bytes = q_count * kSubwarp * sizeof(float);
  for (int residue = 0; residue < source->abc->Kp; ++residue) {
    const float *snapshot = database->host_emissions.data() +
        descriptor.emission_offset +
        static_cast<uint64_t>(residue) * q_count * kSubwarp;
    if (std::memcmp(source->rfv[residue], snapshot, row_bytes) != 0)
      return false;
  }
  for (size_t q = 0; q < q_count; ++q) {
    for (int transition = p7O_BM; transition <= p7O_II;
         ++transition) {
      const auto *source_values = reinterpret_cast<const float *>(
          source->tfv + q * 7 + transition);
      const float *snapshot = database->host_transitions.data() +
          descriptor.transition_offset +
          (q * p7O_NTRANS + transition) * kSubwarp;
      if (std::memcmp(source_values, snapshot,
                      kSubwarp * sizeof(float)) != 0)
        return false;
    }
    const auto *source_dd = reinterpret_cast<const float *>(
        source->tfv + 7 * q_count + q);
    const float *snapshot_dd = database->host_transitions.data() +
        descriptor.transition_offset +
        (q * p7O_NTRANS + p7O_DD) * kSubwarp;
    if (std::memcmp(source_dd, snapshot_dd,
                    kSubwarp * sizeof(float)) != 0)
      return false;
  }
  return true;
}

void free_database_device(plan7_forward_database *database) {
  if (database == nullptr) return;
  cudaFree(database->device_transitions);
  cudaFree(database->device_emissions);
  cudaFree(database->device_profiles);
  database->device_transitions = nullptr;
  database->device_emissions = nullptr;
  database->device_profiles = nullptr;
}

}  // namespace

extern "C" int plan7_forward_database_create(
    const uintptr_t *profile_pointers, size_t profile_count,
    plan7_forward_database **database, char *error, size_t error_size) {
  if (database == nullptr || *database != nullptr ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid Forward database arguments");
    return -1;
  }
  if (profile_count > UINT32_MAX) {
    set_error(error, error_size, "Forward profile count exceeds uint32");
    return -1;
  }
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }

  std::unique_ptr<plan7_forward_database> created(
      new (std::nothrow) plan7_forward_database{});
  if (!created) {
    set_error(error, error_size, "Forward database allocation failed");
    return -1;
  }
  created->device_ordinal = current_device;
  created->generation_id = allocate_forward_database_generation();
  created->provenance_salt = hash_u64(
      hash_u64(kHashOffset, created->generation_id),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(created.get())) ^
          static_cast<uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count()));
  created->alphabet_size = -1;
  created->sealed_source = false;
  created->device_profiles = nullptr;
  created->device_emissions = nullptr;
  created->device_transitions = nullptr;
  created->device_bytes = 0;
  created->pack_milliseconds = 0.0f;
  created->upload_milliseconds = 0.0f;

  const auto pack_begin = std::chrono::steady_clock::now();
  try {
    created->host_profiles.resize(profile_count);
    created->host_device_profiles.resize(profile_count);
    if (profile_count != 0)
      created->source_profile_pointers.assign(
          profile_pointers, profile_pointers + profile_count);
  } catch (...) {
    set_error(error, error_size, "Forward descriptor allocation failed");
    return -1;
  }

  uint64_t emission_total = 0;
  uint64_t transition_total = 0;
  for (size_t profile_index = 0; profile_index < profile_count;
       ++profile_index) {
    const auto *source = reinterpret_cast<const P7_OPROFILE *>(
        profile_pointers[profile_index]);
    if (!valid_profile_storage(source) || !valid_forward_specials(source) ||
        !valid_probability_rows(source)) {
      set_error(error, error_size, "invalid optimized Forward profile");
      return -1;
    }
    if (created->alphabet_size < 0)
      created->alphabet_size = source->abc->Kp;
    else if (created->alphabet_size != source->abc->Kp) {
      set_error(error, error_size, "Forward profile alphabets differ");
      return -1;
    }
    const uint64_t q_count = static_cast<uint64_t>(p7O_NQF(source->M));
    uint64_t emission_count;
    uint64_t transition_count;
    if (!checked_multiply(static_cast<uint64_t>(source->abc->Kp),
                          q_count * kSubwarp, &emission_count) ||
        !checked_multiply(q_count,
                          p7O_NTRANS * kSubwarp, &transition_count) ||
        !checked_add(emission_total, emission_count, &emission_total) ||
        !checked_add(transition_total, transition_count,
                     &transition_total) ||
        emission_total > SIZE_MAX || transition_total > SIZE_MAX) {
      set_error(error, error_size, "Forward packed profile size overflow");
      return -1;
    }
    created->host_profiles[profile_index] = {
      emission_total - emission_count,
      transition_total - transition_count,
      static_cast<uint32_t>(q_count),
      static_cast<uint32_t>(source->M),
      source->xf[p7O_E][p7O_MOVE],
      source->xf[p7O_E][p7O_LOOP],
      source->evparam[p7_FTAU],
      source->evparam[p7_FLAMBDA],
      source->nj,
      source->mode,
      reinterpret_cast<uintptr_t>(source->abc)
    };
    created->host_device_profiles[profile_index] = {
      emission_total - emission_count,
      transition_total - transition_count,
      static_cast<uint32_t>(q_count),
      static_cast<uint32_t>(source->M),
      source->xf[p7O_E][p7O_MOVE],
      source->xf[p7O_E][p7O_LOOP]
    };
  }
  try {
    created->host_emissions.resize(static_cast<size_t>(emission_total));
    created->host_transitions.resize(static_cast<size_t>(transition_total));
  } catch (...) {
    set_error(error, error_size, "Forward packed data allocation failed");
    return -1;
  }

  std::atomic<size_t> next_profile{0};
  std::atomic<unsigned> completed_workers{0};
  const unsigned thread_count = static_cast<unsigned>(std::min<size_t>(
      16, std::max<size_t>(1, profile_count)));
  auto worker = [&]() {
    while (true) {
      const size_t profile_index =
          next_profile.fetch_add(1, std::memory_order_relaxed);
      if (profile_index >= profile_count) break;
      const auto *source = reinterpret_cast<const P7_OPROFILE *>(
          profile_pointers[profile_index]);
      const ForwardProfile descriptor =
          created->host_profiles[profile_index];
      const int q_count = static_cast<int>(descriptor.q);
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const auto *source_values =
            reinterpret_cast<const float *>(source->rfv[residue]);
        float *destination = created->host_emissions.data() +
            descriptor.emission_offset +
            static_cast<uint64_t>(residue) * q_count * kSubwarp;
        std::memcpy(destination, source_values,
                    static_cast<size_t>(q_count) * kSubwarp * sizeof(float));
      }
      for (int q = 0; q < q_count; ++q) {
        for (int transition = p7O_BM; transition <= p7O_II;
             ++transition) {
          const auto *source_values = reinterpret_cast<const float *>(
              source->tfv + q * 7 + transition);
          float *destination = created->host_transitions.data() +
              descriptor.transition_offset +
              (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
                  kSubwarp;
          std::memcpy(destination, source_values,
                      kSubwarp * sizeof(float));
        }
        const auto *source_dd = reinterpret_cast<const float *>(
            source->tfv + 7 * q_count + q);
        float *destination_dd = created->host_transitions.data() +
            descriptor.transition_offset +
            (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) * kSubwarp;
        std::memcpy(destination_dd, source_dd,
                    kSubwarp * sizeof(float));
      }
    }
    completed_workers.fetch_add(1, std::memory_order_release);
  };
  std::vector<std::thread> workers;
  bool worker_failure = false;
  try {
    workers.reserve(thread_count);
    for (unsigned thread = 0; thread < thread_count; ++thread)
      workers.emplace_back(worker);
  } catch (...) {
    worker_failure = true;
  }
  for (auto &thread : workers) {
    try {
      thread.join();
    } catch (...) {
      worker_failure = true;
      if (thread.joinable()) thread.detach();
    }
  }
  while (completed_workers.load(std::memory_order_acquire) < workers.size())
    std::this_thread::yield();
  if (worker_failure || workers.size() != thread_count) {
    set_error(error, error_size, "Forward profile worker launch failed");
    return -1;
  }
  created->pack_milliseconds = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - pack_begin).count();

  size_t profile_bytes;
  size_t emission_bytes;
  size_t transition_bytes;
  if (!checked_bytes(profile_count, sizeof(plan7_forward_device_profile),
                     &profile_bytes) ||
      !checked_bytes(created->host_emissions.size(), sizeof(float),
                     &emission_bytes) ||
      !checked_bytes(created->host_transitions.size(), sizeof(float),
                     &transition_bytes)) {
    set_error(error, error_size, "Forward device allocation size overflow");
    return -1;
  }
  const auto upload_begin = std::chrono::steady_clock::now();
#define CUDA_CREATE(call)                                                     \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      free_database_device(created.get());                                   \
      return -1;                                                              \
    }                                                                         \
  } while (0)
  if (profile_count != 0) {
    CUDA_CREATE(cudaMalloc(&created->device_profiles, profile_bytes));
    CUDA_CREATE(cudaMalloc(&created->device_emissions, emission_bytes));
    CUDA_CREATE(cudaMalloc(&created->device_transitions, transition_bytes));
    CUDA_CREATE(cudaMemcpy(created->device_profiles,
                           created->host_device_profiles.data(), profile_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CREATE(cudaMemcpy(created->device_emissions,
                           created->host_emissions.data(), emission_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CREATE(cudaMemcpy(created->device_transitions,
                           created->host_transitions.data(), transition_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CREATE(cudaDeviceSynchronize());
  }
#undef CUDA_CREATE
  created->device_bytes = static_cast<uint64_t>(profile_bytes) +
                          static_cast<uint64_t>(emission_bytes) +
                          static_cast<uint64_t>(transition_bytes);
  created->upload_milliseconds = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - upload_begin).count();
  *database = created.release();
  return 0;
}

extern "C" int plan7_forward_database_create_snapshot(
    int alphabet_size,
    const plan7_forward_snapshot_profile *profiles, size_t profile_count,
    const float *emissions, size_t emission_count,
    const float *transitions, size_t transition_count,
    const uintptr_t *identity_tokens,
    plan7_forward_database **database, char *error, size_t error_size) {
  if (database == nullptr || *database != nullptr || alphabet_size != 29 ||
      (profile_count != 0 &&
       (profiles == nullptr || identity_tokens == nullptr)) ||
      (emission_count != 0 && emissions == nullptr) ||
      (transition_count != 0 && transitions == nullptr)) {
    set_error(error, error_size, "invalid Forward snapshot arguments");
    return -1;
  }
  if (profile_count > UINT32_MAX) {
    set_error(error, error_size, "Forward profile count exceeds uint32");
    return -1;
  }
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }

  std::unique_ptr<plan7_forward_database> created(
      new (std::nothrow) plan7_forward_database{});
  if (!created) {
    set_error(error, error_size, "Forward database allocation failed");
    return -1;
  }
  created->device_ordinal = current_device;
  created->generation_id = allocate_forward_database_generation();
  created->provenance_salt = hash_u64(
      hash_u64(kHashOffset, created->generation_id),
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(created.get())) ^
          static_cast<uint64_t>(
              std::chrono::steady_clock::now().time_since_epoch().count()));
  created->alphabet_size = alphabet_size;
  created->sealed_source = true;
  created->device_profiles = nullptr;
  created->device_emissions = nullptr;
  created->device_transitions = nullptr;
  created->device_bytes = 0;
  created->pack_milliseconds = 0.0f;
  created->upload_milliseconds = 0.0f;

  const auto pack_begin = std::chrono::steady_clock::now();
  try {
    created->host_profiles.resize(profile_count);
    created->host_device_profiles.resize(profile_count);
    if (profile_count != 0)
      created->source_profile_pointers.assign(
          identity_tokens, identity_tokens + profile_count);
  } catch (...) {
    set_error(error, error_size, "Forward descriptor allocation failed");
    return -1;
  }

  uint64_t expected_emissions = 0;
  uint64_t expected_transitions = 0;
  for (size_t profile_index = 0; profile_index < profile_count;
       ++profile_index) {
    const plan7_forward_snapshot_profile &source = profiles[profile_index];
    if (source.model_length < 1 || source.model_length > kMaximumModelLength ||
        source.q != static_cast<uint32_t>(p7O_NQF(source.model_length)) ||
        source.emission_offset != expected_emissions ||
        source.transition_offset != expected_transitions ||
        (source.mode != p7_LOCAL && source.mode != p7_UNILOCAL) ||
        (source.nj != 0.0f && source.nj != 1.0f) ||
        (source.nj == 0.0f &&
         (source.e_move != 1.0f || source.e_loop != 0.0f)) ||
        (source.nj == 1.0f &&
         (source.e_move != 0.5f || source.e_loop != 0.5f))) {
      set_error(error, error_size, "invalid Forward snapshot descriptor");
      return -1;
    }
    uint64_t profile_emissions;
    uint64_t profile_transitions;
    if (!checked_multiply(alphabet_size,
                          static_cast<uint64_t>(source.q) * kSubwarp,
                          &profile_emissions) ||
        !checked_multiply(source.q, p7O_NTRANS * kSubwarp,
                          &profile_transitions) ||
        !checked_add(expected_emissions, profile_emissions,
                     &expected_emissions) ||
        !checked_add(expected_transitions, profile_transitions,
                     &expected_transitions)) {
      set_error(error, error_size, "Forward snapshot size overflow");
      return -1;
    }
    created->host_profiles[profile_index] = {
      source.emission_offset,
      source.transition_offset,
      source.q,
      source.model_length,
      source.e_move,
      source.e_loop,
      source.f_tau,
      source.f_lambda,
      source.nj,
      source.mode,
      0
    };
    created->host_device_profiles[profile_index] = {
      source.emission_offset,
      source.transition_offset,
      source.q,
      source.model_length,
      source.e_move,
      source.e_loop
    };
  }
  if (expected_emissions != emission_count ||
      expected_transitions != transition_count) {
    set_error(error, error_size, "Forward snapshot arrays have trailing data");
    return -1;
  }
  created->pack_milliseconds = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - pack_begin).count();

  size_t profile_bytes;
  size_t emission_bytes;
  size_t transition_bytes;
  if (!checked_bytes(profile_count, sizeof(plan7_forward_device_profile),
                     &profile_bytes) ||
      !checked_bytes(emission_count, sizeof(float), &emission_bytes) ||
      !checked_bytes(transition_count, sizeof(float), &transition_bytes)) {
    set_error(error, error_size, "Forward device allocation size overflow");
    return -1;
  }
  const auto upload_begin = std::chrono::steady_clock::now();
#define CUDA_STAGE(call)                                                       \
  do {                                                                         \
    status = (call);                                                           \
    if (status != cudaSuccess) {                                               \
      set_cuda_error(error, error_size, #call, status);                        \
      free_database_device(created.get());                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)
  if (profile_count != 0) {
    CUDA_STAGE(cudaMalloc(&created->device_profiles, profile_bytes));
    CUDA_STAGE(cudaMalloc(&created->device_emissions, emission_bytes));
    CUDA_STAGE(cudaMalloc(&created->device_transitions, transition_bytes));
    CUDA_STAGE(cudaMemcpy(created->device_profiles,
                          created->host_device_profiles.data(), profile_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_STAGE(cudaMemcpy(created->device_emissions, emissions, emission_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_STAGE(cudaMemcpy(created->device_transitions, transitions,
                          transition_bytes, cudaMemcpyHostToDevice));
    CUDA_STAGE(cudaDeviceSynchronize());
  }
#undef CUDA_STAGE
  created->device_bytes = static_cast<uint64_t>(profile_bytes) +
                          static_cast<uint64_t>(emission_bytes) +
                          static_cast<uint64_t>(transition_bytes);
  created->upload_milliseconds = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - upload_begin).count();
  *database = created.release();
  return 0;
}

extern "C" int plan7_forward_database_destroy(
    plan7_forward_database **database, char *error, size_t error_size) {
  if (database == nullptr) {
    set_error(error, error_size, "Forward database handle is null");
    return -1;
  }
  plan7_forward_database *value = *database;
  *database = nullptr;
  if (value == nullptr) return 0;

  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool restore_device = false;
  bool device_ready = true;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess && original_device != value->device_ordinal) {
    status = cudaSetDevice(value->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice(value->device_ordinal);
    if (status != cudaSuccess) {
      first_error = status;
      device_ready = false;
    }
  }
#define CUDA_DESTROY(pointer)                                                 \
  do {                                                                        \
    if (device_ready) {                                                       \
      status = cudaFree(pointer);                                             \
      if (status != cudaSuccess && first_error == cudaSuccess)                \
        first_error = status;                                                 \
    }                                                                         \
  } while (0)
  CUDA_DESTROY(value->device_transitions);
  CUDA_DESTROY(value->device_emissions);
  CUDA_DESTROY(value->device_profiles);
#undef CUDA_DESTROY
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (status != cudaSuccess && first_error == cudaSuccess)
      first_error = status;
  }
  delete value;
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy Forward database", first_error);
    return -1;
  }
  return 0;
}

extern "C" size_t plan7_forward_database_profile_count(
    const plan7_forward_database *database) {
  return database == nullptr ? 0 : database->host_profiles.size();
}

extern "C" uint64_t plan7_forward_database_device_bytes(
    const plan7_forward_database *database) {
  return database == nullptr ? 0 : database->device_bytes;
}

extern "C" float plan7_forward_database_pack_milliseconds(
    const plan7_forward_database *database) {
  return database == nullptr ? 0.0f : database->pack_milliseconds;
}

extern "C" float plan7_forward_database_upload_milliseconds(
    const plan7_forward_database *database) {
  return database == nullptr ? 0.0f : database->upload_milliseconds;
}

extern "C" int plan7_forward_database_get_device_view(
    const plan7_forward_database *database,
    plan7_forward_device_view *view, char *error, size_t error_size) {
  if (database == nullptr || view == nullptr) {
    set_error(error, error_size, "invalid Forward device view request");
    return -1;
  }
  std::memset(view, 0, sizeof(*view));
  if (!database->host_profiles.empty() &&
      (database->device_profiles == nullptr ||
       database->device_emissions == nullptr ||
       database->device_transitions == nullptr)) {
    set_error(error, error_size, "Forward device storage is null");
    return -1;
  }
  view->generation_id = database->generation_id;
  view->device_ordinal = database->device_ordinal;
  view->alphabet_size = database->alphabet_size;
  view->profile_count = database->host_profiles.size();
  view->profiles = database->device_profiles;
  view->emissions = database->device_emissions;
  view->transitions = database->device_transitions;
  return 0;
}

extern "C" int plan7_forward_database_get_profile_snapshot(
    const plan7_forward_database *database, size_t profile_index,
    plan7_forward_snapshot_profile *profile,
    char *error, size_t error_size) {
  if (database == nullptr || profile == nullptr ||
      profile_index >= database->host_profiles.size()) {
    set_error(error, error_size, "invalid Forward profile snapshot request");
    return -1;
  }
  const ForwardProfile &source = database->host_profiles[profile_index];
  *profile = {source.emission_offset,
              source.transition_offset,
              source.q,
              source.model_length,
              source.e_move,
              source.e_loop,
              source.f_tau,
              source.f_lambda,
              source.nj,
              source.mode};
  return 0;
}

extern "C" int plan7_forward_workspace_create(
    plan7_forward_workspace **workspace, char *error, size_t error_size) {
  if (workspace == nullptr || *workspace != nullptr) {
    set_error(error, error_size, "invalid Forward workspace output");
    return -1;
  }
  int current_device = -1;
  const cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  auto *created = new (std::nothrow) plan7_forward_workspace{};
  if (created == nullptr) {
    set_error(error, error_size, "Forward workspace allocation failed");
    return -1;
  }
  created->device_ordinal = current_device;
  *workspace = created;
  return 0;
}

extern "C" int plan7_forward_workspace_destroy(
    plan7_forward_workspace **workspace, char *error, size_t error_size) {
  if (workspace == nullptr) {
    set_error(error, error_size, "Forward workspace handle is null");
    return -1;
  }
  plan7_forward_workspace *value = *workspace;
  *workspace = nullptr;
  if (value == nullptr) return 0;
  const int status = destroy_forward_workspace_device(value, error, error_size);
  delete value;
  return status;
}

extern "C" int plan7_forward_workspace_get_statistics(
    const plan7_forward_workspace *workspace,
    plan7_forward_workspace_statistics *statistics,
    char *error, size_t error_size) {
  if (workspace == nullptr || statistics == nullptr) {
    set_error(error, error_size, "invalid Forward workspace statistics");
    return -1;
  }
  *statistics = {};
  statistics->device_bytes = forward_workspace_device_bytes(workspace);
  statistics->dp_capacity_bytes =
      static_cast<uint64_t>(workspace->dp_capacity);
  statistics->xmx_capacity_bytes =
      static_cast<uint64_t>(workspace->xmx_capacity);
  statistics->gather_capacity_bytes =
      static_cast<uint64_t>(workspace->gathered_capacity);
  statistics->growth_count = workspace->growth_count;
  statistics->event_create_count = workspace->event_create_count;
  statistics->run_count = workspace->run_count;
  const size_t capacities[PLAN7_FORWARD_CAPACITY_COUNT] = {
      workspace->candidate_profiles_capacity,
      workspace->candidate_sequences_capacity,
      workspace->filter_scores_capacity,
      workspace->f3_threshold_bits_capacity,
      workspace->length_transitions_capacity,
      workspace->dp_offsets_capacity,
      workspace->x_offsets_capacity,
      workspace->dp_capacity,
      workspace->xmx_capacity,
      workspace->results_capacity,
      workspace->survivor_candidates_capacity,
      workspace->survivor_offsets_capacity,
      workspace->gathered_capacity};
  for (size_t i = 0; i < PLAN7_FORWARD_CAPACITY_COUNT; ++i)
    statistics->capacity_bytes[i] = static_cast<uint64_t>(capacities[i]);
  return 0;
}

static int forward_run_with_workspace_impl(
    plan7_forward_workspace *workspace,
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, bool collect_reason_facts,
    bool retain_device_specials, bool audit_f3,
    int requested_candidates_per_warp,
    const plan7_postfilter_f2_resident_view *postfilter_view,
    char *error, size_t error_size) {
  const auto call_begin = std::chrono::steady_clock::now();
  int candidates_per_warp = kDefaultCandidatesPerWarp;
  if (output == nullptr || *output != nullptr || workspace == nullptr ||
      database == nullptr || batch == nullptr ||
      (profile_count != 0 &&
       (candidate_offsets == nullptr ||
        (!database->sealed_source && source_profile_pointers == nullptr))) ||
      (candidate_count != 0 &&
       (candidate_indices == nullptr || filter_scores == nullptr)) ||
      database->host_profiles.size() != profile_count) {
    set_error(error, error_size, "invalid Forward run arguments");
    return -1;
  }
  if (requested_candidates_per_warp != 0 &&
      requested_candidates_per_warp != 1 &&
      requested_candidates_per_warp != 2 &&
      requested_candidates_per_warp != 4 &&
      requested_candidates_per_warp != 8) {
    set_error(error, error_size,
              "Forward candidates per warp must be 1, 2, 4, or 8");
    return -1;
  }
  if (profile_count == 0 && candidate_count != 0) {
    set_error(error, error_size, "Forward candidates have no profiles");
    return -1;
  }
  if (candidate_count > UINT32_MAX) {
    set_error(error, error_size, "Forward candidate count exceeds uint32");
    return -1;
  }

  /* Snapshot every raw caller array once. Classification and continuation
   * sealing must consume the same filter bits, and validated offsets/indexes
   * must not be mutable while the GIL is released. */
  std::vector<uintptr_t> source_pointer_snapshot;
  std::vector<uint64_t> candidate_offset_snapshot;
  std::vector<uint32_t> candidate_index_snapshot;
  std::vector<float> filter_score_snapshot;
  try {
    if (source_profile_pointers != nullptr && profile_count != 0)
      source_pointer_snapshot.assign(
          source_profile_pointers, source_profile_pointers + profile_count);
    if (profile_count != 0)
      candidate_offset_snapshot.assign(
          candidate_offsets, candidate_offsets + profile_count + 1);
    if (candidate_count != 0) {
      candidate_index_snapshot.assign(
          candidate_indices, candidate_indices + candidate_count);
      filter_score_snapshot.assign(
          filter_scores, filter_scores + candidate_count);
    }
  } catch (...) {
    set_error(error, error_size, "Forward immutable input snapshot failed");
    return -1;
  }
  source_profile_pointers = source_pointer_snapshot.empty()
      ? nullptr : source_pointer_snapshot.data();
  candidate_offsets = candidate_offset_snapshot.empty()
      ? nullptr : candidate_offset_snapshot.data();
  candidate_indices = candidate_index_snapshot.empty()
      ? nullptr : candidate_index_snapshot.data();
  filter_scores = filter_score_snapshot.empty()
      ? nullptr : filter_score_snapshot.data();
  if (profile_count != 0 &&
      (candidate_offsets[0] != 0 ||
       candidate_offsets[profile_count] != candidate_count)) {
    set_error(error, error_size, "invalid Forward candidate offsets");
    return -1;
  }

  plan7_ssv_sequence_batch_view batch_view{};
  if (plan7_ssv_sequence_batch_get_view(
          batch, &batch_view, error, error_size) != 0)
    return -1;

  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != database->device_ordinal ||
      current_device != batch_view.device_ordinal) {
    set_error(error, error_size,
              "Forward inputs belong to a different CUDA device");
    return -1;
  }
  if (current_device != workspace->device_ordinal) {
    set_error(error, error_size,
              "Forward workspace belongs to a different CUDA device");
    return -1;
  }
  const bool resident_f2 = postfilter_view != nullptr;
  if (resident_f2) {
    if (plan7_postfilter_f2_resident_view_validate(
            postfilter_view, error, error_size) != 0)
      return -1;
    if (!postfilter_view->supported ||
        postfilter_view->batch_generation != batch_view.generation_id ||
        postfilter_view->device_ordinal != current_device ||
        postfilter_view->profile_count != profile_count ||
        postfilter_view->selected_count != candidate_count ||
        (candidate_count != 0 &&
         (postfilter_view->host_selected_sources == nullptr ||
          postfilter_view->host_candidates == nullptr ||
          postfilter_view->host_results == nullptr ||
          postfilter_view->device_candidates == nullptr ||
          postfilter_view->device_results == nullptr ||
          postfilter_view->device_selected_sources == nullptr))) {
      set_error(error, error_size, "resident F2 view differs from Forward input");
      return -1;
    }
    size_t expected_profile = 0;
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
      while (expected_profile + 1 < profile_count &&
             candidate >= candidate_offsets[expected_profile + 1])
        ++expected_profile;
      const uint32_t source = postfilter_view->host_selected_sources[candidate];
      if (source >= postfilter_view->source_count) {
        set_error(error, error_size,
                  "resident F2 selected source is out of range");
        return -1;
      }
      const plan7_bias_candidate mapping =
          postfilter_view->host_candidates[source];
      FloatBits resident_filter{};
      FloatBits host_filter{};
      resident_filter.value = postfilter_view->host_results[source].filtersc;
      host_filter.value = filter_scores[candidate];
      if (mapping.profile_index != expected_profile ||
          mapping.sequence_index != candidate_indices[candidate] ||
          postfilter_view->host_results[source].sequence_index !=
              candidate_indices[candidate] ||
          resident_filter.bits != host_filter.bits) {
        set_error(error, error_size,
                  "resident F2 selection differs from host provenance");
        return -1;
      }
    }
  }
  if ((profile_count != 0 &&
       (!device_allocation_on(database->device_profiles, current_device) ||
        !device_allocation_on(database->device_emissions, current_device) ||
        !device_allocation_on(database->device_transitions,
                              current_device))) ||
      (batch_view.sequence_count != 0 &&
       (!device_allocation_on(batch_view.device_residues, current_device) ||
        !device_allocation_on(batch_view.device_offsets, current_device)))) {
    set_error(error, error_size, "Forward input device pointer is invalid");
    return -1;
  }
  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (candidate_offsets[profile] > candidate_offsets[profile + 1]) {
      set_error(error, error_size, "invalid Forward candidate offsets");
      return -1;
    }
    if ((source_profile_pointers != nullptr &&
         source_profile_pointers[profile] !=
             database->source_profile_pointers[profile]) ||
        (!database->sealed_source &&
         (source_profile_pointers == nullptr ||
          !live_profile_matches_snapshot(database, profile)))) {
      set_error(error, error_size,
                "Forward database does not match the source profile row");
      return -1;
    }
  }

  std::unique_ptr<plan7_forward_output> created(
      new (std::nothrow) plan7_forward_output{});
  if (!created) {
    set_error(error, error_size, "Forward output allocation failed");
    return -1;
  }
  try {
    created->results.resize(candidate_count);
    created->special_offsets.assign(candidate_count + 1, 0);
    if (collect_reason_facts)
      created->reason_facts.assign(candidate_count, 0);
  } catch (...) {
    set_error(error, error_size, "Forward output allocation failed");
    return -1;
  }
  created->statistics = {};
  created->f3_device_statistics = {};
  created->input_residency_statistics = {};
  if (resident_f2) {
    created->input_residency_statistics.resident_f2_call_count = 1;
    created->input_residency_statistics.resident_f2_candidate_count =
        candidate_count;
  }
  created->subwarp_statistics.policy_version = kSubwarpPolicyVersion;
  created->subwarp_statistics.requested_candidates_per_warp =
      static_cast<uint32_t>(requested_candidates_per_warp);
  created->subwarp_statistics.candidates_per_warp =
      kDefaultCandidatesPerWarp;
  DoubleBits generation_f3{};
  generation_f3.value = f3;
  const uint64_t output_byte_limit =
      std::min(gathered_byte_budget, kGatheredOutputByteLimit) &
      ~static_cast<uint64_t>(sizeof(float) - 1);
  created->statistics.generation_f3_bits = generation_f3.bits;
  created->statistics.candidate_count = candidate_count;
  created->statistics.output_byte_limit = output_byte_limit;

  for (size_t profile = 0; profile < profile_count; ++profile) {
    for (uint64_t candidate64 = candidate_offsets[profile];
         candidate64 < candidate_offsets[profile + 1]; ++candidate64) {
      const size_t candidate = static_cast<size_t>(candidate64);
      const uint32_t sequence = candidate_indices[candidate];
      if (sequence >= batch_view.sequence_count) {
        set_error(error, error_size,
                  "Forward candidate sequence index is out of range");
        return -1;
      }
      created->results[candidate] = {
        sequence, NAN, PLAN7_FORWARD_ENORESULT,
        PLAN7_FORWARD_CPU_REQUIRED, 0
      };
    }
  }
  if (candidate_count == 0) {
    if (!seal_forward_provenance(created.get(), database, batch_view, nullptr,
                                 filter_scores, generation_f3.bits)) {
      set_error(error, error_size, "Forward provenance sealing failed");
      return -1;
    }
    created->total_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - call_begin).count();
    *output = created.release();
    return 0;
  }
  if (!std::isfinite(f3) || f3 < 0.0 || batch_view.alphabet_size != 29 ||
      !batch_view.host_float_environment_valid ||
      plan7_bias_environment_attested(nullptr, 0) != 1) {
    created->contract_fallback = true;
    if (!seal_forward_provenance(created.get(), database, batch_view, nullptr,
                                 filter_scores, generation_f3.bits)) {
      set_error(error, error_size, "Forward provenance sealing failed");
      return -1;
    }
    created->total_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - call_begin).count();
    *output = created.release();
    return 0;
  }

  ++workspace->run_count;
  std::vector<uint32_t> &host_candidate_profiles =
      workspace->host_candidate_profiles;
  std::vector<uint32_t> &host_candidate_sequences =
      workspace->host_candidate_sequences;
  std::vector<uint32_t> &host_f3_threshold_bits =
      workspace->host_f3_threshold_bits;
  std::vector<ForwardLengthTransitions> &host_length_transitions =
      workspace->host_length_transitions;
  std::vector<uint64_t> &host_dp_offsets = workspace->host_dp_offsets;
  std::vector<uint64_t> &host_x_offsets = workspace->host_x_offsets;
  std::vector<size_t> &tile_boundaries = workspace->tile_boundaries;
  uint64_t maximum_dp_cells = 0;
  uint64_t maximum_x_cells = 0;
  uint64_t model_length_sum = 0;
  uint64_t target_length_sum = 0;
  uint64_t maximum_model_length = 0;
  uint64_t maximum_target_length = 0;
  uint64_t maximum_candidate_work = 0;
  size_t maximum_tile_count = 0;
  try {
    host_candidate_profiles.resize(candidate_count);
    host_candidate_sequences.resize(candidate_count);
    host_f3_threshold_bits.resize(profile_count);
    host_length_transitions.resize(candidate_count);
    host_dp_offsets.assign(candidate_count + 1, 0);
    host_x_offsets.assign(candidate_count + 1, 0);
    tile_boundaries.clear();
    tile_boundaries.push_back(0);
  } catch (...) {
    set_error(error, error_size, "Forward host workspace allocation failed");
    return -1;
  }

  for (size_t profile = 0; profile < profile_count; ++profile) {
    plan7_f3_threshold threshold{};
    if (plan7_forward_compile_f3_threshold(
            database->host_profiles[profile].f_tau,
            database->host_profiles[profile].f_lambda,
            f3, &threshold) != 0) {
      set_error(error, error_size, "Forward F3 threshold compilation failed");
      return -1;
    }
    if (threshold.supported) {
      host_f3_threshold_bits[profile] = threshold.threshold_bits;
      ++created->f3_device_statistics.compiled_profile_count;
    } else {
      host_f3_threshold_bits[profile] = UINT32_C(0x7fc00000);
      ++created->f3_device_statistics.unsupported_profile_count;
    }
  }

  uint64_t work_cells = 0;
  for (size_t profile = 0; profile < profile_count; ++profile) {
    for (uint64_t candidate64 = candidate_offsets[profile];
         candidate64 < candidate_offsets[profile + 1]; ++candidate64) {
      const size_t candidate = static_cast<size_t>(candidate64);
      const uint32_t sequence = candidate_indices[candidate];
      const ForwardProfile &descriptor = database->host_profiles[profile];
      const uint64_t length = batch_view.host_lengths[sequence];
      uint64_t dp_cells;
      uint64_t x_cells;
      uint64_t candidate_work;
      if (!checked_multiply(descriptor.q, 3 * kSubwarp, &dp_cells) ||
          !checked_multiply(length + 1, p7X_NXCELLS, &x_cells) ||
          !checked_multiply(descriptor.model_length, length,
                            &candidate_work) ||
          !checked_add(model_length_sum, descriptor.model_length,
                       &model_length_sum) ||
          !checked_add(target_length_sum, length, &target_length_sum) ||
          !checked_add(host_dp_offsets[candidate], dp_cells,
                       &host_dp_offsets[candidate + 1]) ||
          !checked_add(host_x_offsets[candidate], x_cells,
                       &host_x_offsets[candidate + 1]) ||
          !checked_add(work_cells, candidate_work, &work_cells)) {
        set_error(error, error_size, "Forward work size overflow");
        return -1;
      }
      host_candidate_profiles[candidate] = static_cast<uint32_t>(profile);
      host_candidate_sequences[candidate] = sequence;
      host_length_transitions[candidate] = length_transitions_for(
          descriptor, static_cast<int>(length));
      maximum_model_length = std::max(
          maximum_model_length,
          static_cast<uint64_t>(descriptor.model_length));
      maximum_target_length = std::max(maximum_target_length, length);
      maximum_candidate_work = std::max(
          maximum_candidate_work, candidate_work);
    }
  }
  created->statistics.work_cells = work_cells;
  created->subwarp_statistics.candidate_subwarp_count = candidate_count;
  created->subwarp_statistics.model_length_sum = model_length_sum;
  created->subwarp_statistics.target_length_sum = target_length_sum;
  created->subwarp_statistics.maximum_model_length = maximum_model_length;
  created->subwarp_statistics.maximum_target_length = maximum_target_length;
  created->subwarp_statistics.maximum_candidate_work_cells =
      maximum_candidate_work;

  const uint64_t dp_cell_limit =
      kDpWorkspaceByteLimit / sizeof(float);
  const uint64_t x_cell_limit =
      kXmxWorkspaceByteLimit / sizeof(float);
  for (size_t begin = 0; begin < candidate_count;) {
    size_t end = begin + 1;
    if (host_dp_offsets[end] - host_dp_offsets[begin] > dp_cell_limit ||
        host_x_offsets[end] - host_x_offsets[begin] > x_cell_limit) {
      set_error(error, error_size,
                "one Forward row exceeds the workspace cap");
      return -1;
    }
    while (end < candidate_count &&
           host_dp_offsets[end + 1] - host_dp_offsets[begin] <=
               dp_cell_limit &&
           host_x_offsets[end + 1] - host_x_offsets[begin] <=
               x_cell_limit)
      ++end;
    maximum_dp_cells = std::max(
        maximum_dp_cells,
        host_dp_offsets[end] - host_dp_offsets[begin]);
    maximum_x_cells = std::max(
        maximum_x_cells,
        host_x_offsets[end] - host_x_offsets[begin]);
    maximum_tile_count = std::max(maximum_tile_count, end - begin);
    try {
      tile_boundaries.push_back(end);
    } catch (...) {
      set_error(error, error_size,
                "Forward tile boundary allocation failed");
      return -1;
    }
    begin = end;
  }
  created->statistics.dp_workspace_bytes =
      maximum_dp_cells * sizeof(float);
  created->statistics.xmx_workspace_bytes =
      maximum_x_cells * sizeof(float);

  size_t candidate_bytes;
  size_t threshold_bytes;
  size_t length_transition_bytes;
  size_t offset_bytes;
  size_t result_bytes;
  size_t dp_bytes;
  size_t xmx_bytes;
  size_t survivor_candidate_bytes;
  size_t survivor_offset_bytes;
  size_t scan_workspace_bytes = 0;
  size_t gather_buffer_bytes;
  if (!checked_bytes(candidate_count, sizeof(uint32_t),
                     &candidate_bytes) ||
      !checked_bytes(profile_count, sizeof(uint32_t), &threshold_bytes) ||
      !checked_bytes(candidate_count, sizeof(ForwardLengthTransitions),
                     &length_transition_bytes) ||
      !checked_bytes(candidate_count + 1, sizeof(uint64_t), &offset_bytes) ||
      !checked_bytes(candidate_count, sizeof(ForwardKernelResult),
                     &result_bytes) ||
      maximum_dp_cells > SIZE_MAX / sizeof(float) ||
      maximum_x_cells > SIZE_MAX / sizeof(float) ||
      !checked_bytes(maximum_tile_count, sizeof(uint32_t),
                     &survivor_candidate_bytes) ||
      !checked_bytes(maximum_tile_count + 1, sizeof(uint64_t),
                     &survivor_offset_bytes)) {
    set_error(error, error_size, "Forward device workspace size overflow");
    return -1;
  }
  if (maximum_tile_count >= static_cast<size_t>(INT_MAX)) {
    set_error(error, error_size,
              "Forward survivor tile exceeds the CUB scan item limit");
    return -1;
  }
  dp_bytes = static_cast<size_t>(maximum_dp_cells) * sizeof(float);
  xmx_bytes = static_cast<size_t>(maximum_x_cells) * sizeof(float);

  int maximum_grid_x = 0;
  status = cudaDeviceGetAttribute(
      &maximum_grid_x, cudaDevAttrMaxGridDimX, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid x)", status);
    return -1;
  }
  int multiprocessor_count = 0;
  status = cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, current_device);
  if (status != cudaSuccess || multiprocessor_count <= 0) {
    if (status != cudaSuccess)
      set_cuda_error(error, error_size,
                     "cudaDeviceGetAttribute(multiprocessor count)", status);
    else
      set_error(error, error_size, "CUDA multiprocessor count is invalid");
    return -1;
  }
  int l2_cache_bytes = 0;
  status = cudaDeviceGetAttribute(
      &l2_cache_bytes, cudaDevAttrL2CacheSize, current_device);
  if (status != cudaSuccess || l2_cache_bytes <= 0) {
    if (status != cudaSuccess)
      set_cuda_error(error, error_size,
                     "cudaDeviceGetAttribute(L2 cache size)", status);
    else
      set_error(error, error_size, "CUDA L2 cache size is invalid");
    return -1;
  }
  candidates_per_warp = select_forward_candidates_per_warp(
      requested_candidates_per_warp, maximum_tile_count, work_cells,
      maximum_x_cells * sizeof(float), multiprocessor_count, l2_cache_bytes,
      &created->subwarp_statistics);

  RunBuffers &buffers = workspace->buffers;
#define CUDA_RUN(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)
  if (grow_forward_workspace_buffer(
          &buffers.candidate_profiles,
          &workspace->candidate_profiles_capacity, candidate_bytes,
          &workspace->growth_count, "cudaMalloc(Forward profile indexes)",
          error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.candidate_sequences,
          &workspace->candidate_sequences_capacity, candidate_bytes,
          &workspace->growth_count, "cudaMalloc(Forward sequence indexes)",
          error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.filter_scores, &workspace->filter_scores_capacity,
          candidate_bytes, &workspace->growth_count,
          "cudaMalloc(Forward filter scores)", error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.f3_threshold_bits,
          &workspace->f3_threshold_bits_capacity, threshold_bytes,
          &workspace->growth_count, "cudaMalloc(Forward F3 thresholds)",
          error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.length_transitions,
          &workspace->length_transitions_capacity, length_transition_bytes,
          &workspace->growth_count, "cudaMalloc(Forward length transitions)",
          error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.dp_offsets, &workspace->dp_offsets_capacity, offset_bytes,
          &workspace->growth_count, "cudaMalloc(Forward DP offsets)", error,
          error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.x_offsets, &workspace->x_offsets_capacity, offset_bytes,
          &workspace->growth_count, "cudaMalloc(Forward special offsets)",
          error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.dp, &workspace->dp_capacity, dp_bytes,
          &workspace->growth_count, "cudaMalloc(Forward DP workspace)", error,
          error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.xmx, &workspace->xmx_capacity, xmx_bytes,
          &workspace->growth_count, "cudaMalloc(Forward special workspace)",
          error, error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.results, &workspace->results_capacity, result_bytes,
          &workspace->growth_count, "cudaMalloc(Forward results)", error,
          error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.survivor_candidates,
          &workspace->survivor_candidates_capacity, survivor_candidate_bytes,
          &workspace->growth_count, "cudaMalloc(Forward survivors)", error,
          error_size) != 0 ||
      grow_forward_workspace_buffer(
          &buffers.survivor_offsets, &workspace->survivor_offsets_capacity,
          survivor_offset_bytes, &workspace->growth_count,
          "cudaMalloc(Forward survivor offsets)", error, error_size) != 0)
    return -1;
  status = cub::DeviceScan::ExclusiveSum(
      nullptr, scan_workspace_bytes, buffers.survivor_offsets,
      static_cast<int>(maximum_tile_count + 1));
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "CUB Forward survivor scan workspace query", status);
    return -1;
  }
  gather_buffer_bytes = std::max(xmx_bytes, scan_workspace_bytes);
  if (grow_forward_workspace_buffer(
          &buffers.gathered, &workspace->gathered_capacity,
          gather_buffer_bytes, &workspace->growth_count,
          "cudaMalloc(Forward gather/scan workspace)", error,
          error_size) != 0)
    return -1;

  /* The tile XMX buffer is overwritten by the next tile.  The resident path
   * therefore reserves one generation-owned append buffer, bounded by both
   * the existing output cap and the total possible row bytes.  Persistent
   * legacy work buffers are grown first, so allocation pressure here fails
   * soft without turning an otherwise viable run into an error. */
  if (retain_device_specials && output_byte_limit != 0) {
    const uint64_t total_x_cells = host_x_offsets.back();
    const uint64_t output_cell_limit = output_byte_limit / sizeof(float);
    const uint64_t requested_cells =
        std::min(total_x_cells, output_cell_limit);
    const size_t requested_bytes =
        static_cast<size_t>(requested_cells) * sizeof(float);
    created->residency_statistics.requested_bytes = requested_bytes;
    if (requested_bytes != 0) {
      const auto allocation_begin = std::chrono::steady_clock::now();
      status = cudaMalloc(&created->resident_specials.pointer,
                          requested_bytes);
      created->residency_statistics.allocation_milliseconds =
          std::chrono::duration<float, std::milli>(
              std::chrono::steady_clock::now() - allocation_begin).count();
      if (status == cudaSuccess) {
        created->resident_specials.device_ordinal = current_device;
        created->residency_statistics.allocated_bytes = requested_bytes;
      } else if (status == cudaErrorMemoryAllocation) {
        created->resident_specials.pointer = nullptr;
        ++created->residency_statistics.allocation_fallback_count;
        /* Do not let the optional allocation's per-thread last-error state
         * poison the next legacy kernel launch check. */
        cudaGetLastError();
      } else {
        set_cuda_error(error, error_size,
                       "cudaMalloc(Forward resident specials)", status);
        return -1;
      }
    }
  }
  created->statistics.gather_workspace_bytes =
      static_cast<uint64_t>(gather_buffer_bytes) + survivor_candidate_bytes +
      survivor_offset_bytes;
  const auto input_upload_begin = std::chrono::steady_clock::now();
  if (resident_f2) {
    const auto gather_begin = std::chrono::steady_clock::now();
    const size_t gather_blocks =
        (candidate_count + kGatherThreads - 1) / kGatherThreads;
    gather_resident_f2_inputs_kernel<<<
        static_cast<unsigned>(gather_blocks), kGatherThreads>>>(
        postfilter_view->device_candidates, postfilter_view->device_results,
        postfilter_view->device_selected_sources, candidate_count,
        buffers.candidate_profiles, buffers.candidate_sequences,
        buffers.filter_scores);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaDeviceSynchronize());
    created->input_residency_statistics.resident_f2_call_count = 1;
    created->input_residency_statistics.resident_f2_candidate_count =
        candidate_count;
    created->input_residency_statistics.eliminated_candidate_h2d_bytes =
        static_cast<uint64_t>(candidate_bytes) * 3;
    created->input_residency_statistics.gather_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - gather_begin).count();
  } else {
    CUDA_RUN(cudaMemcpy(buffers.candidate_profiles,
                        host_candidate_profiles.data(),
                        candidate_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.candidate_sequences,
                        host_candidate_sequences.data(),
                        candidate_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.filter_scores, filter_scores,
                        candidate_bytes, cudaMemcpyHostToDevice));
  }
  CUDA_RUN(cudaMemcpy(buffers.f3_threshold_bits,
                      host_f3_threshold_bits.data(),
                      threshold_bytes, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(buffers.length_transitions,
                      host_length_transitions.data(),
                      length_transition_bytes, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(buffers.dp_offsets, host_dp_offsets.data(),
                      offset_bytes, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(buffers.x_offsets, host_x_offsets.data(),
                      offset_bytes, cudaMemcpyHostToDevice));
  created->upload_milliseconds +=
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - input_upload_begin).count();
  if (buffers.begin_event == nullptr) {
    CUDA_RUN(cudaEventCreate(&buffers.begin_event));
    ++workspace->event_create_count;
  }
  if (buffers.end_event == nullptr) {
    CUDA_RUN(cudaEventCreate(&buffers.end_event));
    ++workspace->event_create_count;
  }

  std::vector<ForwardKernelResult> &host_kernel_results =
      workspace->host_kernel_results;
  std::vector<uint64_t> &host_f3_overrides =
      workspace->host_f3_overrides;
  try {
    host_kernel_results.resize(candidate_count);
    host_f3_overrides.reserve(maximum_tile_count);
  } catch (...) {
    set_error(error, error_size, "Forward host result allocation failed");
    return -1;
  }

  const auto total_begin = std::chrono::steady_clock::now();
  uint64_t gathered_cells = 0;
  bool output_cap_exhausted = false;
  for (size_t tile = 0; tile + 1 < tile_boundaries.size(); ++tile) {
    const size_t begin = tile_boundaries[tile];
    const size_t end = tile_boundaries[tile + 1];
    const size_t tile_count = end - begin;
    const size_t candidates_per_block =
        static_cast<size_t>(candidates_per_warp) * kWarpsPerBlock;
    const size_t blocks =
        (tile_count + candidates_per_block - 1) / candidates_per_block;
    if (blocks > static_cast<size_t>(maximum_grid_x)) {
      set_error(error, error_size, "Forward candidate grid is too large");
      return -1;
    }
    CUDA_RUN(cudaEventRecord(buffers.begin_event));
#define LAUNCH_FORWARD(candidates)                                             \
    forward_kernel<candidates><<<static_cast<unsigned>(blocks), kThreads>>>(  \
        batch_view.device_residues, batch_view.device_offsets,                \
        database->device_profiles, database->device_emissions,                \
        database->device_transitions, buffers.candidate_profiles,             \
        buffers.candidate_sequences, buffers.filter_scores,                   \
        buffers.f3_threshold_bits, buffers.length_transitions,                \
        buffers.dp_offsets, buffers.x_offsets, begin, tile_count,             \
        host_dp_offsets[begin], host_x_offsets[begin], buffers.dp,             \
        buffers.xmx, buffers.results)
    switch (candidates_per_warp) {
      case 1: LAUNCH_FORWARD(1); break;
      case 2: LAUNCH_FORWARD(2); break;
      case 4: LAUNCH_FORWARD(4); break;
      case 8: LAUNCH_FORWARD(8); break;
      default:
        set_error(error, error_size,
                  "invalid Forward candidates-per-warp dispatch");
        return -1;
    }
#undef LAUNCH_FORWARD
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaEventRecord(buffers.end_event));
    CUDA_RUN(cudaEventSynchronize(buffers.end_event));
    float elapsed = 0.0f;
    CUDA_RUN(cudaEventElapsedTime(
        &elapsed, buffers.begin_event, buffers.end_event));
    created->statistics.kernel_milliseconds += elapsed;
    ++created->subwarp_statistics.kernel_launch_count;
    created->subwarp_statistics.scheduled_warp_count +=
        static_cast<uint64_t>(blocks) * kWarpsPerBlock;
    created->subwarp_statistics.active_lane_slots +=
        static_cast<uint64_t>(tile_count) * kSubwarp;
    created->subwarp_statistics.issued_lane_slots +=
        static_cast<uint64_t>(blocks) * kThreads;

    const auto download_begin = std::chrono::steady_clock::now();
    CUDA_RUN(cudaMemcpy(host_kernel_results.data() + begin,
                        buffers.results + begin,
                        tile_count * sizeof(ForwardKernelResult),
                        cudaMemcpyDeviceToHost));
    created->statistics.download_milliseconds +=
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - download_begin).count();

    const auto classification_begin = std::chrono::steady_clock::now();
    host_f3_overrides.clear();
    uint64_t tile_gathered_cells = 0;
    size_t survivor_count = 0;
    size_t accepted_candidate_end = output_cap_exhausted ? begin : end;
    for (size_t candidate = begin; candidate < end; ++candidate) {
      created->special_offsets[candidate] = gathered_cells;
      const uint32_t profile_index = host_candidate_profiles[candidate];
      const ForwardProfile &profile =
          database->host_profiles[profile_index];
      const ForwardKernelResult kernel_result =
          host_kernel_results[candidate];
      plan7_forward_result &result = created->results[candidate];
      const uint32_t kernel_status =
          kernel_result.status_and_f3 & kKernelStatusMask;
      const KernelF3Decision device_f3 = static_cast<KernelF3Decision>(
          kernel_result.status_and_f3 >> kKernelF3Shift);
      result.status = static_cast<uint8_t>(kernel_status);
      FloatBits fwdsc{};
      fwdsc.bits = kernel_result.score_bits;
      if (kernel_status == eslOK &&
          batch_view.host_lengths[host_candidate_sequences[candidate]] != 0 &&
          std::isfinite(fwdsc.value) &&
          std::isfinite(filter_scores[candidate]) &&
          std::isfinite(profile.f_tau) &&
          std::isfinite(profile.f_lambda) && profile.f_lambda > 0.0f) {
        result.fwdsc = fwdsc.value;
        bool f3_pass = false;
        if (device_f3 == kKernelF3Reject || device_f3 == kKernelF3Pass) {
          const bool device_pass = device_f3 == kKernelF3Pass;
          if (host_f3_threshold_bits[profile_index] ==
              UINT32_C(0x7fc00000)) {
            set_error(error, error_size,
                      "unsupported Forward profile produced a device F3 decision");
            return -1;
          }
          ++created->f3_device_statistics.device_decision_count;
          if (device_pass)
            ++created->f3_device_statistics.device_pass_count;
          else
            ++created->f3_device_statistics.device_reject_count;
          f3_pass = device_pass;
          if (audit_f3) {
            const float difference = fwdsc.value - filter_scores[candidate];
            const float bit_score = static_cast<float>(
                static_cast<double>(difference) / kLog2);
            const double probability = esl_exp_surv(
                bit_score, profile.f_tau, profile.f_lambda);
            const bool host_pass = !(probability > f3);
            ++created->f3_device_statistics.host_audit_count;
            if (device_pass != host_pass) {
              ++created->f3_device_statistics.decision_mismatch_count;
              set_error(error, error_size,
                        "device and host Forward F3 decisions differ");
              return -1;
            }
          } else {
            ++created->f3_device_statistics.host_decision_avoided_count;
          }
        } else {
          if (device_f3 != kKernelF3Unavailable ||
              host_f3_threshold_bits[profile_index] !=
                  UINT32_C(0x7fc00000)) {
            set_error(error, error_size,
                      "compiled Forward profile has no device F3 decision");
            return -1;
          }
          const float difference = fwdsc.value - filter_scores[candidate];
          const float bit_score = static_cast<float>(
              static_cast<double>(difference) / kLog2);
          const double probability = esl_exp_surv(
              bit_score, profile.f_tau, profile.f_lambda);
          f3_pass = !(probability > f3);
          ++created->f3_device_statistics.host_fallback_count;
          host_f3_overrides.push_back(
              (static_cast<uint64_t>(candidate) << 32) |
              static_cast<uint32_t>(f3_pass ? kKernelF3Pass
                                            : kKernelF3Reject));
        }
        if (!f3_pass) {
          result.action = PLAN7_FORWARD_DEFINITE_REJECT;
        } else {
          const uint64_t x_cells =
              host_x_offsets[candidate + 1] - host_x_offsets[candidate];
          const uint64_t output_cell_limit =
              output_byte_limit / sizeof(float);
          if (output_cap_exhausted ||
              x_cells > output_cell_limit - gathered_cells) {
            result.action = PLAN7_FORWARD_CPU_REQUIRED;
            if (collect_reason_facts)
              created->reason_facts[candidate] |=
                  PLAN7_FORWARD_REASON_OUTPUT_CAP;
            if (!output_cap_exhausted)
              accepted_candidate_end = candidate;
            output_cap_exhausted = true;
            ++created->statistics.output_cap_fallback_count;
          } else if (!checked_add(tile_gathered_cells, x_cells,
                           &tile_gathered_cells) ||
                     !checked_add(gathered_cells, x_cells,
                                  &gathered_cells) ||
                     gathered_cells > SIZE_MAX) {
            set_error(error, error_size,
                      "Forward gathered matrix size overflow");
            return -1;
          } else {
            result.action = PLAN7_FORWARD_DEFINITE_PASS;
            ++survivor_count;
            ++created->statistics.survivor_count;
          }
        }
      } else if (batch_view.host_lengths[
                     host_candidate_sequences[candidate]] == 0) {
        result.status = PLAN7_FORWARD_EMPTY;
      }
      created->special_offsets[candidate + 1] = gathered_cells;
    }
    created->statistics.classification_milliseconds +=
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - classification_begin).count();

    if (survivor_count == 0) continue;
    if (survivor_count > static_cast<size_t>(maximum_grid_x) ||
        tile_gathered_cells > maximum_x_cells) {
      set_error(error, error_size, "Forward gather grid or size is invalid");
      return -1;
    }
    const size_t old_special_count =
        created->specials.size();
    if (tile_gathered_cells > SIZE_MAX - old_special_count) {
      set_error(error, error_size, "Forward special matrix size overflow");
      return -1;
    }
    try {
      created->specials.resize(
          old_special_count + static_cast<size_t>(tile_gathered_cells));
    } catch (...) {
      set_error(error, error_size,
                "Forward special matrix allocation failed");
      return -1;
    }
    CUDA_RUN(cudaEventRecord(buffers.begin_event));
    if (!host_f3_overrides.empty()) {
      const auto override_upload_begin = std::chrono::steady_clock::now();
      CUDA_RUN(cudaMemcpy(buffers.survivor_offsets,
                          host_f3_overrides.data(),
                          host_f3_overrides.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
      created->upload_milliseconds +=
          std::chrono::duration<float, std::milli>(
              std::chrono::steady_clock::now() -
              override_upload_begin).count();
      const size_t override_blocks =
          (host_f3_overrides.size() + kGatherThreads - 1) / kGatherThreads;
      apply_f3_overrides_kernel<<<static_cast<unsigned>(override_blocks),
                                  kGatherThreads>>>(
          buffers.results, buffers.survivor_offsets,
          host_f3_overrides.size());
      CUDA_RUN(cudaGetLastError());
    }
    const size_t compact_blocks =
        (tile_count + kGatherThreads - 1) / kGatherThreads;
    mark_f3_survivors_kernel<<<static_cast<unsigned>(compact_blocks),
                               kGatherThreads>>>(
        buffers.results, begin, tile_count, accepted_candidate_end,
        buffers.survivor_offsets);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cub::DeviceScan::ExclusiveSum(
        buffers.gathered, scan_workspace_bytes, buffers.survivor_offsets,
        static_cast<int>(tile_count)));
    scatter_f3_survivors_kernel<<<static_cast<unsigned>(compact_blocks),
                                  kGatherThreads>>>(
        buffers.results, buffers.survivor_offsets, begin, tile_count,
        accepted_candidate_end, buffers.survivor_candidates);
    CUDA_RUN(cudaGetLastError());
    const size_t count_blocks =
        (survivor_count + 1 + kGatherThreads - 1) / kGatherThreads;
    survivor_cell_counts_kernel<<<static_cast<unsigned>(count_blocks),
                                   kGatherThreads>>>(
        buffers.survivor_candidates, survivor_count, buffers.x_offsets,
        buffers.survivor_offsets);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cub::DeviceScan::ExclusiveSum(
        buffers.gathered, scan_workspace_bytes, buffers.survivor_offsets,
        static_cast<int>(survivor_count + 1)));
    float *gather_destination =
        created->resident_specials.pointer == nullptr
            ? buffers.gathered
            : created->resident_specials.pointer + old_special_count;
    gather_specials_kernel<<<static_cast<unsigned>(survivor_count),
                             kGatherThreads>>>(
        buffers.xmx, buffers.x_offsets, buffers.survivor_candidates,
        buffers.survivor_offsets, survivor_count, host_x_offsets[begin],
        gather_destination);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaEventRecord(buffers.end_event));
    CUDA_RUN(cudaEventSynchronize(buffers.end_event));
    elapsed = 0.0f;
    CUDA_RUN(cudaEventElapsedTime(
        &elapsed, buffers.begin_event, buffers.end_event));
    created->statistics.gather_milliseconds += elapsed;
    if (created->resident_specials.pointer != nullptr)
      created->residency_statistics.materialization_milliseconds += elapsed;
    ++created->f3_device_statistics.device_compaction_run_count;
    created->f3_device_statistics.device_compaction_candidate_count +=
        tile_count;
    created->f3_device_statistics.device_compacted_survivor_count +=
        survivor_count;
    created->f3_device_statistics.survivor_upload_avoided_bytes +=
        static_cast<uint64_t>(survivor_count) * sizeof(uint32_t) +
        static_cast<uint64_t>(survivor_count + 1) * sizeof(uint64_t);
    const auto gather_download_begin = std::chrono::steady_clock::now();
    CUDA_RUN(cudaMemcpy(created->specials.data() + old_special_count,
                        gather_destination,
                        static_cast<size_t>(tile_gathered_cells) *
                            sizeof(float),
                        cudaMemcpyDeviceToHost));
    created->statistics.download_milliseconds +=
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - gather_download_begin).count();
  }
#undef CUDA_RUN
  created->statistics.gathered_xmx_bytes =
      gathered_cells * sizeof(float);
  if (created->resident_specials.pointer != nullptr)
    created->residency_statistics.materialized_bytes =
        created->statistics.gathered_xmx_bytes;
  created->statistics.total_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  if (!seal_forward_provenance(created.get(), database, batch_view,
                               &host_candidate_profiles, filter_scores,
                               generation_f3.bits)) {
    set_error(error, error_size, "Forward provenance sealing failed");
    return -1;
  }
  created->total_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - call_begin).count();
  *output = created.release();
  return 0;
}

extern "C" int plan7_forward_run_with_workspace(
    plan7_forward_workspace *workspace,
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, false, false, false,
      1, nullptr, error, error_size);
}

extern "C" int plan7_forward_run(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_forward_workspace_create(&workspace, error, error_size) != 0)
    return -1;
  const int run_status = plan7_forward_run_with_workspace(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, error, error_size);
  char destroy_error[512] = {0};
  const int destroy_status = plan7_forward_workspace_destroy(
      &workspace, destroy_error, sizeof(destroy_error));
  if (run_status != 0) return run_status;
  if (destroy_status != 0) {
    set_error(error, error_size, destroy_error);
    return -1;
  }
  return 0;
}

extern "C" int plan7_forward_run_batch_workspace(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return plan7_forward_run_with_workspace(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, error, error_size);
}

extern "C" int plan7_forward_run_batch_workspace_variant(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget, int candidates_per_warp,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, false, false, false,
      candidates_per_warp, nullptr, error, error_size);
}

extern "C" int plan7_forward_run_batch_workspace_reason_facts(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, true, false, false,
      1, nullptr, error, error_size);
}

extern "C" int plan7_forward_run_batch_workspace_resident(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, false, true, false,
      1, nullptr, error, error_size);
}

extern "C" int plan7_forward_run_batch_workspace_resident_reason_facts(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, true, true, false,
      1, nullptr, error, error_size);
}

extern "C" int plan7_forward_run_batch_workspace_postfilter_resident(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count,
    const plan7_postfilter_f2_resident_view *postfilter_view,
    double f3, uint64_t gathered_byte_budget, int collect_reason_facts,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, collect_reason_facts != 0, true,
      false, 1, postfilter_view, error, error_size);
}

extern "C" int plan7_forward_run_batch_workspace_f3_audit(
    const plan7_forward_database *database,
    plan7_ssv_sequence_batch *batch,
    const uintptr_t *source_profile_pointers, size_t profile_count,
    const uint64_t *candidate_offsets, const uint32_t *candidate_indices,
    const float *filter_scores, size_t candidate_count, double f3,
    uint64_t gathered_byte_budget,
    plan7_forward_output **output, char *error, size_t error_size) {
  plan7_forward_workspace *workspace = nullptr;
  if (plan7_ssv_sequence_batch_get_forward_workspace(
          batch, &workspace, error, error_size) != 0)
    return -1;
  return forward_run_with_workspace_impl(
      workspace, database, batch, source_profile_pointers, profile_count,
      candidate_offsets, candidate_indices, filter_scores, candidate_count,
      f3, gathered_byte_budget, output, false, false, true,
      1, nullptr, error, error_size);
}

extern "C" int plan7_forward_output_destroy(
    plan7_forward_output **output, char *error, size_t error_size) {
  if (output == nullptr) {
    set_error(error, error_size, "Forward output handle is null");
    return -1;
  }
  delete *output;
  *output = nullptr;
  return 0;
}

extern "C" size_t plan7_forward_output_result_count(
    const plan7_forward_output *output) {
  return output == nullptr ? 0 : output->results.size();
}

extern "C" const plan7_forward_result *plan7_forward_output_results(
    const plan7_forward_output *output) {
  return output == nullptr || output->results.empty()
      ? nullptr : output->results.data();
}

extern "C" size_t plan7_forward_output_reason_count(
    const plan7_forward_output *output) {
  return output == nullptr ? 0 : output->reason_facts.size();
}

extern "C" const uint16_t *plan7_forward_output_reason_facts(
    const plan7_forward_output *output) {
  return output == nullptr || output->reason_facts.empty()
      ? nullptr : output->reason_facts.data();
}

extern "C" const uint64_t *plan7_forward_output_special_offsets(
    const plan7_forward_output *output) {
  return output == nullptr || output->special_offsets.empty()
      ? nullptr : output->special_offsets.data();
}

extern "C" size_t plan7_forward_output_special_count(
    const plan7_forward_output *output) {
  return output == nullptr ? 0 : output->specials.size();
}

extern "C" const float *plan7_forward_output_specials(
    const plan7_forward_output *output) {
  return output == nullptr || output->specials.empty()
      ? nullptr : output->specials.data();
}

extern "C" const plan7_forward_statistics *
plan7_forward_output_statistics(const plan7_forward_output *output) {
  return output == nullptr ? nullptr : &output->statistics;
}

extern "C" const plan7_forward_residency_statistics *
plan7_forward_output_residency_statistics(
    const plan7_forward_output *output) {
  return output == nullptr ? nullptr : &output->residency_statistics;
}

extern "C" const plan7_forward_input_residency_statistics *
plan7_forward_output_input_residency_statistics(
    const plan7_forward_output *output) {
  return output == nullptr ? nullptr : &output->input_residency_statistics;
}

extern "C" const plan7_forward_subwarp_statistics *
plan7_forward_output_subwarp_statistics(
    const plan7_forward_output *output) {
  return output == nullptr ? nullptr : &output->subwarp_statistics;
}

extern "C" int plan7_forward_output_get_resident_view(
    const plan7_forward_output *output,
    plan7_forward_resident_view *view,
    char *error, size_t error_size) {
  if (output == nullptr || view == nullptr) {
    set_error(error, error_size, "invalid Forward resident view request");
    return -1;
  }
  *view = {};
  if (output->resident_specials.pointer == nullptr) return 0;
  if (output->residency_statistics.materialized_bytes !=
          output->specials.size() * sizeof(float) ||
      output->residency_statistics.materialized_bytes >
          output->residency_statistics.allocated_bytes ||
      output->provenance.special_count != output->specials.size() ||
      output->provenance.pass_count > output->results.size() ||
      output->resident_specials.device_ordinal < 0) {
    set_error(error, error_size,
              "Forward resident generation is incomplete");
    return -1;
  }
  view->database_generation = output->provenance.database_generation;
  view->batch_generation = output->provenance.batch_generation;
  view->pass_count = output->provenance.pass_count;
  view->special_count = output->provenance.special_count;
  view->device_ordinal = output->resident_specials.device_ordinal;
  view->specials = output->resident_specials.pointer;
  return 1;
}

extern "C" const plan7_forward_f3_device_statistics *
plan7_forward_output_f3_device_statistics(
    const plan7_forward_output *output) {
  return output == nullptr ? nullptr : &output->f3_device_statistics;
}

extern "C" float plan7_forward_output_upload_milliseconds(
    const plan7_forward_output *output) {
  return output == nullptr ? 0.0f : output->upload_milliseconds;
}

extern "C" float plan7_forward_output_total_milliseconds(
    const plan7_forward_output *output) {
  return output == nullptr ? 0.0f : output->total_milliseconds;
}

extern "C" int plan7_forward_output_contract_fallback(
    const plan7_forward_output *output) {
  return output != nullptr && output->contract_fallback ? 1 : 0;
}

extern "C" const plan7_forward_provenance *
plan7_forward_output_provenance(const plan7_forward_output *output) {
  return output == nullptr ? nullptr : &output->provenance;
}

extern "C" int plan7_forward_database_validate_provenance(
    const plan7_forward_database *database,
    const plan7_forward_provenance *provenance) {
  if (database == nullptr || provenance == nullptr ||
      provenance->database_generation != database->generation_id)
    return 0;
  return provenance->integrity_tag ==
         provenance_integrity_tag(database, *provenance);
}
