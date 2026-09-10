#include "postfilter_cuda.h"
#include "forward_cuda.h"
#include "f3_threshold.h"

#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

extern "C" {
#include <easel.h>
#include <hmmer.h>
#include <impl_sse/impl_sse.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

static_assert(sizeof(plan7_postfilter_result) == PLAN7_POSTFILTER_RECORD_SIZE,
              "post-filter record ABI size changed");
static_assert(offsetof(plan7_postfilter_result, sequence_index) == 0 &&
              offsetof(plan7_postfilter_result, filtersc) == 4 &&
              offsetof(plan7_postfilter_result, msv_numerator) == 8 &&
              offsetof(plan7_postfilter_result, msv_status) == 10 &&
              offsetof(plan7_postfilter_result, action) == 11 &&
              offsetof(plan7_postfilter_result, vfsc) == 12,
              "post-filter record ABI layout changed");
static_assert(sizeof(plan7_forward_snapshot_profile) == 48,
              "Forward snapshot descriptor footprint changed");

namespace {

constexpr int kWarpSize = 32;
constexpr int kForwardSubwarp = 4;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kNegInf = -32768;
constexpr uint64_t kDpByteLimit = UINT64_C(256) << 20;
constexpr size_t kFullMsvCompactSourceChunk = UINT64_C(4) << 20;
constexpr size_t kMaximumTargetLength = 100000;
constexpr double kLog2 = 0.69314718055994529;

enum CandidateState : uint8_t {
  kCandidateCpu = 0,
  kCandidateRawReject = 1,
  kCandidateFinite = 2,
  kCandidateMsvRange = 3
};

struct VitProfile {
  uint64_t ssv_offset;
  uint64_t rbv_offset;
  uint64_t emission_offset;
  uint64_t transition_offset;
  uint32_t q;
  uint32_t model_length;
  int32_t mode;
  int32_t base;
  int32_t ddbound;
  int32_t e_move;
  int32_t e_loop;
  int32_t n_loop;
  int32_t j_loop;
  int32_t c_loop;
  float scale;
  float nj;
  uint8_t msv_tbm;
  uint8_t msv_tec;
  uint8_t msv_base;
  uint8_t msv_bias;
  float msv_scale;
  uint64_t reserved;
};

static_assert(sizeof(VitProfile) == 96,
              "Viterbi descriptor footprint changed");
static_assert(sizeof(uintptr_t) >= sizeof(uint64_t),
              "profile identity tokens require 64-bit uintptr_t");
static_assert(p7O_NTRANS == 8,
              "Viterbi transition-row footprint changed");
static_assert((29 + p7O_NTRANS) * kWarpSize * sizeof(int16_t) == 2368,
              "Viterbi packed-row footprint changed");

struct VitLengthTransitions {
  int16_t n_move;
  int16_t j_move;
  int16_t c_move;
};

struct VitResult {
  int32_t status;
  uint32_t score_bits;
  int32_t numerator;
};

union FloatBits {
  float value;
  uint32_t bits;
};

struct F2DeviceProfile {
  float scale;
  uint32_t msv_threshold_bits;
  uint32_t viterbi_threshold_bits;
};

static_assert(sizeof(F2DeviceProfile) == 12,
              "F2 device profile layout changed");

__device__ __forceinline__ bool f2_threshold_pass(
    float bit_score, uint32_t threshold_bits) {
  FloatBits threshold{};
  threshold.bits = threshold_bits;
  return !isnan(bit_score) && bit_score >= threshold.value;
}

__device__ __forceinline__ bool exact_f2_pass(
    const plan7_postfilter_result &result,
    const F2DeviceProfile &profile) {
  FloatBits vfsc{};
  vfsc.value = result.vfsc;
  if (result.action != PLAN7_BIAS_DEFINITE_PASS ||
      result.msv_status != PLAN7_SSV_OK || !isfinite(result.filtersc) ||
      (!isfinite(result.vfsc) && vfsc.bits != UINT32_C(0x7f800000)) ||
      !isfinite(profile.scale) || profile.scale <= 0.0f)
    return false;

  float usc = __int2float_rn(static_cast<int>(result.msv_numerator));
  usc = __fdiv_rn(usc, profile.scale);
  usc = __fsub_rn(usc, 3.0f);
  const float msv_bit_score = __double2float_rn(__ddiv_rn(
      static_cast<double>(__fsub_rn(usc, result.filtersc)), kLog2));
  if (f2_threshold_pass(msv_bit_score, profile.msv_threshold_bits))
    return true;
  const float viterbi_bit_score = __double2float_rn(__ddiv_rn(
      static_cast<double>(__fsub_rn(result.vfsc, result.filtersc)), kLog2));
  return f2_threshold_pass(
      viterbi_bit_score, profile.viterbi_threshold_bits);
}

__global__ void classify_f2_words_kernel(
    const plan7_bias_candidate *candidates,
    const plan7_postfilter_result *results,
    const F2DeviceProfile *profiles,
    size_t profile_count,
    size_t source_count,
    size_t word_count,
    uint32_t *masks,
    uint64_t *counts) {
  const size_t thread =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t word = thread / 32;
  const unsigned lane = thread & 31U;
  if (word >= word_count) return;
  const size_t source = word * 32 + lane;
  bool pass = false;
  if (source < source_count) {
    const plan7_bias_candidate mapping = candidates[source];
    if (mapping.profile_index < profile_count)
      pass = exact_f2_pass(results[source], profiles[mapping.profile_index]);
  }
  const uint32_t mask = __ballot_sync(UINT32_MAX, pass);
  if (lane == 0) {
    masks[word] = mask;
    counts[word] = static_cast<uint64_t>(__popc(mask));
    if (word + 1 == word_count) counts[word_count] = 0;
  }
}

__global__ void scatter_f2_sources_kernel(
    const uint32_t *masks,
    const uint64_t *ranks,
    size_t word_count,
    uint32_t *selected_sources) {
  const size_t word =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word >= word_count) return;
  uint32_t mask = masks[word];
  uint64_t output = ranks[word];
  while (mask != 0) {
    const unsigned bit = static_cast<unsigned>(__ffs(mask) - 1);
    selected_sources[output++] =
        static_cast<uint32_t>(word * 32 + bit);
    mask &= mask - 1;
  }
}

uint64_t hash_selected_sources(const uint32_t *sources, size_t count) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < count; ++i) {
    uint32_t value = sources[i];
    for (unsigned byte = 0; byte < sizeof(value); ++byte) {
      hash ^= static_cast<uint8_t>(value >> (byte * 8));
      hash *= UINT64_C(1099511628211);
    }
  }
  hash ^= static_cast<uint64_t>(count);
  hash *= UINT64_C(1099511628211);
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

bool checked_bytes(size_t count, size_t size, size_t *bytes) {
  if (size != 0 && count > SIZE_MAX / size) return false;
  *bytes = count * size;
  return true;
}

void saturating_counter_add(uint64_t value, uint64_t *counter) {
  *counter = value > UINT64_MAX - *counter ? UINT64_MAX : *counter + value;
}

bool aligned_vector_address(const void *allocation, uintptr_t *address) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation);
  if (raw == 0 || raw > UINTPTR_MAX - 15) return false;
  *address = (raw + 15) & ~static_cast<uintptr_t>(15);
  return true;
}

bool valid_profile_storage(const P7_OPROFILE *profile) {
  if (profile == nullptr || profile->abc == nullptr ||
      profile->abc->type != eslAMINO || profile->abc->K != 20 ||
      profile->abc->Kp != 29 || profile->M < 1 || profile->M > 100000 ||
      profile->allocM < profile->M ||
      profile->allocM > 100000 || profile->allocQ16 != p7O_NQB(profile->allocM) ||
      profile->allocQ8 != p7O_NQW(profile->allocM) ||
      profile->allocQ16 < p7O_NQB(profile->M) ||
      profile->allocQ8 < p7O_NQW(profile->M) || profile->rbv == nullptr ||
      profile->sbv == nullptr || profile->rwv == nullptr ||
      profile->twv == nullptr || profile->rbv_mem == nullptr ||
      profile->sbv_mem == nullptr || profile->rwv_mem == nullptr ||
      profile->twv_mem == nullptr)
    return false;

  uintptr_t rbv_base;
  uintptr_t sbv_base;
  uintptr_t rwv_base;
  uintptr_t twv_base;
  if (!aligned_vector_address(profile->rbv_mem, &rbv_base) ||
      !aligned_vector_address(profile->sbv_mem, &sbv_base) ||
      !aligned_vector_address(profile->rwv_mem, &rwv_base) ||
      !aligned_vector_address(profile->twv_mem, &twv_base) ||
      reinterpret_cast<uintptr_t>(profile->twv) != twv_base)
    return false;

  const size_t rbv_stride = static_cast<size_t>(profile->allocQ16);
  const size_t sbv_stride = rbv_stride + p7O_EXTRA_SB;
  const size_t rwv_stride = static_cast<size_t>(profile->allocQ8);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const uintptr_t rbv_expected = rbv_base +
        static_cast<size_t>(residue) * rbv_stride * sizeof(__m128i);
    const uintptr_t sbv_expected = sbv_base +
        static_cast<size_t>(residue) * sbv_stride * sizeof(__m128i);
    const uintptr_t rwv_expected = rwv_base +
        static_cast<size_t>(residue) * rwv_stride * sizeof(__m128i);
    if (profile->rbv[residue] == nullptr || profile->sbv[residue] == nullptr ||
        profile->rwv[residue] == nullptr ||
        reinterpret_cast<uintptr_t>(profile->rbv[residue]) != rbv_expected ||
        reinterpret_cast<uintptr_t>(profile->sbv[residue]) != sbv_expected ||
        reinterpret_cast<uintptr_t>(profile->rwv[residue]) != rwv_expected)
      return false;
  }
  return true;
}

bool valid_viterbi_specials(const P7_OPROFILE *profile) {
  if (profile->base_w != 12000 ||
      (profile->nj != 0.0f && profile->nj != 1.0f) ||
      profile->xw[p7O_N][p7O_LOOP] != 0 ||
      profile->xw[p7O_J][p7O_LOOP] != 0 ||
      profile->xw[p7O_C][p7O_LOOP] != 0)
    return false;
  if (profile->nj == 0.0f)
    return profile->xw[p7O_E][p7O_MOVE] == 0 &&
           profile->xw[p7O_E][p7O_LOOP] == kNegInf;
  return profile->xw[p7O_E][p7O_MOVE] < 0 &&
         profile->xw[p7O_E][p7O_MOVE] > kNegInf &&
         profile->xw[p7O_E][p7O_LOOP] ==
             profile->xw[p7O_E][p7O_MOVE];
}

bool valid_forward_storage(const P7_OPROFILE *profile) {
  if (profile == nullptr || profile->abc == nullptr ||
      profile->abc->type != eslAMINO || profile->abc->K != 20 ||
      profile->abc->Kp != 29 || profile->M < 1 || profile->M > 100000 ||
      profile->allocM < profile->M || profile->allocM > 100000 ||
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

bool valid_forward_probability_rows(const P7_OPROFILE *profile) {
  const int q_count = p7O_NQF(profile->M);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const auto *values = reinterpret_cast<const float *>(profile->rfv[residue]);
    for (int cell = 0; cell < q_count * kForwardSubwarp; ++cell)
      if (!std::isfinite(values[cell]) || values[cell] < 0.0f) return false;
  }
  const auto *transitions = reinterpret_cast<const float *>(profile->tfv);
  for (int cell = 0;
       cell < q_count * p7O_NTRANS * kForwardSubwarp; ++cell)
    if (!std::isfinite(transitions[cell]) || transitions[cell] < 0.0f)
      return false;
  return true;
}

VitLengthTransitions length_transitions_for(const VitProfile &profile,
                                            int length) {
  const float numerator = 2.0f + profile.nj;
  const float denominator = static_cast<float>(length) + 2.0f + profile.nj;
  const float pmove = numerator / denominator;
  const float score = roundf(profile.scale * logf(pmove));
  int16_t move;
  if (score >= 32767.0f)
    move = 32767;
  else if (score <= -32768.0f)
    move = -32768;
  else
    move = static_cast<int16_t>(score);
  return {move, move, move};
}

__device__ __forceinline__ unsigned sat_add_u8(unsigned left,
                                                unsigned right) {
  const unsigned value = left + right;
  return value > UINT8_MAX ? UINT8_MAX : value;
}

__device__ __forceinline__ unsigned sat_sub_u8(unsigned left,
                                                unsigned right) {
  return left > right ? left - right : 0;
}

__device__ __forceinline__ int sat_add_i16(int left, int right) {
  const int value = left + right;
  return value > 32767 ? 32767 : (value < kNegInf ? kNegInf : value);
}

template<typename T>
__device__ __forceinline__ T warp_max(T value) {
  for (int delta = 16; delta != 0; delta >>= 1)
    value = max(value, __shfl_xor_sync(UINT32_MAX, value, delta));
  return value;
}

template<typename T>
__device__ __forceinline__ T previous_lane(T value, int lane, T first) {
  value = __shfl_up_sync(UINT32_MAX, value, 1);
  return lane == 0 ? first : value;
}

__device__ __forceinline__ bool raw_f1_survives(
    const plan7_bias_ssv_input result, float null_score,
    const plan7_ssv_f1_profile profile) {
  if (result.status == PLAN7_SSV_ERANGE) return true;
  if (result.status != PLAN7_SSV_OK) return false;
  if (profile.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_REJECT) return false;
  if (profile.cutoff_mode != PLAN7_F1_CUTOFF_SCORE ||
      !isfinite(null_score) || !isfinite(profile.profile.scale) ||
      profile.profile.scale <= 0.0f ||
      !isfinite(profile.cutoff_bit_score))
    return false;
  float score = __int2float_rn(static_cast<int>(result.numerator));
  score = __fdiv_rn(score, profile.profile.scale);
  score = __fsub_rn(score, 3.0f);
  const float delta = __fsub_rn(score, null_score);
  const float bit_score = __double2float_rn(__ddiv_rn(
      static_cast<double>(delta), kLog2));
  return isfinite(score) && isfinite(bit_score) &&
         bit_score >= profile.cutoff_bit_score;
}

__global__ void full_msv_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const uint8_t *compact_scores, const uint8_t *exact_rbv,
    const plan7_ssv_f1_profile *msv_profiles,
    const VitProfile *vit_profiles, const uint8_t *tjb,
    const plan7_bias_candidate *candidates,
    const uint32_t *candidate_indices, const uint64_t *dp_offsets,
    size_t candidate_begin, size_t tile_count, uint64_t tile_dp_begin,
    uint8_t *dp_storage, plan7_bias_ssv_input *msv_results) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (tile_candidate >= tile_count) return;
  const size_t candidate_position = candidate_begin + tile_candidate;
  const size_t candidate = candidate_indices == nullptr
      ? candidate_position
      : static_cast<size_t>(candidate_indices[candidate_position]);
  if (msv_results[candidate].status != PLAN7_SSV_ENORESULT) return;

  const plan7_bias_candidate mapping = candidates[candidate];
  const plan7_ssv_f1_profile profile =
      msv_profiles[mapping.profile_index];
  const VitProfile vit_profile = vit_profiles[mapping.profile_index];
  const int q_count = static_cast<int>(vit_profile.q);
  const uint64_t sequence_start = sequence_offsets[mapping.sequence_index];
  const int sequence_length = static_cast<int>(
      sequence_offsets[mapping.sequence_index + 1] - sequence_start);
  const unsigned candidate_tjb =
      tjb[profile.tjb_offset + mapping.sequence_index];
  uint8_t *dp =
      dp_storage + dp_offsets[candidate_position] - tile_dp_begin;
  for (int q = 0; q < q_count; ++q)
    dp[q * kWarpSize + lane] = 0;

  const int signed_tjb = candidate_tjb < 128
      ? static_cast<int>(candidate_tjb)
      : static_cast<int>(candidate_tjb) - 256;
  const int signed_tbm = profile.profile.tbm < 128
      ? static_cast<int>(profile.profile.tbm)
      : static_cast<int>(profile.profile.tbm) - 256;
  const unsigned tjbm =
      static_cast<unsigned>(signed_tjb + signed_tbm) & 255U;
  unsigned xJ = 0;
  unsigned xB = sat_sub_u8(profile.profile.base, tjbm);

  for (int i = 0; i < sequence_length; ++i) {
    const unsigned residue = residues[sequence_start + i];
    unsigned xE = 0;
    unsigned mpv = previous_lane(
        static_cast<unsigned>(dp[(q_count - 1) * kWarpSize + lane]),
        lane, 0U);
    for (int q = 0; q < q_count; ++q) {
      unsigned score = max(mpv, xB);
      score = sat_add_u8(score, profile.profile.bias);
      const int model_position = q + q_count * lane;
      unsigned cost = UINT8_MAX;
      if (model_position < profile.profile.model_length) {
        if (vit_profile.rbv_offset != UINT64_MAX) {
          cost = exact_rbv[
              vit_profile.rbv_offset +
              static_cast<uint64_t>(model_position) * 29 + residue];
        } else if (residue != 20 && residue != 27 && residue != 28) {
          const unsigned raw = compact_scores[
              profile.profile.score_offset +
              static_cast<uint64_t>(model_position) *
                  profile.profile.score_stride + residue];
          const int signed_score = raw < 128 ? static_cast<int>(raw)
                                             : static_cast<int>(raw) - 256;
          const int decoded =
              signed_score + static_cast<int>(profile.profile.bias);
          cost = static_cast<unsigned>(
              decoded < 0 ? 0 : (decoded > 255 ? 255 : decoded));
        }
      }
      score = sat_sub_u8(score, cost);
      xE = max(xE, score);
      const unsigned old = dp[q * kWarpSize + lane];
      dp[q * kWarpSize + lane] = static_cast<uint8_t>(score);
      mpv = old;
    }
    xE = warp_max(xE);
    if (sat_add_u8(xE, profile.profile.bias) == UINT8_MAX) {
      if (lane == 0)
        msv_results[candidate] = {0, PLAN7_SSV_ERANGE, 0};
      return;
    }
    xE = sat_sub_u8(xE, profile.profile.tec);
    xJ = max(xJ, xE);
    xB = sat_sub_u8(max(static_cast<unsigned>(profile.profile.base), xJ),
                    tjbm);
  }
  if (lane == 0) {
    const int numerator = static_cast<int>(xJ) -
                          static_cast<int>(candidate_tjb) -
                          static_cast<int>(profile.profile.base);
    msv_results[candidate] = {
      static_cast<int16_t>(numerator), PLAN7_SSV_OK, 0
    };
  }
}

__device__ __forceinline__ uint32_t pack_u8x4(const unsigned values[4]) {
  return (values[0] & UINT32_C(0xff)) |
         ((values[1] & UINT32_C(0xff)) << 8) |
         ((values[2] & UINT32_C(0xff)) << 16) |
         ((values[3] & UINT32_C(0xff)) << 24);
}

__device__ __forceinline__ uint32_t warp_max_u8x4(uint32_t value) {
  for (int delta = 16; delta != 0; delta >>= 1)
    value = __vmaxu4(value,
                     __shfl_xor_sync(UINT32_MAX, value, delta));
  return value;
}

/* Execute four independent, same-profile/same-length MSV recurrences in the
 * four bytes of each uint32 lane. The host planner guarantees the grouping;
 * the kernel rechecks it and leaves ENORESULT intact on any disagreement so
 * downstream processing conservatively falls back to the CPU. */
__global__ void full_msv_packed_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const uint8_t *compact_scores, const uint8_t *exact_rbv,
    const plan7_ssv_f1_profile *msv_profiles,
    const VitProfile *vit_profiles, const uint8_t *tjb,
    const plan7_bias_candidate *candidates,
    const uint32_t *candidate_indices, const uint64_t *dp_offsets,
    size_t group_begin, size_t tile_count, uint64_t tile_dp_begin,
    uint8_t *dp_storage, plan7_bias_ssv_input *msv_results) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const size_t tile_group =
      static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (tile_group >= tile_count) return;
  const size_t group = group_begin + tile_group;

  size_t candidate_index[4];
  plan7_bias_candidate mapping[4];
  uint64_t sequence_start[4];
  unsigned candidate_tjb[4];
#pragma unroll
  for (int packed_lane = 0; packed_lane < 4; ++packed_lane) {
    candidate_index[packed_lane] = static_cast<size_t>(
        candidate_indices[group * 4 + packed_lane]);
    if (msv_results[candidate_index[packed_lane]].status !=
        PLAN7_SSV_ENORESULT)
      return;
    mapping[packed_lane] = candidates[candidate_index[packed_lane]];
    sequence_start[packed_lane] =
        sequence_offsets[mapping[packed_lane].sequence_index];
    candidate_tjb[packed_lane] =
        tjb[msv_profiles[mapping[packed_lane].profile_index].tjb_offset +
            mapping[packed_lane].sequence_index];
  }
  const plan7_ssv_f1_profile profile =
      msv_profiles[mapping[0].profile_index];
  const VitProfile vit_profile = vit_profiles[mapping[0].profile_index];
  const int sequence_length = static_cast<int>(
      sequence_offsets[mapping[0].sequence_index + 1] - sequence_start[0]);
#pragma unroll
  for (int packed_lane = 1; packed_lane < 4; ++packed_lane) {
    const int packed_length = static_cast<int>(
        sequence_offsets[mapping[packed_lane].sequence_index + 1] -
        sequence_start[packed_lane]);
    if (mapping[packed_lane].profile_index != mapping[0].profile_index ||
        packed_length != sequence_length)
      return;
  }

  const int q_count = static_cast<int>(vit_profile.q);
  uint32_t *dp = reinterpret_cast<uint32_t *>(
      dp_storage + dp_offsets[group] - tile_dp_begin);
  for (int q = 0; q < q_count; ++q)
    dp[q * kWarpSize + lane] = 0;

  unsigned tjbm_bytes[4];
#pragma unroll
  for (int packed_lane = 0; packed_lane < 4; ++packed_lane) {
    const int signed_tjb = candidate_tjb[packed_lane] < 128
        ? static_cast<int>(candidate_tjb[packed_lane])
        : static_cast<int>(candidate_tjb[packed_lane]) - 256;
    const int signed_tbm = profile.profile.tbm < 128
        ? static_cast<int>(profile.profile.tbm)
        : static_cast<int>(profile.profile.tbm) - 256;
    tjbm_bytes[packed_lane] =
        static_cast<unsigned>(signed_tjb + signed_tbm) & UINT32_C(0xff);
  }
  const uint32_t packed_tjbm = pack_u8x4(tjbm_bytes);
  const uint32_t packed_bias =
      static_cast<uint32_t>(profile.profile.bias) * UINT32_C(0x01010101);
  const uint32_t packed_base =
      static_cast<uint32_t>(profile.profile.base) * UINT32_C(0x01010101);
  const uint32_t packed_tec =
      static_cast<uint32_t>(profile.profile.tec) * UINT32_C(0x01010101);
  uint32_t xJ = 0;
  uint32_t xB = __vsubus4(packed_base, packed_tjbm);
  unsigned active_mask = UINT32_C(0x0f);

  for (int i = 0; i < sequence_length; ++i) {
    unsigned residue[4];
#pragma unroll
    for (int packed_lane = 0; packed_lane < 4; ++packed_lane)
      residue[packed_lane] =
          residues[sequence_start[packed_lane] + static_cast<uint64_t>(i)];
    uint32_t xE = 0;
    uint32_t mpv = previous_lane(
        dp[(q_count - 1) * kWarpSize + lane], lane, UINT32_C(0));
    for (int q = 0; q < q_count; ++q) {
      uint32_t score = __vmaxu4(mpv, xB);
      score = __vaddus4(score, packed_bias);
      const int model_position = q + q_count * lane;
      unsigned cost_bytes[4] = {
          UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX};
      if (model_position < profile.profile.model_length) {
#pragma unroll
        for (int packed_lane = 0; packed_lane < 4; ++packed_lane) {
          if (vit_profile.rbv_offset != UINT64_MAX) {
            cost_bytes[packed_lane] = exact_rbv[
                vit_profile.rbv_offset +
                static_cast<uint64_t>(model_position) * 29 +
                residue[packed_lane]];
          } else if (residue[packed_lane] != 20 &&
                     residue[packed_lane] != 27 &&
                     residue[packed_lane] != 28) {
            const unsigned raw = compact_scores[
                profile.profile.score_offset +
                static_cast<uint64_t>(model_position) *
                    profile.profile.score_stride + residue[packed_lane]];
            const int signed_score = raw < 128 ? static_cast<int>(raw)
                                               : static_cast<int>(raw) - 256;
            const int decoded =
                signed_score + static_cast<int>(profile.profile.bias);
            cost_bytes[packed_lane] = static_cast<unsigned>(
                decoded < 0 ? 0 : (decoded > 255 ? 255 : decoded));
          }
        }
      }
      score = __vsubus4(score, pack_u8x4(cost_bytes));
      xE = __vmaxu4(xE, score);
      const uint32_t old = dp[q * kWarpSize + lane];
      dp[q * kWarpSize + lane] = score;
      mpv = old;
    }
    xE = warp_max_u8x4(xE);
    const uint32_t saturated = __vaddus4(xE, packed_bias);
#pragma unroll
    for (int packed_lane = 0; packed_lane < 4; ++packed_lane) {
      const unsigned bit = 1U << packed_lane;
      if ((active_mask & bit) != 0 &&
          ((saturated >> (packed_lane * 8)) & UINT32_C(0xff)) == UINT8_MAX) {
        if (lane == 0)
          msv_results[candidate_index[packed_lane]] = {
              0, PLAN7_SSV_ERANGE, 0};
        active_mask &= ~bit;
      }
    }
    if (active_mask == 0) return;
    xE = __vsubus4(xE, packed_tec);
    xJ = __vmaxu4(xJ, xE);
    xB = __vsubus4(__vmaxu4(packed_base, xJ), packed_tjbm);
  }
  if (lane == 0) {
#pragma unroll
    for (int packed_lane = 0; packed_lane < 4; ++packed_lane) {
      if ((active_mask & (1U << packed_lane)) == 0) continue;
      const int numerator =
          static_cast<int>((xJ >> (packed_lane * 8)) & UINT32_C(0xff)) -
          static_cast<int>(candidate_tjb[packed_lane]) -
          static_cast<int>(profile.profile.base);
      msv_results[candidate_index[packed_lane]] = {
          static_cast<int16_t>(numerator), PLAN7_SSV_OK, 0};
    }
  }
}

struct FullMsvCandidatePredicate {
  const plan7_bias_ssv_input *results;

  __host__ __device__ bool operator()(const uint32_t candidate) const {
    return results[candidate].status == PLAN7_SSV_ENORESULT;
  }
};

__global__ void postfilter_full_msv_execution_facts_kernel(
    const plan7_bias_ssv_input *msv_results, size_t candidate_count,
    uint16_t *reason_facts) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  if (msv_results[candidate].status == PLAN7_SSV_ENORESULT)
    reason_facts[candidate] |= PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED;
}

__global__ void prepare_bias_inputs_kernel(
    const float *null_scores, const plan7_ssv_f1_profile *profiles,
    const plan7_bias_candidate *candidates,
    const plan7_bias_ssv_input *msv_results, size_t candidate_count,
    uint8_t *states, plan7_bias_ssv_input *bias_inputs) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const plan7_bias_candidate mapping = candidates[candidate];
  const plan7_bias_ssv_input msv = msv_results[candidate];
  CandidateState state = kCandidateCpu;
  plan7_bias_ssv_input bias = {0, PLAN7_SSV_ENORESULT, 0};
  if (msv.status == PLAN7_SSV_ERANGE) {
    state = kCandidateMsvRange;
    bias = {INT16_MAX, PLAN7_SSV_OK, 0};
  } else if (msv.status == PLAN7_SSV_OK) {
    const plan7_ssv_f1_profile profile = profiles[mapping.profile_index];
    if (profile.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_REJECT ||
        (profile.cutoff_mode == PLAN7_F1_CUTOFF_SCORE &&
         !raw_f1_survives(msv, null_scores[mapping.sequence_index],
                          profile))) {
      state = kCandidateRawReject;
    } else if (raw_f1_survives(
                   msv, null_scores[mapping.sequence_index], profile)) {
      state = kCandidateFinite;
      bias = msv;
    }
  }
  states[candidate] = static_cast<uint8_t>(state);
  bias_inputs[candidate] = bias;
}

__global__ void viterbi_kernel(
    const uint8_t *__restrict__ residues,
    const uint64_t *__restrict__ sequence_offsets,
    const VitProfile *__restrict__ profiles,
    const int16_t *__restrict__ emissions,
    const int16_t *__restrict__ transitions,
    const plan7_bias_candidate *__restrict__ candidates,
    const uint8_t *__restrict__ states,
    const plan7_bias_result *__restrict__ bias_results,
    const VitLengthTransitions *__restrict__ length_transitions,
    const uint64_t *__restrict__ dp_offsets,
    size_t candidate_begin, size_t tile_count,
    uint64_t tile_dp_begin, int skip_bias_reject_viterbi,
    int16_t *__restrict__ dp_storage,
    VitResult *__restrict__ results) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (tile_candidate >= tile_count) return;
  const size_t candidate = candidate_begin + tile_candidate;
  if (states[candidate] != kCandidateFinite ||
      bias_results[candidate].action == PLAN7_BIAS_CPU_REQUIRED ||
      (skip_bias_reject_viterbi &&
       bias_results[candidate].action == PLAN7_BIAS_DEFINITE_REJECT))
    return;

  const plan7_bias_candidate mapping = candidates[candidate];
  const VitProfile profile = profiles[mapping.profile_index];
  const int q_count = static_cast<int>(profile.q);
  const uint64_t sequence_start = sequence_offsets[mapping.sequence_index];
  const int sequence_length = static_cast<int>(
      sequence_offsets[mapping.sequence_index + 1] - sequence_start);
  const VitLengthTransitions moves = length_transitions[candidate];
  int16_t *mmx = dp_storage + dp_offsets[candidate] - tile_dp_begin;
  int16_t *imx = mmx + static_cast<uint64_t>(q_count) * kWarpSize;
  int16_t *dmx = imx + static_cast<uint64_t>(q_count) * kWarpSize;
  for (int q = 0; q < q_count; ++q) {
    mmx[q * kWarpSize + lane] = kNegInf;
    imx[q * kWarpSize + lane] = kNegInf;
    dmx[q * kWarpSize + lane] = kNegInf;
  }

  int xN = profile.base;
  int xB = xN + moves.n_move;
  int xJ = kNegInf;
  int xC = kNegInf;
  for (int i = 0; i < sequence_length; ++i) {
    const unsigned residue = residues[sequence_start + i];
    const uint64_t emission_base = profile.emission_offset +
        static_cast<uint64_t>(residue) * q_count * kWarpSize;
    int dcv = kNegInf;
    int xE = kNegInf;
    int dmax = kNegInf;
    int mpv = previous_lane(
        static_cast<int>(mmx[(q_count - 1) * kWarpSize + lane]), lane,
        kNegInf);
    int ipv = previous_lane(
        static_cast<int>(imx[(q_count - 1) * kWarpSize + lane]), lane,
        kNegInf);
    int dpv = previous_lane(
        static_cast<int>(dmx[(q_count - 1) * kWarpSize + lane]), lane,
        kNegInf);
    for (int q = 0; q < q_count; ++q) {
      const uint64_t transition_base = profile.transition_offset +
          static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize + lane;
      int score = sat_add_i16(
          xB, transitions[transition_base + p7O_BM * kWarpSize]);
      score = max(score, sat_add_i16(
          mpv, transitions[transition_base + p7O_MM * kWarpSize]));
      score = max(score, sat_add_i16(
          ipv, transitions[transition_base + p7O_IM * kWarpSize]));
      score = max(score, sat_add_i16(
          dpv, transitions[transition_base + p7O_DM * kWarpSize]));
      score = sat_add_i16(
          score, emissions[emission_base + q * kWarpSize + lane]);
      xE = max(xE, score);
      const int old_m = mmx[q * kWarpSize + lane];
      const int old_i = imx[q * kWarpSize + lane];
      const int old_d = dmx[q * kWarpSize + lane];
      mmx[q * kWarpSize + lane] = static_cast<int16_t>(score);
      dmx[q * kWarpSize + lane] = static_cast<int16_t>(dcv);
      dcv = sat_add_i16(
          score, transitions[transition_base + p7O_MD * kWarpSize]);
      dmax = max(dmax, dcv);
      int insert = sat_add_i16(
          old_m, transitions[transition_base + p7O_MI * kWarpSize]);
      insert = max(insert, sat_add_i16(
          old_i, transitions[transition_base + p7O_II * kWarpSize]));
      imx[q * kWarpSize + lane] = static_cast<int16_t>(insert);
      mpv = old_m;
      ipv = old_i;
      dpv = old_d;
    }
    xE = warp_max(xE);
    if (xE >= 32767) {
      if (lane == 0)
        results[candidate] = {
          PLAN7_SSV_ERANGE, UINT32_C(0x7f800000), INT32_MAX
        };
      return;
    }
    xN += profile.n_loop;
    xC = max(xC + profile.c_loop, xE + profile.e_move);
    xJ = max(xJ + profile.j_loop, xE + profile.e_loop);
    xB = max(xJ + moves.j_move, xN + moves.n_move);

    dmax = warp_max(dmax);
    if (dmax + profile.ddbound > xB) {
      dcv = previous_lane(dcv, lane, kNegInf);
      for (int q = 0; q < q_count; ++q) {
        const uint64_t transition_base = profile.transition_offset +
            static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize + lane;
        const int updated = max(
            dcv, static_cast<int>(dmx[q * kWarpSize + lane]));
        dmx[q * kWarpSize + lane] = static_cast<int16_t>(updated);
        dcv = sat_add_i16(
            updated, transitions[transition_base + p7O_DD * kWarpSize]);
      }
      while (true) {
        dcv = previous_lane(dcv, lane, kNegInf);
        bool complete = true;
        for (int q = 0; q < q_count; ++q) {
          const int current = dmx[q * kWarpSize + lane];
          if (__any_sync(UINT32_MAX, dcv > current) == 0) {
            complete = false;
            break;
          }
          const int updated = max(dcv, current);
          dmx[q * kWarpSize + lane] = static_cast<int16_t>(updated);
          const uint64_t transition_base = profile.transition_offset +
              static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize + lane;
          dcv = sat_add_i16(
              updated, transitions[transition_base + p7O_DD * kWarpSize]);
        }
        if (!complete) break;
      }
    } else {
      dcv = previous_lane(dcv, lane, kNegInf);
      dmx[lane] = static_cast<int16_t>(dcv);
    }
  }

  if (lane == 0) {
    FloatBits score{};
    int numerator = INT32_MIN;
    if (xC > kNegInf) {
      numerator = xC + moves.c_move - profile.base;
      float value = __fadd_rn(__int2float_rn(xC),
                              __int2float_rn(moves.c_move));
      value = __fsub_rn(value, __int2float_rn(profile.base));
      value = __fdiv_rn(value, profile.scale);
      score.value = __fsub_rn(value, 3.0f);
    } else {
      score.bits = UINT32_C(0xff800000);
    }
    results[candidate] = {PLAN7_SSV_OK, score.bits, numerator};
  }
}

__global__ void merge_results_kernel(
    const plan7_bias_candidate *candidates,
    const plan7_bias_ssv_input *msv_results, const uint8_t *states,
    const plan7_bias_result *bias_results, const VitResult *vit_results,
    size_t candidate_count, int skip_bias_reject_viterbi,
    plan7_postfilter_result *results) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const plan7_bias_ssv_input msv = msv_results[candidate];
  plan7_postfilter_result output = {
    candidates[candidate].sequence_index, NAN, msv.numerator,
    msv.status, PLAN7_BIAS_CPU_REQUIRED, NAN
  };
  const CandidateState state =
      static_cast<CandidateState>(states[candidate]);
  if (state == kCandidateRawReject && msv.status == PLAN7_SSV_OK) {
    output.action = PLAN7_BIAS_DEFINITE_REJECT;
  } else if (skip_bias_reject_viterbi &&
             state == kCandidateFinite &&
             msv.status == PLAN7_SSV_OK &&
             isfinite(bias_results[candidate].filtersc) &&
             bias_results[candidate].action == PLAN7_BIAS_DEFINITE_REJECT) {
    /* A direct sparse sealed request authenticates bias filtering and drops
     * this terminal row after packet construction.  +infinity keeps the
     * transient record structurally valid without fabricating a finite
     * Viterbi score; no general/reusable entry point enables this path. */
    output.filtersc = bias_results[candidate].filtersc;
    output.action = PLAN7_BIAS_DEFINITE_REJECT;
    output.vfsc = __int_as_float(0x7f800000);
  } else if (state == kCandidateFinite &&
             msv.status == PLAN7_SSV_OK &&
             isfinite(bias_results[candidate].filtersc) &&
             (bias_results[candidate].action == PLAN7_BIAS_DEFINITE_REJECT ||
              bias_results[candidate].action == PLAN7_BIAS_DEFINITE_PASS)) {
    FloatBits vfsc{};
    vfsc.bits = vit_results[candidate].score_bits;
    if (vit_results[candidate].status == PLAN7_SSV_ERANGE) {
      output.filtersc = bias_results[candidate].filtersc;
      output.action = bias_results[candidate].action;
      output.vfsc = __int_as_float(0x7f800000);
    } else if (vit_results[candidate].status == PLAN7_SSV_OK &&
               isfinite(vfsc.value)) {
      output.filtersc = bias_results[candidate].filtersc;
      output.action = bias_results[candidate].action;
      output.vfsc = vfsc.value;
    }
  }
  results[candidate] = output;
}

__global__ void postfilter_reason_facts_kernel(
    const plan7_bias_profile *profiles,
    const plan7_bias_candidate *candidates,
    const plan7_bias_ssv_input *msv_results,
    const uint8_t *states,
    const plan7_bias_ssv_input *bias_inputs,
    const plan7_bias_result *bias_results,
    const VitResult *vit_results,
    const plan7_postfilter_result *results,
    size_t candidate_count, int skip_bias_reject_viterbi,
    uint16_t *reason_facts) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;

  const CandidateState state = static_cast<CandidateState>(states[candidate]);
  const plan7_bias_result bias = bias_results[candidate];
  const plan7_postfilter_result output = results[candidate];
  uint16_t facts = reason_facts[candidate];
  if (state == kCandidateRawReject)
    facts |= PLAN7_POSTFILTER_REASON_RAW_F1_REJECT;
  else if (state == kCandidateMsvRange)
    facts |= PLAN7_POSTFILTER_REASON_MSV_RANGE_STATE;
  else if (state == kCandidateCpu)
    facts |= PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU;

  if (state == kCandidateFinite) {
    const plan7_bias_ssv_input bias_input = bias_inputs[candidate];
    if (bias_input.status != PLAN7_SSV_OK) {
      facts |= PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO;
    } else if (!isfinite(bias.filtersc)) {
      /* bias_filter_score writes filtersc only after its complete finite
       * recurrence, so a nonfinite retained value is its exact false path. */
      facts |= PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED;
    } else {
      const plan7_bias_candidate mapping = candidates[candidate];
      const plan7_bias_profile profile = profiles[mapping.profile_index];
      float usc = __int2float_rn(static_cast<int>(bias_input.numerator));
      usc = __fdiv_rn(usc, profile.scale);
      usc = __fsub_rn(usc, 3.0f);
      const float bit_score = __double2float_rn(__ddiv_rn(
          static_cast<double>(__fsub_rn(usc, bias.filtersc)), kLog2));
      if (!isfinite(usc) || !isfinite(bit_score))
        facts |= PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE;
      else if (bias.action == PLAN7_BIAS_CPU_REQUIRED)
        facts |= PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED;
    }

    if (bias.action != PLAN7_BIAS_CPU_REQUIRED &&
        !(skip_bias_reject_viterbi &&
          bias.action == PLAN7_BIAS_DEFINITE_REJECT)) {
      facts |= PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED;
      if (vit_results[candidate].status == PLAN7_SSV_ERANGE)
        facts |= PLAN7_POSTFILTER_REASON_VITERBI_ERANGE;
      else if (vit_results[candidate].status != PLAN7_SSV_OK)
        facts |= PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS;
    }
  }

  if (output.action == PLAN7_BIAS_CPU_REQUIRED) {
    facts |= PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED;
    constexpr uint16_t explained =
        PLAN7_POSTFILTER_REASON_MSV_RANGE_STATE |
        PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU |
        PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO |
        PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED |
        PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE |
        PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED |
        PLAN7_POSTFILTER_REASON_VITERBI_ERANGE |
        PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS;
    if ((facts & explained) == 0)
      facts |= PLAN7_POSTFILTER_REASON_OTHER_CPU_REQUIRED;
  } else if (output.action == PLAN7_BIAS_DEFINITE_REJECT) {
    facts |= PLAN7_POSTFILTER_REASON_FINAL_REJECT;
  } else if (output.action == PLAN7_BIAS_DEFINITE_PASS) {
    facts |= PLAN7_POSTFILTER_REASON_FINAL_PASS;
  }
  reason_facts[candidate] = facts;
}

}  // namespace

namespace {

class PackWorkerPool {
 public:
  using Task = void (*)(void *, size_t) noexcept;

  explicit PackWorkerPool(size_t worker_count) {
    try {
      workers_.reserve(worker_count);
      for (size_t i = 0; i < worker_count; ++i)
        workers_.emplace_back([this]() { worker_loop(); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
      }
      ready_.notify_all();
      for (auto &worker : workers_)
        if (worker.joinable()) worker.join();
      throw;
    }
  }

  PackWorkerPool(const PackWorkerPool &) = delete;
  PackWorkerPool &operator=(const PackWorkerPool &) = delete;

  ~PackWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
  }

  void parallel_for(size_t task_count, void *context, Task task) {
    if (task_count == 0) return;
    if (workers_.empty()) {
      for (size_t task_index = 0; task_index < task_count; ++task_index)
        task(context, task_index);
      parallel_run_count_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [this]() { return !active_; });
    task_count_ = task_count;
    task_context_ = context;
    task_ = task;
    next_task_.store(0, std::memory_order_relaxed);
    finished_workers_ = 0;
    active_ = true;
    ++generation_;
    parallel_run_count_.fetch_add(1, std::memory_order_relaxed);
    ready_.notify_all();
    completed_.wait(lock, [this]() { return !active_; });
  }

  size_t worker_count() const noexcept { return workers_.size(); }

  uint64_t parallel_run_count() const noexcept {
    return parallel_run_count_.load(std::memory_order_relaxed);
  }

 private:
  void worker_loop() noexcept {
    uint64_t observed_generation = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this, observed_generation]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) return;
      observed_generation = generation_;
      const size_t task_count = task_count_;
      void *context = task_context_;
      Task task = task_;
      lock.unlock();
      for (;;) {
        const size_t task_index =
            next_task_.fetch_add(1, std::memory_order_relaxed);
        if (task_index >= task_count) break;
        task(context, task_index);
      }
      lock.lock();
      ++finished_workers_;
      if (finished_workers_ == workers_.size()) {
        active_ = false;
        completed_.notify_all();
      }
    }
  }

  std::vector<std::thread> workers_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable completed_;
  std::atomic<size_t> next_task_{0};
  std::atomic<uint64_t> parallel_run_count_{0};
  size_t task_count_ = 0;
  void *task_context_ = nullptr;
  Task task_ = nullptr;
  size_t finished_workers_ = 0;
  uint64_t generation_ = 0;
  bool active_ = false;
  bool stopping_ = false;
};

struct HostProfilePack {
  int alphabet_size = 29;
  std::vector<VitProfile> vit_profiles;
  std::vector<uint8_t> ssv_scores;
  std::vector<uint8_t> exact_rbv;
  std::vector<int16_t> emissions;
  std::vector<int16_t> transitions;
  std::vector<plan7_ssv_profile> ssv_profiles;
  std::vector<float> m_mu;
  std::vector<float> m_lambda;
  std::vector<float> v_mu;
  std::vector<float> v_lambda;
  std::vector<plan7_bias_profile> bias_templates;
  std::vector<plan7_forward_snapshot_profile> forward_profiles;
  std::vector<float> forward_emissions;
  std::vector<float> forward_transitions;
  std::vector<uintptr_t> identity_tokens;
};

uint64_t host_pack_bytes(const HostProfilePack &pack) {
  const uint64_t counts[] = {
      static_cast<uint64_t>(pack.vit_profiles.size()) * sizeof(VitProfile),
      static_cast<uint64_t>(pack.ssv_scores.size()),
      static_cast<uint64_t>(pack.exact_rbv.size()),
      static_cast<uint64_t>(pack.emissions.size()) * sizeof(int16_t),
      static_cast<uint64_t>(pack.transitions.size()) * sizeof(int16_t),
      static_cast<uint64_t>(pack.ssv_profiles.size()) *
          sizeof(plan7_ssv_profile),
      static_cast<uint64_t>(pack.m_mu.size()) * sizeof(float),
      static_cast<uint64_t>(pack.m_lambda.size()) * sizeof(float),
      static_cast<uint64_t>(pack.v_mu.size()) * sizeof(float),
      static_cast<uint64_t>(pack.v_lambda.size()) * sizeof(float),
      static_cast<uint64_t>(pack.bias_templates.size()) *
          sizeof(plan7_bias_profile),
      static_cast<uint64_t>(pack.forward_profiles.size()) *
          sizeof(plan7_forward_snapshot_profile),
      static_cast<uint64_t>(pack.forward_emissions.size()) * sizeof(float),
      static_cast<uint64_t>(pack.forward_transitions.size()) * sizeof(float),
      static_cast<uint64_t>(pack.identity_tokens.size()) * sizeof(uintptr_t)};
  uint64_t total = 0;
  for (const uint64_t count : counts) {
    if (count > UINT64_MAX - total) return UINT64_MAX;
    total += count;
  }
  return total;
}

struct SnapshotTask {
  const uintptr_t *profile_pointers;
  HostProfilePack *pack;
};

void snapshot_profile_task(void *opaque, size_t profile_index) noexcept {
  auto *task = static_cast<SnapshotTask *>(opaque);
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      task->profile_pointers[profile_index]);
  HostProfilePack &pack = *task->pack;
  const VitProfile descriptor = pack.vit_profiles[profile_index];
  const int source_q = p7O_NQW(source->M);
  const int source_qb = std::max(2, (source->M + 15) / 16);
  const int q_count = static_cast<int>(descriptor.q);
  for (int q = 0; q < q_count; ++q) {
    for (int lane = 0; lane < kWarpSize; ++lane) {
      const int model_position = q + q_count * lane + 1;
      if (model_position > source->M) continue;
      const int source_stripe = (model_position - 1) % source_q;
      const int source_lane = (model_position - 1) / source_q;
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const auto *source_bytes = reinterpret_cast<const uint8_t *>(
            source->sbv[residue] + (model_position - 1) % source_qb);
        pack.ssv_scores[
            descriptor.ssv_offset +
            static_cast<uint64_t>(model_position - 1) *
                source->abc->Kp + residue] =
            source_bytes[(model_position - 1) / source_qb];
        if (descriptor.rbv_offset != UINT64_MAX) {
          const auto *source_rbv = reinterpret_cast<const uint8_t *>(
              source->rbv[residue] + (model_position - 1) % source_qb);
          pack.exact_rbv[
              descriptor.rbv_offset +
              static_cast<uint64_t>(model_position - 1) *
                  source->abc->Kp + residue] =
              source_rbv[(model_position - 1) / source_qb];
        }
        const uint64_t destination = descriptor.emission_offset +
            (static_cast<uint64_t>(residue) * q_count + q) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->rwv[residue] + source_stripe);
        pack.emissions[destination] = source_words[source_lane];
      }
      for (int transition = p7O_BM; transition <= p7O_II; ++transition) {
        const uint64_t destination = descriptor.transition_offset +
            (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->twv + source_stripe * 7 + transition);
        pack.transitions[destination] = source_words[source_lane];
      }
      const uint64_t dd_destination = descriptor.transition_offset +
          (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) *
              kWarpSize + lane;
      const auto *dd_words = reinterpret_cast<const int16_t *>(
          source->twv + 7 * source_q + source_stripe);
      pack.transitions[dd_destination] = dd_words[source_lane];
    }
  }

  if (pack.forward_profiles.empty()) return;
  const plan7_forward_snapshot_profile forward =
      pack.forward_profiles[profile_index];
  const int forward_q = static_cast<int>(forward.q);
  for (int residue = 0; residue < source->abc->Kp; ++residue) {
    const auto *source_values =
        reinterpret_cast<const float *>(source->rfv[residue]);
    float *destination = pack.forward_emissions.data() +
        forward.emission_offset +
        static_cast<uint64_t>(residue) * forward_q * kForwardSubwarp;
    std::memcpy(destination, source_values,
                static_cast<size_t>(forward_q) * kForwardSubwarp *
                    sizeof(float));
  }
  for (int q = 0; q < forward_q; ++q) {
    for (int transition = p7O_BM; transition <= p7O_II; ++transition) {
      const auto *source_values = reinterpret_cast<const float *>(
          source->tfv + q * 7 + transition);
      float *destination = pack.forward_transitions.data() +
          forward.transition_offset +
          (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
              kForwardSubwarp;
      std::memcpy(destination, source_values,
                  kForwardSubwarp * sizeof(float));
    }
    const auto *source_dd = reinterpret_cast<const float *>(
        source->tfv + 7 * forward_q + q);
    float *destination_dd = pack.forward_transitions.data() +
        forward.transition_offset +
        (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) *
            kForwardSubwarp;
    std::memcpy(destination_dd, source_dd,
                kForwardSubwarp * sizeof(float));
  }
}

int snapshot_profiles(const uintptr_t *profile_pointers, size_t profile_count,
                      const float *background, PackWorkerPool *workers,
                      HostProfilePack *output, char *error,
                      size_t error_size) {
  if (output == nullptr || workers == nullptr ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid host profile snapshot arguments");
    return -1;
  }
  HostProfilePack pack;
  uint64_t emission_total = 0;
  uint64_t transition_total = 0;
  uint64_t ssv_total = 0;
  uint64_t exact_rbv_total = 0;
  uint64_t forward_emission_total = 0;
  uint64_t forward_transition_total = 0;
  try {
    pack.vit_profiles.resize(profile_count);
    pack.ssv_profiles.resize(profile_count);
    pack.m_mu.resize(profile_count);
    pack.m_lambda.resize(profile_count);
    if (background != nullptr) {
      pack.v_mu.resize(profile_count);
      pack.v_lambda.resize(profile_count);
      pack.bias_templates.resize(profile_count);
      pack.forward_profiles.resize(profile_count);
    }
  } catch (...) {
    set_error(error, error_size, "host profile descriptor allocation failed");
    return -1;
  }

  for (size_t p = 0; p < profile_count; ++p) {
    const auto *source = reinterpret_cast<const P7_OPROFILE *>(
        profile_pointers[p]);
    if (!valid_profile_storage(source) || !valid_viterbi_specials(source) ||
        (background != nullptr &&
         (!valid_forward_storage(source) || !valid_forward_specials(source) ||
          !valid_forward_probability_rows(source))) ||
        source->abc == nullptr || source->abc->Kp != 29 ||
        !isfinite(source->scale_b) || source->scale_b <= 0.0f ||
        !isfinite(source->scale_w) || source->scale_w <= 0.0f ||
        (source->mode != p7_LOCAL && source->mode != p7_UNILOCAL)) {
      set_error(error, error_size, "invalid optimized Viterbi profile");
      return -1;
    }
    const int q = (source->M + kWarpSize - 1) / kWarpSize;
    const uint64_t emission_count =
        static_cast<uint64_t>(source->abc->Kp) * q * kWarpSize;
    const uint64_t ssv_count =
        static_cast<uint64_t>(source->abc->Kp) * source->M;
    const uint64_t transition_count =
        static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize;
    const uint64_t forward_q = background == nullptr
        ? 0 : static_cast<uint64_t>(p7O_NQF(source->M));
    uint64_t forward_emission_count = 0;
    uint64_t forward_transition_count = 0;
    if (background != nullptr &&
        (!checked_multiply(static_cast<uint64_t>(source->abc->Kp),
                           forward_q * kForwardSubwarp,
                           &forward_emission_count) ||
         !checked_multiply(forward_q,
                           p7O_NTRANS * kForwardSubwarp,
                           &forward_transition_count))) {
      set_error(error, error_size, "Forward packed profile size overflow");
      return -1;
    }
    bool needs_exact_rbv = false;
    const int source_qb = std::max(2, (source->M + 15) / 16);
    for (int model_position = 0;
         model_position < source->M && !needs_exact_rbv;
         ++model_position) {
      const int source_stripe = model_position % source_qb;
      const int source_lane = model_position / source_qb;
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const auto *sbv = reinterpret_cast<const uint8_t *>(
            source->sbv[residue] + source_stripe);
        const auto *rbv = reinterpret_cast<const uint8_t *>(
            source->rbv[residue] + source_stripe);
        const unsigned raw = sbv[source_lane];
        const int signed_score = raw < 128 ? static_cast<int>(raw)
                                           : static_cast<int>(raw) - 256;
        const int decoded_value =
            signed_score + static_cast<int>(source->bias_b);
        const unsigned decoded = residue == 20 || residue == 27 ||
                                 residue == 28
            ? UINT8_MAX
            : static_cast<unsigned>(decoded_value < 0 ? 0 :
                                    (decoded_value > 255 ? 255 :
                                     decoded_value));
        if (decoded != rbv[source_lane]) {
          needs_exact_rbv = true;
          break;
        }
      }
    }
    const uint64_t exact_rbv_count = needs_exact_rbv ? ssv_count : 0;
    if (!checked_add(ssv_total, ssv_count, &ssv_total) ||
        !checked_add(exact_rbv_total, exact_rbv_count, &exact_rbv_total) ||
        !checked_add(emission_total, emission_count, &emission_total) ||
        !checked_add(transition_total, transition_count, &transition_total) ||
        !checked_add(forward_emission_total, forward_emission_count,
                     &forward_emission_total) ||
        !checked_add(forward_transition_total, forward_transition_count,
                     &forward_transition_total) ||
        ssv_total > SIZE_MAX || exact_rbv_total > SIZE_MAX ||
        emission_total > SIZE_MAX || transition_total > SIZE_MAX ||
        forward_emission_total > SIZE_MAX ||
        forward_transition_total > SIZE_MAX) {
      set_error(error, error_size, "Viterbi packed profile size overflow");
      return -1;
    }
    VitProfile descriptor{};
    descriptor.ssv_offset = ssv_total - ssv_count;
    descriptor.rbv_offset = needs_exact_rbv
        ? exact_rbv_total - exact_rbv_count
        : UINT64_MAX;
    descriptor.emission_offset = emission_total - emission_count;
    descriptor.transition_offset = transition_total - transition_count;
    descriptor.q = static_cast<uint32_t>(q);
    descriptor.model_length = static_cast<uint32_t>(source->M);
    descriptor.mode = source->mode;
    descriptor.base = source->base_w;
    descriptor.ddbound = source->ddbound_w;
    descriptor.e_move = source->xw[p7O_E][p7O_MOVE];
    descriptor.e_loop = source->xw[p7O_E][p7O_LOOP];
    descriptor.n_loop = source->xw[p7O_N][p7O_LOOP];
    descriptor.j_loop = source->xw[p7O_J][p7O_LOOP];
    descriptor.c_loop = source->xw[p7O_C][p7O_LOOP];
    descriptor.scale = source->scale_w;
    descriptor.nj = source->nj;
    descriptor.msv_tbm = source->tbm_b;
    descriptor.msv_tec = source->tec_b;
    descriptor.msv_base = source->base_b;
    descriptor.msv_bias = source->bias_b;
    descriptor.msv_scale = source->scale_b;
    pack.vit_profiles[p] = descriptor;
    pack.ssv_profiles[p] = {
        descriptor.ssv_offset, ssv_count, 29, source->M,
        source->tbm_b, source->tec_b, source->base_b, source->bias_b,
        source->scale_b};
    pack.m_mu[p] = source->evparam[p7_MMU];
    pack.m_lambda[p] = source->evparam[p7_MLAMBDA];
    if (background != nullptr) {
      pack.v_mu[p] = source->evparam[p7_VMU];
      pack.v_lambda[p] = source->evparam[p7_VLAMBDA];
      pack.forward_profiles[p] = {
        forward_emission_total - forward_emission_count,
        forward_transition_total - forward_transition_count,
        static_cast<uint32_t>(forward_q),
        static_cast<uint32_t>(source->M),
        source->xf[p7O_E][p7O_MOVE],
        source->xf[p7O_E][p7O_LOOP],
        source->evparam[p7_FTAU],
        source->evparam[p7_FLAMBDA],
        source->nj,
        source->mode
      };
    }
    if (background != nullptr &&
        plan7_bias_pack_amino_profile(
            background, source->compo, source->M, source->scale_b,
            PLAN7_BIAS_CUTOFF_INVALID, NAN, &pack.bias_templates[p],
            error, error_size) != 0)
      return -1;
  }

  try {
    pack.ssv_scores.resize(static_cast<size_t>(ssv_total));
    pack.exact_rbv.resize(static_cast<size_t>(exact_rbv_total));
    pack.emissions.assign(static_cast<size_t>(emission_total),
                          static_cast<int16_t>(kNegInf));
    pack.transitions.assign(static_cast<size_t>(transition_total),
                            static_cast<int16_t>(kNegInf));
    pack.forward_emissions.resize(
        static_cast<size_t>(forward_emission_total));
    pack.forward_transitions.resize(
        static_cast<size_t>(forward_transition_total));
  } catch (...) {
    set_error(error, error_size, "Viterbi packed data allocation failed");
    return -1;
  }
  SnapshotTask task{profile_pointers, &pack};
  workers->parallel_for(profile_count, &task, snapshot_profile_task);
  *output = std::move(pack);
  return 0;
}

struct SelectionCopyTask {
  const HostProfilePack *source;
  HostProfilePack *destination;
  const size_t *indices;
};

void copy_selection_profile(void *opaque, size_t output_index) noexcept {
  auto *task = static_cast<SelectionCopyTask *>(opaque);
  const HostProfilePack &source = *task->source;
  HostProfilePack &destination = *task->destination;
  const size_t source_index = task->indices[output_index];
  const VitProfile &from = source.vit_profiles[source_index];
  const VitProfile &to = destination.vit_profiles[output_index];
  const size_t ssv_count =
      static_cast<size_t>(to.model_length) * destination.alphabet_size;
  const size_t emission_count = static_cast<size_t>(to.q) * kWarpSize *
                                destination.alphabet_size;
  const size_t transition_count =
      static_cast<size_t>(to.q) * kWarpSize * p7O_NTRANS;
  std::memcpy(destination.ssv_scores.data() + to.ssv_offset,
              source.ssv_scores.data() + from.ssv_offset, ssv_count);
  if (to.rbv_offset != UINT64_MAX)
    std::memcpy(destination.exact_rbv.data() + to.rbv_offset,
                source.exact_rbv.data() + from.rbv_offset, ssv_count);
  std::memcpy(destination.emissions.data() + to.emission_offset,
              source.emissions.data() + from.emission_offset,
              emission_count * sizeof(int16_t));
  std::memcpy(destination.transitions.data() + to.transition_offset,
              source.transitions.data() + from.transition_offset,
              transition_count * sizeof(int16_t));

  const plan7_forward_snapshot_profile &forward_from =
      source.forward_profiles[source_index];
  const plan7_forward_snapshot_profile &forward_to =
      destination.forward_profiles[output_index];
  const size_t forward_emission_count =
      static_cast<size_t>(forward_to.q) * kForwardSubwarp *
      destination.alphabet_size;
  const size_t forward_transition_count =
      static_cast<size_t>(forward_to.q) * kForwardSubwarp * p7O_NTRANS;
  std::memcpy(
      destination.forward_emissions.data() + forward_to.emission_offset,
      source.forward_emissions.data() + forward_from.emission_offset,
      forward_emission_count * sizeof(float));
  std::memcpy(
      destination.forward_transitions.data() + forward_to.transition_offset,
      source.forward_transitions.data() + forward_from.transition_offset,
      forward_transition_count * sizeof(float));
}

std::atomic<uint64_t> next_profile_session_id{1};
std::atomic<uint64_t> next_profile_identity_token{1};

bool claim_profile_session_id(uint64_t *session_id) {
  uint64_t current = next_profile_session_id.load(std::memory_order_relaxed);
  while (current != 0) {
    const uint64_t next = current == UINT64_MAX ? 0 : current + 1;
    if (next_profile_session_id.compare_exchange_weak(
          current, next, std::memory_order_relaxed,
          std::memory_order_relaxed)) {
      *session_id = current;
      return true;
    }
  }
  return false;
}

bool claim_profile_identity_tokens(size_t token_count,
                                   uint64_t *first_token) {
  if (token_count == 0) {
    *first_token = 0;
    return true;
  }
  const uint64_t count = static_cast<uint64_t>(token_count);
  if (static_cast<size_t>(count) != token_count) return false;
  uint64_t current = next_profile_identity_token.load(
      std::memory_order_relaxed);
  while (current != 0) {
    if (count - 1 > UINT64_MAX - current) return false;
    const uint64_t last = current + count - 1;
    const uint64_t next = last == UINT64_MAX ? 0 : last + 1;
    if (next_profile_identity_token.compare_exchange_weak(
          current, next, std::memory_order_relaxed,
          std::memory_order_relaxed)) {
      *first_token = current;
      return true;
    }
  }
  return false;
}

}  // namespace

struct plan7_viterbi_database {
  int device_ordinal;
  int alphabet_size;
  bool sealed_source;
  std::vector<VitProfile> host_profiles;
  std::vector<uintptr_t> source_profile_pointers;
  std::vector<uintptr_t> source_alphabet_pointers;
  std::vector<uint8_t> host_ssv_scores;
  std::vector<uint8_t> host_exact_rbv;
  std::vector<int16_t> host_emissions;
  std::vector<int16_t> host_transitions;
  std::shared_ptr<const HostProfilePack> sealed_pack;
  VitProfile *device_profiles;
  int16_t *device_emissions;
  int16_t *device_transitions;
  uint8_t *device_exact_rbv;
  size_t emission_count;
  size_t transition_count;
  size_t exact_rbv_count;
};

struct plan7_profile_session {
  uint64_t session_id;
  uint64_t build_worker_count;
  uint64_t build_parallel_run_count;
  uint64_t selection_count;
  size_t profile_count;
  bool chunk_local_pack;
  std::shared_ptr<const HostProfilePack> pack;
  std::vector<uintptr_t> source_profile_pointers;
  std::vector<float> background;
  std::vector<uintptr_t> identity_tokens;
  std::unique_ptr<PackWorkerPool> selection_workers;
  std::mutex operation_mutex;
};

struct plan7_profile_selection {
  uint64_t session_id;
  uint64_t selection_id;
  std::shared_ptr<const HostProfilePack> pack;
};

struct plan7_postfilter_workspace {
  int device_ordinal;
  std::vector<uint64_t> host_msv_offsets;
  std::vector<uint32_t> host_msv_candidate_indices;
  std::vector<uint32_t> host_msv_execution_indices;
  std::vector<uint32_t> host_msv_length_to_class;
  std::vector<uint32_t> host_msv_class_lengths;
  std::vector<std::vector<uint32_t>> host_msv_length_buckets;
  std::vector<uint32_t> host_msv_touched_length_classes;
  const uint64_t *host_msv_length_source;
  size_t host_msv_length_source_count;
  std::vector<uint64_t> host_vit_offsets;
  std::vector<VitLengthTransitions> host_moves;
  std::vector<size_t> msv_tiles;
  std::vector<size_t> vit_tiles;
  uint8_t *device_states;
  plan7_bias_ssv_input *device_bias_inputs;
  plan7_bias_result *device_bias_results;
  VitResult *device_vit_results;
  VitLengthTransitions *device_moves;
  uint64_t *device_msv_offsets;
  uint64_t *device_vit_offsets;
  void *device_dp;
  plan7_postfilter_result *device_results;
  size_t states_capacity;
  size_t bias_inputs_capacity;
  size_t bias_results_capacity;
  size_t vit_results_capacity;
  size_t moves_capacity;
  size_t msv_offsets_capacity;
  size_t vit_offsets_capacity;
  size_t dp_capacity;
  size_t results_capacity;
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
  std::vector<VitLengthTransitions> host_length_transition_table;
  const plan7_bias_candidate *resident_device_candidates;
  const plan7_bias_candidate *resident_host_candidates;
  const plan7_postfilter_result *resident_host_results;
  size_t resident_source_count;
  uint64_t resident_generation;
  bool resident_results_valid;
  bool f2_view_valid;
  uint64_t f2_batch_generation;
  size_t f2_profile_count;
  size_t f2_selected_count;
  uint64_t f2_selected_source_hash;
  std::vector<F2DeviceProfile> host_f2_profiles;
  std::vector<uint32_t> host_f2_selected_sources;
  plan7_postfilter_f2_statistics f2_statistics;
};

namespace {

const std::vector<VitProfile> &database_host_profiles(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->vit_profiles
      : database->host_profiles;
}

bool prepare_target_length_classes(plan7_postfilter_workspace *workspace,
                                   const uint64_t *lengths,
                                   size_t sequence_count,
                                   char *error, size_t error_size) {
  if (workspace->host_msv_length_source == lengths &&
      workspace->host_msv_length_source_count == sequence_count &&
      (sequence_count == 0 ||
       !workspace->host_msv_class_lengths.empty()))
    return true;
  try {
    workspace->host_msv_length_to_class.assign(
        kMaximumTargetLength + 1, UINT32_MAX);
    workspace->host_msv_class_lengths.clear();
    workspace->host_msv_length_buckets.clear();
    for (size_t sequence = 0; sequence < sequence_count; ++sequence) {
      const uint64_t length = lengths[sequence];
      if (length > kMaximumTargetLength) {
        set_error(error, error_size,
                  "post-filter target length is invalid");
        return false;
      }
      uint32_t &length_class =
          workspace->host_msv_length_to_class[length];
      if (length_class == UINT32_MAX) {
        if (workspace->host_msv_class_lengths.size() >= UINT32_MAX) {
          set_error(error, error_size,
                    "post-filter length-class count overflow");
          return false;
        }
        length_class = static_cast<uint32_t>(
            workspace->host_msv_class_lengths.size());
        workspace->host_msv_class_lengths.push_back(
            static_cast<uint32_t>(length));
        workspace->host_msv_length_buckets.emplace_back();
      }
    }
    workspace->host_msv_length_source = lengths;
    workspace->host_msv_length_source_count = sequence_count;
  } catch (...) {
    set_error(error, error_size,
              "post-filter length-class allocation failed");
    return false;
  }
  return true;
}

const std::vector<uint8_t> &database_host_ssv_scores(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->ssv_scores
      : database->host_ssv_scores;
}

const std::vector<uint8_t> &database_host_exact_rbv(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->exact_rbv
      : database->host_exact_rbv;
}

const std::vector<int16_t> &database_host_emissions(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->emissions
      : database->host_emissions;
}

const std::vector<int16_t> &database_host_transitions(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->transitions
      : database->host_transitions;
}

template <typename T>
int grow_workspace_buffer(T **buffer, size_t *capacity, size_t required_bytes,
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
    set_cuda_error(error, error_size, "cudaFree(post-filter workspace)", status);
    return -1;
  }
  *buffer = replacement;
  *capacity = required_bytes;
  ++*growth_count;
  return 0;
}

uint64_t postfilter_workspace_device_bytes(
    const plan7_postfilter_workspace *workspace) {
  if (workspace == nullptr) return 0;
  const size_t capacities[] = {
      workspace->states_capacity,       workspace->bias_inputs_capacity,
      workspace->bias_results_capacity, workspace->vit_results_capacity,
      workspace->moves_capacity,        workspace->msv_offsets_capacity,
      workspace->vit_offsets_capacity,  workspace->dp_capacity,
      workspace->results_capacity};
  uint64_t total = 0;
  for (const size_t capacity : capacities) {
    if (capacity > UINT64_MAX - total) return UINT64_MAX;
    total += static_cast<uint64_t>(capacity);
  }
  return total;
}

int destroy_postfilter_workspace_device(plan7_postfilter_workspace *workspace,
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
#define CUDA_DESTROY_WORKSPACE(pointer)                                        \
  do {                                                                         \
    if (device_ready) {                                                        \
      status = cudaFree(pointer);                                              \
      if (status != cudaSuccess && first_error == cudaSuccess)                \
        first_error = status;                                                  \
    }                                                                          \
  } while (0)
  CUDA_DESTROY_WORKSPACE(workspace->device_results);
  CUDA_DESTROY_WORKSPACE(workspace->device_dp);
  CUDA_DESTROY_WORKSPACE(workspace->device_vit_offsets);
  CUDA_DESTROY_WORKSPACE(workspace->device_msv_offsets);
  CUDA_DESTROY_WORKSPACE(workspace->device_moves);
  CUDA_DESTROY_WORKSPACE(workspace->device_vit_results);
  CUDA_DESTROY_WORKSPACE(workspace->device_bias_results);
  CUDA_DESTROY_WORKSPACE(workspace->device_bias_inputs);
  CUDA_DESTROY_WORKSPACE(workspace->device_states);
#undef CUDA_DESTROY_WORKSPACE
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (status != cudaSuccess && first_error == cudaSuccess)
      first_error = status;
  }
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy post-filter workspace",
                   first_error);
    return -1;
  }
  return 0;
}

}  // namespace

bool live_profile_matches_snapshot(const plan7_viterbi_database *database,
                                   size_t profile_index) {
  if (database->sealed_source) return true;
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      database->source_profile_pointers[profile_index]);
  const VitProfile &descriptor = database_host_profiles(database)[profile_index];
  const auto &host_scores = database_host_ssv_scores(database);
  const auto &host_exact_rbv = database_host_exact_rbv(database);
  const auto &host_emissions = database_host_emissions(database);
  const auto &host_transitions = database_host_transitions(database);
  if (!valid_profile_storage(source) || !valid_viterbi_specials(source) ||
      source->abc->Kp != database->alphabet_size ||
      reinterpret_cast<uintptr_t>(source->abc) !=
          database->source_alphabet_pointers[profile_index] ||
      source->M != static_cast<int>(descriptor.model_length) ||
      source->mode != descriptor.mode ||
      source->tbm_b != descriptor.msv_tbm ||
      source->tec_b != descriptor.msv_tec ||
      source->base_b != descriptor.msv_base ||
      source->bias_b != descriptor.msv_bias ||
      source->scale_b != descriptor.msv_scale ||
      source->base_w != descriptor.base ||
      source->ddbound_w != descriptor.ddbound ||
      source->scale_w != descriptor.scale || source->nj != descriptor.nj ||
      source->xw[p7O_E][p7O_MOVE] != descriptor.e_move ||
      source->xw[p7O_E][p7O_LOOP] != descriptor.e_loop ||
      source->xw[p7O_N][p7O_LOOP] != descriptor.n_loop ||
      source->xw[p7O_J][p7O_LOOP] != descriptor.j_loop ||
      source->xw[p7O_C][p7O_LOOP] != descriptor.c_loop)
    return false;
  for (int residue = 0; residue < source->abc->Kp; ++residue)
    if (source->sbv[residue] == nullptr || source->rbv[residue] == nullptr ||
        source->rwv[residue] == nullptr)
      return false;

  const int source_qw = p7O_NQW(source->M);
  const int source_qb = std::max(2, (source->M + 15) / 16);
  const int q_count = static_cast<int>(descriptor.q);
  for (int q = 0; q < q_count; ++q) {
    for (int lane = 0; lane < kWarpSize; ++lane) {
      const int model_position = q + q_count * lane + 1;
      if (model_position > source->M) continue;
      const int source_word_stripe = (model_position - 1) % source_qw;
      const int source_word_lane = (model_position - 1) / source_qw;
      const int source_byte_stripe = (model_position - 1) % source_qb;
      const int source_byte_lane = (model_position - 1) / source_qb;
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const uint64_t ssv_index = descriptor.ssv_offset +
            static_cast<uint64_t>(model_position - 1) *
                source->abc->Kp + residue;
        const auto *source_sbv = reinterpret_cast<const uint8_t *>(
            source->sbv[residue] + source_byte_stripe);
        const auto *source_rbv = reinterpret_cast<const uint8_t *>(
            source->rbv[residue] + source_byte_stripe);
        if (source_sbv[source_byte_lane] !=
            host_scores[ssv_index])
          return false;
        if (descriptor.rbv_offset != UINT64_MAX) {
          const uint64_t rbv_index = descriptor.rbv_offset +
              static_cast<uint64_t>(model_position - 1) *
                  source->abc->Kp + residue;
          if (source_rbv[source_byte_lane] !=
              host_exact_rbv[rbv_index])
            return false;
        } else {
          const unsigned raw = source_sbv[source_byte_lane];
          const int signed_score = raw < 128 ? static_cast<int>(raw)
                                             : static_cast<int>(raw) - 256;
          const int decoded_value =
              signed_score + static_cast<int>(descriptor.msv_bias);
          const unsigned decoded = residue == 20 || residue == 27 ||
                                   residue == 28
              ? UINT8_MAX
              : static_cast<unsigned>(decoded_value < 0 ? 0 :
                                      (decoded_value > 255 ? 255 :
                                       decoded_value));
          if (source_rbv[source_byte_lane] != decoded) return false;
        }
        const uint64_t emission_index = descriptor.emission_offset +
            (static_cast<uint64_t>(residue) * q_count + q) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->rwv[residue] + source_word_stripe);
        if (source_words[source_word_lane] !=
            host_emissions[emission_index])
          return false;
      }
      for (int transition = p7O_BM; transition <= p7O_II;
           ++transition) {
        const uint64_t transition_index = descriptor.transition_offset +
            (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->twv + source_word_stripe * 7 + transition);
        if (source_words[source_word_lane] !=
            host_transitions[transition_index])
          return false;
      }
      const uint64_t dd_index = descriptor.transition_offset +
          (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) *
              kWarpSize + lane;
      const auto *dd_words = reinterpret_cast<const int16_t *>(
          source->twv + 7 * source_qw + source_word_stripe);
      if (dd_words[source_word_lane] !=
          host_transitions[dd_index])
        return false;
    }
  }
  return true;
}

namespace {

void initialize_viterbi_database(plan7_viterbi_database *database) {
  database->device_ordinal = -1;
  database->alphabet_size = 29;
  database->sealed_source = false;
  database->device_profiles = nullptr;
  database->device_emissions = nullptr;
  database->device_transitions = nullptr;
  database->device_exact_rbv = nullptr;
  database->emission_count = 0;
  database->transition_count = 0;
  database->exact_rbv_count = 0;
}

void release_partial_viterbi_upload(plan7_viterbi_database *database) {
  cudaFree(database->device_exact_rbv);
  cudaFree(database->device_transitions);
  cudaFree(database->device_emissions);
  cudaFree(database->device_profiles);
  database->device_exact_rbv = nullptr;
  database->device_transitions = nullptr;
  database->device_emissions = nullptr;
  database->device_profiles = nullptr;
}

int upload_viterbi_database(plan7_viterbi_database *database, char *error,
                            size_t error_size) {
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  database->device_ordinal = current_device;
  const auto &profiles = database_host_profiles(database);
  const auto &emissions = database_host_emissions(database);
  const auto &transitions = database_host_transitions(database);
  const auto &exact_rbv = database_host_exact_rbv(database);
  database->emission_count = emissions.size();
  database->transition_count = transitions.size();
  database->exact_rbv_count = exact_rbv.size();
  if (profiles.empty()) return 0;

  size_t profile_bytes;
  size_t emission_bytes;
  size_t transition_bytes;
  if (!checked_bytes(profiles.size(), sizeof(VitProfile), &profile_bytes) ||
      !checked_bytes(emissions.size(), sizeof(int16_t), &emission_bytes) ||
      !checked_bytes(transitions.size(), sizeof(int16_t), &transition_bytes)) {
    set_error(error, error_size, "Viterbi device allocation size overflow");
    return -1;
  }
#define CUDA_UPLOAD(call)                                                     \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      release_partial_viterbi_upload(database);                               \
      return -1;                                                              \
    }                                                                         \
  } while (0)
  CUDA_UPLOAD(cudaMalloc(&database->device_profiles, profile_bytes));
  CUDA_UPLOAD(cudaMalloc(&database->device_emissions, emission_bytes));
  CUDA_UPLOAD(cudaMalloc(&database->device_transitions, transition_bytes));
  if (!exact_rbv.empty())
    CUDA_UPLOAD(cudaMalloc(&database->device_exact_rbv, exact_rbv.size()));
  CUDA_UPLOAD(cudaMemcpy(database->device_profiles, profiles.data(),
                         profile_bytes, cudaMemcpyHostToDevice));
  CUDA_UPLOAD(cudaMemcpy(database->device_emissions, emissions.data(),
                         emission_bytes, cudaMemcpyHostToDevice));
  CUDA_UPLOAD(cudaMemcpy(database->device_transitions, transitions.data(),
                         transition_bytes, cudaMemcpyHostToDevice));
  if (!exact_rbv.empty())
    CUDA_UPLOAD(cudaMemcpy(database->device_exact_rbv, exact_rbv.data(),
                           exact_rbv.size(), cudaMemcpyHostToDevice));
#undef CUDA_UPLOAD
  return 0;
}

}  // namespace

extern "C" int plan7_viterbi_database_create(
    const uintptr_t *profile_pointers, size_t profile_count,
    plan7_viterbi_database **database, char *error, size_t error_size) {
  if (database == nullptr || *database != nullptr ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid Viterbi database arguments");
    return -1;
  }
  std::unique_ptr<PackWorkerPool> workers;
  try {
    workers = std::make_unique<PackWorkerPool>(
        std::min<size_t>(16, profile_count));
  } catch (...) {
    set_error(error, error_size, "Viterbi profile worker launch failed");
    return -1;
  }
  HostProfilePack pack;
  if (snapshot_profiles(profile_pointers, profile_count, nullptr,
                        workers.get(), &pack, error, error_size) != 0)
    return -1;
  auto *created = new (std::nothrow) plan7_viterbi_database{};
  if (created == nullptr) {
    set_error(error, error_size, "Viterbi database allocation failed");
    return -1;
  }
  initialize_viterbi_database(created);
  try {
    created->host_profiles = std::move(pack.vit_profiles);
    created->host_ssv_scores = std::move(pack.ssv_scores);
    created->host_exact_rbv = std::move(pack.exact_rbv);
    created->host_emissions = std::move(pack.emissions);
    created->host_transitions = std::move(pack.transitions);
    if (profile_count != 0) {
      created->source_profile_pointers.assign(
          profile_pointers, profile_pointers + profile_count);
      created->source_alphabet_pointers.resize(profile_count);
      for (size_t profile_index = 0; profile_index < profile_count;
           ++profile_index) {
        const auto *source = reinterpret_cast<const P7_OPROFILE *>(
            profile_pointers[profile_index]);
        created->source_alphabet_pointers[profile_index] =
            reinterpret_cast<uintptr_t>(source->abc);
      }
    }
  } catch (...) {
    delete created;
    set_error(error, error_size, "Viterbi descriptor allocation failed");
    return -1;
  }
  if (upload_viterbi_database(created, error, error_size) != 0) {
    delete created;
    return -1;
  }
  *database = created;
  return 0;
}

extern "C" int plan7_viterbi_database_destroy(
    plan7_viterbi_database **database, char *error, size_t error_size) {
  if (database == nullptr) {
    set_error(error, error_size, "Viterbi database handle is null");
    return -1;
  }
  if (*database == nullptr) return 0;
  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool device_ready = true;
  bool restore_device = false;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess &&
      original_device != (*database)->device_ordinal) {
    status = cudaSetDevice((*database)->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice((*database)->device_ordinal);
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
  CUDA_DESTROY((*database)->device_exact_rbv);
  CUDA_DESTROY((*database)->device_transitions);
  CUDA_DESTROY((*database)->device_emissions);
  CUDA_DESTROY((*database)->device_profiles);
#undef CUDA_DESTROY
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (status != cudaSuccess && first_error == cudaSuccess)
      first_error = status;
  }
  delete *database;
  *database = nullptr;
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy Viterbi database", first_error);
    return -1;
  }
  return 0;
}

extern "C" size_t plan7_viterbi_database_profile_count(
    const plan7_viterbi_database *database) {
  return database == nullptr ? 0 : database_host_profiles(database).size();
}

extern "C" int plan7_profile_session_create(
    const uintptr_t *profile_pointers, size_t profile_count,
    const float *background, size_t background_count,
    size_t build_worker_count, size_t selection_worker_count,
    int chunk_local_pack,
    plan7_profile_session **session, char *error, size_t error_size) {
  if (session == nullptr || *session != nullptr || background == nullptr ||
      background_count != 20 ||
      build_worker_count > profile_count ||
      selection_worker_count > profile_count ||
      (chunk_local_pack != 0 && chunk_local_pack != 1) ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid profile session arguments");
    return -1;
  }
  if (plan7_bias_host_environment_attested() != 1) {
    set_error(error, error_size,
              "profile session requires the attested host environment");
    return -1;
  }
  for (size_t residue = 0; residue < background_count; ++residue) {
    if (!isfinite(background[residue]) || background[residue] <= 0.0f) {
      set_error(error, error_size, "invalid profile session background");
      return -1;
    }
  }

  HostProfilePack pack;
  uint64_t build_parallel_run_count = 0;
  if (chunk_local_pack == 0) {
    std::unique_ptr<PackWorkerPool> build_workers;
    try {
      build_workers = std::make_unique<PackWorkerPool>(build_worker_count);
    } catch (...) {
      set_error(error, error_size,
                "profile session build worker launch failed");
      return -1;
    }
    if (snapshot_profiles(profile_pointers, profile_count, background,
                          build_workers.get(), &pack,
                          error, error_size) != 0)
      return -1;
    build_parallel_run_count = build_workers->parallel_run_count();
  }

  uint64_t session_id;
  if (!claim_profile_session_id(&session_id)) {
    set_error(error, error_size, "profile session identity exhausted");
    return -1;
  }
  uint64_t first_identity_token;
  if (!claim_profile_identity_tokens(profile_count, &first_identity_token)) {
    set_error(error, error_size, "profile identity token space exhausted");
    return -1;
  }
  std::vector<uintptr_t> identity_tokens;
  try {
    identity_tokens.resize(profile_count);
    for (size_t profile_index = 0; profile_index < profile_count;
         ++profile_index)
      identity_tokens[profile_index] =
          static_cast<uintptr_t>(first_identity_token + profile_index);
    if (chunk_local_pack == 0)
      pack.identity_tokens = std::move(identity_tokens);
  } catch (...) {
    set_error(error, error_size, "profile session identity allocation failed");
    return -1;
  }

  std::unique_ptr<PackWorkerPool> selection_workers;
  try {
    selection_workers =
        std::make_unique<PackWorkerPool>(selection_worker_count);
  } catch (...) {
    set_error(error, error_size,
              "profile session selection worker launch failed");
    return -1;
  }

  auto *created = new (std::nothrow) plan7_profile_session{};
  if (created == nullptr) {
    set_error(error, error_size, "profile session allocation failed");
    return -1;
  }
  try {
    created->session_id = session_id;
    created->build_worker_count =
        chunk_local_pack == 0 ? build_worker_count : 0;
    created->build_parallel_run_count = build_parallel_run_count;
    created->selection_count = 0;
    created->profile_count = profile_count;
    created->chunk_local_pack = chunk_local_pack != 0;
    if (created->chunk_local_pack) {
      if (profile_count != 0)
        created->source_profile_pointers.assign(
            profile_pointers, profile_pointers + profile_count);
      created->background.assign(background, background + background_count);
      created->identity_tokens = std::move(identity_tokens);
    } else {
      created->pack =
          std::make_shared<const HostProfilePack>(std::move(pack));
    }
    created->selection_workers = std::move(selection_workers);
  } catch (...) {
    delete created;
    set_error(error, error_size, "profile session allocation failed");
    return -1;
  }
  *session = created;
  return 0;
}

extern "C" int plan7_profile_session_destroy(
    plan7_profile_session **session, char *error, size_t error_size) {
  if (session == nullptr) {
    set_error(error, error_size, "profile session handle is null");
    return -1;
  }
  plan7_profile_session *value = *session;
  *session = nullptr;
  delete value;
  return 0;
}

extern "C" int plan7_profile_session_get_statistics(
    const plan7_profile_session *session,
    plan7_profile_session_statistics *statistics,
    char *error, size_t error_size) {
  if (session == nullptr || statistics == nullptr ||
      session->selection_workers == nullptr ||
      (!session->chunk_local_pack && session->pack == nullptr) ||
      (session->chunk_local_pack &&
       (session->source_profile_pointers.size() != session->profile_count ||
        session->background.size() != 20 ||
        session->identity_tokens.size() != session->profile_count))) {
    set_error(error, error_size, "invalid profile session statistics");
    return -1;
  }
  *statistics = {};
  statistics->session_id = session->session_id;
  statistics->profile_count = session->profile_count;
  statistics->worker_count = session->selection_workers->worker_count();
  statistics->build_worker_count = session->build_worker_count;
  statistics->selection_worker_count =
      session->selection_workers->worker_count();
  statistics->selection_count = session->selection_count;
  statistics->build_parallel_run_count =
      session->build_parallel_run_count;
  statistics->selection_parallel_run_count =
      session->selection_workers->parallel_run_count();
  statistics->parallel_run_count =
      statistics->selection_parallel_run_count >
              UINT64_MAX - statistics->build_parallel_run_count
          ? UINT64_MAX
          : statistics->build_parallel_run_count +
                statistics->selection_parallel_run_count;
  statistics->chunk_local_pack = session->chunk_local_pack ? 1 : 0;
  if (session->chunk_local_pack) {
    statistics->profile_pointer_bytes =
        static_cast<uint64_t>(session->source_profile_pointers.size()) *
        sizeof(uintptr_t);
    statistics->identity_token_bytes =
        static_cast<uint64_t>(session->identity_tokens.size()) *
        sizeof(uintptr_t);
    statistics->background_bytes =
        static_cast<uint64_t>(session->background.size()) * sizeof(float);
    const uint64_t first = statistics->profile_pointer_bytes >
            UINT64_MAX - statistics->identity_token_bytes
        ? UINT64_MAX
        : statistics->profile_pointer_bytes +
              statistics->identity_token_bytes;
    statistics->host_bytes = first > UINT64_MAX - statistics->background_bytes
        ? UINT64_MAX
        : first + statistics->background_bytes;
  } else {
    const HostProfilePack &pack = *session->pack;
    statistics->host_bytes = host_pack_bytes(pack);
    statistics->ssv_score_bytes = pack.ssv_scores.size();
    statistics->bias_profile_bytes =
        static_cast<uint64_t>(pack.bias_templates.size()) *
        sizeof(plan7_bias_profile);
    statistics->viterbi_descriptor_bytes =
        static_cast<uint64_t>(pack.vit_profiles.size()) * sizeof(VitProfile);
    statistics->viterbi_emission_bytes =
        static_cast<uint64_t>(pack.emissions.size()) * sizeof(int16_t);
    statistics->viterbi_transition_bytes =
        static_cast<uint64_t>(pack.transitions.size()) * sizeof(int16_t);
    statistics->viterbi_exact_rbv_bytes = pack.exact_rbv.size();
    statistics->forward_descriptor_bytes =
        static_cast<uint64_t>(pack.forward_profiles.size()) *
        sizeof(plan7_forward_snapshot_profile);
    statistics->forward_emission_bytes =
        static_cast<uint64_t>(pack.forward_emissions.size()) * sizeof(float);
    statistics->forward_transition_bytes =
        static_cast<uint64_t>(pack.forward_transitions.size()) * sizeof(float);
  }
  return 0;
}

extern "C" int plan7_profile_session_select(
    plan7_profile_session *session, const size_t *profile_indices,
    size_t profile_count, plan7_profile_selection **selection,
    char *error, size_t error_size) {
  if (session == nullptr || session->selection_workers == nullptr ||
      (!session->chunk_local_pack && session->pack == nullptr) ||
      (session->chunk_local_pack &&
       (session->source_profile_pointers.size() != session->profile_count ||
        session->background.size() != 20 ||
        session->identity_tokens.size() != session->profile_count)) ||
      selection == nullptr ||
      *selection != nullptr ||
      (profile_count != 0 && profile_indices == nullptr)) {
    set_error(error, error_size, "invalid profile selection arguments");
    return -1;
  }
  std::lock_guard<std::mutex> operation(session->operation_mutex);
  if (session->chunk_local_pack) {
    std::vector<uint8_t> seen;
    std::vector<uintptr_t> selected_pointers;
    try {
      seen.assign(session->profile_count, 0);
      selected_pointers.reserve(profile_count);
      for (size_t output_index = 0; output_index < profile_count;
           ++output_index) {
        const size_t source_index = profile_indices[output_index];
        if (source_index >= session->profile_count) {
          set_error(error, error_size,
                    "profile selection index is out of range");
          return -1;
        }
        if (seen[source_index] != 0) {
          set_error(error, error_size,
                    "profile selection indexes must be unique");
          return -1;
        }
        seen[source_index] = 1;
        selected_pointers.push_back(
            session->source_profile_pointers[source_index]);
      }
    } catch (...) {
      set_error(error, error_size, "profile selection allocation failed");
      return -1;
    }

    HostProfilePack selected;
    if (snapshot_profiles(
            selected_pointers.empty() ? nullptr : selected_pointers.data(),
            selected_pointers.size(), session->background.data(),
            session->selection_workers.get(), &selected,
            error, error_size) != 0)
      return -1;
    try {
      selected.identity_tokens.resize(profile_count);
      for (size_t output_index = 0; output_index < profile_count;
           ++output_index)
        selected.identity_tokens[output_index] =
            session->identity_tokens[profile_indices[output_index]];
    } catch (...) {
      set_error(error, error_size, "profile selection allocation failed");
      return -1;
    }
    if (session->selection_count == UINT64_MAX) {
      set_error(error, error_size, "profile selection identity exhausted");
      return -1;
    }
    auto *created = new (std::nothrow) plan7_profile_selection{};
    if (created == nullptr) {
      set_error(error, error_size, "profile selection allocation failed");
      return -1;
    }
    try {
      created->session_id = session->session_id;
      created->selection_id = session->selection_count + 1;
      created->pack =
          std::make_shared<const HostProfilePack>(std::move(selected));
    } catch (...) {
      delete created;
      set_error(error, error_size, "profile selection allocation failed");
      return -1;
    }
    ++session->selection_count;
    *selection = created;
    return 0;
  }
  const HostProfilePack &source = *session->pack;
  HostProfilePack selected;
  selected.alphabet_size = source.alphabet_size;
  std::vector<uint8_t> seen;
  uint64_t ssv_total = 0;
  uint64_t rbv_total = 0;
  uint64_t emission_total = 0;
  uint64_t transition_total = 0;
  uint64_t forward_emission_total = 0;
  uint64_t forward_transition_total = 0;
  try {
    seen.assign(source.vit_profiles.size(), 0);
    selected.vit_profiles.resize(profile_count);
    selected.ssv_profiles.resize(profile_count);
    selected.m_mu.resize(profile_count);
    selected.m_lambda.resize(profile_count);
    selected.v_mu.resize(profile_count);
    selected.v_lambda.resize(profile_count);
    selected.bias_templates.resize(profile_count);
    selected.forward_profiles.resize(profile_count);
    selected.identity_tokens.resize(profile_count);
  } catch (...) {
    set_error(error, error_size, "profile selection allocation failed");
    return -1;
  }

  for (size_t output_index = 0; output_index < profile_count;
       ++output_index) {
    const size_t source_index = profile_indices[output_index];
    if (source_index >= source.vit_profiles.size()) {
      set_error(error, error_size, "profile selection index is out of range");
      return -1;
    }
    if (seen[source_index] != 0) {
      set_error(error, error_size, "profile selection indexes must be unique");
      return -1;
    }
    seen[source_index] = 1;
    const VitProfile &from = source.vit_profiles[source_index];
    const uint64_t ssv_count =
        static_cast<uint64_t>(from.model_length) * source.alphabet_size;
    const uint64_t rbv_count =
        from.rbv_offset == UINT64_MAX ? 0 : ssv_count;
    const uint64_t emission_count =
        static_cast<uint64_t>(from.q) * kWarpSize * source.alphabet_size;
    const uint64_t transition_count =
        static_cast<uint64_t>(from.q) * kWarpSize * p7O_NTRANS;
    const plan7_forward_snapshot_profile &forward_from =
        source.forward_profiles[source_index];
    const uint64_t forward_emission_count =
        static_cast<uint64_t>(forward_from.q) * kForwardSubwarp *
        source.alphabet_size;
    const uint64_t forward_transition_count =
        static_cast<uint64_t>(forward_from.q) * kForwardSubwarp * p7O_NTRANS;
    VitProfile to = from;
    to.ssv_offset = ssv_total;
    to.rbv_offset = rbv_count == 0 ? UINT64_MAX : rbv_total;
    to.emission_offset = emission_total;
    to.transition_offset = transition_total;
    if (!checked_add(ssv_total, ssv_count, &ssv_total) ||
        !checked_add(rbv_total, rbv_count, &rbv_total) ||
        !checked_add(emission_total, emission_count, &emission_total) ||
        !checked_add(transition_total, transition_count, &transition_total) ||
        !checked_add(forward_emission_total, forward_emission_count,
                     &forward_emission_total) ||
        !checked_add(forward_transition_total, forward_transition_count,
                     &forward_transition_total) ||
        ssv_total > SIZE_MAX || rbv_total > SIZE_MAX ||
        emission_total > SIZE_MAX || transition_total > SIZE_MAX ||
        forward_emission_total > SIZE_MAX ||
        forward_transition_total > SIZE_MAX) {
      set_error(error, error_size, "profile selection size overflow");
      return -1;
    }
    selected.vit_profiles[output_index] = to;
    selected.ssv_profiles[output_index] = source.ssv_profiles[source_index];
    selected.ssv_profiles[output_index].score_offset = to.ssv_offset;
    selected.m_mu[output_index] = source.m_mu[source_index];
    selected.m_lambda[output_index] = source.m_lambda[source_index];
    selected.v_mu[output_index] = source.v_mu[source_index];
    selected.v_lambda[output_index] = source.v_lambda[source_index];
    selected.bias_templates[output_index] =
        source.bias_templates[source_index];
    selected.forward_profiles[output_index] = forward_from;
    selected.forward_profiles[output_index].emission_offset =
        forward_emission_total - forward_emission_count;
    selected.forward_profiles[output_index].transition_offset =
        forward_transition_total - forward_transition_count;
    selected.identity_tokens[output_index] =
        source.identity_tokens[source_index];
  }
  try {
    selected.ssv_scores.resize(static_cast<size_t>(ssv_total));
    selected.exact_rbv.resize(static_cast<size_t>(rbv_total));
    selected.emissions.resize(static_cast<size_t>(emission_total));
    selected.transitions.resize(static_cast<size_t>(transition_total));
    selected.forward_emissions.resize(
        static_cast<size_t>(forward_emission_total));
    selected.forward_transitions.resize(
        static_cast<size_t>(forward_transition_total));
  } catch (...) {
    set_error(error, error_size, "profile selection data allocation failed");
    return -1;
  }
  SelectionCopyTask task{&source, &selected, profile_indices};
  session->selection_workers->parallel_for(
      profile_count, &task, copy_selection_profile);

  if (session->selection_count == UINT64_MAX) {
    set_error(error, error_size, "profile selection identity exhausted");
    return -1;
  }
  auto *created = new (std::nothrow) plan7_profile_selection{};
  if (created == nullptr) {
    set_error(error, error_size, "profile selection allocation failed");
    return -1;
  }
  try {
    created->session_id = session->session_id;
    created->selection_id = session->selection_count + 1;
    created->pack =
        std::make_shared<const HostProfilePack>(std::move(selected));
  } catch (...) {
    delete created;
    set_error(error, error_size, "profile selection allocation failed");
    return -1;
  }
  ++session->selection_count;
  *selection = created;
  return 0;
}

extern "C" int plan7_profile_selection_destroy(
    plan7_profile_selection **selection, char *error, size_t error_size) {
  if (selection == nullptr) {
    set_error(error, error_size, "profile selection handle is null");
    return -1;
  }
  plan7_profile_selection *value = *selection;
  *selection = nullptr;
  delete value;
  return 0;
}

extern "C" int plan7_profile_selection_get_view(
    const plan7_profile_selection *selection,
    plan7_profile_selection_view *view,
    char *error, size_t error_size) {
  if (selection == nullptr || selection->pack == nullptr || view == nullptr) {
    set_error(error, error_size, "invalid profile selection view");
    return -1;
  }
  const HostProfilePack &pack = *selection->pack;
  *view = {};
  view->session_id = selection->session_id;
  view->selection_id = selection->selection_id;
  view->profile_count = pack.ssv_profiles.size();
  view->packed_scores = pack.ssv_scores.empty()
      ? nullptr : pack.ssv_scores.data();
  view->packed_score_count = pack.ssv_scores.size();
  view->profiles = pack.ssv_profiles.empty()
      ? nullptr : pack.ssv_profiles.data();
  view->m_mu = pack.m_mu.empty() ? nullptr : pack.m_mu.data();
  view->m_lambda = pack.m_lambda.empty() ? nullptr : pack.m_lambda.data();
  view->v_mu = pack.v_mu.empty() ? nullptr : pack.v_mu.data();
  view->v_lambda = pack.v_lambda.empty() ? nullptr : pack.v_lambda.data();
  view->bias_templates = pack.bias_templates.empty()
      ? nullptr : pack.bias_templates.data();
  view->identity_tokens = pack.identity_tokens.empty()
      ? nullptr : pack.identity_tokens.data();
  view->host_bytes = host_pack_bytes(pack);
  return 0;
}

extern "C" int plan7_profile_selection_snapshot_equal_for_test(
    const plan7_profile_selection *left,
    const plan7_profile_selection *right,
    char *error, size_t error_size) {
  if (left == nullptr || left->pack == nullptr ||
      right == nullptr || right->pack == nullptr) {
    set_error(error, error_size,
              "invalid profile selection snapshot comparison");
    return -1;
  }
  const HostProfilePack &a = *left->pack;
  const HostProfilePack &b = *right->pack;
  const auto equal_bytes = [](const auto &x, const auto &y) {
    if (x.size() != y.size()) return false;
    return x.empty() ||
        std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
  };
  return a.alphabet_size == b.alphabet_size &&
      equal_bytes(a.vit_profiles, b.vit_profiles) &&
      equal_bytes(a.ssv_scores, b.ssv_scores) &&
      equal_bytes(a.exact_rbv, b.exact_rbv) &&
      equal_bytes(a.emissions, b.emissions) &&
      equal_bytes(a.transitions, b.transitions) &&
      equal_bytes(a.ssv_profiles, b.ssv_profiles) &&
      equal_bytes(a.m_mu, b.m_mu) &&
      equal_bytes(a.m_lambda, b.m_lambda) &&
      equal_bytes(a.v_mu, b.v_mu) &&
      equal_bytes(a.v_lambda, b.v_lambda) &&
      equal_bytes(a.bias_templates, b.bias_templates) &&
      equal_bytes(a.forward_profiles, b.forward_profiles) &&
      equal_bytes(a.forward_emissions, b.forward_emissions) &&
      equal_bytes(a.forward_transitions, b.forward_transitions)
      ? 1 : 0;
}

extern "C" int plan7_profile_selection_stage_viterbi(
    const plan7_profile_selection *selection,
    plan7_viterbi_database **database,
    char *error, size_t error_size) {
  if (selection == nullptr || selection->pack == nullptr ||
      database == nullptr || *database != nullptr) {
    set_error(error, error_size, "invalid staged Viterbi arguments");
    return -1;
  }
  auto *created = new (std::nothrow) plan7_viterbi_database{};
  if (created == nullptr) {
    set_error(error, error_size, "staged Viterbi allocation failed");
    return -1;
  }
  initialize_viterbi_database(created);
  created->sealed_source = true;
  created->sealed_pack = selection->pack;
  created->alphabet_size = selection->pack->alphabet_size;
  try {
    created->source_profile_pointers = selection->pack->identity_tokens;
  } catch (...) {
    delete created;
    set_error(error, error_size, "staged Viterbi identity allocation failed");
    return -1;
  }
  if (upload_viterbi_database(created, error, error_size) != 0) {
    delete created;
    return -1;
  }
  *database = created;
  return 0;
}

extern "C" int plan7_profile_selection_stage_forward(
    const plan7_profile_selection *selection,
    plan7_forward_database **database,
    char *error, size_t error_size) {
  if (selection == nullptr || selection->pack == nullptr) {
    set_error(error, error_size, "invalid staged Forward selection");
    return -1;
  }
  const HostProfilePack &pack = *selection->pack;
  return plan7_forward_database_create_snapshot(
      pack.alphabet_size,
      pack.forward_profiles.empty() ? nullptr : pack.forward_profiles.data(),
      pack.forward_profiles.size(),
      pack.forward_emissions.empty() ? nullptr : pack.forward_emissions.data(),
      pack.forward_emissions.size(),
      pack.forward_transitions.empty()
          ? nullptr : pack.forward_transitions.data(),
      pack.forward_transitions.size(),
      pack.identity_tokens.empty() ? nullptr : pack.identity_tokens.data(),
      database, error, error_size);
}

extern "C" int plan7_postfilter_workspace_create(
    plan7_postfilter_workspace **workspace, char *error, size_t error_size) {
  if (workspace == nullptr || *workspace != nullptr) {
    set_error(error, error_size, "invalid post-filter workspace output");
    return -1;
  }
  int current_device = -1;
  const cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  auto *created = new (std::nothrow) plan7_postfilter_workspace{};
  if (created == nullptr) {
    set_error(error, error_size, "post-filter workspace allocation failed");
    return -1;
  }
  created->device_ordinal = current_device;
  *workspace = created;
  return 0;
}

extern "C" int plan7_postfilter_workspace_destroy(
    plan7_postfilter_workspace **workspace, char *error, size_t error_size) {
  if (workspace == nullptr) {
    set_error(error, error_size, "post-filter workspace handle is null");
    return -1;
  }
  plan7_postfilter_workspace *value = *workspace;
  *workspace = nullptr;
  if (value == nullptr) return 0;
  const int status = destroy_postfilter_workspace_device(
      value, error, error_size);
  delete value;
  return status;
}

extern "C" int plan7_postfilter_workspace_get_statistics(
    const plan7_postfilter_workspace *workspace,
    plan7_postfilter_workspace_statistics *statistics,
    char *error, size_t error_size) {
  if (workspace == nullptr || statistics == nullptr) {
    set_error(error, error_size, "invalid post-filter workspace statistics");
    return -1;
  }
  *statistics = {};
  statistics->device_bytes = postfilter_workspace_device_bytes(workspace);
  statistics->dp_capacity_bytes =
      static_cast<uint64_t>(workspace->dp_capacity);
  statistics->growth_count = workspace->growth_count;
  statistics->run_count = workspace->run_count;
  statistics->full_msv_compaction_run_count =
      workspace->full_msv_compaction_run_count;
  statistics->full_msv_compaction_chunk_count =
      workspace->full_msv_compaction_chunk_count;
  statistics->full_msv_compaction_source_count =
      workspace->full_msv_compaction_source_count;
  statistics->full_msv_compaction_selected_count =
      workspace->full_msv_compaction_selected_count;
  statistics->full_msv_legacy_run_count =
      workspace->full_msv_legacy_run_count;
  statistics->full_msv_launch_candidate_count =
      workspace->full_msv_launch_candidate_count;
  statistics->full_msv_launch_candidate_avoided_count =
      workspace->full_msv_launch_candidate_avoided_count;
  statistics->full_msv_index_d2h_bytes =
      workspace->full_msv_index_d2h_bytes;
  statistics->full_msv_packed_run_count =
      workspace->full_msv_packed_run_count;
  statistics->full_msv_packed_group_count =
      workspace->full_msv_packed_group_count;
  statistics->full_msv_packed_candidate_count =
      workspace->full_msv_packed_candidate_count;
  statistics->full_msv_scalar_candidate_count =
      workspace->full_msv_scalar_candidate_count;
  statistics->vit_length_cache_run_count =
      workspace->vit_length_cache_run_count;
  statistics->vit_length_cache_entry_count =
      workspace->vit_length_cache_entry_count;
  statistics->vit_length_cache_candidate_count =
      workspace->vit_length_cache_candidate_count;
  statistics->vit_length_direct_candidate_count =
      workspace->vit_length_direct_candidate_count;
  statistics->vit_length_cache_build_ns =
      workspace->vit_length_cache_build_ns;
  statistics->vit_length_candidate_plan_ns =
      workspace->vit_length_candidate_plan_ns;
  const size_t capacities[PLAN7_POSTFILTER_CAPACITY_COUNT] = {
      workspace->states_capacity,
      workspace->bias_inputs_capacity,
      workspace->bias_results_capacity,
      workspace->vit_results_capacity,
      workspace->moves_capacity,
      workspace->msv_offsets_capacity,
      workspace->vit_offsets_capacity,
      workspace->dp_capacity,
      workspace->results_capacity};
  for (size_t i = 0; i < PLAN7_POSTFILTER_CAPACITY_COUNT; ++i)
    statistics->capacity_bytes[i] = static_cast<uint64_t>(capacities[i]);
  return 0;
}

extern "C" int plan7_viterbi_database_matches_ssv(
    const plan7_viterbi_database *database,
    const uint8_t *packed_scores, size_t packed_score_count,
    const uintptr_t *source_profile_pointers,
    const plan7_ssv_f1_profile *profiles, size_t profile_count,
    char *error, size_t error_size) {
  if (database == nullptr ||
      (profile_count != 0 &&
       (profiles == nullptr || packed_scores == nullptr ||
        source_profile_pointers == nullptr)) ||
      database_host_profiles(database).size() != profile_count) {
    set_error(error, error_size, "Viterbi and SSV profile counts differ");
    return -1;
  }
  const auto &host_profiles = database_host_profiles(database);
  const auto &host_scores = database_host_ssv_scores(database);
  for (size_t p = 0; p < profile_count; ++p) {
    const VitProfile &vit = host_profiles[p];
    const plan7_ssv_profile &msv = profiles[p].profile;
    const uint64_t expected_count =
        static_cast<uint64_t>(vit.model_length) * database->alphabet_size;
    if (source_profile_pointers[p] !=
          database->source_profile_pointers[p] ||
        (!database->sealed_source &&
         !live_profile_matches_snapshot(database, p)) ||
        vit.model_length != static_cast<uint32_t>(msv.model_length) ||
        msv.score_offset != vit.ssv_offset ||
        msv.score_count != expected_count ||
        msv.score_stride != database->alphabet_size ||
        msv.score_offset > packed_score_count ||
        msv.score_count > packed_score_count - msv.score_offset ||
        vit.msv_tbm != msv.tbm || vit.msv_tec != msv.tec ||
        vit.msv_base != msv.base || vit.msv_bias != msv.bias ||
        vit.msv_scale != msv.scale ||
        std::memcmp(host_scores.data() + vit.ssv_offset,
                    packed_scores + msv.score_offset,
                    static_cast<size_t>(expected_count)) != 0) {
      set_error(error, error_size,
                "Viterbi database does not match the SSV profile row");
      return -1;
    }
  }
  if (packed_score_count != host_scores.size()) {
    set_error(error, error_size,
              "SSV profile scores have trailing bytes");
    return -1;
  }
  return 0;
}

namespace {

struct ScopedReasonFacts {
  uint16_t *device = nullptr;
  ~ScopedReasonFacts() { cudaFree(device); }
};

int postfilter_candidates_device_with_workspace_impl(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, uint16_t *host_reason_facts,
    plan7_postfilter_reason_statistics *reason_statistics,
    int execution_policy, bool skip_bias_reject_viterbi,
    char *error, size_t error_size) {
  if (workspace == nullptr || database == nullptr || device_residues == nullptr ||
      device_sequence_offsets == nullptr || host_sequence_lengths == nullptr ||
      device_null_scores == nullptr || device_compact_scores == nullptr ||
      device_f1_profiles == nullptr || device_tjb == nullptr ||
      device_length_logp == nullptr || device_length_log1mp == nullptr ||
      device_bias_profiles == nullptr || device_candidates == nullptr ||
      host_candidates == nullptr || device_msv_inputs == nullptr ||
      (candidate_count != 0 && host_results == nullptr)) {
    set_error(error, error_size, "invalid post-filter device buffers");
    return -1;
  }
  if (reason_statistics != nullptr)
    *reason_statistics = {};
  workspace->resident_results_valid = false;
  workspace->f2_view_valid = false;
  if (candidate_count == 0) {
    if (workspace->resident_generation == UINT64_MAX) {
      set_error(error, error_size,
                "post-filter resident generation overflow");
      return -1;
    }
    ++workspace->resident_generation;
    workspace->resident_device_candidates = device_candidates;
    workspace->resident_host_candidates = host_candidates;
    workspace->resident_host_results = host_results;
    workspace->resident_source_count = 0;
    workspace->resident_results_valid = true;
    return 0;
  }
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != database->device_ordinal) {
    set_error(error, error_size,
              "Viterbi database belongs to a different CUDA device");
    return -1;
  }
  if (current_device != workspace->device_ordinal) {
    set_error(error, error_size,
              "post-filter workspace belongs to a different CUDA device");
    return -1;
  }
  ++workspace->run_count;

  if (candidate_count == SIZE_MAX) {
    set_error(error, error_size, "post-filter candidate count overflow");
    return -1;
  }
  if (execution_policy < PLAN7_GPU_EXECUTION_POLICY_AUTO ||
      execution_policy > PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT) {
    set_error(error, error_size, "invalid post-filter execution policy");
    return -1;
  }
  const char *full_msv_policy = std::getenv("PLAN7_GPU_FULL_MSV_POLICY");
  const bool force_legacy_full_msv =
      (full_msv_policy != nullptr &&
       std::strcmp(full_msv_policy, "legacy") == 0) ||
      execution_policy == PLAN7_GPU_EXECUTION_POLICY_SIMPLE;
  const bool force_compact_full_msv =
      (full_msv_policy != nullptr &&
       std::strcmp(full_msv_policy, "compact") == 0) ||
      execution_policy == PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT;
  if (force_compact_full_msv && candidate_count > UINT32_MAX) {
    set_error(error, error_size,
              "compact full-MSV candidate count exceeds uint32 range");
    return -1;
  }
  const bool compact_full_msv =
      !force_legacy_full_msv && candidate_count <= UINT32_MAX &&
      (force_compact_full_msv || candidate_count >= 65536);
  const char *full_msv_arithmetic =
      std::getenv("PLAN7_GPU_FULL_MSV_ARITHMETIC");
  const bool force_scalar_full_msv =
      (full_msv_arithmetic != nullptr &&
       std::strcmp(full_msv_arithmetic, "scalar") == 0) ||
      execution_policy == PLAN7_GPU_EXECUTION_POLICY_SIMPLE;
  const bool allow_packed_full_msv =
      compact_full_msv && !force_scalar_full_msv;
  std::vector<uint64_t> &host_msv_offsets = workspace->host_msv_offsets;
  std::vector<uint32_t> &host_msv_candidate_indices =
      workspace->host_msv_candidate_indices;
  std::vector<uint32_t> &host_msv_execution_indices =
      workspace->host_msv_execution_indices;
  std::vector<uint64_t> &host_vit_offsets = workspace->host_vit_offsets;
  std::vector<VitLengthTransitions> &host_moves = workspace->host_moves;
  std::vector<size_t> &msv_tiles = workspace->msv_tiles;
  std::vector<size_t> &vit_tiles = workspace->vit_tiles;
  const auto &profiles = database_host_profiles(database);
  const char *vit_length_policy =
      std::getenv("PLAN7_GPU_VIT_LENGTH_CACHE");
  const bool vit_length_cache_requested =
      vit_length_policy == nullptr
          ? (execution_policy != PLAN7_GPU_EXECUTION_POLICY_SIMPLE &&
             candidate_count >= 65536)
          : std::strcmp(vit_length_policy, "0") != 0;
  const bool vit_length_cache_audit =
      vit_length_policy != nullptr &&
      std::strcmp(vit_length_policy, "audit") == 0;
  const bool vit_length_cache_reference =
      vit_length_policy != nullptr &&
      std::strcmp(vit_length_policy, "reference") == 0;
  const bool vit_length_cache_force =
      vit_length_cache_audit ||
      (vit_length_policy != nullptr &&
       std::strcmp(vit_length_policy, "force") == 0);
  size_t vit_length_table_entries = 0;
  bool use_vit_length_cache = false;
  if (vit_length_cache_requested) {
    if (!prepare_target_length_classes(
            workspace, host_sequence_lengths, sequence_count,
            error, error_size))
      return -1;
    const size_t length_class_count =
        workspace->host_msv_class_lengths.size();
    if (profiles.size() != 0 &&
        length_class_count > SIZE_MAX / profiles.size()) {
      set_error(error, error_size,
                "Viterbi length-transition table size overflow");
      return -1;
    }
    vit_length_table_entries = profiles.size() *
        workspace->host_msv_class_lengths.size();
    use_vit_length_cache = !vit_length_cache_reference &&
        (vit_length_cache_force ||
         vit_length_table_entries < candidate_count);
  }
  if (use_vit_length_cache) {
    const auto cache_started = std::chrono::steady_clock::now();
    try {
      workspace->host_length_transition_table.resize(
          vit_length_table_entries);
      const size_t length_class_count =
          workspace->host_msv_class_lengths.size();
      for (size_t profile_index = 0; profile_index < profiles.size();
           ++profile_index) {
        for (size_t length_class = 0;
             length_class < length_class_count; ++length_class) {
          workspace->host_length_transition_table[
              profile_index * length_class_count + length_class] =
              length_transitions_for(
                  profiles[profile_index],
                  static_cast<int>(
                      workspace->host_msv_class_lengths[length_class]));
        }
      }
    } catch (...) {
      set_error(error, error_size,
                "Viterbi length-transition cache allocation failed");
      return -1;
    }
    const auto cache_elapsed = std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - cache_started).count();
    ++workspace->vit_length_cache_run_count;
    saturating_counter_add(vit_length_table_entries,
                           &workspace->vit_length_cache_entry_count);
    saturating_counter_add(candidate_count,
                           &workspace->vit_length_cache_candidate_count);
    saturating_counter_add(
        cache_elapsed < 0 ? 0 : static_cast<uint64_t>(cache_elapsed),
        &workspace->vit_length_cache_build_ns);
  } else if (vit_length_cache_requested) {
    saturating_counter_add(candidate_count,
                           &workspace->vit_length_direct_candidate_count);
  }
  uint64_t maximum_msv_bytes = 0;
  uint64_t maximum_vit_cells = 0;
  constexpr uint64_t kVitCellLimit = kDpByteLimit / sizeof(int16_t);
  const auto candidate_plan_started = std::chrono::steady_clock::now();
  try {
    if (compact_full_msv) {
      host_msv_offsets.clear();
      host_msv_candidate_indices.clear();
      host_msv_execution_indices.clear();
    } else {
      host_msv_offsets.assign(candidate_count + 1, 0);
    }
    host_vit_offsets.assign(candidate_count + 1, 0);
    host_moves.resize(candidate_count);
    msv_tiles.clear();
    vit_tiles.clear();
    msv_tiles.push_back(0);
    vit_tiles.push_back(0);
    for (size_t c = 0; c < candidate_count; ++c) {
      const plan7_bias_candidate mapping = host_candidates[c];
      if (mapping.profile_index >= profiles.size() ||
          mapping.sequence_index >= sequence_count ||
          host_sequence_lengths[mapping.sequence_index] > 100000) {
        set_error(error, error_size, "invalid post-filter candidate mapping");
        return -1;
      }
      const VitProfile &profile = profiles[mapping.profile_index];
      const int length = static_cast<int>(
          host_sequence_lengths[mapping.sequence_index]);
      if (use_vit_length_cache) {
        const uint32_t length_class =
            workspace->host_msv_length_to_class[
                static_cast<size_t>(length)];
        const size_t length_class_count =
            workspace->host_msv_class_lengths.size();
        if (length_class >= length_class_count) {
          set_error(error, error_size,
                    "Viterbi length-transition class is invalid");
          return -1;
        }
        host_moves[c] = workspace->host_length_transition_table[
            static_cast<size_t>(mapping.profile_index) *
                length_class_count + length_class];
        if (vit_length_cache_audit) {
          const VitLengthTransitions reference =
              length_transitions_for(profile, length);
          if (reference.n_move != host_moves[c].n_move ||
              reference.j_move != host_moves[c].j_move ||
              reference.c_move != host_moves[c].c_move) {
            set_error(error, error_size,
                      "Viterbi length-transition cache mismatch");
            return -1;
          }
        }
      } else {
        host_moves[c] = length_transitions_for(profile, length);
      }
      const uint64_t msv_cells =
          static_cast<uint64_t>(profile.q) * kWarpSize;
      const uint64_t vit_cells = msv_cells * 3;
      if ((!compact_full_msv &&
           !checked_add(host_msv_offsets[c], msv_cells,
                        &host_msv_offsets[c + 1])) ||
          !checked_add(host_vit_offsets[c], vit_cells,
                       &host_vit_offsets[c + 1])) {
        set_error(error, error_size, "post-filter DP offset overflow");
        return -1;
      }
    }
    if (!compact_full_msv) {
      for (size_t begin = 0; begin < candidate_count;) {
        size_t end = begin + 1;
        while (end < candidate_count &&
               host_msv_offsets[end + 1] - host_msv_offsets[begin] <=
                   kDpByteLimit)
          ++end;
        maximum_msv_bytes = std::max(
            maximum_msv_bytes,
            host_msv_offsets[end] - host_msv_offsets[begin]);
        msv_tiles.push_back(end);
        begin = end;
      }
    }
    for (size_t begin = 0; begin < candidate_count;) {
      size_t end = begin + 1;
      while (end < candidate_count &&
             host_vit_offsets[end + 1] - host_vit_offsets[begin] <=
                 kVitCellLimit)
        ++end;
      maximum_vit_cells = std::max(
          maximum_vit_cells,
          host_vit_offsets[end] - host_vit_offsets[begin]);
      vit_tiles.push_back(end);
      begin = end;
    }
  } catch (...) {
    set_error(error, error_size, "post-filter host workspace allocation failed");
    return -1;
  }
  if (vit_length_cache_requested) {
    const auto plan_elapsed = std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - candidate_plan_started).count();
    saturating_counter_add(
        plan_elapsed < 0 ? 0 : static_cast<uint64_t>(plan_elapsed),
        &workspace->vit_length_candidate_plan_ns);
  }
  const uint64_t maximum_vit_bytes = maximum_vit_cells * sizeof(int16_t);
  if (maximum_msv_bytes > kDpByteLimit ||
      maximum_vit_bytes > kDpByteLimit ||
      maximum_msv_bytes > SIZE_MAX || maximum_vit_bytes > SIZE_MAX) {
    set_error(error, error_size, "post-filter DP tile exceeds 256 MiB");
    return -1;
  }

  size_t candidate_bytes;
  size_t vit_offset_bytes;
  size_t move_bytes;
  size_t bias_input_bytes;
  size_t bias_result_bytes;
  size_t vit_result_bytes;
  size_t post_result_bytes;
  size_t reason_fact_bytes = 0;
  if (!checked_bytes(candidate_count + 1, sizeof(uint64_t),
                     &vit_offset_bytes) ||
      !checked_bytes(candidate_count, sizeof(VitLengthTransitions),
                     &move_bytes) ||
      !checked_bytes(candidate_count, sizeof(plan7_bias_ssv_input),
                     &bias_input_bytes) ||
      !checked_bytes(candidate_count, sizeof(plan7_bias_result),
                     &bias_result_bytes) ||
      !checked_bytes(candidate_count, sizeof(VitResult), &vit_result_bytes) ||
      !checked_bytes(candidate_count, sizeof(plan7_postfilter_result),
                     &post_result_bytes) ||
      (host_reason_facts != nullptr &&
       !checked_bytes(candidate_count, sizeof(uint16_t), &reason_fact_bytes)) ||
      !checked_bytes(candidate_count, 1, &candidate_bytes)) {
    set_error(error, error_size, "post-filter workspace size overflow");
    return -1;
  }

  const unsigned blocks = static_cast<unsigned>(
      (candidate_count - 1) / kThreads + 1);
  ScopedReasonFacts reason_storage;

#define CUDA_RUN(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  if (grow_workspace_buffer(&workspace->device_states,
                            &workspace->states_capacity, candidate_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter states)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_bias_inputs,
                            &workspace->bias_inputs_capacity, bias_input_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter bias inputs)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_bias_results,
                            &workspace->bias_results_capacity,
                            bias_result_bytes, &workspace->growth_count,
                            "cudaMalloc(post-filter bias results)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_vit_results,
                            &workspace->vit_results_capacity, vit_result_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter Viterbi results)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_moves,
                            &workspace->moves_capacity, move_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter length transitions)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_vit_offsets,
                            &workspace->vit_offsets_capacity,
                            vit_offset_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter Viterbi offsets)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_results,
                            &workspace->results_capacity, post_result_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter results)", error,
                            error_size) != 0)
    return -1;
  if (host_reason_facts != nullptr) {
    CUDA_RUN(cudaMalloc(&reason_storage.device, reason_fact_bytes));
    CUDA_RUN(cudaMemset(reason_storage.device, 0, reason_fact_bytes));
    postfilter_full_msv_execution_facts_kernel<<<blocks, kThreads>>>(
        device_msv_inputs, candidate_count, reason_storage.device);
    CUDA_RUN(cudaGetLastError());
  }
  CUDA_RUN(cudaMemcpy(workspace->device_moves, host_moves.data(), move_bytes,
                      cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(workspace->device_vit_offsets, host_vit_offsets.data(),
                      vit_offset_bytes, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemset(workspace->device_vit_results, 0xff, vit_result_bytes));

  if (!compact_full_msv) {
    size_t msv_offset_bytes;
    const uint64_t dp_bytes_u64 =
        std::max(maximum_msv_bytes, maximum_vit_bytes);
    if (!checked_bytes(candidate_count + 1, sizeof(uint64_t),
                       &msv_offset_bytes) ||
        grow_workspace_buffer(&workspace->device_msv_offsets,
                              &workspace->msv_offsets_capacity,
                              msv_offset_bytes, &workspace->growth_count,
                              "cudaMalloc(post-filter MSV offsets)", error,
                              error_size) != 0 ||
        grow_workspace_buffer(&workspace->device_dp,
                              &workspace->dp_capacity,
                              static_cast<size_t>(dp_bytes_u64),
                              &workspace->growth_count,
                              "cudaMalloc(post-filter DP workspace)", error,
                              error_size) != 0)
      return -1;
    CUDA_RUN(cudaMemcpy(workspace->device_msv_offsets,
                        host_msv_offsets.data(), msv_offset_bytes,
                        cudaMemcpyHostToDevice));
    ++workspace->full_msv_legacy_run_count;
    saturating_counter_add(candidate_count,
                           &workspace->full_msv_launch_candidate_count);
    for (size_t tile = 0; tile + 1 < msv_tiles.size(); ++tile) {
      const size_t tile_begin = msv_tiles[tile];
      const size_t tile_count = msv_tiles[tile + 1] - tile_begin;
      full_msv_kernel<<<
          static_cast<unsigned>((tile_count - 1) / kWarpsPerBlock + 1),
          kThreads>>>(
          device_residues, device_sequence_offsets, device_compact_scores,
          database->device_exact_rbv, device_f1_profiles,
          database->device_profiles, device_tjb, device_candidates, nullptr,
          workspace->device_msv_offsets, tile_begin, tile_count,
          host_msv_offsets[tile_begin],
          static_cast<uint8_t *>(workspace->device_dp), device_msv_inputs);
      CUDA_RUN(cudaGetLastError());
    }
  } else {
    ++workspace->full_msv_compaction_run_count;
    saturating_counter_add(candidate_count,
                           &workspace->full_msv_compaction_source_count);
    if (allow_packed_full_msv &&
        !prepare_target_length_classes(
            workspace, host_sequence_lengths, sequence_count,
            error, error_size))
      return -1;
    for (size_t source_begin = 0; source_begin < candidate_count;) {
      const size_t source_count = std::min(
          kFullMsvCompactSourceChunk, candidate_count - source_begin);
      size_t index_capacity_bytes;
      size_t offset_capacity_bytes;
      if (!checked_bytes(source_count, sizeof(uint32_t),
                         &index_capacity_bytes) ||
          !checked_bytes(source_count + 1, sizeof(uint64_t),
                         &offset_capacity_bytes) ||
          index_capacity_bytes > SIZE_MAX - 7) {
        set_error(error, error_size,
                  "compact full-MSV workspace size overflow");
        return -1;
      }
      const size_t offset_start = (index_capacity_bytes + 7) & ~size_t{7};
      if (offset_capacity_bytes > SIZE_MAX - offset_start) {
        set_error(error, error_size,
                  "compact full-MSV workspace size overflow");
        return -1;
      }
      const size_t msv_storage_bytes =
          offset_start + offset_capacity_bytes;
      if (grow_workspace_buffer(&workspace->device_msv_offsets,
                                &workspace->msv_offsets_capacity,
                                msv_storage_bytes,
                                &workspace->growth_count,
                                "cudaMalloc(compact full-MSV storage)", error,
                                error_size) != 0)
        return -1;
      auto *msv_storage = reinterpret_cast<uint8_t *>(
          workspace->device_msv_offsets);
      auto *device_selected = reinterpret_cast<uint32_t *>(msv_storage);
      auto *device_selected_count = reinterpret_cast<uint32_t *>(
          msv_storage + offset_start);
      const cub::CountingInputIterator<uint32_t> source_indices(
          static_cast<uint32_t>(source_begin));
      const FullMsvCandidatePredicate select_full_msv{device_msv_inputs};
      size_t select_workspace_bytes = 0;
      CUDA_RUN(cub::DeviceSelect::If(
          nullptr, select_workspace_bytes, source_indices, device_selected,
          device_selected_count, static_cast<int>(source_count),
          select_full_msv));
      if (grow_workspace_buffer(&workspace->device_dp,
                                &workspace->dp_capacity,
                                select_workspace_bytes,
                                &workspace->growth_count,
                                "cudaMalloc(full-MSV selection workspace)",
                                error, error_size) != 0)
        return -1;
      CUDA_RUN(cub::DeviceSelect::If(
          workspace->device_dp, select_workspace_bytes, source_indices,
          device_selected, device_selected_count,
          static_cast<int>(source_count), select_full_msv));
      uint32_t selected_count_u32 = 0;
      CUDA_RUN(cudaMemcpy(&selected_count_u32, device_selected_count,
                          sizeof(selected_count_u32),
                          cudaMemcpyDeviceToHost));
      const size_t selected_count =
          static_cast<size_t>(selected_count_u32);
      if (selected_count > source_count) {
        set_error(error, error_size,
                  "compact full-MSV selected count is invalid");
        return -1;
      }
      ++workspace->full_msv_compaction_chunk_count;
      saturating_counter_add(selected_count,
          &workspace->full_msv_compaction_selected_count);
      saturating_counter_add(selected_count,
          &workspace->full_msv_launch_candidate_count);
      saturating_counter_add(source_count - selected_count,
          &workspace->full_msv_launch_candidate_avoided_count);
      source_begin += source_count;
      if (selected_count == 0) continue;

      size_t selected_index_bytes;
      if (!checked_bytes(selected_count, sizeof(uint32_t),
                         &selected_index_bytes)) {
        set_error(error, error_size,
                  "compact full-MSV selected-index size overflow");
        return -1;
      }
      try {
        host_msv_candidate_indices.resize(selected_count);
      } catch (...) {
        set_error(error, error_size,
                  "compact full-MSV host index allocation failed");
        return -1;
      }
      CUDA_RUN(cudaMemcpy(host_msv_candidate_indices.data(), device_selected,
                          selected_index_bytes, cudaMemcpyDeviceToHost));
      saturating_counter_add(selected_index_bytes,
                             &workspace->full_msv_index_d2h_bytes);
      for (size_t selected = 0; selected < selected_count; ++selected) {
        const size_t candidate = host_msv_candidate_indices[selected];
        const size_t source_chunk_begin = source_begin - source_count;
        if (candidate < source_chunk_begin || candidate >= source_begin ||
            (selected != 0 &&
             candidate <= host_msv_candidate_indices[selected - 1])) {
          set_error(error, error_size,
                    "compact full-MSV index order is invalid");
          return -1;
        }
      }

      size_t packed_group_count = 0;
      size_t packed_candidate_count = 0;
      size_t scalar_candidate_count = selected_count;
      uint64_t chunk_maximum_msv_bytes = 0;
      try {
        host_msv_execution_indices.clear();
        host_msv_execution_indices.reserve(selected_count);
        if (allow_packed_full_msv) {
          for (auto &bucket : workspace->host_msv_length_buckets)
            bucket.clear();
          size_t scalar_write = 0;
          for (size_t profile_begin = 0;
               profile_begin < selected_count;) {
            const uint32_t profile_index = host_candidates[
                host_msv_candidate_indices[profile_begin]].profile_index;
            size_t profile_end = profile_begin + 1;
            while (profile_end < selected_count &&
                   host_candidates[host_msv_candidate_indices[profile_end]].
                           profile_index == profile_index)
              ++profile_end;
            workspace->host_msv_touched_length_classes.clear();
            for (size_t selected = profile_begin;
                 selected < profile_end; ++selected) {
              const uint32_t candidate =
                  host_msv_candidate_indices[selected];
              const plan7_bias_candidate mapping = host_candidates[candidate];
              const uint64_t length =
                  host_sequence_lengths[mapping.sequence_index];
              const uint32_t length_class =
                  workspace->host_msv_length_to_class[length];
              if (length_class >=
                  workspace->host_msv_length_buckets.size()) {
                set_error(error, error_size,
                          "packed full-MSV length class is invalid");
                return -1;
              }
              auto &bucket =
                  workspace->host_msv_length_buckets[length_class];
              if (bucket.empty())
                workspace->host_msv_touched_length_classes.push_back(
                    length_class);
              bucket.push_back(candidate);
            }
            for (const uint32_t length_class :
                 workspace->host_msv_touched_length_classes) {
              auto &bucket =
                  workspace->host_msv_length_buckets[length_class];
              const size_t groupable = (bucket.size() / 4) * 4;
              host_msv_execution_indices.insert(
                  host_msv_execution_indices.end(), bucket.begin(),
                  bucket.begin() + groupable);
              for (size_t scalar = groupable; scalar < bucket.size();
                   ++scalar)
                host_msv_candidate_indices[scalar_write++] = bucket[scalar];
              bucket.clear();
            }
            profile_begin = profile_end;
          }
          packed_candidate_count = host_msv_execution_indices.size();
          packed_group_count = packed_candidate_count / 4;
          scalar_candidate_count = scalar_write;
          host_msv_execution_indices.insert(
              host_msv_execution_indices.end(),
              host_msv_candidate_indices.begin(),
              host_msv_candidate_indices.begin() + scalar_write);
        } else {
          host_msv_execution_indices.assign(
              host_msv_candidate_indices.begin(),
              host_msv_candidate_indices.end());
        }
        if (host_msv_execution_indices.size() != selected_count) {
          set_error(error, error_size,
                    "packed full-MSV candidate partition is invalid");
          return -1;
        }

        const size_t work_unit_count =
            packed_group_count + scalar_candidate_count;
        host_msv_offsets.assign(work_unit_count + 1, 0);
        const auto &profiles = database_host_profiles(database);
        for (size_t work = 0; work < work_unit_count; ++work) {
          const bool packed = work < packed_group_count;
          const size_t execution_position = packed
              ? work * 4
              : packed_candidate_count + work - packed_group_count;
          const size_t candidate =
              host_msv_execution_indices[execution_position];
          const plan7_bias_candidate mapping = host_candidates[candidate];
          uint64_t msv_bytes =
              static_cast<uint64_t>(profiles[mapping.profile_index].q) *
              kWarpSize;
          if (packed && !checked_multiply(msv_bytes, 4, &msv_bytes)) {
            set_error(error, error_size,
                      "packed full-MSV DP size overflow");
            return -1;
          }
          if (!checked_add(host_msv_offsets[work], msv_bytes,
                           &host_msv_offsets[work + 1])) {
            set_error(error, error_size,
                      "compact full-MSV DP offset overflow");
            return -1;
          }
        }
        msv_tiles.clear();
        msv_tiles.push_back(0);
        const auto append_tiles = [&](size_t range_begin,
                                      size_t range_end) {
          for (size_t begin = range_begin; begin < range_end;) {
            size_t end = begin + 1;
            while (end < range_end &&
                   host_msv_offsets[end + 1] - host_msv_offsets[begin] <=
                       kDpByteLimit)
              ++end;
            chunk_maximum_msv_bytes = std::max(
                chunk_maximum_msv_bytes,
                host_msv_offsets[end] - host_msv_offsets[begin]);
            msv_tiles.push_back(end);
            begin = end;
          }
        };
        append_tiles(0, packed_group_count);
        if (msv_tiles.back() != packed_group_count)
          msv_tiles.push_back(packed_group_count);
        append_tiles(packed_group_count, work_unit_count);
      } catch (...) {
        set_error(error, error_size,
                  "compact full-MSV execution plan allocation failed");
        return -1;
      }
      const size_t work_unit_count =
          packed_group_count + scalar_candidate_count;
      size_t selected_offset_bytes;
      if (chunk_maximum_msv_bytes > kDpByteLimit ||
          chunk_maximum_msv_bytes > SIZE_MAX ||
          !checked_bytes(work_unit_count + 1, sizeof(uint64_t),
                         &selected_offset_bytes) ||
          selected_offset_bytes > offset_capacity_bytes) {
        set_error(error, error_size,
                  "compact full-MSV DP workspace size overflow");
        return -1;
      }
      if (grow_workspace_buffer(&workspace->device_dp,
                                &workspace->dp_capacity,
                                static_cast<size_t>(chunk_maximum_msv_bytes),
                                &workspace->growth_count,
                                "cudaMalloc(compact full-MSV DP workspace)",
                                error, error_size) != 0)
        return -1;
      auto *device_selected_offsets = reinterpret_cast<uint64_t *>(
          msv_storage + offset_start);
      CUDA_RUN(cudaMemcpy(device_selected,
                          host_msv_execution_indices.data(),
                          selected_index_bytes, cudaMemcpyHostToDevice));
      CUDA_RUN(cudaMemcpy(device_selected_offsets, host_msv_offsets.data(),
                          selected_offset_bytes, cudaMemcpyHostToDevice));
      if (packed_group_count != 0) {
        ++workspace->full_msv_packed_run_count;
        saturating_counter_add(packed_group_count,
                               &workspace->full_msv_packed_group_count);
        saturating_counter_add(packed_candidate_count,
                               &workspace->full_msv_packed_candidate_count);
      }
      saturating_counter_add(scalar_candidate_count,
                             &workspace->full_msv_scalar_candidate_count);
      for (size_t tile = 0; tile + 1 < msv_tiles.size(); ++tile) {
        const size_t tile_begin = msv_tiles[tile];
        const size_t tile_count = msv_tiles[tile + 1] - tile_begin;
        if (tile_count == 0) continue;
        if (tile_begin < packed_group_count) {
          full_msv_packed_kernel<<<
              static_cast<unsigned>((tile_count - 1) / kWarpsPerBlock + 1),
              kThreads>>>(
              device_residues, device_sequence_offsets,
              device_compact_scores, database->device_exact_rbv,
              device_f1_profiles, database->device_profiles, device_tjb,
              device_candidates, device_selected, device_selected_offsets,
              tile_begin, tile_count, host_msv_offsets[tile_begin],
              static_cast<uint8_t *>(workspace->device_dp),
              device_msv_inputs);
        } else {
          const size_t scalar_begin = tile_begin - packed_group_count;
          full_msv_kernel<<<
              static_cast<unsigned>((tile_count - 1) / kWarpsPerBlock + 1),
              kThreads>>>(
              device_residues, device_sequence_offsets,
              device_compact_scores, database->device_exact_rbv,
              device_f1_profiles, database->device_profiles, device_tjb,
              device_candidates,
              device_selected + packed_candidate_count,
              device_selected_offsets + packed_group_count, scalar_begin,
              tile_count, host_msv_offsets[tile_begin],
              static_cast<uint8_t *>(workspace->device_dp),
              device_msv_inputs);
        }
        CUDA_RUN(cudaGetLastError());
      }
    }
    if (grow_workspace_buffer(&workspace->device_dp,
                              &workspace->dp_capacity,
                              static_cast<size_t>(maximum_vit_bytes),
                              &workspace->growth_count,
                              "cudaMalloc(post-filter Viterbi DP workspace)",
                              error, error_size) != 0)
      return -1;
  }
  prepare_bias_inputs_kernel<<<blocks, kThreads>>>(
      device_null_scores, device_f1_profiles, device_candidates,
      device_msv_inputs, candidate_count, workspace->device_states,
      workspace->device_bias_inputs);
  CUDA_RUN(cudaGetLastError());
  if (plan7_bias_filter_candidates_device(
        device_residues, device_sequence_offsets, device_length_logp,
        device_length_log1mp, device_bias_profiles, device_candidates,
        workspace->device_bias_inputs, candidate_count,
        workspace->device_bias_results,
        error, error_size) != 0)
    return -1;

  for (size_t tile = 0; tile + 1 < vit_tiles.size(); ++tile) {
    const size_t tile_begin = vit_tiles[tile];
    const size_t tile_count = vit_tiles[tile + 1] - tile_begin;
    viterbi_kernel<<<
        static_cast<unsigned>((tile_count - 1) / kWarpsPerBlock + 1),
        kThreads>>>(
        device_residues, device_sequence_offsets, database->device_profiles,
        database->device_emissions, database->device_transitions,
        device_candidates, workspace->device_states,
        workspace->device_bias_results, workspace->device_moves,
        workspace->device_vit_offsets, tile_begin, tile_count,
        host_vit_offsets[tile_begin], skip_bias_reject_viterbi ? 1 : 0,
        static_cast<int16_t *>(workspace->device_dp),
        workspace->device_vit_results);
    CUDA_RUN(cudaGetLastError());
  }
  merge_results_kernel<<<blocks, kThreads>>>(
      device_candidates, device_msv_inputs, workspace->device_states,
      workspace->device_bias_results, workspace->device_vit_results,
      candidate_count, skip_bias_reject_viterbi ? 1 : 0,
      workspace->device_results);
  CUDA_RUN(cudaGetLastError());
  if (host_reason_facts != nullptr) {
    postfilter_reason_facts_kernel<<<blocks, kThreads>>>(
        device_bias_profiles, device_candidates, device_msv_inputs,
        workspace->device_states, workspace->device_bias_inputs,
        workspace->device_bias_results, workspace->device_vit_results,
        workspace->device_results, candidate_count,
        skip_bias_reject_viterbi ? 1 : 0, reason_storage.device);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaMemcpy(host_reason_facts, reason_storage.device,
                        reason_fact_bytes, cudaMemcpyDeviceToHost));
    if (reason_statistics == nullptr) {
      set_error(error, error_size,
                "post-filter reason statistics output is null");
      return -1;
    }
    reason_statistics->candidate_count = candidate_count;
    const auto &profiles = database_host_profiles(database);
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
      const plan7_bias_candidate mapping = host_candidates[candidate];
      const uint64_t length = host_sequence_lengths[mapping.sequence_index];
      const uint64_t model_length =
          static_cast<uint64_t>(profiles[mapping.profile_index].model_length);
      uint64_t cells = 0;
      if (!checked_multiply(length, model_length, &cells)) {
        set_error(error, error_size,
                  "post-filter reason work-cell product overflow");
        return -1;
      }
      if (host_reason_facts[candidate] &
          PLAN7_POSTFILTER_REASON_FULL_MSV_EXECUTED) {
        ++reason_statistics->full_msv_execution_count;
        if (!checked_add(reason_statistics->full_msv_work_cells, cells,
                         &reason_statistics->full_msv_work_cells)) {
          set_error(error, error_size,
                    "post-filter full-MSV work-cell total overflow");
          return -1;
        }
      }
      if (host_reason_facts[candidate] &
          PLAN7_POSTFILTER_REASON_VITERBI_EXECUTED) {
        ++reason_statistics->viterbi_execution_count;
        if (!checked_add(reason_statistics->viterbi_work_cells, cells,
                         &reason_statistics->viterbi_work_cells)) {
          set_error(error, error_size,
                    "post-filter Viterbi work-cell total overflow");
          return -1;
        }
      }
    }
    if (!checked_add(reason_statistics->full_msv_work_cells,
                     reason_statistics->viterbi_work_cells,
                     &reason_statistics->work_cells)) {
      set_error(error, error_size,
                "post-filter work-cell total overflow");
      return -1;
    }
  }
  CUDA_RUN(cudaMemcpy(host_results, workspace->device_results, post_result_bytes,
                      cudaMemcpyDeviceToHost));
  if (workspace->resident_generation == UINT64_MAX) {
    set_error(error, error_size, "post-filter resident generation overflow");
    return -1;
  }
  ++workspace->resident_generation;
  workspace->resident_device_candidates = device_candidates;
  workspace->resident_host_candidates = host_candidates;
  workspace->resident_host_results = host_results;
  workspace->resident_source_count = candidate_count;
  workspace->resident_results_valid = true;
  return 0;
#undef CUDA_RUN
}

}  // namespace

extern "C" int plan7_postfilter_workspace_compact_f2(
    plan7_postfilter_workspace *workspace, uint64_t batch_generation,
    const plan7_ssv_profile *profiles, const float *m_mu, const float *m_lambda,
    const float *v_mu, const float *v_lambda, size_t profile_count, double f2,
    int host_environment_attested, plan7_postfilter_f2_resident_view *view,
    char *error, size_t error_size) {
  if (workspace == nullptr || view == nullptr || batch_generation == 0 ||
      (profile_count != 0 &&
       (profiles == nullptr || m_mu == nullptr || m_lambda == nullptr ||
        v_mu == nullptr || v_lambda == nullptr)) ||
      !std::isfinite(f2) || f2 < 0.0 || f2 > 1.0) {
    set_error(error, error_size, "invalid resident F2 compaction request");
    return -1;
  }
  std::memset(view, 0, sizeof(*view));
  workspace->f2_view_valid = false;
  const auto total_begin = std::chrono::steady_clock::now();
  plan7_postfilter_f2_statistics statistics{};
  statistics.source_count = workspace->resident_source_count;
  statistics.run_count = 1;
  if (!workspace->resident_results_valid) {
    set_error(error, error_size, "resident post-filter results are unavailable");
    return -1;
  }
  if (workspace->resident_source_count > UINT32_MAX) {
    set_error(error, error_size, "resident F2 source count exceeds uint32");
    return -1;
  }
  if (workspace->resident_source_count != 0 && profile_count == 0) {
    set_error(error, error_size, "resident F2 sources have no profiles");
    return -1;
  }

  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != workspace->device_ordinal) {
    set_error(error, error_size,
              "resident F2 workspace belongs to a different CUDA device");
    return -1;
  }

  const auto compile_begin = std::chrono::steady_clock::now();
  try {
    workspace->host_f2_profiles.resize(profile_count);
  } catch (...) {
    set_error(error, error_size, "resident F2 profile allocation failed");
    return -1;
  }
  bool supported = host_environment_attested == 1;
  for (size_t profile = 0; profile < profile_count; ++profile) {
    plan7_f2_threshold msv{};
    plan7_f2_threshold viterbi{};
    if (plan7_postfilter_compile_f2_threshold(
            m_mu[profile], m_lambda[profile], f2, &msv) != 0 ||
        plan7_postfilter_compile_f2_threshold(
            v_mu[profile], v_lambda[profile], f2, &viterbi) != 0) {
      set_error(error, error_size, "resident F2 threshold compilation failed");
      return -1;
    }
    if (!std::isfinite(profiles[profile].scale) ||
        profiles[profile].scale <= 0.0f || !msv.supported ||
        !viterbi.supported) {
      supported = false;
      ++statistics.unsupported_profile_count;
    } else {
      ++statistics.compiled_profile_count;
    }
    workspace->host_f2_profiles[profile] = {
        profiles[profile].scale, msv.threshold_bits,
        viterbi.threshold_bits};
  }
  if (!host_environment_attested)
    statistics.unsupported_profile_count = profile_count;
  statistics.compile_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - compile_begin).count();

  workspace->f2_batch_generation = batch_generation;
  workspace->f2_profile_count = profile_count;
  workspace->f2_selected_count = 0;
  workspace->f2_selected_source_hash = hash_selected_sources(nullptr, 0);
  workspace->host_f2_selected_sources.clear();
  workspace->f2_statistics = statistics;
  workspace->f2_view_valid = true;
  if (!supported) {
    workspace->f2_statistics.total_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - total_begin).count();
    view->batch_generation = batch_generation;
    view->workspace_generation = workspace->resident_generation;
    view->device_ordinal = current_device;
    view->supported = 0;
    view->profile_count = profile_count;
    view->source_count = workspace->resident_source_count;
    view->host_candidates = workspace->resident_host_candidates;
    view->host_results = workspace->resident_host_results;
    view->device_candidates = workspace->resident_device_candidates;
    view->device_results = workspace->device_results;
    view->owner = workspace;
    view->statistics = workspace->f2_statistics;
    return 0;
  }

  const size_t source_count = workspace->resident_source_count;
  const size_t word_count = (source_count + 31) / 32;
  size_t profile_bytes = 0;
  size_t mask_bytes = 0;
  size_t rank_bytes = 0;
  if (!checked_bytes(profile_count, sizeof(F2DeviceProfile), &profile_bytes) ||
      !checked_bytes(word_count, sizeof(uint32_t), &mask_bytes) ||
      !checked_bytes(word_count + 1, sizeof(uint64_t), &rank_bytes)) {
    set_error(error, error_size, "resident F2 workspace size overflow");
    return -1;
  }
  if (grow_workspace_buffer(
          &workspace->device_moves, &workspace->moves_capacity,
          profile_bytes, &workspace->growth_count,
          "cudaMalloc(resident F2 profiles)", error, error_size) != 0 ||
      grow_workspace_buffer(
          &workspace->device_bias_inputs, &workspace->bias_inputs_capacity,
          mask_bytes, &workspace->growth_count,
          "cudaMalloc(resident F2 masks)", error, error_size) != 0 ||
      grow_workspace_buffer(
          &workspace->device_msv_offsets, &workspace->msv_offsets_capacity,
          rank_bytes, &workspace->growth_count,
          "cudaMalloc(resident F2 counts)", error, error_size) != 0 ||
      grow_workspace_buffer(
          &workspace->device_vit_offsets, &workspace->vit_offsets_capacity,
          rank_bytes, &workspace->growth_count,
          "cudaMalloc(resident F2 ranks)", error, error_size) != 0)
    return -1;

  auto *device_profiles =
      reinterpret_cast<F2DeviceProfile *>(workspace->device_moves);
  auto *device_masks =
      reinterpret_cast<uint32_t *>(workspace->device_bias_inputs);
  auto *device_counts = workspace->device_msv_offsets;
  auto *device_ranks = workspace->device_vit_offsets;
  const auto upload_begin = std::chrono::steady_clock::now();
  if (profile_bytes != 0) {
    status = cudaMemcpy(device_profiles, workspace->host_f2_profiles.data(),
                        profile_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      set_cuda_error(error, error_size, "upload resident F2 profiles", status);
      return -1;
    }
  }
  statistics.upload_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - upload_begin).count();

  size_t scan_workspace_bytes = 0;
  status = cub::DeviceScan::ExclusiveSum(
      nullptr, scan_workspace_bytes, device_counts, device_ranks,
      static_cast<int>(word_count + 1));
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "CUB resident F2 scan workspace query", status);
    return -1;
  }
  if (grow_workspace_buffer(
          &workspace->device_dp, &workspace->dp_capacity,
          scan_workspace_bytes, &workspace->growth_count,
          "cudaMalloc(resident F2 scan workspace)", error,
          error_size) != 0)
    return -1;

  cudaEvent_t begin_event = nullptr;
  cudaEvent_t classify_event = nullptr;
  cudaEvent_t scan_event = nullptr;
#define F2_CUDA(call, label)                                                  \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      if (begin_event != nullptr) cudaEventDestroy(begin_event);              \
      if (classify_event != nullptr) cudaEventDestroy(classify_event);        \
      if (scan_event != nullptr) cudaEventDestroy(scan_event);                \
      set_cuda_error(error, error_size, (label), status);                     \
      workspace->f2_view_valid = false;                                      \
      return -1;                                                              \
    }                                                                         \
  } while (0)
  F2_CUDA(cudaEventCreate(&begin_event), "create resident F2 begin event");
  F2_CUDA(cudaEventCreate(&classify_event),
          "create resident F2 classify event");
  F2_CUDA(cudaEventCreate(&scan_event), "create resident F2 scan event");
  F2_CUDA(cudaEventRecord(begin_event), "record resident F2 begin event");
  if (word_count != 0) {
    const size_t thread_count = word_count * 32;
    const size_t block_count = (thread_count + kThreads - 1) / kThreads;
    classify_f2_words_kernel<<<static_cast<unsigned>(block_count), kThreads>>>(
        workspace->resident_device_candidates, workspace->device_results,
        device_profiles, profile_count, source_count, word_count, device_masks,
        device_counts);
    F2_CUDA(cudaGetLastError(), "launch resident F2 classification");
  } else {
    F2_CUDA(cudaMemset(device_counts, 0, sizeof(uint64_t)),
            "initialize empty resident F2 count");
  }
  F2_CUDA(cudaEventRecord(classify_event),
          "record resident F2 classify event");
  F2_CUDA(cub::DeviceScan::ExclusiveSum(
              workspace->device_dp, scan_workspace_bytes, device_counts,
              device_ranks, static_cast<int>(word_count + 1)),
          "scan resident F2 masks");
  F2_CUDA(cudaEventRecord(scan_event), "record resident F2 scan event");
  F2_CUDA(cudaEventSynchronize(scan_event),
          "synchronize resident F2 scan event");
  F2_CUDA(cudaEventElapsedTime(
              &statistics.kernel_milliseconds, begin_event, classify_event),
          "time resident F2 classification");
  F2_CUDA(cudaEventElapsedTime(
              &statistics.scan_milliseconds, classify_event, scan_event),
          "time resident F2 scan");
  cudaEventDestroy(begin_event);
  cudaEventDestroy(classify_event);
  cudaEventDestroy(scan_event);
  begin_event = classify_event = scan_event = nullptr;

  uint64_t selected_count64 = 0;
  const auto count_download_begin = std::chrono::steady_clock::now();
  status = cudaMemcpy(&selected_count64, device_ranks + word_count,
                      sizeof(selected_count64), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "download resident F2 selected count", status);
    workspace->f2_view_valid = false;
    return -1;
  }
  if (selected_count64 > source_count || selected_count64 > UINT32_MAX) {
    set_error(error, error_size, "resident F2 selected count is invalid");
    workspace->f2_view_valid = false;
    return -1;
  }
  const size_t selected_count = static_cast<size_t>(selected_count64);
  size_t selected_bytes = 0;
  if (!checked_bytes(selected_count, sizeof(uint32_t), &selected_bytes) ||
      grow_workspace_buffer(
          &workspace->device_vit_results, &workspace->vit_results_capacity,
          selected_bytes, &workspace->growth_count,
          "cudaMalloc(resident F2 selected sources)", error,
          error_size) != 0) {
    workspace->f2_view_valid = false;
    return -1;
  }
  auto *device_selected_sources =
      reinterpret_cast<uint32_t *>(workspace->device_vit_results);
  if (word_count != 0) {
    const size_t block_count = (word_count + kThreads - 1) / kThreads;
    scatter_f2_sources_kernel<<<static_cast<unsigned>(block_count), kThreads>>>(
        device_masks, device_ranks, word_count, device_selected_sources);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      set_cuda_error(error, error_size,
                     "launch resident F2 stable scatter", status);
      workspace->f2_view_valid = false;
      return -1;
    }
  }
  try {
    workspace->host_f2_selected_sources.resize(selected_count);
  } catch (...) {
    set_error(error, error_size,
              "resident F2 host selection allocation failed");
    workspace->f2_view_valid = false;
    return -1;
  }
  if (selected_bytes != 0) {
    status = cudaMemcpy(workspace->host_f2_selected_sources.data(),
                        device_selected_sources, selected_bytes,
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      set_cuda_error(error, error_size,
                     "download resident F2 selected sources", status);
      workspace->f2_view_valid = false;
      return -1;
    }
  }
  statistics.download_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - count_download_begin).count();
  statistics.source_count = source_count;
  statistics.selected_count = selected_count;
  statistics.mask_word_count = word_count;
  statistics.selected_d2h_bytes = selected_bytes;
  statistics.total_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  workspace->f2_selected_count = selected_count;
  workspace->f2_selected_source_hash = hash_selected_sources(
      workspace->host_f2_selected_sources.data(), selected_count);
  workspace->f2_statistics = statistics;
  workspace->f2_view_valid = true;

  view->batch_generation = batch_generation;
  view->workspace_generation = workspace->resident_generation;
  view->selected_source_hash = workspace->f2_selected_source_hash;
  view->device_ordinal = current_device;
  view->supported = 1;
  view->profile_count = profile_count;
  view->source_count = source_count;
  view->selected_count = selected_count;
  view->host_selected_sources =
      workspace->host_f2_selected_sources.empty()
          ? nullptr : workspace->host_f2_selected_sources.data();
  view->host_candidates = workspace->resident_host_candidates;
  view->host_results = workspace->resident_host_results;
  view->device_candidates = workspace->resident_device_candidates;
  view->device_results = workspace->device_results;
  view->device_selected_sources =
      selected_count == 0 ? nullptr : device_selected_sources;
  view->owner = workspace;
  view->statistics = statistics;
#undef F2_CUDA
  return 0;
}

extern "C" int plan7_postfilter_f2_resident_view_validate(
    const plan7_postfilter_f2_resident_view *view,
    char *error, size_t error_size) {
  if (view == nullptr || view->owner == nullptr) {
    set_error(error, error_size, "resident F2 view is null");
    return -1;
  }
  const plan7_postfilter_workspace *workspace = view->owner;
  if (!workspace->f2_view_valid || !workspace->resident_results_valid ||
      view->workspace_generation != workspace->resident_generation ||
      view->batch_generation != workspace->f2_batch_generation ||
      view->device_ordinal != workspace->device_ordinal ||
      view->profile_count != workspace->f2_profile_count ||
      view->source_count != workspace->resident_source_count ||
      view->selected_count != workspace->f2_selected_count ||
      view->selected_source_hash != workspace->f2_selected_source_hash ||
      view->host_candidates != workspace->resident_host_candidates ||
      view->host_results != workspace->resident_host_results ||
      view->device_candidates != workspace->resident_device_candidates ||
      view->device_results != workspace->device_results ||
      (view->selected_count != 0 &&
       (view->host_selected_sources !=
            workspace->host_f2_selected_sources.data() ||
        view->device_selected_sources != reinterpret_cast<const uint32_t *>(
            workspace->device_vit_results)))) {
    set_error(error, error_size, "resident F2 view identity changed");
    return -1;
  }
  return 0;
}

extern "C" int plan7_postfilter_candidates_device_with_workspace(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, char *error, size_t error_size) {
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, nullptr, nullptr, PLAN7_GPU_EXECUTION_POLICY_AUTO,
      false, error, error_size);
}

extern "C" int plan7_postfilter_candidates_device_with_workspace_reason_facts(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, uint16_t *reason_facts,
    plan7_postfilter_reason_statistics *reason_statistics,
    char *error, size_t error_size) {
  if (reason_facts == nullptr || reason_statistics == nullptr) {
    set_error(error, error_size, "post-filter reason output is null");
    return -1;
  }
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, reason_facts, reason_statistics,
      PLAN7_GPU_EXECUTION_POLICY_AUTO, false, error, error_size);
}

extern "C" int plan7_postfilter_candidates_device_with_workspace_policy(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, int execution_policy,
    char *error, size_t error_size) {
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, nullptr, nullptr, execution_policy, false,
      error, error_size);
}

extern "C" int
plan7_postfilter_candidates_device_with_workspace_reason_facts_policy(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, uint16_t *reason_facts,
    plan7_postfilter_reason_statistics *reason_statistics,
    int execution_policy, char *error, size_t error_size) {
  if (reason_facts == nullptr || reason_statistics == nullptr) {
    set_error(error, error_size, "post-filter reason output is null");
    return -1;
  }
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, reason_facts, reason_statistics, execution_policy,
      false, error, error_size);
}

extern "C" int
plan7_postfilter_candidates_device_with_workspace_fixed_bias_policy(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, int execution_policy,
    char *error, size_t error_size) {
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, nullptr, nullptr, execution_policy, true,
      error, error_size);
}

extern "C" int
plan7_postfilter_candidates_device_with_workspace_fixed_bias_reason_facts_policy(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, uint16_t *reason_facts,
    plan7_postfilter_reason_statistics *reason_statistics,
    int execution_policy, char *error, size_t error_size) {
  if (reason_facts == nullptr || reason_statistics == nullptr) {
    set_error(error, error_size, "post-filter reason output is null");
    return -1;
  }
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, reason_facts, reason_statistics, execution_policy, true,
      error, error_size);
}

extern "C" int plan7_postfilter_candidates_device(
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, char *error, size_t error_size) {
  plan7_postfilter_workspace *workspace = nullptr;
  if (plan7_postfilter_workspace_create(&workspace, error, error_size) != 0)
    return -1;
  const int run_status = plan7_postfilter_candidates_device_with_workspace(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, error, error_size);
  char destroy_error[512] = {0};
  const int destroy_status = plan7_postfilter_workspace_destroy(
      &workspace, destroy_error, sizeof(destroy_error));
  if (run_status != 0) return run_status;
  if (destroy_status != 0) {
    set_error(error, error_size, destroy_error);
    return -1;
  }
  return 0;
}
