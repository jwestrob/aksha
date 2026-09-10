#include "ssv_cuda.h"
#include "forward_cuda.h"
#include "postfilter_cuda.h"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>

extern "C" {
#include <easel.h>
#include <esl_gumbel.h>
}

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct plan7_ssv_sequence_batch {
  uint64_t generation_id;
  int device_ordinal;
  int alphabet_size;
  int execution_policy_mode;
  int execution_policy_locked;
  size_t sequence_count;
  uint64_t *host_lengths;
  float *host_null_scores;
  float *host_bias_logp;
  float *host_bias_log1mp;
  float *host_tjb_log_terms;
  uint32_t *host_length_class_indices;
  float *host_length_class_log_terms;
  size_t length_class_count;
  uint8_t *host_tjb;
  size_t host_tjb_capacity;
  plan7_ssv_f1_profile *host_f1_profiles;
  size_t host_f1_profile_capacity;
  uint8_t *host_f1_scores;
  size_t host_f1_score_capacity;
  size_t host_f1_score_count;
  plan7_bias_candidate *host_bias_candidates;
  size_t host_bias_candidate_capacity;
  size_t *host_candidate_offsets;
  size_t host_candidate_offset_capacity;
  uint8_t *device_residues;
  size_t device_residue_capacity;
  uint64_t *device_offsets;
  size_t device_offset_capacity;
  float *device_null_scores;
  size_t device_null_score_capacity;
  float *device_bias_logp;
  size_t device_bias_logp_capacity;
  float *device_bias_log1mp;
  size_t device_bias_log1mp_capacity;
  uint32_t *device_length_class_indices;
  size_t device_length_class_index_capacity;
  uint8_t *device_f1_compact_tjb;
  size_t device_f1_compact_tjb_capacity;
  uint8_t *device_f1_raw_xe;
  size_t device_f1_raw_xe_capacity;
  uint8_t *device_tjb;
  size_t device_tjb_capacity;
  plan7_ssv_result *device_results;
  size_t device_result_capacity;
  uint8_t *device_scores;
  size_t device_score_capacity;
  plan7_ssv_profile *device_profiles;
  size_t device_profile_capacity;
  plan7_ssv_f1_profile *device_f1_profiles;
  size_t device_f1_profile_capacity;
  uint32_t *device_f1_profile_packed_scores;
  size_t device_f1_profile_packed_score_capacity;
  void *device_f1_profile_packed_quartets;
  size_t device_f1_profile_packed_quartet_capacity;
  uint32_t *device_f1_scalar_profile_indices;
  size_t device_f1_scalar_profile_index_capacity;
  uint32_t *device_candidate_words;
  size_t device_candidate_word_capacity;
  uint64_t *device_candidate_word_counts;
  size_t device_candidate_word_count_capacity;
  uint64_t *device_candidate_word_offsets;
  size_t device_candidate_word_offset_capacity;
  size_t *device_candidate_profile_offsets;
  size_t device_candidate_profile_offset_capacity;
  uint8_t *device_candidate_scan_workspace;
  size_t device_candidate_scan_workspace_capacity;
  plan7_bias_profile *device_bias_profiles;
  size_t device_bias_profile_capacity;
  plan7_bias_candidate *device_bias_candidates;
  size_t device_bias_candidate_capacity;
  plan7_bias_ssv_input *device_bias_ssv_inputs;
  size_t device_bias_ssv_input_capacity;
  plan7_bias_result *device_bias_results;
  size_t device_bias_result_capacity;
  float cached_tjb_scale;
  int tjb_cache_valid;
  size_t cached_f1_profile_count;
  size_t cached_f1_candidate_count;
  int f1_cache_valid;
  int f1_device_candidates_valid;
  int f1_raw_xe_valid;
  int f1_compact_ssv_inputs_valid;
  int bias_length_terms_device_valid;
  int host_float_environment_valid;
  uint64_t input_device_bytes;
  plan7_postfilter_workspace *postfilter_workspace;
  plan7_forward_workspace *forward_workspace;
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
  uint64_t execution_policy_f1_run_count;
  uint64_t f1_raw_xe_run_count;
  uint64_t f1_raw_xe_logical_pair_count;
  uint64_t f1_raw_xe_sidecar_bytes_written;
  uint64_t f1_raw_xe_candidate_gather_count;
  uint64_t f1_candidate_ssv_replay_count;
  uint64_t f1_candidate_ssv_replay_avoided_count;
  uint64_t f1_raw_xe_fallback_run_count;
};

namespace {

std::atomic<uint64_t> next_sequence_batch_generation{1};

uint64_t allocate_sequence_batch_generation()
{
  uint64_t generation = next_sequence_batch_generation.fetch_add(
    1, std::memory_order_relaxed);
  if (generation == 0)
    generation = next_sequence_batch_generation.fetch_add(
      1, std::memory_order_relaxed);
  return generation;
}

}  // namespace

static_assert(sizeof(plan7_ssv_result) == 6,
              "plan7_ssv_result ABI size changed");
static_assert(offsetof(plan7_ssv_result, numerator) == 4,
              "plan7_ssv_result ABI layout changed");
static_assert(sizeof(float) == 4, "plan7 profile ABI requires binary32 float");
static_assert(sizeof(plan7_ssv_profile) == 32,
              "plan7_ssv_profile ABI size changed");
static_assert(sizeof(plan7_ssv_f1_profile) == 48,
              "plan7 fused F1 profile ABI size changed");
static_assert(offsetof(plan7_ssv_profile, score_offset) == 0 &&
              offsetof(plan7_ssv_profile, score_count) == 8 &&
              offsetof(plan7_ssv_profile, score_stride) == 16 &&
              offsetof(plan7_ssv_profile, model_length) == 20 &&
              offsetof(plan7_ssv_profile, tbm) == 24 &&
              offsetof(plan7_ssv_profile, scale) == 28,
              "plan7_ssv_profile ABI layout changed");

namespace {

constexpr int kThreads = 256;
constexpr int kSequencesPerBlock = 4;
constexpr int kExtraScoreVectors = 17;
constexpr uint64_t kMaximumTargetLength = 100000;
constexpr uint64_t kMaximumModelLength = 100000;
constexpr float kEvparamUnset = -99999.0f;
constexpr size_t kProfilePackedMinimumProfiles = 32;
constexpr int kProfilesPerPackedWord = 4;
constexpr size_t kLengthClassMinimumSequences = 256;

struct ProfilePackedQuartet {
  uint64_t score_word_offset;
  uint32_t maximum_model_length;
  uint32_t profile_indices[kProfilesPerPackedWord];
};

static_assert(sizeof(ProfilePackedQuartet) == 32,
              "profile-packed quartet layout changed");

/* Exact requested device bytes for the current amino-profile packers. The
 * descriptor and row constants mirror private structs/layouts in
 * postfilter_cuda.cu and forward_cuda.cu. Their boundary tests are the ABI
 * tripwire for this allocation-free estimator. */
constexpr uint64_t kAminoAlphabetSize = 29;
constexpr uint64_t kViterbiDescriptorBytes = 96;
constexpr uint64_t kViterbiBytesPerQ = 2368;
constexpr uint64_t kForwardDescriptorBytes = 32;
constexpr uint64_t kForwardBytesPerQ = 592;
constexpr uint64_t kBiasDescriptorBytes = 272;
static_assert(sizeof(plan7_bias_profile) == kBiasDescriptorBytes,
              "bias profile footprint changed");
static_assert(PLAN7_SSV_CAPACITY_POSTFILTER_RESULTS -
                  PLAN7_SSV_CAPACITY_POSTFILTER_STATES + 1 ==
                PLAN7_POSTFILTER_CAPACITY_COUNT,
              "post-filter capacity mapping changed");
static_assert(PLAN7_SSV_CAPACITY_FORWARD_GATHERED -
                  PLAN7_SSV_CAPACITY_FORWARD_CANDIDATE_PROFILES + 1 ==
                PLAN7_FORWARD_CAPACITY_COUNT,
              "Forward capacity mapping changed");

__device__ __forceinline__ int
saturating_signed_subtract(int left, int right)
{
  int value = left - right;
  return value > INT8_MAX ? INT8_MAX : (value < INT8_MIN ? INT8_MIN : value);
}

__device__ __forceinline__ plan7_ssv_result
ssv_result_from_raw_xe(unsigned raw_xE,
                       uint8_t length_tjb,
                       uint8_t tbm,
                       uint8_t tec,
                       uint8_t base,
                       uint8_t bias,
                       bool empty)
{
  plan7_ssv_result result = {
    static_cast<uint8_t>(raw_xE),
    static_cast<uint8_t>(empty ? PLAN7_SSV_EMPTY : PLAN7_SSV_OK),
    length_tjb, 0, 0
  };
  if (empty) return result;
  if (static_cast<unsigned>(length_tjb) + tbm + tec + bias >= 127U) {
    result.status = PLAN7_SSV_ENORESULT;
  } else if (raw_xE >= 255U - bias) {
    result.status =
      static_cast<int>(base) - static_cast<int>(length_tjb) -
          static_cast<int>(tbm) < 128
        ? PLAN7_SSV_ENORESULT
        : PLAN7_SSV_ERANGE;
  } else {
    unsigned adjusted = raw_xE + base - length_tjb - tbm;
    adjusted -= 128U;
    if (adjusted >= 255U - bias) {
      result.status = PLAN7_SSV_ERANGE;
    } else {
      const unsigned xJ = adjusted - tec;
      if (xJ > base) {
        result.status = PLAN7_SSV_ENORESULT;
      } else {
        result.numerator = static_cast<int16_t>(
          static_cast<int>(xJ) - static_cast<int>(length_tjb) -
          static_cast<int>(base));
      }
    }
  }
  return result;
}

template<bool CompactScores>
__device__ __forceinline__ void
ssv_filter_block(const uint8_t *scores,
                 int score_stride,
                 int model_length,
                 const uint8_t *residues,
                 const uint64_t *offsets,
                 size_t sequence,
                 uint8_t length_tjb,
                 uint8_t tbm,
                 uint8_t tec,
                 uint8_t base,
                 uint8_t bias,
                 plan7_ssv_result *result_out,
                 unsigned *maxima)
{
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  const int Q = max(2, (model_length + 15) / 16);
  unsigned local_maximum = 128;

  if (length == 0) {
    if (threadIdx.x == 0)
      *result_out = {128, PLAN7_SSV_EMPTY, length_tjb, 0, 0};
    return;
  }

  const int diagonal_count = model_length + length - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    int value = INT8_MIN;

    while (i < length && k < model_length) {
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      unsigned raw_cost;
      if constexpr (CompactScores) {
        raw_cost = scores[static_cast<size_t>(k) * score_stride + residue];
      } else {
        const int q = k % Q;
        const int lane = k / Q;
        raw_cost =
          scores[static_cast<size_t>(residue) * score_stride + 16 * q + lane];
      }
      const int cost = raw_cost < 128 ? static_cast<int>(raw_cost)
                                     : static_cast<int>(raw_cost) - 256;
      value = saturating_signed_subtract(value, cost);
      const unsigned raw_value = value < 0 ? static_cast<unsigned>(value + 256)
                                           : static_cast<unsigned>(value);
      local_maximum = max(local_maximum, raw_value);
      ++i;
      ++k;
    }
  }

  for (int width = 16; width > 0; width >>= 1)
    local_maximum = max(
      local_maximum,
      __shfl_down_sync(UINT32_MAX, local_maximum, width));
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) maxima[warp] = local_maximum;
  __syncthreads();
  if (warp == 0) {
    local_maximum = lane < blockDim.x / 32 ? maxima[lane] : 128;
    for (int width = 16; width > 0; width >>= 1)
      local_maximum = max(
        local_maximum,
        __shfl_down_sync(UINT32_MAX, local_maximum, width));
    if (lane == 0) maxima[0] = local_maximum;
  }

  if (threadIdx.x == 0) {
    const unsigned raw_xE = maxima[0];
    *result_out = ssv_result_from_raw_xe(
      raw_xE, length_tjb, tbm, tec, base, bias, false);
  }
}

__global__ void
ssv_kernel(const uint8_t *scores,
           int score_stride,
           int model_length,
           const uint8_t *residues,
           const uint64_t *offsets,
           const uint8_t *tjb,
           uint8_t tbm,
           uint8_t tec,
           uint8_t base,
           uint8_t bias,
           plan7_ssv_result *results)
{
  __shared__ unsigned maxima[kThreads];
  const size_t sequence = static_cast<size_t>(blockIdx.x);
  ssv_filter_block<false>(scores, score_stride, model_length, residues, offsets,
                          sequence, tjb[sequence], tbm, tec, base, bias,
                          &results[sequence], maxima);
}

__global__ void
ssv_many_kernel(const uint8_t *packed_scores,
                const plan7_ssv_profile *profiles,
                size_t sequence_count,
                const uint8_t *residues,
                const uint64_t *offsets,
                const uint8_t *profile_major_tjb,
                plan7_ssv_result *profile_major_results)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ plan7_ssv_profile profile_descriptor;
  const size_t sequence = static_cast<size_t>(blockIdx.x);
  const size_t profile = static_cast<size_t>(blockIdx.y);
  const size_t result_index = profile * sequence_count + sequence;

  if (threadIdx.x == 0) profile_descriptor = profiles[profile];
  __syncthreads();
  ssv_filter_block<true>(
    packed_scores + profile_descriptor.score_offset,
    profile_descriptor.score_stride,
    profile_descriptor.model_length,
    residues,
    offsets,
    sequence,
    profile_major_tjb[result_index],
    profile_descriptor.tbm,
    profile_descriptor.tec,
    profile_descriptor.base,
    profile_descriptor.bias,
    &profile_major_results[result_index],
    maxima);
}

__device__ __forceinline__ bool
f1_requires_cpu(const plan7_ssv_result result,
                float null_score,
                const plan7_ssv_f1_profile profile)
{
  if (result.status != PLAN7_SSV_OK)
    return true;
  if (profile.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_REJECT)
    return false;
  if (profile.cutoff_mode != PLAN7_F1_CUTOFF_SCORE ||
      !isfinite(null_score) || !isfinite(profile.profile.scale) ||
      profile.profile.scale <= 0.0f ||
      !isfinite(profile.cutoff_bit_score))
    return true;

  /* Match HMMER's binary32/binary64 score evaluation order explicitly. */
  float score = __int2float_rn(static_cast<int>(result.numerator));
  score = __fdiv_rn(score, profile.profile.scale);
  score = __fsub_rn(score, 3.0f);
  const float delta = __fsub_rn(score, null_score);
  const double bit_score_double = __ddiv_rn(
    static_cast<double>(delta), 0.69314718055994529);
  const float bit_score = __double2float_rn(bit_score_double);
  if (!isfinite(score) || !isfinite(bit_score))
    return true;
  return bit_score >= profile.cutoff_bit_score;
}

__global__ void
ssv_f1_mask_many_kernel(const uint8_t *packed_scores,
                        const plan7_ssv_f1_profile *profiles,
                        size_t sequence_count,
                        const uint8_t *residues,
                        const uint64_t *offsets,
                        const float *null_scores,
                        const uint8_t *tjb,
                        size_t words_per_profile,
                        uint32_t *candidate_words,
                        uint8_t *raw_xe)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ plan7_ssv_f1_profile profile_descriptor;
  const size_t profile = static_cast<size_t>(blockIdx.y);

  if (threadIdx.x == 0) profile_descriptor = profiles[profile];
  __syncthreads();

  const size_t first_sequence =
    static_cast<size_t>(blockIdx.x) * kSequencesPerBlock;
  for (int iteration = 0; iteration < kSequencesPerBlock; ++iteration) {
    const size_t sequence = first_sequence + static_cast<size_t>(iteration);
    if (sequence >= sequence_count) break;

    if (profile_descriptor.cutoff_mode == PLAN7_F1_CUTOFF_INVALID ||
        profile_descriptor.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_CPU) {
      if (threadIdx.x == 0) {
        const size_t word = profile * words_per_profile + sequence / 32;
        atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
      }
      continue;
    }

    /* A certified reject needs ordinary unsigned arithmetic below to remain
     * monotone. Real HMMER profiles satisfy this bound. Unknown raw callers
     * are retained for exact SSV instead of relying on unsigned underflow. */
    const unsigned transition_total =
      static_cast<unsigned>(
        tjb[profile_descriptor.tjb_offset + sequence]) +
      static_cast<unsigned>(profile_descriptor.profile.tbm) +
      static_cast<unsigned>(profile_descriptor.profile.tec);
    const bool force_retain =
      static_cast<unsigned>(profile_descriptor.profile.base) <
      transition_total;
    if (force_retain && raw_xe == nullptr) {
      if (threadIdx.x == 0) {
        const size_t word = profile * words_per_profile + sequence / 32;
        atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
      }
      continue;
    }

    plan7_ssv_result result;
    ssv_filter_block<true>(
      packed_scores + profile_descriptor.profile.score_offset,
      profile_descriptor.profile.score_stride,
      profile_descriptor.profile.model_length,
      residues,
      offsets,
      sequence,
      tjb[profile_descriptor.tjb_offset + sequence],
      profile_descriptor.profile.tbm,
      profile_descriptor.profile.tec,
      profile_descriptor.profile.base,
      profile_descriptor.profile.bias,
      &result,
      maxima);
    if (threadIdx.x == 0) {
      if (raw_xe != nullptr)
        raw_xe[profile * sequence_count + sequence] = result.xE;
      if (force_retain ||
          f1_requires_cpu(result, null_scores[sequence], profile_descriptor)) {
        const size_t word = profile * words_per_profile + sequence / 32;
        atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
      }
    }
    __syncthreads();
  }
}

/* This deliberately mirrors ssv_filter_block instead of changing the
 * production primitive. Each reduced score is the minimum signed exact cost
 * over its residue class, so signed saturating subtraction produces an
 * optimistic state. If that optimistic path could cross the signed-byte wrap,
 * the unchanged overflow/J-state guards retain it for exact SSV. */
__device__ __forceinline__ void
ssv_f0_filter_block(const uint8_t *coarse_scores,
                    size_t coarse_score_offset,
                    size_t class_count,
                    const uint8_t *residue_classes,
                    int model_length,
                    const uint8_t *residues,
                    const uint64_t *offsets,
                    size_t sequence,
                    uint8_t length_tjb,
                    uint8_t tbm,
                    uint8_t tec,
                    uint8_t base,
                    uint8_t bias,
                    plan7_ssv_result *result_out,
                    unsigned *maxima)
{
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  unsigned local_maximum = 128;

  if (length == 0) {
    if (threadIdx.x == 0)
      *result_out = {128, PLAN7_SSV_EMPTY, length_tjb, 0, 0};
    return;
  }

  const int diagonal_count = model_length + length - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    int value = INT8_MIN;

    while (i < length && k < model_length) {
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      const unsigned residue_class = residue_classes[residue];
      const unsigned raw_cost = coarse_scores[
        coarse_score_offset + static_cast<size_t>(k) * class_count +
        residue_class];
      const int cost = raw_cost < 128 ? static_cast<int>(raw_cost)
                                     : static_cast<int>(raw_cost) - 256;
      value = saturating_signed_subtract(value, cost);
      const unsigned raw_value = value < 0 ? static_cast<unsigned>(value + 256)
                                           : static_cast<unsigned>(value);
      local_maximum = max(local_maximum, raw_value);
      ++i;
      ++k;
    }
  }

  for (int width = 16; width > 0; width >>= 1)
    local_maximum = max(
      local_maximum,
      __shfl_down_sync(UINT32_MAX, local_maximum, width));
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) maxima[warp] = local_maximum;
  __syncthreads();
  if (warp == 0) {
    local_maximum = lane < blockDim.x / 32 ? maxima[lane] : 128;
    for (int width = 16; width > 0; width >>= 1)
      local_maximum = max(
        local_maximum,
        __shfl_down_sync(UINT32_MAX, local_maximum, width));
    if (lane == 0) maxima[0] = local_maximum;
  }

  if (threadIdx.x == 0) {
    const unsigned raw_xE = maxima[0];
    plan7_ssv_result result = {
      static_cast<uint8_t>(raw_xE), PLAN7_SSV_OK,
      static_cast<uint8_t>(length_tjb), 0, 0
    };
    if (length_tjb + tbm + tec + bias >= 127) {
      result.status = PLAN7_SSV_ENORESULT;
    } else if (raw_xE >= 255U - bias) {
      result.status =
        static_cast<int>(base) - static_cast<int>(length_tjb) -
            static_cast<int>(tbm) < 128
          ? PLAN7_SSV_ENORESULT
          : PLAN7_SSV_ERANGE;
    } else {
      unsigned adjusted = raw_xE + base - length_tjb - tbm;
      adjusted -= 128;
      if (adjusted >= 255U - bias) {
        result.status = PLAN7_SSV_ERANGE;
      } else {
        const unsigned xJ = adjusted - tec;
        if (xJ > base) {
          result.status = PLAN7_SSV_ENORESULT;
        } else {
          result.numerator = static_cast<int16_t>(
            static_cast<int>(xJ) - static_cast<int>(length_tjb) -
            static_cast<int>(base));
        }
      }
    }
    *result_out = result;
  }
}

__global__ void
ssv_f0_mask_many_kernel(const uint8_t *coarse_scores,
                        const plan7_ssv_f1_profile *profiles,
                        size_t sequence_count,
                        const uint8_t *residues,
                        const uint64_t *offsets,
                        const float *null_scores,
                        const uint8_t *tjb,
                        const uint8_t *residue_classes,
                        size_t exact_alphabet_size,
                        size_t class_count,
                        size_t words_per_profile,
                        uint32_t *candidate_words)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ plan7_ssv_f1_profile profile_descriptor;
  const size_t profile = static_cast<size_t>(blockIdx.y);
  if (threadIdx.x == 0) profile_descriptor = profiles[profile];
  __syncthreads();

  const size_t first_sequence =
    static_cast<size_t>(blockIdx.x) * kSequencesPerBlock;
  for (int iteration = 0; iteration < kSequencesPerBlock; ++iteration) {
    const size_t sequence = first_sequence + static_cast<size_t>(iteration);
    if (sequence >= sequence_count) break;
    if (profile_descriptor.cutoff_mode == PLAN7_F1_CUTOFF_INVALID ||
        profile_descriptor.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_CPU) {
      if (threadIdx.x == 0) {
        const size_t word = profile * words_per_profile + sequence / 32;
        atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
      }
      continue;
    }

    plan7_ssv_result result;
    const size_t model_offset =
      static_cast<size_t>(profile_descriptor.profile.score_offset) /
      exact_alphabet_size;
    ssv_f0_filter_block(
      coarse_scores, model_offset * class_count, class_count,
      residue_classes, profile_descriptor.profile.model_length, residues,
      offsets, sequence,
      tjb[profile_descriptor.tjb_offset + sequence],
      profile_descriptor.profile.tbm, profile_descriptor.profile.tec,
      profile_descriptor.profile.base, profile_descriptor.profile.bias,
      &result, maxima);
    if (threadIdx.x == 0 &&
        f1_requires_cpu(result, null_scores[sequence], profile_descriptor)) {
      const size_t word = profile * words_per_profile + sequence / 32;
      atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
    }
    __syncthreads();
  }
}

__device__ __forceinline__ plan7_ssv_result
ssv_result_for_local_gain(unsigned gain,
                          uint8_t length_tjb,
                          const plan7_ssv_profile profile)
{
  const unsigned raw_xE = 128U + gain;
  return ssv_result_from_raw_xe(
    raw_xE, length_tjb, profile.tbm, profile.tec, profile.base,
    profile.bias, false);
}

/* Return the first pre-wrap integer gain that the exact F1 decision cannot
 * reject. Zero means that this pair is unsupported/unconditional. */
__device__ __forceinline__ int
compile_seed_required_gain(float null_score,
                           uint8_t length_tjb,
                           const plan7_ssv_f1_profile profile)
{
  if (profile.cutoff_mode == PLAN7_F1_CUTOFF_INVALID ||
      profile.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_CPU ||
      !isfinite(null_score) ||
      static_cast<unsigned>(profile.profile.base) <
        static_cast<unsigned>(length_tjb) + profile.profile.tbm +
          profile.profile.tec)
    return 0;

  int first_required = -1;
  for (int gain = 0; gain < 128; ++gain) {
    const plan7_ssv_result result = ssv_result_for_local_gain(
      static_cast<unsigned>(gain), length_tjb, profile.profile);
    const bool required = f1_requires_cpu(result, null_score, profile);
    if (required && first_required < 0)
      first_required = gain;
    else if (!required && first_required >= 0)
      return 0;
  }
  return first_required > 0 ? first_required : 0;
}

template<int MaximumWordLength>
__device__ __forceinline__ bool
ssv_has_bounded_seed(const uint8_t *scores,
                     int score_stride,
                     int model_length,
                     const uint8_t *residues,
                     const uint64_t *offsets,
                     size_t sequence,
                     int threshold)
{
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  const int diagonal_count = model_length + length - 1;
  bool found = false;

  for (int diagonal = threadIdx.x;
       diagonal < diagonal_count && !found;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    int prefix = 0;
    int queue_values[MaximumWordLength + 1];
    int queue_indices[MaximumWordLength + 1];
    int head = 0;
    int tail = 1;
    int step = 0;
    queue_values[0] = 0;
    queue_indices[0] = 0;

    while (i < length && k < model_length) {
      ++step;
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      const unsigned raw_cost = scores[
        static_cast<size_t>(k) * score_stride + residue];
      const int cost = raw_cost < 128 ? static_cast<int>(raw_cost)
                                     : static_cast<int>(raw_cost) - 256;
      prefix -= cost;

      while (head < tail &&
             queue_indices[head % (MaximumWordLength + 1)] <
               step - MaximumWordLength)
        ++head;
      if (head < tail &&
          prefix - queue_values[head % (MaximumWordLength + 1)] >= threshold) {
        found = true;
        break;
      }
      while (head < tail &&
             queue_values[(tail - 1) % (MaximumWordLength + 1)] >= prefix)
        --tail;
      queue_values[tail % (MaximumWordLength + 1)] = prefix;
      queue_indices[tail % (MaximumWordLength + 1)] = step;
      ++tail;
      ++i;
      ++k;
    }
  }
  return found;
}

template<int MaximumWordLength>
__global__ void
ssv_seed_mask_many_kernel(const uint8_t *packed_scores,
                          const plan7_ssv_f1_profile *profiles,
                          size_t sequence_count,
                          const uint8_t *residues,
                          const uint64_t *offsets,
                          const float *null_scores,
                          const uint8_t *tjb,
                          size_t indexed_alphabet_size,
                          size_t words_per_profile,
                          uint32_t *candidate_words,
                          uint32_t *unsupported_words)
{
  __shared__ plan7_ssv_f1_profile profile_descriptor;
  __shared__ int required_gain;
  __shared__ int seed_threshold;
  const size_t profile = static_cast<size_t>(blockIdx.y);
  if (threadIdx.x == 0) profile_descriptor = profiles[profile];
  __syncthreads();

  const size_t first_sequence =
    static_cast<size_t>(blockIdx.x) * kSequencesPerBlock;
  for (int iteration = 0; iteration < kSequencesPerBlock; ++iteration) {
    const size_t sequence = first_sequence + static_cast<size_t>(iteration);
    if (sequence >= sequence_count) break;
    const uint64_t target_start = offsets[sequence];
    const uint64_t target_stop = offsets[sequence + 1];
    bool noncanonical = target_start == target_stop;
    for (uint64_t index = target_start + threadIdx.x;
         index < target_stop;
         index += blockDim.x)
      noncanonical = noncanonical || residues[index] >= indexed_alphabet_size;
    const int unsupported = __syncthreads_or(noncanonical);

    if (threadIdx.x == 0) {
      required_gain = unsupported ? 0 : compile_seed_required_gain(
        null_scores[sequence],
        tjb[profile_descriptor.tjb_offset + sequence],
        profile_descriptor);
      if (required_gain > 0) {
        const uint64_t target_length = target_stop - target_start;
        const uint64_t maximum_alignment = min(
          target_length,
          static_cast<uint64_t>(profile_descriptor.profile.model_length));
        const uint64_t chunks =
          (maximum_alignment + MaximumWordLength - 1) / MaximumWordLength;
        seed_threshold = static_cast<int>(
          (static_cast<uint64_t>(required_gain) + chunks - 1) / chunks);
      } else {
        seed_threshold = 0;
      }
    }
    __syncthreads();

    bool found = required_gain <= 0;
    if (!found) {
      found = ssv_has_bounded_seed<MaximumWordLength>(
        packed_scores + profile_descriptor.profile.score_offset,
        profile_descriptor.profile.score_stride,
        profile_descriptor.profile.model_length,
        residues, offsets, sequence, seed_threshold);
      found = __syncthreads_or(found);
    }
    if (threadIdx.x == 0 && found) {
      const size_t word = profile * words_per_profile + sequence / 32;
      const uint32_t bit = UINT32_C(1) << (sequence % 32);
      atomicOr(&candidate_words[word], bit);
      if (required_gain <= 0) atomicOr(&unsupported_words[word], bit);
    }
    __syncthreads();
  }
}

__global__ void
pack_profile_quartet_scores_kernel(
  const uint8_t *scalar_scores,
  const plan7_ssv_f1_profile *profiles,
  const ProfilePackedQuartet *quartets,
  bool identity_padding,
  uint32_t *quartet_scores)
{
  const size_t quartet_index = static_cast<size_t>(blockIdx.y);
  const ProfilePackedQuartet quartet = quartets[quartet_index];
  const size_t word_count =
    static_cast<size_t>(quartet.maximum_model_length) * 29;
  for (size_t word = static_cast<size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       word < word_count;
       word += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t model_position = word / 29;
    const size_t residue = word % 29;
    uint32_t packed = 0;
#pragma unroll
    for (int lane = 0; lane < kProfilesPerPackedWord; ++lane) {
      const uint32_t profile_index = quartet.profile_indices[lane];
      const plan7_ssv_profile profile = profiles[profile_index].profile;
      uint8_t cost = identity_padding ? UINT8_C(0) : UINT8_C(0x80);
      if (model_position < static_cast<size_t>(profile.model_length))
        cost = scalar_scores[
          profile.score_offset + model_position * profile.score_stride +
          residue];
      packed |= static_cast<uint32_t>(cost) << (lane * 8);
    }
    quartet_scores[quartet.score_word_offset + word] = packed;
  }
}

template<bool IdentityPadding>
__device__ __forceinline__ void
ssv_profile_packed_filter_block(
  const uint32_t *scores,
  const ProfilePackedQuartet quartet,
  const plan7_ssv_f1_profile *profiles,
  const uint8_t *residues,
  const uint64_t *offsets,
  size_t sequence,
  const uint8_t *tjb,
  plan7_ssv_result *results,
  unsigned *maxima)
{
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  if (length == 0) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int lane = 0; lane < kProfilesPerPackedWord; ++lane) {
        const uint32_t profile_index = quartet.profile_indices[lane];
        results[lane] = {
          128, PLAN7_SSV_EMPTY,
          tjb[profiles[profile_index].tjb_offset + sequence], 0, 0};
      }
    }
    return;
  }

  uint32_t local_maximum = UINT32_C(0x80808080);
  const int diagonal_count = quartet.maximum_model_length + length - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    uint32_t value = UINT32_C(0x80808080);
    while (i < length && k < static_cast<int>(quartet.maximum_model_length)) {
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      const uint32_t cost = scores[
        quartet.score_word_offset + static_cast<size_t>(k) * 29 + residue];
      const uint32_t next = __vsubss4(value, cost);
      if constexpr (IdentityPadding) {
        value = next;
      } else {
        uint32_t active = 0;
#pragma unroll
        for (int lane = 0; lane < kProfilesPerPackedWord; ++lane) {
          const uint32_t profile_index = quartet.profile_indices[lane];
          if (k < profiles[profile_index].profile.model_length)
            active |= UINT32_C(0xff) << (lane * 8);
        }
        value = (next & active) | (value & ~active);
      }
      local_maximum = __vmaxu4(local_maximum, value);
      ++i;
      ++k;
    }
  }

  for (int width = 16; width > 0; width >>= 1)
    local_maximum = __vmaxu4(
      local_maximum,
      __shfl_down_sync(UINT32_MAX, local_maximum, width));
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) maxima[warp] = local_maximum;
  __syncthreads();
  if (warp == 0) {
    local_maximum = lane < blockDim.x / 32
      ? maxima[lane] : UINT32_C(0x80808080);
    for (int width = 16; width > 0; width >>= 1)
      local_maximum = __vmaxu4(
        local_maximum,
        __shfl_down_sync(UINT32_MAX, local_maximum, width));
    if (lane == 0) maxima[0] = local_maximum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int packed_lane = 0; packed_lane < kProfilesPerPackedWord;
         ++packed_lane) {
      const uint32_t profile_index = quartet.profile_indices[packed_lane];
      const plan7_ssv_profile profile = profiles[profile_index].profile;
      const uint8_t length_tjb = tjb[
        profiles[profile_index].tjb_offset + sequence];
      const unsigned raw_xE =
        (maxima[0] >> (packed_lane * 8)) & UINT32_C(0xff);
      results[packed_lane] = ssv_result_from_raw_xe(
        raw_xE, length_tjb, profile.tbm, profile.tec, profile.base,
        profile.bias, false);
    }
  }
}

template<bool IdentityPadding>
__global__ void
ssv_f1_mask_profile_packed_kernel(
  const uint32_t *quartet_scores,
  const ProfilePackedQuartet *quartets,
  const plan7_ssv_f1_profile *profiles,
  size_t sequence_count,
  const uint8_t *residues,
  const uint64_t *offsets,
  const float *null_scores,
  const uint8_t *tjb,
  size_t words_per_profile,
  uint32_t *candidate_words,
  uint8_t *raw_xe)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ ProfilePackedQuartet quartet;
  __shared__ plan7_ssv_result results[kProfilesPerPackedWord];
  if (threadIdx.x == 0) quartet = quartets[blockIdx.y];
  __syncthreads();

  const size_t first_sequence =
    static_cast<size_t>(blockIdx.x) * kSequencesPerBlock;
  for (int iteration = 0; iteration < kSequencesPerBlock; ++iteration) {
    const size_t sequence = first_sequence + static_cast<size_t>(iteration);
    if (sequence >= sequence_count) break;
    ssv_profile_packed_filter_block<IdentityPadding>(
      quartet_scores, quartet, profiles, residues, offsets, sequence, tjb,
      results, maxima);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int lane = 0; lane < kProfilesPerPackedWord; ++lane) {
        const uint32_t profile_index = quartet.profile_indices[lane];
        if (raw_xe != nullptr)
          raw_xe[static_cast<size_t>(profile_index) * sequence_count +
                 sequence] = results[lane].xE;
        if (f1_requires_cpu(results[lane], null_scores[sequence],
                            profiles[profile_index])) {
          const size_t word =
            static_cast<size_t>(profile_index) * words_per_profile +
            sequence / 32;
          atomicOr(&candidate_words[word],
                   UINT32_C(1) << (sequence % 32));
        }
      }
    }
    __syncthreads();
  }
}

__global__ void
ssv_f1_mask_indexed_kernel(const uint8_t *packed_scores,
                           const plan7_ssv_f1_profile *profiles,
                           const uint32_t *profile_indices,
                           size_t sequence_count,
                           const uint8_t *residues,
                           const uint64_t *offsets,
                           const float *null_scores,
                           const uint8_t *tjb,
                           size_t words_per_profile,
                           uint32_t *candidate_words,
                           uint8_t *raw_xe)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ plan7_ssv_f1_profile profile_descriptor;
  __shared__ uint32_t profile_index;
  if (threadIdx.x == 0) {
    profile_index = profile_indices[blockIdx.y];
    profile_descriptor = profiles[profile_index];
  }
  __syncthreads();

  const size_t first_sequence =
    static_cast<size_t>(blockIdx.x) * kSequencesPerBlock;
  for (int iteration = 0; iteration < kSequencesPerBlock; ++iteration) {
    const size_t sequence = first_sequence + static_cast<size_t>(iteration);
    if (sequence >= sequence_count) break;
    if (profile_descriptor.cutoff_mode == PLAN7_F1_CUTOFF_INVALID ||
        profile_descriptor.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_CPU) {
      if (threadIdx.x == 0) {
        const size_t word =
          static_cast<size_t>(profile_index) * words_per_profile +
          sequence / 32;
        atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
      }
      continue;
    }
    plan7_ssv_result result;
    ssv_filter_block<true>(
      packed_scores + profile_descriptor.profile.score_offset,
      profile_descriptor.profile.score_stride,
      profile_descriptor.profile.model_length,
      residues, offsets, sequence,
      tjb[profile_descriptor.tjb_offset + sequence],
      profile_descriptor.profile.tbm, profile_descriptor.profile.tec,
      profile_descriptor.profile.base, profile_descriptor.profile.bias,
      &result, maxima);
    if (threadIdx.x == 0) {
      if (raw_xe != nullptr)
        raw_xe[static_cast<size_t>(profile_index) * sequence_count +
               sequence] = result.xE;
      if (f1_requires_cpu(result, null_scores[sequence], profile_descriptor)) {
        const size_t word = static_cast<size_t>(profile_index) *
                            words_per_profile + sequence / 32;
        atomicOr(&candidate_words[word], UINT32_C(1) << (sequence % 32));
      }
    }
    __syncthreads();
  }
}

__global__ void
expand_length_class_tjb_kernel(const uint8_t *compact_tjb,
                               const uint32_t *sequence_length_classes,
                               size_t sequence_count,
                               size_t length_class_count,
                               size_t dense_count,
                               uint8_t *dense_tjb)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (index >= dense_count) return;
  const size_t sequence = index % sequence_count;
  const size_t row = index / sequence_count;
  const uint32_t length_class = sequence_length_classes[sequence];
  if (length_class < length_class_count)
    dense_tjb[index] = compact_tjb[row * length_class_count + length_class];
}

__global__ void
candidate_word_counts_kernel(const uint32_t *candidate_words,
                             size_t word_count,
                             uint64_t *word_counts)
{
  const size_t word = static_cast<size_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (word < word_count)
    word_counts[word] = static_cast<uint64_t>(__popc(candidate_words[word]));
}

__global__ void
candidate_profile_offsets_kernel(const uint64_t *word_counts,
                                 const uint64_t *word_offsets,
                                 size_t words_per_profile,
                                 size_t profile_count,
                                 size_t *profile_offsets)
{
  const size_t profile = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  if (profile < profile_count) {
    profile_offsets[profile] = static_cast<size_t>(
      word_offsets[profile * words_per_profile]);
  } else if (profile == profile_count) {
    const size_t word_count = profile_count * words_per_profile;
    profile_offsets[profile] = static_cast<size_t>(
      word_offsets[word_count - 1] + word_counts[word_count - 1]);
  }
}

__global__ void
candidate_scatter_kernel(const uint32_t *candidate_words,
                         const uint64_t *word_offsets,
                         size_t word_count,
                         size_t words_per_profile,
                         size_t sequence_count,
                         const uint8_t *raw_xe,
                         const plan7_ssv_f1_profile *profiles,
                         const uint8_t *tjb,
                         const uint64_t *sequence_offsets,
                         plan7_bias_candidate *candidates,
                         plan7_bias_ssv_input *ssv_inputs)
{
  const size_t word_index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  if (word_index >= word_count) return;

  uint32_t word = candidate_words[word_index];
  size_t output = static_cast<size_t>(word_offsets[word_index]);
  const uint32_t profile = static_cast<uint32_t>(
    word_index / words_per_profile);
  const uint32_t sequence_base = static_cast<uint32_t>(
    (word_index % words_per_profile) * 32);
  while (word != 0) {
    const unsigned bit = static_cast<unsigned>(__ffs(word) - 1);
    const uint32_t sequence = sequence_base + bit;
    candidates[output] = {profile, sequence};
    if (ssv_inputs != nullptr) {
      const plan7_ssv_f1_profile descriptor = profiles[profile];
      const uint8_t length_tjb =
        tjb[descriptor.tjb_offset + sequence];
      const bool empty = sequence_offsets[sequence] ==
                         sequence_offsets[sequence + 1];
      const plan7_ssv_result result = ssv_result_from_raw_xe(
        raw_xe[static_cast<size_t>(profile) * sequence_count + sequence],
        length_tjb, descriptor.profile.tbm, descriptor.profile.tec,
        descriptor.profile.base, descriptor.profile.bias, empty);
      ssv_inputs[output] = {result.numerator, result.status, 0};
    }
    ++output;
    word &= word - 1;
  }
}

__global__ void
ssv_bias_candidates_kernel(const uint8_t *packed_scores,
                           const plan7_ssv_f1_profile *profiles,
                           size_t sequence_count,
                           const uint8_t *residues,
                           const uint64_t *offsets,
                           const uint8_t *tjb,
                           const plan7_bias_candidate *candidates,
                           plan7_bias_ssv_input *ssv_inputs)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ plan7_ssv_f1_profile profile_descriptor;
  __shared__ plan7_bias_candidate candidate_descriptor;
  const size_t candidate_index = static_cast<size_t>(blockIdx.x);

  if (threadIdx.x == 0) {
    candidate_descriptor = candidates[candidate_index];
    profile_descriptor = profiles[candidate_descriptor.profile_index];
  }
  __syncthreads();

  plan7_ssv_result result;
  ssv_filter_block<true>(
    packed_scores + profile_descriptor.profile.score_offset,
    profile_descriptor.profile.score_stride,
    profile_descriptor.profile.model_length,
    residues,
    offsets,
    candidate_descriptor.sequence_index,
    tjb[profile_descriptor.tjb_offset + candidate_descriptor.sequence_index],
    profile_descriptor.profile.tbm,
    profile_descriptor.profile.tec,
    profile_descriptor.profile.base,
    profile_descriptor.profile.bias,
    &result,
    maxima);
  if (threadIdx.x == 0) {
    ssv_inputs[candidate_index] = {
      result.numerator, result.status, 0
    };
  }
}

void
set_error(char *error, size_t error_size, const char *message)
{
  if (error != nullptr && error_size != 0)
    snprintf(error, error_size, "%s", message);
}

void
set_cuda_error(char *error, size_t error_size, const char *operation,
               cudaError_t status)
{
  if (error != nullptr && error_size != 0)
    snprintf(error, error_size, "%s: %s", operation, cudaGetErrorString(status));
}

void
fill_cpu_required_results(plan7_ssv_result *results, size_t result_count)
{
  for (size_t index = 0; index < result_count; ++index)
    results[index] = {0, PLAN7_SSV_ENORESULT, 0, 0, 0};
}

bool
checked_product(size_t left, size_t right, size_t *product)
{
  if (right != 0 && left > SIZE_MAX / right) return false;
  *product = left * right;
  return true;
}

bool
checked_add_u64(uint64_t left, uint64_t right, uint64_t *sum)
{
  if (right > UINT64_MAX - left) return false;
  *sum = left + right;
  return true;
}

bool
checked_product_u64(uint64_t left, uint64_t right, uint64_t *product)
{
  if (right != 0 && left > UINT64_MAX / right) return false;
  *product = left * right;
  return true;
}

bool
compute_null_score(uint64_t length, float *ret_null_score)
{
  float length_f;
  float p1;
  float null_score;

  if (plan7_bias_host_environment_attested() != 1 ||
      length == 0 || length > kMaximumTargetLength ||
      ret_null_score == nullptr)
    return false;
  length_f = static_cast<float>(length);
  p1 = length_f / static_cast<float>(length + 1);
  null_score = static_cast<float>(
    static_cast<double>(length_f) * log(static_cast<double>(p1)) +
    log(1.0 - static_cast<double>(p1)));
  if (!isfinite(null_score)) return false;
  *ret_null_score = null_score;
  return true;
}

bool
compute_bit_score(int16_t numerator,
                  float null_score,
                  float scale,
                  float *ret_bit_score)
{
  float score;
  float delta;
  float bit_score;

  if (plan7_bias_host_environment_attested() != 1 ||
      !isfinite(null_score) || !isfinite(scale) || scale <= 0.0f ||
      ret_bit_score == nullptr)
    return false;
  score = static_cast<float>(numerator);
  score /= scale;
  score -= 3.0;
  delta = score - null_score;
  bit_score = static_cast<float>(
    static_cast<double>(delta) / eslCONST_LOG2);
  if (!isfinite(score) || !isfinite(bit_score)) return false;
  *ret_bit_score = bit_score;
  return true;
}

float
ordered_to_float(uint32_t ordered)
{
  const uint32_t bits = (ordered & UINT32_C(0x80000000)) != 0
                          ? ordered ^ UINT32_C(0x80000000)
                          : ~ordered;
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

bool
valid_f1_parameters(float m_mu, float m_lambda, double f1)
{
  return isfinite(m_mu) && isfinite(m_lambda) && m_lambda > 0.0f &&
         m_mu != kEvparamUnset && m_lambda != kEvparamUnset && isfinite(f1) &&
         f1 >= 0.0 && f1 <= 1.0;
}

bool
f1_probability(float bit_score,
               float m_mu,
               float m_lambda,
               double *ret_probability)
{
  const double probability = esl_gumbel_surv(
    static_cast<double>(bit_score), static_cast<double>(m_mu),
    static_cast<double>(m_lambda));
  if (!isfinite(probability) || probability < 0.0 || probability > 1.0)
    return false;
  *ret_probability = probability;
  return true;
}

bool
uses_smallx_survivor_branch(float bit_score,
                            float m_mu,
                            float m_lambda,
                            bool *ret_uses_smallx)
{
  const double y = static_cast<double>(m_lambda) *
                   (static_cast<double>(bit_score) -
                    static_cast<double>(m_mu));
  const double tail_term = exp(-y);
  if (isnan(tail_term)) return false;
  *ret_uses_smallx = tail_term < eslSMALLX1;
  return true;
}

bool
f1_branch_is_nonmonotone(float m_mu,
                         float m_lambda,
                         double f1,
                         bool *ret_nonmonotone)
{
  constexpr uint32_t kLowestFinite = UINT32_C(0x00800000);
  constexpr uint32_t kHighestFinite = UINT32_C(0xff7fffff);
  const double lowest_jump_probability = 1.0 - exp(-eslSMALLX1);
  bool low_uses_smallx;
  bool high_uses_smallx;
  uint32_t low = kLowestFinite;
  uint32_t high = kHighestFinite;

  *ret_nonmonotone = false;
  if (f1 < lowest_jump_probability || f1 >= eslSMALLX1) return true;
  if (!uses_smallx_survivor_branch(-FLT_MAX, m_mu, m_lambda,
                                   &low_uses_smallx) ||
      !uses_smallx_survivor_branch(FLT_MAX, m_mu, m_lambda,
                                   &high_uses_smallx))
    return false;
  if (low_uses_smallx == high_uses_smallx) return true;
  if (low_uses_smallx || !high_uses_smallx) return false;

  while (high - low > 1) {
    const uint32_t middle = low + (high - low) / 2;
    bool uses_smallx;
    if (!uses_smallx_survivor_branch(
          ordered_to_float(middle), m_mu, m_lambda, &uses_smallx))
      return false;
    if (uses_smallx)
      high = middle;
    else
      low = middle;
  }

  double before_probability;
  double after_probability;
  if (!f1_probability(ordered_to_float(low), m_mu, m_lambda,
                      &before_probability) ||
      !f1_probability(ordered_to_float(high), m_mu, m_lambda,
                      &after_probability))
    return false;
  *ret_nonmonotone =
    before_probability <= f1 && after_probability > f1;
  return true;
}

int
derive_f1_cutoff(float m_mu,
                 float m_lambda,
                 double f1,
                 float *ret_bit_score)
{
  constexpr uint32_t kLowestFinite = UINT32_C(0x00800000);
  constexpr uint32_t kHighestFinite = UINT32_C(0xff7fffff);
  double low_probability;
  double high_probability;
  bool branch_is_nonmonotone;
  uint32_t low = kLowestFinite;
  uint32_t high = kHighestFinite;

  if (ret_bit_score != nullptr) *ret_bit_score = NAN;
  if (plan7_bias_host_environment_attested() != 1 ||
      !valid_f1_parameters(m_mu, m_lambda, f1) ||
      !f1_probability(-FLT_MAX, m_mu, m_lambda, &low_probability) ||
      !f1_probability(FLT_MAX, m_mu, m_lambda, &high_probability) ||
      !f1_branch_is_nonmonotone(m_mu, m_lambda, f1,
                                &branch_is_nonmonotone) ||
      branch_is_nonmonotone ||
      low_probability < high_probability)
    return PLAN7_F1_CUTOFF_INVALID;

  /* The survivor is nonincreasing for lambda > 0. */
  if (low_probability <= f1) return PLAN7_F1_CUTOFF_ALWAYS_CPU;
  if (high_probability > f1) return PLAN7_F1_CUTOFF_ALWAYS_REJECT;

  while (high - low > 1) {
    const uint32_t middle = low + (high - low) / 2;
    double probability;
    if (!f1_probability(ordered_to_float(middle), m_mu, m_lambda,
                        &probability))
      return PLAN7_F1_CUTOFF_INVALID;
    if (probability > low_probability || probability < high_probability)
      return PLAN7_F1_CUTOFF_INVALID;
    if (probability > f1) {
      low = middle;
      low_probability = probability;
    } else {
      high = middle;
      high_probability = probability;
    }
  }

  const float cutoff = ordered_to_float(high);
  const float predecessor = ordered_to_float(high - 1);
  double cutoff_probability;
  double predecessor_probability;
  if (!isfinite(cutoff) ||
      !f1_probability(cutoff, m_mu, m_lambda, &cutoff_probability) ||
      !f1_probability(predecessor, m_mu, m_lambda,
                      &predecessor_probability) ||
      predecessor_probability < cutoff_probability ||
      predecessor_probability <= f1 || cutoff_probability > f1)
    return PLAN7_F1_CUTOFF_INVALID;
  if (ret_bit_score != nullptr) *ret_bit_score = cutoff;
  return PLAN7_F1_CUTOFF_SCORE;
}

int
cutoff_decision_with_null_score(uint8_t status,
                                int16_t numerator,
                                float null_score,
                                float scale,
                                int cutoff_mode,
                                float cutoff_bit_score)
{
  float bit_score;
  if (status != PLAN7_SSV_OK ||
      !compute_bit_score(numerator, null_score, scale, &bit_score))
    return PLAN7_F1_CPU_REQUIRED;

  switch (cutoff_mode) {
    case PLAN7_F1_CUTOFF_SCORE:
      if (!isfinite(cutoff_bit_score)) return PLAN7_F1_CPU_REQUIRED;
      return bit_score < cutoff_bit_score ? PLAN7_F1_DEFINITE_REJECT
                                          : PLAN7_F1_CPU_REQUIRED;
    case PLAN7_F1_CUTOFF_ALWAYS_REJECT:
      return PLAN7_F1_DEFINITE_REJECT;
    case PLAN7_F1_CUTOFF_ALWAYS_CPU:
    default:
      return PLAN7_F1_CPU_REQUIRED;
  }
}

template<typename T>
int
grow_device_buffer(T **buffer,
                   size_t *capacity,
                   size_t required_bytes,
                   const char *malloc_name,
                   const char *free_name,
                   char *error,
                   size_t error_size)
{
  T *replacement = nullptr;
  cudaError_t status;

  if (required_bytes <= *capacity) return 0;
  status = cudaMalloc(&replacement, required_bytes);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, malloc_name, status);
    return -1;
  }
  status = cudaFree(*buffer);
  if (status != cudaSuccess) {
    cudaFree(replacement);
    set_cuda_error(error, error_size, free_name, status);
    return -1;
  }
  *buffer = replacement;
  *capacity = required_bytes;
  return 0;
}

void
invalidate_f1_device_candidates(plan7_ssv_sequence_batch *batch)
{
  batch->f1_device_candidates_valid = 0;
  batch->f1_compact_ssv_inputs_valid = 0;
  batch->cached_f1_candidate_count = 0;
}

void
invalidate_f1_cache(plan7_ssv_sequence_batch *batch)
{
  batch->f1_cache_valid = 0;
  batch->f1_raw_xe_valid = 0;
  invalidate_f1_device_candidates(batch);
}

bool
validate_compact_profile(const plan7_ssv_profile *profile,
                         int alphabet_size,
                         size_t packed_score_count,
                         size_t expected_offset,
                         char *error,
                         size_t error_size)
{
  size_t expected_score_count;

  if (profile->model_length < 1 || profile->model_length > 100000 ||
      profile->score_stride < 1 || !isfinite(profile->scale) ||
      profile->scale <= 0.0f) {
    set_error(error, error_size, "invalid packed profile dimensions or scale");
    return false;
  }
  if (profile->score_stride != alphabet_size ||
      !checked_product(static_cast<size_t>(profile->model_length),
                       static_cast<size_t>(alphabet_size),
                       &expected_score_count) ||
      profile->score_count != expected_score_count ||
      profile->score_offset != expected_offset ||
      expected_offset > packed_score_count ||
      profile->score_count >
        static_cast<uint64_t>(packed_score_count - expected_offset)) {
    set_error(error, error_size, "invalid compact profile score range");
    return false;
  }
  return true;
}

bool
validate_bias_profile(const plan7_bias_profile *profile)
{
  if (profile == nullptr || profile->reserved != 0 ||
      !isfinite(profile->t10) || profile->t10 < 0.0f ||
      !isfinite(profile->t11) || profile->t11 < 0.0f ||
      !isfinite(profile->scale) || profile->scale <= 0.0f ||
      profile->cutoff_mode < PLAN7_BIAS_CUTOFF_INVALID ||
      profile->cutoff_mode > PLAN7_BIAS_CUTOFF_ALWAYS_PASS ||
      (profile->cutoff_mode == PLAN7_BIAS_CUTOFF_SCORE &&
       !isfinite(profile->cutoff_bit_score)) ||
      !isfinite(profile->pi0) || profile->pi0 < 0.0f ||
      !isfinite(profile->pi1) || profile->pi1 < 0.0f ||
      !isfinite(profile->t02) || profile->t02 < 0.0f ||
      !isfinite(profile->t12) || profile->t12 < 0.0f)
    return false;
  for (int residue = 0; residue < 29; ++residue)
    for (int state = 0; state < 2; ++state)
      if (!isfinite(profile->eo[residue][state]) ||
          profile->eo[residue][state] < 0.0f)
        return false;
  return true;
}

uint8_t
compute_tjb_from_log_term(float scale, float log_term)
{
  const float cost = -1.0f * roundf(scale * log_term);
  return cost > 255.0f ? 255 : static_cast<uint8_t>(cost);
}

float
compute_tjb_log_term(uint64_t length)
{
  return logf(3.0f / static_cast<float>(length + 3));
}

uint8_t
compute_tjb(float scale, uint64_t length)
{
  return compute_tjb_from_log_term(scale, compute_tjb_log_term(length));
}

int
destroy_sequence_batch(plan7_ssv_sequence_batch *batch,
                       char *error,
                       size_t error_size)
{
  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool device_ready = true;
  bool restore_device = false;
  int workspace_status = 0;

  if (batch == nullptr) return 0;
  if (plan7_forward_workspace_destroy(
        &batch->forward_workspace, error, error_size) != 0)
    workspace_status = -1;
  if (plan7_postfilter_workspace_destroy(
        &batch->postfilter_workspace,
        workspace_status == 0 ? error : nullptr,
        workspace_status == 0 ? error_size : 0) != 0)
    workspace_status = -1;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess && original_device != batch->device_ordinal) {
    status = cudaSetDevice(batch->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice(batch->device_ordinal);
    if (status != cudaSuccess) {
      first_error = status;
      device_ready = false;
    }
  }

#define CUDA_FREE(pointer)                                                    \
  do {                                                                        \
    if (device_ready) {                                                       \
      status = cudaFree(pointer);                                             \
      if (first_error == cudaSuccess && status != cudaSuccess)                \
        first_error = status;                                                 \
    }                                                                         \
  } while (0)

  CUDA_FREE(batch->device_scores);
  CUDA_FREE(batch->device_profiles);
  CUDA_FREE(batch->device_f1_profiles);
  CUDA_FREE(batch->device_f1_profile_packed_scores);
  CUDA_FREE(batch->device_f1_profile_packed_quartets);
  CUDA_FREE(batch->device_f1_scalar_profile_indices);
  CUDA_FREE(batch->device_f1_raw_xe);
  CUDA_FREE(batch->device_candidate_words);
  CUDA_FREE(batch->device_candidate_word_counts);
  CUDA_FREE(batch->device_candidate_word_offsets);
  CUDA_FREE(batch->device_candidate_profile_offsets);
  CUDA_FREE(batch->device_candidate_scan_workspace);
  CUDA_FREE(batch->device_bias_profiles);
  CUDA_FREE(batch->device_bias_candidates);
  CUDA_FREE(batch->device_bias_ssv_inputs);
  CUDA_FREE(batch->device_bias_results);
  CUDA_FREE(batch->device_bias_log1mp);
  CUDA_FREE(batch->device_bias_logp);
  CUDA_FREE(batch->device_f1_compact_tjb);
  CUDA_FREE(batch->device_length_class_indices);
  CUDA_FREE(batch->device_results);
  CUDA_FREE(batch->device_tjb);
  CUDA_FREE(batch->device_null_scores);
  CUDA_FREE(batch->device_offsets);
  CUDA_FREE(batch->device_residues);
#undef CUDA_FREE
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (first_error == cudaSuccess && status != cudaSuccess)
      first_error = status;
  }
  free(batch->host_tjb);
  free(batch->host_f1_profiles);
  free(batch->host_f1_scores);
  free(batch->host_bias_candidates);
  free(batch->host_candidate_offsets);
  free(batch->host_length_class_log_terms);
  free(batch->host_length_class_indices);
  free(batch->host_tjb_log_terms);
  free(batch->host_bias_log1mp);
  free(batch->host_bias_logp);
  free(batch->host_null_scores);
  free(batch->host_lengths);
  free(batch);
  if (first_error != cudaSuccess) {
    if (workspace_status == 0)
      set_cuda_error(error, error_size, "destroy CUDA sequence batch",
                     first_error);
    return -1;
  }
  return workspace_status;
}

}  // namespace

extern "C" int
plan7_cuda_device_count(char *error, size_t error_size)
{
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDeviceCount", status);
    return -1;
  }
  return count;
}

extern "C" int
plan7_cuda_memory_info(int *device_ordinal,
                       uint64_t *free_bytes,
                       uint64_t *total_bytes,
                       char *error,
                       size_t error_size)
{
  int current_device = -1;
  size_t free_value = 0;
  size_t total_value = 0;
  cudaError_t status;
  if (device_ordinal == nullptr || free_bytes == nullptr ||
      total_bytes == nullptr) {
    set_error(error, error_size, "CUDA memory information output is null");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  status = cudaMemGetInfo(&free_value, &total_value);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaMemGetInfo", status);
    return -1;
  }
  *device_ordinal = current_device;
  *free_bytes = static_cast<uint64_t>(free_value);
  *total_bytes = static_cast<uint64_t>(total_value);
  return 0;
}

extern "C" int
plan7_validate_device_ordinal(int owner_device,
                              int current_device,
                              char *error,
                              size_t error_size)
{
  if (owner_device < 0 || current_device < 0) {
    set_error(error, error_size, "CUDA device ordinal is unavailable");
    return -1;
  }
  if (owner_device != current_device) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  return 0;
}

extern "C" int
plan7_profile_footprint_compute(const uint32_t *model_lengths,
                                size_t profile_count,
                                plan7_profile_footprint *footprint,
                                char *error,
                                size_t error_size)
{
  plan7_profile_footprint result{};
  if (footprint == nullptr ||
      (profile_count != 0 && model_lengths == nullptr)) {
    set_error(error, error_size, "invalid profile footprint arguments");
    return -1;
  }
  result.profile_count = static_cast<uint64_t>(profile_count);
  for (size_t i = 0; i < profile_count; ++i) {
    const uint64_t model_length = model_lengths[i];
    if (model_length < 1 || model_length > kMaximumModelLength) {
      set_error(error, error_size, "model length must be in [1, 100000]");
      return -1;
    }
    const uint64_t viterbi_q = (model_length + 31) / 32;
    const uint64_t forward_raw_q = (model_length + 3) / 4;
    const uint64_t forward_q = forward_raw_q < 2 ? 2 : forward_raw_q;
    uint64_t ssv_bytes;
    uint64_t viterbi_rows;
    uint64_t viterbi_bytes;
    uint64_t forward_rows;
    uint64_t forward_bytes;
    if (!checked_product_u64(kAminoAlphabetSize, model_length, &ssv_bytes) ||
        !checked_product_u64(kViterbiBytesPerQ, viterbi_q,
                             &viterbi_rows) ||
        !checked_add_u64(kViterbiDescriptorBytes, viterbi_rows,
                         &viterbi_bytes) ||
        !checked_product_u64(kForwardBytesPerQ, forward_q,
                             &forward_rows) ||
        !checked_add_u64(kForwardDescriptorBytes, forward_rows,
                         &forward_bytes) ||
        !checked_add_u64(result.ssv_device_bytes, ssv_bytes,
                         &result.ssv_device_bytes) ||
        !checked_add_u64(result.viterbi_device_bytes, viterbi_bytes,
                         &result.viterbi_device_bytes) ||
        !checked_add_u64(result.viterbi_exact_rbv_upper_bytes, ssv_bytes,
                         &result.viterbi_exact_rbv_upper_bytes) ||
        !checked_add_u64(result.forward_device_bytes, forward_bytes,
                         &result.forward_device_bytes) ||
        !checked_add_u64(result.bias_device_bytes, kBiasDescriptorBytes,
                         &result.bias_device_bytes)) {
      set_error(error, error_size, "profile footprint size overflow");
      return -1;
    }
  }
  if (!checked_add_u64(result.ssv_device_bytes,
                       result.viterbi_device_bytes,
                       &result.minimum_device_bytes) ||
      !checked_add_u64(result.minimum_device_bytes,
                       result.forward_device_bytes,
                       &result.minimum_device_bytes) ||
      !checked_add_u64(result.minimum_device_bytes,
                       result.bias_device_bytes,
                       &result.minimum_device_bytes) ||
      !checked_add_u64(result.minimum_device_bytes,
                       result.viterbi_exact_rbv_upper_bytes,
                       &result.maximum_device_bytes)) {
    set_error(error, error_size, "profile footprint total overflow");
    return -1;
  }
  *footprint = result;
  return 0;
}

extern "C" int
plan7_profile_slice_cell_count(uint64_t profile_count,
                               uint64_t target_count,
                               uint64_t cell_limit,
                               uint64_t *cell_count,
                               char *error,
                               size_t error_size)
{
  uint64_t result;
  if (cell_count == nullptr) {
    set_error(error, error_size, "profile slice cell output is null");
    return -1;
  }
  if (!checked_product_u64(profile_count, target_count, &result)) {
    set_error(error, error_size, "profile slice cell count overflow");
    return -1;
  }
  *cell_count = result;
  if (result > cell_limit) {
    set_error(error, error_size, "profile slice exceeds cell limit");
    return -1;
  }
  return 0;
}

extern "C" int
plan7_simulate_allocate_before_free(
  const uint64_t *current_capacities,
  const uint64_t *required_capacities,
  size_t capacity_count,
  uint64_t free_bytes,
  uint64_t *final_capacities,
  plan7_allocation_simulation *simulation,
  char *error,
  size_t error_size)
{
  plan7_allocation_simulation result{};
  uint64_t net_growth = 0;
  if (simulation == nullptr ||
      (capacity_count != 0 &&
       (current_capacities == nullptr || required_capacities == nullptr ||
        final_capacities == nullptr))) {
    set_error(error, error_size, "invalid allocation simulation arguments");
    return -1;
  }
  result.fits = 1;
  result.first_unfit_index = UINT64_MAX;
  for (size_t i = 0; i < capacity_count; ++i) {
    const uint64_t current = current_capacities[i];
    const uint64_t required = required_capacities[i];
    final_capacities[i] = current < required ? required : current;
    if (required <= current) continue;

    uint64_t temporary_bytes;
    uint64_t next_growth;
    if (!checked_add_u64(net_growth, required, &temporary_bytes) ||
        !checked_add_u64(net_growth, required - current, &next_growth)) {
      set_error(error, error_size, "allocation simulation size overflow");
      return -1;
    }
    if (temporary_bytes > result.peak_additional_bytes)
      result.peak_additional_bytes = temporary_bytes;
    if (temporary_bytes > free_bytes && result.fits) {
      result.fits = 0;
      result.first_unfit_index = static_cast<uint64_t>(i);
    }
    net_growth = next_growth;
    ++result.growth_count;
  }
  result.final_additional_bytes = net_growth;
  result.final_free_bytes = result.fits ? free_bytes - net_growth : 0;
  *simulation = result;
  return 0;
}

extern "C" int
plan7_tjb_for_length(float scale, uint64_t length)
{
  if (!isfinite(scale) || scale <= 0.0f || length > kMaximumTargetLength)
    return -1;
  return compute_tjb(scale, length);
}

extern "C" int
plan7_ssv_f1_decision(uint8_t status,
                      int16_t numerator,
                      uint64_t length,
                      float scale,
                      float m_mu,
                      float m_lambda,
                      double f1,
                      double *ret_p)
{
  float null_score;
  float bit_score;
  double probability;

  if (ret_p != nullptr) *ret_p = NAN;
  if (status != PLAN7_SSV_OK || length == 0 ||
      length > kMaximumTargetLength || !isfinite(scale) || scale <= 0.0f ||
      !valid_f1_parameters(m_mu, m_lambda, f1))
    return PLAN7_F1_CPU_REQUIRED;

  /* Preserve HMMER 3.4's float/double evaluation order exactly. */
  if (!compute_null_score(length, &null_score) ||
      !compute_bit_score(numerator, null_score, scale, &bit_score) ||
      !f1_probability(bit_score, m_mu, m_lambda, &probability))
    return PLAN7_F1_CPU_REQUIRED;
  if (ret_p != nullptr) *ret_p = probability;
  return probability > f1 ? PLAN7_F1_DEFINITE_REJECT
                          : PLAN7_F1_CPU_REQUIRED;
}

extern "C" int
plan7_ssv_f1_cutoff(float m_mu,
                    float m_lambda,
                    double f1,
                    float *ret_bit_score)
{
  return derive_f1_cutoff(m_mu, m_lambda, f1, ret_bit_score);
}

extern "C" int
plan7_ssv_f1_cutoff_decision(uint8_t status,
                             int16_t numerator,
                             uint64_t length,
                             float scale,
                             int cutoff_mode,
                             float cutoff_bit_score)
{
  float null_score;
  if (!compute_null_score(length, &null_score))
    return PLAN7_F1_CPU_REQUIRED;
  return cutoff_decision_with_null_score(status, numerator, null_score, scale,
                                         cutoff_mode, cutoff_bit_score);
}

extern "C" int
plan7_ssv_sequence_batch_create(const uint8_t *residues,
                                size_t residue_count,
                                const uint64_t *offsets,
                                size_t offset_count,
                                int alphabet_size,
                                plan7_ssv_sequence_batch **batch_out,
                                char *error,
                                size_t error_size)
{
  plan7_ssv_sequence_batch *batch = nullptr;
  cudaError_t status;
  size_t sequence_count;
  size_t offset_bytes;
  size_t result_bytes;
  size_t null_score_bytes;
  size_t length_class_index_bytes;
  uint32_t *length_to_class = nullptr;
  int rc = -1;
  int device_ordinal;

  if (batch_out == nullptr) {
    set_error(error, error_size, "sequence batch output is null");
    return -1;
  }
  *batch_out = nullptr;
  status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (alphabet_size < 1 || offset_count == 0 || offsets == nullptr ||
      (residue_count != 0 && residues == nullptr)) {
    set_error(error, error_size, "invalid sequence batch buffers");
    return -1;
  }
  sequence_count = offset_count - 1;
  if (sequence_count > INT_MAX) {
    set_error(error, error_size, "too many sequences in CUDA batch");
    return -1;
  }
  if (offsets[0] != 0 || offsets[sequence_count] != residue_count) {
    set_error(error, error_size, "invalid sequence offsets");
    return -1;
  }
  for (size_t i = 0; i < sequence_count; ++i) {
    if (offsets[i] > offsets[i + 1] ||
        offsets[i + 1] - offsets[i] > kMaximumTargetLength) {
      set_error(error, error_size, "invalid target length or offsets");
      return -1;
    }
  }
  for (size_t i = 0; i < residue_count; ++i) {
    if (residues[i] >= alphabet_size) {
      set_error(error, error_size, "digital residue is outside the alphabet");
      return -1;
    }
  }
  if (!checked_product(offset_count, sizeof(*offsets), &offset_bytes) ||
      !checked_product(sequence_count, sizeof(plan7_ssv_result),
                       &result_bytes) ||
      !checked_product(sequence_count, sizeof(float), &null_score_bytes) ||
      !checked_product(sequence_count, sizeof(uint32_t),
                       &length_class_index_bytes) ||
      (residue_count == 0 ? 1 : residue_count) > SIZE_MAX - offset_bytes) {
    set_error(error, error_size, "sequence batch size overflow");
    return -1;
  }

  batch = static_cast<plan7_ssv_sequence_batch *>(calloc(1, sizeof(*batch)));
  if (batch == nullptr) {
    set_error(error, error_size, "host sequence batch allocation failed");
    return -1;
  }
  batch->alphabet_size = alphabet_size;
  batch->device_ordinal = device_ordinal;
  batch->generation_id = allocate_sequence_batch_generation();
  batch->sequence_count = sequence_count;
  batch->host_float_environment_valid =
    plan7_bias_host_environment_attested();
  if (sequence_count == 0) {
    *batch_out = batch;
    return 0;
  }
  batch->host_lengths =
    static_cast<uint64_t *>(malloc(sequence_count * sizeof(uint64_t)));
  batch->host_null_scores =
    static_cast<float *>(malloc(sequence_count * sizeof(float)));
  batch->host_bias_logp =
    static_cast<float *>(malloc(sequence_count * sizeof(float)));
  batch->host_bias_log1mp =
    static_cast<float *>(malloc(sequence_count * sizeof(float)));
  batch->host_tjb_log_terms =
    static_cast<float *>(malloc(sequence_count * sizeof(float)));
  batch->host_length_class_indices =
    static_cast<uint32_t *>(malloc(length_class_index_bytes));
  batch->host_length_class_log_terms =
    static_cast<float *>(malloc(null_score_bytes));
  batch->host_tjb = static_cast<uint8_t *>(malloc(sequence_count));
  length_to_class = static_cast<uint32_t *>(
    malloc((static_cast<size_t>(kMaximumTargetLength) + 1) *
           sizeof(uint32_t)));
  if (batch->host_lengths == nullptr || batch->host_null_scores == nullptr ||
      batch->host_bias_logp == nullptr ||
      batch->host_bias_log1mp == nullptr ||
      batch->host_tjb_log_terms == nullptr ||
      batch->host_length_class_indices == nullptr ||
      batch->host_length_class_log_terms == nullptr ||
      batch->host_tjb == nullptr || length_to_class == nullptr) {
    set_error(error, error_size, "host sequence metadata allocation failed");
    goto cleanup;
  }
  std::fill_n(length_to_class,
              static_cast<size_t>(kMaximumTargetLength) + 1,
              UINT32_MAX);
  batch->host_tjb_capacity = sequence_count;
  for (size_t i = 0; i < sequence_count; ++i) {
    batch->host_lengths[i] = offsets[i + 1] - offsets[i];
    if (!compute_null_score(batch->host_lengths[i],
                            &batch->host_null_scores[i]))
      batch->host_null_scores[i] = NAN;
    if (plan7_bias_length_terms(batch->host_lengths[i],
                                &batch->host_bias_logp[i],
                                &batch->host_bias_log1mp[i]) != 0) {
      batch->host_bias_logp[i] = NAN;
      batch->host_bias_log1mp[i] = NAN;
    }
    batch->host_tjb_log_terms[i] =
      compute_tjb_log_term(batch->host_lengths[i]);
    uint32_t length_class = length_to_class[batch->host_lengths[i]];
    if (length_class == UINT32_MAX) {
      length_class = static_cast<uint32_t>(batch->length_class_count++);
      length_to_class[batch->host_lengths[i]] = length_class;
      batch->host_length_class_log_terms[length_class] =
        batch->host_tjb_log_terms[i];
    }
    batch->host_length_class_indices[i] = length_class;
  }
  free(length_to_class);
  length_to_class = nullptr;

#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      goto cleanup;                                                           \
    }                                                                         \
  } while (0)

  CUDA_TRY(cudaMalloc(&batch->device_residues,
                      residue_count == 0 ? 1 : residue_count));
  batch->device_residue_capacity = residue_count == 0 ? 1 : residue_count;
  CUDA_TRY(cudaMalloc(&batch->device_offsets, offset_bytes));
  batch->device_offset_capacity = offset_bytes;
  CUDA_TRY(cudaMalloc(&batch->device_null_scores, null_score_bytes));
  batch->device_null_score_capacity = null_score_bytes;
  CUDA_TRY(cudaMalloc(&batch->device_length_class_indices,
                      length_class_index_bytes));
  batch->device_length_class_index_capacity = length_class_index_bytes;
  CUDA_TRY(cudaMalloc(&batch->device_tjb, sequence_count));
  batch->device_tjb_capacity = sequence_count;
  CUDA_TRY(cudaMalloc(&batch->device_results, result_bytes));
  batch->device_result_capacity = result_bytes;
  if (residue_count != 0)
    CUDA_TRY(cudaMemcpy(batch->device_residues, residues, residue_count,
                        cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(batch->device_offsets, offsets, offset_bytes,
                      cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(batch->device_null_scores, batch->host_null_scores,
                      null_score_bytes, cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(batch->device_length_class_indices,
                      batch->host_length_class_indices,
                      length_class_index_bytes,
                      cudaMemcpyHostToDevice));
  batch->input_device_bytes = static_cast<uint64_t>(
      (residue_count == 0 ? 1 : residue_count) + offset_bytes);
  *batch_out = batch;
  rc = 0;

cleanup:
  free(length_to_class);
  if (rc != 0) destroy_sequence_batch(batch, nullptr, 0);
#undef CUDA_TRY
  return rc;
}

extern "C" int
plan7_ssv_sequence_batch_destroy(plan7_ssv_sequence_batch **batch_out,
                                 char *error,
                                 size_t error_size)
{
  plan7_ssv_sequence_batch *batch;

  if (batch_out == nullptr) {
    set_error(error, error_size, "sequence batch pointer is null");
    return -1;
  }
  batch = *batch_out;
  *batch_out = nullptr;
  return destroy_sequence_batch(batch, error, error_size);
}

extern "C" int
plan7_ssv_sequence_batch_set_execution_policy(
  plan7_ssv_sequence_batch *batch,
  int policy,
  char *error,
  size_t error_size)
{
  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  if (policy < PLAN7_GPU_EXECUTION_POLICY_AUTO ||
      policy > PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT) {
    set_error(error, error_size, "invalid GPU execution policy");
    return -1;
  }
  if (batch->execution_policy_locked) {
    set_error(error, error_size, "GPU execution policy is already in use");
    return -1;
  }
  batch->execution_policy_mode = policy;
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_get_execution_policy_statistics(
  const plan7_ssv_sequence_batch *batch,
  plan7_gpu_execution_policy_statistics *statistics,
  char *error,
  size_t error_size)
{
  if (batch == nullptr || statistics == nullptr) {
    set_error(error, error_size, "invalid GPU execution policy statistics");
    return -1;
  }
  memset(statistics, 0, sizeof(*statistics));
  statistics->version = PLAN7_GPU_EXECUTION_POLICY_VERSION;
  statistics->mode = static_cast<uint32_t>(batch->execution_policy_mode);
  statistics->target_count = batch->sequence_count;
  statistics->length_class_count = batch->length_class_count;
  statistics->f1_run_count = batch->execution_policy_f1_run_count;
  statistics->forward_candidates_per_warp =
    PLAN7_GPU_EXECUTION_POLICY_FORWARD_CANDIDATES_PER_WARP;
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_get_view(const plan7_ssv_sequence_batch *batch,
                                  plan7_ssv_sequence_batch_view *view,
                                  char *error,
                                  size_t error_size)
{
  if (batch == nullptr || view == nullptr) {
    set_error(error, error_size, "sequence batch view argument is null");
    return -1;
  }
  memset(view, 0, sizeof(*view));
  if (batch->sequence_count != 0 &&
      (batch->host_lengths == nullptr || batch->device_residues == nullptr ||
       batch->device_offsets == nullptr)) {
    set_error(error, error_size, "sequence batch input storage is null");
    return -1;
  }
  view->generation_id = batch->generation_id;
  view->device_ordinal = batch->device_ordinal;
  view->alphabet_size = batch->alphabet_size;
  view->host_float_environment_valid =
    batch->host_float_environment_valid;
  view->sequence_count = batch->sequence_count;
  view->host_lengths = batch->host_lengths;
  view->device_residues = batch->device_residues;
  view->device_offsets = batch->device_offsets;
  view->input_device_bytes = batch->input_device_bytes;
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_get_forward_workspace(
  plan7_ssv_sequence_batch *batch,
  plan7_forward_workspace **workspace,
  char *error,
  size_t error_size)
{
  if (batch == nullptr || workspace == nullptr) {
    set_error(error, error_size, "invalid Forward workspace request");
    return -1;
  }
  int current_device = -1;
  const cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  if (batch->forward_workspace == nullptr &&
      plan7_forward_workspace_create(
        &batch->forward_workspace, error, error_size) != 0)
    return -1;
  *workspace = batch->forward_workspace;
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_get_workspace_statistics(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_workspace_statistics *statistics,
  char *error,
  size_t error_size)
{
  if (batch == nullptr || statistics == nullptr) {
    set_error(error, error_size, "invalid sequence workspace statistics");
    return -1;
  }
  memset(statistics, 0, sizeof(*statistics));
  statistics->f1_device_compaction_run_count =
    batch->f1_device_compaction_run_count;
  statistics->f1_host_expansion_run_count = batch->f1_host_expansion_run_count;
  statistics->f1_candidate_upload_count = batch->f1_candidate_upload_count;
  statistics->f1_candidate_upload_avoided_count =
    batch->f1_candidate_upload_avoided_count;
  statistics->f1_profile_packed_run_count =
    batch->f1_profile_packed_run_count;
  statistics->f1_profile_packed_quartet_count =
    batch->f1_profile_packed_quartet_count;
  statistics->f1_profile_packed_profile_count =
    batch->f1_profile_packed_profile_count;
  statistics->f1_profile_scalar_profile_count =
    batch->f1_profile_scalar_profile_count;
  statistics->f1_profile_packed_score_bytes =
    batch->f1_profile_packed_score_bytes;
  statistics->f1_identity_padding_run_count =
    batch->f1_identity_padding_run_count;
  statistics->f1_identity_padding_quartet_count =
    batch->f1_identity_padding_quartet_count;
  statistics->f1_identity_padding_profile_count =
    batch->f1_identity_padding_profile_count;
  statistics->f1_length_class_run_count =
    batch->f1_length_class_run_count;
  statistics->f1_length_class_value_count =
    batch->f1_length_class_value_count;
  statistics->f1_length_compact_h2d_bytes =
    batch->f1_length_compact_h2d_bytes;
  statistics->f1_length_dense_h2d_bytes_avoided =
    batch->f1_length_dense_h2d_bytes_avoided;
  statistics->f1_length_dense_materialized_bytes =
    batch->f1_length_dense_materialized_bytes;
  statistics->f1_raw_xe_run_count = batch->f1_raw_xe_run_count;
  statistics->f1_raw_xe_logical_pair_count =
    batch->f1_raw_xe_logical_pair_count;
  statistics->f1_raw_xe_sidecar_bytes_written =
    batch->f1_raw_xe_sidecar_bytes_written;
  statistics->f1_raw_xe_candidate_gather_count =
    batch->f1_raw_xe_candidate_gather_count;
  statistics->f1_candidate_ssv_replay_count =
    batch->f1_candidate_ssv_replay_count;
  statistics->f1_candidate_ssv_replay_avoided_count =
    batch->f1_candidate_ssv_replay_avoided_count;
  statistics->f1_raw_xe_fallback_run_count =
    batch->f1_raw_xe_fallback_run_count;
  if (batch->postfilter_workspace != nullptr) {
    plan7_postfilter_workspace_statistics postfilter{};
    if (plan7_postfilter_workspace_get_statistics(
          batch->postfilter_workspace, &postfilter, error, error_size) != 0)
      return -1;
    statistics->postfilter_device_bytes = postfilter.device_bytes;
    statistics->postfilter_dp_capacity_bytes =
      postfilter.dp_capacity_bytes;
    statistics->postfilter_growth_count = postfilter.growth_count;
    statistics->postfilter_run_count = postfilter.run_count;
    statistics->full_msv_compaction_run_count =
      postfilter.full_msv_compaction_run_count;
    statistics->full_msv_compaction_chunk_count =
      postfilter.full_msv_compaction_chunk_count;
    statistics->full_msv_compaction_source_count =
      postfilter.full_msv_compaction_source_count;
    statistics->full_msv_compaction_selected_count =
      postfilter.full_msv_compaction_selected_count;
    statistics->full_msv_legacy_run_count =
      postfilter.full_msv_legacy_run_count;
    statistics->full_msv_launch_candidate_count =
      postfilter.full_msv_launch_candidate_count;
    statistics->full_msv_launch_candidate_avoided_count =
      postfilter.full_msv_launch_candidate_avoided_count;
    statistics->full_msv_index_d2h_bytes =
      postfilter.full_msv_index_d2h_bytes;
    statistics->full_msv_packed_run_count =
      postfilter.full_msv_packed_run_count;
    statistics->full_msv_packed_group_count =
      postfilter.full_msv_packed_group_count;
    statistics->full_msv_packed_candidate_count =
      postfilter.full_msv_packed_candidate_count;
    statistics->full_msv_scalar_candidate_count =
      postfilter.full_msv_scalar_candidate_count;
    statistics->vit_length_cache_run_count =
      postfilter.vit_length_cache_run_count;
    statistics->vit_length_cache_entry_count =
      postfilter.vit_length_cache_entry_count;
    statistics->vit_length_cache_candidate_count =
      postfilter.vit_length_cache_candidate_count;
    statistics->vit_length_direct_candidate_count =
      postfilter.vit_length_direct_candidate_count;
    statistics->vit_length_cache_build_ns =
      postfilter.vit_length_cache_build_ns;
    statistics->vit_length_candidate_plan_ns =
      postfilter.vit_length_candidate_plan_ns;
  }
  if (batch->forward_workspace != nullptr) {
    plan7_forward_workspace_statistics forward{};
    if (plan7_forward_workspace_get_statistics(
          batch->forward_workspace, &forward, error, error_size) != 0)
      return -1;
    statistics->forward_device_bytes = forward.device_bytes;
    statistics->forward_dp_capacity_bytes = forward.dp_capacity_bytes;
    statistics->forward_xmx_capacity_bytes = forward.xmx_capacity_bytes;
    statistics->forward_gather_capacity_bytes = forward.gather_capacity_bytes;
    statistics->forward_growth_count = forward.growth_count;
    statistics->forward_event_create_count = forward.event_create_count;
    statistics->forward_run_count = forward.run_count;
  }
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_get_memory_snapshot(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_memory_snapshot *snapshot,
  char *error,
  size_t error_size)
{
  int current_device = -1;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t status;
  if (batch == nullptr || snapshot == nullptr) {
    set_error(error, error_size, "invalid sequence batch memory snapshot");
    return -1;
  }
  memset(snapshot, 0, sizeof(*snapshot));
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (plan7_validate_device_ordinal(batch->device_ordinal, current_device,
                                    error, error_size) != 0)
    return -1;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaMemGetInfo", status);
    return -1;
  }

  snapshot->device_ordinal = batch->device_ordinal;
  snapshot->cuda_free_bytes = static_cast<uint64_t>(free_bytes);
  snapshot->cuda_total_bytes = static_cast<uint64_t>(total_bytes);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_INPUT_RESIDUES] =
      static_cast<uint64_t>(batch->device_residue_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_INPUT_OFFSETS] =
      static_cast<uint64_t>(batch->device_offset_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_INPUT_NULL_SCORES] =
      static_cast<uint64_t>(batch->device_null_score_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_LENGTH_TJB] =
      static_cast<uint64_t>(batch->device_tjb_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_RESULTS] =
      static_cast<uint64_t>(batch->device_result_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_COMPACT_SCORES] =
      static_cast<uint64_t>(batch->device_score_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_PROFILES] =
      static_cast<uint64_t>(batch->device_profile_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_F1_PROFILES] =
      static_cast<uint64_t>(batch->device_f1_profile_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_CANDIDATE_WORDS] =
      static_cast<uint64_t>(batch->device_candidate_word_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_CANDIDATE_WORD_COUNTS] =
      static_cast<uint64_t>(batch->device_candidate_word_count_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_CANDIDATE_WORD_OFFSETS] =
      static_cast<uint64_t>(batch->device_candidate_word_offset_capacity);
  snapshot->device_capacity_bytes[
      PLAN7_SSV_CAPACITY_CANDIDATE_PROFILE_OFFSETS] =
      static_cast<uint64_t>(batch->device_candidate_profile_offset_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_CANDIDATE_SCAN_WORKSPACE] =
      static_cast<uint64_t>(batch->device_candidate_scan_workspace_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_F1_PROFILE_PACKED_SCORES] =
      static_cast<uint64_t>(batch->device_f1_profile_packed_score_capacity);
  snapshot->device_capacity_bytes[
      PLAN7_SSV_CAPACITY_F1_PROFILE_PACKED_QUARTETS] =
      static_cast<uint64_t>(batch->device_f1_profile_packed_quartet_capacity);
  snapshot->device_capacity_bytes[
      PLAN7_SSV_CAPACITY_F1_SCALAR_PROFILE_INDICES] =
      static_cast<uint64_t>(batch->device_f1_scalar_profile_index_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_LENGTH_CLASS_INDICES] =
      static_cast<uint64_t>(batch->device_length_class_index_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_F1_COMPACT_TJB] =
      static_cast<uint64_t>(batch->device_f1_compact_tjb_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_F1_RAW_XE] =
      static_cast<uint64_t>(batch->device_f1_raw_xe_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_BIAS_PROFILES] =
      static_cast<uint64_t>(batch->device_bias_profile_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_BIAS_CANDIDATES] =
      static_cast<uint64_t>(batch->device_bias_candidate_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_BIAS_SSV_INPUTS] =
      static_cast<uint64_t>(batch->device_bias_ssv_input_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_BIAS_RESULTS] =
      static_cast<uint64_t>(batch->device_bias_result_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_BIAS_LOGP] =
      static_cast<uint64_t>(batch->device_bias_logp_capacity);
  snapshot->device_capacity_bytes[PLAN7_SSV_CAPACITY_BIAS_LOG1MP] =
      static_cast<uint64_t>(batch->device_bias_log1mp_capacity);

  if (batch->postfilter_workspace != nullptr) {
    plan7_postfilter_workspace_statistics postfilter{};
    if (plan7_postfilter_workspace_get_statistics(
          batch->postfilter_workspace, &postfilter, error, error_size) != 0)
      return -1;
    for (size_t i = 0; i < PLAN7_POSTFILTER_CAPACITY_COUNT; ++i)
      snapshot->device_capacity_bytes[
          PLAN7_SSV_CAPACITY_POSTFILTER_STATES + i] =
          postfilter.capacity_bytes[i];
  }
  if (batch->forward_workspace != nullptr) {
    plan7_forward_workspace_statistics forward{};
    if (plan7_forward_workspace_get_statistics(
          batch->forward_workspace, &forward, error, error_size) != 0)
      return -1;
    for (size_t i = 0; i < PLAN7_FORWARD_CAPACITY_COUNT; ++i)
      snapshot->device_capacity_bytes[
          PLAN7_SSV_CAPACITY_FORWARD_CANDIDATE_PROFILES + i] =
          forward.capacity_bytes[i];
  }

  for (size_t i = 0; i < PLAN7_SSV_DEVICE_CAPACITY_COUNT; ++i) {
    if (!checked_add_u64(snapshot->persistent_device_bytes,
                         snapshot->device_capacity_bytes[i],
                         &snapshot->persistent_device_bytes)) {
      set_error(error, error_size, "sequence capacity total overflow");
      return -1;
    }
  }
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_filter(plan7_ssv_sequence_batch *batch,
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
                                size_t error_size)
{
  uint8_t *new_device_scores = nullptr;
  cudaError_t status;
  size_t score_bytes;
  int current_device;

  if (batch == nullptr || model_length < 1 || model_length > 100000 ||
      alphabet_size < 1 || alphabet_size != batch->alphabet_size) {
    set_error(error, error_size, "invalid SSV batch dimensions");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  const int Q = model_length < 17 ? 2 : (model_length + 15) / 16;
  const int minimum_stride = 16 * (Q + kExtraScoreVectors);
  if (score_stride < minimum_stride || striped_scores == nullptr ||
      !checked_product(static_cast<size_t>(score_stride),
                       static_cast<size_t>(alphabet_size), &score_bytes) ||
      striped_score_count < score_bytes) {
    set_error(error, error_size, "invalid striped score buffer");
    return -1;
  }
  if (!isfinite(scale) || scale <= 0.0f ||
      (batch->sequence_count != 0 &&
       (results == nullptr || result_count < batch->sequence_count))) {
    set_error(error, error_size, "invalid SSV result buffer or scale");
    return -1;
  }
  if (batch->sequence_count == 0) return 0;
  invalidate_f1_cache(batch);
  if (!batch->host_float_environment_valid ||
      plan7_bias_host_environment_attested() != 1) {
    fill_cpu_required_results(results, batch->sequence_count);
    return 0;
  }

#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  if (score_bytes > batch->device_score_capacity) {
    CUDA_TRY(cudaMalloc(&new_device_scores, score_bytes));
    status = cudaFree(batch->device_scores);
    if (status != cudaSuccess) {
      cudaFree(new_device_scores);
      set_cuda_error(error, error_size, "cudaFree(device_scores)", status);
      return -1;
    }
    batch->device_scores = new_device_scores;
    batch->device_score_capacity = score_bytes;
  }
  CUDA_TRY(cudaMemcpy(batch->device_scores, striped_scores, score_bytes,
                      cudaMemcpyHostToDevice));
  if (!batch->tjb_cache_valid || scale != batch->cached_tjb_scale) {
    for (size_t i = 0; i < batch->sequence_count; ++i)
      batch->host_tjb[i] =
        compute_tjb_from_log_term(scale, batch->host_tjb_log_terms[i]);
    batch->tjb_cache_valid = 0;
    CUDA_TRY(cudaMemcpy(batch->device_tjb, batch->host_tjb,
                        batch->sequence_count, cudaMemcpyHostToDevice));
    batch->cached_tjb_scale = scale;
    batch->tjb_cache_valid = 1;
  }

  ssv_kernel<<<static_cast<unsigned>(batch->sequence_count), kThreads>>>(
    batch->device_scores, score_stride, model_length, batch->device_residues,
    batch->device_offsets, batch->device_tjb, tbm, tec, base, bias,
    batch->device_results);
  CUDA_TRY(cudaGetLastError());
  CUDA_TRY(cudaMemcpy(results, batch->device_results,
                      batch->sequence_count * sizeof(*results),
                      cudaMemcpyDeviceToHost));
#undef CUDA_TRY
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_filter_many(
  plan7_ssv_sequence_batch *batch,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const plan7_ssv_profile *profiles,
  size_t profile_count,
  plan7_ssv_result *profile_major_results,
  size_t result_count,
  char *error,
  size_t error_size)
{
  cudaError_t status;
  size_t cell_count;
  size_t result_bytes;
  size_t profile_bytes;
  uint8_t *new_host_tjb;
  int current_device;
  int maximum_grid_x;
  int maximum_grid_y;
  size_t expected_score_offset = 0;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  if (profile_count == 0) return 0;

  status = cudaDeviceGetAttribute(
    &maximum_grid_x, cudaDevAttrMaxGridDimX, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid x)", status);
    return -1;
  }
  status = cudaDeviceGetAttribute(
    &maximum_grid_y, cudaDevAttrMaxGridDimY, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid y)", status);
    return -1;
  }
  if (batch->sequence_count > static_cast<size_t>(maximum_grid_x) ||
      profile_count > static_cast<size_t>(maximum_grid_y)) {
    set_error(error, error_size, "multi-profile CUDA grid is too large");
    return -1;
  }
  if (packed_scores == nullptr || profiles == nullptr) {
    set_error(error, error_size, "packed profile buffers are null");
    return -1;
  }
  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (!validate_compact_profile(
          &profiles[profile], batch->alphabet_size, packed_score_count,
          expected_score_offset, error, error_size))
      return -1;
    expected_score_offset += static_cast<size_t>(profiles[profile].score_count);
  }
  if (expected_score_offset != packed_score_count) {
    set_error(error, error_size, "compact profile scores have trailing bytes");
    return -1;
  }

  if (!checked_product(profile_count, batch->sequence_count, &cell_count) ||
      !checked_product(cell_count, sizeof(plan7_ssv_result), &result_bytes) ||
      !checked_product(profile_count, sizeof(plan7_ssv_profile),
                       &profile_bytes)) {
    set_error(error, error_size, "multi-profile batch size overflow");
    return -1;
  }
  if (cell_count != 0 &&
      (profile_major_results == nullptr || result_count < cell_count)) {
    set_error(error, error_size, "multi-profile result buffer is too short");
    return -1;
  }
  if (batch->sequence_count == 0) return 0;
  invalidate_f1_cache(batch);
  if (!batch->host_float_environment_valid ||
      plan7_bias_host_environment_attested() != 1) {
    fill_cpu_required_results(profile_major_results, cell_count);
    return 0;
  }

  if (cell_count > batch->host_tjb_capacity) {
    new_host_tjb = static_cast<uint8_t *>(
      realloc(batch->host_tjb, cell_count));
    if (new_host_tjb == nullptr) {
      set_error(error, error_size,
                "host multi-profile transition allocation failed");
      return -1;
    }
    batch->host_tjb = new_host_tjb;
    batch->host_tjb_capacity = cell_count;
  }
  if (grow_device_buffer(&batch->device_scores,
                         &batch->device_score_capacity,
                         packed_score_count,
                         "cudaMalloc(packed profile scores)",
                         "cudaFree(packed profile scores)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_profiles,
                         &batch->device_profile_capacity,
                         profile_bytes,
                         "cudaMalloc(profile descriptors)",
                         "cudaFree(profile descriptors)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_tjb,
                         &batch->device_tjb_capacity,
                         cell_count,
                         "cudaMalloc(profile transitions)",
                         "cudaFree(profile transitions)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_results,
                         &batch->device_result_capacity,
                         result_bytes,
                         "cudaMalloc(profile results)",
                         "cudaFree(profile results)",
                         error,
                         error_size) != 0)
    return -1;

#define CUDA_TRY_MANY(call)                                                   \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  CUDA_TRY_MANY(cudaMemcpy(batch->device_scores,
                           packed_scores,
                           packed_score_count,
                           cudaMemcpyHostToDevice));
  CUDA_TRY_MANY(cudaMemcpy(batch->device_profiles,
                           profiles,
                           profile_bytes,
                           cudaMemcpyHostToDevice));
  for (size_t profile = 0; profile < profile_count; ++profile) {
    const size_t row = profile * batch->sequence_count;
    if (profile != 0 && profiles[profile].scale == profiles[profile - 1].scale) {
      memcpy(batch->host_tjb + row,
             batch->host_tjb + row - batch->sequence_count,
             batch->sequence_count);
    } else {
      for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence)
        batch->host_tjb[row + sequence] = compute_tjb_from_log_term(
          profiles[profile].scale, batch->host_tjb_log_terms[sequence]);
    }
  }
  batch->tjb_cache_valid = 0;
  CUDA_TRY_MANY(cudaMemcpy(batch->device_tjb,
                           batch->host_tjb,
                           cell_count,
                           cudaMemcpyHostToDevice));

  const dim3 grid(static_cast<unsigned>(batch->sequence_count),
                  static_cast<unsigned>(profile_count));
  ssv_many_kernel<<<grid, kThreads>>>(
    batch->device_scores,
    batch->device_profiles,
    batch->sequence_count,
    batch->device_residues,
    batch->device_offsets,
    batch->device_tjb,
    batch->device_results);
  CUDA_TRY_MANY(cudaGetLastError());
  CUDA_TRY_MANY(cudaMemcpy(profile_major_results,
                           batch->device_results,
                           result_bytes,
                           cudaMemcpyDeviceToHost));
#undef CUDA_TRY_MANY
  return 0;
}

static int
sequence_batch_f1_mask_many_impl(
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
  bool copy_candidate_words,
  char *error,
  size_t error_size)
{
  cudaError_t status;
  size_t expected_score_offset = 0;
  size_t words_per_profile;
  size_t required_word_count;
  size_t candidate_word_bytes;
  size_t f1_profile_bytes;
  size_t unique_tjb_rows = 0;
  size_t tjb_count;
  size_t compact_tjb_count = 0;
  size_t host_tjb_count;
  bool use_length_classes = false;
  bool raw_xe_reconstructable = false;
  size_t raw_xe_count = 0;
  size_t packed_score_word_count = 0;
  size_t packed_score_bytes = 0;
  size_t packed_quartet_bytes = 0;
  size_t scalar_profile_index_bytes = 0;
  size_t maximum_quartet_score_words = 0;
  std::vector<ProfilePackedQuartet> profile_quartets;
  std::vector<uint32_t> scalar_profile_indices;
  int current_device;
  int maximum_grid_x;
  int maximum_grid_y;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  batch->execution_policy_locked = 1;
  ++batch->execution_policy_f1_run_count;
  invalidate_f1_cache(batch);
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  if (profile_count == 0) return 0;
  if (packed_scores == nullptr || profiles == nullptr || m_mu == nullptr ||
      m_lambda == nullptr) {
    set_error(error, error_size, "fused F1 profile buffers are null");
    return -1;
  }

  status = cudaDeviceGetAttribute(
    &maximum_grid_x, cudaDevAttrMaxGridDimX, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid x)", status);
    return -1;
  }
  status = cudaDeviceGetAttribute(
    &maximum_grid_y, cudaDevAttrMaxGridDimY, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid y)", status);
    return -1;
  }
  if (batch->sequence_count > static_cast<size_t>(maximum_grid_x) ||
      profile_count > static_cast<size_t>(maximum_grid_y)) {
    set_error(error, error_size, "fused F1 CUDA grid is too large");
    return -1;
  }

  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (!validate_compact_profile(
          &profiles[profile], batch->alphabet_size, packed_score_count,
          expected_score_offset, error, error_size))
      return -1;
    expected_score_offset += static_cast<size_t>(profiles[profile].score_count);
  }
  if (expected_score_offset != packed_score_count) {
    set_error(error, error_size, "compact profile scores have trailing bytes");
    return -1;
  }

  if (batch->sequence_count > SIZE_MAX - 31) {
    set_error(error, error_size, "fused F1 mask size overflow");
    return -1;
  }
  words_per_profile = (batch->sequence_count + 31) / 32;
  if (!checked_product(profile_count, words_per_profile,
                       &required_word_count) ||
      !checked_product(required_word_count, sizeof(uint32_t),
                       &candidate_word_bytes) ||
      !checked_product(profile_count, sizeof(plan7_ssv_f1_profile),
                       &f1_profile_bytes)) {
    set_error(error, error_size, "fused F1 mask size overflow");
    return -1;
  }
  if (copy_candidate_words && required_word_count != 0 &&
      (profile_major_candidate_words == nullptr ||
       candidate_word_count < required_word_count)) {
    set_error(error, error_size, "fused F1 mask buffer is too short");
    return -1;
  }
  if (batch->sequence_count == 0) return 0;

  if (f1_profile_bytes > batch->host_f1_profile_capacity) {
    void *replacement = realloc(batch->host_f1_profiles, f1_profile_bytes);
    if (replacement == nullptr) {
      set_error(error, error_size, "host fused F1 profile allocation failed");
      return -1;
    }
    batch->host_f1_profiles =
      static_cast<plan7_ssv_f1_profile *>(replacement);
    batch->host_f1_profile_capacity = f1_profile_bytes;
  }
  if (packed_score_count > batch->host_f1_score_capacity) {
    void *replacement = realloc(batch->host_f1_scores, packed_score_count);
    if (replacement == nullptr) {
      set_error(error, error_size, "host fused profile score allocation failed");
      return -1;
    }
    batch->host_f1_scores = static_cast<uint8_t *>(replacement);
    batch->host_f1_score_capacity = packed_score_count;
  }
  memcpy(batch->host_f1_scores, packed_scores, packed_score_count);
  batch->host_f1_score_count = packed_score_count;

  const bool float_environment_valid =
    batch->host_float_environment_valid &&
    plan7_bias_host_environment_attested() == 1;
  for (size_t profile = 0; profile < profile_count; ++profile) {
    plan7_ssv_f1_profile *f1_profile = &batch->host_f1_profiles[profile];
    float cutoff = NAN;
    size_t tjb_offset = SIZE_MAX;
    f1_profile->profile = profiles[profile];
    f1_profile->cutoff_mode = float_environment_valid
      ? derive_f1_cutoff(m_mu[profile], m_lambda[profile], f1, &cutoff)
      : PLAN7_F1_CUTOFF_INVALID;
    f1_profile->cutoff_bit_score = cutoff;

    for (size_t previous = 0; previous < profile; ++previous) {
      if (profiles[previous].scale == profiles[profile].scale) {
        tjb_offset = static_cast<size_t>(
          batch->host_f1_profiles[previous].tjb_offset);
        break;
      }
    }
    if (tjb_offset == SIZE_MAX) {
      if (!checked_product(unique_tjb_rows, batch->sequence_count,
                           &tjb_offset)) {
        set_error(error, error_size, "fused F1 transition size overflow");
        return -1;
      }
      ++unique_tjb_rows;
    }
    f1_profile->tjb_offset = tjb_offset;
  }
  const char *raw_xe_policy = getenv("PLAN7_GPU_F1_RAW_XE");
  size_t raw_xe_logical_pair_count = 0;
  const bool auto_raw_xe =
    raw_xe_policy == nullptr &&
    batch->execution_policy_mode != PLAN7_GPU_EXECUTION_POLICY_SIMPLE &&
    checked_product(profile_count, batch->sequence_count,
                    &raw_xe_logical_pair_count) &&
    raw_xe_logical_pair_count >= 65536;
  raw_xe_reconstructable =
    (auto_raw_xe ||
     (raw_xe_policy != nullptr && strcmp(raw_xe_policy, "1") == 0)) &&
    float_environment_valid;
  if (raw_xe_reconstructable) {
    for (size_t profile = 0; profile < profile_count; ++profile) {
      const int mode = batch->host_f1_profiles[profile].cutoff_mode;
      if (mode != PLAN7_F1_CUTOFF_SCORE &&
          mode != PLAN7_F1_CUTOFF_ALWAYS_REJECT) {
        raw_xe_reconstructable = false;
        break;
      }
    }
  }

  const char *profile_policy = getenv("PLAN7_GPU_SSV_PROFILE_POLICY");
  const bool profile_packing_enabled =
    (profile_policy == nullptr || strcmp(profile_policy, "scalar") != 0) &&
    batch->execution_policy_mode != PLAN7_GPU_EXECUTION_POLICY_SIMPLE;
  const char *identity_padding_policy =
    getenv("PLAN7_GPU_SSV_IDENTITY_PADDING");
  const bool use_identity_padding =
    identity_padding_policy == nullptr ||
    strcmp(identity_padding_policy, "1") == 0;
  const size_t profile_packed_minimum =
    batch->execution_policy_mode == PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT
      ? kProfilesPerPackedWord : kProfilePackedMinimumProfiles;
  if (profile_packing_enabled &&
      profile_count >= profile_packed_minimum) {
    std::vector<uint32_t> eligible;
    try {
      eligible.reserve(profile_count);
      scalar_profile_indices.reserve(profile_count);
      profile_quartets.reserve(profile_count / kProfilesPerPackedWord);
      for (size_t profile = 0; profile < profile_count; ++profile) {
        const int mode = batch->host_f1_profiles[profile].cutoff_mode;
        if (mode == PLAN7_F1_CUTOFF_SCORE ||
            mode == PLAN7_F1_CUTOFF_ALWAYS_REJECT)
          eligible.push_back(static_cast<uint32_t>(profile));
        else
          scalar_profile_indices.push_back(static_cast<uint32_t>(profile));
      }
      std::stable_sort(
        eligible.begin(), eligible.end(),
        [batch](uint32_t left, uint32_t right) {
          return batch->host_f1_profiles[left].profile.model_length <
                 batch->host_f1_profiles[right].profile.model_length;
        });
      size_t cursor = 0;
      while (cursor < eligible.size()) {
        if (eligible.size() - cursor >= kProfilesPerPackedWord) {
          const int minimum_model_length =
            batch->host_f1_profiles[eligible[cursor]].profile.model_length;
          const int maximum_model_length =
            batch->host_f1_profiles[
              eligible[cursor + kProfilesPerPackedWord - 1]]
              .profile.model_length;
          const int tolerated_spread =
            std::max(16, minimum_model_length / 4);
          if (maximum_model_length <=
              minimum_model_length + tolerated_spread) {
            ProfilePackedQuartet quartet{};
            quartet.score_word_offset = packed_score_word_count;
            quartet.maximum_model_length =
              static_cast<uint32_t>(maximum_model_length);
            for (int lane = 0; lane < kProfilesPerPackedWord; ++lane)
              quartet.profile_indices[lane] = eligible[cursor + lane];
            size_t quartet_words;
            if (!checked_product(
                  static_cast<size_t>(maximum_model_length), 29,
                  &quartet_words) ||
                quartet_words > SIZE_MAX - packed_score_word_count) {
              set_error(error, error_size,
                        "profile-packed F1 score size overflow");
              return -1;
            }
            packed_score_word_count += quartet_words;
            maximum_quartet_score_words =
              std::max(maximum_quartet_score_words, quartet_words);
            profile_quartets.push_back(quartet);
            cursor += kProfilesPerPackedWord;
            continue;
          }
        }
        scalar_profile_indices.push_back(eligible[cursor]);
        ++cursor;
      }
    } catch (...) {
      set_error(error, error_size,
                "profile-packed F1 execution plan allocation failed");
      return -1;
    }
  }
  if (profile_quartets.empty()) scalar_profile_indices.clear();
  if (!checked_product(packed_score_word_count, sizeof(uint32_t),
                       &packed_score_bytes) ||
      !checked_product(profile_quartets.size(), sizeof(ProfilePackedQuartet),
                       &packed_quartet_bytes) ||
      !checked_product(scalar_profile_indices.size(), sizeof(uint32_t),
                       &scalar_profile_index_bytes)) {
    set_error(error, error_size, "profile-packed F1 allocation size overflow");
    return -1;
  }
  if (!checked_product(unique_tjb_rows, batch->sequence_count, &tjb_count)) {
    set_error(error, error_size, "fused F1 transition size overflow");
    return -1;
  }
  if (!checked_product(unique_tjb_rows, batch->length_class_count,
                       &compact_tjb_count)) {
    set_error(error, error_size,
              "length-class F1 transition size overflow");
    return -1;
  }
  const char *length_policy = getenv("PLAN7_GPU_SSV_LENGTH_METADATA");
  const bool force_length_classes =
    length_policy != nullptr && strcmp(length_policy, "compact") == 0;
  const bool disable_length_classes =
    length_policy != nullptr && strcmp(length_policy, "expanded") == 0;
  const size_t expansion_blocks = (tjb_count + kThreads - 1) / kThreads;
  use_length_classes =
    !disable_length_classes && batch->device_length_class_indices != nullptr &&
    batch->length_class_count != 0 &&
    expansion_blocks <= static_cast<size_t>(maximum_grid_x) &&
    (force_length_classes ||
     batch->execution_policy_mode == PLAN7_GPU_EXECUTION_POLICY_THROUGHPUT ||
     (batch->sequence_count >= kLengthClassMinimumSequences &&
      batch->length_class_count <= batch->sequence_count / 2)) &&
    batch->execution_policy_mode != PLAN7_GPU_EXECUTION_POLICY_SIMPLE;
  host_tjb_count = use_length_classes ? compact_tjb_count : tjb_count;
  if (host_tjb_count > batch->host_tjb_capacity) {
    void *replacement = realloc(batch->host_tjb, host_tjb_count);
    if (replacement == nullptr) {
      set_error(error, error_size,
                "host fused F1 transition allocation failed");
      return -1;
    }
    batch->host_tjb = static_cast<uint8_t *>(replacement);
    batch->host_tjb_capacity = tjb_count;
  }
  for (size_t profile = 0; profile < profile_count; ++profile) {
    const size_t row_offset = static_cast<size_t>(
      batch->host_f1_profiles[profile].tjb_offset);
    bool first_for_scale = true;
    for (size_t previous = 0; previous < profile; ++previous) {
      if (batch->host_f1_profiles[previous].tjb_offset ==
          batch->host_f1_profiles[profile].tjb_offset) {
        first_for_scale = false;
        break;
      }
    }
    if (!first_for_scale) continue;
    if (use_length_classes) {
      const size_t compact_row =
        (row_offset / batch->sequence_count) * batch->length_class_count;
      for (size_t length_class = 0;
           length_class < batch->length_class_count; ++length_class) {
        const uint8_t value = compute_tjb_from_log_term(
          profiles[profile].scale,
          batch->host_length_class_log_terms[length_class]);
        batch->host_tjb[compact_row + length_class] = value;
      }
    } else {
      for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
        const uint8_t value = compute_tjb_from_log_term(
          profiles[profile].scale, batch->host_tjb_log_terms[sequence]);
        batch->host_tjb[row_offset + sequence] = value;
      }
    }
  }
  if (raw_xe_reconstructable &&
      !checked_product(profile_count, batch->sequence_count, &raw_xe_count)) {
    set_error(error, error_size, "F1 raw-xE sidecar size overflow");
    return -1;
  }
  batch->tjb_cache_valid = 0;

  if (grow_device_buffer(&batch->device_scores,
                         &batch->device_score_capacity,
                         packed_score_count,
                         "cudaMalloc(fused profile scores)",
                         "cudaFree(fused profile scores)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_f1_profiles,
                         &batch->device_f1_profile_capacity,
                         f1_profile_bytes,
                         "cudaMalloc(fused F1 profiles)",
                         "cudaFree(fused F1 profiles)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_f1_profile_packed_scores,
                         &batch->device_f1_profile_packed_score_capacity,
                         packed_score_bytes,
                         "cudaMalloc(profile-packed F1 scores)",
                         "cudaFree(profile-packed F1 scores)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_f1_profile_packed_quartets,
                         &batch->device_f1_profile_packed_quartet_capacity,
                         packed_quartet_bytes,
                         "cudaMalloc(profile-packed F1 quartets)",
                         "cudaFree(profile-packed F1 quartets)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_f1_scalar_profile_indices,
                         &batch->device_f1_scalar_profile_index_capacity,
                         scalar_profile_index_bytes,
                         "cudaMalloc(profile-packed F1 scalar indexes)",
                         "cudaFree(profile-packed F1 scalar indexes)",
                         error,
                         error_size) != 0 ||
      (use_length_classes &&
       grow_device_buffer(&batch->device_f1_compact_tjb,
                          &batch->device_f1_compact_tjb_capacity,
                          compact_tjb_count,
                          "cudaMalloc(length-class F1 transitions)",
                          "cudaFree(length-class F1 transitions)",
                          error,
                          error_size) != 0) ||
      grow_device_buffer(&batch->device_tjb,
                         &batch->device_tjb_capacity,
                         tjb_count,
                         "cudaMalloc(fused profile transitions)",
                         "cudaFree(fused profile transitions)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_candidate_words,
                         &batch->device_candidate_word_capacity,
                         candidate_word_bytes,
                         "cudaMalloc(fused F1 mask)",
                         "cudaFree(fused F1 mask)",
                         error,
                         error_size) != 0 ||
      (raw_xe_reconstructable &&
       grow_device_buffer(&batch->device_f1_raw_xe,
                          &batch->device_f1_raw_xe_capacity,
                          raw_xe_count,
                          "cudaMalloc(F1 raw-xE sidecar)",
                          "cudaFree(F1 raw-xE sidecar)",
                          error,
                          error_size) != 0))
    return -1;

#define CUDA_TRY_FUSED(call)                                                  \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  CUDA_TRY_FUSED(cudaMemcpy(batch->device_scores,
                            packed_scores,
                            packed_score_count,
                            cudaMemcpyHostToDevice));
  CUDA_TRY_FUSED(cudaMemcpy(batch->device_f1_profiles,
                            batch->host_f1_profiles,
                            f1_profile_bytes,
                            cudaMemcpyHostToDevice));
  if (packed_quartet_bytes != 0)
    CUDA_TRY_FUSED(cudaMemcpy(
      batch->device_f1_profile_packed_quartets,
      profile_quartets.data(), packed_quartet_bytes, cudaMemcpyHostToDevice));
  if (scalar_profile_index_bytes != 0)
    CUDA_TRY_FUSED(cudaMemcpy(
      batch->device_f1_scalar_profile_indices,
      scalar_profile_indices.data(), scalar_profile_index_bytes,
      cudaMemcpyHostToDevice));
  if (use_length_classes) {
    CUDA_TRY_FUSED(cudaMemcpy(batch->device_f1_compact_tjb,
                              batch->host_tjb,
                              compact_tjb_count,
                              cudaMemcpyHostToDevice));
    expand_length_class_tjb_kernel<<<
      static_cast<unsigned>(expansion_blocks), kThreads>>>(
        batch->device_f1_compact_tjb,
        batch->device_length_class_indices,
        batch->sequence_count,
        batch->length_class_count,
        tjb_count,
        batch->device_tjb);
    CUDA_TRY_FUSED(cudaGetLastError());
  } else {
    CUDA_TRY_FUSED(cudaMemcpy(batch->device_tjb,
                              batch->host_tjb,
                              tjb_count,
                              cudaMemcpyHostToDevice));
  }
  CUDA_TRY_FUSED(cudaMemset(batch->device_candidate_words,
                            0,
                            candidate_word_bytes));

  const unsigned sequence_blocks = static_cast<unsigned>(
    (batch->sequence_count + kSequencesPerBlock - 1) /
    kSequencesPerBlock);
  if (profile_quartets.empty()) {
    const dim3 grid(sequence_blocks, static_cast<unsigned>(profile_count));
    ssv_f1_mask_many_kernel<<<grid, kThreads>>>(
      batch->device_scores, batch->device_f1_profiles,
      batch->sequence_count, batch->device_residues, batch->device_offsets,
      batch->device_null_scores, batch->device_tjb, words_per_profile,
      batch->device_candidate_words,
      raw_xe_reconstructable ? batch->device_f1_raw_xe : nullptr);
    CUDA_TRY_FUSED(cudaGetLastError());
  } else {
    const size_t requested_pack_blocks =
      (maximum_quartet_score_words + kThreads - 1) / kThreads;
    const unsigned pack_blocks = static_cast<unsigned>(std::min(
      requested_pack_blocks, static_cast<size_t>(maximum_grid_x)));
    const auto *device_quartets =
      static_cast<const ProfilePackedQuartet *>(
        batch->device_f1_profile_packed_quartets);
    pack_profile_quartet_scores_kernel<<<
      dim3(pack_blocks, static_cast<unsigned>(profile_quartets.size())),
      kThreads>>>(
      batch->device_scores, batch->device_f1_profiles, device_quartets,
        use_identity_padding,
        batch->device_f1_profile_packed_scores);
    CUDA_TRY_FUSED(cudaGetLastError());
    if (use_identity_padding) {
      ssv_f1_mask_profile_packed_kernel<true><<<
        dim3(sequence_blocks, static_cast<unsigned>(profile_quartets.size())),
        kThreads>>>(
          batch->device_f1_profile_packed_scores, device_quartets,
          batch->device_f1_profiles, batch->sequence_count,
          batch->device_residues, batch->device_offsets,
          batch->device_null_scores, batch->device_tjb, words_per_profile,
          batch->device_candidate_words,
          raw_xe_reconstructable ? batch->device_f1_raw_xe : nullptr);
    } else {
      ssv_f1_mask_profile_packed_kernel<false><<<
        dim3(sequence_blocks, static_cast<unsigned>(profile_quartets.size())),
        kThreads>>>(
          batch->device_f1_profile_packed_scores, device_quartets,
          batch->device_f1_profiles, batch->sequence_count,
          batch->device_residues, batch->device_offsets,
          batch->device_null_scores, batch->device_tjb, words_per_profile,
          batch->device_candidate_words,
          raw_xe_reconstructable ? batch->device_f1_raw_xe : nullptr);
    }
    CUDA_TRY_FUSED(cudaGetLastError());
    if (!scalar_profile_indices.empty()) {
      ssv_f1_mask_indexed_kernel<<<
        dim3(sequence_blocks,
             static_cast<unsigned>(scalar_profile_indices.size())),
        kThreads>>>(
          batch->device_scores, batch->device_f1_profiles,
          batch->device_f1_scalar_profile_indices, batch->sequence_count,
          batch->device_residues, batch->device_offsets,
          batch->device_null_scores, batch->device_tjb, words_per_profile,
          batch->device_candidate_words,
          raw_xe_reconstructable ? batch->device_f1_raw_xe : nullptr);
      CUDA_TRY_FUSED(cudaGetLastError());
    }
    ++batch->f1_profile_packed_run_count;
    batch->f1_profile_packed_quartet_count += profile_quartets.size();
    batch->f1_profile_packed_profile_count +=
      profile_quartets.size() * kProfilesPerPackedWord;
    batch->f1_profile_scalar_profile_count += scalar_profile_indices.size();
    batch->f1_profile_packed_score_bytes += packed_score_bytes;
    if (use_identity_padding) {
      ++batch->f1_identity_padding_run_count;
      batch->f1_identity_padding_quartet_count += profile_quartets.size();
      batch->f1_identity_padding_profile_count +=
        profile_quartets.size() * kProfilesPerPackedWord;
    }
  }
  if (copy_candidate_words)
    CUDA_TRY_FUSED(cudaMemcpy(profile_major_candidate_words,
                              batch->device_candidate_words,
                              candidate_word_bytes,
                              cudaMemcpyDeviceToHost));
  batch->cached_f1_profile_count = profile_count;
  batch->f1_cache_valid = 1;
  batch->f1_raw_xe_valid = raw_xe_reconstructable ? 1 : 0;
  if (raw_xe_reconstructable) {
    ++batch->f1_raw_xe_run_count;
    batch->f1_raw_xe_logical_pair_count += raw_xe_count;
    batch->f1_raw_xe_sidecar_bytes_written += raw_xe_count;
  } else {
    ++batch->f1_raw_xe_fallback_run_count;
  }
  if (use_length_classes) {
    ++batch->f1_length_class_run_count;
    batch->f1_length_class_value_count += batch->length_class_count;
    batch->f1_length_compact_h2d_bytes += compact_tjb_count;
    batch->f1_length_dense_h2d_bytes_avoided += tjb_count;
    batch->f1_length_dense_materialized_bytes += tjb_count;
  }
#undef CUDA_TRY_FUSED
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_f1_mask_many(
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
  size_t error_size)
{
  const int status = sequence_batch_f1_mask_many_impl(
    batch, packed_scores, packed_score_count, profiles, profile_count,
    m_mu, m_lambda, f1, profile_major_candidate_words, candidate_word_count,
    true, error, error_size);
  if (status == 0 && batch != nullptr)
    ++batch->f1_host_expansion_run_count;
  return status;
}

extern "C" int
plan7_ssv_sequence_batch_f1_compact_many(
  plan7_ssv_sequence_batch *batch,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const plan7_ssv_profile *profiles,
  size_t profile_count,
  const float *m_mu,
  const float *m_lambda,
  double f1,
  char *error,
  size_t error_size)
{
  size_t words_per_profile;
  size_t word_count;
  size_t word_bytes;
  size_t profile_offset_bytes;
  size_t candidate_bytes;
  size_t ssv_input_bytes;
  size_t scan_workspace_bytes = 0;
  cudaError_t status;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  invalidate_f1_device_candidates(batch);
  if (batch->sequence_count > SIZE_MAX - 31) {
    set_error(error, error_size, "fused F1 mask size overflow");
    return -1;
  }
  words_per_profile = (batch->sequence_count + 31) / 32;
  if (!checked_product(profile_count, words_per_profile, &word_count)) {
    set_error(error, error_size, "fused F1 mask size overflow");
    return -1;
  }
  if (word_count > static_cast<size_t>(INT_MAX)) {
    set_error(error, error_size,
              "fused F1 mask exceeds the CUB scan item limit");
    return -1;
  }

  if (sequence_batch_f1_mask_many_impl(
        batch, packed_scores, packed_score_count, profiles, profile_count,
        m_mu, m_lambda, f1, nullptr, 0, false, error, error_size) != 0)
    return -1;

  if (profile_count == SIZE_MAX ||
      !checked_product(profile_count + 1, sizeof(size_t),
                       &profile_offset_bytes)) {
    set_error(error, error_size, "F1 candidate offset size overflow");
    return -1;
  }
  if (profile_offset_bytes > batch->host_candidate_offset_capacity) {
    void *replacement = realloc(batch->host_candidate_offsets,
                                profile_offset_bytes);
    if (replacement == nullptr) {
      set_error(error, error_size, "host F1 candidate offset allocation failed");
      return -1;
    }
    batch->host_candidate_offsets = static_cast<size_t *>(replacement);
    batch->host_candidate_offset_capacity = profile_offset_bytes;
  }

  if (word_count == 0) {
    for (size_t profile = 0; profile <= profile_count; ++profile)
      batch->host_candidate_offsets[profile] = 0;
    batch->cached_f1_profile_count = profile_count;
    batch->cached_f1_candidate_count = 0;
    batch->f1_cache_valid = 1;
    batch->f1_device_candidates_valid = 1;
    batch->f1_compact_ssv_inputs_valid = batch->f1_raw_xe_valid;
    ++batch->f1_device_compaction_run_count;
    return 0;
  }

  if (!checked_product(word_count, sizeof(uint64_t), &word_bytes)) {
    set_error(error, error_size, "F1 candidate scan size overflow");
    return -1;
  }
  if (grow_device_buffer(&batch->device_candidate_word_counts,
                         &batch->device_candidate_word_count_capacity,
                         word_bytes,
                         "cudaMalloc(F1 candidate word counts)",
                         "cudaFree(F1 candidate word counts)",
                         error, error_size) != 0 ||
      grow_device_buffer(&batch->device_candidate_word_offsets,
                         &batch->device_candidate_word_offset_capacity,
                         word_bytes,
                         "cudaMalloc(F1 candidate word offsets)",
                         "cudaFree(F1 candidate word offsets)",
                         error, error_size) != 0 ||
      grow_device_buffer(&batch->device_candidate_profile_offsets,
                         &batch->device_candidate_profile_offset_capacity,
                         profile_offset_bytes,
                         "cudaMalloc(F1 candidate profile offsets)",
                         "cudaFree(F1 candidate profile offsets)",
                         error, error_size) != 0)
    return -1;
  status = cub::DeviceScan::ExclusiveSum(
    nullptr, scan_workspace_bytes, batch->device_candidate_word_counts,
    batch->device_candidate_word_offsets, static_cast<int>(word_count));
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "CUB F1 scan workspace query", status);
    return -1;
  }
  if (grow_device_buffer(&batch->device_candidate_scan_workspace,
                         &batch->device_candidate_scan_workspace_capacity,
                         scan_workspace_bytes,
                         "cudaMalloc(F1 candidate scan workspace)",
                         "cudaFree(F1 candidate scan workspace)",
                         error, error_size) != 0)
    return -1;

#define CUDA_TRY_COMPACT(call)                                                \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  const unsigned word_blocks = static_cast<unsigned>(
    (word_count + kThreads - 1) / kThreads);
  candidate_word_counts_kernel<<<word_blocks, kThreads>>>(
    batch->device_candidate_words, word_count,
    batch->device_candidate_word_counts);
  CUDA_TRY_COMPACT(cudaGetLastError());
  CUDA_TRY_COMPACT(cub::DeviceScan::ExclusiveSum(
    batch->device_candidate_scan_workspace, scan_workspace_bytes,
    batch->device_candidate_word_counts,
    batch->device_candidate_word_offsets, static_cast<int>(word_count)));

  const unsigned profile_blocks = static_cast<unsigned>(
    (profile_count + 1 + kThreads - 1) / kThreads);
  candidate_profile_offsets_kernel<<<profile_blocks, kThreads>>>(
    batch->device_candidate_word_counts,
    batch->device_candidate_word_offsets,
    words_per_profile,
    profile_count,
    batch->device_candidate_profile_offsets);
  CUDA_TRY_COMPACT(cudaGetLastError());
  CUDA_TRY_COMPACT(cudaMemcpy(batch->host_candidate_offsets,
                              batch->device_candidate_profile_offsets,
                              profile_offset_bytes,
                              cudaMemcpyDeviceToHost));

  const size_t candidate_count = batch->host_candidate_offsets[profile_count];
  if (!checked_product(candidate_count, sizeof(plan7_bias_candidate),
                       &candidate_bytes) ||
      !checked_product(candidate_count, sizeof(plan7_bias_ssv_input),
                       &ssv_input_bytes)) {
    set_error(error, error_size, "F1 candidate mapping size overflow");
    return -1;
  }
  if (candidate_bytes > batch->host_bias_candidate_capacity) {
    void *replacement = realloc(batch->host_bias_candidates, candidate_bytes);
    if (replacement == nullptr) {
      set_error(error, error_size, "host F1 candidate allocation failed");
      return -1;
    }
    batch->host_bias_candidates =
      static_cast<plan7_bias_candidate *>(replacement);
    batch->host_bias_candidate_capacity = candidate_bytes;
  }
  if (grow_device_buffer(&batch->device_bias_candidates,
                         &batch->device_bias_candidate_capacity,
                         candidate_bytes,
                         "cudaMalloc(compact F1 candidates)",
                         "cudaFree(compact F1 candidates)",
                         error, error_size) != 0 ||
      (batch->f1_raw_xe_valid &&
       grow_device_buffer(&batch->device_bias_ssv_inputs,
                          &batch->device_bias_ssv_input_capacity,
                          ssv_input_bytes,
                          "cudaMalloc(compact F1 SSV inputs)",
                          "cudaFree(compact F1 SSV inputs)",
                          error, error_size) != 0))
    return -1;

  if (candidate_count != 0) {
    candidate_scatter_kernel<<<word_blocks, kThreads>>>(
      batch->device_candidate_words,
      batch->device_candidate_word_offsets,
      word_count,
      words_per_profile,
      batch->sequence_count,
      batch->f1_raw_xe_valid ? batch->device_f1_raw_xe : nullptr,
      batch->device_f1_profiles,
      batch->device_tjb,
      batch->device_offsets,
      batch->device_bias_candidates,
      batch->f1_raw_xe_valid ? batch->device_bias_ssv_inputs : nullptr);
    CUDA_TRY_COMPACT(cudaGetLastError());
    CUDA_TRY_COMPACT(cudaMemcpy(batch->host_bias_candidates,
                                batch->device_bias_candidates,
                                candidate_bytes,
                                cudaMemcpyDeviceToHost));
  }

  batch->cached_f1_profile_count = profile_count;
  batch->cached_f1_candidate_count = candidate_count;
  batch->f1_device_candidates_valid = 1;
  batch->f1_compact_ssv_inputs_valid = batch->f1_raw_xe_valid;
  if (batch->f1_raw_xe_valid)
    batch->f1_raw_xe_candidate_gather_count += candidate_count;
  ++batch->f1_device_compaction_run_count;
#undef CUDA_TRY_COMPACT
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_get_f1_candidate_view(
  const plan7_ssv_sequence_batch *batch,
  plan7_ssv_f1_candidate_view *view,
  char *error,
  size_t error_size)
{
  if (batch == nullptr || view == nullptr) {
    set_error(error, error_size, "invalid F1 candidate view request");
    return -1;
  }
  if (!batch->f1_cache_valid || !batch->f1_device_candidates_valid ||
      batch->host_candidate_offsets == nullptr) {
    set_error(error, error_size, "device-compacted F1 candidates are unavailable");
    return -1;
  }
  view->profile_count = batch->cached_f1_profile_count;
  view->candidate_count = batch->cached_f1_candidate_count;
  view->candidate_offsets = batch->host_candidate_offsets;
  view->candidates = batch->host_bias_candidates;
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_evaluate_f0_many(
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
  size_t error_size)
{
  using Clock = std::chrono::steady_clock;
  uint8_t *device_coarse_scores = nullptr;
  uint8_t *device_residue_classes = nullptr;
  uint32_t *device_coarse_words = nullptr;
  cudaEvent_t upload_start = nullptr;
  cudaEvent_t upload_stop = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  cudaError_t status = cudaSuccess;
  std::vector<uint8_t> coarse_scores;
  std::vector<uint32_t> exact_words;
  std::vector<uint32_t> coarse_words;
  std::vector<uint8_t> class_seen;
  size_t words_per_profile = 0;
  size_t word_count = 0;
  size_t word_bytes = 0;
  size_t coarse_score_count = 0;
  uint64_t residue_total = 0;
  int rc = -1;

  if (statistics == nullptr || batch == nullptr ||
      (profile_count != 0 &&
       (profile_statistics == nullptr || profiles == nullptr ||
        packed_scores == nullptr || m_mu == nullptr || m_lambda == nullptr)) ||
      profile_statistics_count != profile_count ||
      residue_classes == nullptr || residue_class_count !=
        static_cast<size_t>(batch->alphabet_size) ||
      class_count < 2 || class_count > residue_class_count) {
    set_error(error, error_size, "invalid F0 evaluator arguments");
    return -1;
  }
  memset(statistics, 0, sizeof(*statistics));
  if (profile_count != 0)
    memset(profile_statistics, 0,
           profile_count * sizeof(*profile_statistics));
  statistics->profile_count = profile_count;
  statistics->sequence_count = batch->sequence_count;
  statistics->class_count = class_count;

  try {
    class_seen.assign(class_count, 0);
    for (size_t residue = 0; residue < residue_class_count; ++residue) {
      if (residue_classes[residue] >= class_count) {
        set_error(error, error_size, "F0 residue class is out of range");
        return -1;
      }
      class_seen[residue_classes[residue]] = 1;
    }
    if (std::find(class_seen.begin(), class_seen.end(), 0) !=
        class_seen.end()) {
      set_error(error, error_size, "F0 residue partition has an empty class");
      return -1;
    }
    if (batch->sequence_count > SIZE_MAX - 31 ||
        !checked_product(profile_count,
                         (batch->sequence_count + 31) / 32,
                         &word_count) ||
        !checked_product(word_count, sizeof(uint32_t), &word_bytes)) {
      set_error(error, error_size, "F0 candidate mask size overflow");
      return -1;
    }
    words_per_profile = (batch->sequence_count + 31) / 32;
    for (size_t profile = 0; profile < profile_count; ++profile) {
      size_t profile_coarse_count;
      if (!checked_product(static_cast<size_t>(profiles[profile].model_length),
                           class_count, &profile_coarse_count) ||
          profile_coarse_count > SIZE_MAX - coarse_score_count) {
        set_error(error, error_size, "F0 score table size overflow");
        return -1;
      }
      coarse_score_count += profile_coarse_count;
    }
    coarse_scores.resize(coarse_score_count);
    exact_words.resize(word_count);
    coarse_words.resize(word_count);
  } catch (...) {
    set_error(error, error_size, "F0 evaluator host allocation failed");
    return -1;
  }

  {
    const auto start = Clock::now();
    if (sequence_batch_f1_mask_many_impl(
          batch, packed_scores, packed_score_count, profiles, profile_count,
          m_mu, m_lambda, f1, nullptr, 0, false, error, error_size) != 0)
      goto cleanup;
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      set_cuda_error(error, error_size, "F0 exact F1 synchronization", status);
      goto cleanup;
    }
    statistics->exact_generation_milliseconds =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  if (profile_count == 0 || batch->sequence_count == 0) {
    rc = 0;
    goto cleanup;
  }

  {
    const auto start = Clock::now();
    size_t output = 0;
    for (size_t profile = 0; profile < profile_count; ++profile) {
      const plan7_ssv_profile &descriptor = profiles[profile];
      for (int k = 0; k < descriptor.model_length; ++k) {
        int minimum[256];
        for (size_t residue_class = 0; residue_class < class_count;
             ++residue_class)
          minimum[residue_class] = INT_MAX;
        for (size_t residue = 0; residue < residue_class_count; ++residue) {
          const uint8_t raw = packed_scores[
            descriptor.score_offset +
            static_cast<size_t>(k) * descriptor.score_stride + residue];
          const int cost = raw < 128 ? static_cast<int>(raw)
                                     : static_cast<int>(raw) - 256;
          /* HMMER's signed-byte SSV pack guarantees cost >= -bias. This is
           * the step-size invariant that makes crossing the signed-byte wrap
           * hit the existing 255-bias overflow guard before a reject can be
           * certified. Refuse non-HMMER tables rather than weakening the
           * upper-bound proof. */
          if (cost < -static_cast<int>(descriptor.bias)) {
            set_error(error, error_size,
                      "F0 profile score violates the HMMER bias bound");
            goto cleanup;
          }
          minimum[residue_classes[residue]] =
            std::min(minimum[residue_classes[residue]], cost);
        }
        for (size_t residue_class = 0; residue_class < class_count;
             ++residue_class)
          coarse_scores[output++] = static_cast<uint8_t>(
            minimum[residue_class]);
      }
    }
    if (output != coarse_score_count) {
      set_error(error, error_size, "F0 score table size changed");
      goto cleanup;
    }
    statistics->coarse_table_build_milliseconds =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }

#define CUDA_TRY_F0(call, label)                                             \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, label, status);                       \
      goto cleanup;                                                           \
    }                                                                         \
  } while (0)

  CUDA_TRY_F0(cudaEventCreate(&upload_start), "F0 upload event create");
  CUDA_TRY_F0(cudaEventCreate(&upload_stop), "F0 upload event create");
  CUDA_TRY_F0(cudaEventCreate(&kernel_start), "F0 kernel event create");
  CUDA_TRY_F0(cudaEventCreate(&kernel_stop), "F0 kernel event create");
  CUDA_TRY_F0(cudaMalloc(&device_coarse_scores, coarse_score_count),
              "cudaMalloc(F0 scores)");
  CUDA_TRY_F0(cudaMalloc(&device_residue_classes, residue_class_count),
              "cudaMalloc(F0 residue classes)");
  CUDA_TRY_F0(cudaMalloc(&device_coarse_words, word_bytes),
              "cudaMalloc(F0 mask)");
  CUDA_TRY_F0(cudaEventRecord(upload_start), "F0 upload event record");
  CUDA_TRY_F0(cudaMemcpy(device_coarse_scores, coarse_scores.data(),
                         coarse_score_count, cudaMemcpyHostToDevice),
              "cudaMemcpy(F0 scores)");
  CUDA_TRY_F0(cudaMemcpy(device_residue_classes, residue_classes,
                         residue_class_count, cudaMemcpyHostToDevice),
              "cudaMemcpy(F0 residue classes)");
  CUDA_TRY_F0(cudaMemset(device_coarse_words, 0, word_bytes),
              "cudaMemset(F0 mask)");
  CUDA_TRY_F0(cudaEventRecord(upload_stop), "F0 upload event record");
  CUDA_TRY_F0(cudaEventRecord(kernel_start), "F0 kernel event record");
  {
    const unsigned sequence_blocks = static_cast<unsigned>(
      (batch->sequence_count + kSequencesPerBlock - 1) /
      kSequencesPerBlock);
    ssv_f0_mask_many_kernel<<<
      dim3(sequence_blocks, static_cast<unsigned>(profile_count)), kThreads>>>(
        device_coarse_scores, batch->device_f1_profiles,
        batch->sequence_count, batch->device_residues, batch->device_offsets,
        batch->device_null_scores, batch->device_tjb,
        device_residue_classes, residue_class_count, class_count,
        words_per_profile, device_coarse_words);
  }
  CUDA_TRY_F0(cudaGetLastError(), "F0 kernel launch");
  CUDA_TRY_F0(cudaEventRecord(kernel_stop), "F0 kernel event record");
  CUDA_TRY_F0(cudaEventSynchronize(kernel_stop), "F0 kernel synchronize");
  {
    float milliseconds = 0.0f;
    CUDA_TRY_F0(cudaEventElapsedTime(
                  &milliseconds, upload_start, upload_stop),
                "F0 upload event elapsed");
    statistics->coarse_upload_milliseconds = milliseconds;
    CUDA_TRY_F0(cudaEventElapsedTime(
                  &milliseconds, kernel_start, kernel_stop),
                "F0 kernel event elapsed");
    statistics->coarse_kernel_milliseconds = milliseconds;
  }
  CUDA_TRY_F0(cudaMemcpy(exact_words.data(), batch->device_candidate_words,
                         word_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy(exact F1 mask)");
  CUDA_TRY_F0(cudaMemcpy(coarse_words.data(), device_coarse_words,
                         word_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy(F0 mask)");

  {
    const auto start = Clock::now();
    for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
      if (batch->host_lengths[sequence] > UINT64_MAX - residue_total) {
        set_error(error, error_size, "F0 residue total overflow");
        goto cleanup;
      }
      residue_total += batch->host_lengths[sequence];
    }
    for (size_t profile = 0; profile < profile_count; ++profile) {
      plan7_f0_profile_statistics &row = profile_statistics[profile];
      const uint64_t model_length = static_cast<uint64_t>(
        profiles[profile].model_length);
      row.logical_pair_count = batch->sequence_count;
      if (!checked_product_u64(model_length, residue_total,
                               &row.logical_cell_count)) {
        set_error(error, error_size, "F0 logical cell count overflow");
        goto cleanup;
      }
      for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
        const size_t word = profile * words_per_profile + sequence / 32;
        const uint32_t bit = UINT32_C(1) << (sequence % 32);
        const bool exact = (exact_words[word] & bit) != 0;
        const bool coarse = (coarse_words[word] & bit) != 0;
        row.exact_candidate_count += exact;
        row.coarse_candidate_count += coarse;
        row.certified_reject_count += !coarse;
        row.false_reject_count += exact && !coarse;
        if (coarse) {
          uint64_t cells;
          if (!checked_product_u64(model_length,
                                   batch->host_lengths[sequence], &cells) ||
              !checked_add_u64(row.survivor_exact_cell_count, cells,
                               &row.survivor_exact_cell_count)) {
            set_error(error, error_size,
                      "F0 survivor exact cell count overflow");
            goto cleanup;
          }
        }
      }
      if (!checked_add_u64(statistics->logical_pair_count,
                           row.logical_pair_count,
                           &statistics->logical_pair_count) ||
          !checked_add_u64(statistics->exact_candidate_count,
                           row.exact_candidate_count,
                           &statistics->exact_candidate_count) ||
          !checked_add_u64(statistics->coarse_candidate_count,
                           row.coarse_candidate_count,
                           &statistics->coarse_candidate_count) ||
          !checked_add_u64(statistics->certified_reject_count,
                           row.certified_reject_count,
                           &statistics->certified_reject_count) ||
          !checked_add_u64(statistics->false_reject_count,
                           row.false_reject_count,
                           &statistics->false_reject_count) ||
          !checked_add_u64(statistics->logical_cell_count,
                           row.logical_cell_count,
                           &statistics->logical_cell_count) ||
          !checked_add_u64(statistics->survivor_exact_cell_count,
                           row.survivor_exact_cell_count,
                           &statistics->survivor_exact_cell_count)) {
        set_error(error, error_size, "F0 aggregate count overflow");
        goto cleanup;
      }
    }
    statistics->analysis_milliseconds =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  statistics->coarse_table_bytes = coarse_score_count;
  if (coarse_score_count > UINT64_MAX - residue_class_count ||
      coarse_score_count + residue_class_count > UINT64_MAX - word_bytes) {
    set_error(error, error_size, "F0 temporary byte count overflow");
    goto cleanup;
  }
  statistics->temporary_device_bytes =
    coarse_score_count + residue_class_count + word_bytes;
  rc = 0;

cleanup:
  if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
  if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
  if (upload_stop != nullptr) cudaEventDestroy(upload_stop);
  if (upload_start != nullptr) cudaEventDestroy(upload_start);
  if (device_coarse_words != nullptr) cudaFree(device_coarse_words);
  if (device_residue_classes != nullptr) cudaFree(device_residue_classes);
  if (device_coarse_scores != nullptr) cudaFree(device_coarse_scores);
#undef CUDA_TRY_F0
  return rc;
}

extern "C" int
plan7_ssv_sequence_batch_evaluate_seed_many(
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
  size_t error_size)
{
  using Clock = std::chrono::steady_clock;
  uint32_t *device_seed_words = nullptr;
  uint32_t *device_unsupported_words = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_stop = nullptr;
  cudaError_t status = cudaSuccess;
  std::vector<uint32_t> exact_words;
  std::vector<uint32_t> seed_words;
  std::vector<uint32_t> unsupported_words;
  size_t words_per_profile = 0;
  size_t word_count = 0;
  size_t word_bytes = 0;
  uint64_t residue_total = 0;
  int rc = -1;

  if (statistics == nullptr || batch == nullptr ||
      (profile_count != 0 &&
       (profile_statistics == nullptr || profiles == nullptr ||
        packed_scores == nullptr || m_mu == nullptr || m_lambda == nullptr)) ||
      profile_statistics_count != profile_count ||
      indexed_alphabet_size == 0 ||
      indexed_alphabet_size > static_cast<size_t>(batch->alphabet_size) ||
      (maximum_word_length != 1 && maximum_word_length != 2 &&
       maximum_word_length != 4 && maximum_word_length != 8 &&
       maximum_word_length != 16 && maximum_word_length != 32)) {
    set_error(error, error_size, "invalid mandatory-seed evaluator arguments");
    return -1;
  }
  memset(statistics, 0, sizeof(*statistics));
  if (profile_count != 0)
    memset(profile_statistics, 0,
           profile_count * sizeof(*profile_statistics));
  statistics->profile_count = profile_count;
  statistics->sequence_count = batch->sequence_count;
  statistics->maximum_word_length = maximum_word_length;

  try {
    if (batch->sequence_count > SIZE_MAX - 31 ||
        !checked_product(profile_count,
                         (batch->sequence_count + 31) / 32,
                         &word_count) ||
        !checked_product(word_count, sizeof(uint32_t), &word_bytes)) {
      set_error(error, error_size, "mandatory-seed mask size overflow");
      return -1;
    }
    words_per_profile = (batch->sequence_count + 31) / 32;
    exact_words.resize(word_count);
    seed_words.resize(word_count);
    unsupported_words.resize(word_count);
  } catch (...) {
    set_error(error, error_size, "mandatory-seed host allocation failed");
    return -1;
  }
  for (size_t profile = 0; profile < profile_count; ++profile) {
    const plan7_ssv_profile &descriptor = profiles[profile];
    for (int position = 0; position < descriptor.model_length; ++position) {
      for (int residue = 0; residue < batch->alphabet_size; ++residue) {
        const uint8_t raw = packed_scores[
          descriptor.score_offset +
          static_cast<size_t>(position) * descriptor.score_stride + residue];
        const int cost = raw < 128 ? static_cast<int>(raw)
                                   : static_cast<int>(raw) - 256;
        if (cost < -static_cast<int>(descriptor.bias)) {
          set_error(error, error_size,
                    "mandatory-seed score violates the HMMER bias bound");
          return -1;
        }
      }
    }
  }

  {
    const auto start = Clock::now();
    if (sequence_batch_f1_mask_many_impl(
          batch, packed_scores, packed_score_count, profiles, profile_count,
          m_mu, m_lambda, f1, nullptr, 0, false, error, error_size) != 0)
      goto cleanup_seed;
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      set_cuda_error(error, error_size,
                     "mandatory-seed exact F1 synchronization", status);
      goto cleanup_seed;
    }
    statistics->exact_generation_milliseconds =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  if (profile_count == 0 || batch->sequence_count == 0) {
    rc = 0;
    goto cleanup_seed;
  }

#define CUDA_TRY_SEED(call, label)                                           \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, label, status);                       \
      goto cleanup_seed;                                                      \
    }                                                                         \
  } while (0)

  CUDA_TRY_SEED(cudaEventCreate(&kernel_start),
                "mandatory-seed kernel event create");
  CUDA_TRY_SEED(cudaEventCreate(&kernel_stop),
                "mandatory-seed kernel event create");
  CUDA_TRY_SEED(cudaMalloc(&device_seed_words, word_bytes),
                "cudaMalloc(mandatory-seed mask)");
  CUDA_TRY_SEED(cudaMalloc(&device_unsupported_words, word_bytes),
                "cudaMalloc(mandatory-seed unsupported mask)");
  CUDA_TRY_SEED(cudaMemset(device_seed_words, 0, word_bytes),
                "cudaMemset(mandatory-seed mask)");
  CUDA_TRY_SEED(cudaMemset(device_unsupported_words, 0, word_bytes),
                "cudaMemset(mandatory-seed unsupported mask)");
  CUDA_TRY_SEED(cudaEventRecord(kernel_start),
                "mandatory-seed kernel event record");
  {
    const dim3 grid(
      static_cast<unsigned>((batch->sequence_count + kSequencesPerBlock - 1) /
                            kSequencesPerBlock),
      static_cast<unsigned>(profile_count));
#define LAUNCH_SEED(length)                                                   \
    ssv_seed_mask_many_kernel<length><<<grid, kThreads>>>(                    \
      batch->device_scores, batch->device_f1_profiles,                        \
      batch->sequence_count, batch->device_residues, batch->device_offsets,   \
      batch->device_null_scores, batch->device_tjb, indexed_alphabet_size,    \
      words_per_profile, device_seed_words, device_unsupported_words)
    switch (maximum_word_length) {
      case 1: LAUNCH_SEED(1); break;
      case 2: LAUNCH_SEED(2); break;
      case 4: LAUNCH_SEED(4); break;
      case 8: LAUNCH_SEED(8); break;
      case 16: LAUNCH_SEED(16); break;
      case 32: LAUNCH_SEED(32); break;
      default: break;
    }
#undef LAUNCH_SEED
  }
  CUDA_TRY_SEED(cudaGetLastError(), "mandatory-seed kernel launch");
  CUDA_TRY_SEED(cudaEventRecord(kernel_stop),
                "mandatory-seed kernel event record");
  CUDA_TRY_SEED(cudaEventSynchronize(kernel_stop),
                "mandatory-seed kernel synchronize");
  {
    float milliseconds = 0.0f;
    CUDA_TRY_SEED(cudaEventElapsedTime(&milliseconds, kernel_start, kernel_stop),
                  "mandatory-seed kernel event elapsed");
    statistics->seed_kernel_milliseconds = milliseconds;
  }
  CUDA_TRY_SEED(cudaMemcpy(exact_words.data(), batch->device_candidate_words,
                           word_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(exact F1 mask for mandatory-seed)");
  CUDA_TRY_SEED(cudaMemcpy(seed_words.data(), device_seed_words,
                           word_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(mandatory-seed mask)");
  CUDA_TRY_SEED(cudaMemcpy(unsupported_words.data(), device_unsupported_words,
                           word_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(mandatory-seed unsupported mask)");

  {
    const auto start = Clock::now();
    for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
      if (batch->host_lengths[sequence] > UINT64_MAX - residue_total) {
        set_error(error, error_size, "mandatory-seed residue total overflow");
        goto cleanup_seed;
      }
      residue_total += batch->host_lengths[sequence];
    }
    for (size_t profile = 0; profile < profile_count; ++profile) {
      plan7_seed_profile_statistics &row = profile_statistics[profile];
      const uint64_t model_length = static_cast<uint64_t>(
        profiles[profile].model_length);
      row.logical_pair_count = batch->sequence_count;
      if (!checked_product_u64(model_length, residue_total,
                               &row.logical_cell_count)) {
        set_error(error, error_size,
                  "mandatory-seed logical cell count overflow");
        goto cleanup_seed;
      }
      for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
        const size_t word = profile * words_per_profile + sequence / 32;
        const uint32_t bit = UINT32_C(1) << (sequence % 32);
        const bool exact = (exact_words[word] & bit) != 0;
        const bool seed = (seed_words[word] & bit) != 0;
        const bool unsupported = (unsupported_words[word] & bit) != 0;
        row.exact_candidate_count += exact;
        row.seed_candidate_count += seed;
        row.certified_reject_count += !seed;
        row.false_reject_count += exact && !seed;
        row.unsupported_pair_count += unsupported;
        if (seed) {
          uint64_t cells;
          if (!checked_product_u64(model_length,
                                   batch->host_lengths[sequence], &cells) ||
              !checked_add_u64(row.survivor_exact_cell_count, cells,
                               &row.survivor_exact_cell_count)) {
            set_error(error, error_size,
                      "mandatory-seed survivor cell count overflow");
            goto cleanup_seed;
          }
        }
      }
#define ADD_SEED_FIELD(field)                                                 \
      if (!checked_add_u64(statistics->field, row.field, &statistics->field)) {\
        set_error(error, error_size,                                          \
                  "mandatory-seed aggregate count overflow");                \
        goto cleanup_seed;                                                    \
      }
      ADD_SEED_FIELD(logical_pair_count);
      ADD_SEED_FIELD(exact_candidate_count);
      ADD_SEED_FIELD(seed_candidate_count);
      ADD_SEED_FIELD(certified_reject_count);
      ADD_SEED_FIELD(false_reject_count);
      ADD_SEED_FIELD(unsupported_pair_count);
      ADD_SEED_FIELD(logical_cell_count);
      ADD_SEED_FIELD(survivor_exact_cell_count);
#undef ADD_SEED_FIELD
    }
    statistics->analysis_milliseconds =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  }
  if (word_bytes > UINT64_MAX / 2) {
    set_error(error, error_size, "mandatory-seed temporary byte overflow");
    goto cleanup_seed;
  }
  statistics->temporary_device_bytes = static_cast<uint64_t>(word_bytes) * 2;
  rc = 0;

cleanup_seed:
  if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
  if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
  if (device_unsupported_words != nullptr) cudaFree(device_unsupported_words);
  if (device_seed_words != nullptr) cudaFree(device_seed_words);
#undef CUDA_TRY_SEED
  return rc;
}

extern "C" int
plan7_ssv_sequence_batch_bias_candidates_many(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  plan7_bias_result *results,
  size_t result_count,
  char *error,
  size_t error_size)
{
  cudaError_t status;
  size_t profile_bytes;
  size_t candidate_bytes;
  size_t ssv_input_bytes;
  size_t result_bytes;
  size_t length_term_bytes;
  int current_device;
  int maximum_grid_x;
  bool use_cached_device_candidates;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  if (!batch->f1_cache_valid ||
      profile_count != batch->cached_f1_profile_count) {
    set_error(error, error_size,
              "bias candidates do not match the cached fused F1 pass");
    return -1;
  }
  if ((profile_count != 0 &&
       (bias_profiles == nullptr || candidate_offsets == nullptr)) ||
      (candidate_count != 0 &&
       (candidate_indices == nullptr || results == nullptr)) ||
      result_count < candidate_count) {
    set_error(error, error_size, "invalid bias candidate buffers");
    return -1;
  }
  if (profile_count == 0) {
    if (candidate_count != 0) {
      set_error(error, error_size, "bias candidates have no profiles");
      return -1;
    }
    return 0;
  }
  if (candidate_offsets[0] != 0 ||
      candidate_offsets[profile_count] != candidate_count) {
    set_error(error, error_size, "invalid bias candidate offsets");
    return -1;
  }

  use_cached_device_candidates = batch->f1_device_candidates_valid &&
    batch->cached_f1_profile_count == profile_count &&
    batch->cached_f1_candidate_count == candidate_count &&
    batch->host_candidate_offsets != nullptr &&
    batch->host_bias_candidates != nullptr;

  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (candidate_offsets[profile] > candidate_offsets[profile + 1]) {
      set_error(error, error_size, "invalid bias candidate offsets");
      return -1;
    }
    if (use_cached_device_candidates &&
        (batch->host_candidate_offsets[profile] != candidate_offsets[profile] ||
         batch->host_candidate_offsets[profile + 1] !=
           candidate_offsets[profile + 1]))
      use_cached_device_candidates = false;
    for (size_t candidate = candidate_offsets[profile];
         candidate < candidate_offsets[profile + 1]; ++candidate) {
      if (candidate_indices[candidate] >= batch->sequence_count) {
        set_error(error, error_size,
                  "bias candidate sequence index is out of range");
        return -1;
      }
      if (use_cached_device_candidates &&
          (batch->host_bias_candidates[candidate].profile_index != profile ||
           batch->host_bias_candidates[candidate].sequence_index !=
             candidate_indices[candidate]))
        use_cached_device_candidates = false;
    }
  }
  if (candidate_count == 0) return 0;

  if (batch->alphabet_size != 29 ||
      !batch->host_float_environment_valid ||
      plan7_bias_environment_attested(nullptr, 0) != 1) {
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
      results[candidate] = {
        candidate_indices[candidate], NAN, 0, PLAN7_SSV_ENORESULT,
        PLAN7_BIAS_CPU_REQUIRED
      };
    }
    return 0;
  }

  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (!validate_bias_profile(&bias_profiles[profile]) ||
        bias_profiles[profile].scale !=
          batch->host_f1_profiles[profile].profile.scale ||
        bias_profiles[profile].cutoff_mode !=
          batch->host_f1_profiles[profile].cutoff_mode ||
        (bias_profiles[profile].cutoff_mode == PLAN7_BIAS_CUTOFF_SCORE &&
         bias_profiles[profile].cutoff_bit_score !=
           batch->host_f1_profiles[profile].cutoff_bit_score)) {
      set_error(error, error_size,
                "bias profile differs from the cached fused F1 profile");
      return -1;
    }
  }
  if (!checked_product(profile_count, sizeof(plan7_bias_profile),
                       &profile_bytes) ||
      !checked_product(candidate_count, sizeof(plan7_bias_candidate),
                       &candidate_bytes) ||
      !checked_product(candidate_count, sizeof(plan7_bias_ssv_input),
                       &ssv_input_bytes) ||
      !checked_product(candidate_count, sizeof(plan7_bias_result),
                       &result_bytes) ||
      !checked_product(batch->sequence_count, sizeof(float),
                       &length_term_bytes)) {
    set_error(error, error_size, "bias candidate batch size overflow");
    return -1;
  }

  if (candidate_bytes > batch->device_bias_candidate_capacity)
    use_cached_device_candidates = false;
  if (!use_cached_device_candidates)
    invalidate_f1_device_candidates(batch);
  if (!use_cached_device_candidates &&
      candidate_bytes > batch->host_bias_candidate_capacity) {
    void *replacement = realloc(batch->host_bias_candidates, candidate_bytes);
    if (replacement == nullptr) {
      set_error(error, error_size, "host bias candidate allocation failed");
      return -1;
    }
    batch->host_bias_candidates =
      static_cast<plan7_bias_candidate *>(replacement);
    batch->host_bias_candidate_capacity = candidate_bytes;
  }
  if (!use_cached_device_candidates) {
    for (size_t profile = 0; profile < profile_count; ++profile) {
      for (size_t candidate = candidate_offsets[profile];
           candidate < candidate_offsets[profile + 1]; ++candidate) {
        batch->host_bias_candidates[candidate] = {
          static_cast<uint32_t>(profile), candidate_indices[candidate]
        };
      }
    }
  }

  status = cudaDeviceGetAttribute(
    &maximum_grid_x, cudaDevAttrMaxGridDimX, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid x)", status);
    return -1;
  }
  if (candidate_count > static_cast<size_t>(maximum_grid_x)) {
    set_error(error, error_size, "sparse SSV candidate grid is too large");
    return -1;
  }

  if (grow_device_buffer(&batch->device_bias_profiles,
                         &batch->device_bias_profile_capacity,
                         profile_bytes,
                         "cudaMalloc(bias profiles)",
                         "cudaFree(bias profiles)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_candidates,
                         &batch->device_bias_candidate_capacity,
                         candidate_bytes,
                         "cudaMalloc(bias candidates)",
                         "cudaFree(bias candidates)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_ssv_inputs,
                         &batch->device_bias_ssv_input_capacity,
                         ssv_input_bytes,
                         "cudaMalloc(sparse SSV results)",
                         "cudaFree(sparse SSV results)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_results,
                         &batch->device_bias_result_capacity,
                         result_bytes,
                         "cudaMalloc(bias results)",
                         "cudaFree(bias results)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_logp,
                         &batch->device_bias_logp_capacity,
                         length_term_bytes,
                         "cudaMalloc(bias length logp)",
                         "cudaFree(bias length logp)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_log1mp,
                         &batch->device_bias_log1mp_capacity,
                         length_term_bytes,
                         "cudaMalloc(bias length log1mp)",
                         "cudaFree(bias length log1mp)",
                         error,
                         error_size) != 0)
    return -1;

#define CUDA_TRY_BIAS(call)                                                   \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  CUDA_TRY_BIAS(cudaMemcpy(batch->device_bias_profiles,
                           bias_profiles,
                           profile_bytes,
                           cudaMemcpyHostToDevice));
  if (use_cached_device_candidates) {
    ++batch->f1_candidate_upload_avoided_count;
  } else {
    CUDA_TRY_BIAS(cudaMemcpy(batch->device_bias_candidates,
                             batch->host_bias_candidates,
                             candidate_bytes,
                             cudaMemcpyHostToDevice));
    ++batch->f1_candidate_upload_count;
  }
  if (!batch->bias_length_terms_device_valid) {
    CUDA_TRY_BIAS(cudaMemcpy(batch->device_bias_logp,
                             batch->host_bias_logp,
                             length_term_bytes,
                             cudaMemcpyHostToDevice));
    CUDA_TRY_BIAS(cudaMemcpy(batch->device_bias_log1mp,
                             batch->host_bias_log1mp,
                             length_term_bytes,
                             cudaMemcpyHostToDevice));
    batch->bias_length_terms_device_valid = 1;
  }

  if (use_cached_device_candidates &&
      batch->f1_compact_ssv_inputs_valid) {
    batch->f1_candidate_ssv_replay_avoided_count += candidate_count;
  } else {
    ssv_bias_candidates_kernel<<<static_cast<unsigned>(candidate_count),
                                 kThreads>>>(
      batch->device_scores,
      batch->device_f1_profiles,
      batch->sequence_count,
      batch->device_residues,
      batch->device_offsets,
      batch->device_tjb,
      batch->device_bias_candidates,
      batch->device_bias_ssv_inputs);
    CUDA_TRY_BIAS(cudaGetLastError());
    batch->f1_candidate_ssv_replay_count += candidate_count;
  }
  if (plan7_bias_filter_candidates_device(
        batch->device_residues,
        batch->device_offsets,
        batch->device_bias_logp,
        batch->device_bias_log1mp,
        batch->device_bias_profiles,
        batch->device_bias_candidates,
        batch->device_bias_ssv_inputs,
        candidate_count,
        batch->device_bias_results,
        error,
        error_size) != 0)
    return -1;
  CUDA_TRY_BIAS(cudaMemcpy(results,
                           batch->device_bias_results,
                           result_bytes,
                           cudaMemcpyDeviceToHost));
#undef CUDA_TRY_BIAS
  return 0;
}

static int
sequence_batch_postfilter_candidates_many_impl(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const plan7_viterbi_database *viterbi_database,
  plan7_postfilter_result *results,
  size_t result_count,
  uint16_t *reason_facts,
  size_t reason_count,
  plan7_postfilter_reason_statistics *reason_statistics,
  bool fixed_bias_filter,
  char *error,
  size_t error_size)
{
  cudaError_t status;
  size_t profile_bytes;
  size_t candidate_bytes;
  size_t ssv_input_bytes;
  size_t length_term_bytes;
  int current_device;
  int maximum_grid_x;
  bool use_cached_device_candidates;

  if (reason_statistics != nullptr) *reason_statistics = {};

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  if (!batch->f1_cache_valid ||
      profile_count != batch->cached_f1_profile_count) {
    set_error(error, error_size,
              "post-filter candidates do not match the cached fused F1 pass");
    return -1;
  }
  if ((profile_count != 0 &&
       (bias_profiles == nullptr || candidate_offsets == nullptr)) ||
      (candidate_count != 0 &&
       (candidate_indices == nullptr || results == nullptr ||
        (reason_facts != nullptr && reason_count < candidate_count))) ||
      ((reason_facts == nullptr) != (reason_statistics == nullptr)) ||
      result_count < candidate_count || viterbi_database == nullptr ||
      (profile_count != 0 && source_profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid post-filter candidate buffers");
    return -1;
  }
  if (profile_count == 0) {
    if (candidate_count != 0 ||
        plan7_viterbi_database_profile_count(viterbi_database) != 0) {
      set_error(error, error_size, "post-filter candidates have no profiles");
      return -1;
    }
    return 0;
  }
  if (candidate_offsets[0] != 0 ||
      candidate_offsets[profile_count] != candidate_count) {
    set_error(error, error_size, "invalid post-filter candidate offsets");
    return -1;
  }
  use_cached_device_candidates = batch->f1_device_candidates_valid &&
    batch->cached_f1_profile_count == profile_count &&
    batch->cached_f1_candidate_count == candidate_count &&
    batch->host_candidate_offsets != nullptr &&
    batch->host_bias_candidates != nullptr;
  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (candidate_offsets[profile] > candidate_offsets[profile + 1]) {
      set_error(error, error_size, "invalid post-filter candidate offsets");
      return -1;
    }
    if (use_cached_device_candidates &&
        (batch->host_candidate_offsets[profile] != candidate_offsets[profile] ||
         batch->host_candidate_offsets[profile + 1] !=
           candidate_offsets[profile + 1]))
      use_cached_device_candidates = false;
    for (size_t candidate = candidate_offsets[profile];
         candidate < candidate_offsets[profile + 1]; ++candidate) {
      if (candidate_indices[candidate] >= batch->sequence_count) {
        set_error(error, error_size,
                  "post-filter candidate sequence index is out of range");
        return -1;
      }
      if (use_cached_device_candidates &&
          (batch->host_bias_candidates[candidate].profile_index != profile ||
           batch->host_bias_candidates[candidate].sequence_index !=
             candidate_indices[candidate]))
        use_cached_device_candidates = false;
    }
  }
  if (candidate_count == 0) return 0;
  if (batch->alphabet_size != 29 ||
      !batch->host_float_environment_valid ||
      plan7_bias_environment_attested(nullptr, 0) != 1) {
    if (reason_statistics != nullptr) {
      *reason_statistics = {};
      reason_statistics->candidate_count = candidate_count;
    }
    for (size_t profile = 0; profile < profile_count; ++profile) {
      for (size_t candidate = candidate_offsets[profile];
           candidate < candidate_offsets[profile + 1]; ++candidate) {
        results[candidate] = {
          candidate_indices[candidate], NAN, 0, PLAN7_SSV_ENORESULT,
          PLAN7_BIAS_CPU_REQUIRED, NAN
        };
        if (reason_facts != nullptr)
          reason_facts[candidate] =
              PLAN7_POSTFILTER_REASON_CONTRACT_FALLBACK |
              PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED;
      }
    }
    return 0;
  }
  if (plan7_viterbi_database_matches_ssv(
        viterbi_database, batch->host_f1_scores,
        batch->host_f1_score_count, source_profile_pointers,
        batch->host_f1_profiles, profile_count,
        error, error_size) != 0)
    return -1;
  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (!validate_bias_profile(&bias_profiles[profile]) ||
        bias_profiles[profile].scale !=
          batch->host_f1_profiles[profile].profile.scale ||
        bias_profiles[profile].cutoff_mode !=
          batch->host_f1_profiles[profile].cutoff_mode ||
        (bias_profiles[profile].cutoff_mode == PLAN7_BIAS_CUTOFF_SCORE &&
         bias_profiles[profile].cutoff_bit_score !=
           batch->host_f1_profiles[profile].cutoff_bit_score)) {
      set_error(error, error_size,
                "bias profile differs from the cached fused F1 profile");
      return -1;
    }
  }
  if (!checked_product(profile_count, sizeof(plan7_bias_profile),
                       &profile_bytes) ||
      !checked_product(candidate_count, sizeof(plan7_bias_candidate),
                       &candidate_bytes) ||
      !checked_product(candidate_count, sizeof(plan7_bias_ssv_input),
                       &ssv_input_bytes) ||
      !checked_product(batch->sequence_count, sizeof(float),
                       &length_term_bytes)) {
    set_error(error, error_size, "post-filter candidate batch size overflow");
    return -1;
  }
  if (candidate_bytes > batch->device_bias_candidate_capacity)
    use_cached_device_candidates = false;
  if (!use_cached_device_candidates)
    invalidate_f1_device_candidates(batch);
  if (!use_cached_device_candidates &&
      candidate_bytes > batch->host_bias_candidate_capacity) {
    void *replacement = realloc(batch->host_bias_candidates, candidate_bytes);
    if (replacement == nullptr) {
      set_error(error, error_size,
                "host post-filter candidate allocation failed");
      return -1;
    }
    batch->host_bias_candidates =
      static_cast<plan7_bias_candidate *>(replacement);
    batch->host_bias_candidate_capacity = candidate_bytes;
  }
  if (!use_cached_device_candidates) {
    for (size_t profile = 0; profile < profile_count; ++profile) {
      for (size_t candidate = candidate_offsets[profile];
           candidate < candidate_offsets[profile + 1]; ++candidate) {
        batch->host_bias_candidates[candidate] = {
          static_cast<uint32_t>(profile), candidate_indices[candidate]
        };
      }
    }
  }

  status = cudaDeviceGetAttribute(
    &maximum_grid_x, cudaDevAttrMaxGridDimX, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid x)", status);
    return -1;
  }
  if (candidate_count > static_cast<size_t>(maximum_grid_x)) {
    set_error(error, error_size, "sparse SSV candidate grid is too large");
    return -1;
  }

  if (grow_device_buffer(&batch->device_bias_profiles,
                         &batch->device_bias_profile_capacity,
                         profile_bytes,
                         "cudaMalloc(post-filter bias profiles)",
                         "cudaFree(post-filter bias profiles)",
                         error, error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_candidates,
                         &batch->device_bias_candidate_capacity,
                         candidate_bytes,
                         "cudaMalloc(post-filter candidates)",
                         "cudaFree(post-filter candidates)",
                         error, error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_ssv_inputs,
                         &batch->device_bias_ssv_input_capacity,
                         ssv_input_bytes,
                         "cudaMalloc(post-filter sparse SSV results)",
                         "cudaFree(post-filter sparse SSV results)",
                         error, error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_logp,
                         &batch->device_bias_logp_capacity,
                         length_term_bytes,
                         "cudaMalloc(post-filter bias length logp)",
                         "cudaFree(post-filter bias length logp)",
                         error, error_size) != 0 ||
      grow_device_buffer(&batch->device_bias_log1mp,
                         &batch->device_bias_log1mp_capacity,
                         length_term_bytes,
                         "cudaMalloc(post-filter bias length log1mp)",
                         "cudaFree(post-filter bias length log1mp)",
                         error, error_size) != 0)
    return -1;

#define CUDA_TRY_POSTFILTER(call)                                             \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  CUDA_TRY_POSTFILTER(cudaMemcpy(batch->device_bias_profiles,
                                 bias_profiles, profile_bytes,
                                 cudaMemcpyHostToDevice));
  if (use_cached_device_candidates) {
    ++batch->f1_candidate_upload_avoided_count;
  } else {
    CUDA_TRY_POSTFILTER(cudaMemcpy(batch->device_bias_candidates,
                                   batch->host_bias_candidates,
                                   candidate_bytes, cudaMemcpyHostToDevice));
    ++batch->f1_candidate_upload_count;
  }
  if (!batch->bias_length_terms_device_valid) {
    CUDA_TRY_POSTFILTER(cudaMemcpy(batch->device_bias_logp,
                                   batch->host_bias_logp,
                                   length_term_bytes,
                                   cudaMemcpyHostToDevice));
    CUDA_TRY_POSTFILTER(cudaMemcpy(batch->device_bias_log1mp,
                                   batch->host_bias_log1mp,
                                   length_term_bytes,
                                   cudaMemcpyHostToDevice));
    batch->bias_length_terms_device_valid = 1;
  }
  if (use_cached_device_candidates &&
      batch->f1_compact_ssv_inputs_valid) {
    batch->f1_candidate_ssv_replay_avoided_count += candidate_count;
  } else {
    ssv_bias_candidates_kernel<<<static_cast<unsigned>(candidate_count),
                                 kThreads>>>(
      batch->device_scores,
      batch->device_f1_profiles,
      batch->sequence_count,
      batch->device_residues,
      batch->device_offsets,
      batch->device_tjb,
      batch->device_bias_candidates,
      batch->device_bias_ssv_inputs);
    CUDA_TRY_POSTFILTER(cudaGetLastError());
    batch->f1_candidate_ssv_replay_count += candidate_count;
  }
  if (batch->postfilter_workspace == nullptr &&
      plan7_postfilter_workspace_create(
        &batch->postfilter_workspace, error, error_size) != 0)
    return -1;
  const int postfilter_status = reason_facts == nullptr
      ? (fixed_bias_filter
        ? plan7_postfilter_candidates_device_with_workspace_fixed_bias_policy(
          batch->postfilter_workspace, viterbi_database,
          batch->device_residues, batch->device_offsets, batch->host_lengths,
          batch->sequence_count, batch->device_null_scores,
          batch->device_scores, batch->device_f1_profiles, batch->device_tjb,
          batch->device_bias_logp, batch->device_bias_log1mp,
          batch->device_bias_profiles, batch->device_bias_candidates,
          batch->host_bias_candidates, batch->device_bias_ssv_inputs,
          candidate_count, results, batch->execution_policy_mode,
          error, error_size)
        : plan7_postfilter_candidates_device_with_workspace_policy(
          batch->postfilter_workspace, viterbi_database,
          batch->device_residues, batch->device_offsets, batch->host_lengths,
          batch->sequence_count, batch->device_null_scores,
          batch->device_scores, batch->device_f1_profiles, batch->device_tjb,
          batch->device_bias_logp, batch->device_bias_log1mp,
          batch->device_bias_profiles, batch->device_bias_candidates,
          batch->host_bias_candidates, batch->device_bias_ssv_inputs,
          candidate_count, results, batch->execution_policy_mode,
          error, error_size))
      : (fixed_bias_filter
        ? plan7_postfilter_candidates_device_with_workspace_fixed_bias_reason_facts_policy(
          batch->postfilter_workspace, viterbi_database,
          batch->device_residues, batch->device_offsets, batch->host_lengths,
          batch->sequence_count, batch->device_null_scores,
          batch->device_scores, batch->device_f1_profiles, batch->device_tjb,
          batch->device_bias_logp, batch->device_bias_log1mp,
          batch->device_bias_profiles, batch->device_bias_candidates,
          batch->host_bias_candidates, batch->device_bias_ssv_inputs,
          candidate_count, results, reason_facts, reason_statistics,
          batch->execution_policy_mode, error, error_size)
        : plan7_postfilter_candidates_device_with_workspace_reason_facts_policy(
          batch->postfilter_workspace, viterbi_database,
          batch->device_residues, batch->device_offsets, batch->host_lengths,
          batch->sequence_count, batch->device_null_scores,
          batch->device_scores, batch->device_f1_profiles, batch->device_tjb,
          batch->device_bias_logp, batch->device_bias_log1mp,
          batch->device_bias_profiles, batch->device_bias_candidates,
          batch->host_bias_candidates, batch->device_bias_ssv_inputs,
          candidate_count, results, reason_facts, reason_statistics,
          batch->execution_policy_mode, error, error_size));
  if (postfilter_status != 0)
    return -1;
#undef CUDA_TRY_POSTFILTER
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_postfilter_candidates_many(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const plan7_viterbi_database *viterbi_database,
  plan7_postfilter_result *results,
  size_t result_count,
  char *error,
  size_t error_size)
{
  return sequence_batch_postfilter_candidates_many_impl(
      batch, bias_profiles, profile_count, candidate_offsets,
      candidate_indices, candidate_count, source_profile_pointers,
      viterbi_database, results, result_count, nullptr, 0, nullptr,
      false, error, error_size);
}

extern "C" int
plan7_ssv_sequence_batch_postfilter_candidates_many_reason_facts(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const plan7_viterbi_database *viterbi_database,
  plan7_postfilter_result *results,
  size_t result_count,
  uint16_t *reason_facts,
  size_t reason_count,
  plan7_postfilter_reason_statistics *reason_statistics,
  char *error,
  size_t error_size)
{
  if (candidate_count != 0 &&
      (reason_facts == nullptr || reason_count < candidate_count ||
       reason_statistics == nullptr)) {
    set_error(error, error_size, "post-filter reason output is too small");
    return -1;
  }
  return sequence_batch_postfilter_candidates_many_impl(
      batch, bias_profiles, profile_count, candidate_offsets,
      candidate_indices, candidate_count, source_profile_pointers,
      viterbi_database, results, result_count, reason_facts, reason_count,
      reason_statistics, false,
      error, error_size);
}

extern "C" int
plan7_ssv_sequence_batch_postfilter_candidates_many_fixed_bias(
  plan7_ssv_sequence_batch *batch,
  const plan7_bias_profile *bias_profiles,
  size_t profile_count,
  const size_t *candidate_offsets,
  const uint32_t *candidate_indices,
  size_t candidate_count,
  const uintptr_t *source_profile_pointers,
  const plan7_viterbi_database *viterbi_database,
  plan7_postfilter_result *results,
  size_t result_count,
  uint16_t *reason_facts,
  size_t reason_count,
  plan7_postfilter_reason_statistics *reason_statistics,
  char *error,
  size_t error_size)
{
  if ((reason_facts == nullptr) != (reason_statistics == nullptr) ||
      (reason_facts != nullptr && reason_count < candidate_count)) {
    set_error(error, error_size, "fixed-bias reason output is invalid");
    return -1;
  }
  return sequence_batch_postfilter_candidates_many_impl(
      batch, bias_profiles, profile_count, candidate_offsets,
      candidate_indices, candidate_count, source_profile_pointers,
      viterbi_database, results, result_count, reason_facts, reason_count,
      reason_statistics, true, error, error_size);
}

extern "C" int plan7_ssv_sequence_batch_compact_postfilter_f2(
    plan7_ssv_sequence_batch *batch, const plan7_ssv_profile *profiles,
    const float *m_mu, const float *m_lambda, const float *v_mu,
    const float *v_lambda, size_t profile_count, double f2,
    plan7_postfilter_f2_resident_view *view,
    char *error, size_t error_size) {
  if (batch == nullptr || view == nullptr) {
    set_error(error, error_size, "invalid sequence-batch F2 compaction");
    return -1;
  }
  if (batch->postfilter_workspace == nullptr) {
    set_error(error, error_size, "post-filter workspace is unavailable");
    return -1;
  }
  return plan7_postfilter_workspace_compact_f2(
      batch->postfilter_workspace, batch->generation_id, profiles,
      m_mu, m_lambda, v_mu, v_lambda, profile_count, f2,
      batch->host_float_environment_valid, view, error, error_size);
}

extern "C" int
plan7_ssv_sequence_batch_f1_candidates_many(
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
  size_t error_size)
{
  size_t cell_count;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  if (profile_count == 0) return 0;
  if (scales == nullptr || m_mu == nullptr || m_lambda == nullptr ||
      candidate_counts == nullptr) {
    set_error(error, error_size, "F1 profile buffers are null");
    return -1;
  }
  if ((candidate_offsets == nullptr) != (candidate_indices == nullptr)) {
    set_error(error, error_size, "F1 candidate output buffers differ");
    return -1;
  }
  if (!checked_product(profile_count, batch->sequence_count, &cell_count)) {
    set_error(error, error_size, "F1 candidate batch size overflow");
    return -1;
  }
  if (cell_count != 0 &&
      (profile_major_results == nullptr || result_count < cell_count ||
       batch->host_lengths == nullptr || batch->host_null_scores == nullptr)) {
    set_error(error, error_size, "F1 candidate buffer is too short");
    return -1;
  }
  if (candidate_indices != nullptr) {
    size_t expected_offset = 0;
    for (size_t profile = 0; profile < profile_count; ++profile) {
      if (candidate_offsets[profile] != expected_offset ||
          expected_offset > candidate_index_count ||
          candidate_counts[profile] > candidate_index_count - expected_offset) {
        set_error(error, error_size, "invalid F1 candidate offsets");
        return -1;
      }
      expected_offset += candidate_counts[profile];
    }
    if (expected_offset != candidate_index_count) {
      set_error(error, error_size, "invalid F1 candidate count");
      return -1;
    }
  }

  for (size_t profile = 0; profile < profile_count; ++profile) {
    float cutoff = NAN;
    const int cutoff_mode =
      derive_f1_cutoff(m_mu[profile], m_lambda[profile], f1, &cutoff);
    size_t candidate_count = 0;
    const size_t row = profile * batch->sequence_count;
    const size_t expected_candidate_count = candidate_indices != nullptr
                                              ? candidate_counts[profile]
                                              : 0;

    for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
      const plan7_ssv_result result = profile_major_results[row + sequence];
      int action;
      if (cutoff_mode == PLAN7_F1_CUTOFF_INVALID) {
        action = plan7_ssv_f1_decision(
          result.status, result.numerator, batch->host_lengths[sequence],
          scales[profile], m_mu[profile], m_lambda[profile], f1, nullptr);
      } else {
        action = cutoff_decision_with_null_score(
          result.status, result.numerator, batch->host_null_scores[sequence],
          scales[profile], cutoff_mode, cutoff);
      }
      if (action == PLAN7_F1_CPU_REQUIRED) {
        if (candidate_indices != nullptr) {
          if (candidate_count >= expected_candidate_count) {
            set_error(error, error_size, "F1 candidate count changed");
            return -1;
          }
          candidate_indices[candidate_offsets[profile] + candidate_count] =
            static_cast<uint32_t>(sequence);
        }
        ++candidate_count;
      }
    }
    if (candidate_indices != nullptr &&
        candidate_count != expected_candidate_count) {
      set_error(error, error_size, "F1 candidate count changed");
      return -1;
    }
    candidate_counts[profile] = candidate_count;
  }
  return 0;
}

extern "C" int
plan7_ssv_filter_cuda(const uint8_t *striped_scores,
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
                      size_t error_size)
{
  plan7_ssv_sequence_batch *batch = nullptr;
  int rc;
  int destroy_rc;

  if (offset_count != sequence_count + 1) {
    set_error(error, error_size, "invalid sequence offsets");
    return -1;
  }
  rc = plan7_ssv_sequence_batch_create(
    residues, residue_count, offsets, offset_count, alphabet_size, &batch,
    error, error_size);
  if (rc == 0)
    rc = plan7_ssv_sequence_batch_filter(
      batch, striped_scores, striped_score_count, score_stride, model_length,
      alphabet_size, tbm, tec, base, bias, scale, results, result_count, error,
      error_size);
  destroy_rc = plan7_ssv_sequence_batch_destroy(
    &batch, rc == 0 ? error : nullptr, rc == 0 ? error_size : 0);
  if (rc == 0 && destroy_rc != 0) rc = -1;
  return rc;
}
