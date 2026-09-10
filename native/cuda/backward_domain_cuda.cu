#include "backward_domain_cuda.h"

#include <cuda_runtime.h>

extern "C" {
#include <easel.h>
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
              "Backward/domain CUDA requires binary32 float");
static_assert(sizeof(plan7_backward_domain_candidate) == 8,
              "Backward candidate ABI changed");
static_assert(sizeof(plan7_backward_domain_result) ==
                  PLAN7_BACKWARD_DOMAIN_RECORD_SIZE,
              "Backward result ABI changed");
static_assert(sizeof(plan7_domain_posterior) ==
                  PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE,
              "Domain posterior ABI changed");
static_assert(sizeof(plan7_simple_region) == PLAN7_BACKWARD_DOMAIN_REGION_SIZE,
              "Simple region ABI changed");
static_assert(sizeof(plan7_backward_domain_resident_region) ==
                  PLAN7_BACKWARD_DOMAIN_RESIDENT_REGION_SIZE,
              "Resident Backward region ABI changed");
static_assert(sizeof(plan7_forward_provenance) == 72,
              "Forward provenance ABI changed");
static_assert(sizeof(plan7_backward_domain_provenance) == 112,
              "Backward/domain provenance ABI changed");
static_assert(sizeof(plan7_forward_device_profile) == 32,
              "Forward device profile ABI changed");
static_assert(p7X_NXCELLS == 6 && p7O_NTRANS == 8,
              "HMMER Forward/Backward layout changed");

namespace {

constexpr int kThreads = 256;
constexpr int kSubwarp = 4;
constexpr int kCandidatesPerBlock = kThreads / 32;
constexpr uint64_t kDpWorkspaceByteLimit = UINT64_C(256) << 20;
constexpr uint64_t kBackwardSpecialByteLimit = UINT64_C(256) << 20;
constexpr uint64_t kForwardSpecialByteLimit = UINT64_C(384) << 20;
constexpr uint64_t kPosteriorWorkspaceByteLimit = UINT64_C(384) << 20;
constexpr uint64_t kSimpleRegionOutputByteLimit = UINT64_C(64) << 20;
constexpr uint64_t kResultOutputByteLimit = UINT64_C(256) << 20;
constexpr uint64_t kMaximumTargetLength = 100000;
constexpr uint64_t kMaximumModelLength = 100000;
constexpr uint64_t kMaximumRowWorkCells =
    PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS;
constexpr uint64_t kMaximumRunWorkCells =
    PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS;

union FloatBits {
  float value;
  uint32_t bits;
};

constexpr uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr uint64_t kHashPrime = UINT64_C(1099511628211);

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

bool matches_forward_provenance(
    const plan7_forward_provenance &sealed,
    const plan7_forward_device_view &profiles,
    const plan7_ssv_sequence_batch_view &sequences,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const uint64_t *offsets,
    const float *specials, size_t special_count) {
  if (sealed.database_generation != profiles.generation_id ||
      sealed.batch_generation != sequences.generation_id ||
      sealed.pass_count != candidate_count ||
      sealed.special_count != special_count)
    return false;
  uint64_t row_hash = hash_u64(kHashOffset, UINT64_C(0x524f5753));
  uint64_t special_hash = hash_u64(kHashOffset, UINT64_C(0x584d5821));
  for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
    if (offsets[candidate] > offsets[candidate + 1] ||
        offsets[candidate + 1] > special_count)
      return false;
    row_hash = hash_u32(row_hash, candidates[candidate].profile_index);
    row_hash = hash_u32(row_hash, candidates[candidate].sequence_index);
    const uint64_t begin = offsets[candidate];
    const uint64_t end = offsets[candidate + 1];
    special_hash = hash_u64(special_hash, end - begin);
    for (uint64_t cell = begin; cell < end; ++cell) {
      FloatBits bits{};
      bits.value = specials[static_cast<size_t>(cell)];
      special_hash = hash_u32(special_hash, bits.bits);
    }
  }
  row_hash = hash_u64(row_hash, candidate_count);
  special_hash = hash_u64(special_hash, special_count);
  return row_hash == sealed.row_hash &&
         special_hash == sealed.special_hash;
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

bool valid_thresholds(float rt1, float rt2, float rt3,
                      float guard_band) {
  return std::isfinite(rt1) && std::isfinite(rt2) && std::isfinite(rt3) &&
         std::isfinite(guard_band) && rt1 >= 0.0f && rt1 <= 1.0f &&
         rt2 >= 0.0f && rt2 <= 1.0f && rt3 >= 0.0f && rt3 <= 1.0f &&
         guard_band >= 0.0f && guard_band <= 1.0f;
}

void classify_regions_host(const plan7_domain_posterior *posterior, size_t L,
                           float rt1, float rt2, float rt3, float guard,
                           uint32_t *uncertain_count,
                           uint32_t *region_count,
                           uint32_t *multidomain_count,
                           plan7_simple_region *region_output,
                           size_t region_capacity) {
  uint32_t uncertain = 0;
  uint32_t regions = 0;
  uint32_t multidomain = 0;
  int64_t region_begin = -1;
  bool triggered = false;
  for (size_t j = 1; j <= L; ++j) {
    const float bocc = posterior[j].btot - posterior[j - 1].btot;
    const float eocc = posterior[j].etot - posterior[j - 1].etot;
    const float left = posterior[j].mocc - bocc;
    const float right = posterior[j].mocc - eocc;
    if (!triggered) {
      if (std::fabs(posterior[j].mocc - rt1) <= guard) ++uncertain;
      if (std::fabs(left - rt2) <= guard) ++uncertain;
      if (left < rt2 || region_begin == -1)
        region_begin = static_cast<int64_t>(j);
      if (posterior[j].mocc >= rt1) triggered = true;
    } else {
      if (std::fabs(right - rt2) <= guard) ++uncertain;
      if (right >= rt2) continue;
      if (regions < region_capacity && region_output != nullptr)
        region_output[regions] = {
            static_cast<uint32_t>(region_begin), static_cast<uint32_t>(j)};
      ++regions;
      float maximum = -1.0f;
      const size_t begin = static_cast<size_t>(region_begin);
      for (size_t z = begin; z <= j; ++z) {
        const float left_expected =
            posterior[z].etot - posterior[begin - 1].etot;
        const float right_expected =
            posterior[j].btot - posterior[z - 1].btot;
        maximum = std::max(maximum,
                           std::min(left_expected, right_expected));
      }
      if (std::fabs(maximum - rt3) <= guard) ++uncertain;
      if (maximum >= rt3) ++multidomain;
      region_begin = -1;
      triggered = false;
    }
  }
  *uncertain_count = uncertain;
  *region_count = regions;
  *multidomain_count = multidomain;
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

template <bool CollectReasonFacts>
__device__ __forceinline__ void add_backward_reason(
    uint32_t *reason_facts, size_t candidate, uint32_t reason) {
  if constexpr (CollectReasonFacts) reason_facts[candidate] |= reason;
}

template <bool CollectReasonFacts>
__global__ void backward_domain_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const plan7_forward_device_profile *profiles, const float *emissions,
    const float *transitions,
    const plan7_backward_domain_candidate *candidates,
    const uint64_t *forward_offsets, const float *forward_specials,
    const uint64_t *dp_offsets, const uint64_t *backward_offsets,
    const uint64_t *posterior_offsets, size_t candidate_count,
    float rt1, float rt2, float rt3, float guard_band,
    float *dp_storage, float *backward_specials,
    plan7_domain_posterior *posteriors,
    plan7_backward_domain_result *results,
    uint32_t *reason_facts) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  if (lane >= kSubwarp) return;
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * kCandidatesPerBlock + warp_in_block;
  if (candidate >= candidate_count) return;
  const int sublane = lane;
  const unsigned mask = 0xFU;
  const plan7_backward_domain_candidate row = candidates[candidate];
  const plan7_forward_device_profile profile = profiles[row.profile_index];
  const int Q = static_cast<int>(profile.q);
  const uint64_t sequence_begin = sequence_offsets[row.sequence_index];
  const int L = static_cast<int>(sequence_offsets[row.sequence_index + 1] -
                                 sequence_begin);
  const float nj = profile.e_loop == 0.0f ? 0.0f : 1.0f;
  const float move = __fdiv_rn(add_rn(2.0f, nj),
                               add_rn(static_cast<float>(L),
                                      add_rn(2.0f, nj)));
  const float loop = sub_rn(1.0f, move);
  const float *forward = forward_specials + forward_offsets[candidate];
  float *backward = backward_specials + backward_offsets[candidate];
  plan7_domain_posterior *posterior =
      posteriors + posterior_offsets[candidate];
  float *mmx = dp_storage + dp_offsets[candidate];
  float *dmx = mmx + static_cast<uint64_t>(Q) * kSubwarp;
  float *imx = dmx + static_cast<uint64_t>(Q) * kSubwarp;

  float xJ = 0.0f;
  float xB = 0.0f;
  float xN = 0.0f;
  float xC = move;
  float xE = mul_rn(xC, profile.e_move);
  float xEv = xE;
  float dcv = 0.0f;
  for (int q = 0; q < Q; ++q) {
    mmx[q * kSubwarp + sublane] = xEv;
    dmx[q * kSubwarp + sublane] = xEv;
    imx[q * kSubwarp + sublane] = 0.0f;
  }

  float dpv = shift_left(dmx[(Q - 1) * kSubwarp + sublane], mask, sublane);
  for (int q = Q - 1; q >= 0; --q) {
    dcv = mul_rn(dpv, transition_value(
        profile, transitions, q, p7O_DD, sublane));
    dmx[q * kSubwarp + sublane] =
        add_rn(dmx[q * kSubwarp + sublane], dcv);
    dpv = dmx[q * kSubwarp + sublane];
  }
  for (int pass = 1; pass < 4; ++pass) {
    dcv = shift_left(dcv, mask, sublane);
    for (int q = Q - 1; q >= 0; --q) {
      dcv = mul_rn(dcv, transition_value(
          profile, transitions, q, p7O_DD, sublane));
      dmx[q * kSubwarp + sublane] =
          add_rn(dmx[q * kSubwarp + sublane], dcv);
    }
  }
  dcv = shift_left(dmx[sublane], mask, sublane);
  for (int q = Q - 1; q >= 0; --q) {
    mmx[q * kSubwarp + sublane] = add_rn(
        mmx[q * kSubwarp + sublane],
        mul_rn(dcv, transition_value(
            profile, transitions, q, p7O_MD, sublane)));
    dcv = dmx[q * kSubwarp + sublane];
  }

  float scale = forward[L * p7X_NXCELLS + p7X_SCALE];
  float totscale = static_cast<float>(log(static_cast<double>(scale)));
  if (scale > 1.0f) {
    xE = __fdiv_rn(xE, scale);
    xN = __fdiv_rn(xN, scale);
    xC = __fdiv_rn(xC, scale);
    xJ = __fdiv_rn(xJ, scale);
    xB = __fdiv_rn(xB, scale);
    const float inverse = __fdiv_rn(1.0f, scale);
    for (int q = 0; q < Q; ++q) {
      mmx[q * kSubwarp + sublane] =
          mul_rn(mmx[q * kSubwarp + sublane], inverse);
      dmx[q * kSubwarp + sublane] =
          mul_rn(dmx[q * kSubwarp + sublane], inverse);
      imx[q * kSubwarp + sublane] =
          mul_rn(imx[q * kSubwarp + sublane], inverse);
    }
  }
  if (sublane == 0) {
    backward[L * p7X_NXCELLS + p7X_E] = xE;
    backward[L * p7X_NXCELLS + p7X_N] = xN;
    backward[L * p7X_NXCELLS + p7X_J] = xJ;
    backward[L * p7X_NXCELLS + p7X_B] = xB;
    backward[L * p7X_NXCELLS + p7X_C] = xC;
    backward[L * p7X_NXCELLS + p7X_SCALE] = scale;
  }

  bool own_scales = false;
  for (int i = L - 1; i >= 1; --i) {
    const unsigned residue = residues[sequence_begin + i];
    float tmmv = shift_left(transition_value(
        profile, transitions, 0, p7O_MM, sublane), mask, sublane);
    float timv = shift_left(transition_value(
        profile, transitions, 0, p7O_IM, sublane), mask, sublane);
    float tdmv = shift_left(transition_value(
        profile, transitions, 0, p7O_DM, sublane), mask, sublane);
    float mpv = mul_rn(mmx[sublane], emission_value(
        profile, emissions, residue, 0, sublane));
    mpv = shift_left(mpv, mask, sublane);
    float xBv = 0.0f;
    for (int q = Q - 1; q >= 0; --q) {
      const float ipv = imx[q * kSubwarp + sublane];
      const float new_i = add_rn(
          mul_rn(ipv, transition_value(
              profile, transitions, q, p7O_II, sublane)),
          mul_rn(mpv, timv));
      const float new_d = mul_rn(mpv, tdmv);
      const float new_m = add_rn(
          mul_rn(ipv, transition_value(
              profile, transitions, q, p7O_MI, sublane)),
          mul_rn(mpv, tmmv));
      mpv = mul_rn(mmx[q * kSubwarp + sublane], emission_value(
          profile, emissions, residue, q, sublane));
      imx[q * kSubwarp + sublane] = new_i;
      dmx[q * kSubwarp + sublane] = new_d;
      mmx[q * kSubwarp + sublane] = new_m;
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
    xE = add_rn(mul_rn(xC, profile.e_move),
                mul_rn(xJ, profile.e_loop));
    xEv = xE;

    dpv = add_rn(dmx[sublane], xEv);
    dpv = shift_left(dpv, mask, sublane);
    for (int q = Q - 1; q >= 0; --q) {
      dcv = mul_rn(dpv, transition_value(
          profile, transitions, q, p7O_DD, sublane));
      dmx[q * kSubwarp + sublane] = add_rn(
          dmx[q * kSubwarp + sublane], add_rn(dcv, xEv));
      dpv = dmx[q * kSubwarp + sublane];
      mmx[q * kSubwarp + sublane] =
          add_rn(mmx[q * kSubwarp + sublane], xEv);
    }
    for (int pass = 1; pass < 4; ++pass) {
      dcv = shift_left(dcv, mask, sublane);
      for (int q = Q - 1; q >= 0; --q) {
        dcv = mul_rn(dcv, transition_value(
            profile, transitions, q, p7O_DD, sublane));
        dmx[q * kSubwarp + sublane] =
            add_rn(dmx[q * kSubwarp + sublane], dcv);
      }
    }
    dcv = shift_left(dmx[sublane], mask, sublane);
    for (int q = Q - 1; q >= 0; --q) {
      mmx[q * kSubwarp + sublane] = add_rn(
          mmx[q * kSubwarp + sublane],
          mul_rn(dcv, transition_value(
              profile, transitions, q, p7O_MD, sublane)));
      dcv = dmx[q * kSubwarp + sublane];
    }

    if (xB > 1.0e16f) own_scales = true;
    scale = own_scales ? (xB > 1.0e4f ? xB : 1.0f)
                       : forward[i * p7X_NXCELLS + p7X_SCALE];
    if (scale > 1.0f) {
      xE = __fdiv_rn(xE, scale);
      xN = __fdiv_rn(xN, scale);
      xJ = __fdiv_rn(xJ, scale);
      xB = __fdiv_rn(xB, scale);
      xC = __fdiv_rn(xC, scale);
      const float inverse = __fdiv_rn(1.0f, scale);
      for (int q = 0; q < Q; ++q) {
        mmx[q * kSubwarp + sublane] =
            mul_rn(mmx[q * kSubwarp + sublane], inverse);
        dmx[q * kSubwarp + sublane] =
            mul_rn(dmx[q * kSubwarp + sublane], inverse);
        imx[q * kSubwarp + sublane] =
            mul_rn(imx[q * kSubwarp + sublane], inverse);
      }
      if (sublane == 0)
        totscale = static_cast<float>(
            static_cast<double>(totscale) + log(static_cast<double>(scale)));
    }
    if (sublane == 0) {
      backward[i * p7X_NXCELLS + p7X_E] = xE;
      backward[i * p7X_NXCELLS + p7X_N] = xN;
      backward[i * p7X_NXCELLS + p7X_J] = xJ;
      backward[i * p7X_NXCELLS + p7X_B] = xB;
      backward[i * p7X_NXCELLS + p7X_C] = xC;
      backward[i * p7X_NXCELLS + p7X_SCALE] = scale;
    }
  }

  const unsigned first_residue = residues[sequence_begin];
  float xBv = 0.0f;
  for (int q = 0; q < Q; ++q) {
    float mpv = mul_rn(mmx[q * kSubwarp + sublane], emission_value(
        profile, emissions, first_residue, q, sublane));
    mpv = mul_rn(mpv, transition_value(
        profile, transitions, q, p7O_BM, sublane));
    xBv = add_rn(xBv, mpv);
  }
  xB = horizontal_sum_sse(xBv, mask, sublane);
  xN = add_rn(mul_rn(xB, move), mul_rn(xN, loop));
  if (sublane == 0) {
    backward[p7X_B] = xB;
    backward[p7X_C] = 0.0f;
    backward[p7X_J] = 0.0f;
    backward[p7X_N] = xN;
    backward[p7X_E] = 0.0f;
    backward[p7X_SCALE] = 1.0f;
  }
  __syncwarp(mask);

  if (sublane == 0) {
    plan7_backward_domain_result result{};
    result.profile_index = row.profile_index;
    result.sequence_index = row.sequence_index;
    result.status = PLAN7_BACKWARD_DOMAIN_OK;
    result.route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
    result.has_own_scales = own_scales ? 1 : 0;
    if (own_scales)
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          PLAN7_BACKWARD_DOMAIN_REASON_HAS_OWN_SCALES);
    if (isnan(xN) || xN == 0.0f || isinf(xN)) {
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          PLAN7_BACKWARD_DOMAIN_REASON_TERMINAL_SCORE_INVALID);
      result.backward_score = nanf("");
      result.status = PLAN7_BACKWARD_DOMAIN_ERANGE;
      results[candidate] = result;
      return;
    }
    result.backward_score = static_cast<float>(
        static_cast<double>(totscale) + log(static_cast<double>(xN)));
    posterior[0] = {0.0f, 0.0f, 0.0f};
    float scaleproduct = __fdiv_rn(1.0f, backward[p7X_N]);
    bool finite = isfinite(scaleproduct);
    for (int i = 1; i <= L; ++i) {
      float value = mul_rn(forward[(i - 1) * p7X_NXCELLS + p7X_B],
                           backward[(i - 1) * p7X_NXCELLS + p7X_B]);
      value = mul_rn(value,
                     forward[(i - 1) * p7X_NXCELLS + p7X_SCALE]);
      value = mul_rn(value, scaleproduct);
      posterior[i].btot = add_rn(posterior[i - 1].btot, value);
      if (own_scales) {
        const float ratio = __fdiv_rn(
            forward[(i - 1) * p7X_NXCELLS + p7X_SCALE],
            backward[(i - 1) * p7X_NXCELLS + p7X_SCALE]);
        scaleproduct = mul_rn(scaleproduct, ratio);
      }
      value = mul_rn(forward[i * p7X_NXCELLS + p7X_E],
                     backward[i * p7X_NXCELLS + p7X_E]);
      value = mul_rn(value, forward[i * p7X_NXCELLS + p7X_SCALE]);
      value = mul_rn(value, scaleproduct);
      posterior[i].etot = add_rn(posterior[i - 1].etot, value);

      float njcp = mul_rn(forward[(i - 1) * p7X_NXCELLS + p7X_N],
                          backward[i * p7X_NXCELLS + p7X_N]);
      njcp = mul_rn(njcp, loop);
      njcp = mul_rn(njcp, scaleproduct);
      float component = mul_rn(
          forward[(i - 1) * p7X_NXCELLS + p7X_J],
          backward[i * p7X_NXCELLS + p7X_J]);
      component = mul_rn(component, loop);
      component = mul_rn(component, scaleproduct);
      njcp = add_rn(njcp, component);
      component = mul_rn(
          forward[(i - 1) * p7X_NXCELLS + p7X_C],
          backward[i * p7X_NXCELLS + p7X_C]);
      component = mul_rn(component, loop);
      component = mul_rn(component, scaleproduct);
      njcp = add_rn(njcp, component);
      posterior[i].mocc = sub_rn(1.0f, njcp);
      finite = finite && isfinite(scaleproduct) &&
               isfinite(posterior[i].btot) &&
               isfinite(posterior[i].etot) &&
               isfinite(posterior[i].mocc);
    }
    if (!finite || !isfinite(result.backward_score)) {
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          PLAN7_BACKWARD_DOMAIN_REASON_POSTERIOR_OR_BACKWARD_SCORE_NONFINITE);
      result.status = PLAN7_BACKWARD_DOMAIN_ERANGE;
      results[candidate] = result;
      return;
    }

    uint32_t uncertain = 0;
    uint32_t regions = 0;
    uint32_t multidomain = 0;
    int region_begin = -1;
    bool triggered = false;
    for (int j = 1; j <= L; ++j) {
      const float bocc = sub_rn(posterior[j].btot,
                                posterior[j - 1].btot);
      const float eocc = sub_rn(posterior[j].etot,
                                posterior[j - 1].etot);
      const float left = sub_rn(posterior[j].mocc, bocc);
      const float right = sub_rn(posterior[j].mocc, eocc);
      if (!triggered) {
        if (fabsf(sub_rn(posterior[j].mocc, rt1)) <= guard_band) ++uncertain;
        if (fabsf(sub_rn(left, rt2)) <= guard_band) ++uncertain;
        if (left < rt2 || region_begin == -1) region_begin = j;
        if (posterior[j].mocc >= rt1) triggered = true;
      } else {
        if (fabsf(sub_rn(right, rt2)) <= guard_band) ++uncertain;
        if (right >= rt2) continue;
        ++regions;
        float maximum = -1.0f;
        for (int z = region_begin; z <= j; ++z) {
          const float left_expected = sub_rn(
              posterior[z].etot, posterior[region_begin - 1].etot);
          const float right_expected = sub_rn(
              posterior[j].btot, posterior[z - 1].btot);
          maximum = fmaxf(maximum, fminf(left_expected, right_expected));
        }
        if (fabsf(sub_rn(maximum, rt3)) <= guard_band) ++uncertain;
        if (maximum >= rt3) ++multidomain;
        region_begin = -1;
        triggered = false;
      }
    }
    result.uncertain_count = uncertain;
    result.region_count = regions;
    result.multidomain_count = multidomain;
    if (uncertain != 0)
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          PLAN7_BACKWARD_DOMAIN_REASON_THRESHOLD_UNCERTAIN);
    if (multidomain != 0)
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          PLAN7_BACKWARD_DOMAIN_REASON_MULTIDOMAIN);
    result.nexpected = posterior[L].btot;
    if (!(result.nexpected >= 0.0f) ||
        !(result.nexpected <= static_cast<float>(L))) {
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          PLAN7_BACKWARD_DOMAIN_REASON_NEXPECTED_INVALID);
      result.status = PLAN7_BACKWARD_DOMAIN_ERANGE;
      results[candidate] = result;
      return;
    }
    if (uncertain == 0 && multidomain == 0) {
      result.route = regions == 0 ? PLAN7_BACKWARD_DOMAIN_NO_REGIONS
                                  : PLAN7_BACKWARD_DOMAIN_SIMPLE;
      add_backward_reason<CollectReasonFacts>(
          reason_facts, candidate,
          regions == 0 ? PLAN7_BACKWARD_DOMAIN_REASON_NO_REGIONS
                       : PLAN7_BACKWARD_DOMAIN_REASON_SIMPLE);
    }
    results[candidate] = result;
  }
}

__global__ void gather_simple_regions_kernel(
    const plan7_domain_posterior *posteriors,
    const uint64_t *posterior_offsets,
    const uint64_t *region_offsets,
    const uint64_t *sequence_offsets,
    const plan7_backward_domain_candidate *candidates,
    const plan7_backward_domain_result *results,
    size_t candidate_count, float rt1, float rt2,
    plan7_simple_region *regions,
    plan7_backward_domain_resident_region *resident_regions) {
  const size_t candidate = static_cast<size_t>(blockIdx.x);
  if (candidate >= candidate_count || threadIdx.x != 0) return;
  const uint64_t output_begin = region_offsets[candidate];
  const uint64_t output_end = region_offsets[candidate + 1];
  if (output_begin == output_end) return;
  const uint64_t input_begin = posterior_offsets[candidate];
  const uint64_t input_end = posterior_offsets[candidate + 1];
  const plan7_domain_posterior *posterior = posteriors + input_begin;
  const int L = static_cast<int>(input_end - input_begin - 1);
  uint64_t output = output_begin;
  int region_begin = -1;
  bool triggered = false;
  for (int j = 1; j <= L; ++j) {
    const float bocc = __fsub_rn(posterior[j].btot,
                                 posterior[j - 1].btot);
    const float eocc = __fsub_rn(posterior[j].etot,
                                 posterior[j - 1].etot);
    const float left = __fsub_rn(posterior[j].mocc, bocc);
    const float right = __fsub_rn(posterior[j].mocc, eocc);
    if (!triggered) {
      if (left < rt2 || region_begin == -1) region_begin = j;
      if (posterior[j].mocc >= rt1) triggered = true;
    } else if (right < rt2) {
      if (output < output_end) {
        regions[output] = {static_cast<uint32_t>(region_begin),
                           static_cast<uint32_t>(j)};
        if (resident_regions != nullptr) {
          const plan7_backward_domain_candidate source = candidates[candidate];
          const uint64_t target_length =
              sequence_offsets[source.sequence_index + 1] -
              sequence_offsets[source.sequence_index];
          resident_regions[output] = {
              source.profile_index,
              source.sequence_index,
              static_cast<uint32_t>(region_begin),
              static_cast<uint32_t>(j),
              static_cast<uint32_t>(target_length),
              results[candidate].has_own_scales,
              {0, 0, 0}};
        }
      }
      ++output;
      region_begin = -1;
      triggered = false;
    }
  }
}

struct DeviceBuffers {
  plan7_backward_domain_candidate *candidates = nullptr;
  uint64_t *forward_offsets = nullptr;
  uint64_t *dp_offsets = nullptr;
  uint64_t *backward_offsets = nullptr;
  uint64_t *posterior_offsets = nullptr;
  uint64_t *region_offsets = nullptr;
  float *forward_specials = nullptr;
  float *dp = nullptr;
  float *backward_specials = nullptr;
  plan7_domain_posterior *posteriors = nullptr;
  plan7_simple_region *regions = nullptr;
  plan7_backward_domain_resident_region *resident_regions = nullptr;
  plan7_backward_domain_result *results = nullptr;
  uint32_t *reason_facts = nullptr;
};

void free_device_buffers(DeviceBuffers *buffers) {
  if (buffers == nullptr) return;
  cudaFree(buffers->reason_facts);
  cudaFree(buffers->results);
  cudaFree(buffers->resident_regions);
  cudaFree(buffers->regions);
  cudaFree(buffers->posteriors);
  cudaFree(buffers->backward_specials);
  cudaFree(buffers->dp);
  cudaFree(buffers->forward_specials);
  cudaFree(buffers->posterior_offsets);
  cudaFree(buffers->region_offsets);
  cudaFree(buffers->backward_offsets);
  cudaFree(buffers->dp_offsets);
  cudaFree(buffers->forward_offsets);
  cudaFree(buffers->candidates);
  *buffers = {};
}

}  // namespace

struct ResidentBackwardRegions {
  plan7_backward_domain_resident_region *pointer = nullptr;
  int device_ordinal = -1;
  uint64_t database_generation = 0;
  uint64_t batch_generation = 0;
  uint64_t result_hash = 0;
  uint64_t region_hash = 0;
  uint64_t row_count = 0;
  uint64_t region_count = 0;
  bool valid = false;

  ~ResidentBackwardRegions() {
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

  ResidentBackwardRegions() = default;
  ResidentBackwardRegions(const ResidentBackwardRegions &) = delete;
  ResidentBackwardRegions &operator=(const ResidentBackwardRegions &) = delete;
};

struct plan7_backward_domain_output {
  std::vector<plan7_backward_domain_result> results;
  std::vector<uint64_t> posterior_offsets;
  std::vector<plan7_domain_posterior> posteriors;
  std::vector<uint64_t> region_offsets;
  std::vector<plan7_simple_region> regions;
  std::vector<uint32_t> reason_facts;
  plan7_backward_domain_statistics statistics;
  plan7_backward_domain_residency_statistics residency_statistics{};
  ResidentBackwardRegions resident_regions;
  plan7_backward_domain_provenance provenance;
  float rt1 = NAN;
  float rt2 = NAN;
  float rt3 = NAN;
  float guard_band = NAN;
  bool sealed = false;
};

namespace {

template <typename SourceIndex>
bool merge_backward_reason_facts(
    const SourceIndex *active_sources, const uint32_t *active_facts,
    size_t active_count, uint32_t *source_facts, size_t source_count) {
  if ((active_count != 0 &&
       (active_sources == nullptr || active_facts == nullptr)) ||
      (source_count != 0 && source_facts == nullptr))
    return false;
  size_t previous = 0;
  bool have_previous = false;
  for (size_t active = 0; active < active_count; ++active) {
    const uint64_t source = static_cast<uint64_t>(active_sources[active]);
    if (source >= source_count || (have_previous && source <= previous))
      return false;
    source_facts[static_cast<size_t>(source)] = active_facts[active];
    previous = static_cast<size_t>(source);
    have_previous = true;
  }
  return true;
}

bool seal_backward_domain_provenance(
    plan7_backward_domain_output *output,
    const plan7_forward_provenance &forward,
    float rt1, float rt2, float rt3, float guard_band) {
  if (output == nullptr ||
      output->region_offsets.size() != output->results.size() + 1)
    return false;
  plan7_backward_domain_provenance sealed{};
  sealed.forward = forward;
  uint64_t threshold_hash =
      hash_u64(kHashOffset, UINT64_C(0x54485245));
  for (const float value : {rt1, rt2, rt3, guard_band}) {
    FloatBits bits{};
    bits.value = value;
    threshold_hash = hash_u32(threshold_hash, bits.bits);
  }
  uint64_t result_hash = hash_u64(kHashOffset, UINT64_C(0x52455355));
  for (const plan7_backward_domain_result &result : output->results) {
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
  }
  uint64_t region_hash = hash_u64(kHashOffset, UINT64_C(0x5245474e));
  for (size_t candidate = 0; candidate < output->results.size(); ++candidate) {
    const uint64_t begin = output->region_offsets[candidate];
    const uint64_t end = output->region_offsets[candidate + 1];
    if (begin > end || end > output->regions.size()) return false;
    region_hash = hash_u64(region_hash, end - begin);
    for (uint64_t index = begin; index < end; ++index) {
      region_hash = hash_u32(region_hash,
                             output->regions[index].begin);
      region_hash = hash_u32(region_hash,
                             output->regions[index].end);
    }
  }
  sealed.candidate_count = output->results.size();
  sealed.region_count = output->regions.size();
  sealed.threshold_hash = hash_u64(
      threshold_hash, PLAN7_BACKWARD_DOMAIN_RECORD_VERSION);
  sealed.result_hash = hash_u64(result_hash, sealed.candidate_count);
  sealed.region_hash = hash_u64(region_hash, sealed.region_count);
  output->provenance = sealed;
  return true;
}

int backward_domain_run_impl(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const plan7_forward_provenance *provenance,
    const uint64_t *forward_offsets,
    const float *forward_specials, size_t forward_special_count,
    const plan7_forward_resident_view *resident_view,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget,
    bool unsealed_test, bool collect_reason_facts,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  const auto total_begin = std::chrono::steady_clock::now();
  if (output == nullptr || *output != nullptr || database == nullptr ||
      batch == nullptr || (!unsealed_test && provenance == nullptr) ||
      forward_offsets == nullptr ||
      (candidate_count != 0 &&
       candidates == nullptr) ||
      (forward_special_count != 0 && forward_specials == nullptr) ||
      !valid_thresholds(rt1, rt2, rt3, guard_band)) {
    set_error(error, error_size, "invalid Backward/domain run arguments");
    return -1;
  }
  if (candidate_count > UINT32_MAX) {
    set_error(error, error_size, "Backward/domain candidate count exceeds uint32");
    return -1;
  }
  if (unsealed_test && resident_view != nullptr) {
    set_error(error, error_size,
              "test-only Backward/domain cannot consume resident Forward state");
    return -1;
  }
  if (candidate_count >
          kResultOutputByteLimit / sizeof(plan7_backward_domain_result) ||
      forward_special_count >
          kForwardSpecialByteLimit / sizeof(float)) {
    set_error(error, error_size,
              "Backward/domain input exceeds bounded host output limits");
    return -1;
  }

  /* The raw C seam accepts caller-owned arrays. Snapshot them once before
   * validation so validation, GPU upload, and the output seal all attest the
   * exact same bytes even if another thread mutates the caller's storage. */
  plan7_forward_provenance provenance_snapshot{};
  if (!unsealed_test) provenance_snapshot = *provenance;
  std::vector<plan7_backward_domain_candidate> candidate_snapshot;
  std::vector<uint64_t> offset_snapshot;
  std::vector<float> special_snapshot;
  try {
    if (candidate_count != 0)
      candidate_snapshot.assign(candidates, candidates + candidate_count);
    offset_snapshot.assign(forward_offsets,
                           forward_offsets + candidate_count + 1);
    if (forward_special_count != 0)
      special_snapshot.assign(forward_specials,
                              forward_specials + forward_special_count);
  } catch (...) {
    set_error(error, error_size,
              "Backward/domain immutable input snapshot failed");
    return -1;
  }
  candidates = candidate_snapshot.empty() ? nullptr : candidate_snapshot.data();
  forward_offsets = offset_snapshot.data();
  forward_specials = special_snapshot.empty() ? nullptr : special_snapshot.data();

  plan7_forward_device_view profile_view{};
  plan7_ssv_sequence_batch_view sequence_view{};
  if (plan7_forward_database_get_device_view(
          database, &profile_view, error, error_size) != 0 ||
      plan7_ssv_sequence_batch_get_view(
          batch, &sequence_view, error, error_size) != 0)
    return -1;
  int current_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&current_device);
  if (cuda_status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", cuda_status);
    return -1;
  }
  if (profile_view.device_ordinal != current_device ||
      sequence_view.device_ordinal != current_device) {
    set_error(error, error_size,
              "Backward/domain inputs belong to a different CUDA device");
    return -1;
  }
  if (profile_view.alphabet_size != 29 ||
      sequence_view.alphabet_size != 29) {
    set_error(error, error_size,
              "Backward/domain requires the amino alphabet");
    return -1;
  }
  if (candidate_count != 0 &&
      (!device_allocation_on(profile_view.profiles, current_device) ||
       !device_allocation_on(profile_view.emissions, current_device) ||
       !device_allocation_on(profile_view.transitions, current_device) ||
       !device_allocation_on(sequence_view.device_residues, current_device) ||
       !device_allocation_on(sequence_view.device_offsets, current_device))) {
    set_error(error, error_size,
              "Backward/domain input device pointer is invalid");
    return -1;
  }
  if (resident_view != nullptr &&
      (resident_view->database_generation !=
           provenance_snapshot.database_generation ||
       resident_view->batch_generation != provenance_snapshot.batch_generation ||
       resident_view->pass_count != candidate_count ||
       resident_view->special_count != forward_special_count ||
       resident_view->device_ordinal != current_device ||
       (forward_special_count != 0 &&
        (resident_view->specials == nullptr ||
         !device_allocation_on(resident_view->specials, current_device))))) {
    set_error(error, error_size,
              "Backward/domain resident Forward generation mismatch");
    return -1;
  }
  if (forward_offsets == nullptr || forward_offsets[0] != 0 ||
      forward_offsets[candidate_count] != forward_special_count) {
    set_error(error, error_size, "invalid Backward/domain Forward offsets");
    return -1;
  }
  if (!unsealed_test &&
      (plan7_forward_database_validate_provenance(
           database, &provenance_snapshot) != 1 ||
       !matches_forward_provenance(
           provenance_snapshot, profile_view, sequence_view, candidates,
           candidate_count, forward_offsets, forward_specials,
           forward_special_count))) {
    set_error(error, error_size,
              "Backward/domain Forward provenance mismatch");
    return -1;
  }

  std::unique_ptr<plan7_backward_domain_output> created(
      new (std::nothrow) plan7_backward_domain_output{});
  if (!created) {
    set_error(error, error_size, "Backward/domain output allocation failed");
    return -1;
  }
  created->rt1 = rt1;
  created->rt2 = rt2;
  created->rt3 = rt3;
  created->guard_band = guard_band;
  try {
    created->results.resize(candidate_count);
    created->posterior_offsets.assign(candidate_count + 1, 0);
    created->region_offsets.assign(candidate_count + 1, 0);
    if (collect_reason_facts)
      created->reason_facts.assign(candidate_count, 0);
  } catch (...) {
    set_error(error, error_size, "Backward/domain output allocation failed");
    return -1;
  }
  created->statistics.candidate_count = candidate_count;
  created->residency_statistics.resident_input_count =
      resident_view == nullptr ? 0 : 1;
  created->statistics.output_byte_limit = std::min(
      posterior_byte_budget,
      static_cast<uint64_t>(PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES));

  std::vector<plan7_forward_snapshot_profile> profile_snapshots;
  try {
    profile_snapshots.resize(profile_view.profile_count);
  } catch (...) {
    set_error(error, error_size, "Backward/domain profile allocation failed");
    return -1;
  }
  for (size_t profile = 0; profile < profile_view.profile_count; ++profile) {
    if (plan7_forward_database_get_profile_snapshot(
            database, profile, &profile_snapshots[profile],
            error, error_size) != 0)
      return -1;
  }

  std::vector<size_t> active_sources;
  std::vector<plan7_backward_domain_candidate> active_candidates;
  std::vector<uint64_t> active_forward_offsets;
  std::vector<uint64_t> active_dp_offsets;
  std::vector<uint64_t> active_backward_offsets;
  std::vector<uint64_t> active_posterior_offsets;
  std::vector<float> active_forward_specials;
  uint64_t requested_posterior_bytes = 0;
  uint64_t dp_bytes = 0;
  uint64_t backward_bytes = 0;
  uint64_t forward_bytes = 0;
  uint64_t work_cells = 0;
  const bool use_resident_forward = resident_view != nullptr;
  try {
    active_sources.reserve(candidate_count);
    active_candidates.reserve(candidate_count);
    if (!use_resident_forward) {
      active_forward_specials.reserve(forward_special_count);
      active_forward_offsets.push_back(0);
    } else {
      active_forward_offsets.reserve(candidate_count + 1);
    }
    active_dp_offsets.push_back(0);
    active_backward_offsets.push_back(0);
    active_posterior_offsets.push_back(0);
  } catch (...) {
    set_error(error, error_size, "Backward/domain host allocation failed");
    return -1;
  }

  uint32_t previous_profile = 0;
  uint32_t previous_sequence = 0;
  bool have_previous = false;
  for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
    const auto row = candidates[candidate];
    if (row.profile_index >= profile_view.profile_count ||
        row.sequence_index >= sequence_view.sequence_count) {
      set_error(error, error_size,
                "Backward/domain candidate index is out of range");
      return -1;
    }
    if (have_previous &&
        (row.profile_index < previous_profile ||
         (row.profile_index == previous_profile &&
          row.sequence_index <= previous_sequence))) {
      set_error(error, error_size,
                "Backward/domain candidates are not profile-major ordered");
      return -1;
    }
    have_previous = true;
    previous_profile = row.profile_index;
    previous_sequence = row.sequence_index;
    const uint64_t L = sequence_view.host_lengths[row.sequence_index];
    const auto profile = profile_snapshots[row.profile_index];
    if (L > kMaximumTargetLength ||
        profile.model_length < 1 ||
        profile.model_length > kMaximumModelLength ||
        profile.q != std::max<uint32_t>(
                         2, (profile.model_length + 3) / 4)) {
      set_error(error, error_size,
                "Backward/domain candidate dimensions are invalid");
      return -1;
    }
    uint64_t expected_forward_cells;
    if (!checked_add(L, 1, &expected_forward_cells) ||
        !checked_multiply(expected_forward_cells, p7X_NXCELLS,
                          &expected_forward_cells) ||
        forward_offsets[candidate] > forward_offsets[candidate + 1] ||
        forward_offsets[candidate + 1] - forward_offsets[candidate] !=
            expected_forward_cells) {
      set_error(error, error_size,
                "Backward/domain Forward row has the wrong size");
      return -1;
    }
    plan7_backward_domain_result &result = created->results[candidate];
    result = {};
    result.profile_index = row.profile_index;
    result.sequence_index = row.sequence_index;
    result.backward_score = NAN;
    result.nexpected = NAN;
    result.route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
    if (L == 0) {
      result.status = PLAN7_BACKWARD_DOMAIN_EMPTY;
      if (collect_reason_facts)
        created->reason_facts[candidate] |=
            PLAN7_BACKWARD_DOMAIN_REASON_TARGET_EMPTY;
      continue;
    }
    bool forward_valid = true;
    for (uint64_t cell = forward_offsets[candidate];
         cell < forward_offsets[candidate + 1]; ++cell)
      if (!std::isfinite(forward_specials[cell])) {
        forward_valid = false;
        if (collect_reason_facts)
          created->reason_facts[candidate] |=
              PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SPECIAL_NONFINITE;
        break;
      }
    if (forward_valid) {
      for (uint64_t i = 0; i <= L; ++i) {
        const float row_scale = forward_specials[
            forward_offsets[candidate] + i * p7X_NXCELLS + p7X_SCALE];
        if (!(row_scale >= 1.0f) || !std::isfinite(row_scale)) {
          forward_valid = false;
          if (collect_reason_facts)
            created->reason_facts[candidate] |=
                PLAN7_BACKWARD_DOMAIN_REASON_FORWARD_SCALE_INVALID;
          break;
        }
      }
    }
    if (!forward_valid) {
      result.status = PLAN7_BACKWARD_DOMAIN_ERANGE;
      continue;
    }
    if (!sequence_view.host_float_environment_valid) {
      result.status = PLAN7_BACKWARD_DOMAIN_ENORESULT;
      if (collect_reason_facts)
        created->reason_facts[candidate] |=
            PLAN7_BACKWARD_DOMAIN_REASON_HOST_FLOAT_ENV_INVALID;
      continue;
    }
    if (profile.mode != p7_LOCAL || profile.nj != 1.0f) {
      /* The compact continuation currently supports canonical multihit-local
       * rows only. Do not emit a route that its HMMER consumer must reject. */
      result.status = PLAN7_BACKWARD_DOMAIN_ENORESULT;
      if (collect_reason_facts)
        created->reason_facts[candidate] |=
            PLAN7_BACKWARD_DOMAIN_REASON_MODE_OR_NJ_UNSUPPORTED;
      continue;
    }

    uint64_t posterior_cells;
    uint64_t row_posterior_bytes;
    uint64_t row_dp_cells;
    uint64_t row_dp_bytes;
    uint64_t row_special_bytes;
    uint64_t row_work_cells;
    if (!checked_add(L, 1, &posterior_cells) ||
        !checked_multiply(posterior_cells,
                          sizeof(plan7_domain_posterior),
                          &row_posterior_bytes) ||
        !checked_multiply(static_cast<uint64_t>(profile.q),
                          3 * kSubwarp, &row_dp_cells) ||
        !checked_multiply(row_dp_cells, sizeof(float), &row_dp_bytes) ||
        !checked_multiply(expected_forward_cells, sizeof(float),
                          &row_special_bytes) ||
        !checked_multiply(L, profile.model_length, &row_work_cells)) {
      set_error(error, error_size, "Backward/domain work size overflow");
      return -1;
    }
    if (row_work_cells > kMaximumRowWorkCells ||
        work_cells > kMaximumRunWorkCells - row_work_cells) {
      result.status = PLAN7_BACKWARD_DOMAIN_OK;
      ++created->statistics.work_cap_fallback_count;
      if (collect_reason_facts)
        created->reason_facts[candidate] |=
            PLAN7_BACKWARD_DOMAIN_REASON_WORK_CAP;
      continue;
    }
    uint64_t next_posterior_bytes;
    uint64_t next_dp_bytes;
    uint64_t next_backward_bytes;
    uint64_t next_forward_bytes;
    if (!checked_add(requested_posterior_bytes, row_posterior_bytes,
                     &next_posterior_bytes) ||
        !checked_add(dp_bytes, row_dp_bytes, &next_dp_bytes) ||
        !checked_add(backward_bytes, row_special_bytes,
                     &next_backward_bytes) ||
        !checked_add(forward_bytes, row_special_bytes,
                     &next_forward_bytes)) {
      set_error(error, error_size, "Backward/domain workspace size overflow");
      return -1;
    }
    if (next_posterior_bytes > kPosteriorWorkspaceByteLimit ||
        next_dp_bytes > kDpWorkspaceByteLimit ||
        next_backward_bytes > kBackwardSpecialByteLimit ||
        next_forward_bytes > kForwardSpecialByteLimit) {
      result.status = PLAN7_BACKWARD_DOMAIN_OK;
      ++created->statistics.output_cap_fallback_count;
      if (collect_reason_facts)
        created->reason_facts[candidate] |=
            PLAN7_BACKWARD_DOMAIN_REASON_WORKSPACE_CAP;
      continue;
    }
    active_sources.push_back(candidate);
    active_candidates.push_back(row);
    if (use_resident_forward) {
      active_forward_offsets.push_back(forward_offsets[candidate]);
    } else {
      active_forward_specials.insert(
          active_forward_specials.end(),
          forward_specials + forward_offsets[candidate],
          forward_specials + forward_offsets[candidate + 1]);
      active_forward_offsets.push_back(
          active_forward_offsets.back() + expected_forward_cells);
    }
    active_dp_offsets.push_back(active_dp_offsets.back() + row_dp_cells);
    active_backward_offsets.push_back(
        active_backward_offsets.back() + expected_forward_cells);
    active_posterior_offsets.push_back(
        active_posterior_offsets.back() + posterior_cells);
    requested_posterior_bytes = next_posterior_bytes;
    dp_bytes = next_dp_bytes;
    backward_bytes = next_backward_bytes;
    forward_bytes = next_forward_bytes;
    if (!checked_add(work_cells, row_work_cells, &work_cells)) {
      set_error(error, error_size, "Backward/domain work cell overflow");
      return -1;
    }
  }

  if (use_resident_forward)
    active_forward_offsets.push_back(forward_special_count);

  const size_t active_count = active_candidates.size();
  std::vector<plan7_backward_domain_result> active_results;
  active_results.resize(active_count);
  std::vector<uint32_t> active_reason_facts;
  if (collect_reason_facts) active_reason_facts.resize(active_count);

  DeviceBuffers buffers{};
  if (active_count != 0) {
    size_t candidate_bytes;
    size_t result_bytes;
    size_t offset_bytes;
    size_t forward_special_bytes;
    size_t posterior_bytes;
    size_t reason_bytes = 0;
    if (!checked_bytes(active_count,
                       sizeof(plan7_backward_domain_candidate),
                       &candidate_bytes) ||
        !checked_bytes(active_count, sizeof(plan7_backward_domain_result),
                       &result_bytes) ||
        !checked_bytes(active_count + 1, sizeof(uint64_t), &offset_bytes) ||
        (!use_resident_forward &&
         !checked_bytes(active_forward_specials.size(), sizeof(float),
                        &forward_special_bytes)) ||
        (use_resident_forward && forward_bytes > SIZE_MAX) ||
        requested_posterior_bytes > SIZE_MAX ||
        (collect_reason_facts &&
         !checked_bytes(active_count, sizeof(uint32_t), &reason_bytes))) {
      set_error(error, error_size, "Backward/domain device size overflow");
      return -1;
    }
    if (use_resident_forward)
      forward_special_bytes = static_cast<size_t>(forward_bytes);
    posterior_bytes = static_cast<size_t>(requested_posterior_bytes);
    const auto upload_begin = std::chrono::steady_clock::now();
    cudaEvent_t begin_event = nullptr;
    cudaEvent_t end_event = nullptr;
#define CUDA_RUN(call)                                                        \
    do {                                                                      \
      cuda_status = (call);                                                   \
      if (cuda_status != cudaSuccess) {                                       \
        set_cuda_error(error, error_size, #call, cuda_status);                \
        if (end_event != nullptr) cudaEventDestroy(end_event);                \
        if (begin_event != nullptr) cudaEventDestroy(begin_event);            \
        free_device_buffers(&buffers);                                        \
        return -1;                                                            \
      }                                                                       \
    } while (0)
    CUDA_RUN(cudaMalloc(&buffers.candidates, candidate_bytes));
    CUDA_RUN(cudaMalloc(&buffers.forward_offsets, offset_bytes));
    CUDA_RUN(cudaMalloc(&buffers.dp_offsets, offset_bytes));
    CUDA_RUN(cudaMalloc(&buffers.backward_offsets, offset_bytes));
    CUDA_RUN(cudaMalloc(&buffers.posterior_offsets, offset_bytes));
    if (!use_resident_forward)
      CUDA_RUN(cudaMalloc(&buffers.forward_specials, forward_special_bytes));
    CUDA_RUN(cudaMalloc(&buffers.dp, static_cast<size_t>(dp_bytes)));
    CUDA_RUN(cudaMalloc(&buffers.backward_specials,
                        static_cast<size_t>(backward_bytes)));
    CUDA_RUN(cudaMalloc(&buffers.posteriors, posterior_bytes));
    CUDA_RUN(cudaMalloc(&buffers.results, result_bytes));
    if (collect_reason_facts) {
      CUDA_RUN(cudaMalloc(&buffers.reason_facts, reason_bytes));
      CUDA_RUN(cudaMemset(buffers.reason_facts, 0, reason_bytes));
    }
    CUDA_RUN(cudaMemcpy(buffers.candidates, active_candidates.data(),
                        candidate_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.forward_offsets,
                        active_forward_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.dp_offsets, active_dp_offsets.data(),
                        offset_bytes, cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.backward_offsets,
                        active_backward_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
    CUDA_RUN(cudaMemcpy(buffers.posterior_offsets,
                        active_posterior_offsets.data(), offset_bytes,
                        cudaMemcpyHostToDevice));
    if (!use_resident_forward) {
      const auto forward_upload_begin = std::chrono::steady_clock::now();
      CUDA_RUN(cudaMemcpy(buffers.forward_specials,
                          active_forward_specials.data(),
                          forward_special_bytes, cudaMemcpyHostToDevice));
      created->residency_statistics.forward_special_upload_milliseconds =
          std::chrono::duration<float, std::milli>(
              std::chrono::steady_clock::now() - forward_upload_begin).count();
      created->residency_statistics.forward_special_h2d_bytes =
          forward_special_bytes;
    } else {
      created->residency_statistics.eliminated_forward_special_h2d_bytes =
          forward_special_bytes;
    }
    created->statistics.upload_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - upload_begin).count();

    CUDA_RUN(cudaEventCreate(&begin_event));
    CUDA_RUN(cudaEventCreate(&end_event));
    CUDA_RUN(cudaEventRecord(begin_event));
    const size_t block_count =
        (active_count + kCandidatesPerBlock - 1) / kCandidatesPerBlock;
    const float *kernel_forward_specials = use_resident_forward
        ? resident_view->specials : buffers.forward_specials;
    if (collect_reason_facts) {
      backward_domain_kernel<true><<<
          static_cast<unsigned>(block_count), kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.candidates,
          buffers.forward_offsets, kernel_forward_specials,
          buffers.dp_offsets, buffers.backward_offsets,
          buffers.posterior_offsets, active_count, rt1, rt2, rt3, guard_band,
          buffers.dp, buffers.backward_specials, buffers.posteriors,
          buffers.results, buffers.reason_facts);
    } else {
      backward_domain_kernel<false><<<
          static_cast<unsigned>(block_count), kThreads>>>(
          sequence_view.device_residues, sequence_view.device_offsets,
          profile_view.profiles, profile_view.emissions,
          profile_view.transitions, buffers.candidates,
          buffers.forward_offsets, kernel_forward_specials,
          buffers.dp_offsets, buffers.backward_offsets,
          buffers.posterior_offsets, active_count, rt1, rt2, rt3, guard_band,
          buffers.dp, buffers.backward_specials, buffers.posteriors,
          buffers.results, nullptr);
    }
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaEventRecord(end_event));
    CUDA_RUN(cudaEventSynchronize(end_event));
    CUDA_RUN(cudaEventElapsedTime(&created->statistics.kernel_milliseconds,
                                  begin_event, end_event));
    cudaEventDestroy(end_event);
    cudaEventDestroy(begin_event);
    end_event = nullptr;
    begin_event = nullptr;

    const auto download_begin = std::chrono::steady_clock::now();
    CUDA_RUN(cudaMemcpy(active_results.data(), buffers.results, result_bytes,
                        cudaMemcpyDeviceToHost));
    if (collect_reason_facts)
      CUDA_RUN(cudaMemcpy(active_reason_facts.data(), buffers.reason_facts,
                          reason_bytes, cudaMemcpyDeviceToHost));

    std::vector<uint64_t> active_region_offsets;
    std::vector<uint64_t> active_diagnostic_offsets;
    try {
      active_region_offsets.assign(active_count + 1, 0);
      active_diagnostic_offsets.assign(active_count + 1, 0);
    } catch (...) {
      if (end_event != nullptr) cudaEventDestroy(end_event);
      if (begin_event != nullptr) cudaEventDestroy(begin_event);
      free_device_buffers(&buffers);
      set_error(error, error_size,
                "Backward/domain compact output allocation failed");
      return -1;
    }

    uint64_t region_count = 0;
    uint64_t diagnostic_cells = 0;
    for (size_t active = 0; active < active_count; ++active) {
      plan7_backward_domain_result &result = active_results[active];
      if (unsealed_test) {
        result.route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
        result.region_count = 0;
      }
      active_region_offsets[active] = region_count;
      if (result.route == PLAN7_BACKWARD_DOMAIN_SIMPLE) {
        uint64_t next_regions;
        uint64_t next_region_bytes;
        if (!checked_add(region_count, result.region_count, &next_regions) ||
            !checked_multiply(next_regions, sizeof(plan7_simple_region),
                              &next_region_bytes) ||
            next_region_bytes > kSimpleRegionOutputByteLimit) {
          result.route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
          ++created->statistics.output_cap_fallback_count;
          if (collect_reason_facts)
            active_reason_facts[active] |=
                PLAN7_BACKWARD_DOMAIN_REASON_REGION_OUTPUT_CAP;
        } else {
          region_count = next_regions;
        }
      }
      active_region_offsets[active + 1] = region_count;

      active_diagnostic_offsets[active] = diagnostic_cells;
      if (result.status == PLAN7_BACKWARD_DOMAIN_OK) {
        const uint64_t row_cells = active_posterior_offsets[active + 1] -
                                   active_posterior_offsets[active];
        uint64_t next_cells;
        uint64_t next_bytes;
        if (checked_add(diagnostic_cells, row_cells, &next_cells) &&
            checked_multiply(next_cells, sizeof(plan7_domain_posterior),
                             &next_bytes) &&
            next_bytes <= created->statistics.output_byte_limit) {
          diagnostic_cells = next_cells;
        } else {
          ++created->statistics.posterior_omitted_count;
        }
      }
      active_diagnostic_offsets[active + 1] = diagnostic_cells;
    }

    size_t region_bytes = 0;
    size_t region_offset_bytes = 0;
    size_t resident_region_bytes = 0;
    if (!checked_bytes(static_cast<size_t>(region_count),
                       sizeof(plan7_simple_region), &region_bytes) ||
        !checked_bytes(static_cast<size_t>(region_count),
                       sizeof(plan7_backward_domain_resident_region),
                       &resident_region_bytes) ||
        !checked_bytes(active_count + 1, sizeof(uint64_t),
                       &region_offset_bytes) ||
        diagnostic_cells > SIZE_MAX) {
      free_device_buffers(&buffers);
      set_error(error, error_size,
                "Backward/domain compact output size overflow");
      return -1;
    }
    try {
      created->regions.resize(static_cast<size_t>(region_count));
      created->posteriors.resize(static_cast<size_t>(diagnostic_cells));
    } catch (...) {
      free_device_buffers(&buffers);
      set_error(error, error_size,
                "Backward/domain compact output allocation failed");
      return -1;
    }

    if (region_count != 0) {
      CUDA_RUN(cudaMalloc(&buffers.region_offsets, region_offset_bytes));
      CUDA_RUN(cudaMalloc(&buffers.regions, region_bytes));
      if (!unsealed_test) {
        created->residency_statistics.resident_region_requested_bytes =
            resident_region_bytes;
        const auto resident_allocation_begin =
            std::chrono::steady_clock::now();
        cuda_status = cudaMalloc(&buffers.resident_regions,
                                 resident_region_bytes);
        created->residency_statistics.
            resident_region_allocation_milliseconds =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() -
                resident_allocation_begin).count();
        if (cuda_status == cudaSuccess) {
          created->residency_statistics.resident_region_allocated_bytes =
              resident_region_bytes;
        } else {
          buffers.resident_regions = nullptr;
          ++created->residency_statistics.
              resident_region_allocation_fallback_count;
          cudaGetLastError();
        }
      }
      CUDA_RUN(cudaMemcpy(buffers.region_offsets,
                          active_region_offsets.data(), region_offset_bytes,
                          cudaMemcpyHostToDevice));
      const auto resident_materialization_begin =
          std::chrono::steady_clock::now();
      gather_simple_regions_kernel<<<static_cast<unsigned>(active_count), 32>>>(
          buffers.posteriors, buffers.posterior_offsets,
          buffers.region_offsets, sequence_view.device_offsets,
          buffers.candidates, buffers.results, active_count, rt1, rt2,
          buffers.regions, buffers.resident_regions);
      CUDA_RUN(cudaGetLastError());
      CUDA_RUN(cudaMemcpy(created->regions.data(), buffers.regions,
                          region_bytes, cudaMemcpyDeviceToHost));
      if (buffers.resident_regions != nullptr) {
        created->residency_statistics.resident_region_materialized_bytes =
            resident_region_bytes;
        created->residency_statistics.
            resident_region_materialization_milliseconds =
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() -
                resident_materialization_begin).count();
      }
    }
    for (size_t active = 0; active < active_count; ++active) {
      const uint64_t output_begin = active_diagnostic_offsets[active];
      const uint64_t output_end = active_diagnostic_offsets[active + 1];
      if (output_begin == output_end) continue;
      const uint64_t input_begin = active_posterior_offsets[active];
      const size_t row_bytes = static_cast<size_t>(output_end - output_begin) *
                               sizeof(plan7_domain_posterior);
      CUDA_RUN(cudaMemcpy(created->posteriors.data() + output_begin,
                          buffers.posteriors + input_begin, row_bytes,
                          cudaMemcpyDeviceToHost));
    }

    for (size_t active = 0; active < active_count; ++active)
      created->results[active_sources[active]] = active_results[active];
    if (collect_reason_facts && !merge_backward_reason_facts(
            active_sources.data(), active_reason_facts.data(), active_count,
            created->reason_facts.data(), created->reason_facts.size())) {
      free_device_buffers(&buffers);
      set_error(error, error_size,
                "Backward/domain reason source remap changed");
      return -1;
    }
    size_t active = 0;
    uint64_t posterior_offset = 0;
    uint64_t region_offset = 0;
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
      created->posterior_offsets[candidate] = posterior_offset;
      created->region_offsets[candidate] = region_offset;
      if (active < active_count && active_sources[active] == candidate) {
        posterior_offset = active_diagnostic_offsets[active + 1];
        region_offset = active_region_offsets[active + 1];
        ++active;
      }
    }
    created->posterior_offsets[candidate_count] = posterior_offset;
    created->region_offsets[candidate_count] = region_offset;
    if (buffers.resident_regions != nullptr) {
      created->resident_regions.pointer = buffers.resident_regions;
      created->resident_regions.device_ordinal = current_device;
      created->resident_regions.database_generation =
          provenance_snapshot.database_generation;
      created->resident_regions.batch_generation =
          provenance_snapshot.batch_generation;
      created->resident_regions.row_count = candidate_count;
      created->resident_regions.region_count = region_count;
      buffers.resident_regions = nullptr;
    }
    created->statistics.download_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - download_begin).count();
#undef CUDA_RUN
    free_device_buffers(&buffers);
  }

  for (size_t candidate = 0; candidate < created->results.size(); ++candidate) {
    const plan7_backward_domain_result &result = created->results[candidate];
    if (result.has_own_scales) ++created->statistics.own_scale_count;
    if (result.uncertain_count != 0)
      ++created->statistics.threshold_uncertain_count;
    if (result.route == PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED) {
      ++created->statistics.cpu_required_count;
      if (collect_reason_facts) {
        constexpr uint32_t explained = UINT32_C(0x0000ffff);
        if ((created->reason_facts[candidate] & explained) == 0)
          created->reason_facts[candidate] |=
              PLAN7_BACKWARD_DOMAIN_REASON_OTHER_CPU_REQUIRED;
        created->reason_facts[candidate] |=
            PLAN7_BACKWARD_DOMAIN_REASON_FINAL_CPU_REQUIRED;
      }
      if (result.multidomain_count != 0)
        ++created->statistics.multidomain_fallback_count;
    } else {
      ++created->statistics.device_result_count;
      if (result.route == PLAN7_BACKWARD_DOMAIN_NO_REGIONS) {
        ++created->statistics.no_region_count;
        if (collect_reason_facts)
          created->reason_facts[candidate] |=
              PLAN7_BACKWARD_DOMAIN_REASON_NO_REGIONS;
      } else if (result.route == PLAN7_BACKWARD_DOMAIN_SIMPLE) {
        ++created->statistics.simple_count;
        if (collect_reason_facts)
          created->reason_facts[candidate] |=
              PLAN7_BACKWARD_DOMAIN_REASON_SIMPLE;
      }
    }
  }
  created->statistics.work_cells = work_cells;
  created->statistics.dp_workspace_bytes = dp_bytes;
  created->statistics.backward_special_workspace_bytes = backward_bytes;
  created->statistics.forward_special_workspace_bytes = forward_bytes;
  created->statistics.posterior_bytes =
      static_cast<uint64_t>(created->posteriors.size()) *
      sizeof(plan7_domain_posterior);
  created->statistics.simple_region_bytes =
      static_cast<uint64_t>(created->regions.size()) *
      sizeof(plan7_simple_region);
  created->statistics.total_milliseconds =
      std::chrono::duration<float, std::milli>(
          std::chrono::steady_clock::now() - total_begin).count();
  if (unsealed_test) {
    created->provenance = {};
  } else {
    if (!seal_backward_domain_provenance(
            created.get(), provenance_snapshot, rt1, rt2, rt3, guard_band)) {
      set_error(error, error_size,
                "Backward/domain output provenance sealing failed");
      return -1;
    }
    created->sealed = true;
    if (created->resident_regions.pointer != nullptr) {
      created->resident_regions.result_hash = created->provenance.result_hash;
      created->resident_regions.region_hash = created->provenance.region_hash;
      created->resident_regions.valid = true;
    }
  }
  *output = created.release();
  return 0;
}

}  // namespace

extern "C" int plan7_backward_domain_run(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const plan7_forward_provenance *provenance,
    const uint64_t *forward_offsets,
    const float *forward_specials, size_t forward_special_count,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  try {
    return backward_domain_run_impl(
        database, batch, candidates, candidate_count, provenance,
        forward_offsets, forward_specials, forward_special_count, nullptr,
        rt1, rt2, rt3, guard_band, posterior_byte_budget, false, false,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "Backward/domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size, "Backward/domain unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_backward_domain_run_with_reason_facts(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const plan7_forward_provenance *provenance,
    const uint64_t *forward_offsets,
    const float *forward_specials, size_t forward_special_count,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  try {
    return backward_domain_run_impl(
        database, batch, candidates, candidate_count, provenance,
        forward_offsets, forward_specials, forward_special_count, nullptr,
        rt1, rt2, rt3, guard_band, posterior_byte_budget, false, true,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "Backward/domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size, "Backward/domain unexpected native failure");
    return -1;
  }
}

namespace {

int backward_domain_run_from_forward_output_impl(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const uint64_t *forward_offsets,
    const plan7_forward_output *forward_output,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget, bool collect_reason_facts,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  if (forward_output == nullptr) {
    set_error(error, error_size,
              "Backward/domain resident Forward output is null");
    return -1;
  }
  plan7_forward_resident_view resident_view{};
  const int resident_status = plan7_forward_output_get_resident_view(
      forward_output, &resident_view, error, error_size);
  if (resident_status != 1) {
    if (resident_status == 0)
      set_error(error, error_size,
                "Backward/domain resident Forward output is unavailable");
    return -1;
  }
  const plan7_forward_provenance *provenance =
      plan7_forward_output_provenance(forward_output);
  const float *host_specials =
      plan7_forward_output_specials(forward_output);
  const size_t special_count =
      plan7_forward_output_special_count(forward_output);
  return backward_domain_run_impl(
      database, batch, candidates, candidate_count, provenance,
      forward_offsets, host_specials, special_count, &resident_view,
      rt1, rt2, rt3, guard_band, posterior_byte_budget, false,
      collect_reason_facts, output, error, error_size);
}

}  // namespace

extern "C" int plan7_backward_domain_run_from_forward_output(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const uint64_t *forward_offsets,
    const plan7_forward_output *forward_output,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  try {
    return backward_domain_run_from_forward_output_impl(
        database, batch, candidates, candidate_count, forward_offsets,
        forward_output, rt1, rt2, rt3, guard_band, posterior_byte_budget,
        false, output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "Backward/domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size, "Backward/domain unexpected native failure");
    return -1;
  }
}

extern "C" int
plan7_backward_domain_run_from_forward_output_with_reason_facts(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const uint64_t *forward_offsets,
    const plan7_forward_output *forward_output,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  try {
    return backward_domain_run_from_forward_output_impl(
        database, batch, candidates, candidate_count, forward_offsets,
        forward_output, rt1, rt2, rt3, guard_band, posterior_byte_budget,
        true, output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "Backward/domain host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size, "Backward/domain unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_backward_domain_route_all_cpu_from_forward_output(
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const uint64_t *forward_offsets,
    const plan7_forward_output *forward_output,
    float rt1, float rt2, float rt3, float guard_band,
    int collect_reason_facts,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  const auto total_begin = std::chrono::steady_clock::now();
  try {
    if (output == nullptr || *output != nullptr || forward_output == nullptr ||
        forward_offsets == nullptr ||
        (candidate_count != 0 && candidates == nullptr) ||
        !valid_thresholds(rt1, rt2, rt3, guard_band) ||
        collect_reason_facts < 0 || collect_reason_facts > 1) {
      set_error(error, error_size,
                "invalid CPU-domain ownership arguments");
      return -1;
    }
    const plan7_forward_provenance *provenance =
        plan7_forward_output_provenance(forward_output);
    const size_t special_count =
        plan7_forward_output_special_count(forward_output);
    if (provenance == nullptr || provenance->pass_count != candidate_count ||
        provenance->special_count != special_count ||
        forward_offsets[0] != 0 ||
        forward_offsets[candidate_count] != special_count) {
      set_error(error, error_size,
                "CPU-domain ownership Forward provenance differs");
      return -1;
    }

    std::unique_ptr<plan7_backward_domain_output> created(
        new (std::nothrow) plan7_backward_domain_output{});
    if (!created) {
      set_error(error, error_size,
                "CPU-domain ownership output allocation failed");
      return -1;
    }
    created->rt1 = rt1;
    created->rt2 = rt2;
    created->rt3 = rt3;
    created->guard_band = guard_band;
    created->results.resize(candidate_count);
    created->posterior_offsets.assign(candidate_count + 1, 0);
    created->region_offsets.assign(candidate_count + 1, 0);
    if (collect_reason_facts)
      created->reason_facts.assign(
          candidate_count,
          PLAN7_BACKWARD_DOMAIN_REASON_OTHER_CPU_REQUIRED |
              PLAN7_BACKWARD_DOMAIN_REASON_FINAL_CPU_REQUIRED);

    uint32_t previous_profile = 0;
    uint32_t previous_sequence = 0;
    bool have_previous = false;
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
      const auto row = candidates[candidate];
      if (forward_offsets[candidate] > forward_offsets[candidate + 1] ||
          (have_previous &&
           (row.profile_index < previous_profile ||
            (row.profile_index == previous_profile &&
             row.sequence_index <= previous_sequence)))) {
        set_error(error, error_size,
                  "CPU-domain ownership rows are not canonical");
        return -1;
      }
      have_previous = true;
      previous_profile = row.profile_index;
      previous_sequence = row.sequence_index;
      auto &result = created->results[candidate];
      result = {};
      result.profile_index = row.profile_index;
      result.sequence_index = row.sequence_index;
      result.backward_score = NAN;
      result.nexpected = NAN;
      result.status = PLAN7_BACKWARD_DOMAIN_OK;
      result.route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
    }
    created->statistics.candidate_count = candidate_count;
    created->statistics.cpu_required_count = candidate_count;
    created->statistics.output_byte_limit = 0;
    created->statistics.total_milliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - total_begin).count();
    if (!seal_backward_domain_provenance(
            created.get(), *provenance, rt1, rt2, rt3, guard_band)) {
      set_error(error, error_size,
                "CPU-domain ownership provenance sealing failed");
      return -1;
    }
    created->sealed = true;
    *output = created.release();
    return 0;
  } catch (const std::bad_alloc &) {
    set_error(error, error_size,
              "CPU-domain ownership output allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "CPU-domain ownership unexpected native failure");
    return -1;
  }
}

extern "C" int plan7_backward_domain_unsealed_test_run(
    const plan7_forward_database *database,
    const plan7_ssv_sequence_batch *batch,
    const plan7_backward_domain_candidate *candidates,
    size_t candidate_count, const uint64_t *forward_offsets,
    const float *forward_specials, size_t forward_special_count,
    float rt1, float rt2, float rt3, float guard_band,
    uint64_t posterior_byte_budget,
    plan7_backward_domain_output **output,
    char *error, size_t error_size) {
  try {
    return backward_domain_run_impl(
        database, batch, candidates, candidate_count, nullptr,
        forward_offsets, forward_specials, forward_special_count, nullptr,
        rt1, rt2, rt3, guard_band, posterior_byte_budget, true, false,
        output, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size,
              "Backward/domain test host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size,
              "Backward/domain unexpected test failure");
    return -1;
  }
}

extern "C" int plan7_backward_domain_output_destroy(
    plan7_backward_domain_output **output, char *error, size_t error_size) {
  if (output == nullptr) {
    set_error(error, error_size, "Backward/domain output handle is null");
    return -1;
  }
  delete *output;
  *output = nullptr;
  return 0;
}

extern "C" size_t plan7_backward_domain_output_result_count(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? 0 : output->results.size();
}

extern "C" const plan7_backward_domain_result *
plan7_backward_domain_output_results(
    const plan7_backward_domain_output *output) {
  return output == nullptr || output->results.empty()
             ? nullptr : output->results.data();
}

extern "C" size_t plan7_backward_domain_output_reason_count(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? 0 : output->reason_facts.size();
}

extern "C" const uint32_t *plan7_backward_domain_output_reason_facts(
    const plan7_backward_domain_output *output) {
  return output == nullptr || output->reason_facts.empty()
             ? nullptr : output->reason_facts.data();
}

extern "C" int plan7_backward_domain_merge_reason_facts_for_test(
    const uint64_t *active_sources, const uint32_t *active_facts,
    size_t active_count, uint32_t *source_facts, size_t source_count) {
  return merge_backward_reason_facts(
             active_sources, active_facts, active_count,
             source_facts, source_count)
             ? 0
             : -1;
}

extern "C" const uint64_t *
plan7_backward_domain_output_posterior_offsets(
    const plan7_backward_domain_output *output) {
  return output == nullptr || output->posterior_offsets.empty()
             ? nullptr : output->posterior_offsets.data();
}

extern "C" size_t plan7_backward_domain_output_posterior_count(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? 0 : output->posteriors.size();
}

extern "C" const plan7_domain_posterior *
plan7_backward_domain_output_posteriors(
    const plan7_backward_domain_output *output) {
  return output == nullptr || output->posteriors.empty()
             ? nullptr : output->posteriors.data();
}

extern "C" const uint64_t *
plan7_backward_domain_output_region_offsets(
    const plan7_backward_domain_output *output) {
  return output == nullptr || output->region_offsets.empty()
             ? nullptr : output->region_offsets.data();
}

extern "C" size_t plan7_backward_domain_output_region_count(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? 0 : output->regions.size();
}

extern "C" const plan7_simple_region *
plan7_backward_domain_output_regions(
    const plan7_backward_domain_output *output) {
  return output == nullptr || output->regions.empty()
             ? nullptr : output->regions.data();
}

extern "C" const plan7_backward_domain_provenance *
plan7_backward_domain_output_provenance(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? nullptr : &output->provenance;
}

extern "C" const plan7_backward_domain_statistics *
plan7_backward_domain_output_statistics(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? nullptr : &output->statistics;
}

extern "C" const plan7_backward_domain_residency_statistics *
plan7_backward_domain_output_residency_statistics(
    const plan7_backward_domain_output *output) {
  return output == nullptr ? nullptr : &output->residency_statistics;
}

extern "C" int plan7_backward_domain_output_get_resident_view(
    const plan7_backward_domain_output *output,
    plan7_backward_domain_resident_view *view,
    char *error, size_t error_size) {
  if (output == nullptr || view == nullptr) {
    set_error(error, error_size,
              "invalid Backward/domain resident view request");
    return -1;
  }
  *view = {};
  if (output->resident_regions.pointer == nullptr ||
      !output->resident_regions.valid)
    return 0;
  const ResidentBackwardRegions &resident = output->resident_regions;
  const uint64_t expected_bytes =
      resident.region_count * sizeof(plan7_backward_domain_resident_region);
  if (!output->sealed || resident.device_ordinal < 0 ||
      resident.database_generation !=
          output->provenance.forward.database_generation ||
      resident.batch_generation !=
          output->provenance.forward.batch_generation ||
      resident.result_hash != output->provenance.result_hash ||
      resident.region_hash != output->provenance.region_hash ||
      resident.row_count != output->provenance.candidate_count ||
      resident.region_count != output->provenance.region_count ||
      resident.region_count != output->regions.size() ||
      output->residency_statistics.resident_region_allocated_bytes !=
          expected_bytes ||
      output->residency_statistics.resident_region_materialized_bytes !=
          expected_bytes ||
      expected_bytes >
          output->residency_statistics.resident_region_requested_bytes) {
    set_error(error, error_size,
              "Backward/domain resident generation is incomplete");
    return -1;
  }
  view->database_generation = resident.database_generation;
  view->batch_generation = resident.batch_generation;
  view->result_hash = resident.result_hash;
  view->region_hash = resident.region_hash;
  view->row_count = resident.row_count;
  view->region_count = resident.region_count;
  view->device_ordinal = resident.device_ordinal;
  view->regions = resident.pointer;
  return 1;
}

extern "C" int plan7_backward_domain_output_is_production_calibrated(
    const plan7_backward_domain_output *output) {
  if (output == nullptr || !output->sealed || output->rt1 != 0.25f ||
      output->rt2 != 0.10f || output->rt3 != 0.20f ||
      !std::isfinite(output->guard_band) || output->guard_band < 2.0e-4f)
    return 0;
  uint64_t threshold_hash =
      hash_u64(kHashOffset, UINT64_C(0x54485245));
  for (const float value : {
           output->rt1, output->rt2, output->rt3, output->guard_band}) {
    FloatBits bits{};
    bits.value = value;
    threshold_hash = hash_u32(threshold_hash, bits.bits);
  }
  return output->provenance.threshold_hash == hash_u64(
      threshold_hash, PLAN7_BACKWARD_DOMAIN_RECORD_VERSION);
}

extern "C" int plan7_backward_domain_output_apply_test_fault(
    plan7_backward_domain_output *output, int fault,
    char *error, size_t error_size) {
  if (output == nullptr || !output->sealed) {
    set_error(error, error_size,
              "Backward/domain test fault requires a sealed output");
    return -1;
  }
  switch (fault) {
    case PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH:
      output->provenance.result_hash ^= UINT64_C(1);
      return 0;
    case PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH:
      output->provenance.threshold_hash ^= UINT64_C(1);
      return 0;
    case PLAN7_BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE:
      for (auto &result : output->results) {
        if (result.route != PLAN7_BACKWARD_DOMAIN_SIMPLE) continue;
        if (!result.has_own_scales) {
          result.has_own_scales = 1;
          ++output->statistics.own_scale_count;
        }
        if (!seal_backward_domain_provenance(
                output, output->provenance.forward,
                output->rt1, output->rt2, output->rt3,
                output->guard_band)) {
          set_error(error, error_size,
                    "Backward/domain own-scale test reseal failed");
          return -1;
        }
        /* This test intentionally mutates a sealed host row after the device
         * descriptor generation was materialized.  Keep the old allocation
         * for ordinary ownership cleanup, but force the consumer through the
         * exact legacy host replay. */
        output->resident_regions.valid = false;
        return 0;
      }
      set_error(error, error_size,
                "Backward/domain test output has no SIMPLE row");
      return -1;
    default:
      set_error(error, error_size,
                "unknown Backward/domain test fault");
      return -1;
  }
}

namespace {

int backward_domain_cpu_oracle_impl(
    uintptr_t source_profile_pointer,
    const uint8_t *residues, size_t residue_count,
    const float *forward_specials, size_t forward_special_count,
    float rt1, float rt2, float rt3, float guard_band,
    plan7_backward_domain_result *result,
    plan7_domain_posterior *posteriors, size_t posterior_count,
    plan7_simple_region *regions, size_t region_capacity,
    size_t *region_count,
    char *error, size_t error_size) {
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      source_profile_pointer);
  if (source == nullptr || residues == nullptr || residue_count == 0 ||
      forward_specials == nullptr || result == nullptr ||
      posteriors == nullptr || region_count == nullptr ||
      (region_capacity != 0 && regions == nullptr) ||
      residue_count > kMaximumTargetLength ||
      posterior_count != residue_count + 1 ||
      forward_special_count != (residue_count + 1) * p7X_NXCELLS ||
      !valid_thresholds(rt1, rt2, rt3, guard_band) ||
      source->abc == nullptr || source->abc->type != eslAMINO ||
      source->M < 1 || source->M > static_cast<int>(kMaximumModelLength) ||
      !p7_oprofile_IsLocal(source)) {
    set_error(error, error_size, "invalid Backward/domain CPU oracle input");
    return -1;
  }
  *region_count = 0;
  for (size_t i = 0; i < residue_count; ++i)
    if (residues[i] >= static_cast<uint8_t>(source->abc->Kp)) {
      set_error(error, error_size, "CPU oracle residue is out of range");
      return -1;
    }
  for (size_t i = 0; i < forward_special_count; ++i)
    if (!std::isfinite(forward_specials[i])) {
      set_error(error, error_size, "CPU oracle Forward state is not finite");
      return -1;
    }

  std::vector<ESL_DSQ> dsq;
  std::vector<float> btot;
  std::vector<float> etot;
  std::vector<float> mocc;
  try {
    dsq.resize(residue_count + 2, eslDSQ_SENTINEL);
    btot.resize(residue_count + 1);
    etot.resize(residue_count + 1);
    mocc.resize(residue_count + 1);
  } catch (...) {
    set_error(error, error_size, "CPU oracle host allocation failed");
    return -1;
  }

  P7_OPROFILE *profile = p7_oprofile_Clone(source);
  P7_OMX *forward = nullptr;
  P7_OMX *backward = nullptr;
  if (profile == nullptr ||
      p7_oprofile_ReconfigLength(
          profile, static_cast<int>(residue_count)) != eslOK ||
      (forward = p7_omx_Create(profile->M, 0,
                               static_cast<int>(residue_count))) == nullptr ||
      (backward = p7_omx_Create(profile->M, 0,
                                static_cast<int>(residue_count))) == nullptr) {
    p7_omx_Destroy(backward);
    p7_omx_Destroy(forward);
    p7_oprofile_Destroy(profile);
    set_error(error, error_size, "CPU oracle workspace allocation failed");
    return -1;
  }
  std::copy(residues, residues + residue_count, dsq.begin() + 1);
  forward->M = profile->M;
  forward->L = static_cast<int>(residue_count);
  forward->has_own_scales = TRUE;
  std::memcpy(forward->xmx, forward_specials,
              forward_special_count * sizeof(float));
  float backward_score = NAN;
  const int status = p7_BackwardParser(
      dsq.data(), static_cast<int>(residue_count), profile,
      forward, backward, &backward_score);
  *result = {};
  result->profile_index = 0;
  result->sequence_index = 0;
  result->backward_score = backward_score;
  result->nexpected = NAN;
  result->status = static_cast<uint8_t>(status);
  result->route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
  result->has_own_scales = backward->has_own_scales ? 1 : 0;
  int decoding_status = status;
  if (status == eslOK) {
    P7_DOMAINDEF ddef{};
    ddef.btot = btot.data();
    ddef.etot = etot.data();
    ddef.mocc = mocc.data();
    ddef.Lalloc = static_cast<int>(residue_count);
    decoding_status = p7_DomainDecoding(profile, forward, backward, &ddef);
    bool finite = decoding_status == eslOK;
    for (size_t i = 0; i <= residue_count; ++i) {
      posteriors[i] = {btot[i], etot[i], mocc[i]};
      finite = finite && std::isfinite(btot[i]) &&
               std::isfinite(etot[i]) && std::isfinite(mocc[i]);
    }
    if (finite) {
      classify_regions_host(posteriors, residue_count, rt1, rt2, rt3,
                            guard_band, &result->uncertain_count,
                            &result->region_count,
                            &result->multidomain_count,
                            regions, region_capacity);
      *region_count = result->region_count;
      result->nexpected = posteriors[residue_count].btot;
      result->status = PLAN7_BACKWARD_DOMAIN_OK;
      if (result->uncertain_count == 0 &&
          result->multidomain_count == 0)
        result->route = result->region_count == 0
                            ? PLAN7_BACKWARD_DOMAIN_NO_REGIONS
                            : PLAN7_BACKWARD_DOMAIN_SIMPLE;
      if (result->region_count > region_capacity) {
        decoding_status = eslERANGE;
        result->status = PLAN7_BACKWARD_DOMAIN_ERANGE;
        result->route = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED;
      }
    } else {
      result->status = PLAN7_BACKWARD_DOMAIN_ERANGE;
    }
  }
  p7_omx_Destroy(backward);
  p7_omx_Destroy(forward);
  p7_oprofile_Destroy(profile);
  return decoding_status == eslOK ? 0 : decoding_status;
}

}  // namespace

extern "C" int plan7_backward_domain_cpu_oracle(
    uintptr_t source_profile_pointer,
    const uint8_t *residues, size_t residue_count,
    const float *forward_specials, size_t forward_special_count,
    float rt1, float rt2, float rt3, float guard_band,
    plan7_backward_domain_result *result,
    plan7_domain_posterior *posteriors, size_t posterior_count,
    plan7_simple_region *regions, size_t region_capacity,
    size_t *region_count,
    char *error, size_t error_size) {
  try {
    return backward_domain_cpu_oracle_impl(
        source_profile_pointer, residues, residue_count,
        forward_specials, forward_special_count,
        rt1, rt2, rt3, guard_band, result,
        posteriors, posterior_count, regions, region_capacity,
        region_count, error, error_size);
  } catch (const std::bad_alloc &) {
    set_error(error, error_size, "CPU oracle host allocation failed");
    return -1;
  } catch (...) {
    set_error(error, error_size, "CPU oracle unexpected native failure");
    return -1;
  }
}
