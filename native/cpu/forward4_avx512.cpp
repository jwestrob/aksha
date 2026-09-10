#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

extern "C" {
#include "easel.h"
#include "esl_alphabet.h"
#include "hmmer.h"
#include "impl_sse/impl_sse.h"
}

#include "avx512_tail.h"

namespace {

constexpr int kCandidates = 4;

struct Forward4Result {
  int M = 0;
  int L = 0;
  int Q = 0;
  std::vector<__m512> dp;
  std::array<std::vector<float>, kCandidates> xmx;
  std::array<float, kCandidates> totscale{};
  std::array<float, kCandidates> score{};
  std::array<int, kCandidates> status{};
  std::array<int, kCandidates> has_own_scales{};

  void resize(int model_length, int sequence_length) {
    M = model_length;
    L = sequence_length;
    Q = p7O_NQF(M);
    dp.assign(static_cast<size_t>(Q) * p7X_NSCELLS,
              _mm512_setzero_ps());
    for (auto &row : xmx) {
      row.assign(static_cast<size_t>(L + 1) * p7X_NXCELLS, 0.0f);
    }
  }
};

thread_local Forward4Result forward_output;
thread_local Forward4Result backward_output;

std::atomic<uint64_t> tail_madvise_calls{0};
std::atomic<uint64_t> tail_madvise_released_bytes{0};

bool madvise_full_pages(void *allocation, size_t allocation_bytes,
                        size_t page_size, uint64_t *released_bytes) {
  if (allocation == nullptr || allocation_bytes < page_size) return true;
  const uintptr_t start = reinterpret_cast<uintptr_t>(allocation);
  const uintptr_t first =
      ((start + page_size - 1u) / page_size) * page_size;
  const uintptr_t end =
      ((start + allocation_bytes) / page_size) * page_size;
  if (end <= first) return true;
  if (madvise(reinterpret_cast<void *>(first),
              static_cast<size_t>(end - first), MADV_DONTNEED) != 0) {
    return false;
  }
  *released_bytes += static_cast<uint64_t>(end - first);
  return true;
}

bool madvise_result_pages(Forward4Result *result, size_t page_size,
                          uint64_t *released_bytes) {
  bool ok = madvise_full_pages(
      result->dp.data(), result->dp.capacity() * sizeof(__m512), page_size,
      released_bytes);
  for (auto &row : result->xmx) {
    ok = madvise_full_pages(row.data(), row.capacity() * sizeof(float),
                            page_size, released_bytes) && ok;
  }
  return ok;
}

inline __m512 pack4(__m128 a, __m128 b, __m128 c, __m128 d) {
  __m512 value = _mm512_castps128_ps512(a);
  value = _mm512_insertf32x4(value, b, 1);
  value = _mm512_insertf32x4(value, c, 2);
  return _mm512_insertf32x4(value, d, 3);
}

inline __m512 broadcast128(__m128 value) {
  return _mm512_broadcast_f32x4(value);
}

inline __m512 repeat_candidate_lanes(__m128 value) {
  const __m512i indexes = _mm512_setr_epi32(
      0, 0, 0, 0, 1, 1, 1, 1,
      2, 2, 2, 2, 3, 3, 3, 3);
  return _mm512_permutexvar_ps(indexes, broadcast128(value));
}

inline __m512 rightshiftz_quarters(__m512 value) {
  return _mm512_castsi512_ps(
      _mm512_bslli_epi128(_mm512_castps_si512(value), 4));
}

inline __m512 leftshiftz_quarters(__m512 value) {
  return _mm512_castsi512_ps(
      _mm512_bsrli_epi128(_mm512_castps_si512(value), 4));
}

inline __mmask16 expand_candidate_mask(unsigned mask4) {
  unsigned mask16 = 0;
  for (int c = 0; c < kCandidates; ++c) {
    if ((mask4 & (1u << c)) != 0) mask16 |= 0xFu << (4 * c);
  }
  return static_cast<__mmask16>(mask16);
}

inline unsigned collapse_changed_mask(__mmask16 mask16) {
  unsigned mask4 = 0;
  for (int c = 0; c < kCandidates; ++c) {
    if ((static_cast<unsigned>(mask16) & (0xFu << (4 * c))) != 0) {
      mask4 |= 1u << c;
    }
  }
  return mask4;
}

inline __m128 candidate_mask_ps(unsigned mask4) {
  return _mm_castsi128_ps(_mm_setr_epi32(
      (mask4 & 0x1u) ? -1 : 0,
      (mask4 & 0x2u) ? -1 : 0,
      (mask4 & 0x4u) ? -1 : 0,
      (mask4 & 0x8u) ? -1 : 0));
}

inline __m128 quarter_sums(__m512 value) {
  value = _mm512_add_ps(
      value,
      _mm512_shuffle_ps(value, value, _MM_SHUFFLE(0, 3, 2, 1)));
  value = _mm512_add_ps(
      value,
      _mm512_shuffle_ps(value, value, _MM_SHUFFLE(1, 0, 3, 2)));
  alignas(64) float lanes[16];
  _mm512_store_ps(lanes, value);
  return _mm_setr_ps(lanes[0], lanes[4], lanes[8], lanes[12]);
}

inline void store_special(std::array<std::vector<float>, kCandidates> &xmx,
                          int row, int state, __m128 value) {
  alignas(16) float lanes[kCandidates];
  _mm_store_ps(lanes, value);
  for (int c = 0; c < kCandidates; ++c) {
    xmx[c][static_cast<size_t>(row) * p7X_NXCELLS + state] = lanes[c];
  }
}

inline void store_special_rows(
    std::array<std::vector<float>, kCandidates> &xmx,
    const std::array<int, kCandidates> &rows,
    unsigned active4,
    int state,
    __m128 value) {
  alignas(16) float lanes[kCandidates];
  _mm_store_ps(lanes, value);
  for (int c = 0; c < kCandidates; ++c) {
    if ((active4 & (1u << c)) != 0) {
      xmx[c][static_cast<size_t>(rows[c]) * p7X_NXCELLS + state] = lanes[c];
    }
  }
}

__attribute__((noinline)) int forward4_parser(
    const std::array<const ESL_DSQ *, kCandidates> &dsq,
    int L,
    const P7_OPROFILE *om,
    Forward4Result *out) {
  if (out == nullptr || om == nullptr || L < 0) return eslEINVAL;
  if (!p7_oprofile_IsLocal(om)) return eslEINVAL;
  if (out->M != om->M || out->L != L) out->resize(om->M, L);

  const int Q = p7O_NQF(om->M);
  auto &dp = out->dp;
  const __m512 zero = _mm512_setzero_ps();
  std::fill(dp.begin(), dp.end(), zero);

  __m128 xE = _mm_setzero_ps();
  __m128 xN = _mm_set1_ps(1.0f);
  __m128 xJ = _mm_setzero_ps();
  __m128 xB = _mm_set1_ps(om->xf[p7O_N][p7O_MOVE]);
  __m128 xC = _mm_setzero_ps();
  out->totscale.fill(0.0f);
  out->status.fill(eslOK);
  out->has_own_scales.fill(1);

  store_special(out->xmx, 0, p7X_E, xE);
  store_special(out->xmx, 0, p7X_N, xN);
  store_special(out->xmx, 0, p7X_J, xJ);
  store_special(out->xmx, 0, p7X_B, xB);
  store_special(out->xmx, 0, p7X_C, xC);
  store_special(out->xmx, 0, p7X_SCALE, _mm_set1_ps(1.0f));

  for (int i = 1; i <= L; ++i) {
    const __m128 *rp[kCandidates] = {
        om->rfv[dsq[0][i]], om->rfv[dsq[1][i]],
        om->rfv[dsq[2][i]], om->rfv[dsq[3][i]]};
    const __m128 *tp = om->tfv;
    __m512 dcv = zero;
    __m512 xEv = zero;
    const __m512 xBv = repeat_candidate_lanes(xB);

    __m512 mpv = rightshiftz_quarters(
        dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_M]);
    __m512 dpv = rightshiftz_quarters(
        dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_D]);
    __m512 ipv = rightshiftz_quarters(
        dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_I]);

    for (int q = 0; q < Q; ++q) {
      __m512 sv = _mm512_mul_ps(xBv, broadcast128(*tp++));
      sv = _mm512_add_ps(sv, _mm512_mul_ps(mpv, broadcast128(*tp++)));
      sv = _mm512_add_ps(sv, _mm512_mul_ps(ipv, broadcast128(*tp++)));
      sv = _mm512_add_ps(sv, _mm512_mul_ps(dpv, broadcast128(*tp++)));
      const __m512 emission = pack4(rp[0][q], rp[1][q], rp[2][q], rp[3][q]);
      sv = _mm512_mul_ps(sv, emission);
      xEv = _mm512_add_ps(xEv, sv);

      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      mpv = dp[base + p7X_M];
      dpv = dp[base + p7X_D];
      ipv = dp[base + p7X_I];
      dp[base + p7X_M] = sv;
      dp[base + p7X_D] = dcv;

      dcv = _mm512_mul_ps(sv, broadcast128(*tp++));
      sv = _mm512_mul_ps(mpv, broadcast128(*tp++));
      dp[base + p7X_I] = _mm512_add_ps(
          sv, _mm512_mul_ps(ipv, broadcast128(*tp++)));
    }

    dcv = rightshiftz_quarters(dcv);
    dp[p7X_D] = zero;
    tp = om->tfv + 7 * Q;
    for (int q = 0; q < Q; ++q) {
      const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
      dp[index] = _mm512_add_ps(dcv, dp[index]);
      dcv = _mm512_mul_ps(dp[index], broadcast128(*tp++));
    }

    if (om->M < 100) {
      for (int j = 1; j < 4; ++j) {
        dcv = rightshiftz_quarters(dcv);
        tp = om->tfv + 7 * Q;
        for (int q = 0; q < Q; ++q) {
          const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
          dp[index] = _mm512_add_ps(dcv, dp[index]);
          dcv = _mm512_mul_ps(dcv, broadcast128(*tp++));
        }
      }
    } else {
      unsigned active4 = 0xFu;
      for (int j = 1; j < 4 && active4 != 0; ++j) {
        const __mmask16 active16 = expand_candidate_mask(active4);
        dcv = _mm512_maskz_mov_ps(active16, rightshiftz_quarters(dcv));
        tp = om->tfv + 7 * Q;
        __mmask16 changed16 = 0;
        for (int q = 0; q < Q; ++q) {
          const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
          const __m512 old = dp[index];
          const __m512 sum = _mm512_add_ps(dcv, old);
          changed16 |= _mm512_mask_cmp_ps_mask(
              active16, sum, old, _CMP_GT_OQ);
          dp[index] = _mm512_mask_mov_ps(old, active16, sum);
          dcv = _mm512_maskz_mov_ps(
              active16,
              _mm512_mul_ps(dcv, broadcast128(*tp++)));
        }
        active4 = collapse_changed_mask(changed16);
      }
    }

    for (int q = 0; q < Q; ++q) {
      xEv = _mm512_add_ps(
          dp[static_cast<size_t>(q) * p7X_NSCELLS + p7X_D], xEv);
    }
    xE = quarter_sums(xEv);

    xN = _mm_mul_ps(xN, _mm_set1_ps(om->xf[p7O_N][p7O_LOOP]));
    xC = _mm_add_ps(
        _mm_mul_ps(xC, _mm_set1_ps(om->xf[p7O_C][p7O_LOOP])),
        _mm_mul_ps(xE, _mm_set1_ps(om->xf[p7O_E][p7O_MOVE])));
    xJ = _mm_add_ps(
        _mm_mul_ps(xJ, _mm_set1_ps(om->xf[p7O_J][p7O_LOOP])),
        _mm_mul_ps(xE, _mm_set1_ps(om->xf[p7O_E][p7O_LOOP])));
    xB = _mm_add_ps(
        _mm_mul_ps(xJ, _mm_set1_ps(om->xf[p7O_J][p7O_MOVE])),
        _mm_mul_ps(xN, _mm_set1_ps(om->xf[p7O_N][p7O_MOVE])));

    const __m128 scale_mask = _mm_cmpgt_ps(xE, _mm_set1_ps(1.0e4f));
    const unsigned scale_bits = static_cast<unsigned>(_mm_movemask_ps(scale_mask));
    __m128 scale_row = _mm_set1_ps(1.0f);
    if (scale_bits != 0) {
      alignas(16) float xE_lanes[kCandidates];
      alignas(16) float xN_lanes[kCandidates];
      alignas(16) float xC_lanes[kCandidates];
      alignas(16) float xJ_lanes[kCandidates];
      alignas(16) float xB_lanes[kCandidates];
      alignas(16) float inverse_lanes[kCandidates];
      alignas(16) float scale_lanes[kCandidates];
      _mm_store_ps(xE_lanes, xE);
      _mm_store_ps(xN_lanes, xN);
      _mm_store_ps(xC_lanes, xC);
      _mm_store_ps(xJ_lanes, xJ);
      _mm_store_ps(xB_lanes, xB);
      for (int c = 0; c < kCandidates; ++c) {
        if ((scale_bits & (1u << c)) != 0) {
          xN_lanes[c] = xN_lanes[c] / xE_lanes[c];
          xC_lanes[c] = xC_lanes[c] / xE_lanes[c];
          xJ_lanes[c] = xJ_lanes[c] / xE_lanes[c];
          xB_lanes[c] = xB_lanes[c] / xE_lanes[c];
          inverse_lanes[c] = 1.0f / xE_lanes[c];
          scale_lanes[c] = xE_lanes[c];
          out->totscale[c] = static_cast<float>(
              static_cast<double>(out->totscale[c]) +
              ::log(static_cast<double>(xE_lanes[c])));
        } else {
          inverse_lanes[c] = 1.0f;
          scale_lanes[c] = 1.0f;
        }
      }
      const __m128 inverse = _mm_load_ps(inverse_lanes);
      scale_row = _mm_load_ps(scale_lanes);
      xN = _mm_load_ps(xN_lanes);
      xC = _mm_load_ps(xC_lanes);
      xJ = _mm_load_ps(xJ_lanes);
      xB = _mm_load_ps(xB_lanes);
      xE = _mm_blendv_ps(xE, _mm_set1_ps(1.0f), scale_mask);
      const __m512 inverse_quarters = repeat_candidate_lanes(inverse);
      for (auto &cell : dp) cell = _mm512_mul_ps(cell, inverse_quarters);
    }

    store_special(out->xmx, i, p7X_SCALE, scale_row);
    store_special(out->xmx, i, p7X_E, xE);
    store_special(out->xmx, i, p7X_N, xN);
    store_special(out->xmx, i, p7X_J, xJ);
    store_special(out->xmx, i, p7X_B, xB);
    store_special(out->xmx, i, p7X_C, xC);
  }

  alignas(16) float xC_lanes[kCandidates];
  _mm_store_ps(xC_lanes, xC);
  for (int c = 0; c < kCandidates; ++c) {
    if (std::isnan(xC_lanes[c]) ||
        (L > 0 && xC_lanes[c] == 0.0f) ||
        std::isinf(xC_lanes[c])) {
      out->status[c] = eslERANGE;
      out->score[c] = 0.0f;
    } else {
      const float terminal = xC_lanes[c] * om->xf[p7O_C][p7O_MOVE];
      out->score[c] = static_cast<float>(
          static_cast<double>(out->totscale[c]) +
          ::log(static_cast<double>(terminal)));
    }
  }
  return eslOK;
}

__attribute__((noinline)) int forward4_parser_varlen(
    const std::array<const ESL_DSQ *, kCandidates> &dsq,
    const std::array<int, kCandidates> &lengths,
    const P7_OPROFILE *om,
    Forward4Result *out) {
  if (out == nullptr || om == nullptr || !p7_oprofile_IsLocal(om)) {
    return eslEINVAL;
  }
  const int max_length = *std::max_element(lengths.begin(), lengths.end());
  if (max_length < 1) return eslEINVAL;
  for (int c = 0; c < kCandidates; ++c) {
    if (dsq[c] == nullptr || lengths[c] < 1 || lengths[c] > 100000) {
      return eslEINVAL;
    }
  }
  if (out->M != om->M || out->L != max_length) {
    out->resize(om->M, max_length);
  }

  alignas(16) float move_lanes[kCandidates];
  alignas(16) float loop_lanes[kCandidates];
  for (int c = 0; c < kCandidates; ++c) {
    const float move = (2.0f + om->nj) /
                       (static_cast<float>(lengths[c]) + 2.0f + om->nj);
    move_lanes[c] = move;
    loop_lanes[c] = 1.0f - move;
  }
  const __m128 xmove = _mm_load_ps(move_lanes);
  const __m128 xloop = _mm_load_ps(loop_lanes);
  const __m128 emove = _mm_set1_ps(om->xf[p7O_E][p7O_MOVE]);
  const __m128 eloop = _mm_set1_ps(om->xf[p7O_E][p7O_LOOP]);

  const int Q = p7O_NQF(om->M);
  auto &dp = out->dp;
  const __m512 zero = _mm512_setzero_ps();
  std::fill(dp.begin(), dp.end(), zero);

  __m128 xE = _mm_setzero_ps();
  __m128 xN = _mm_set1_ps(1.0f);
  __m128 xJ = _mm_setzero_ps();
  __m128 xB = xmove;
  __m128 xC = _mm_setzero_ps();
  out->totscale.fill(0.0f);
  out->status.fill(eslOK);
  out->has_own_scales.fill(1);

  const std::array<int, kCandidates> row0 = {0, 0, 0, 0};
  store_special_rows(out->xmx, row0, 0xFu, p7X_E, xE);
  store_special_rows(out->xmx, row0, 0xFu, p7X_N, xN);
  store_special_rows(out->xmx, row0, 0xFu, p7X_J, xJ);
  store_special_rows(out->xmx, row0, 0xFu, p7X_B, xB);
  store_special_rows(out->xmx, row0, 0xFu, p7X_C, xC);
  store_special_rows(
      out->xmx, row0, 0xFu, p7X_SCALE, _mm_set1_ps(1.0f));

  for (int i = 1; i <= max_length; ++i) {
    unsigned active4 = 0;
    std::array<int, kCandidates> rows{};
    const __m128 *rp[kCandidates];
    for (int c = 0; c < kCandidates; ++c) {
      rows[c] = i;
      if (i <= lengths[c]) active4 |= 1u << c;
      const int residue_index = i <= lengths[c] ? i : 1;
      rp[c] = om->rfv[dsq[c][residue_index]];
    }
    if (active4 == 0) break;
    const __mmask16 active16 = expand_candidate_mask(active4);
    const __m128 active_ps = candidate_mask_ps(active4);
    const __m128 *tp = om->tfv;
    __m512 dcv = zero;
    __m512 xEv = zero;
    const __m512 xBv = repeat_candidate_lanes(xB);

    __m512 mpv = rightshiftz_quarters(
        dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_M]);
    __m512 dpv = rightshiftz_quarters(
        dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_D]);
    __m512 ipv = rightshiftz_quarters(
        dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_I]);

    for (int q = 0; q < Q; ++q) {
      __m512 sv = _mm512_mul_ps(xBv, broadcast128(*tp++));
      sv = _mm512_add_ps(sv, _mm512_mul_ps(mpv, broadcast128(*tp++)));
      sv = _mm512_add_ps(sv, _mm512_mul_ps(ipv, broadcast128(*tp++)));
      sv = _mm512_add_ps(sv, _mm512_mul_ps(dpv, broadcast128(*tp++)));
      const __m512 emission = pack4(rp[0][q], rp[1][q], rp[2][q], rp[3][q]);
      sv = _mm512_maskz_mov_ps(active16, _mm512_mul_ps(sv, emission));
      xEv = _mm512_add_ps(xEv, sv);

      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      const __m512 old_m = dp[base + p7X_M];
      const __m512 old_d = dp[base + p7X_D];
      const __m512 old_i = dp[base + p7X_I];
      mpv = old_m;
      dpv = old_d;
      ipv = old_i;
      dp[base + p7X_M] = _mm512_mask_mov_ps(old_m, active16, sv);
      dp[base + p7X_D] = _mm512_mask_mov_ps(old_d, active16, dcv);

      dcv = _mm512_maskz_mov_ps(
          active16, _mm512_mul_ps(sv, broadcast128(*tp++)));
      sv = _mm512_mul_ps(mpv, broadcast128(*tp++));
      const __m512 new_i = _mm512_add_ps(
          sv, _mm512_mul_ps(ipv, broadcast128(*tp++)));
      dp[base + p7X_I] = _mm512_mask_mov_ps(old_i, active16, new_i);
    }

    dcv = _mm512_maskz_mov_ps(active16, rightshiftz_quarters(dcv));
    dp[p7X_D] = _mm512_mask_mov_ps(dp[p7X_D], active16, zero);
    tp = om->tfv + 7 * Q;
    for (int q = 0; q < Q; ++q) {
      const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
      const __m512 sum = _mm512_add_ps(dcv, dp[index]);
      dp[index] = _mm512_mask_mov_ps(dp[index], active16, sum);
      dcv = _mm512_maskz_mov_ps(
          active16, _mm512_mul_ps(dp[index], broadcast128(*tp++)));
    }

    if (om->M < 100) {
      for (int j = 1; j < 4; ++j) {
        dcv = _mm512_maskz_mov_ps(active16, rightshiftz_quarters(dcv));
        tp = om->tfv + 7 * Q;
        for (int q = 0; q < Q; ++q) {
          const size_t index =
              static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
          const __m512 sum = _mm512_add_ps(dcv, dp[index]);
          dp[index] = _mm512_mask_mov_ps(dp[index], active16, sum);
          dcv = _mm512_maskz_mov_ps(
              active16, _mm512_mul_ps(dcv, broadcast128(*tp++)));
        }
      }
    } else {
      unsigned delete_active4 = active4;
      for (int j = 1; j < 4 && delete_active4 != 0; ++j) {
        const __mmask16 delete_active16 =
            expand_candidate_mask(delete_active4);
        dcv = _mm512_maskz_mov_ps(
            delete_active16, rightshiftz_quarters(dcv));
        tp = om->tfv + 7 * Q;
        __mmask16 changed16 = 0;
        for (int q = 0; q < Q; ++q) {
          const size_t index =
              static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
          const __m512 old = dp[index];
          const __m512 sum = _mm512_add_ps(dcv, old);
          changed16 |= _mm512_mask_cmp_ps_mask(
              delete_active16, sum, old, _CMP_GT_OQ);
          dp[index] = _mm512_mask_mov_ps(old, delete_active16, sum);
          dcv = _mm512_maskz_mov_ps(
              delete_active16,
              _mm512_mul_ps(dcv, broadcast128(*tp++)));
        }
        delete_active4 = collapse_changed_mask(changed16);
      }
    }

    for (int q = 0; q < Q; ++q) {
      xEv = _mm512_mask_add_ps(
          xEv, active16, xEv,
          dp[static_cast<size_t>(q) * p7X_NSCELLS + p7X_D]);
    }
    const __m128 next_xE = quarter_sums(xEv);
    const __m128 next_xN = _mm_mul_ps(xN, xloop);
    const __m128 next_xC = _mm_add_ps(
        _mm_mul_ps(xC, xloop), _mm_mul_ps(next_xE, emove));
    const __m128 next_xJ = _mm_add_ps(
        _mm_mul_ps(xJ, xloop), _mm_mul_ps(next_xE, eloop));
    const __m128 next_xB = _mm_add_ps(
        _mm_mul_ps(next_xJ, xmove), _mm_mul_ps(next_xN, xmove));
    xE = _mm_blendv_ps(xE, next_xE, active_ps);
    xN = _mm_blendv_ps(xN, next_xN, active_ps);
    xC = _mm_blendv_ps(xC, next_xC, active_ps);
    xJ = _mm_blendv_ps(xJ, next_xJ, active_ps);
    xB = _mm_blendv_ps(xB, next_xB, active_ps);

    const __m128 scale_mask = _mm_and_ps(
        _mm_cmpgt_ps(xE, _mm_set1_ps(1.0e4f)), active_ps);
    const unsigned scale_bits =
        static_cast<unsigned>(_mm_movemask_ps(scale_mask));
    alignas(16) float scale_lanes[kCandidates] = {1.0f, 1.0f, 1.0f, 1.0f};
    alignas(16) float inverse_lanes[kCandidates] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (scale_bits != 0) {
      alignas(16) float xE_lanes[kCandidates];
      alignas(16) float xN_lanes[kCandidates];
      alignas(16) float xC_lanes[kCandidates];
      alignas(16) float xJ_lanes[kCandidates];
      alignas(16) float xB_lanes[kCandidates];
      _mm_store_ps(xE_lanes, xE);
      _mm_store_ps(xN_lanes, xN);
      _mm_store_ps(xC_lanes, xC);
      _mm_store_ps(xJ_lanes, xJ);
      _mm_store_ps(xB_lanes, xB);
      for (int c = 0; c < kCandidates; ++c) {
        if ((scale_bits & (1u << c)) != 0) {
          xN_lanes[c] = xN_lanes[c] / xE_lanes[c];
          xC_lanes[c] = xC_lanes[c] / xE_lanes[c];
          xJ_lanes[c] = xJ_lanes[c] / xE_lanes[c];
          xB_lanes[c] = xB_lanes[c] / xE_lanes[c];
          inverse_lanes[c] = 1.0f / xE_lanes[c];
          scale_lanes[c] = xE_lanes[c];
          out->totscale[c] = static_cast<float>(
              static_cast<double>(out->totscale[c]) +
              ::log(static_cast<double>(xE_lanes[c])));
        }
      }
      const __m128 inverse = _mm_load_ps(inverse_lanes);
      xN = _mm_load_ps(xN_lanes);
      xC = _mm_load_ps(xC_lanes);
      xJ = _mm_load_ps(xJ_lanes);
      xB = _mm_load_ps(xB_lanes);
      xE = _mm_blendv_ps(xE, _mm_set1_ps(1.0f), scale_mask);
      const __m512 inverse_quarters = repeat_candidate_lanes(inverse);
      for (auto &cell : dp) cell = _mm512_mul_ps(cell, inverse_quarters);
    }

    const __m128 scale_row = _mm_load_ps(scale_lanes);
    store_special_rows(out->xmx, rows, active4, p7X_SCALE, scale_row);
    store_special_rows(out->xmx, rows, active4, p7X_E, xE);
    store_special_rows(out->xmx, rows, active4, p7X_N, xN);
    store_special_rows(out->xmx, rows, active4, p7X_J, xJ);
    store_special_rows(out->xmx, rows, active4, p7X_B, xB);
    store_special_rows(out->xmx, rows, active4, p7X_C, xC);
  }

  alignas(16) float xC_lanes[kCandidates];
  _mm_store_ps(xC_lanes, xC);
  for (int c = 0; c < kCandidates; ++c) {
    if (std::isnan(xC_lanes[c]) || xC_lanes[c] == 0.0f ||
        std::isinf(xC_lanes[c])) {
      out->status[c] = eslERANGE;
      out->score[c] = 0.0f;
    } else {
      const float terminal = xC_lanes[c] * move_lanes[c];
      out->score[c] = static_cast<float>(
          static_cast<double>(out->totscale[c]) +
          ::log(static_cast<double>(terminal)));
    }
  }
  return eslOK;
}

__attribute__((noinline)) int backward4_parser(
    const std::array<const ESL_DSQ *, kCandidates> &dsq,
    int L,
    const P7_OPROFILE *om,
    const Forward4Result &fwd,
    Forward4Result *out) {
  if (out == nullptr || om == nullptr || L < 1) return eslEINVAL;
  if (!p7_oprofile_IsLocal(om) || fwd.M != om->M || fwd.L != L) {
    return eslEINVAL;
  }
  if (out->M != om->M || out->L != L) out->resize(om->M, L);

  const int Q = p7O_NQF(om->M);
  auto &dp = out->dp;
  const __m512 zero = _mm512_setzero_ps();
  std::fill(dp.begin(), dp.end(), zero);
  out->totscale.fill(0.0f);
  out->status.fill(eslOK);
  out->has_own_scales.fill(0);

  __m128 xJ = _mm_setzero_ps();
  __m128 xB = _mm_setzero_ps();
  __m128 xN = _mm_setzero_ps();
  __m128 xC = _mm_set1_ps(om->xf[p7O_C][p7O_MOVE]);
  __m128 xE = _mm_mul_ps(xC, _mm_set1_ps(om->xf[p7O_E][p7O_MOVE]));
  __m512 xEv = repeat_candidate_lanes(xE);
  __m512 dcv = zero;
  for (int q = 0; q < Q; ++q) {
    const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
    dp[base + p7X_M] = xEv;
    dp[base + p7X_D] = xEv;
    dp[base + p7X_I] = zero;
  }

  const __m128 *tp = om->tfv + 8 * Q - 1;
  __m512 dpv = leftshiftz_quarters(
      dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_D]);
  for (int q = Q - 1; q >= 0; --q) {
    dcv = _mm512_mul_ps(dpv, broadcast128(*tp--));
    const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
    dp[index] = _mm512_add_ps(dp[index], dcv);
    dpv = dp[index];
  }
  for (int j = 1; j < 4; ++j) {
    tp = om->tfv + 8 * Q - 1;
    dcv = leftshiftz_quarters(dcv);
    for (int q = Q - 1; q >= 0; --q) {
      dcv = _mm512_mul_ps(dcv, broadcast128(*tp--));
      const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
      dp[index] = _mm512_add_ps(dp[index], dcv);
    }
  }
  tp = om->tfv + 7 * Q - 3;
  dcv = leftshiftz_quarters(dp[p7X_D]);
  for (int q = Q - 1; q >= 0; --q) {
    const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
    dp[base + p7X_M] = _mm512_add_ps(
        dp[base + p7X_M], _mm512_mul_ps(dcv, broadcast128(*tp)));
    tp -= 7;
    dcv = dp[base + p7X_D];
  }

  alignas(16) float scale_lanes[kCandidates];
  for (int c = 0; c < kCandidates; ++c) {
    scale_lanes[c] = fwd.xmx[c][static_cast<size_t>(L) * p7X_NXCELLS + p7X_SCALE];
  }
  __m128 scale = _mm_load_ps(scale_lanes);
  const __m128 scale_mask = _mm_cmpgt_ps(scale, _mm_set1_ps(1.0f));
  if (_mm_movemask_ps(scale_mask) != 0) {
    alignas(16) float inverse_lanes[kCandidates];
    alignas(16) float xE_lanes[kCandidates];
    alignas(16) float xN_lanes[kCandidates];
    alignas(16) float xC_lanes[kCandidates];
    alignas(16) float xJ_lanes[kCandidates];
    alignas(16) float xB_lanes[kCandidates];
    _mm_store_ps(xE_lanes, xE);
    _mm_store_ps(xN_lanes, xN);
    _mm_store_ps(xC_lanes, xC);
    _mm_store_ps(xJ_lanes, xJ);
    _mm_store_ps(xB_lanes, xB);
    for (int c = 0; c < kCandidates; ++c) {
      inverse_lanes[c] = scale_lanes[c] > 1.0f ? 1.0f / scale_lanes[c] : 1.0f;
      if (scale_lanes[c] > 1.0f) {
        xE_lanes[c] = xE_lanes[c] / scale_lanes[c];
        xN_lanes[c] = xN_lanes[c] / scale_lanes[c];
        xC_lanes[c] = xC_lanes[c] / scale_lanes[c];
        xJ_lanes[c] = xJ_lanes[c] / scale_lanes[c];
        xB_lanes[c] = xB_lanes[c] / scale_lanes[c];
      }
      out->totscale[c] = static_cast<float>(
          ::log(static_cast<double>(scale_lanes[c])));
    }
    const __m128 inverse = _mm_load_ps(inverse_lanes);
    xE = _mm_load_ps(xE_lanes);
    xN = _mm_load_ps(xN_lanes);
    xC = _mm_load_ps(xC_lanes);
    xJ = _mm_load_ps(xJ_lanes);
    xB = _mm_load_ps(xB_lanes);
    const __m512 inverse_quarters = repeat_candidate_lanes(inverse);
    for (auto &cell : dp) cell = _mm512_mul_ps(cell, inverse_quarters);
  }
  store_special(out->xmx, L, p7X_SCALE, scale);
  store_special(out->xmx, L, p7X_E, xE);
  store_special(out->xmx, L, p7X_N, xN);
  store_special(out->xmx, L, p7X_J, xJ);
  store_special(out->xmx, L, p7X_B, xB);
  store_special(out->xmx, L, p7X_C, xC);

  for (int i = L - 1; i >= 1; --i) {
    const __m128 *rp[kCandidates] = {
        om->rfv[dsq[0][i + 1]], om->rfv[dsq[1][i + 1]],
        om->rfv[dsq[2][i + 1]], om->rfv[dsq[3][i + 1]]};
    tp = om->tfv + 7 * Q - 1;

    __m512 tmmv = leftshiftz_quarters(broadcast128(om->tfv[1]));
    __m512 timv = leftshiftz_quarters(broadcast128(om->tfv[2]));
    __m512 tdmv = leftshiftz_quarters(broadcast128(om->tfv[3]));
    __m512 mpv = _mm512_mul_ps(
        dp[p7X_M], pack4(rp[0][0], rp[1][0], rp[2][0], rp[3][0]));
    mpv = leftshiftz_quarters(mpv);
    __m512 xBv = zero;

    for (int q = Q - 1; q >= 0; --q) {
      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      const __m512 ipv = dp[base + p7X_I];
      dp[base + p7X_I] = _mm512_add_ps(
          _mm512_mul_ps(ipv, broadcast128(*tp)),
          _mm512_mul_ps(mpv, timv));
      --tp;
      dp[base + p7X_D] = _mm512_mul_ps(mpv, tdmv);
      const __m512 mcv = _mm512_add_ps(
          _mm512_mul_ps(ipv, broadcast128(*tp)),
          _mm512_mul_ps(mpv, tmmv));
      tp -= 2;
      mpv = _mm512_mul_ps(
          dp[base + p7X_M],
          pack4(rp[0][q], rp[1][q], rp[2][q], rp[3][q]));
      dp[base + p7X_M] = mcv;
      tdmv = broadcast128(*tp--);
      timv = broadcast128(*tp--);
      tmmv = broadcast128(*tp--);
      xBv = _mm512_add_ps(
          xBv, _mm512_mul_ps(mpv, broadcast128(*tp--)));
    }

    xB = quarter_sums(xBv);
    xC = _mm_mul_ps(xC, _mm_set1_ps(om->xf[p7O_C][p7O_LOOP]));
    xJ = _mm_add_ps(
        _mm_mul_ps(xB, _mm_set1_ps(om->xf[p7O_J][p7O_MOVE])),
        _mm_mul_ps(xJ, _mm_set1_ps(om->xf[p7O_J][p7O_LOOP])));
    xN = _mm_add_ps(
        _mm_mul_ps(xB, _mm_set1_ps(om->xf[p7O_N][p7O_MOVE])),
        _mm_mul_ps(xN, _mm_set1_ps(om->xf[p7O_N][p7O_LOOP])));
    xE = _mm_add_ps(
        _mm_mul_ps(xC, _mm_set1_ps(om->xf[p7O_E][p7O_MOVE])),
        _mm_mul_ps(xJ, _mm_set1_ps(om->xf[p7O_E][p7O_LOOP])));
    xEv = repeat_candidate_lanes(xE);

    tp = om->tfv + 8 * Q - 1;
    dpv = leftshiftz_quarters(_mm512_add_ps(dp[p7X_D], xEv));
    for (int q = Q - 1; q >= 0; --q) {
      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      dcv = _mm512_mul_ps(dpv, broadcast128(*tp--));
      dp[base + p7X_D] = _mm512_add_ps(
          dp[base + p7X_D], _mm512_add_ps(dcv, xEv));
      dpv = dp[base + p7X_D];
      dp[base + p7X_M] = _mm512_add_ps(dp[base + p7X_M], xEv);
    }
    for (int j = 1; j < 4; ++j) {
      dcv = leftshiftz_quarters(dcv);
      tp = om->tfv + 8 * Q - 1;
      for (int q = Q - 1; q >= 0; --q) {
        dcv = _mm512_mul_ps(dcv, broadcast128(*tp--));
        const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
        dp[index] = _mm512_add_ps(dp[index], dcv);
      }
    }
    dcv = leftshiftz_quarters(dp[p7X_D]);
    tp = om->tfv + 7 * Q - 3;
    for (int q = Q - 1; q >= 0; --q) {
      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      dp[base + p7X_M] = _mm512_add_ps(
          dp[base + p7X_M], _mm512_mul_ps(dcv, broadcast128(*tp)));
      tp -= 7;
      dcv = dp[base + p7X_D];
    }

    alignas(16) float xB_lanes[kCandidates];
    _mm_store_ps(xB_lanes, xB);
    for (int c = 0; c < kCandidates; ++c) {
      if (static_cast<double>(xB_lanes[c]) > 1.0e16) {
        out->has_own_scales[c] = 1;
      }
      if (out->has_own_scales[c] != 0) {
        scale_lanes[c] = xB_lanes[c] > 1.0e4f ? xB_lanes[c] : 1.0f;
      } else {
        scale_lanes[c] = fwd.xmx[c][static_cast<size_t>(i) * p7X_NXCELLS + p7X_SCALE];
      }
    }
    scale = _mm_load_ps(scale_lanes);
    if (_mm_movemask_ps(_mm_cmpgt_ps(scale, _mm_set1_ps(1.0f))) != 0) {
      alignas(16) float inverse_lanes[kCandidates];
      alignas(16) float xE_lanes[kCandidates];
      alignas(16) float xN_lanes[kCandidates];
      alignas(16) float xJ_lanes[kCandidates];
      alignas(16) float xB_scaled_lanes[kCandidates];
      alignas(16) float xC_lanes[kCandidates];
      _mm_store_ps(xE_lanes, xE);
      _mm_store_ps(xN_lanes, xN);
      _mm_store_ps(xJ_lanes, xJ);
      _mm_store_ps(xB_scaled_lanes, xB);
      _mm_store_ps(xC_lanes, xC);
      for (int c = 0; c < kCandidates; ++c) {
        inverse_lanes[c] = scale_lanes[c] > 1.0f ? 1.0f / scale_lanes[c] : 1.0f;
        if (scale_lanes[c] > 1.0f) {
          xE_lanes[c] = xE_lanes[c] / scale_lanes[c];
          xN_lanes[c] = xN_lanes[c] / scale_lanes[c];
          xJ_lanes[c] = xJ_lanes[c] / scale_lanes[c];
          xB_scaled_lanes[c] = xB_scaled_lanes[c] / scale_lanes[c];
          xC_lanes[c] = xC_lanes[c] / scale_lanes[c];
          out->totscale[c] = static_cast<float>(
              static_cast<double>(out->totscale[c]) +
              ::log(static_cast<double>(scale_lanes[c])));
        }
      }
      const __m128 inverse = _mm_load_ps(inverse_lanes);
      xE = _mm_load_ps(xE_lanes);
      xN = _mm_load_ps(xN_lanes);
      xJ = _mm_load_ps(xJ_lanes);
      xB = _mm_load_ps(xB_scaled_lanes);
      xC = _mm_load_ps(xC_lanes);
      const __m512 inverse_quarters = repeat_candidate_lanes(inverse);
      for (auto &cell : dp) cell = _mm512_mul_ps(cell, inverse_quarters);
    }
    store_special(out->xmx, i, p7X_SCALE, scale);
    store_special(out->xmx, i, p7X_E, xE);
    store_special(out->xmx, i, p7X_N, xN);
    store_special(out->xmx, i, p7X_J, xJ);
    store_special(out->xmx, i, p7X_B, xB);
    store_special(out->xmx, i, p7X_C, xC);
  }

  tp = om->tfv;
  const __m128 *rp[kCandidates] = {
      om->rfv[dsq[0][1]], om->rfv[dsq[1][1]],
      om->rfv[dsq[2][1]], om->rfv[dsq[3][1]]};
  __m512 xBv = zero;
  for (int q = 0; q < Q; ++q) {
    __m512 mpv = _mm512_mul_ps(
        dp[static_cast<size_t>(q) * p7X_NSCELLS + p7X_M],
        pack4(rp[0][q], rp[1][q], rp[2][q], rp[3][q]));
    mpv = _mm512_mul_ps(mpv, broadcast128(*tp));
    tp += 7;
    xBv = _mm512_add_ps(xBv, mpv);
  }
  xB = quarter_sums(xBv);
  xN = _mm_add_ps(
      _mm_mul_ps(xB, _mm_set1_ps(om->xf[p7O_N][p7O_MOVE])),
      _mm_mul_ps(xN, _mm_set1_ps(om->xf[p7O_N][p7O_LOOP])));
  store_special(out->xmx, 0, p7X_B, xB);
  store_special(out->xmx, 0, p7X_C, _mm_setzero_ps());
  store_special(out->xmx, 0, p7X_J, _mm_setzero_ps());
  store_special(out->xmx, 0, p7X_N, xN);
  store_special(out->xmx, 0, p7X_E, _mm_setzero_ps());
  store_special(out->xmx, 0, p7X_SCALE, _mm_set1_ps(1.0f));

  alignas(16) float xN_lanes[kCandidates];
  _mm_store_ps(xN_lanes, xN);
  for (int c = 0; c < kCandidates; ++c) {
    if (std::isnan(xN_lanes[c]) || xN_lanes[c] == 0.0f ||
        std::isinf(xN_lanes[c])) {
      out->status[c] = eslERANGE;
      out->score[c] = 0.0f;
    } else {
      out->score[c] = static_cast<float>(
          static_cast<double>(out->totscale[c]) +
          ::log(static_cast<double>(xN_lanes[c])));
    }
  }
  return eslOK;
}

__attribute__((noinline)) int backward4_parser_varlen(
    const std::array<const ESL_DSQ *, kCandidates> &dsq,
    const std::array<int, kCandidates> &lengths,
    const P7_OPROFILE *om,
    const std::array<const float *, kCandidates> &forward_xmx,
    const std::array<uint64_t, kCandidates> &forward_xmx_counts,
    Forward4Result *out) {
  if (out == nullptr || om == nullptr || !p7_oprofile_IsLocal(om)) {
    return eslEINVAL;
  }
  const int max_length = *std::max_element(lengths.begin(), lengths.end());
  if (max_length < 1) return eslEINVAL;
  for (int c = 0; c < kCandidates; ++c) {
    if (lengths[c] < 1 || forward_xmx[c] == nullptr ||
        forward_xmx_counts[c] <
            static_cast<uint64_t>(lengths[c] + 1) * p7X_NXCELLS) {
      return eslEINVAL;
    }
  }
  if (out->M != om->M || out->L != max_length) {
    out->resize(om->M, max_length);
  }

  alignas(16) float move_lanes[kCandidates];
  alignas(16) float loop_lanes[kCandidates];
  for (int c = 0; c < kCandidates; ++c) {
    const float move = (2.0f + om->nj) /
                       (static_cast<float>(lengths[c]) + 2.0f + om->nj);
    move_lanes[c] = move;
    loop_lanes[c] = 1.0f - move;
  }
  const __m128 xmove = _mm_load_ps(move_lanes);
  const __m128 xloop = _mm_load_ps(loop_lanes);
  const __m128 emove = _mm_set1_ps(om->xf[p7O_E][p7O_MOVE]);
  const __m128 eloop = _mm_set1_ps(om->xf[p7O_E][p7O_LOOP]);

  const int Q = p7O_NQF(om->M);
  auto &dp = out->dp;
  const __m512 zero = _mm512_setzero_ps();
  std::fill(dp.begin(), dp.end(), zero);
  out->totscale.fill(0.0f);
  out->status.fill(eslOK);
  out->has_own_scales.fill(0);

  __m128 xJ = _mm_setzero_ps();
  __m128 xB = _mm_setzero_ps();
  __m128 xN = _mm_setzero_ps();
  __m128 xC = xmove;
  __m128 xE = _mm_mul_ps(xC, emove);
  __m512 xEv = repeat_candidate_lanes(xE);
  __m512 dcv = zero;
  for (int q = 0; q < Q; ++q) {
    const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
    dp[base + p7X_M] = xEv;
    dp[base + p7X_D] = xEv;
    dp[base + p7X_I] = zero;
  }

  const __m128 *tp = om->tfv + 8 * Q - 1;
  __m512 dpv = leftshiftz_quarters(
      dp[static_cast<size_t>(Q - 1) * p7X_NSCELLS + p7X_D]);
  for (int q = Q - 1; q >= 0; --q) {
    dcv = _mm512_mul_ps(dpv, broadcast128(*tp--));
    const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
    dp[index] = _mm512_add_ps(dp[index], dcv);
    dpv = dp[index];
  }
  for (int j = 1; j < 4; ++j) {
    tp = om->tfv + 8 * Q - 1;
    dcv = leftshiftz_quarters(dcv);
    for (int q = Q - 1; q >= 0; --q) {
      dcv = _mm512_mul_ps(dcv, broadcast128(*tp--));
      const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
      dp[index] = _mm512_add_ps(dp[index], dcv);
    }
  }
  tp = om->tfv + 7 * Q - 3;
  dcv = leftshiftz_quarters(dp[p7X_D]);
  for (int q = Q - 1; q >= 0; --q) {
    const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
    dp[base + p7X_M] = _mm512_add_ps(
        dp[base + p7X_M], _mm512_mul_ps(dcv, broadcast128(*tp)));
    tp -= 7;
    dcv = dp[base + p7X_D];
  }

  alignas(16) float scale_lanes[kCandidates];
  for (int c = 0; c < kCandidates; ++c) {
    scale_lanes[c] = forward_xmx[c][
        static_cast<size_t>(lengths[c]) * p7X_NXCELLS + p7X_SCALE];
    out->totscale[c] = static_cast<float>(
        ::log(static_cast<double>(scale_lanes[c])));
  }
  __m128 scale = _mm_load_ps(scale_lanes);
  if (_mm_movemask_ps(_mm_cmpgt_ps(scale, _mm_set1_ps(1.0f))) != 0) {
    alignas(16) float inverse_lanes[kCandidates];
    alignas(16) float xE_lanes[kCandidates];
    alignas(16) float xN_lanes[kCandidates];
    alignas(16) float xC_lanes[kCandidates];
    alignas(16) float xJ_lanes[kCandidates];
    alignas(16) float xB_lanes[kCandidates];
    _mm_store_ps(xE_lanes, xE);
    _mm_store_ps(xN_lanes, xN);
    _mm_store_ps(xC_lanes, xC);
    _mm_store_ps(xJ_lanes, xJ);
    _mm_store_ps(xB_lanes, xB);
    for (int c = 0; c < kCandidates; ++c) {
      inverse_lanes[c] = scale_lanes[c] > 1.0f ? 1.0f / scale_lanes[c] : 1.0f;
      if (scale_lanes[c] > 1.0f) {
        xE_lanes[c] = xE_lanes[c] / scale_lanes[c];
        xN_lanes[c] = xN_lanes[c] / scale_lanes[c];
        xC_lanes[c] = xC_lanes[c] / scale_lanes[c];
        xJ_lanes[c] = xJ_lanes[c] / scale_lanes[c];
        xB_lanes[c] = xB_lanes[c] / scale_lanes[c];
      }
    }
    const __m128 inverse = _mm_load_ps(inverse_lanes);
    xE = _mm_load_ps(xE_lanes);
    xN = _mm_load_ps(xN_lanes);
    xC = _mm_load_ps(xC_lanes);
    xJ = _mm_load_ps(xJ_lanes);
    xB = _mm_load_ps(xB_lanes);
    const __m512 inverse_quarters = repeat_candidate_lanes(inverse);
    for (auto &cell : dp) cell = _mm512_mul_ps(cell, inverse_quarters);
  }
  store_special_rows(out->xmx, lengths, 0xFu, p7X_SCALE, scale);
  store_special_rows(out->xmx, lengths, 0xFu, p7X_E, xE);
  store_special_rows(out->xmx, lengths, 0xFu, p7X_N, xN);
  store_special_rows(out->xmx, lengths, 0xFu, p7X_J, xJ);
  store_special_rows(out->xmx, lengths, 0xFu, p7X_B, xB);
  store_special_rows(out->xmx, lengths, 0xFu, p7X_C, xC);

  for (int step = 0; step < max_length - 1; ++step) {
    std::array<int, kCandidates> rows{};
    unsigned active4 = 0;
    for (int c = 0; c < kCandidates; ++c) {
      rows[c] = lengths[c] - 1 - step;
      if (rows[c] >= 1) active4 |= 1u << c;
    }
    if (active4 == 0) break;
    const __mmask16 active16 = expand_candidate_mask(active4);
    const __m128 active_ps = candidate_mask_ps(active4);
    const __m128 *rp[kCandidates];
    for (int c = 0; c < kCandidates; ++c) {
      const int residue_index = rows[c] >= 1 ? rows[c] + 1 : 1;
      rp[c] = om->rfv[dsq[c][residue_index]];
    }
    tp = om->tfv + 7 * Q - 1;

    __m512 tmmv = leftshiftz_quarters(broadcast128(om->tfv[1]));
    __m512 timv = leftshiftz_quarters(broadcast128(om->tfv[2]));
    __m512 tdmv = leftshiftz_quarters(broadcast128(om->tfv[3]));
    __m512 mpv = _mm512_mul_ps(
        dp[p7X_M], pack4(rp[0][0], rp[1][0], rp[2][0], rp[3][0]));
    mpv = _mm512_maskz_mov_ps(active16, leftshiftz_quarters(mpv));
    __m512 xBv = zero;
    for (int q = Q - 1; q >= 0; --q) {
      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      const __m512 old_i = dp[base + p7X_I];
      const __m512 new_i = _mm512_add_ps(
          _mm512_mul_ps(old_i, broadcast128(*tp)),
          _mm512_mul_ps(mpv, timv));
      --tp;
      const __m512 new_d = _mm512_mul_ps(mpv, tdmv);
      const __m512 new_m = _mm512_add_ps(
          _mm512_mul_ps(old_i, broadcast128(*tp)),
          _mm512_mul_ps(mpv, tmmv));
      tp -= 2;
      mpv = _mm512_mul_ps(
          dp[base + p7X_M],
          pack4(rp[0][q], rp[1][q], rp[2][q], rp[3][q]));
      dp[base + p7X_I] = _mm512_mask_mov_ps(old_i, active16, new_i);
      dp[base + p7X_D] = _mm512_mask_mov_ps(
          dp[base + p7X_D], active16, new_d);
      dp[base + p7X_M] = _mm512_mask_mov_ps(
          dp[base + p7X_M], active16, new_m);
      tdmv = broadcast128(*tp--);
      timv = broadcast128(*tp--);
      tmmv = broadcast128(*tp--);
      xBv = _mm512_mask_add_ps(
          xBv, active16, xBv,
          _mm512_mul_ps(mpv, broadcast128(*tp--)));
    }

    const __m128 next_xB = quarter_sums(xBv);
    const __m128 next_xC = _mm_mul_ps(xC, xloop);
    const __m128 next_xJ = _mm_add_ps(
        _mm_mul_ps(next_xB, xmove), _mm_mul_ps(xJ, xloop));
    const __m128 next_xN = _mm_add_ps(
        _mm_mul_ps(next_xB, xmove), _mm_mul_ps(xN, xloop));
    const __m128 next_xE = _mm_add_ps(
        _mm_mul_ps(next_xC, emove), _mm_mul_ps(next_xJ, eloop));
    xB = _mm_blendv_ps(xB, next_xB, active_ps);
    xC = _mm_blendv_ps(xC, next_xC, active_ps);
    xJ = _mm_blendv_ps(xJ, next_xJ, active_ps);
    xN = _mm_blendv_ps(xN, next_xN, active_ps);
    xE = _mm_blendv_ps(xE, next_xE, active_ps);
    xEv = repeat_candidate_lanes(xE);

    tp = om->tfv + 8 * Q - 1;
    dpv = _mm512_maskz_mov_ps(
        active16, leftshiftz_quarters(_mm512_add_ps(dp[p7X_D], xEv)));
    for (int q = Q - 1; q >= 0; --q) {
      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      dcv = _mm512_maskz_mov_ps(
          active16, _mm512_mul_ps(dpv, broadcast128(*tp--)));
      const __m512 new_d = _mm512_add_ps(
          dp[base + p7X_D], _mm512_add_ps(dcv, xEv));
      dp[base + p7X_D] = _mm512_mask_mov_ps(
          dp[base + p7X_D], active16, new_d);
      dpv = dp[base + p7X_D];
      dp[base + p7X_M] = _mm512_mask_add_ps(
          dp[base + p7X_M], active16, dp[base + p7X_M], xEv);
    }
    for (int j = 1; j < 4; ++j) {
      dcv = _mm512_maskz_mov_ps(active16, leftshiftz_quarters(dcv));
      tp = om->tfv + 8 * Q - 1;
      for (int q = Q - 1; q >= 0; --q) {
        dcv = _mm512_maskz_mov_ps(
            active16, _mm512_mul_ps(dcv, broadcast128(*tp--)));
        const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + p7X_D;
        dp[index] = _mm512_mask_add_ps(
            dp[index], active16, dp[index], dcv);
      }
    }
    dcv = _mm512_maskz_mov_ps(active16, leftshiftz_quarters(dp[p7X_D]));
    tp = om->tfv + 7 * Q - 3;
    for (int q = Q - 1; q >= 0; --q) {
      const size_t base = static_cast<size_t>(q) * p7X_NSCELLS;
      const __m512 add = _mm512_mul_ps(dcv, broadcast128(*tp));
      dp[base + p7X_M] = _mm512_mask_add_ps(
          dp[base + p7X_M], active16, dp[base + p7X_M], add);
      tp -= 7;
      dcv = _mm512_maskz_mov_ps(active16, dp[base + p7X_D]);
    }

    alignas(16) float xB_lanes[kCandidates];
    _mm_store_ps(xB_lanes, xB);
    for (int c = 0; c < kCandidates; ++c) {
      if ((active4 & (1u << c)) == 0) {
        scale_lanes[c] = 1.0f;
        continue;
      }
      if (static_cast<double>(xB_lanes[c]) > 1.0e16) {
        out->has_own_scales[c] = 1;
      }
      if (out->has_own_scales[c] != 0) {
        scale_lanes[c] = xB_lanes[c] > 1.0e4f ? xB_lanes[c] : 1.0f;
      } else {
        scale_lanes[c] = forward_xmx[c][
            static_cast<size_t>(rows[c]) * p7X_NXCELLS + p7X_SCALE];
      }
    }
    scale = _mm_load_ps(scale_lanes);
    if (_mm_movemask_ps(_mm_cmpgt_ps(scale, _mm_set1_ps(1.0f))) != 0) {
      alignas(16) float inverse_lanes[kCandidates];
      alignas(16) float xE_lanes[kCandidates];
      alignas(16) float xN_lanes[kCandidates];
      alignas(16) float xJ_lanes[kCandidates];
      alignas(16) float xB_scaled_lanes[kCandidates];
      alignas(16) float xC_lanes[kCandidates];
      _mm_store_ps(xE_lanes, xE);
      _mm_store_ps(xN_lanes, xN);
      _mm_store_ps(xJ_lanes, xJ);
      _mm_store_ps(xB_scaled_lanes, xB);
      _mm_store_ps(xC_lanes, xC);
      for (int c = 0; c < kCandidates; ++c) {
        inverse_lanes[c] = scale_lanes[c] > 1.0f ? 1.0f / scale_lanes[c] : 1.0f;
        if (scale_lanes[c] > 1.0f) {
          xE_lanes[c] = xE_lanes[c] / scale_lanes[c];
          xN_lanes[c] = xN_lanes[c] / scale_lanes[c];
          xJ_lanes[c] = xJ_lanes[c] / scale_lanes[c];
          xB_scaled_lanes[c] = xB_scaled_lanes[c] / scale_lanes[c];
          xC_lanes[c] = xC_lanes[c] / scale_lanes[c];
          out->totscale[c] = static_cast<float>(
              static_cast<double>(out->totscale[c]) +
              ::log(static_cast<double>(scale_lanes[c])));
        }
      }
      const __m128 inverse = _mm_load_ps(inverse_lanes);
      xE = _mm_load_ps(xE_lanes);
      xN = _mm_load_ps(xN_lanes);
      xJ = _mm_load_ps(xJ_lanes);
      xB = _mm_load_ps(xB_scaled_lanes);
      xC = _mm_load_ps(xC_lanes);
      const __m512 inverse_quarters = repeat_candidate_lanes(inverse);
      for (auto &cell : dp) cell = _mm512_mul_ps(cell, inverse_quarters);
    }
    store_special_rows(out->xmx, rows, active4, p7X_SCALE, scale);
    store_special_rows(out->xmx, rows, active4, p7X_E, xE);
    store_special_rows(out->xmx, rows, active4, p7X_N, xN);
    store_special_rows(out->xmx, rows, active4, p7X_J, xJ);
    store_special_rows(out->xmx, rows, active4, p7X_B, xB);
    store_special_rows(out->xmx, rows, active4, p7X_C, xC);
  }

  tp = om->tfv;
  const __m128 *rp[kCandidates] = {
      om->rfv[dsq[0][1]], om->rfv[dsq[1][1]],
      om->rfv[dsq[2][1]], om->rfv[dsq[3][1]]};
  __m512 xBv = zero;
  for (int q = 0; q < Q; ++q) {
    __m512 mpv = _mm512_mul_ps(
        dp[static_cast<size_t>(q) * p7X_NSCELLS + p7X_M],
        pack4(rp[0][q], rp[1][q], rp[2][q], rp[3][q]));
    mpv = _mm512_mul_ps(mpv, broadcast128(*tp));
    tp += 7;
    xBv = _mm512_add_ps(xBv, mpv);
  }
  xB = quarter_sums(xBv);
  xN = _mm_add_ps(_mm_mul_ps(xB, xmove), _mm_mul_ps(xN, xloop));
  const std::array<int, kCandidates> row0 = {0, 0, 0, 0};
  store_special_rows(out->xmx, row0, 0xFu, p7X_B, xB);
  store_special_rows(out->xmx, row0, 0xFu, p7X_C, _mm_setzero_ps());
  store_special_rows(out->xmx, row0, 0xFu, p7X_J, _mm_setzero_ps());
  store_special_rows(out->xmx, row0, 0xFu, p7X_N, xN);
  store_special_rows(out->xmx, row0, 0xFu, p7X_E, _mm_setzero_ps());
  store_special_rows(out->xmx, row0, 0xFu, p7X_SCALE, _mm_set1_ps(1.0f));

  alignas(16) float xN_lanes[kCandidates];
  _mm_store_ps(xN_lanes, xN);
  for (int c = 0; c < kCandidates; ++c) {
    if (std::isnan(xN_lanes[c]) || xN_lanes[c] == 0.0f ||
        std::isinf(xN_lanes[c])) {
      out->status[c] = eslERANGE;
      out->score[c] = 0.0f;
    } else {
      out->score[c] = static_cast<float>(
          static_cast<double>(out->totscale[c]) +
          ::log(static_cast<double>(xN_lanes[c])));
    }
  }
  return eslOK;
}

uint32_t float_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void compare_exact(const std::array<P7_OMX *, kCandidates> &scalar,
                   const std::array<int, kCandidates> &scalar_status,
                   const std::array<float, kCandidates> &scalar_score,
                   const Forward4Result &packed,
                   const std::array<int, kCandidates> &lengths) {
  for (int c = 0; c < kCandidates; ++c) {
    if (scalar_status[c] != packed.status[c]) {
      fail("status mismatch for candidate " + std::to_string(c));
    }
    if (float_bits(scalar_score[c]) != float_bits(packed.score[c])) {
      fail("score mismatch for candidate " + std::to_string(c));
    }
    if (float_bits(scalar[c]->totscale) != float_bits(packed.totscale[c])) {
      fail("totscale mismatch for candidate " + std::to_string(c));
    }
    if (scalar[c]->has_own_scales != packed.has_own_scales[c]) {
      fail("scale ownership mismatch for candidate " + std::to_string(c));
    }
    for (int i = 0; i <= lengths[c]; ++i) {
      for (int s = 0; s < p7X_NXCELLS; ++s) {
        const size_t index = static_cast<size_t>(i) * p7X_NXCELLS + s;
        if (float_bits(scalar[c]->xmx[index]) !=
            float_bits(packed.xmx[c][index])) {
          fail("special mismatch candidate=" + std::to_string(c) +
               " row=" + std::to_string(i) +
               " state=" + std::to_string(s));
        }
      }
    }
  }

  alignas(64) float lanes[16];
  for (int q = 0; q < packed.Q; ++q) {
    for (int state = 0; state < p7X_NSCELLS; ++state) {
      const size_t index = static_cast<size_t>(q) * p7X_NSCELLS + state;
      _mm512_store_ps(lanes, packed.dp[index]);
      for (int c = 0; c < kCandidates; ++c) {
        alignas(16) float reference[4];
        _mm_store_ps(reference, scalar[c]->dpf[0][index]);
        for (int lane = 0; lane < 4; ++lane) {
          if (float_bits(reference[lane]) !=
              float_bits(lanes[4 * c + lane])) {
            fail("DP mismatch candidate=" + std::to_string(c) +
                 " q=" + std::to_string(q) +
                 " state=" + std::to_string(state) +
                 " lane=" + std::to_string(lane));
          }
        }
      }
    }
  }
}

double milliseconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}  // namespace

#if defined(__x86_64__)
extern "C" int plan7_avx512_tail_available(void) {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl");
}
#else
extern "C" int plan7_avx512_tail_available(void) { return 0; }
#endif

extern "C" int plan7_avx512_forward4_varlen(
    const ESL_DSQ *const sequences[PLAN7_AVX512_TAIL_LANES],
    const int lengths[PLAN7_AVX512_TAIL_LANES],
    const P7_OPROFILE *profile,
    const float *forward_xmx[PLAN7_AVX512_TAIL_LANES],
    uint64_t forward_xmx_counts[PLAN7_AVX512_TAIL_LANES],
    float forward_scores[PLAN7_AVX512_TAIL_LANES],
    float forward_totscales[PLAN7_AVX512_TAIL_LANES],
    int forward_statuses[PLAN7_AVX512_TAIL_LANES],
    uint64_t *elapsed_ns) {
  if (!plan7_avx512_tail_available() || sequences == nullptr ||
      lengths == nullptr || profile == nullptr || forward_xmx == nullptr ||
      forward_xmx_counts == nullptr || forward_scores == nullptr ||
      forward_totscales == nullptr || forward_statuses == nullptr ||
      elapsed_ns == nullptr || !p7_oprofile_IsLocal(profile)) {
    return eslEINVAL;
  }

  std::array<const ESL_DSQ *, kCandidates> sequence_array{};
  std::array<int, kCandidates> length_array{};
  for (int c = 0; c < kCandidates; ++c) {
    if (sequences[c] == nullptr || lengths[c] < 1 || lengths[c] > 100000) {
      return eslEINVAL;
    }
    sequence_array[c] = sequences[c];
    length_array[c] = lengths[c];
  }

  try {
    const auto start = std::chrono::steady_clock::now();
    const int status = forward4_parser_varlen(
        sequence_array, length_array, profile, &forward_output);
    *elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    if (status != eslOK) return status;
    for (int c = 0; c < kCandidates; ++c) {
      forward_xmx[c] = forward_output.xmx[c].data();
      forward_xmx_counts[c] =
          (static_cast<uint64_t>(lengths[c]) + 1u) * p7X_NXCELLS;
      forward_scores[c] = forward_output.score[c];
      forward_totscales[c] = forward_output.totscale[c];
      forward_statuses[c] = forward_output.status[c];
    }
  } catch (const std::bad_alloc &) {
    return eslEMEM;
  } catch (...) {
    return eslEINVAL;
  }
  return eslOK;
}

extern "C" int plan7_avx512_backward4_varlen(
    const ESL_DSQ *const sequences[PLAN7_AVX512_TAIL_LANES],
    const int lengths[PLAN7_AVX512_TAIL_LANES],
    const P7_OPROFILE *profile,
    const float *const forward_xmx[PLAN7_AVX512_TAIL_LANES],
    const uint64_t forward_xmx_counts[PLAN7_AVX512_TAIL_LANES],
    const float *backward_xmx[PLAN7_AVX512_TAIL_LANES],
    uint64_t backward_xmx_counts[PLAN7_AVX512_TAIL_LANES],
    float backward_scores[PLAN7_AVX512_TAIL_LANES],
    float backward_totscales[PLAN7_AVX512_TAIL_LANES],
    int backward_has_own_scales[PLAN7_AVX512_TAIL_LANES],
    uint64_t *elapsed_ns) {
  if (!plan7_avx512_tail_available() || sequences == nullptr ||
      lengths == nullptr || profile == nullptr || forward_xmx == nullptr ||
      forward_xmx_counts == nullptr || backward_xmx == nullptr ||
      backward_xmx_counts == nullptr || backward_scores == nullptr ||
      backward_totscales == nullptr ||
      backward_has_own_scales == nullptr || elapsed_ns == nullptr ||
      !p7_oprofile_IsLocal(profile)) {
    return eslEINVAL;
  }

  std::array<const ESL_DSQ *, kCandidates> sequence_array{};
  std::array<int, kCandidates> length_array{};
  for (int c = 0; c < kCandidates; ++c) {
    if (sequences[c] == nullptr || forward_xmx[c] == nullptr ||
        lengths[c] < 1 || lengths[c] > 100000) {
      return eslEINVAL;
    }
    const uint64_t expected =
        (static_cast<uint64_t>(lengths[c]) + 1u) * p7X_NXCELLS;
    if (forward_xmx_counts[c] != expected) return eslEINVAL;
    sequence_array[c] = sequences[c];
    length_array[c] = lengths[c];
  }

  try {
    const auto start = std::chrono::steady_clock::now();
    std::array<const float *, kCandidates> forward_array{};
    std::array<uint64_t, kCandidates> forward_count_array{};
    for (int c = 0; c < kCandidates; ++c) {
      forward_array[c] = forward_xmx[c];
      forward_count_array[c] = forward_xmx_counts[c];
    }
    const int status = backward4_parser_varlen(
        sequence_array, length_array, profile, forward_array,
        forward_count_array,
        &backward_output);
    *elapsed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    if (status != eslOK) return status;
    for (int c = 0; c < kCandidates; ++c) {
      if (backward_output.status[c] != eslOK) {
        return backward_output.status[c];
      }
      backward_xmx[c] = backward_output.xmx[c].data();
      backward_xmx_counts[c] =
          (static_cast<uint64_t>(lengths[c]) + 1u) * p7X_NXCELLS;
      backward_scores[c] = backward_output.score[c];
      backward_totscales[c] = backward_output.totscale[c];
      backward_has_own_scales[c] = backward_output.has_own_scales[c];
    }
  } catch (const std::bad_alloc &) {
    return eslEMEM;
  } catch (...) {
    return eslEINVAL;
  }
  return eslOK;
}

extern "C" int plan7_avx512_tail_madvise_pages(
    uint64_t *released_bytes) {
  if (released_bytes == nullptr) return eslEINVAL;
  *released_bytes = 0;
  tail_madvise_calls.fetch_add(1, std::memory_order_relaxed);
  const long observed_page_size = sysconf(_SC_PAGESIZE);
  if (observed_page_size <= 0) return eslESYS;
  const size_t page_size = static_cast<size_t>(observed_page_size);
  bool ok = madvise_result_pages(
      &forward_output, page_size, released_bytes);
  ok = madvise_result_pages(
      &backward_output, page_size, released_bytes) && ok;
  tail_madvise_released_bytes.fetch_add(
      *released_bytes, std::memory_order_relaxed);
  return ok ? eslOK : eslESYS;
}

extern "C" void plan7_avx512_tail_madvise_statistics(
    uint64_t *call_count, uint64_t *released_bytes) {
  if (call_count != nullptr) {
    *call_count = tail_madvise_calls.load(std::memory_order_relaxed);
  }
  if (released_bytes != nullptr) {
    *released_bytes =
        tail_madvise_released_bytes.load(std::memory_order_relaxed);
  }
}

extern "C" void plan7_avx512_tail_madvise_statistics_reset(void) {
  tail_madvise_calls.store(0, std::memory_order_relaxed);
  tail_madvise_released_bytes.store(0, std::memory_order_relaxed);
}

#ifndef PLAN7_AVX512_TAIL_LIBRARY
int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: " << argv[0] << " HMMFILE LENGTH [ITERATIONS]\n";
    return 2;
  }
  const char *hmm_path = argv[1];
  const int L = std::atoi(argv[2]);
  const int iterations = argc == 4 ? std::atoi(argv[3]) : 20;
  if (L < 1 || iterations < 1) fail("length and iterations must be positive");

  ESL_ALPHABET *abc = nullptr;
  P7_HMMFILE *hfp = nullptr;
  P7_HMM *hmm = nullptr;
  if (p7_hmmfile_Open(hmm_path, nullptr, &hfp, nullptr) != eslOK ||
      p7_hmmfile_Read(hfp, &abc, &hmm) != eslOK) {
    fail("could not read HMM");
  }
  p7_hmmfile_Close(hfp);

  P7_BG *bg = p7_bg_Create(abc);
  P7_PROFILE *gm = p7_profile_Create(hmm->M, abc);
  P7_OPROFILE *om = p7_oprofile_Create(hmm->M, abc);
  if (bg == nullptr || gm == nullptr || om == nullptr) fail("allocation failed");
  p7_bg_SetLength(bg, L);
  if (p7_ProfileConfig(hmm, bg, gm, L, p7_LOCAL) != eslOK ||
      p7_oprofile_Convert(gm, om) != eslOK ||
      p7_oprofile_ReconfigLength(om, L) != eslOK) {
    fail("profile configuration failed");
  }

  const std::string amino = "ACDEFGHIKLMNPQRSTVWY";
  std::array<std::vector<ESL_DSQ>, kCandidates> dsq_storage;
  std::array<const ESL_DSQ *, kCandidates> dsq{};
  for (int c = 0; c < kCandidates; ++c) {
    std::string sequence;
    sequence.reserve(static_cast<size_t>(L));
    for (int i = 0; i < L; ++i) {
      const size_t index = static_cast<size_t>(
          (i * (c + 3) + c * c * 7 + i / 17) % static_cast<int>(amino.size()));
      sequence.push_back(amino[index]);
    }
    dsq_storage[c].resize(static_cast<size_t>(L) + 2);
    if (esl_abc_Digitize(abc, sequence.c_str(), dsq_storage[c].data()) != eslOK) {
      fail("sequence digitization failed");
    }
    dsq[c] = dsq_storage[c].data();
  }

  std::array<P7_OMX *, kCandidates> scalar_fwd{};
  std::array<P7_OMX *, kCandidates> scalar_bck{};
  for (auto &matrix : scalar_fwd) {
    matrix = p7_omx_Create(hmm->M, 0, L);
    if (matrix == nullptr) fail("matrix allocation failed");
  }
  for (auto &matrix : scalar_bck) {
    matrix = p7_omx_Create(hmm->M, 0, L);
    if (matrix == nullptr) fail("matrix allocation failed");
  }
  Forward4Result packed_fwd;
  Forward4Result packed_bck;
  packed_fwd.resize(hmm->M, L);
  packed_bck.resize(hmm->M, L);
  std::array<float, kCandidates> scalar_fwd_score{};
  std::array<float, kCandidates> scalar_bck_score{};
  std::array<int, kCandidates> scalar_fwd_status{};
  std::array<int, kCandidates> scalar_bck_status{};

  for (int c = 0; c < kCandidates; ++c) {
    scalar_fwd_status[c] = p7_ForwardParser(
        dsq[c], L, om, scalar_fwd[c], &scalar_fwd_score[c]);
  }
  if (forward4_parser(dsq, L, om, &packed_fwd) != eslOK) {
    fail("packed Forward call failed");
  }
  compare_exact(
      scalar_fwd, scalar_fwd_status, scalar_fwd_score, packed_fwd,
      std::array<int, kCandidates>{L, L, L, L});
  for (int c = 0; c < kCandidates; ++c) {
    scalar_bck_status[c] = p7_BackwardParser(
        dsq[c], L, om, scalar_fwd[c], scalar_bck[c], &scalar_bck_score[c]);
  }
  if (backward4_parser(dsq, L, om, packed_fwd, &packed_bck) != eslOK) {
    fail("packed Backward call failed");
  }
  compare_exact(
      scalar_bck, scalar_bck_status, scalar_bck_score, packed_bck,
      std::array<int, kCandidates>{L, L, L, L});

  const std::array<int, kCandidates> varied_lengths = {
      std::max(1, (L * 3) / 4),
      std::max(1, (L * 5) / 6),
      std::max(1, (L * 11) / 12),
      L};
  std::array<P7_OPROFILE *, kCandidates> varied_om{};
  std::array<P7_OMX *, kCandidates> varied_fwd{};
  std::array<P7_OMX *, kCandidates> varied_bck{};
  std::array<float, kCandidates> varied_fwd_score{};
  std::array<float, kCandidates> varied_bck_score{};
  std::array<int, kCandidates> varied_fwd_status{};
  std::array<int, kCandidates> varied_bck_status{};
  Forward4Result varied_fwd_input;
  Forward4Result varied_fwd_packed;
  Forward4Result varied_bck_packed;
  std::array<const float *, kCandidates> varied_forward_xmx{};
  std::array<uint64_t, kCandidates> varied_forward_xmx_counts{};
  varied_fwd_input.resize(hmm->M, L);
  varied_bck_packed.resize(hmm->M, L);
  const size_t transition_bytes =
      static_cast<size_t>(8 * p7O_NQF(hmm->M)) * sizeof(__m128);
  for (int c = 0; c < kCandidates; ++c) {
    varied_om[c] = p7_oprofile_Clone(om);
    varied_fwd[c] = p7_omx_Create(hmm->M, 0, L);
    varied_bck[c] = p7_omx_Create(hmm->M, 0, L);
    if (varied_om[c] == nullptr || varied_fwd[c] == nullptr ||
        varied_bck[c] == nullptr) {
      fail("varied-length allocation failed");
    }
    p7_oprofile_ReconfigLength(varied_om[c], varied_lengths[c]);
    if (std::memcmp(varied_om[c]->tfv, om->tfv, transition_bytes) != 0) {
      fail("target length unexpectedly changed core Forward transitions");
    }
    const int status = p7_ForwardParser(
        dsq[c], varied_lengths[c], varied_om[c], varied_fwd[c],
        &varied_fwd_score[c]);
    varied_fwd_status[c] = status;
    if (status != eslOK) fail("varied-length Forward failed");
    const size_t xmx_count =
        static_cast<size_t>(varied_lengths[c] + 1) * p7X_NXCELLS;
    std::copy_n(
        varied_fwd[c]->xmx, xmx_count, varied_fwd_input.xmx[c].begin());
    varied_forward_xmx[c] = varied_fwd_input.xmx[c].data();
    varied_forward_xmx_counts[c] = xmx_count;
  }
  if (forward4_parser_varlen(
          dsq, varied_lengths, om, &varied_fwd_packed) != eslOK) {
    fail("varied-length packed Forward call failed");
  }
  compare_exact(
      varied_fwd, varied_fwd_status, varied_fwd_score, varied_fwd_packed,
      varied_lengths);
  for (int c = 0; c < kCandidates; ++c) {
    varied_bck_status[c] = p7_BackwardParser(
        dsq[c], varied_lengths[c], varied_om[c], varied_fwd[c], varied_bck[c],
        &varied_bck_score[c]);
  }
  if (backward4_parser_varlen(
          dsq, varied_lengths, om, varied_forward_xmx,
          varied_forward_xmx_counts,
          &varied_bck_packed) != eslOK) {
    fail("varied-length packed Backward call failed");
  }
  compare_exact(
      varied_bck, varied_bck_status, varied_bck_score, varied_bck_packed,
      varied_lengths);

  volatile float checksum = 0.0f;
  std::vector<double> scalar_fwd_ms;
  std::vector<double> packed_fwd_ms;
  std::vector<double> scalar_bck_ms;
  std::vector<double> packed_bck_ms;
  std::vector<double> varied_scalar_fwd_ms;
  std::vector<double> varied_packed_fwd_ms;
  std::vector<double> varied_scalar_bck_ms;
  std::vector<double> varied_packed_bck_ms;
  constexpr int kRounds = 7;
  for (int round = 0; round < kRounds; ++round) {
    if ((round & 1) == 0) {
      const auto start_scalar = std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        for (int c = 0; c < kCandidates; ++c) {
          p7_ForwardParser(
              dsq[c], L, om, scalar_fwd[c], &scalar_fwd_score[c]);
          checksum += scalar_fwd_score[c];
        }
      }
      scalar_fwd_ms.push_back(milliseconds_since(start_scalar));

      const auto start_packed = std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        forward4_parser(dsq, L, om, &packed_fwd);
        checksum += packed_fwd.score[0];
      }
      packed_fwd_ms.push_back(milliseconds_since(start_packed));
    } else {
      const auto start_packed = std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        forward4_parser(dsq, L, om, &packed_fwd);
        checksum += packed_fwd.score[0];
      }
      packed_fwd_ms.push_back(milliseconds_since(start_packed));

      const auto start_scalar = std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        for (int c = 0; c < kCandidates; ++c) {
          p7_ForwardParser(
              dsq[c], L, om, scalar_fwd[c], &scalar_fwd_score[c]);
          checksum += scalar_fwd_score[c];
        }
      }
      scalar_fwd_ms.push_back(milliseconds_since(start_scalar));
    }

    if ((round & 1) == 0) {
      const auto start_varied_scalar_fwd =
          std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        for (int c = 0; c < kCandidates; ++c) {
          p7_ForwardParser(
              dsq[c], varied_lengths[c], varied_om[c], varied_fwd[c],
              &varied_fwd_score[c]);
          checksum += varied_fwd_score[c];
        }
      }
      varied_scalar_fwd_ms.push_back(
          milliseconds_since(start_varied_scalar_fwd));

      const auto start_varied_packed_fwd =
          std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        forward4_parser_varlen(dsq, varied_lengths, om, &varied_fwd_packed);
        checksum += varied_fwd_packed.score[0];
      }
      varied_packed_fwd_ms.push_back(
          milliseconds_since(start_varied_packed_fwd));
    } else {
      const auto start_varied_packed_fwd =
          std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        forward4_parser_varlen(dsq, varied_lengths, om, &varied_fwd_packed);
        checksum += varied_fwd_packed.score[0];
      }
      varied_packed_fwd_ms.push_back(
          milliseconds_since(start_varied_packed_fwd));

      const auto start_varied_scalar_fwd =
          std::chrono::steady_clock::now();
      for (int it = 0; it < iterations; ++it) {
        for (int c = 0; c < kCandidates; ++c) {
          p7_ForwardParser(
              dsq[c], varied_lengths[c], varied_om[c], varied_fwd[c],
              &varied_fwd_score[c]);
          checksum += varied_fwd_score[c];
        }
      }
      varied_scalar_fwd_ms.push_back(
          milliseconds_since(start_varied_scalar_fwd));
    }

    const auto start_scalar_bck = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
      for (int c = 0; c < kCandidates; ++c) {
        p7_BackwardParser(
            dsq[c], L, om, scalar_fwd[c], scalar_bck[c], &scalar_bck_score[c]);
        checksum += scalar_bck_score[c];
      }
    }
    scalar_bck_ms.push_back(milliseconds_since(start_scalar_bck));

    const auto start_packed_bck = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
      backward4_parser(dsq, L, om, packed_fwd, &packed_bck);
      checksum += packed_bck.score[0];
    }
    packed_bck_ms.push_back(milliseconds_since(start_packed_bck));

    const auto start_varied_scalar_bck = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
      for (int c = 0; c < kCandidates; ++c) {
        p7_BackwardParser(
            dsq[c], varied_lengths[c], varied_om[c], varied_fwd[c],
            varied_bck[c], &varied_bck_score[c]);
        checksum += varied_bck_score[c];
      }
    }
    varied_scalar_bck_ms.push_back(
        milliseconds_since(start_varied_scalar_bck));

    const auto start_varied_packed_bck = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
      backward4_parser_varlen(
          dsq, varied_lengths, om, varied_forward_xmx,
          varied_forward_xmx_counts, &varied_bck_packed);
      checksum += varied_bck_packed.score[0];
    }
    varied_packed_bck_ms.push_back(
        milliseconds_since(start_varied_packed_bck));
  }

  const double scalar_fwd_median = median(scalar_fwd_ms);
  const double packed_fwd_median = median(packed_fwd_ms);
  const double scalar_bck_median = median(scalar_bck_ms);
  const double packed_bck_median = median(packed_bck_ms);
  const double varied_scalar_fwd_median = median(varied_scalar_fwd_ms);
  const double varied_packed_fwd_median = median(varied_packed_fwd_ms);
  const double varied_scalar_bck_median = median(varied_scalar_bck_ms);
  const double varied_packed_bck_median = median(varied_packed_bck_ms);
  std::cout << "{\n"
            << "  \"status\": \"PASS\",\n"
            << "  \"model\": \"" << (hmm->name == nullptr ? "" : hmm->name) << "\",\n"
            << "  \"M\": " << hmm->M << ",\n"
            << "  \"L\": " << L << ",\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"forward_scalar_four_median_ms\": " << scalar_fwd_median << ",\n"
            << "  \"forward_avx512_four_median_ms\": " << packed_fwd_median << ",\n"
            << "  \"forward_speedup\": " << scalar_fwd_median / packed_fwd_median << ",\n"
            << "  \"backward_scalar_four_median_ms\": " << scalar_bck_median << ",\n"
            << "  \"backward_avx512_four_median_ms\": " << packed_bck_median << ",\n"
            << "  \"backward_speedup\": " << scalar_bck_median / packed_bck_median << ",\n"
            << "  \"varied_lengths\": [" << varied_lengths[0] << ", "
            << varied_lengths[1] << ", " << varied_lengths[2] << ", "
            << varied_lengths[3] << "],\n"
            << "  \"varied_forward_scalar_four_median_ms\": "
            << varied_scalar_fwd_median << ",\n"
            << "  \"varied_forward_avx512_four_median_ms\": "
            << varied_packed_fwd_median << ",\n"
            << "  \"varied_forward_speedup\": "
            << varied_scalar_fwd_median / varied_packed_fwd_median << ",\n"
            << "  \"varied_backward_scalar_four_median_ms\": "
            << varied_scalar_bck_median << ",\n"
            << "  \"varied_backward_avx512_four_median_ms\": "
            << varied_packed_bck_median << ",\n"
            << "  \"varied_backward_speedup\": "
            << varied_scalar_bck_median / varied_packed_bck_median << ",\n"
            << "  \"checksum\": " << checksum << "\n"
            << "}\n";

  for (auto *matrix : scalar_fwd) p7_omx_Destroy(matrix);
  for (auto *matrix : scalar_bck) p7_omx_Destroy(matrix);
  for (auto *matrix : varied_fwd) p7_omx_Destroy(matrix);
  for (auto *matrix : varied_bck) p7_omx_Destroy(matrix);
  for (auto *profile : varied_om) p7_oprofile_Destroy(profile);
  p7_oprofile_Destroy(om);
  p7_profile_Destroy(gm);
  p7_bg_Destroy(bg);
  p7_hmm_Destroy(hmm);
  esl_alphabet_Destroy(abc);
  return 0;
}
#endif
