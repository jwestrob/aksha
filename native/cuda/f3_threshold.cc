#include "f3_threshold.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" {
#include <esl_exponential.h>
#include <esl_gumbel.h>
}

namespace {

static_assert(sizeof(float) == sizeof(uint32_t),
              "F3 threshold compilation requires binary32 float");
static_assert(sizeof(double) == sizeof(uint64_t),
              "F3 threshold provenance requires binary64 double");
static_assert(std::numeric_limits<float>::is_iec559,
              "F3 threshold compilation requires IEEE-754 float");
static_assert(std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<float>::digits == 24 &&
                  std::numeric_limits<float>::max_exponent == 128,
              "F3 threshold compilation requires IEEE binary32");
static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53,
              "F3 threshold compilation requires IEEE binary64");

constexpr uint32_t kSignBit = UINT32_C(0x80000000);
constexpr uint32_t kNegativeInfinityBits = UINT32_C(0xff800000);
constexpr uint32_t kPositiveInfinityBits = UINT32_C(0x7f800000);
constexpr uint32_t kQuietNanBits = UINT32_C(0x7fc00000);

template <typename To, typename From>
To bit_copy(const From &source) {
  static_assert(sizeof(To) == sizeof(From), "bit-copy width changed");
  To destination;
  std::memcpy(&destination, &source, sizeof(destination));
  return destination;
}

uint32_t ordered_key(uint32_t bits) {
  return (bits & kSignBit) != 0 ? ~bits : bits ^ kSignBit;
}

uint32_t bits_from_ordered_key(uint32_t key) {
  return (key & kSignBit) != 0 ? key ^ kSignBit : ~key;
}

bool oracle_pass(uint32_t bit_score_bits, float tau, float lambda, double f3) {
  const float bit_score = bit_copy<float>(bit_score_bits);
  const double probability = esl_exp_surv(
      static_cast<double>(bit_score), static_cast<double>(tau),
      static_cast<double>(lambda));
  return !(probability > f3);
}

bool gumbel_oracle_pass(uint32_t bit_score_bits, float mu, float lambda,
                        double f2) {
  const float bit_score = bit_copy<float>(bit_score_bits);
  const double probability = esl_gumbel_surv(
      static_cast<double>(bit_score), static_cast<double>(mu),
      static_cast<double>(lambda));
  return !(probability > f2);
}

void mark_unsupported(plan7_f3_threshold *result,
                      plan7_f3_threshold_reason reason) {
  result->supported = 0;
  result->reason = static_cast<uint8_t>(reason);
}

void mark_unsupported(plan7_f2_threshold *result,
                      plan7_f3_threshold_reason reason) {
  result->supported = 0;
  result->reason = static_cast<uint8_t>(reason);
}

}  // namespace

extern "C" int plan7_forward_f3_oracle_pass_bits(
    uint32_t bit_score_bits, float tau, float lambda, double f3) {
  return oracle_pass(bit_score_bits, tau, lambda, f3) ? 1 : 0;
}

extern "C" int plan7_forward_compile_f3_threshold(
    float tau, float lambda, double f3, plan7_f3_threshold *result) {
  if (result == nullptr) return -1;
  std::memset(result, 0, sizeof(*result));
  result->tau_bits = bit_copy<uint32_t>(tau);
  result->lambda_bits = bit_copy<uint32_t>(lambda);
  result->f3_bits = bit_copy<uint64_t>(f3);

  if (!std::isfinite(tau) || !std::isfinite(lambda) || lambda <= 0.0f ||
      !std::isfinite(f3) || f3 < 0.0 || f3 > 1.0) {
    mark_unsupported(result, PLAN7_F3_THRESHOLD_REASON_INVALID_PARAMETERS);
    return 0;
  }

  const uint32_t first_key = ordered_key(kNegativeInfinityBits);
  const uint32_t last_key = ordered_key(kPositiveInfinityBits);
  const bool negative_infinity_pass =
      oracle_pass(kNegativeInfinityBits, tau, lambda, f3);
  const bool positive_infinity_pass =
      oracle_pass(kPositiveInfinityBits, tau, lambda, f3);
  result->negative_infinity_pass = negative_infinity_pass ? 1 : 0;
  result->positive_infinity_pass = positive_infinity_pass ? 1 : 0;
  result->quiet_nan_oracle_pass =
      oracle_pass(kQuietNanBits, tau, lambda, f3) ? 1 : 0;
  result->nan_requires_fallback = result->quiet_nan_oracle_pass;

  if (!positive_infinity_pass) {
    mark_unsupported(result, PLAN7_F3_THRESHOLD_REASON_NO_NUMERIC_PASS);
    return 0;
  }

  uint32_t threshold_key = first_key;
  if (!negative_infinity_pass) {
    uint32_t reject_key = first_key;
    uint32_t pass_key = last_key;
    while (pass_key - reject_key > 1) {
      const uint32_t middle_key =
          reject_key + (pass_key - reject_key) / 2;
      if (oracle_pass(bits_from_ordered_key(middle_key), tau, lambda, f3))
        pass_key = middle_key;
      else
        reject_key = middle_key;
    }
    threshold_key = pass_key;
  }

  result->threshold_bits = bits_from_ordered_key(threshold_key);
  result->threshold_pass =
      oracle_pass(result->threshold_bits, tau, lambda, f3) ? 1 : 0;
  if (threshold_key != first_key) {
    result->has_predecessor = 1;
    result->predecessor_bits = bits_from_ordered_key(threshold_key - 1);
    result->predecessor_pass =
        oracle_pass(result->predecessor_bits, tau, lambda, f3) ? 1 : 0;
  }
  if (threshold_key != last_key) {
    result->has_successor = 1;
    result->successor_bits = bits_from_ordered_key(threshold_key + 1);
    result->successor_pass =
        oracle_pass(result->successor_bits, tau, lambda, f3) ? 1 : 0;
  }

  if (!result->threshold_pass ||
      (result->has_predecessor && result->predecessor_pass) ||
      (result->has_successor && !result->successor_pass)) {
    mark_unsupported(result, PLAN7_F3_THRESHOLD_REASON_CERTIFICATE_FAILED);
    return 0;
  }

  result->supported = 1;
  result->reason = PLAN7_F3_THRESHOLD_REASON_NONE;
  return 0;
}

extern "C" int plan7_postfilter_f2_oracle_pass_bits(
    uint32_t bit_score_bits, float mu, float lambda, double f2) {
  return gumbel_oracle_pass(bit_score_bits, mu, lambda, f2) ? 1 : 0;
}

extern "C" int plan7_postfilter_compile_f2_threshold(
    float mu, float lambda, double f2, plan7_f2_threshold *result) {
  if (result == nullptr) return -1;
  std::memset(result, 0, sizeof(*result));
  result->mu_bits = bit_copy<uint32_t>(mu);
  result->lambda_bits = bit_copy<uint32_t>(lambda);
  result->f2_bits = bit_copy<uint64_t>(f2);

  if (!std::isfinite(mu) || !std::isfinite(lambda) || lambda <= 0.0f ||
      !std::isfinite(f2) || f2 < 0.0 || f2 > 1.0) {
    mark_unsupported(result, PLAN7_F3_THRESHOLD_REASON_INVALID_PARAMETERS);
    return 0;
  }

  const uint32_t first_key = ordered_key(kNegativeInfinityBits);
  const uint32_t last_key = ordered_key(kPositiveInfinityBits);
  const bool negative_infinity_pass =
      gumbel_oracle_pass(kNegativeInfinityBits, mu, lambda, f2);
  const bool positive_infinity_pass =
      gumbel_oracle_pass(kPositiveInfinityBits, mu, lambda, f2);
  result->negative_infinity_pass = negative_infinity_pass ? 1 : 0;
  result->positive_infinity_pass = positive_infinity_pass ? 1 : 0;
  result->quiet_nan_oracle_pass =
      gumbel_oracle_pass(kQuietNanBits, mu, lambda, f2) ? 1 : 0;
  result->nan_requires_fallback = result->quiet_nan_oracle_pass;

  if (!positive_infinity_pass) {
    mark_unsupported(result, PLAN7_F3_THRESHOLD_REASON_NO_NUMERIC_PASS);
    return 0;
  }

  uint32_t threshold_key = first_key;
  if (!negative_infinity_pass) {
    uint32_t reject_key = first_key;
    uint32_t pass_key = last_key;
    while (pass_key - reject_key > 1) {
      const uint32_t middle_key =
          reject_key + (pass_key - reject_key) / 2;
      if (gumbel_oracle_pass(
              bits_from_ordered_key(middle_key), mu, lambda, f2))
        pass_key = middle_key;
      else
        reject_key = middle_key;
    }
    threshold_key = pass_key;
  }

  result->threshold_bits = bits_from_ordered_key(threshold_key);
  result->threshold_pass = gumbel_oracle_pass(
      result->threshold_bits, mu, lambda, f2) ? 1 : 0;
  if (threshold_key != first_key) {
    result->has_predecessor = 1;
    result->predecessor_bits = bits_from_ordered_key(threshold_key - 1);
    result->predecessor_pass = gumbel_oracle_pass(
        result->predecessor_bits, mu, lambda, f2) ? 1 : 0;
  }
  if (threshold_key != last_key) {
    result->has_successor = 1;
    result->successor_bits = bits_from_ordered_key(threshold_key + 1);
    result->successor_pass = gumbel_oracle_pass(
        result->successor_bits, mu, lambda, f2) ? 1 : 0;
  }

  if (!result->threshold_pass ||
      (result->has_predecessor && result->predecessor_pass) ||
      (result->has_successor && !result->successor_pass)) {
    mark_unsupported(result, PLAN7_F3_THRESHOLD_REASON_CERTIFICATE_FAILED);
    return 0;
  }

  result->supported = 1;
  result->reason = PLAN7_F3_THRESHOLD_REASON_NONE;
  return 0;
}
