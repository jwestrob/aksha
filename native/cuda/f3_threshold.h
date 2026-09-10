#ifndef PLAN7_GPU_F3_THRESHOLD_H
#define PLAN7_GPU_F3_THRESHOLD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_f3_threshold_reason {
  PLAN7_F3_THRESHOLD_REASON_NONE = 0,
  PLAN7_F3_THRESHOLD_REASON_INVALID_PARAMETERS = 1,
  PLAN7_F3_THRESHOLD_REASON_NO_NUMERIC_PASS = 2,
  PLAN7_F3_THRESHOLD_REASON_CERTIFICATE_FAILED = 3
};

/* Exact host-compiled boundary for the HMMER Forward/F3 predicate.
 *
 * For supported parameters, every non-NaN binary32 bit_score is accepted by
 * the host oracle if and only if bit_score >= threshold. Infinities and both
 * signed zero encodings are part of the ordered search domain. A NaN score is
 * deliberately excluded: HMMER's `P > F3` predicate accepts a NaN P-value,
 * whereas an ordered device comparison rejects NaN. Production callers must
 * therefore retain their existing non-NaN input guard (or fall back).
 */
typedef struct plan7_f3_threshold {
  uint32_t tau_bits;
  uint32_t lambda_bits;
  uint64_t f3_bits;
  uint32_t threshold_bits;
  uint32_t predecessor_bits;
  uint32_t successor_bits;
  uint8_t supported;
  uint8_t reason;
  uint8_t has_predecessor;
  uint8_t has_successor;
  uint8_t negative_infinity_pass;
  uint8_t predecessor_pass;
  uint8_t threshold_pass;
  uint8_t successor_pass;
  uint8_t positive_infinity_pass;
  uint8_t quiet_nan_oracle_pass;
  uint8_t nan_requires_fallback;
  uint8_t reserved;
} plan7_f3_threshold;

/* Exact host-compiled boundary for either HMMER Gumbel/F2 predicate.  For
 * supported parameters, every non-NaN binary32 bit_score survives F2 if and
 * only if it is at or above threshold_bits.  The same NaN exclusion and
 * predecessor/threshold/successor certificate used by F3 applies here. */
typedef struct plan7_f2_threshold {
  uint32_t mu_bits;
  uint32_t lambda_bits;
  uint64_t f2_bits;
  uint32_t threshold_bits;
  uint32_t predecessor_bits;
  uint32_t successor_bits;
  uint8_t supported;
  uint8_t reason;
  uint8_t has_predecessor;
  uint8_t has_successor;
  uint8_t negative_infinity_pass;
  uint8_t predecessor_pass;
  uint8_t threshold_pass;
  uint8_t successor_pass;
  uint8_t positive_infinity_pass;
  uint8_t quiet_nan_oracle_pass;
  uint8_t nan_requires_fallback;
  uint8_t reserved;
} plan7_f2_threshold;

/* Compile by binary-searching the same esl_exp_surv() predicate that HMMER
 * uses. Returns 0 for both supported and conservative-unsupported results;
 * inspect result->supported/reason. Returns -1 only for a null result pointer.
 */
int plan7_forward_compile_f3_threshold(float tau, float lambda, double f3,
                                       plan7_f3_threshold *result);

/* Exact host oracle exposed only for boundary validation. The score is passed
 * as raw binary32 bits so tests cover signed zero, infinities, and NaNs.
 */
int plan7_forward_f3_oracle_pass_bits(uint32_t bit_score_bits, float tau,
                                      float lambda, double f3);

/* Compile and expose the exact linked esl_gumbel_surv() predicate used by
 * HMMER's MSV/Viterbi F2 gates. */
int plan7_postfilter_compile_f2_threshold(float mu, float lambda, double f2,
                                          plan7_f2_threshold *result);
int plan7_postfilter_f2_oracle_pass_bits(uint32_t bit_score_bits, float mu,
                                         float lambda, double f2);

#ifdef __cplusplus
}
#endif

#endif
