#ifndef PLAN7_GPU_BIAS_CUDA_H
#define PLAN7_GPU_BIAS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum plan7_bias_action {
  PLAN7_BIAS_CPU_REQUIRED = 0,
  PLAN7_BIAS_DEFINITE_REJECT = 1,
  PLAN7_BIAS_DEFINITE_PASS = 2
};

enum plan7_bias_cutoff_mode {
  PLAN7_BIAS_CUTOFF_INVALID = 0,
  PLAN7_BIAS_CUTOFF_SCORE = 1,
  PLAN7_BIAS_CUTOFF_ALWAYS_REJECT = 2,
  PLAN7_BIAS_CUTOFF_ALWAYS_PASS = 3
};

/* A CUDA math target admitted by the bias-filter correctness attestation.
 * Keep these values stable: they are also exposed in the private Python test
 * and provenance API. */
enum plan7_bias_cuda_target {
  PLAN7_BIAS_CUDA_UNATTESTED = 0,
  PLAN7_BIAS_CUDA_SM75_RTX2080_TI = 1,
  PLAN7_BIAS_CUDA_SM90_H200 = 2
};

enum { PLAN7_BIAS_LIBM_BUILD_ID_SIZE = 20 };

/* Runtime identity captured before any exact bias-filter kernel is launched.
 * UUID and PCI address are provenance, rather than allow-list keys: all full
 * H200 devices of the admitted product/architecture may be used. */
typedef struct {
  int32_t device_ordinal;
  int32_t runtime_version;
  int32_t driver_version;
  int32_t cudart_version;
  int32_t nvcc_major;
  int32_t nvcc_minor;
  int32_t nvcc_build;
  int32_t compute_major;
  int32_t compute_minor;
  int32_t multiprocessor_count;
  int32_t pci_domain_id;
  int32_t pci_bus_id;
  int32_t pci_device_id;
  uint64_t total_global_memory;
  uint8_t uuid[16];
  char pci_bus_address[32];
  char name[256];
} plan7_bias_cuda_identity;

/* Host CPUID state that affects the attested scalar reference calculations.
 * This separate value object keeps the target allow-list host-unit-testable
 * without requiring either admitted GPU. */
typedef struct {
  uint32_t vendor_ebx;
  uint32_t vendor_edx;
  uint32_t vendor_ecx;
  uint32_t family;
  uint32_t model;
  uint32_t stepping;
  uint32_t leaf1_ecx;
  uint32_t leaf1_edx;
  uint32_t leaf7_ebx;
  uint32_t xcr0_low;
  uint32_t xcr0_high;
} plan7_bias_host_identity;

typedef struct {
  float t10;
  float t11;
  float scale;
  float cutoff_bit_score;
  int32_t cutoff_mode;
  uint32_t reserved;
  float pi0;
  float pi1;
  float t02;
  float t12;
  float eo[29][2];
} plan7_bias_profile;

typedef struct {
  uint32_t profile_index;
  uint32_t sequence_index;
} plan7_bias_candidate;

typedef struct {
  int16_t numerator;
  uint8_t status;
  uint8_t reserved;
} plan7_bias_ssv_input;

typedef struct {
  uint32_t sequence_index;
  float filtersc;
  int16_t ssv_numerator;
  uint8_t ssv_status;
  uint8_t action;
} plan7_bias_result;

int plan7_bias_pack_amino_profile(const float *background,
                                  const float *composition,
                                  int model_length,
                                  float scale,
                                  int cutoff_mode,
                                  float cutoff_bit_score,
                                  plan7_bias_profile *profile,
                                  char *error,
                                  size_t error_size);

int plan7_bias_length_terms(uint64_t length,
                            float *length_logp,
                            float *length_log1mp);

int plan7_bias_filter_score_host(const plan7_bias_profile *profile,
                                 const uint8_t *residues,
                                 uint64_t length,
                                 float *filtersc);

int plan7_bias_rebias_decision(uint8_t ssv_status,
                               int16_t ssv_numerator,
                               float scale,
                               float filtersc,
                               int cutoff_mode,
                               float cutoff_bit_score,
                               float *bit_score);

/* Returns 1 only when host binary32/binary64 operations use the attested
 * rounding and denormal modes. */
int plan7_bias_host_environment_attested(void);

/* Capture current identities. These do not attest them and never launch a
 * kernel. They are exposed for immutable benchmark provenance. */
int plan7_bias_current_cuda_identity(plan7_bias_cuda_identity *identity,
                                     char *reason,
                                     size_t reason_size);
int plan7_bias_current_host_identity(plan7_bias_host_identity *identity,
                                     char *reason,
                                     size_t reason_size);

/* Pure host-side allow-list boundaries used by the runtime gate and unit
 * tests. CUDA identity returns a plan7_bias_cuda_target value. */
int plan7_bias_cuda_identity_target(
  const plan7_bias_cuda_identity *identity,
  char *reason,
  size_t reason_size);
int plan7_bias_host_identity_attested(
  const plan7_bias_host_identity *identity,
  int cuda_target,
  char *reason,
  size_t reason_size);

/* Capture the loaded libm GNU build ID and apply the target-specific exact
 * allow-list. The pure predicate keeps this boundary independently testable. */
int plan7_bias_current_libm_build_id(
  uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE]);
int plan7_bias_libm_build_id_attested(
  const uint8_t *build_id,
  size_t build_id_size,
  int cuda_target);

/* Returns 1 only for an explicitly admitted host/device math target. */
int plan7_bias_environment_attested(char *reason, size_t reason_size);

/* Environment classification behind plan7_bias_environment_attested.
 *
 * The bias filter, Forward and domain kernels evaluate log() on the device;
 * the CPU pipeline evaluates the host libm log(). They agree only when the two
 * implementations agree. An environment is trusted in one of two ways:
 *   BUILTIN: the exact libm build, glibc version and CPU signature attested
 *            offline when the target was admitted (the original allow-list);
 *   SELF:    an exhaustive in-process audit (plan7_bias_self_attest) compared
 *            the device and host log over every positive finite binary32
 *            input on this libm build, CPU identity and admitted device, or a
 *            cached record of that audit was re-bound to the live identity
 *            (plan7_bias_register_attestation).
 * Negative codes say which requirement failed; only NOT_ATTESTED can be
 * resolved by self-attestation. */
enum plan7_bias_environment_code {
  PLAN7_BIAS_ENV_ATTESTED_BUILTIN = 1,
  PLAN7_BIAS_ENV_ATTESTED_SELF = 2,
  PLAN7_BIAS_ENV_UNSUPPORTED_PLATFORM = -1,
  PLAN7_BIAS_ENV_FLOAT_ENVIRONMENT = -2,
  PLAN7_BIAS_ENV_CUDA_DEVICE = -3,
  PLAN7_BIAS_ENV_HOST_FEATURES = -4,
  PLAN7_BIAS_ENV_LIBM_UNAVAILABLE = -5,
  PLAN7_BIAS_ENV_NOT_ATTESTED = -6
};

/* Version of the exhaustive audit below. Cached records from a different
 * version are never accepted. */
enum { PLAN7_BIAS_ATTESTATION_ALGORITHM_VERSION = 1 };

/* Every positive finite binary32 value: bit patterns 0x00000001..0x7f7fffff. */
enum { PLAN7_BIAS_ATTESTATION_INPUT_COUNT = 2139095039 };

typedef struct {
  int32_t algorithm_version;
  int32_t cuda_target;
  int32_t host_threads;
  int32_t attested;
  uint64_t inputs_checked;
  uint64_t float_mismatches;
  uint64_t double_mismatches;
  uint32_t maximum_float_ulp;
  uint32_t first_float_mismatch_input;
  uint32_t first_float_mismatch_host;
  uint32_t first_float_mismatch_device;
  uint64_t maximum_double_ulp;
  uint32_t first_double_mismatch_input;
  uint32_t reserved;
  double seconds;
  uint8_t libm_build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE];
  uint8_t reserved_padding[4];
  plan7_bias_host_identity host;
  uint8_t reserved_host_padding[4];
  plan7_bias_cuda_identity cuda;
} plan7_bias_self_attestation;

/* Classify the live environment; returns a plan7_bias_environment_code. */
int plan7_bias_classify_environment(char *reason, size_t reason_size);

/* Exhaustively compare the device log used by the production kernels with the
 * host libm log() over every positive finite binary32 input, at the binary32
 * rounding used by the bias filter and at binary64 (reported). With zero
 * binary32 mismatches the live libm build, CPU identity and device target are
 * registered as trusted for this process. Returns 0 when the audit ran (see
 * report->attested), -1 when it could not run (reason set). */
int plan7_bias_self_attest(int host_threads,
                           plan7_bias_self_attestation *report,
                           char *reason,
                           size_t reason_size);

/* Re-bind a cached audit to this process. Registers and returns 1 only when the
 * libm build ID, the complete host identity and the device target all equal
 * the live environment, the platform requirements hold and the record's audit
 * is the current algorithm with zero binary32 mismatches; otherwise returns 0
 * and leaves the registration unchanged. */
int plan7_bias_register_attestation(const uint8_t *build_id,
                                    size_t build_id_size,
                                    const plan7_bias_host_identity *host,
                                    int cuda_target,
                                    int algorithm_version,
                                    uint64_t inputs_checked,
                                    uint64_t float_mismatches,
                                    char *reason,
                                    size_t reason_size);

/* Forget any registration (tests and diagnostics). */
void plan7_bias_clear_attestation(void);

/* Device identity for one visible CUDA ordinal, without changing the calling
 * thread's current device. */
int plan7_bias_cuda_device_count(int *count, char *reason, size_t reason_size);
int plan7_bias_cuda_identity_for_device(int device_ordinal,
                                        plan7_bias_cuda_identity *identity,
                                        char *reason,
                                        size_t reason_size);

/* Internal device-buffer entry point used by the resident SSV batch. */
int plan7_bias_filter_candidates_device(
  const uint8_t *device_residues,
  const uint64_t *device_offsets,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_ssv_input *device_ssv_inputs,
  size_t candidate_count,
  plan7_bias_result *device_results,
  char *error,
  size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
