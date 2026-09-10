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
