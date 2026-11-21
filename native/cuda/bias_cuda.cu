#include "bias_cuda.h"

#include <cuda_runtime.h>

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <link.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__GLIBC__)
#include <cpuid.h>
#include <gnu/libc-version.h>
#include <xmmintrin.h>
#endif

static_assert(sizeof(float) == 4,
              "bias ABI requires binary32 float");
static_assert(sizeof(plan7_bias_profile) == 272,
              "plan7_bias_profile ABI size changed");
static_assert(sizeof(plan7_bias_candidate) == 8,
              "plan7_bias_candidate ABI size changed");
static_assert(sizeof(plan7_bias_ssv_input) == 4,
              "plan7_bias_ssv_input ABI size changed");
static_assert(sizeof(plan7_bias_result) == 12,
              "plan7_bias_result ABI size changed");
static_assert(sizeof(plan7_bias_cuda_identity) == 368,
              "plan7_bias_cuda_identity ABI size changed");
static_assert(sizeof(plan7_bias_host_identity) == 44,
              "plan7_bias_host_identity ABI size changed");
static_assert(sizeof(plan7_bias_self_attestation) == 520,
              "plan7_bias_self_attestation ABI size changed");

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kMaximumTargetLength = 100000;
constexpr double kLog2 = 0.69314718055994529;

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

bool
uuid_is_zero(const uint8_t uuid[16])
{
  uint8_t combined = 0;
  for (size_t index = 0; index < 16; ++index) combined |= uuid[index];
  return combined == 0;
}

float
host_add(float left, float right)
{
  const float result = left + right;
  return result;
}

float
host_mul(float left, float right)
{
  const float result = left * right;
  return result;
}

float
host_div(float left, float right)
{
  const float result = left / right;
  return result;
}

bool
valid_profile_inputs(const float *background,
                     const float *composition,
                     int model_length,
                     float scale,
                     int cutoff_mode,
                     float cutoff_bit_score)
{
  if (background == nullptr || composition == nullptr || model_length < 1 ||
      model_length > static_cast<int>(kMaximumTargetLength) ||
      !isfinite(scale) || scale <= 0.0f ||
      cutoff_mode < PLAN7_BIAS_CUTOFF_INVALID ||
      cutoff_mode > PLAN7_BIAS_CUTOFF_ALWAYS_PASS ||
      (cutoff_mode == PLAN7_BIAS_CUTOFF_SCORE &&
       !isfinite(cutoff_bit_score)))
    return false;
  for (int residue = 0; residue < 20; ++residue) {
    if (!isfinite(background[residue]) || background[residue] <= 0.0f ||
        !isfinite(composition[residue]) || composition[residue] < 0.0f)
      return false;
  }
  return true;
}

struct build_id_search {
  uintptr_t symbol;
  bool module_found;
  bool build_id_found;
  uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE];
};

int
find_attested_build_id(struct dl_phdr_info *info, size_t, void *opaque)
{
  build_id_search *search = static_cast<build_id_search *>(opaque);
  bool contains_symbol = false;
  for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
    const ElfW(Phdr) &header = info->dlpi_phdr[index];
    if (header.p_type != PT_LOAD) continue;
    const uintptr_t first =
      static_cast<uintptr_t>(info->dlpi_addr) + header.p_vaddr;
    const uintptr_t last = first + header.p_memsz;
    if (search->symbol >= first && search->symbol < last) {
      contains_symbol = true;
      break;
    }
  }
  if (!contains_symbol) return 0;
  search->module_found = true;

  for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
    const ElfW(Phdr) &header = info->dlpi_phdr[index];
    if (header.p_type != PT_NOTE) continue;
    const unsigned char *cursor = reinterpret_cast<const unsigned char *>(
      static_cast<uintptr_t>(info->dlpi_addr) + header.p_vaddr);
    const unsigned char *end = cursor + header.p_memsz;
    while (cursor + sizeof(ElfW(Nhdr)) <= end) {
      const ElfW(Nhdr) *note = reinterpret_cast<const ElfW(Nhdr) *>(cursor);
      cursor += sizeof(*note);
      const size_t name_size = (note->n_namesz + 3U) & ~size_t(3U);
      const size_t descriptor_size = (note->n_descsz + 3U) & ~size_t(3U);
      if (name_size > static_cast<size_t>(end - cursor)) break;
      const unsigned char *name = cursor;
      cursor += name_size;
      if (descriptor_size > static_cast<size_t>(end - cursor)) break;
      const unsigned char *descriptor = cursor;
      cursor += descriptor_size;
      if (note->n_type == NT_GNU_BUILD_ID && note->n_namesz == 4 &&
          memcmp(name, "GNU", 4) == 0 &&
          note->n_descsz == PLAN7_BIAS_LIBM_BUILD_ID_SIZE) {
        memcpy(search->build_id, descriptor,
               PLAN7_BIAS_LIBM_BUILD_ID_SIZE);
        search->build_id_found = true;
        return 1;
      }
    }
  }
  return 1;
}

bool
loaded_libm_build_id(uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE])
{
  void *symbol = dlsym(RTLD_DEFAULT, "log");
  if (symbol == nullptr) return false;
  build_id_search search = {
    reinterpret_cast<uintptr_t>(symbol), false, false, {}
  };
  dl_iterate_phdr(find_attested_build_id, &search);
  if (!search.module_found || !search.build_id_found) return false;
  memcpy(build_id, search.build_id, PLAN7_BIAS_LIBM_BUILD_ID_SIZE);
  return true;
}

void
set_degenerate_odds(const float *background,
                    const float *composition,
                    uint32_t mask,
                    float odds[2])
{
  float normal_numerator = 0.0f;
  float bias_numerator = 0.0f;
  float denominator = 0.0f;
  for (int residue = 0; residue < 20; ++residue) {
    if ((mask & (UINT32_C(1) << residue)) == 0) continue;
    normal_numerator = host_add(normal_numerator, background[residue]);
    bias_numerator = host_add(bias_numerator, composition[residue]);
    denominator = host_add(denominator, background[residue]);
  }
  odds[0] = denominator > 0.0f
              ? host_div(normal_numerator, denominator)
              : 0.0f;
  odds[1] = denominator > 0.0f
              ? host_div(bias_numerator, denominator)
              : 0.0f;
}

__device__ __forceinline__ float
device_add(float left, float right)
{
  return __fadd_rn(left, right);
}

__device__ __forceinline__ float
device_mul(float left, float right)
{
  return __fmul_rn(left, right);
}

__device__ __forceinline__ float
device_div(float left, float right)
{
  return __fdiv_rn(left, right);
}

__device__ __forceinline__ float
device_log_to_float(float value)
{
  return __double2float_rn(log(static_cast<double>(value)));
}

/* Self-attestation kernel: the production binary32 log of the bias filter and
 * the binary64 log used by the Forward/domain kernels, for one input range. */
__global__ void
bias_log_attestation_kernel(uint32_t first,
                            uint32_t count,
                            float *binary32,
                            double *binary64)
{
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float value = __uint_as_float(first + index);
  binary32[index] = device_log_to_float(value);
  binary64[index] = log(static_cast<double>(value));
}

__device__ __forceinline__ bool
bias_filter_score(const uint8_t *residues,
                  const uint64_t *offsets,
                  uint32_t sequence,
                  const plan7_bias_profile &profile,
                  float length_logp,
                  float length_log1mp,
                  float *filtersc)
{
  const uint64_t start = offsets[sequence];
  const uint64_t length_u64 = offsets[sequence + 1] - start;
  if (length_u64 == 0 || length_u64 > kMaximumTargetLength) return false;
  const int length = static_cast<int>(length_u64);
  const float length_f = __int2float_rn(length);
  const float p1 = device_div(length_f, __int2float_rn(length + 1));
  const float t00 = p1;
  const float t01 = __fsub_rn(1.0f, p1);

  unsigned residue = residues[start];
  float d0 = device_mul(profile.eo[residue][0], profile.pi0);
  float d1 = device_mul(profile.eo[residue][1], profile.pi1);
  float maximum = d0 > 0.0f ? d0 : 0.0f;
  maximum = d1 > maximum ? d1 : maximum;
  if (!(maximum > 0.0f) || !isfinite(maximum)) return false;
  d0 = device_div(d0, maximum);
  d1 = device_div(d1, maximum);
  float score = device_log_to_float(maximum);

  for (int position = 1; position < length; ++position) {
    residue = residues[start + static_cast<uint64_t>(position)];
    float next0 = device_add(device_mul(d0, t00),
                             device_mul(d1, profile.t10));
    next0 = device_mul(next0, profile.eo[residue][0]);
    float next1 = device_add(device_mul(d0, t01),
                             device_mul(d1, profile.t11));
    next1 = device_mul(next1, profile.eo[residue][1]);
    maximum = next0 > 0.0f ? next0 : 0.0f;
    maximum = next1 > maximum ? next1 : maximum;
    if (!(maximum > 0.0f) || !isfinite(maximum)) return false;
    d0 = device_div(next0, maximum);
    d1 = device_div(next1, maximum);
    score = device_add(score, device_log_to_float(maximum));
  }

  const float end = device_add(device_mul(d0, profile.t02),
                               device_mul(d1, profile.t12));
  if (!(end > 0.0f) || !isfinite(end)) return false;
  score = device_add(score, device_log_to_float(end));
  score = device_add(score, length_logp);
  score = device_add(score, length_log1mp);
  if (!isfinite(score)) return false;
  *filtersc = score;
  return true;
}

__global__ void
bias_candidates_kernel(const uint8_t *residues,
                       const uint64_t *offsets,
                       const float *length_logp,
                       const float *length_log1mp,
                       const plan7_bias_profile *profiles,
                       const plan7_bias_candidate *candidates,
                       const plan7_bias_ssv_input *ssv_inputs,
                       size_t candidate_count,
                       plan7_bias_result *results)
{
  const size_t candidate_index =
    static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate_index >= candidate_count) return;

  const plan7_bias_candidate candidate = candidates[candidate_index];
  const plan7_bias_ssv_input ssv = ssv_inputs[candidate_index];
  const plan7_bias_profile &profile = profiles[candidate.profile_index];
  plan7_bias_result result = {
    candidate.sequence_index, NAN, ssv.numerator, ssv.status,
    PLAN7_BIAS_CPU_REQUIRED
  };
  if (ssv.status != 0) {
    results[candidate_index] = result;
    return;
  }

  float filtersc;
  if (!bias_filter_score(residues, offsets, candidate.sequence_index, profile,
                         length_logp[candidate.sequence_index],
                         length_log1mp[candidate.sequence_index], &filtersc)) {
    results[candidate_index] = result;
    return;
  }
  result.filtersc = filtersc;

  float usc = __int2float_rn(static_cast<int>(ssv.numerator));
  usc = device_div(usc, profile.scale);
  usc = __fsub_rn(usc, 3.0f);
  const float delta = __fsub_rn(usc, filtersc);
  const float bit_score = __double2float_rn(
    __ddiv_rn(static_cast<double>(delta), kLog2));
  if (!isfinite(usc) || !isfinite(bit_score)) {
    results[candidate_index] = result;
    return;
  }

  switch (profile.cutoff_mode) {
    case PLAN7_BIAS_CUTOFF_SCORE:
      if (isfinite(profile.cutoff_bit_score))
        result.action = bit_score < profile.cutoff_bit_score
                          ? PLAN7_BIAS_DEFINITE_REJECT
                          : PLAN7_BIAS_DEFINITE_PASS;
      break;
    case PLAN7_BIAS_CUTOFF_ALWAYS_REJECT:
      result.action = PLAN7_BIAS_DEFINITE_REJECT;
      break;
    case PLAN7_BIAS_CUTOFF_ALWAYS_PASS:
      result.action = PLAN7_BIAS_DEFINITE_PASS;
      break;
    default:
      break;
  }
  results[candidate_index] = result;
}

}  // namespace

extern "C" int
plan7_bias_pack_amino_profile(const float *background,
                              const float *composition,
                              int model_length,
                              float scale,
                              int cutoff_mode,
                              float cutoff_bit_score,
                              plan7_bias_profile *profile,
                              char *error,
                              size_t error_size)
{
  if (plan7_bias_host_environment_attested() != 1) {
    set_error(error, error_size,
              "bias host floating-point environment is not attested");
    return -1;
  }
  if (profile == nullptr ||
      !valid_profile_inputs(background, composition, model_length, scale,
                            cutoff_mode, cutoff_bit_score)) {
    set_error(error, error_size, "invalid amino bias profile");
    return -1;
  }

  memset(profile, 0, sizeof(*profile));
  const float biased_length = host_div(static_cast<float>(model_length), 8.0f);
  const float biased_denominator = host_add(biased_length, 1.0f);
  profile->t10 = host_div(1.0f, biased_denominator);
  profile->t11 = host_div(biased_length, biased_denominator);
  profile->scale = scale;
  profile->cutoff_mode = cutoff_mode;
  profile->cutoff_bit_score = cutoff_bit_score;
  profile->pi0 = 0x1.ff7ceep-1f;
  profile->pi1 = 0x1.0624dep-10f;
  profile->t02 = 1.0f;
  profile->t12 = 1.0f;

  for (int residue = 0; residue < 20; ++residue) {
    profile->eo[residue][0] = host_div(background[residue],
                                       background[residue]);
    profile->eo[residue][1] = host_div(composition[residue],
                                       background[residue]);
  }

  profile->eo[20][0] = profile->eo[20][1] = 1.0f;
  set_degenerate_odds(background, composition,
                      (UINT32_C(1) << 2) | (UINT32_C(1) << 11),
                      profile->eo[21]);
  set_degenerate_odds(background, composition,
                      (UINT32_C(1) << 7) | (UINT32_C(1) << 9),
                      profile->eo[22]);
  set_degenerate_odds(background, composition,
                      (UINT32_C(1) << 3) | (UINT32_C(1) << 13),
                      profile->eo[23]);
  set_degenerate_odds(background, composition, UINT32_C(1) << 8,
                      profile->eo[24]);
  set_degenerate_odds(background, composition, UINT32_C(1) << 1,
                      profile->eo[25]);
  set_degenerate_odds(background, composition, UINT32_C(0x000fffff),
                      profile->eo[26]);
  profile->eo[27][0] = profile->eo[27][1] = 1.0f;
  profile->eo[28][0] = profile->eo[28][1] = 1.0f;
  return 0;
}

extern "C" int
plan7_bias_length_terms(uint64_t length,
                        float *length_logp,
                        float *length_log1mp)
{
  if (plan7_bias_host_environment_attested() != 1 ||
      length == 0 || length > kMaximumTargetLength ||
      length_logp == nullptr || length_log1mp == nullptr)
    return -1;
  const float length_f = static_cast<float>(length);
  const float p1 = host_div(length_f, static_cast<float>(length + 1));
  *length_logp = host_mul(length_f, logf(p1));
  *length_log1mp = logf(1.0f - p1);
  return isfinite(*length_logp) && isfinite(*length_log1mp) ? 0 : -1;
}

extern "C" int
plan7_bias_filter_score_host(const plan7_bias_profile *profile,
                             const uint8_t *residues,
                             uint64_t length,
                             float *filtersc)
{
  if (plan7_bias_host_environment_attested() != 1 ||
      profile == nullptr || residues == nullptr || filtersc == nullptr ||
      length == 0 || length > kMaximumTargetLength)
    return -1;
  for (uint64_t position = 0; position < length; ++position)
    if (residues[position] >= 29) return -1;

  const float length_f = static_cast<float>(length);
  const float p1 = host_div(length_f, static_cast<float>(length + 1));
  const float t00 = p1;
  const float t01 = 1.0f - p1;
  unsigned residue = residues[0];
  if (!isfinite(profile->pi0) || !isfinite(profile->pi1) ||
      !isfinite(profile->t02) || !isfinite(profile->t12) ||
      profile->pi0 < 0.0f || profile->pi1 < 0.0f ||
      profile->t02 < 0.0f || profile->t12 < 0.0f)
    return -1;
  float d0 = host_mul(profile->eo[residue][0], profile->pi0);
  float d1 = host_mul(profile->eo[residue][1], profile->pi1);
  float maximum = d0 > 0.0f ? d0 : 0.0f;
  maximum = d1 > maximum ? d1 : maximum;
  if (!(maximum > 0.0f) || !isfinite(maximum)) return -1;
  d0 = host_div(d0, maximum);
  d1 = host_div(d1, maximum);
  float score = static_cast<float>(log(static_cast<double>(maximum)));

  for (uint64_t position = 1; position < length; ++position) {
    residue = residues[position];
    float next0 = 0.0f;
    next0 = host_add(next0, host_mul(d0, t00));
    next0 = host_add(next0, host_mul(d1, profile->t10));
    next0 = host_mul(next0, profile->eo[residue][0]);
    float next1 = 0.0f;
    next1 = host_add(next1, host_mul(d0, t01));
    next1 = host_add(next1, host_mul(d1, profile->t11));
    next1 = host_mul(next1, profile->eo[residue][1]);
    maximum = next0 > 0.0f ? next0 : 0.0f;
    maximum = next1 > maximum ? next1 : maximum;
    if (!(maximum > 0.0f) || !isfinite(maximum)) return -1;
    d0 = host_div(next0, maximum);
    d1 = host_div(next1, maximum);
    score = host_add(score,
                     static_cast<float>(log(static_cast<double>(maximum))));
  }

  float end = 0.0f;
  end = host_add(end, host_mul(d0, profile->t02));
  end = host_add(end, host_mul(d1, profile->t12));
  if (!(end > 0.0f) || !isfinite(end)) return -1;
  score = host_add(score, static_cast<float>(log(static_cast<double>(end))));
  score = host_add(score, host_mul(length_f, logf(p1)));
  score = host_add(score, logf(1.0f - p1));
  if (!isfinite(score)) return -1;
  *filtersc = score;
  return 0;
}

extern "C" int
plan7_bias_rebias_decision(uint8_t ssv_status,
                           int16_t ssv_numerator,
                           float scale,
                           float filtersc,
                           int cutoff_mode,
                           float cutoff_bit_score,
                           float *bit_score)
{
  if (bit_score != nullptr) *bit_score = NAN;
  if (plan7_bias_host_environment_attested() != 1 ||
      ssv_status != 0 || !isfinite(scale) || scale <= 0.0f ||
      !isfinite(filtersc))
    return PLAN7_BIAS_CPU_REQUIRED;
  float usc = static_cast<float>(ssv_numerator);
  usc = host_div(usc, scale);
  usc = usc - 3.0f;
  const float delta = usc - filtersc;
  const float score = static_cast<float>(static_cast<double>(delta) / kLog2);
  if (!isfinite(usc) || !isfinite(score)) return PLAN7_BIAS_CPU_REQUIRED;
  if (bit_score != nullptr) *bit_score = score;
  switch (cutoff_mode) {
    case PLAN7_BIAS_CUTOFF_SCORE:
      if (!isfinite(cutoff_bit_score)) return PLAN7_BIAS_CPU_REQUIRED;
      return score < cutoff_bit_score ? PLAN7_BIAS_DEFINITE_REJECT
                                      : PLAN7_BIAS_DEFINITE_PASS;
    case PLAN7_BIAS_CUTOFF_ALWAYS_REJECT:
      return PLAN7_BIAS_DEFINITE_REJECT;
    case PLAN7_BIAS_CUTOFF_ALWAYS_PASS:
      return PLAN7_BIAS_DEFINITE_PASS;
    default:
      return PLAN7_BIAS_CPU_REQUIRED;
  }
}

extern "C" int
plan7_bias_host_environment_attested(void)
{
#if !defined(__x86_64__) || !defined(__GLIBC__)
  return 0;
#else
  constexpr unsigned kMxcsrRoundingFtzDazMask = UINT32_C(0x0000e040);
  return fegetround() == FE_TONEAREST &&
         (_mm_getcsr() & kMxcsrRoundingFtzDazMask) == 0;
#endif
}

namespace {

int
capture_cuda_identity(int requested_ordinal,
                      plan7_bias_cuda_identity *identity,
                      char *reason,
                      size_t reason_size)
{
  if (identity == nullptr) {
    set_error(reason, reason_size, "bias CUDA identity output is null");
    return -1;
  }
  memset(identity, 0, sizeof(*identity));
  identity->device_ordinal = -1;
  identity->pci_domain_id = -1;
  identity->pci_bus_id = -1;
  identity->pci_device_id = -1;
  identity->cudart_version = CUDART_VERSION;
  identity->nvcc_major = __CUDACC_VER_MAJOR__;
  identity->nvcc_minor = __CUDACC_VER_MINOR__;
  identity->nvcc_build = __CUDACC_VER_BUILD__;

  cudaError_t status = cudaRuntimeGetVersion(&identity->runtime_version);
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "cudaRuntimeGetVersion", status);
    return -1;
  }
  status = cudaDriverGetVersion(&identity->driver_version);
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "cudaDriverGetVersion", status);
    return -1;
  }
  if (requested_ordinal < 0) {
    status = cudaGetDevice(&identity->device_ordinal);
    if (status != cudaSuccess) {
      set_cuda_error(reason, reason_size, "cudaGetDevice", status);
      return -1;
    }
  } else {
    identity->device_ordinal = requested_ordinal;
  }

  cudaDeviceProp properties;
  status = cudaGetDeviceProperties(&properties, identity->device_ordinal);
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "cudaGetDeviceProperties", status);
    return -1;
  }
  identity->compute_major = properties.major;
  identity->compute_minor = properties.minor;
  identity->multiprocessor_count = properties.multiProcessorCount;
  identity->pci_domain_id = properties.pciDomainID;
  identity->pci_bus_id = properties.pciBusID;
  identity->pci_device_id = properties.pciDeviceID;
  identity->total_global_memory =
    static_cast<uint64_t>(properties.totalGlobalMem);
  memcpy(identity->uuid, properties.uuid.bytes, sizeof(identity->uuid));
  snprintf(identity->name, sizeof(identity->name), "%s", properties.name);
  status = cudaDeviceGetPCIBusId(
    identity->pci_bus_address, static_cast<int>(sizeof(identity->pci_bus_address)),
    identity->device_ordinal);
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "cudaDeviceGetPCIBusId", status);
    return -1;
  }
  if (reason != nullptr && reason_size != 0) reason[0] = '\0';
  return 0;
}

}  // namespace

extern "C" int
plan7_bias_current_cuda_identity(plan7_bias_cuda_identity *identity,
                                 char *reason,
                                 size_t reason_size)
{
  return capture_cuda_identity(-1, identity, reason, reason_size);
}

extern "C" int
plan7_bias_cuda_identity_for_device(int device_ordinal,
                                    plan7_bias_cuda_identity *identity,
                                    char *reason,
                                    size_t reason_size)
{
  if (device_ordinal < 0) {
    set_error(reason, reason_size, "CUDA device ordinal must be nonnegative");
    return -1;
  }
  return capture_cuda_identity(device_ordinal, identity, reason, reason_size);
}

extern "C" int
plan7_bias_cuda_device_count(int *count, char *reason, size_t reason_size)
{
  if (count == nullptr) {
    set_error(reason, reason_size, "CUDA device count output is null");
    return -1;
  }
  *count = 0;
  const cudaError_t status = cudaGetDeviceCount(count);
  if (status != cudaSuccess) {
    *count = 0;
    set_cuda_error(reason, reason_size, "cudaGetDeviceCount", status);
    return -1;
  }
  if (reason != nullptr && reason_size != 0) reason[0] = '\0';
  return 0;
}

extern "C" int
plan7_bias_current_host_identity(plan7_bias_host_identity *identity,
                                 char *reason,
                                 size_t reason_size)
{
  if (identity == nullptr) {
    set_error(reason, reason_size, "bias host identity output is null");
    return -1;
  }
  memset(identity, 0, sizeof(*identity));
#if !defined(__x86_64__) || !defined(__GLIBC__)
  set_error(reason, reason_size,
            "bias math is not attested for this host architecture");
  return -1;
#else
  const unsigned vendor_max = __get_cpuid_max(0, nullptr);
  unsigned vendor_eax;
  unsigned leaf1_eax;
  unsigned leaf1_ebx;
  unsigned leaf7_eax;
  unsigned leaf7_ecx;
  unsigned leaf7_edx;
  if (vendor_max < 7 ||
      !__get_cpuid(0, &vendor_eax, &identity->vendor_ebx,
                   &identity->vendor_ecx, &identity->vendor_edx) ||
      !__get_cpuid(1, &leaf1_eax, &leaf1_ebx,
                   &identity->leaf1_ecx, &identity->leaf1_edx) ||
      !__get_cpuid_count(7, 0, &leaf7_eax, &identity->leaf7_ebx,
                         &leaf7_ecx, &leaf7_edx)) {
    set_error(reason, reason_size, "bias CPUID identity is unavailable");
    return -1;
  }
  const unsigned base_model = (leaf1_eax >> 4) & UINT32_C(0x0f);
  const unsigned base_family = (leaf1_eax >> 8) & UINT32_C(0x0f);
  const unsigned extended_model = (leaf1_eax >> 16) & UINT32_C(0x0f);
  const unsigned extended_family = (leaf1_eax >> 20) & UINT32_C(0xff);
  identity->stepping = leaf1_eax & UINT32_C(0x0f);
  identity->family = base_family == UINT32_C(0x0f)
                       ? base_family + extended_family
                       : base_family;
  identity->model =
    (base_family == UINT32_C(0x06) || base_family == UINT32_C(0x0f))
      ? base_model | (extended_model << 4)
      : base_model;
  if ((identity->leaf1_ecx & bit_OSXSAVE) != 0) {
    __asm__ volatile("xgetbv" : "=a"(identity->xcr0_low),
                                  "=d"(identity->xcr0_high) : "c"(0));
  }
  if (reason != nullptr && reason_size != 0) reason[0] = '\0';
  return 0;
#endif
}

extern "C" int
plan7_bias_cuda_identity_target(const plan7_bias_cuda_identity *identity,
                                char *reason,
                                size_t reason_size)
{
  if (identity == nullptr) {
    set_error(reason, reason_size, "bias CUDA identity is null");
    return PLAN7_BIAS_CUDA_UNATTESTED;
  }
  if (identity->cudart_version != 12050 ||
      identity->nvcc_major != 12 || identity->nvcc_minor != 5 ||
      identity->nvcc_build != 82) {
    set_error(reason, reason_size,
              "bias math is not attested for this CUDA toolkit build");
    return PLAN7_BIAS_CUDA_UNATTESTED;
  }
  if (identity->runtime_version != 12050) {
    set_error(reason, reason_size,
              "bias math is not attested for this CUDA runtime version");
    return PLAN7_BIAS_CUDA_UNATTESTED;
  }
  if (identity->driver_version < identity->runtime_version) {
    set_error(reason, reason_size,
              "bias math requires a driver supporting CUDA runtime 12.5");
    return PLAN7_BIAS_CUDA_UNATTESTED;
  }
  if (identity->device_ordinal < 0 || identity->multiprocessor_count <= 0 ||
      identity->total_global_memory == 0 || uuid_is_zero(identity->uuid) ||
      identity->pci_domain_id < 0 || identity->pci_bus_id < 0 ||
      identity->pci_bus_id > 255 || identity->pci_device_id < 0 ||
      identity->pci_device_id > 31 || identity->pci_bus_address[0] == '\0' ||
      memchr(identity->pci_bus_address, '\0',
             sizeof(identity->pci_bus_address)) == nullptr ||
      memchr(identity->name, '\0', sizeof(identity->name)) == nullptr) {
    set_error(reason, reason_size,
              "bias CUDA device identity or provenance is incomplete");
    return PLAN7_BIAS_CUDA_UNATTESTED;
  }
  if (identity->compute_major == 7 && identity->compute_minor == 5 &&
      strcmp(identity->name, "NVIDIA GeForce RTX 2080 Ti") == 0 &&
      identity->multiprocessor_count == 68 &&
      identity->total_global_memory >= UINT64_C(10) * 1024 * 1024 * 1024) {
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
    return PLAN7_BIAS_CUDA_SM75_RTX2080_TI;
  }
  const bool named_h200 = strcmp(identity->name, "NVIDIA H200") == 0 ||
                          strcmp(identity->name, "NVIDIA H200 NVL") == 0;
  if (identity->compute_major == 9 && identity->compute_minor == 0 &&
      named_h200 && identity->multiprocessor_count >= 100 &&
      identity->total_global_memory >=
        UINT64_C(100) * 1024 * 1024 * 1024) {
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
    return PLAN7_BIAS_CUDA_SM90_H200;
  }
  set_error(reason, reason_size,
            "bias math is not attested for this CUDA product/architecture");
  return PLAN7_BIAS_CUDA_UNATTESTED;
}

namespace {

#if defined(__x86_64__) && defined(__GLIBC__)
/* Vendor and instruction-set requirements of the attested host reference.
 * They are independent of the family/model/stepping signature that the
 * built-in allow-list additionally pins. */
bool
host_features_supported(const plan7_bias_host_identity *identity,
                        char *reason,
                        size_t reason_size)
{
  const uint32_t required_leaf1_ecx =
    bit_FMA | bit_SSE4_1 | bit_OSXSAVE | bit_AVX;
  const uint32_t required_leaf7_ebx =
    bit_AVX2 | bit_AVX512F | bit_AVX512DQ | bit_AVX512CD |
    bit_AVX512BW | bit_AVX512VL;
  if (identity->vendor_ebx != UINT32_C(0x756e6547) ||
      identity->vendor_edx != UINT32_C(0x49656e69) ||
      identity->vendor_ecx != UINT32_C(0x6c65746e) ||
      (identity->leaf1_edx & bit_SSE2) == 0 ||
      (identity->leaf1_ecx & required_leaf1_ecx) != required_leaf1_ecx ||
      (identity->leaf7_ebx & required_leaf7_ebx) != required_leaf7_ebx ||
      (identity->xcr0_low & UINT32_C(0x000000e6)) !=
        UINT32_C(0x000000e6) ||
      identity->xcr0_high != 0) {
    set_error(reason, reason_size,
              "bias math is not attested for this CPU vendor or feature set");
    return false;
  }
  return true;
}
#endif

}  // namespace

extern "C" int
plan7_bias_host_identity_attested(const plan7_bias_host_identity *identity,
                                  int cuda_target,
                                  char *reason,
                                  size_t reason_size)
{
#if !defined(__x86_64__) || !defined(__GLIBC__)
  (void) identity;
  (void) cuda_target;
  set_error(reason, reason_size,
            "bias math is not attested for this host architecture");
  return 0;
#else
  if (identity == nullptr) {
    set_error(reason, reason_size, "bias host identity is null");
    return 0;
  }
  if (!host_features_supported(identity, reason, reason_size)) return 0;
  if (cuda_target == PLAN7_BIAS_CUDA_SM75_RTX2080_TI &&
      identity->family == 6 && identity->model == 85 &&
      identity->stepping == 4) {
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
    return 1;
  }
  if (cuda_target == PLAN7_BIAS_CUDA_SM90_H200 &&
      identity->family == 6 && identity->model == 143 &&
      identity->stepping == 8) {
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
    return 1;
  }
  set_error(reason, reason_size,
            "bias math is not attested for this CPU/accelerator pairing");
  return 0;
#endif
}

extern "C" int
plan7_bias_current_libm_build_id(
  uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE])
{
  if (build_id == nullptr) return -1;
  memset(build_id, 0, PLAN7_BIAS_LIBM_BUILD_ID_SIZE);
  return loaded_libm_build_id(build_id) ? 0 : -1;
}

extern "C" int
plan7_bias_libm_build_id_attested(const uint8_t *build_id,
                                  size_t build_id_size,
                                  int cuda_target)
{
  constexpr uint8_t kSm75LibmBuildId[PLAN7_BIAS_LIBM_BUILD_ID_SIZE] = {
    0xe6, 0xf0, 0x50, 0x69, 0x61, 0x20, 0xae, 0xb5, 0x13, 0x4a,
    0x14, 0xe3, 0x8b, 0xc5, 0x4e, 0x8c, 0xe1, 0xbd, 0xc0, 0xb5
  };
  constexpr uint8_t kSm90H200LibmBuildId[PLAN7_BIAS_LIBM_BUILD_ID_SIZE] = {
    0xb2, 0xb4, 0xcd, 0x0f, 0x73, 0xa7, 0xc7, 0xa6, 0xda, 0x5b,
    0xa4, 0x12, 0x6b, 0xce, 0xb3, 0x3c, 0xe6, 0x88, 0x53, 0xd5
  };
  if (build_id == nullptr ||
      build_id_size != PLAN7_BIAS_LIBM_BUILD_ID_SIZE)
    return 0;
  if (cuda_target == PLAN7_BIAS_CUDA_SM75_RTX2080_TI)
    return memcmp(build_id, kSm75LibmBuildId, sizeof(kSm75LibmBuildId)) == 0;
  if (cuda_target == PLAN7_BIAS_CUDA_SM90_H200)
    return memcmp(build_id, kSm90H200LibmBuildId,
                  sizeof(kSm90H200LibmBuildId)) == 0;
  return 0;
}

namespace {

#if defined(__x86_64__) && defined(__GLIBC__)
bool
host_identity_equal(const plan7_bias_host_identity &left,
                    const plan7_bias_host_identity &right)
{
  return left.vendor_ebx == right.vendor_ebx &&
         left.vendor_edx == right.vendor_edx &&
         left.vendor_ecx == right.vendor_ecx &&
         left.family == right.family && left.model == right.model &&
         left.stepping == right.stepping &&
         left.leaf1_ecx == right.leaf1_ecx &&
         left.leaf1_edx == right.leaf1_edx &&
         left.leaf7_ebx == right.leaf7_ebx &&
         left.xcr0_low == right.xcr0_low &&
         left.xcr0_high == right.xcr0_high;
}

/* One process-wide registration: the libm build, CPU identity and device
 * target for which an exhaustive audit (live or cached) has zero binary32
 * mismatches. Classification re-checks every field on each call. */
struct registered_attestation {
  bool valid;
  int target;
  uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE];
  plan7_bias_host_identity host;
};

std::mutex registered_attestation_mutex;
registered_attestation registered_attestation_state = {};

void
register_attestation(int target,
                     const uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE],
                     const plan7_bias_host_identity &host)
{
  std::lock_guard<std::mutex> lock(registered_attestation_mutex);
  registered_attestation_state.valid = true;
  registered_attestation_state.target = target;
  memcpy(registered_attestation_state.build_id, build_id,
         PLAN7_BIAS_LIBM_BUILD_ID_SIZE);
  registered_attestation_state.host = host;
}

bool
registered_attestation_matches(
  int target,
  const uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE],
  const plan7_bias_host_identity &host)
{
  std::lock_guard<std::mutex> lock(registered_attestation_mutex);
  return registered_attestation_state.valid &&
         registered_attestation_state.target == target &&
         memcmp(registered_attestation_state.build_id, build_id,
                PLAN7_BIAS_LIBM_BUILD_ID_SIZE) == 0 &&
         host_identity_equal(registered_attestation_state.host, host);
}

/* The live inputs shared by classification, audit and registration. */
struct live_environment {
  plan7_bias_cuda_identity cuda;
  plan7_bias_host_identity host;
  uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE];
  int target;
};

/* Returns 0 when every platform requirement holds, otherwise the failing
 * plan7_bias_environment_code. Nothing here depends on the libm build. */
int
capture_live_environment(live_environment *live,
                         char *reason,
                         size_t reason_size)
{
  memset(live, 0, sizeof(*live));
  if (plan7_bias_host_environment_attested() != 1) {
    set_error(reason, reason_size,
              "bias host floating-point environment is not attested "
              "(it requires round-to-nearest with FTZ and DAZ disabled)");
    return PLAN7_BIAS_ENV_FLOAT_ENVIRONMENT;
  }
  if (plan7_bias_current_cuda_identity(&live->cuda, reason, reason_size) != 0)
    return PLAN7_BIAS_ENV_CUDA_DEVICE;
  live->target = plan7_bias_cuda_identity_target(&live->cuda, reason,
                                                 reason_size);
  if (live->target == PLAN7_BIAS_CUDA_UNATTESTED)
    return PLAN7_BIAS_ENV_CUDA_DEVICE;
  if (plan7_bias_current_host_identity(&live->host, reason, reason_size) != 0)
    return PLAN7_BIAS_ENV_HOST_FEATURES;
  if (!host_features_supported(&live->host, reason, reason_size))
    return PLAN7_BIAS_ENV_HOST_FEATURES;
  if (!loaded_libm_build_id(live->build_id)) {
    set_error(reason, reason_size,
              "the loaded libm GNU build ID is unavailable");
    return PLAN7_BIAS_ENV_LIBM_UNAVAILABLE;
  }
  return 0;
}

void
build_id_hex(const uint8_t build_id[PLAN7_BIAS_LIBM_BUILD_ID_SIZE],
             char text[2 * PLAN7_BIAS_LIBM_BUILD_ID_SIZE + 1])
{
  static const char digits[] = "0123456789abcdef";
  for (size_t index = 0; index < PLAN7_BIAS_LIBM_BUILD_ID_SIZE; ++index) {
    text[2 * index] = digits[build_id[index] >> 4];
    text[2 * index + 1] = digits[build_id[index] & 0x0f];
  }
  text[2 * PLAN7_BIAS_LIBM_BUILD_ID_SIZE] = '\0';
}

using host_log_function = double (*)(double);

uint32_t
float_bits(float value)
{
  uint32_t raw;
  memcpy(&raw, &value, sizeof(raw));
  return raw;
}

uint64_t
double_bits(double value)
{
  uint64_t raw;
  memcpy(&raw, &value, sizeof(raw));
  return raw;
}

struct audit_counts {
  uint64_t float_mismatches;
  uint64_t double_mismatches;
  uint32_t maximum_float_ulp;
  uint64_t maximum_double_ulp;
  uint32_t first_float_input;
  uint32_t first_float_host;
  uint32_t first_float_device;
  uint32_t first_double_input;
};

/* Host side of the audit for one slice of a chunk. The host value is the
 * libm log() that HMMER/Easel call, reached through the same symbol whose
 * module supplied the build ID; the binary32 value is its round-to-nearest
 * conversion, exactly as the host reference computes (float)log((double)x). */
void
audit_slice(host_log_function host_log,
            uint32_t first,
            const float *device_binary32,
            const double *device_binary64,
            uint32_t begin,
            uint32_t end,
            audit_counts *counts)
{
  for (uint32_t index = begin; index < end; ++index) {
    const uint32_t raw = first + index;
    float value;
    memcpy(&value, &raw, sizeof(value));
    const double host = host_log(static_cast<double>(value));
    const uint64_t host64 = double_bits(host);
    const uint64_t device64 = double_bits(device_binary64[index]);
    if (host64 != device64) {
      if (counts->double_mismatches == 0) counts->first_double_input = raw;
      ++counts->double_mismatches;
      const uint64_t ulp = host64 > device64 ? host64 - device64
                                             : device64 - host64;
      counts->maximum_double_ulp = std::max(counts->maximum_double_ulp, ulp);
    }
    const uint32_t host32 = float_bits(static_cast<float>(host));
    const uint32_t device32 = float_bits(device_binary32[index]);
    if (host32 != device32) {
      if (counts->float_mismatches == 0) {
        counts->first_float_input = raw;
        counts->first_float_host = host32;
        counts->first_float_device = device32;
      }
      ++counts->float_mismatches;
      const uint32_t ulp = host32 > device32 ? host32 - device32
                                             : device32 - host32;
      counts->maximum_float_ulp = std::max(counts->maximum_float_ulp, ulp);
    }
  }
}
#endif

}  // namespace

extern "C" int
plan7_bias_classify_environment(char *reason, size_t reason_size)
{
#if !defined(__x86_64__) || !defined(__GLIBC__)
  set_error(reason, reason_size,
            "bias math is not attested for this host architecture");
  return PLAN7_BIAS_ENV_UNSUPPORTED_PLATFORM;
#else
  live_environment live;
  const int prerequisite = capture_live_environment(&live, reason,
                                                    reason_size);
  if (prerequisite != 0) return prerequisite;
  /* Built-in allow-list: the environments attested offline when each target
   * was admitted (unchanged). */
  if (strcmp(gnu_get_libc_version(), "2.35") == 0 &&
      plan7_bias_libm_build_id_attested(live.build_id, sizeof(live.build_id),
                                        live.target) == 1 &&
      plan7_bias_host_identity_attested(&live.host, live.target, nullptr,
                                        0) == 1) {
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
    return PLAN7_BIAS_ENV_ATTESTED_BUILTIN;
  }
  if (registered_attestation_matches(live.target, live.build_id, live.host)) {
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
    return PLAN7_BIAS_ENV_ATTESTED_SELF;
  }
  char hex[2 * PLAN7_BIAS_LIBM_BUILD_ID_SIZE + 1];
  build_id_hex(live.build_id, hex);
  if (reason != nullptr && reason_size != 0)
    snprintf(reason, reason_size,
             "bias math is not attested for this host math library "
             "(glibc %s, libm GNU build ID %s) on this CPU and device: no "
             "built-in attestation and no exhaustive self-attestation "
             "registered in this process",
             gnu_get_libc_version(), hex);
  return PLAN7_BIAS_ENV_NOT_ATTESTED;
#endif
}

extern "C" int
plan7_bias_environment_attested(char *reason, size_t reason_size)
{
  return plan7_bias_classify_environment(reason, reason_size) > 0 ? 1 : 0;
}

extern "C" void
plan7_bias_clear_attestation(void)
{
#if defined(__x86_64__) && defined(__GLIBC__)
  std::lock_guard<std::mutex> lock(registered_attestation_mutex);
  registered_attestation_state = {};
#endif
}

extern "C" int
plan7_bias_register_attestation(const uint8_t *build_id,
                                size_t build_id_size,
                                const plan7_bias_host_identity *host,
                                int cuda_target,
                                int algorithm_version,
                                uint64_t inputs_checked,
                                uint64_t float_mismatches,
                                char *reason,
                                size_t reason_size)
{
#if !defined(__x86_64__) || !defined(__GLIBC__)
  (void) build_id; (void) build_id_size; (void) host; (void) cuda_target;
  (void) algorithm_version; (void) inputs_checked; (void) float_mismatches;
  set_error(reason, reason_size,
            "bias math is not attested for this host architecture");
  return 0;
#else
  if (build_id == nullptr || build_id_size != PLAN7_BIAS_LIBM_BUILD_ID_SIZE ||
      host == nullptr) {
    set_error(reason, reason_size, "invalid bias attestation record");
    return 0;
  }
  if (algorithm_version != PLAN7_BIAS_ATTESTATION_ALGORITHM_VERSION) {
    set_error(reason, reason_size,
              "bias attestation record uses a different audit algorithm");
    return 0;
  }
  if (inputs_checked != PLAN7_BIAS_ATTESTATION_INPUT_COUNT ||
      float_mismatches != 0) {
    set_error(reason, reason_size,
              "bias attestation record is not a complete zero-mismatch audit");
    return 0;
  }
  live_environment live;
  if (capture_live_environment(&live, reason, reason_size) != 0) return 0;
  if (live.target != cuda_target) {
    set_error(reason, reason_size,
              "bias attestation record is for a different CUDA target");
    return 0;
  }
  if (memcmp(live.build_id, build_id, PLAN7_BIAS_LIBM_BUILD_ID_SIZE) != 0) {
    set_error(reason, reason_size,
              "bias attestation record is for a different libm build");
    return 0;
  }
  if (!host_identity_equal(live.host, *host)) {
    set_error(reason, reason_size,
              "bias attestation record is for a different CPU identity");
    return 0;
  }
  register_attestation(live.target, live.build_id, live.host);
  if (reason != nullptr && reason_size != 0) reason[0] = '\0';
  return 1;
#endif
}

extern "C" int
plan7_bias_self_attest(int host_threads,
                       plan7_bias_self_attestation *report,
                       char *reason,
                       size_t reason_size)
{
  if (report == nullptr) {
    set_error(reason, reason_size, "bias self-attestation report is null");
    return -1;
  }
  memset(report, 0, sizeof(*report));
  report->algorithm_version = PLAN7_BIAS_ATTESTATION_ALGORITHM_VERSION;
#if !defined(__x86_64__) || !defined(__GLIBC__)
  (void) host_threads;
  set_error(reason, reason_size,
            "bias math is not attested for this host architecture");
  return -1;
#else
  if (host_threads < 1 || host_threads > 1024) {
    set_error(reason, reason_size,
              "bias self-attestation needs 1 to 1024 host threads");
    return -1;
  }
  live_environment live;
  if (capture_live_environment(&live, reason, reason_size) != 0) return -1;
  host_log_function host_log = reinterpret_cast<host_log_function>(
    dlsym(RTLD_DEFAULT, "log"));
  if (host_log == nullptr) {
    set_error(reason, reason_size, "host libm log() is unavailable");
    return -1;
  }
  report->cuda_target = live.target;
  report->host_threads = host_threads;
  memcpy(report->libm_build_id, live.build_id, sizeof(report->libm_build_id));
  report->host = live.host;
  report->cuda = live.cuda;

  constexpr uint32_t kFirst = UINT32_C(1);
  constexpr uint32_t kLast = UINT32_C(0x7f7fffff);
  constexpr uint32_t kChunk = UINT32_C(1) << 24;
  float *device_binary32 = nullptr;
  double *device_binary64 = nullptr;
  float *host_binary32 = nullptr;
  double *host_binary64 = nullptr;
  bool pinned = true;
  int status_code = 0;
  cudaError_t status = cudaMalloc(&device_binary32, kChunk * sizeof(float));
  if (status == cudaSuccess)
    status = cudaMalloc(&device_binary64, kChunk * sizeof(double));
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "bias self-attestation cudaMalloc",
                   status);
    status_code = -1;
  }
  if (status_code == 0 &&
      (cudaMallocHost(&host_binary32, kChunk * sizeof(float)) != cudaSuccess ||
       cudaMallocHost(&host_binary64, kChunk * sizeof(double)) !=
         cudaSuccess)) {
    cudaGetLastError();
    if (host_binary32 != nullptr) cudaFreeHost(host_binary32);
    if (host_binary64 != nullptr) cudaFreeHost(host_binary64);
    pinned = false;
    host_binary32 = new (std::nothrow) float[kChunk];
    host_binary64 = new (std::nothrow) double[kChunk];
    if (host_binary32 == nullptr || host_binary64 == nullptr) {
      set_error(reason, reason_size,
                "bias self-attestation host buffers are unavailable");
      status_code = -1;
    }
  }

  audit_counts total = {};
  total.first_float_input = 0;
  const auto started = std::chrono::steady_clock::now();
  std::vector<audit_counts> slices(static_cast<size_t>(host_threads));
  for (uint64_t first64 = kFirst; status_code == 0 && first64 <= kLast;
       first64 += kChunk) {
    const uint32_t first = static_cast<uint32_t>(first64);
    const uint32_t count = static_cast<uint32_t>(
      std::min<uint64_t>(kChunk, static_cast<uint64_t>(kLast) - first64 + 1));
    bias_log_attestation_kernel<<<(count + kThreads - 1) / kThreads,
                                  kThreads>>>(first, count, device_binary32,
                                              device_binary64);
    status = cudaGetLastError();
    if (status == cudaSuccess)
      status = cudaMemcpy(host_binary32, device_binary32,
                          static_cast<size_t>(count) * sizeof(float),
                          cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
      status = cudaMemcpy(host_binary64, device_binary64,
                          static_cast<size_t>(count) * sizeof(double),
                          cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      set_cuda_error(reason, reason_size, "bias self-attestation chunk",
                     status);
      status_code = -1;
      break;
    }
    std::fill(slices.begin(), slices.end(), audit_counts{});
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(host_threads));
    for (int worker = 0; worker < host_threads; ++worker) {
      const uint32_t begin = static_cast<uint32_t>(
        static_cast<uint64_t>(count) * worker / host_threads);
      const uint32_t end = static_cast<uint32_t>(
        static_cast<uint64_t>(count) * (worker + 1) / host_threads);
      workers.emplace_back(audit_slice, host_log, first, host_binary32,
                           host_binary64, begin, end, &slices[worker]);
    }
    for (std::thread &thread : workers) thread.join();
    /* Slices are ordered, so the first nonzero slice holds the first
     * mismatch of the chunk; chunks are visited in increasing order. */
    for (const audit_counts &slice : slices) {
      if (slice.float_mismatches != 0 && total.float_mismatches == 0) {
        total.first_float_input = slice.first_float_input;
        total.first_float_host = slice.first_float_host;
        total.first_float_device = slice.first_float_device;
      }
      if (slice.double_mismatches != 0 && total.double_mismatches == 0)
        total.first_double_input = slice.first_double_input;
      total.float_mismatches += slice.float_mismatches;
      total.double_mismatches += slice.double_mismatches;
      total.maximum_float_ulp = std::max(total.maximum_float_ulp,
                                         slice.maximum_float_ulp);
      total.maximum_double_ulp = std::max(total.maximum_double_ulp,
                                          slice.maximum_double_ulp);
    }
    report->inputs_checked += count;
  }
  report->seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();

  if (device_binary32 != nullptr) cudaFree(device_binary32);
  if (device_binary64 != nullptr) cudaFree(device_binary64);
  if (pinned) {
    if (host_binary32 != nullptr) cudaFreeHost(host_binary32);
    if (host_binary64 != nullptr) cudaFreeHost(host_binary64);
  } else {
    delete[] host_binary32;
    delete[] host_binary64;
  }
  if (status_code != 0) return -1;

  report->float_mismatches = total.float_mismatches;
  report->double_mismatches = total.double_mismatches;
  report->maximum_float_ulp = total.maximum_float_ulp;
  report->maximum_double_ulp = total.maximum_double_ulp;
  report->first_float_mismatch_input = total.first_float_input;
  report->first_float_mismatch_host = total.first_float_host;
  report->first_float_mismatch_device = total.first_float_device;
  report->first_double_mismatch_input = total.first_double_input;
  report->attested =
    report->inputs_checked == PLAN7_BIAS_ATTESTATION_INPUT_COUNT &&
    total.float_mismatches == 0;
  if (report->attested) {
    register_attestation(live.target, live.build_id, live.host);
    if (reason != nullptr && reason_size != 0) reason[0] = '\0';
  } else {
    set_error(reason, reason_size,
              report->inputs_checked == PLAN7_BIAS_ATTESTATION_INPUT_COUNT
                ? "device and host log differ on at least one input"
                : "bias self-attestation did not cover every input");
  }
  return 0;
#endif
}

extern "C" int
plan7_bias_filter_candidates_device(
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
  size_t error_size)
{
  if (candidate_count == 0) return 0;
  if (device_residues == nullptr || device_offsets == nullptr ||
      device_length_logp == nullptr || device_length_log1mp == nullptr ||
      device_profiles == nullptr || device_candidates == nullptr ||
      device_ssv_inputs == nullptr || device_results == nullptr) {
    set_error(error, error_size, "bias CUDA device buffer is null");
    return -1;
  }
  if (plan7_bias_environment_attested(error, error_size) != 1) return -1;
  if (candidate_count >
      static_cast<size_t>(UINT_MAX) * static_cast<size_t>(kThreads)) {
    set_error(error, error_size, "bias CUDA candidate grid is too large");
    return -1;
  }

  const unsigned blocks = static_cast<unsigned>(
    (candidate_count + kThreads - 1) / kThreads);
  bias_candidates_kernel<<<blocks, kThreads>>>(
    device_residues, device_offsets, device_length_logp,
    device_length_log1mp, device_profiles, device_candidates,
    device_ssv_inputs, candidate_count, device_results);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "bias candidate kernel", status);
    return -1;
  }
  return 0;
}
