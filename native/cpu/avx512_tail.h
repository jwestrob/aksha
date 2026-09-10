#ifndef PLAN7_GPU_AVX512_TAIL_H
#define PLAN7_GPU_AVX512_TAIL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char ESL_DSQ;
typedef struct p7_oprofile_s P7_OPROFILE;

enum { PLAN7_AVX512_TAIL_LANES = 4 };

/* Return nonzero only when the process and OS can execute every instruction
 * used by the exact four-candidate Backward parser. */
int plan7_avx512_tail_available(void);

/* Run four independent parsing-mode Forward recurrences. Target lengths may
 * differ; each 128-bit quarter uses the exact N/J/C length transitions that
 * p7_oprofile_ReconfigLength() would install for that candidate. */
int plan7_avx512_forward4_varlen(
    const ESL_DSQ *const sequences[PLAN7_AVX512_TAIL_LANES],
    const int lengths[PLAN7_AVX512_TAIL_LANES],
    const P7_OPROFILE *profile,
    const float *forward_xmx[PLAN7_AVX512_TAIL_LANES],
    uint64_t forward_xmx_counts[PLAN7_AVX512_TAIL_LANES],
    float forward_scores[PLAN7_AVX512_TAIL_LANES],
    float forward_totscales[PLAN7_AVX512_TAIL_LANES],
    int forward_statuses[PLAN7_AVX512_TAIL_LANES],
    uint64_t *elapsed_ns);

/* Run four independent HMMER parsing-mode Backward recurrences in the four
 * 128-bit quarters of one ZMM register. Inputs may have unequal lengths. The
 * supplied Forward arrays are the exact E/N/J/B/C/SCALE trajectories already
 * authenticated by the continuation journal.
 *
 * Output pointers refer to thread-local immutable storage and remain valid
 * until the next call on the same thread. The caller must consume all four
 * rows before invoking this function again. */
int plan7_avx512_backward4_varlen(
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
    uint64_t *elapsed_ns);

/* Discard only complete resident pages from the two thread-local result
 * backings. The result objects, sizes, capacities, and data pointers remain
 * unchanged; callers must have consumed every pointer returned by the most
 * recent four-lane calls before invoking this function. */
int plan7_avx512_tail_madvise_pages(uint64_t *released_bytes);

/* Process-wide observability for the private, caller-controlled release. */
void plan7_avx512_tail_madvise_statistics(
    uint64_t *call_count,
    uint64_t *released_bytes);
void plan7_avx512_tail_madvise_statistics_reset(void);

#ifdef __cplusplus
}
#endif

#endif
