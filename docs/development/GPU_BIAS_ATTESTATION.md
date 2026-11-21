# GPU bias-math attestation that survives OS updates

Date: 2026-11-21. Branch `fix/gpu-environment-attestation`.

## What failed

The CUDA composition-bias filter computes `(float)log((double)x)` on the device;
the CPU pipeline (Easel `esl_hmm_Forward`) computes it with the host libm. The
two agree only if the device `log` and the host libm `log` agree, so
`plan7_bias_environment_attested` admitted only environments audited offline:
glibc 2.35, one pinned libm GNU build ID per GPU target and one pinned CPU
signature. The audit itself (hmmer_gpu `oracle/bias_log_attestation.cu`,
`results/cuda/bias-log-attestation-sm75.json`) compared the device and host
values for every positive finite binary32 input (2,139,095,039 values) and
required zero mismatches.

On 2026-11-17 biotite's H200 node moved to Ubuntu 24.04.5 (glibc 2.39, libm
build ID `d53fcbff...`, kernel 6.8.0-146, driver 580.178.04). The allow-list no
longer matched, every bias-dependent stage reported "not attested", and the
sealed F2 compaction then raised `post-filter workspace is unavailable`: the
post-filter returned early (all candidates `CPU_REQUIRED`) before creating the
workspace that the compaction requires. Every GPU search aborted.

## Design chosen: self-attestation at first use

The CUDA extension now runs the same exhaustive audit itself
(`plan7_bias_self_attest`): the production device function
(`device_log_to_float`) versus the host libm `log`, reached through the same
dynamic symbol whose module supplies the build ID, for every positive finite
binary32 input, zero binary32 mismatches required. On success the process
registers the live libm build ID, the complete CPU identity and the device
target; classification re-checks all three on every call.

Aksha (`aksha/gpu_attestation.py`) runs the audit once per environment and caches
the record under a key made of everything the result depends on: the CUDA
extension's SHA-256, the audit algorithm, glibc, libm build ID, CPU identity
(vendor, family/model/stepping, feature words, XCR0), device product, compute
capability, SM count, CUDA runtime, driver API level and nvcc build. A later
process re-binds the record through `plan7_bias_register_attestation`, which
refuses anything that does not equal the live identity or is not a complete
zero-mismatch audit of the current algorithm. Any OS, library, driver, CPU,
device or extension change selects a new key and a new audit. The built-in
environments stay trusted without an audit.

Correctness policy: unchanged. The criterion is the one that admitted the
original targets (exhaustive, zero binary32 mismatches); only where and when it
runs changed. The device allow-list (H200, RTX 2080 Ti; CUDA 12.5.82 build,
runtime 12050) and the floating-point environment check are unchanged. The
pinned CPU signature now applies only to the built-in records; a self-attested
record is bound to its exact CPU identity instead.

Measured on the H200 node after the upgrade (glibc 2.39, 28 host threads): zero
binary32 mismatches over 2,139,095,039 inputs, 1.95 s.

## Graceful degradation

Untrusted (audit disabled, failed, or a non-admitted device/host): Aksha keeps
only the exact GPU SSV filter and runs the bias filter and every later stage on
the CPU pipeline, printing one warning line that names the untrusted component
and how to restore GPU use. Independently, the native post-filter now
invalidates resident results whenever it retains every candidate for the CPU
(or receives none), and the F2 compaction reports an unsupported view instead of
raising, so the session route also completes through the host records
(`ASTRA_GPU_UNTRUSTED_NATIVE_FALLBACK=1` exercises it).

## Readiness check

`aksha gpu-check --gpu-manifest DB=PATH ... [--attest] [--json]` reports, per
manifest and CUDA device, `GPU`, `GPU after self-attestation`, `SSV only` or
`refused`, with the reason; GPU searches run the same evaluation at start.

## Options rejected

- Add the new build ID to the allow-list after a manual audit: fixes today and
  keeps the single point of failure for the next glibc update.
- Make results independent of the host libm (vendor a fixed `log` into the
  private Easel/HMMER runtime or a correctly rounded implementation on both
  sides): changes the CPU reference itself, needs a runtime rebuild and a full
  requalification, and departs from "HMMER 3.4 with the host libm defines
  correctness".
- Audit at every process start with no cache: correct and cheap on an H200
  (about 2 s), but the readiness check could then never answer without a GPU
  run, and slower hosts pay it on every search. The cache makes the audit a
  one-time cost per environment while every key field still forces a re-audit.
- Accept sampled or partial audits: weaker than the original exhaustive
  criterion.

## Characterization (not gating)

The audit also reports binary64 agreement of `log((double)x)` over the same
inputs, which the Forward, Backward and domain-rescore kernels use before
rounding sums to binary32. On the upgraded H200 node: 215,332 inputs differ by
one binary64 ulp (CUDA `log` is not correctly rounded; glibc's is), zero at
binary32. Their exactness therefore rests on the subsequent binary32 rounding
and on full-output identity tests, as before this change; a binary64 gate would
disable those kernels everywhere.
