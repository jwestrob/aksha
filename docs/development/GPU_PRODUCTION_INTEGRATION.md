# GPU production integration (promoted candidate)

## Source map

The Astra integration starts at `f8b61d8`, which is already a descendant of
CPU-production main `7e81fc4`. It therefore retains installed CPU profile
streaming, the native TSV sink, and the console allocator launcher while
adding the measured GPU cell-cap, global continuation-window, chunk-local
profile-pack, and pressed-profile-stream stack.

The paired plan7_gpu integration starts at exact full-gate tree `ce46998` and
adds the tested logical HMMER implementation plus request-local Forward,
Pipeline-release, AVX-release, intra-row-release, and filter-tail SIMD control
seams. That patch set reproduces private PyHMMER ABI SHA-256
`282988762a47af957ed04e05e97ca49387ec2eb7e4e485cabee1f9d05ee0dffb`.
The runtime-qualified commits are Astra `7a03f15` and plan7_gpu `964ca98`;
the latter is followed only by source-hygiene commit `dd40d9d`, which deletes
an obsolete benchmark submission script. The exact validation DSOs have
SHA-256 `61a9f2bc29723e03f0b96aa351cc1d4c1250a21a587d267e86924a7204f1ffe5`
(`_native`) and
`54cf18e2cbe85c1b2f0a790302d51a8842945c869fd99fbbaeb0e141ca68d477`
(`_pipeline`). The Astra wheel has SHA-256
`fd7b6d4c4e309a49008774a17b50b6adb52d635df93b1373b1f0a23cb5ae5022`.

## Automatic request policy

The measured bundle is selected only when all of these facts are available at
the normal authenticated GPU preflight boundary:

- an installed pressed database has passed its GPU manifest attestation;
- exactly one installed pressed database is GPU-mapped for the request;
- the effective HMMER options contain only finite global E=1e-15;
- the four pressed files total at least 4 GiB;
- the request has exactly 64 search threads and more than 65,536 targets;
- persistent profile caching, serial mode, legacy overlap, and every private
  path-affecting tuning overrides are absent. Metrics-only profiling
  controls do not alter eligibility.

That immutable request-local decision selects 100,000,000 profile-target
cells per chunk, streamed chunk-local profile sessions, a pooled global
continuation window of four, exact sharded continuation at ratio 3/2, a
1,300,000,000-work-unit Pipeline page-release threshold, AVX result-page
release, logical HMMER intra-row release at 16 MiB, sparse journal v3, and
request-local hybrid Forward ownership below or equal to 200,000 cells. The
Forward cutoff is bound while the shared target batch is constructed; the
page-release hooks save and
restore prior programmatic state. The policy is passed through Python objects
and scoped extension hooks; it never writes process environment variables.

Every failed predicate keeps the previous eager/profile-session scheduler and
default core behavior. An explicit private setting disables the automatic
bundle and remains authoritative, providing the rollback path. CPU requests,
custom HMM inputs, small databases, cascade/domain/inclusion/bit-score
requests, cached or multi-database sessions, non-1e-15 thresholds, and
non-64-thread requests therefore keep their measured paths.

The separate retained PFAM policy is selected only for one attested installed
GPU database with at least 256 profiles, no persistent profile cache, exactly
64 threads, more than 65,536 targets, gathering cutoffs, and no private path
override. It keeps eager profile loading and default page release, while
enabling sparse journal v3, filter-tail SIMD, the pooled global continuation
window of four, exact 3/2 sharding, and request-local hybrid Forward ownership
below or equal to 200,000 cells. The explicit filter-tail SIMD and test-fallback
environment controls both disable this automatic path.

## Allocator launcher

The allocation-free Linux/glibc console launcher applies
`MALLOC_ARENA_MAX=24` to every `astra search --threads>=64` request, including
GPU-manifest searches. This avoids a second FASTA parse or late allocator
mutation. A user-supplied `MALLOC_ARENA_MAX` remains authoritative;
`ASTRA_CPU_MALLOC_ARENA_MAX=0` remains the compatible opt-out despite its
legacy CPU-specific name. Direct/library APIs retain their prior behavior.

AVX, logical page-release, and filter-tail SIMD settings are process-global in
their underlying private runtimes. Astra-managed calls serialize, save, and
restore them. Private direct plan7_gpu/PyHMMER callers must not mutate or use
those private release hooks concurrently with an Astra-managed tuned request.

## Promotion evidence

The focused host gates passed for the automatic selectors, default-off and
explicit-override paths, multi-database rejection, launcher precedence, CPU
profile streaming/native sink, global-window failure ordering, request-local
state restoration, and exact known Forward/Viterbi rows. The exact ABI-282
SM75/SM90 build retained the qualified normalized SASS properties: the SM90
Forward kernel used 64 registers with no local memory, stack, or spills, and
the SM90 Viterbi kernel did the same.

Full PFAM production-auto job 1189279 completed the exact gathering-cutoff
request with output SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`.
It selected Forward 200k, filter-tail SIMD, sparse journal v3, window four,
3/2 sharding, and launcher arena 24 while leaving KOFAM-only release controls
off. Request wall time was 218.980 seconds, process peak RSS was 6,238,016 KiB,
cgroup peak was 6,351,958,016 bytes, and HBM peak was 1,598 MiB. This was
1.147% faster and used 5.066% less process RSS than retained exact job 1188661.
The Slurm step reported failure only because its completed-result summarizer
read a stale telemetry location; replaying the corrected summarizer over the
immutable result produced a PASS summary with SHA-256
`4de70fee61672021f2cf2f157a9a954199b6c6ffb7b5d07bff0438cee0602b7c`.

Full KOFAM production-auto job 1189282 completed the exact PLM2_5, E=1e-15
request with raw-order output SHA-256
`fdd134e107fcc688be6a749493082c96eec6ce71e1c8b05e9bef0b8da076abc7`.
The real automatic route selected Forward 200k, streamed 333-profile chunks,
sparse journal v3, window four, 3/2 sharding, Pipeline release at 1.3 billion,
AVX release, logical 16 MiB intra-row release, the 100-million-cell cap, and
launcher arena 24. Request wall time was 1,393.690 seconds, process peak RSS
was 9,086,124 KiB, cgroup peak was 9,618,771,968 bytes, and HBM peak was
2,976 MiB, all within the full-gate limits. Its Slurm step reported failure
only after the exact result was written because the evidence wrapper counted
its own final intra-row reset as a third swap; the production request itself
made the expected enable-and-restore pair.

These exact-output and resource gates promote the paired source trees as the
production integration candidate. Neither integration branch has been merged
to main or pushed.
