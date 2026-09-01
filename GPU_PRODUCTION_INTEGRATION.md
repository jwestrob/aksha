# GPU production integration (prepared, not promoted)

## Source map

The Astra integration starts at `f8b61d8`, which is already a descendant of
CPU-production main `7e81fc4`. It therefore retains installed CPU profile
streaming, the native TSV sink, and the console allocator launcher while
adding the measured GPU cell-cap, global continuation-window, chunk-local
profile-pack, and pressed-profile-stream stack.

The paired plan7_gpu integration starts at exact full-gate tree `ce46998` and
adds the tested logical HMMER implementation from `e55a2dd` plus `f7e2172`.
That patch set reproduces private PyHMMER ABI SHA-256
`282988762a47af957ed04e05e97ca49387ec2eb7e4e485cabee1f9d05ee0dffb`.
The paired prepared core commit is `bc06307`; it supplies request-local shard,
Pipeline-release, and AVX-release controls without changing their existing
environment-controlled defaults.

## Automatic request policy

The measured bundle is selected only when all of these facts are available at
the normal authenticated GPU preflight boundary:

- an installed pressed database has passed its GPU manifest attestation;
- exactly one installed pressed database is GPU-mapped for the request;
- the effective HMMER options contain only finite global E=1e-15;
- the four pressed files total at least 4 GiB;
- the request has exactly 64 search threads and more than 65,536 targets;
- persistent profile caching, serial mode, legacy overlap, and every private
  path-affecting private tuning overrides are absent. Metrics-only profiling
  controls do not alter eligibility.

That immutable request-local decision selects 100,000,000 profile-target
cells per chunk, streamed chunk-local profile sessions, a pooled global
continuation window of four, exact sharded continuation at ratio 3/2, a
1,300,000,000-work-unit Pipeline page-release threshold, AVX result-page
release, logical HMMER intra-row release at 16 MiB, and request-local hybrid
Forward ownership below or equal to 200,000 cells. The Forward cutoff is bound
while the shared target batch is constructed; the page-release hooks save and
restore prior programmatic state. The policy is passed through Python objects
and scoped extension hooks; it never writes process environment variables.

Every failed predicate keeps the previous eager/profile-session scheduler and
default core behavior. An explicit private setting disables the automatic
bundle and remains authoritative, providing the rollback path. CPU requests,
custom HMM inputs, PFAM/small databases, GA/cascade/domain/inclusion/bit-score
requests, cached or multi-database sessions, non-1e-15 thresholds, and
non-64-thread requests therefore keep their measured paths.

## Allocator launcher

The allocation-free Linux/glibc console launcher applies
`MALLOC_ARENA_MAX=24` to every `astra search --threads>=64` request, including
GPU-manifest searches. This avoids a second FASTA parse or late allocator
mutation. A user-supplied `MALLOC_ARENA_MAX` remains authoritative;
`ASTRA_CPU_MALLOC_ARENA_MAX=0` remains the compatible opt-out despite its
legacy CPU-specific name. Direct/library APIs retain their prior behavior.

No branch in this preparation has been merged to main or pushed. A coordinated
wheel/build and exact GPU plus PFAM/default regression gates are still required
before promotion.
