# Persistent PFAM ProfileSession design

## Objective and measured opportunity

The bounded ready queue already keeps generation and continuation busy.  Its
remaining cold-start cost on the RTX 2080 Ti is dominated by immutable PFAM
state construction:

- `load_pressed_profiles`: about 12.4 seconds per process;
- `ProfileSession` host snapshot: about 2.9 seconds per process;
- retained immutable host snapshot: 1,148,474,819 bytes.

The smallest safe next step is to retain that state in a deliberately
long-lived process and reuse it across sequential searches.  Target sequences,
CUDA `SequenceBatch` objects, candidate batches, thresholds, output files, and
continuation state remain owned by one search and are always destroyed at its
normal boundary.  The CUDA runtime context naturally remains alive for the
life of the worker process; this design does not retain target-dependent device
memory.

## Ownership model

`GPUProfileSessionCache` is explicit and process-local.  It is not a module
singleton and the normal one-shot CLI does not create one.  A long-lived caller
owns it and passes it to successive calls:

```python
from astra import search
from astra.gpu_profile_cache import GPUProfileSessionCache

with GPUProfileSessionCache() as cache:
    search.main(first_args, gpu_profile_session_cache=cache)
    search.main(second_args, gpu_profile_session_cache=cache)
```

Each search receives an exclusive lease whose interface is the subset of
`ProfileSession` used by Astra.  Closing the lease returns it to the cache;
closing the cache destroys the native session.  If cache shutdown begins while
a lease is active, destruction is deferred until that lease returns.  A second
concurrent lease fails rather than silently doubling PFAM memory or changing
queue ordering.  Astra first takes a lightweight reservation before constructing
the per-search CUDA target batch, so a concurrent request fails while still
host-only instead of transiently doubling target memory.

The cache has exactly one entry.  Replacing an idle key closes the old session
before allocating the new one, so the cache never retains two 1.15 GB profile
snapshots.  The first seam intentionally supports one mapped database per
search; ordinary multi-database searches retain their previous uncached path.

## Exact key and invalidation

The key contains:

1. the canonical pressed-database base path;
2. the SHA256 of the exact trusted manifest;
3. the pinned stat token of all `.h3m/.h3i/.h3f/.h3p` members;
4. the PyHMMER version and private-ABI SHA256;
5. SHA256 identities of the loaded plan7 adapter, native extension, and
   continuation extension;
6. an opaque device identity (currently the CUDA ordinal supplied by the
   target batch, conservatively segregating sessions even though today's host
   snapshot is device-neutral);
7. construction/selection worker counts; and
8. a versioned canonical amino/local/multihit/exact-Forward semantics token.

Every acquisition revalidates the manifest before a hit can be returned.  A
database stat change, manifest change, runtime/ABI change, device change, or
relevant profile option change is a miss and closes the idle entry.  A closed
or malformed resident session is never returned.  On a cold miss,
`load_pressed_profiles(..., manifest=...)` performs its existing second pinned
validation while loading, preserving the validation/load race closure.

Search thresholds (`F1`, `F2`, `F3`, bias filtering, reporting and inclusion
cutoffs) are intentionally not cached: they are applied to fresh per-search
selections/candidate batches and cannot affect the immutable session snapshot.

## Production boundary still needed

This commit supplies the safe resource seam, but it does not invent a daemon
protocol.  Reuse requires a caller that deliberately keeps one Python process
alive.  Before exposing a general service, the narrow next production boundary
should be a batch/worker entry point with one request at a time, per-request
logging handlers, unique output directories, cancellation that always returns
the lease, and worker retirement on cache close failure.  No cache should be
serialized or shared across processes.

## Fair persistent CPU comparison

The performance experiment must not compare a warm GPU database with a cold CPU
database.  Use two long-lived worker processes under the same 16-core affinity:

- GPU worker: owns one `GPUProfileSessionCache`, rebuilds `SequenceBatch` for
  each distinct target set, and closes every target/candidate resource after
  the request;
- CPU worker: parses the exact same manifest-authenticated pressed PFAM HMMs
  once into an immutable tuple, then reuses that tuple across requests while
  constructing a fresh PyHMMER pipeline for each target set.

Keep both workers alive and dispatch distinct 10k-target jobs in a balanced
alternating order.  Record two results rather than hiding setup:

1. cold end-to-end latency and RSS, including the one database load/build;
2. warm per-request latency and total batch makespan after both arms report a
   cache hit.

The GPU gate should require `profile_cache_hit=true`, zero warm profile-load and
session-build time, the same native session identity across warm requests,
monotonic selection identities, bounded RSS, the existing 14-chunk/1+15 worker
topology, and exact canonical TSV/semantic hashes for every job.  The CPU gate
should similarly prove one PFAM parse and a fresh search pipeline per job.

Based only on the already measured decomposition, a warm GPU request can remove
roughly 15.3 seconds from the current approximately 91-second whole-process
path.  That is a hypothesis for the persistent benchmark, not a claimed result.
