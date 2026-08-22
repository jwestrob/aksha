# GPU ready-queue experiment

This change isolates one scheduling question: whether retaining more completed
GPU candidate batches reduces continuation stalls. It does not change model
grouping, chunk order, CUDA generation, continuation, thresholds, or output.

## Configuration and memory contract

With neither variable set, Astra uses the existing `queue.Queue(maxsize=1)`
implementation. This is the regression control.

The experimental implementation is selected with:

- `ASTRA_GPU_READY_QUEUE_DEPTH`: exactly `1`, `2`, or `4`;
- `ASTRA_GPU_READY_QUEUE_BYTES`: a canonical positive decimal byte count no
  larger than 2^63 - 1.

Depths 2 and 4 require the byte variable. Supplying a byte value also selects
the weighted implementation at depth 1. Invalid configuration is rejected
before Astra creates its search output directory.

The byte ceiling applies only to batches stored in the ready FIFO. It excludes
the batch currently owned by continuation and the batch currently being built
by the producer. Consequently, at configured depth `d`:

- at most `d` completed batches and the configured number of their bytes are
  queued;
- generation can be at most `d + 1` chunks ahead of continuation;
- at most `d + 2` candidate batches can be live: one consumer, `d` queued, and
  one producer.

The queue charges the adapter's authoritative `CandidateBatch.resident_bytes`.
That value is the exact incremental storage retained only because the batch is
alive: every batch-owned host or native allocation once, with no double charge
for views. Shared target storage, shared profile/session storage, persistent
generation workspaces, and allocator estimates are excluded. `sys.getsizeof`
and inferred sizes are not accepted. Missing, non-integer, negative, changing,
or over-ceiling charges are errors.

An over-ceiling batch is rejected; it is never admitted as an oversize
singleton. Its error occupies its canonical ordinal, so all earlier published
chunks finish first and no later chunk is selected. On every exit Astra stops
the producer, drains queued references, waits for the producer-owned completion
event, retries thread join through interruptions, drains again, and requires
final queued count and bytes to both be zero. A current continuation/output
error retains precedence over later producer and cleanup errors.

## Metrics and replay evidence

`ASTRA_GPU_OVERLAP_TIMING=1` already opts a request into metrics. The queue
adds:

- configured item and byte capacities;
- item and byte high-water marks;
- final queued item and byte counts;
- total generated and consumed candidate bytes and maximum batch bytes;
- one ordered generation record per successfully dequeued chunk, including
  monotonic selection, generation, wait, and dequeue timestamps plus exact
  resident bytes when byte accounting is enabled;
- one ordered continuation record per attempted chunk, including monotonic
  start, finish, duration, and completion state.

The detailed records are absent when metrics are not requested. Snapshotting
copies record dictionaries so later metric mutation cannot rewrite a captured
snapshot.

The first H200 comparison should use the weighted implementation for all three
experimental arms with the same cap:

```text
ASTRA_GPU_READY_QUEUE_DEPTH=1 ASTRA_GPU_READY_QUEUE_BYTES=1073741824
ASTRA_GPU_READY_QUEUE_DEPTH=2 ASTRA_GPU_READY_QUEUE_BYTES=1073741824
ASTRA_GPU_READY_QUEUE_DEPTH=4 ASTRA_GPU_READY_QUEUE_BYTES=1073741824
```

The 1 GiB cap exceeds the largest observed full-PFAM batch and the largest
observed adjacent four-batch ready payload, while bounding the experimental
FIFO independently of depth. Retain a separate no-environment depth-one run to
prove the default path remains the audited implementation.

Before any GPU launch, replay the recorded service intervals as a deterministic
single-producer/single-consumer discrete-event schedule:

1. Preserve the exact recorded chunk order; never sort profile ordinals.
2. Admit a generated batch only when both item and byte predicates hold.
3. Subtract its queued bytes at dequeue, before continuation begins.
4. Reproduce the measured depth-one wait, idle, high-water, and wall accounting
   from the trace.
5. Replay depths 2 and 4 with the common cap. With fixed service intervals,
   increasing depth must not increase simulated consumer stall, producer block,
   or completion time.
6. Use replay only to decide which real arms are worth running. Acceptance of a
   real run requires identical biological output and provenance, exact queue
   invariants, zero final queue state, and observed scheduling movement in the
   predicted direction; no arbitrary speed threshold is introduced here.

## Persistent continuation executor design

Executor and `pyhmmer.plan7.Pipeline` reuse is intentionally a separate change.
Today `plan7_gpu.astra_search` creates a `ThreadPoolExecutor` and thread-local
Pipeline set for every outer candidate batch. A future request-scoped
`ContinuationExecutor` should:

- own one fixed worker pool from request entry through request exit;
- permit exactly one active candidate iterator at a time;
- keep each Pipeline exclusively on its worker thread;
- key Pipeline reuse on the complete normalized immutable Pipeline
  configuration and alphabet, replacing rather than mutating on a key change;
- clear after every row and poison/replace a Pipeline after construction,
  search, or clear failure;
- cancel and join every per-batch future before releasing that CandidateBatch;
- reject concurrent iterators and shut down all workers before the target batch
  or request resources close.

Deliver it in stages: persistent executor with fresh per-call Pipelines first,
then Astra request ownership, then same-key Pipeline reuse. Re-capture service
traces after each stage because changing continuation setup changes the queue
replay inputs. It must not be stored in the profile-session cache: targets,
candidates, thresholds, continuation state, and Pipelines remain request-owned.

