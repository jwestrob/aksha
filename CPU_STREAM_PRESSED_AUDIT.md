# Bounded pressed-profile CPU residency

## Evidence for the memory target

The ordinary Astra CPU path reads every pressed HMM into a Python list before
starting `pyhmmer.hmmsearch`.  The installed databases differ sharply in total
model length and pressed payload:

| database | models | summed M | `.h3m` bytes |
|---|---:|---:|---:|
| PFAM | 27,481 | 4,374,119 | 854,484,190 |
| KOFAM | 27,816 | 16,387,763 | 3,171,683,747 |

A bounded 1,000-profile read measured about 204--213 resident bytes per model
position after subtracting the PyHMMER process baseline.  Extrapolation puts
the eager HMM object list near 0.9 GiB for PFAM and 3.3 GiB for KOFAM before
worker pipelines, targets, or results.

MetaCerberus obtains lower KOFAM memory by a different execution shape: it
splits targets and opens them as streaming `SequenceFile` inputs in four-CPU
tasks.  Its lower peak is not explained solely by stricter output filtering.
The preserved KOFAM run emitted 4,561,845 raw HMM rows versus Astra's
2,994,532 data rows, though the database, domain threshold, columns, and final
semantics differ.  For PFAM, MetaCerberus emitted 180,291 pre-overlap HMM rows
under E=1e-15 plus domain score 60 versus Astra's 383,234 GA-threshold data
rows.  MetaCerberus's measured aggregate cgroup peak was 4.624 GiB, while the
historical Astra PFAM process peak was already lower at 2.501 GiB.  The acute
gap is therefore the long-profile KOFAM execution shape, not a universal
Astra object leak.

## Prototype

Commit `41d0965` adds an opt-in private `_PressedHMMStream` selected by
`ASTRA_CPU_STREAM_PRESSED=1`.  It applies only to ordinary bulk CPU search
with a fixed threshold policy.  The pressed database is reopened once and
read in existing 2,000-profile chunks.  A completed chunk and all of its HMM
objects are released before the next chunk is read.  GPU, cascade, and
MacSyFinder paths retain their established behavior.  Public APIs are
unchanged.

The retained implementation preflights fixed GA/NC/TC availability with a
bounded one-profile-at-a-time pass.  If a pressed database mixes profiles
with and without the requested cutoff, it returns to the ordinary eager path
before executing any query; this preserves the eager path's global cutoff
grouping, profile/output order, and error order.  The profile-chunk generator
and each result iterator are also closed explicitly on completion,
cancellation, or failure, so the pressed file descriptor cannot outlive the
search scope.  User-supplied, unpressed, cascade, MacSyFinder, GPU, and default
non-opt-in paths remain untouched.

The unit oracle presses five real toy HMMs, forces 2-profile chunks, and
compares eager and streamed TSV bytes.  Both tests pass.  The existing GPU
test module has one pre-existing mock-expectation failure at this parent
commit because it does not expect the already-retained
`continuation_pools=None` keyword; the streaming tests do not touch that path.

## Focused PFAM gate

Slurm job 1187459 used 64 verified physical cores on `node-344-8t-1` and the
full 27,481-profile PFAM database against the first 4,096 PLM2_5 targets.  It
ran measured eager/stream/stream/eager arms with no smoke or prime request.
All four outputs were byte-identical: 607,184 bytes, 6,045 lines, SHA-256
`71afaf3a065c53e12a34755c22a0c7592bd87e634da2c981c3d4e2d57f1f04e4`.

| mode | median wall | median peak RSS |
|---|---:|---:|
| eager | 10.30 s | 1,764,596 KiB |
| streamed | 9.82 s | 942,740 KiB |

The exact focused result is a 46.57% RSS reduction and 4.66% wall reduction.
It passes the predeclared advance gate (at least 20% less RSS and no more than
1% runtime regression).  No full PFAM or KOFAM job was submitted by this
experiment.

## CPU portability of other retained GPU work

The completion-balanced GPU continuation scheduler does not directly improve
ordinary `pyhmmer.hmmsearch`: PyHMMER's query dispatcher already keeps a
shared HMM work queue filled.  Its ordered result deque can retain many
completed `TopHits` behind a slow older profile, which is a memory target, but
not a worker-refill bottleneck until the queue drains.

Profile sharding is algorithmically portable, and PyHMMER already has exact
target-parallel search plus `TopHits.merge`.  A useful pure-CPU hybrid would
keep normal profiles query-parallel and split predicted stragglers by target
residue count.  The GPU sparse-v3 sharding implementation itself is not
portable because it begins after authenticated F2 candidate generation.

The exact AVX-512 four-candidate Forward/Backward primitive is also portable
in principle but not drop-in.  It begins at the private filter-score seam;
stock CPU `hmmsearch` calls the monolithic Plan7 pipeline once per target.  A
pure-CPU port needs an internal prefilter seam that batches four F2 survivors,
runs Forward4/Backward4, then resumes scalar domain work in canonical target
order.  Existing full-PFAM continuation evidence bounds the likely benefit:
it removed 15.92% aggregate continuation CPU and 4.37% continuation wall when
the GPU had already supplied F2 scores.  Pure CPU includes additional F0--F2
work, so 15.92% is an optimistic whole-search ceiling rather than an expected
speedup.
