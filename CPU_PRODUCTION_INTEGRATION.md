# CPU production integration

## Retained implementation

The production CPU command now combines three independently exact changes:

1. Installed pressed-profile databases are streamed in bounded 2,000-profile
   chunks when the conservative automatic selector proves the request is an
   eligible large ordinary CPU bulk search.
2. When the exact plan7_gpu renderer ABI is installed, ordinary sorted
   `TopHits` rows are rendered natively.  Missing, mismatched, long-target,
   unsorted, or malformed private state uses Astra's public wrapper path.
3. The Linux/glibc console launcher sets `MALLOC_ARENA_MAX=24` before importing
   PyHMMER for explicit CPU searches with at least 64 threads.  Existing user
   settings win, `ASTRA_CPU_MALLOC_ARENA_MAX=0` opts out, and other platforms,
   commands, GPU searches, small thread counts, and library calls are unchanged.

## Exact full-workload evidence

All measurements below used 64 verified physical cores.

Full PLM2_5 x PFAM job 1187611 compared eager/public output against
streamed/native output in matched A/B arms.  Every arm reproduced SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.

| PFAM path | Mean wall (s) | Mean peak RSS (KiB) | Mean sink (s) |
| --- | ---: | ---: | ---: |
| eager + public | 286.553 | 2,668,402 | 4.994 |
| streamed + native | 285.708 | 1,814,110 | 1.993 |

The retained path reduced process RSS 32.02%, reduced sink time 60.09%, and
reduced wall 0.29%.

Full PLM2_5 x KOFAM job 1188063 tested the corrected native renderer together
with profile streaming and the 24-arena allocator policy at global sequence
E=1e-15.  It reproduced SHA-256
`fdd134e107fcc688be6a749493082c96eec6ce71e1c8b05e9bef0b8da076abc7`,
canonical SHA-256
`cfe8a3e4df17e243150036e55c4287c0606ae22a912c48336cb07f07bdf2ddef`,
301,039,592 bytes, and 2,994,533 lines.

| KOFAM path | Wall (s) | Peak RSS (KiB) | Sink (s) |
| --- | ---: | ---: | ---: |
| eager/public control mean | 2,189.609 | 16,384,764 | 32.234 |
| retained candidate | 2,182.566 | 11,973,724 | 13.182 |

The retained path reduced process RSS 26.92%, whole-cgroup current memory
26.72%, sink time 59.11%, and wall 0.32%.

The allocator was also isolated on full PFAM in job 1187652.  At 24 arenas it
reduced RSS 14.59% with a 0.59% wall difference and reproduced the exact PFAM
oracle.  Its automatic scope remains limited to Linux/glibc CPU searches with
an explicit thread count of at least 64.

## MetaCerberus comparison

The relevant preserved MetaCerberus value is its PFAM HMM-search portion,
693.97 s on 64 physical cores; downstream `getStats` is excluded.  The exact
retained Astra CPU PFAM result above is 285.708 s on 64 physical cores, or
2.43x faster by observed wall time.  Threshold/output semantics are different,
so this is an end-user HMM-search timing comparison rather than a claim of
identical work.

For memory, the retained Astra PFAM cgroup peak is about 1.72 GiB versus the
preserved MetaCerberus aggregate peak of 4.624 GiB.  KOFAM remains the larger
Astra memory target at 11.42 GiB process RSS even after the 26.92% reduction;
the production changes improve it materially without trading away runtime.
