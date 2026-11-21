# Changelog

## Unreleased (0.2.4.dev0)

Needs a new `aksha-cuda12` build for the native parts; the application changes also
work with `aksha-cuda12==0.1.1`, which then falls back as described below instead of
aborting.

- GPU bias math survives operating-system updates. The CUDA backend trusted only the
  glibc 2.35 / libm build and CPU signature audited when each GPU was admitted, so the
  Ubuntu 24.04 upgrade (glibc 2.39) disabled it and every GPU search aborted. The CUDA
  extension now runs that same exhaustive audit itself: every positive finite binary32
  input, device `log` versus host libm `log`, zero mismatches required. A GPU search
  runs it once per environment (about 2 s on an H200 with 28 threads) and caches the
  result under a key of the CUDA extension, libm build ID, glibc, CPU identity, device
  product, CUDA runtime and driver; any change selects a fresh audit. The built-in
  environments stay trusted without an audit.
- A GPU search in an untrusted environment completes on the exact path: the GPU runs
  only the SSV filter, the bias filter and every later stage run on the CPU, and one
  warning line names what is untrusted and how to restore GPU use. The native
  post-filter no longer raises "post-filter workspace is unavailable" when it retains
  every candidate for the CPU (or receives none): the resident F2 compaction reports an
  unsupported view and the host records are used.
- `aksha gpu-check --gpu-manifest DB=PATH ...` reports, per database manifest and CUDA
  device, whether a search would run on the GPU, run the one-time self-attestation
  first, fall back to SSV-only, or refuse (manifest or device), and why. `--attest`
  runs the self-attestation now; `--json` prints the report. Exit status 0 ready,
  2 pending, 1 fallback. GPU searches run the same evaluation at start.
- `AKSHA_GPU_SELF_ATTEST=auto|never|force` controls the search-time audit;
  `AKSHA_GPU_ATTESTATION_DIR` moves the record cache (default
  `$XDG_CACHE_HOME/aksha/gpu-attestation`, else `~/.cache/aksha/gpu-attestation`).
  `ASTRA_GPU_UNTRUSTED_NATIVE_FALLBACK=1` (audit control) keeps the GPU session route
  in an untrusted environment and relies on the native per-stage CPU fallback.

## 0.2.3 (2026-11-20)

Application-only release; it uses the published `aksha-runtime==0.1.1` and
`aksha-cuda12==0.1.1` wheels unchanged.

- KOfam cutoffs follow `ko_list`'s `score_type`. A `full` KO's threshold is the
  sequence cutoff and a `domain` KO's threshold the domain cutoff; the other slot is
  -1,000,000 bits. `--cut_ga`/`--cut_tc`/`--cut_nc` searches of KOFAM and RP16 now assign
  KOs by KofamScan's rule; earlier installs also required a domain to reach a
  full-score threshold, which dropped hits just above it. A passing full-score hit
  now lists all of its domains.
- `aksha initialize --restamp --hmms KOFAM,RP16` re-stamps installed databases from
  the local `ko_list` and presses them again, without downloading. The registry
  records `"cutoff_encoding": "score_type"`, and a search with bit cutoffs warns when
  a KOFAM or RP16 install lacks it.
- RP16 markers are stamped from `ko_list` at install, including those copied from a
  local KOFAM.
- `aksha search --cpu-parallel {auto,queries,targets}` chooses how CPU searches spread
  work over threads (`targets` shares each HMM across workers); `auto` keeps the
  runtime's default. Results are identical across strategies. GPU searches reject it.
- `aksha --version` prints the installed version.

## 0.2.2 (2026-11-12)

Application-only release; it uses the published `aksha-runtime==0.1.1` and
`aksha-cuda12==0.1.1` wheels unchanged.

- `aksha initialize --hmms KOFAM` completes on pandas 3. The `ko_list` thresholds are
  read with `pd.to_numeric`, which works with the str column pandas 3 infers.
- A KOFAM install is recorded in Aksha's own registry (the config directory's
  `hmm_databases.json`), so `aksha search --installed_hmms KOFAM` finds it when the
  databases live under a different `db_path`.
- dbCAN installs from the current dbCAN HMM release
  (`https://pro.unl.edu/dbCAN2/download/Databases/V14/dbCAN-HMMdb-V14.txt`). A single-file
  download that is neither HMM text nor an archive now stops the install with a message,
  leaving nothing marked as installed.
- `aksha initialize --hmms A,B,...` keeps every database's record: each installer now writes
  only its own registry entry, where KOFAM's or HydDB's record could be overwritten by a later
  install in the same command.
- Rerunning a KOFAM install whose profiles are already extracted (for example after the
  pandas 3 failure) resumes from them instead of downloading the 1.6 GB archive again.
- New database: `CCTyper`, CRISPRCasTyper v1.9.0's 704 Cas profiles with its typing tables
  (`aksha initialize --hmms CCTyper`; needs `zstandard`, now a dependency).
- Package metadata pins the runtime and CUDA wheels that are on PyPI (0.1.1).

## 0.2.1 (2026-09-10)

First release on PyPI: application `aksha`, runtime `aksha-runtime==0.1.1` and optional
`aksha-cuda12==0.1.1`.
