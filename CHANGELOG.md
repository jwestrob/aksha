# Changelog

## Unreleased

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
