# Changelog

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
- New database: `CCTyper`, CRISPRCasTyper v1.9.0's 704 Cas profiles with its typing tables
  (`aksha initialize --hmms CCTyper`; needs `zstandard`, now a dependency).
- Package metadata pins the runtime and CUDA wheels that are on PyPI (0.1.1).

## 0.2.1 (2026-09-10)

First release on PyPI: application `aksha`, runtime `aksha-runtime==0.1.1` and optional
`aksha-cuda12==0.1.1`.
