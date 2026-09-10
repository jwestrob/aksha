# Aksha: HMM-based sequence search and retrieval

Aksha combines HMM database installation, protein sequence search, and exact
CPU/GPU acceleration of [PyHMMER](https://pyhmmer.readthedocs.io/) and HMMER 3.4.
Previously developed as Astra and Astra-GPU, it now has one production source
repository containing the application, native backend, and release recipes.

## Installation

The prepared versions are `aksha==0.2.0`, `aksha-runtime==0.1.0` and optional
`aksha-cuda12==0.1.0`. PyPI publication is pending; after upload:

```bash
python -m pip install aksha
# Optional NVIDIA backend:
python -m pip install 'aksha[gpu]'
```

Before publication, add `--find-links /path/to/release-bundle`.
Pip selects matching native wheels and ordinary dependencies automatically.
No source checkout, compiler, CUDA toolkit, container or environment image
is needed. The command and Python package are both named `aksha`.

Current wheels support Linux x86-64, CPython 3.12 and SSE4.1 CPUs; AVX-512 is
optional. GPU execution requires a compatible NVIDIA GPU/driver; H200 was
tested. ARM, macOS, Windows and other Python versions are not yet supported.
See the [installation guide](https://github.com/jwestrob/aksha/blob/main/release/INSTALL.md)
for support details.

## Usage

```bash
aksha --help
aksha initialize --show_available
aksha initialize --hmms PFAM
aksha search --prot_in proteins.faa --installed_hmms PFAM --cut_ga --outdir results
```

Use `--hmm_in` for custom HMMs instead of downloading a database. Cutoff
options depend on the database; PFAM provides curated gathering thresholds.
See the [database setup guide](https://github.com/jwestrob/aksha/blob/main/initialize_usage_guide.md)
for storage locations, existing databases and reinstalling safely.
GPU installation does not automatically enable GPU searches: use a local
pressed-database manifest and `--gpu-manifest DB=PATH`, documented in
the [installation guide](https://github.com/jwestrob/aksha/blob/main/release/INSTALL.md).

Large eligible CPU searches stream pressed profiles in bounded chunks. The
qualified allocator policy applies to console searches using 64 or more
threads. Legacy `ASTRA_*` tuning variables remain unchanged, including
`ASTRA_CPU_STREAM_PRESSED` and `ASTRA_CPU_MALLOC_ARENA_MAX`; existing
`MALLOC_ARENA_MAX` values are respected. No numerical policy changed.

## Source and development

- `aksha/`: application, CLI, database setup and output.
- `native/`: production CPU/CUDA sources, private bindings and five upstream patches.
- `release/`: single-checkout recipes, provenance and checks.

The root pyproject directly declares the matching runtime and GPU extra.
Editable installs use that same contract; there is no second app variant or
hidden dependency rewrite. Private names `astra_pyhmmer`, `plan7_gpu`,
`libastra_hmmer.so` and `libastra_easel.so` remain stable for ABI compatibility.
Stock `pyhmmer` may coexist but its native objects are not interchangeable.

See the [release notes](https://github.com/jwestrob/aksha/blob/main/RELEASE.md)
and [build guide](https://github.com/jwestrob/aksha/blob/main/release/BUILD.md). This
naming release promotes no experimental algorithms or new benchmark claims.
The root CPU/GPU integration, audit and experiment documents are historical
engineering records; their old names, paths and timings are not current
installation instructions.

## License and contributions

Original application code uses [MIT](https://github.com/jwestrob/aksha/blob/main/LICENSE);
original native additions carry their MIT grant in the
[native README](https://github.com/jwestrob/aksha/blob/main/native/README.md).
Upstream notices remain intact: [inventory](https://github.com/jwestrob/aksha/blob/main/release/THIRD_PARTY_NOTICES.md).
Contributions and database suggestions are welcome. Future upstream work
with Martin Larralde starts with cleanup, a narrow reusable interface proposal
and existing evidence, not the experimental development tree.
