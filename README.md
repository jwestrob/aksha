# Aksha

HMMER-powered sequence search with a clean Python API and CLI. Built on PyHMMER, supports protein and nucleotide workflows, and manages reference databases for you.

---

## Install

Requires Python 3.10+.

```bash
pip install -e .          # editable install
pip install -e .[dev]     # with dev/test extras
```

## Quickstart

```bash
aksha database install PFAM                                        # one-time setup
aksha search --sequences dummy_dataset --hmms PFAM --cut_ga -o results.tsv
head results.tsv
```

Add `-v` for verbose logging.

## CLI Commands (overview)
- `aksha search` – hmmsearch style: protein sequences vs HMM profiles.
- `aksha scan` – hmmscan style: HMM profiles vs protein sequences.
- `aksha phmmer` – protein vs protein DB.
- `aksha jackhmmer` – iterative protein search.
- `aksha nhmmer` – nucleotide sequences vs nucleotide HMMs.
- `aksha database` – list/install databases (e.g., PFAM); custom paths also supported.

## Inputs
- Sequences: file or directory; accepts `.faa .fa .fasta .fas .fna`. Default mode is protein; use `nhmmer` for nucleotide.
- HMMs: single `.hmm` file, directory of `.hmm` files, or installed DB name (e.g., `PFAM`).
- Threads: `--threads 0` uses all available cores.

## Thresholds (cutoffs)
- `--cut_ga`, `--cut_tc`, `--cut_nc`: use gathering/trusted/noise cutoffs if present in the HMMs.
- `--cascade`: tries trusted → gathering → noise → none (first available).
- Numeric: `-E/--evalue`, `-T/--bitscore`, `--domE`, `--domT`.
- If nothing is given, PyHMMER inclusion defaults apply.

## Outputs
- File layout:
  - If `-o` ends with `.tsv`, results are written there.
  - If `-o` is a directory, it is created if missing and `search_results.tsv` (or `scan_results.tsv`, etc.) is written inside.
- Columns: `sequence_id, hmm_name, evalue, bitscore, c_evalue, i_evalue, dom_bitscore, env_from, env_to, ali_from, ali_to`.
- Large runs: when hits exceed ~100k, collection spills to disk (`.aksha_temp_results.tsv` in the output dir) before finalizing.

## Databases & Paths
- Default directories follow platform conventions via [platformdirs](https://github.com/tox-dev/platformdirs):
  - Linux: `~/.local/share/aksha`, `~/.config/aksha`
  - macOS: `~/Library/Application Support/aksha`
  - Windows: `%APPDATA%\aksha`
- Environment overrides: `AKSHA_DATA_DIR`, `AKSHA_CONFIG_DIR`.
- Installed DB metadata is tracked in `databases.json` in the data dir.

## Troubleshooting
- **0 or very few hits**: thresholds too strict (`--cut_ga` can be tight); try `-E 1e-3` or `-T 20`. Ensure protein vs nucleotide mode matches your sequences/HMMs.
- **HMM parse count looks too small**: your input might be a directory with a single multi-model `.hmm`; we parse all models. Re-run with `-v` to see the “Parsed N HMMs” log.
- **Slow searches**: use `--threads 0` (all cores) and keep sequences grouped by file; avoid extremely small files in large counts.

## Library API

Everything is importable from `aksha` directly.

```python
import aksha

# Search protein sequences against an HMM database
result = aksha.search(
    sequences="dummy_dataset",
    hmms="PFAM",
    thresholds=aksha.ThresholdOptions(cut_ga=True),
    threads=4,
)

# Results as a pandas DataFrame (one row per domain hit)
df = result.to_dataframe()

# Or write straight to TSV (disk-backed results skip the DataFrame entirely)
result.to_csv("results.tsv")

# Iterate over hits lazily — works for both in-memory and disk-backed results
for hit in result:
    print(hit.sequence_id, hit.hmm_name, hit.bitscore)
    for dom in hit.domains:
        print(f"  {dom.env_from}-{dom.env_to}  {dom.bitscore:.1f}")
```

Other search modes follow the same pattern:

```python
# hmmscan (sequences as queries against HMM DB)
result = aksha.scan("proteins.faa", "PFAM")

# Nucleotide search
result = aksha.nhmmer("genomes/", "rRNA_HMMs/")

# Protein-vs-protein (no HMMs needed)
result = aksha.phmmer("query.faa", "target.faa")

# Iterative search (builds HMM from hits each round)
result = aksha.jackhmmer("query.faa", "target.faa", max_iterations=5)
```

Database management:

```python
aksha.install_database("PFAM")
aksha.list_databases()               # returns list of DatabaseInfo
aksha.register_custom_database(      # bring your own HMMs
    name="my_db",
    path="/path/to/models.hmm",
    molecule_type=aksha.MoleculeType.PROTEIN,
)
```

## Development
- Tests: `pytest`
- Lint: `ruff check`
- Type check: `mypy`

For deeper design notes and module-by-module behavior, see `AKSHA_SPEC.md`.
