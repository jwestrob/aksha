# Aksha

HMMER-powered sequence search with a clean Python API and CLI. Built on PyHMMER, supports protein and nucleotide workflows, and manages reference databases for you.

---

## Install
- Prereqs: Python 3.10+, pip, a C compiler for PyHMMER wheels if no binary is available.
- Editable install: `pip install -e .`
- Optional dev extras: `pip install -e .[dev]`

## Quickstart (protein, PFAM)
- Install PFAM once: `aksha database install PFAM`
- Run a search on the provided toy data:  
  `aksha search --sequences dummy_dataset --hmms PFAM --cut_ga -o results.tsv`
- Open TSV: `cat results.tsv | head`
- Verbose logging for debugging: add `-v`.

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
- Default data dir: `~/.local/share/aksha`; config dir: `~/.config/aksha`.
- Environment overrides: `AKSHA_DATA_DIR`, `AKSHA_CONFIG_DIR`, or XDG vars.
- Installed DB metadata is tracked in `databases.json` in the data dir.

## Troubleshooting
- **0 or very few hits**: thresholds too strict (`--cut_ga` can be tight); try `-E 1e-3` or `-T 20`. Ensure protein vs nucleotide mode matches your sequences/HMMs.
- **HMM parse count looks too small**: your input might be a directory with a single multi-model `.hmm`; we parse all models. Re-run with `-v` to see the “Parsed N HMMs” log.
- **Slow searches**: use `--threads 0` (all cores) and keep sequences grouped by file; avoid extremely small files in large counts.

## Library API (tiny teaser)
```python
from aksha.search import search
from aksha.types import ThresholdOptions

result = search(
    sequences="dummy_dataset",
    hmms="PFAM",
    thresholds=ThresholdOptions(cut_ga=True),
    threads=4,
)
df = result.to_dataframe()
```

## Development
- Tests: `pytest`
- Lint: `ruff check`
- Type check: `mypy`

For deeper design notes and module-by-module behavior, see `AKSHA_SPEC.md`.
