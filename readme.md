# Aksha

**Big metagenomes. Fast answers.**

Aksha accelerates HMM-based sequence searches so you can spend more time on
biology and less time waiting for annotations. Built on [HMMER](http://hmmer.org/)
and [PyHMMER](https://pyhmmer.readthedocs.io/), it combines fast CPU searches,
optional NVIDIA GPU acceleration, database management and sequence retrieval
in one command-line tool.

## How fast?

**300,186 proteins. 27,481 Pfam models. 3 minutes 39 seconds.**

That's a complete PLM2_5 metagenome search using Aksha's production GPU path
on one NVIDIA H200. No GPU? The retained CPU benchmark was **2.43× faster
than MetaCerberus 1.4 on the same 64 physical CPU cores**.

| Tool / configuration | Pfam runtime | Speedup over MetaCerberus |
| --- | ---: | ---: |
| MetaCerberus 1.4, CPU | 11m 34s | — |
| Aksha, CPU | 4m 46s | 2.43× |
| Aksha, H200 GPU + CPU | 3m 39s | 3.17× |

Aksha's CPU and GPU outputs matched exactly in these runs. CPU comparisons
used the same machine; the GPU run used a separate H200 node with 64 host
cores. MetaCerberus timing covers its HMM search/filter/parse pipeline, not
additional reporting; its thresholds and output rules differ, so this is a
workflow comparison, not identical-work benchmarking. Measurements predate
wheel packaging. [CPU evidence](https://github.com/jwestrob/aksha/blob/main/docs/development/CPU_PRODUCTION_INTEGRATION.md)
and [GPU evidence](https://github.com/jwestrob/aksha/blob/main/docs/development/GPU_PRODUCTION_INTEGRATION.md).

## More than a fast search

- Search proteins or nucleotide sequences using supported databases or your own HMMs.
- Download and manage databases such as Pfam and KOfam from the same tool.
- Use a database's recommended cutoffs, or choose your own score and E-value thresholds.
- Export tabular results and optionally retrieve matching sequences for downstream analysis.

## Installation

The PyPI release is being prepared. Once published:

```bash
python -m pip install aksha
# With optional GPU support:
python -m pip install 'aksha[gpu]'
```

Currently supported: Linux x86-64 with Python 3.12. GPU use requires a
compatible NVIDIA GPU and driver. ARM, macOS and Windows are not yet supported.
See the [installation guide](https://github.com/jwestrob/aksha/blob/main/release/INSTALL.md)
for hardware requirements, local-wheel installation and GPU setup.

## Quick start

Download the Pfam protein-family database, then search your protein sequences:

```bash
aksha initialize --hmms PFAM
aksha search --prot_in proteins.faa --installed_hmms PFAM --cut_ga --outdir results
```

Results are written to `results/`. The example uses Pfam's built-in score
thresholds. You can also supply your own models with `--hmm_in` instead of
`--installed_hmms`. Run `aksha --help` or `aksha search --help` for more options.

The [database setup guide](https://github.com/jwestrob/aksha/blob/main/docs/database-setup.md)
explains available databases, storage locations and reusing existing files.
For source builds, see the [maintainer guide](https://github.com/jwestrob/aksha/blob/main/release/BUILD.md).

## License

Aksha's original code is MIT-licensed; third-party components retain their own
licenses. See the [application license](https://github.com/jwestrob/aksha/blob/main/LICENSE),
[native-code license](https://github.com/jwestrob/aksha/blob/main/native/README.md)
and [third-party notices](https://github.com/jwestrob/aksha/blob/main/release/THIRD_PARTY_NOTICES.md).
