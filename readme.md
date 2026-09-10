# Aksha

Aksha searches biological sequences against collections of protein-family
models, helping you annotate genes and find sequences of interest. It uses
[HMMER](http://hmmer.org/) through [PyHMMER](https://pyhmmer.readthedocs.io/),
with optional NVIDIA GPU acceleration for protein searches.

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
