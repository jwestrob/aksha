
# Astra: Scalable HMM-based Sequence Search and Retrieval

<img src="img/astra_logo.png" width="50%">
	
## Overview

Astra is a Python package designed to facilitate bioinformatic workflows involving Hidden Markov Models (HMMs). It serves as a wrapper around [PyHMMER](https://pyhmmer.readthedocs.io/en/stable/index.html) and [HMMER](http://hmmer.org/), as well as a compendium of downloadable HMM databases. It automates the process of downloading and searching with custom or pre-installed HMM databases. Astra aims to streamline bioinformatic analyses, allowing for greater flexibility and ease of use. I intend to make Astra into a capable command-line tool as well as a python library.

## Features

(Just what I've implemented so far! This list will grow!)

- **Initialize**: Download and install HMM databases directly from various sources with a simple command.
- **Search**: Perform advanced HMM searches on sequence data with customizable options.

## Installation

To install the package, use pip:

```bash
pip install astra-hmm
```

The distribution is named `astra-hmm` because `astra` was already taken on
PyPI; the command and the importable package are both still `astra`.

To install from a clone:

```bash
pip install -e .
```

Dependencies (all installed automatically via pip):

- [pyhmmer](https://pyhmmer.readthedocs.io/) >= 0.10
- pandas >= 2.0
- tqdm
- requests
- platformdirs

Python 3.9 or newer. No external binaries are required — HMMER itself comes
bundled with PyHMMER.

## Usage

### Initialization

This is not necessary if you have locally installed HMMs. You can specify those without ever running 'Astra initialize'.

If you would like to view the available HMM databases for install:
```bash
Astra initialize --show_available
```


To initialize and download one of these databases:

```bash
Astra initialize --hmms database_name
```

Or to install them all (takes quite a bit of time!):

```bash
Astra initialize --hmms all_prot
```


### Search

To perform an HMM search:

```bash
Astra search --prot_in your_fasta_file --installed_hmms database_name  --cut_ga --outdir example_output
```

#### Combined Search

To perform an HMM search using custom HMM files:

```bash
Astra search --prot_in your_fasta_file --hmm_in custom_db --installed_hmms pre_installed_db --cut_ga --outdir example_output
```

#### CPU memory policy

Large ordinary CPU searches of installed pressed databases automatically read
profiles in bounded chunks.  Custom/unpressed databases, small databases,
GPU, cascade, and MacSyFinder searches keep the ordinary eager path.  Set
`ASTRA_CPU_STREAM_PRESSED=0` to disable this optimization; `auto` is the
default and `1` requests it without overriding the safety checks.

On Linux/glibc, an `astra search --threads 64` (or higher) console invocation
also starts with a validated 24-arena allocator limit.  Existing
`MALLOC_ARENA_MAX` values are preserved.  Set
`ASTRA_CPU_MALLOC_ARENA_MAX=0` to opt out, or set it to a positive integer to
choose another value.  Programmatic Astra use and lower-thread searches are
unchanged.

## Contributing & License

Contributions are welcome! Especially if you have database suggestions. Feel free to raise an issue if you'd like to add a database to the installable list. Feature requests will be considered and implemented if I have the time and ability.

---

MIT License

Copyright (c) 2023 Jacob West-Roberts

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
