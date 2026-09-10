# Installing Aksha

PyPI publication is pending. After upload, on Linux x86-64/CPython 3.12:

```bash
python -m pip install aksha
# Optional NVIDIA backend:
python -m pip install 'aksha[gpu]'
aksha --help
```

Before upload add `--find-links /path/to/release-bundle`. App version 0.2.0
pins runtime/CUDA 0.1.0. Pip selects the components automatically; ordinary
dependencies are downloaded separately. This is not a complete offline wheelhouse.

CPU requires SSE4.1; AVX-512 is optional. Native wheels have manylinux2014 /
glibc2.17 tags; ordinary dependencies have their own system requirements.
ARM, macOS, Windows and other Python versions are not supported by these files.
The CUDA backend uses CUDA12.5, SM75/SM90 cubins and compute75 PTX. H200 was
tested; other embedded targets are not hardware-qualification claims.
A compatible NVIDIA driver is required, but no CUDA toolkit is needed.
See [NVIDIA PTX compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Databases and GPU execution

```bash
aksha initialize --show_available
aksha initialize --hmms PFAM
aksha search --prot_in proteins.faa --installed_hmms PFAM --cut_ga --outdir results
```

Use `--hmm_in` for your own HMMs. No biological database is bundled.
See the [database setup guide](https://github.com/jwestrob/aksha/blob/main/initialize_usage_guide.md)
for choosing storage, reusing existing files and the destructive `--force` option.
The GPU extra makes the backend available; enable it with a local manifest:

```bash
python -c 'from plan7_gpu.pressed_manifest import create_pressed_manifest; create_pressed_manifest("/path/to/pressed/PFAM", "PFAM.manifest.json")'
aksha search --prot_in proteins.faa --installed_hmms PFAM --cut_ga --outdir results --gpu-manifest PFAM=PFAM.manifest.json
```

Use the pressed prefix without `.h3m`. Manifest generation authenticates the
database, not an HMM search. Regenerate after database/native-runtime changes.
Choose cutoff options appropriate to the database.

## Compatibility

Use a fresh environment when moving from old Astra packages. Aksha has its own
configuration directory (`~/.config/Aksha` on Linux by default), but the
catalog's `db_path` still defaults to `$HOME/.config/Astra`. Configuration and
database storage are separate; neither existing files nor installed flags are
automatically migrated. Set `db_path` before new downloads, or reuse HMM files
directly with `--hmm_in` instead of redownloading.
Legacy `ASTRA_*` knobs and private native names are preserved.
Do not inject older runtimes through PYTHONPATH/LD_LIBRARY_PATH/LD_PRELOAD.
Stock `pyhmmer` can coexist; its C objects are not interchangeable with
`astra_pyhmmer`. No new full-workload performance qualification is claimed.
