# Single-checkout release build

The application, native sources, five pinned patches and recipes are here.
No second clone is required, including when using the source distribution.

## Future native builds

In a suitable Linux x86-64/CPython3.12 maintainer toolchain:

```bash
python -m pip install -r release/build-requirements.txt auditwheel==6.4.2 patchelf==0.17.2.4
python release/prepare.py --output /path/to/new-prepared
python release/build.py /path/to/new-prepared --manylinux manylinux_2_17_x86_64 --cuda-home /path/to/cuda-12.5
```

Use `--sdist /path/to/pyhmmer-0.12.0.tar.gz` to prepare offline; its hash is
always checked. Omit `--cuda-home` for CPU-only construction. Preparation
copies the app unchanged; only upstream PyHMMER receives the established
private namespace transformation. Native sources are already namespaced.

The original toolchain was GCC10.2.1, glibc2.17 and CUDA12.5. A suitable genuine
manylinux toolchain plus audit/repair is required, not merely retagging.
Precise math flags and post-repair ABI checks are preserved. A recompile is
not promised bit-identical. These are maintainer, not user, requirements.

## This metadata-only naming release

```bash
python release/bundle.py --baseline /path/to/pip-release-candidate-v4 --output /path/to/release-bundle
```

The baseline is an artifact/evidence input, not a second development repo.
The recipe verifies frozen hashes/reviews, builds only the Python app/source
distribution, and repackages native metadata. All native payload bytes must
remain identical; the app must equal the deterministic rename of the tested
app. Drift stops the recipe. Native ABI/library names remain unchanged.

The destination is stable. Existing manifest-listed files must match their
saved hashes before replacement. Obsolete generated filenames are removed
only after the new bundle is complete; original baseline wheels and reviews
are untouched.

For the renamed bundle, run `install_smoke.py --bundle BUNDLE --baseline BASELINE
--output NEW_DIRECTORY` on a small allocated CPU node. It creates a fresh venv,
resolves CPU and GPU-extra dependencies from local wheels, checks imports/ABI,
and compares one tiny real-fixture CLI output against the authenticated old
app. The same searches record time/RSS; they are not benchmark figures.
GPU execution is not repeated because every native payload byte is unchanged.
After inspecting the result and successful job exit, use
`install_smoke.py --bundle BUNDLE --finalize NEW_DIRECTORY/result.json` to
attach that evidence without rebuilding or changing the tested wheels.

Check wheel/source metadata with `python -m twine check` before the user
publishes; see [PUBLISH.md](PUBLISH.md). Recipes never read ~/.pypirc or upload.
Full benchmarks and ARM builds are separate work.

## Internal compatibility

Private module/library names (`astra_pyhmmer`, `plan7_gpu`, `libastra_hmmer.so`
and `libastra_easel.so`) and existing `ASTRA_*` tuning variables remain stable.
These include `ASTRA_CPU_STREAM_PRESSED` and `ASTRA_CPU_MALLOC_ARENA_MAX`;
existing `MALLOC_ARENA_MAX` values are respected. Stock PyHMMER can coexist,
but its native objects are not interchangeable with the private runtime's.

Historical integration, tuning and release-handoff records are archived in
[docs/development](../docs/development/). They retain their original names,
paths and measurements and are excluded from new source distributions.
