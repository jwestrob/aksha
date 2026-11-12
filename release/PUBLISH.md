# User-operated publication

## Application-only release (0.2.2)

`aksha-runtime==0.1.1` and `aksha-cuda12==0.1.1` are already on PyPI and stay unchanged.
From a clean checkout of the release commit:

```bash
python -m pip install build twine
python -m build --outdir dist
twine check dist/aksha-0.2.2-py3-none-any.whl dist/aksha-0.2.2.tar.gz
twine upload dist/aksha-0.2.2-py3-none-any.whl dist/aksha-0.2.2.tar.gz
```

## Native bundle release

Nothing is uploaded by the build/check recipes. Once `release-manifest.json`
says `READY_FOR_USER_UPLOAD`, open a shell in the stable `release-bundle`
directory and run:

```bash
sha256sum --check SHA256SUMS
twine check *.whl aksha-0.2.0.tar.gz
twine upload aksha_runtime-*.whl aksha_cuda12-*.whl aksha-*.whl aksha-0.2.0.tar.gz
```

The upload command uses your configured PyPI credentials; never put the token
in a command, repo, or log. Upload the matching runtime and CUDA distributions
before the app. Availability is not a reservation: PyPI must accept each new
project name. Do not overwrite/delete an existing published version. If an
upload is interrupted, inspect which exact files PyPI accepted and resume only
the missing files, keeping the same tested artifacts.

After all uploads succeed, verify the project pages and do one small fresh
`pip install aksha` check from PyPI (without local `--find-links`). Mark
publication complete in the docs only then. GitHub repository renaming is a
separate operation; local paths are intentionally unchanged in this release.
