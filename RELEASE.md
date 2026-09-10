# Aksha consolidated release

This checkout is the canonical source for the application and native runtime.
Application Git history remains here; the original Astra-GPU repository and
its history remain intact. No experimental tree was merged wholesale.
`release/source-migration.json` records original revisions and file hashes.

The imported native sources are byte-identical to the qualified, privately
namespaced v4 build inputs. The retained Forward4 unit is now
`native/cpu/forward4_avx512.cpp`. The application is mechanically renamed
to `aksha` and directly imports the same private runtime as the tested wheels.
Legacy `ASTRA_*` knobs and native-library names remain unchanged.

| Distribution | Contents |
| --- | --- |
| `aksha==0.2.0` | Python application and `aksha` command |
| `aksha-runtime==0.1.0` | Private PyHMMER/HMMER/Easel and CPU continuation |
| `aksha-cuda12==0.1.0` | Optional CUDA extension |

Plain `aksha` selects two wheels; `aksha[gpu]` selects three. The root
pyproject and released metadata both pin the same internal dependencies.

Final-version labels describe the packaging release, not a new scientific
implementation. Native wheels are metadata-only repackagings of the tested
rc2 binaries, with byte-identity assertions and the ABI guard intact. The app
wheel must match the deterministic rename of the tested application.

Existing CPU/GPU exactness and ownership evidence is retained. Only a bounded
fresh installation and small exact-output check accompany the rename. Full
benchmarking and ARM are separate work. Publication is performed by the user;
no recipe reads credentials or uploads artifacts.

The 2026-09-10 renamed-wheel install check completed as Slurm1194528: CPU and
GPU-extra installation/import/ABI/CLI checks passed in a fresh environment,
and the one tiny real-fixture output matched exactly. The 13 launcher tests
also passed in that environment. All three wheels and the sdist passed Twine
checks; standalone sdist preparation reproduced all 655 pinned numerical
source files without a second checkout or native compilation.
The stable handoff remains
`/groups/banfield/projects/environmental/sr/srvp2020/Jacob/hmmer_gpu/build/pip-release-candidate-v4/release-bundle`.
Its manifest records `READY_FOR_USER_UPLOAD`, meaning local wheel readiness,
not publication on PyPI. See [publication instructions](release/PUBLISH.md)
for the user's commands. The original v4
artifacts/evidence remain untouched outside this refreshed handoff directory.

The subsequent documentation cleanup updates this checkout's database guide
and repository URLs to `https://github.com/jwestrob/aksha`. The owner renamed
the GitHub repository, and that canonical name and `main` branch were verified
before the source push. Local checkout and artifact paths are unchanged.
Historical native provenance retains the original repo name.
The already checked handoff artifacts are intentionally unchanged, not
rebuilt or requalified for documentation edits. Their embedded README and
repository metadata reflect the earlier handoff snapshot. Pushing source to
GitHub does not upload the wheels: PyPI publication remains pending until an
upload is confirmed.

The source distribution contains app/native sources, five pinned patches and
build recipes. Upstream PyHMMER is fetched by recorded SHA-256 for future native
builds, or supplied offline with `--sdist`. No other Git checkout is required.

Original licenses are in LICENSE and native/README.md; upstream notices retain
their scope. Cleanup-first collaboration with Martin Larralde remains future
work, not authorization for contact, an issue or a PR.
