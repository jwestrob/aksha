# License inventory

Original application: MIT, root LICENSE, Jacob West-Roberts (2023).
Original native additions: MIT, native/README.md, Jacob West-Roberts (2026),
explicitly selected by the owner. The latter grant is included in both native
wheels as AKSHA-NATIVE-LICENSE.txt; it does not relicense upstream work.

| Included component | License | Original notice |
| --- | --- | --- |
| PyHMMER 0.12.0, Martin Larralde | MIT | COPYING |
| HMMER | BSD-3-Clause | vendor/hmmer/LICENSE |
| Easel | BSD-2-Clause | vendor/easel/LICENSE |
| Statically linked NVIDIA CUDA runtime | NVIDIA CUDA EULA | NVIDIA-CUDA-EULA.txt |

Original wheel notices are preserved byte-for-byte. The five pinned additive
patches are in native/patches; upstream file-level notices remain intact when
preparing sources. The original upstream archive is pinned in source-lock.json
and may be supplied offline. This is an Aksha modification, not an upstream
PyHMMER release. Separately installed dependencies retain their own licenses.
No CUDA toolkit or environment image is delivered to users.
