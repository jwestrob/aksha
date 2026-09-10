from __future__ import annotations

from array import array
import hashlib
import importlib.util
from pathlib import Path
import platform
import sys


def _module_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {name}")
    return Path(spec.origin).resolve(strict=True)


def _abi_files() -> list[tuple[str, Path]]:
    package = _module_path("astra_pyhmmer").parent
    bundled = package.parent / "astra_pyhmmer.libs"
    files = [
        ("verifier", Path(__file__).resolve(strict=True)),
        ("extension/plan7", _module_path("astra_pyhmmer.plan7")),
        ("extension/easel", _module_path("astra_pyhmmer.easel")),
        ("library/hmmer", (bundled / "libastra_hmmer.so").resolve(strict=True)),
        ("library/easel", (bundled / "libastra_easel.so").resolve(strict=True)),
    ]
    for root, label in (
        (package, "package-pxd"),
        (bundled / "cython" / "include", "bundled-pxd"),
        (bundled / "include", "bundled-header"),
    ):
        patterns = ("*.pxd",) if label != "bundled-header" else ("*.h",)
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                files.append((f"{label}/{path.relative_to(root)}", path))
    return files


def pyhmmer_abi_fingerprint() -> str:
    digest = hashlib.sha256(b"plan7_gpu astra_pyhmmer private ABI v1\0")
    for label, path in _abi_files():
        encoded_label = str(label).encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "little"))
        digest.update(encoded_label)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_private_abi_platform() -> None:
    machine = platform.machine().lower()
    if sys.platform != "linux" or machine not in {"x86_64", "amd64"}:
        raise ImportError(
            "plan7_gpu._pipeline requires Linux/x86_64 for its private ABI"
        )
    if array("I").itemsize != 4 or array("Q").itemsize != 8:
        raise ImportError(
            "plan7_gpu._pipeline requires 32-bit array('I') and 64-bit array('Q')"
        )


if __name__ == "__main__":
    validate_private_abi_platform()
    print(pyhmmer_abi_fingerprint())
