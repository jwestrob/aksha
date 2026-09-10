"""Prepare the native build from this single checkout and a pinned upstream sdist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def namespace(text):
    # Includes Cython's import/type names and pickle module names. Never alias
    # stock modules in sys.modules or pretend the two C type systems coincide.
    result = re.sub(r"\bpyhmmer\b", "astra_pyhmmer", text)
    # Package relocation is not a renaming of upstream documentation/projects.
    return result.replace("astra_pyhmmer.readthedocs.io", "pyhmmer.readthedocs.io")


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"expected one occurrence in {path}: {old!r}")
    path.write_text(text.replace(old, new))


def extract_sdist(archive, output):
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            parts = Path(member.name).parts
            if (not parts or parts[0] != "pyhmmer-0.12.0" or ".." in parts
                    or not (member.isdir() or member.isfile() or member.issym())):
                raise ValueError(f"unexpected source archive member: {member.name}")
            if member.issym():
                target = (output / member.name).parent / member.linkname
                root = (output / "pyhmmer-0.12.0").resolve()
                if Path(member.linkname).is_absolute() or not target.resolve().is_relative_to(root):
                    raise ValueError(f"source symlink escapes source tree: {member.name}")
        handle.extractall(output, filter="data")


def prepare(output, sdist=None):
    output = output.resolve()
    # Preserve failed attempts as well as successful ones; no recursive cleanup.
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads((HERE / "source-lock.json").read_text())
    recipes = output / "recipes"
    recipes.mkdir()
    for path in HERE.iterdir():
        if path.is_file():
            shutil.copyfile(path, recipes / path.name)
    manifest = {"schema": "astra-pip-preparation-v1", "status": "PREPARING",
        "source_lock": lock, "source_lock_file": record(HERE / "source-lock.json"),
        "production_promoted": False, "publish_allowed": False,
        "packaging_sources": [record(p) for p in sorted(recipes.iterdir()) if p.is_file()]}
    save(output / "preparation.json", manifest)
    archive = output / "pyhmmer-0.12.0.tar.gz"
    if sdist:
        shutil.copyfile(sdist, archive)
    else:
        with urllib.request.urlopen(lock["sdist_url"], timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
    if record(archive)["sha256"] != lock["sdist_sha256"]:
        raise ValueError("upstream source SHA256 mismatch")
    extract_sdist(archive, output)
    upstream = output / "pyhmmer-0.12.0"
    patches = output / "patches"
    patches.mkdir()
    for name, expected in lock["patches"]:
        patch = patches / name
        shutil.copyfile(ROOT / "native/patches" / name, patch)
        if record(patch)["sha256"] != expected:
            raise ValueError(f"patch mismatch: {name}")
        result = subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
            cwd=upstream, text=True, capture_output=True)
        save(patches / (name + ".json"), {"returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr})
        result.check_returncode()
    numerical = {str(p.relative_to(upstream)): record(p)["sha256"]
                 for p in sorted((upstream / "vendor").rglob("*")) if p.is_file()}
    manifest["patched_numerical_sources"] = numerical

    # Only bindings/declarations change namespace. HMMER/Easel source does not.
    bindings = upstream / "src/pyhmmer"
    bindings.rename(upstream / "src/astra_pyhmmer")
    transformed = []
    for base in (upstream / "src/astra_pyhmmer", upstream / "include"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".pyx", ".pxd", ".pxi", ".pyi"}:
                old = path.read_text()
                new = namespace(old)
                if new != old:
                    path.write_text(new)
                    transformed.append(str(path.relative_to(upstream)))
    replace_once(upstream / "src/CMakeLists.txt", "add_subdirectory(pyhmmer)",
                 "add_subdirectory(astra_pyhmmer)")
    replace_once(upstream / "src/astra_pyhmmer/platform/CMakeLists.txt",
                 '"include" "pyhmmer" "platform"',
                 '"include" "astra_pyhmmer" "platform"')
    replace_once(upstream / "CMakeLists.txt",
        "project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)",
        "project(astra_pyhmmer VERSION 0.12.0 LANGUAGES C)")
    for target, output_name in (("libhmmer", "astra_hmmer"), ("libeasel", "astra_easel")):
        cmake = upstream / "src" / ("hmmer" if target == "libhmmer" else "easel") / "CMakeLists.txt"
        with cmake.open("a") as handle:
            handle.write(f'\nset_target_properties({target} PROPERTIES OUTPUT_NAME "{output_name}")\n')
    replace_once(upstream / "src/astra_pyhmmer/__init__.py", "from . import errors",
        "from ._cpu_check import require_sse41 as _require_sse41\n"
        "_require_sse41()\ndel _require_sse41\n\nfrom . import errors")
    (upstream / "pyproject.toml").write_text(
        (HERE / "runtime-pyproject.toml.in").read_text().replace("@VERSION@", lock["runtime_version"]))
    (upstream / "ASTRA_RUNTIME.md").write_text(
        "# Aksha private native runtime\n\nBuilt from PyHMMER 0.12.0 "
        "by Martin Larralde and the HMMER/Easel projects, with Astra's qualified "
        "additive patches. Upstream notices are preserved. This is not an upstream "
        "PyHMMER release. Original Aksha native additions are MIT-licensed; "
        "upstream licenses remain unchanged.\n")
    assert numerical == {str(p.relative_to(upstream)): record(p)["sha256"]
        for p in sorted((upstream / "vendor").rglob("*")) if p.is_file()}
    manifest["namespace_transforms"] = transformed

    native = output / "native"
    shutil.copytree(ROOT / "native", native, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    app = output / "app"
    app.mkdir()
    shutil.copytree(ROOT / "aksha", app / "aksha", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "readme.md", "LICENSE"):
        shutil.copyfile(ROOT / name, app / name)
    manifest.update(status="PREPARED_NOT_VALIDATED", sdist=record(archive),
        staged_files=[record(p) for tree in (upstream, native, app)
                      for p in sorted(tree.rglob("*")) if p.is_file()],
        limitations=["Local CPU packaging proof; not portable-wheel or scientific qualification.",
            "Stock and private PyHMMER objects are distinct; no implicit interoperability.",
            "GPU extra requires the separately built, exact-version CUDA wheel.",
            "Preparation does not compile, validate or publish a release."])
    save(output / "preparation.json", manifest)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.sdist)
