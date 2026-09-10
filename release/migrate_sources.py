"""One-time authenticated source consolidation; no builds, jobs or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rename_app(data):
    """Only public Python/CLI branding; legacy ASTRA_* knobs stay unchanged."""
    text = data.decode()
    return re.sub(r"\bAstra\b", "Aksha", re.sub(r"\bastra\b", "aksha", text)).encode()


def namespace(data):
    text = re.sub(r"\bpyhmmer\b", "astra_pyhmmer", data.decode())
    return text.replace("astra_pyhmmer.readthedocs.io", "pyhmmer.readthedocs.io").encode()


def main(candidate, native_repo):
    candidate, native_repo = candidate.resolve(), native_repo.resolve()
    prep = json.loads((candidate / "preparation.json").read_text())
    lock = prep["source_lock"]
    assert subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip() == lock["astra_revision"]
    for entry in prep["staged_files"]:
        data = Path(entry["path"]).read_bytes()
        assert digest(data) == entry["sha256"] and len(data) == entry["bytes"], entry["path"]
    assert not (ROOT / "aksha").exists() and not (ROOT / "native").exists()
    app_inputs, native_inputs = [], []
    for source in sorted((candidate / "astra/astra").glob("*")):
        if not source.is_file():
            continue
        original = ROOT / "astra" / source.name
        data = original.read_bytes()
        assert (namespace(data) if source.suffix == ".py" else data) == source.read_bytes(), original
        app_inputs.append((source, rename_app(source.read_bytes()) if source.suffix == ".py" else source.read_bytes()))
    authenticated = {Path(row["path"]): row for row in prep["staged_files"]}
    for source in sorted((candidate / "native").rglob("*")):
        if source.is_file() and source.resolve() in authenticated:
            relative = source.relative_to(candidate / "native").as_posix()
            relative = relative.replace("experiments/post448_forward4_avx512.cpp", "cpu/forward4_avx512.cpp")
            native_inputs.append((source, ROOT / "native" / relative))
    for name, expected in lock["patches"]:
        assert digest((candidate / "patches" / name).read_bytes()) == expected

    # All guards run before the mechanical move/copy. Git retains app history;
    # the original native repository and its complete history are untouched.
    (ROOT / "astra").rename(ROOT / "aksha")
    rows = []
    for source, data in app_inputs:
        target = ROOT / "aksha" / source.name
        target.write_bytes(data)
        rows.append({"path": str(target.relative_to(ROOT)), "sha256": digest(data),
                     "prepared_path": str(source.relative_to(candidate)), "prepared_sha256": digest(source.read_bytes()),
                     "change": "app namespace/branding only"})
    for source, target in native_inputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": str(target.relative_to(ROOT)), "sha256": digest(target.read_bytes()),
                     "prepared_path": str(source.relative_to(candidate)), "change": "byte-identical native source"})
    for name, _ in lock["patches"]:
        target = ROOT / "native/patches" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(candidate / "patches" / name, target)
    for path in (ROOT / "tests").glob("*.py"):
        path.write_bytes(rename_app(namespace(path.read_bytes())))
    for name in ("prepare.py", "build.py", "cpu_check.c", "build-requirements.txt", "runtime-pyproject.toml.in"):
        shutil.copyfile(native_repo / "release" / name, ROOT / "release" / name)
    native_license = (native_repo / "README.md").read_text().split("\nMIT License\n", 1)[1]
    (ROOT / "native/README.md").write_text(
        "# Aksha native runtime\n\nImported production sources retain the private `plan7_gpu`, "
        "`astra_pyhmmer` and native DSO names for ABI compatibility. The five upstream patches "
        "are in `patches/`; no paused CPU campaign is included.\n\n"
        "Original native additions use the owner's MIT grant below; upstream code retains "
        "its own notices. See ../release/THIRD_PARTY_NOTICES.md.\n\nMIT License\n" + native_license)
    lock.update(app_name="aksha", runtime_name="aksha-runtime", gpu_name="aksha-cuda12",
                app_version="0.2.0", runtime_version="0.1.0", gpu_version="0.1.0",
                schema="aksha-consolidated-source-v1")
    lock.pop("astra_version")
    (ROOT / "release/source-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    report = {"schema": "aksha-source-migration-v1", "native_repository": "https://github.com/jwestrob/astra-gpu",
              "native_revision": lock["native_revision"], "application_revision": lock["astra_revision"],
              "original_preparation_sha256": digest((candidate / "preparation.json").read_bytes()),
              "files": rows, "native_implementation_changed": False,
              "scope": "one canonical checkout; original repository histories preserved; no publication"}
    (ROOT / "release/source-migration.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CONSOLIDATED", len(app_inputs), "app files;", len(native_inputs), "native files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("native_repo", type=Path)
    args = parser.parse_args()
    main(args.candidate, args.native_repo)
