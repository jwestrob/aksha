"""Package the Aksha rename using authenticated existing native binaries. No upload."""
import argparse
import copy
import email
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

from packaging.requirements import Requirement
from wheel.wheelfile import WheelFile

from migrate_sources import rename_app
from prepare import ROOT, HERE, record, save


def native_license():
    return ("MIT License\n" + (ROOT / "native/README.md").read_text().split("\nMIT License\n", 1)[1]).encode()


def repack_native(source, destination, name, version, runtime_version):
    filename = name.replace("-", "_") + "-" + version + "-" + source.name.split("-", 2)[2]
    target = destination / filename
    with zipfile.ZipFile(source) as old:
        old_meta, = [n for n in old.namelist() if n.endswith(".dist-info/METADATA")]
        old_info = old_meta.rsplit("/", 1)[0]
        new_info = name.replace("-", "_") + "-" + version + ".dist-info"
        md = email.message_from_bytes(old.read(old_meta))
        md.replace_header("Name", name)
        md.replace_header("Version", version)
        requirements = md.get_all("Requires-Dist", [])
        if requirements:
            del md["Requires-Dist"]
            for value in requirements:
                req = Requirement(value)
                if req.name == "astra-hmm-runtime":
                    value = "aksha-runtime==" + runtime_version
                md["Requires-Dist"] = value
        licenses = md.get_all("License-File", [])
        if licenses:
            del md["License-File"]
        for value in licenses:
            md["License-File"] = value.removeprefix("licenses/")
        md["License-File"] = "AKSHA-NATIVE-LICENSE.txt"
        if md["Summary"]:
            md.replace_header("Summary", "Coordinated " + ("CUDA backend" if name == "aksha-cuda12" else "CPU runtime") + " for Aksha")
        if md["Description-Content-Type"]:
            md.replace_header("Description-Content-Type", "text/markdown")
        else:
            md["Description-Content-Type"] = "text/markdown"
        md.set_payload((HERE / "THIRD_PARTY_NOTICES.md").read_text())
        with WheelFile(target, "w") as new:
            for info in old.infolist():
                if info.filename == old_info + "/RECORD":
                    continue
                renamed = copy.copy(info)
                if info.filename.startswith(old_info + "/"):
                    renamed.filename = new_info + info.filename[len(old_info):]
                new.writestr(renamed, md.as_bytes() if info.filename == old_meta else old.read(info.filename))
            new.writestr(new_info + "/licenses/AKSHA-NATIVE-LICENSE.txt", native_license())
        with WheelFile(target) as new:
            for filename in new.namelist():
                new.read(filename)  # Verify every RECORD digest.
            old_payload = {n for n in old.namelist() if not n.startswith(old_info + "/")}
            new_payload = {n for n in new.namelist() if not n.startswith(new_info + "/")}
            assert old_payload == new_payload
            for filename in old_payload:
                assert old.read(filename) == new.read(filename), filename
            for filename in licenses:
                relative = filename.removeprefix("licenses/")
                assert old.read(old_info + "/licenses/" + relative) == new.read(new_info + "/licenses/" + relative)
    return {"original": record(source), "artifact": target.name,
            "native_payload_byte_identical": True, "changes": "distribution name/version/dependencies, description, native MIT notice; RECORD recomputed"}


def verify_app(original, renamed):
    with zipfile.ZipFile(original) as old, WheelFile(renamed) as new:
        expected = set()
        for name in old.namelist():
            if not name.startswith("astra/") or name.endswith("/"):
                continue
            target = "aksha/" + name.removeprefix("astra/")
            expected.add(target)
            value = rename_app(old.read(name)) if name.endswith(".py") else old.read(name)
            assert new.read(target) == value, target
        assert expected == {n for n in new.namelist() if ".dist-info/" not in n and not n.endswith("/")}
        entry, = [n for n in new.namelist() if n.endswith(".dist-info/entry_points.txt")]
        assert "aksha = aksha._launcher:main" in new.read(entry).decode()
        for name in new.namelist():
            new.read(name)
    return len(expected)


def dependency_graph(wheels, lock):
    infos, owners = {}, {}
    for path in wheels:
        with zipfile.ZipFile(path) as archive:
            metadata_path, = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            info_dir = metadata_path.rsplit("/", 1)[0]
            md = email.message_from_bytes(archive.read(metadata_path))
            infos[md["Name"]] = md
            for value in md.get_all("License-File", []):
                assert info_dir + "/licenses/" + value in archive.namelist()
            for name in archive.namelist():
                if ".dist-info/" in name or name.endswith("/"):
                    continue
                assert not name.startswith(("astra/", "pyhmmer/", "pyhmmer.libs/")), name
                assert name not in owners, "overlapping owners: " + name
                owners[name] = md["Name"]
    versions = {"aksha": lock["app_version"], "aksha-runtime": lock["runtime_version"], "aksha-cuda12": lock["gpu_version"]}
    assert set(infos) == set(versions)
    assert all(infos[name]["Version"] == version for name, version in versions.items())
    assert infos["aksha"].get_all("Provides-Extra", []) == ["gpu"]
    selections = {}
    for extra in ("", "gpu"):
        selected, pending = set(), ["aksha"]
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            selected.add(name)
            for value in infos[name].get_all("Requires-Dist", []):
                req = Requirement(value)
                assert not req.name.startswith("astra-hmm")
                if req.marker and not req.marker.evaluate({"extra": extra if name == "aksha" else ""}):
                    continue
                if req.name in versions:
                    assert str(req.specifier) == "==" + versions[req.name]
                    pending.append(req.name)
        selections[extra or "cpu"] = sorted(selected)
    assert selections["cpu"] == ["aksha", "aksha-runtime"]
    assert selections["gpu"] == sorted(versions)
    return selections


def main(baseline, output):
    baseline = baseline.resolve(strict=True)
    output = output.resolve()
    build = json.loads((baseline / "build-result.json").read_text())
    assert build["cuda_built"] and build["post_repair_abi_equal"]
    for row in build["wheels"]:
        assert record(row["path"]) == row
    prep = json.loads((baseline / "preparation.json").read_text())
    migration = json.loads((HERE / "source-migration.json").read_text())
    assert record(baseline / "preparation.json")["sha256"] == migration["original_preparation_sha256"]
    for row in migration["files"]:
        assert record(ROOT / row["path"])["sha256"] == row["sha256"], row["path"]
    lock = json.loads((HERE / "source-lock.json").read_text())
    for name, expected in lock["patches"]:
        assert record(ROOT / "native/patches" / name)["sha256"] == expected
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "aksha" and project["version"] == lock["app_version"]
    approvals = json.loads((baseline / "qualification/approved-gates.json").read_text())
    reviews = {}
    for name in ("smoke-cpu", "smoke-gpu"):
        row = approvals[name]
        assert record(row["path"]) == row
        review = json.loads(Path(row["path"]).read_text())
        assert review["passed"]
        reviews[name] = review

    previous, old_files = None, set()
    if output.exists():
        previous = json.loads((output / "release-manifest.json").read_text())
        for row in previous["files"]:
            assert Path(row["path"]).name == row["path"]
            actual = record(output / row["path"])
            assert (actual["sha256"], actual["bytes"]) == (row["sha256"], row["bytes"]), "preserve edited bundle: " + row["path"]
            old_files.add(row["path"])
        assert {p.name for p in output.iterdir()} <= old_files | {"release-manifest.json", "SHA256SUMS"}, "unknown bundle files"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aksha-bundle-", dir=output.parent) as temporary:
        staged = Path(temporary)
        command = [sys.executable, "-m", "build", "--wheel", "--sdist", "--no-isolation", "--outdir", str(staged), str(ROOT)]
        with (staged / "app-build.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        app = staged / ("aksha-" + lock["app_version"] + "-py3-none-any.whl")
        originals = {Path(row["path"]).name.split("-", 1)[0]: Path(row["path"]) for row in build["wheels"]}
        app_files = verify_app(originals["astra_hmm"], app)
        changes = [repack_native(originals[old], staged, name, lock[version], lock["runtime_version"])
                   for old, name, version in (("astra_hmm_runtime", "aksha-runtime", "runtime_version"),
                                              ("astra_hmm_cuda12", "aksha-cuda12", "gpu_version"))]
        graph = dependency_graph(sorted(staged.glob("*.whl")), lock)
        sdist = staged / ("aksha-" + lock["app_version"] + ".tar.gz")
        with tarfile.open(sdist) as archive:
            names = archive.getnames()
            for required in ("aksha/search.py", "native/cpu/forward4_avx512.cpp", "native/python/plan7_gpu/_native.pyx", "release/prepare.py"):
                assert sdist.name.removesuffix(".tar.gz") + "/" + required in names, required
            assert not any("/experiments/" in n or "/.git/" in n for n in names)
        for name in ("INSTALL.md", "BUILD.md", "PUBLISH.md", "THIRD_PARTY_NOTICES.md"):
            shutil.copyfile(HERE / name, staged / name)
        report = {"schema": "aksha-release-handoff-v1", "status": "RENAMED_AWAITING_SMALL_INSTALL_CHECK",
                  "source_lock": lock, "source_migration": migration,
                  "private_abi": build["private_abi"], "native_wheel_changes": changes,
                  "application_payload_matches_rename": app_files, "single_command_install": graph,
                  "existing_installation_reviews": reviews, "native_license_declared": True,
                  "native_implementation_changed": False, "full_benchmark_qualified": False,
                  "published": False, "publication_method": "user-operated Twine; no credential reads by recipes",
                  "app_build_command": command,
                  "files": [{**record(p), "path": p.name} for p in sorted(staged.iterdir()) if p.is_file()]}
        save(staged / "release-manifest.json", report)
        (staged / "SHA256SUMS").write_text("".join(record(p)["sha256"] + "  " + p.name + "\n"
            for p in sorted(staged.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
        output.mkdir(exist_ok=True)
        new_files = {p.name for p in staged.iterdir()}
        for path in sorted(staged.iterdir()):
            os.replace(path, output / path.name)
        obsolete = sorted(old_files - new_files)
        for name in obsolete:
            (output / name).unlink()
    print(json.dumps({"bundle": str(output), "selection": graph, "native_bytes_unchanged": True,
                      "application_files_checked": app_files, "obsolete_generated_files_replaced": obsolete}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline, args.output)
