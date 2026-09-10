"""One small allocated-node rename/install check. No upload or full benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from prepare import record, save


def verified_bundle(bundle):
    report = json.loads((bundle / "release-manifest.json").read_text())
    for row in report["files"]:
        assert Path(row["path"]).name == row["path"]
        assert {**record(bundle / row["path"]), "path": row["path"]} == row
    return report


def smoke(bundle, baseline, output):
    assert os.environ.get("SLURM_JOB_ID"), "run this tiny check on an allocated node"
    report = verified_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for name in list(environment):
        if name in {"PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD"} or name.startswith("ASTRA_"):
            environment.pop(name)
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       LC_ALL="C", XDG_CONFIG_HOME=str(output / "config"),
                       XDG_CACHE_HOME=str(output / "cache"))

    def run(label, command):
        with (output / (label + ".log")).open("w") as log:
            subprocess.run([str(x) for x in command], cwd=output, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    run("venv", [sys.executable, "-m", "venv", output / "venv"])
    python = output / "venv/bin/python"
    cli = output / "venv/bin/aksha"
    pip = [python, "-m", "pip"]
    install = pip + ["install", "--no-index", "--only-binary=:all:", "--find-links", bundle,
                     "--find-links", baseline / "qualification-wheelhouse"]
    run("install-cpu", install + ["--report", output / "pip-cpu.json", "aksha==0.2.0"])
    run("pip-check-cpu", pip + ["check"])
    run("imports", [python, "-I", "-c", r'''
import importlib.metadata as md, importlib.util, json, pathlib, sys
import aksha, astra_pyhmmer
from aksha import search
from plan7_gpu import _abi, _pipeline
assert search.astra_pyhmmer is astra_pyhmmer
assert importlib.util.find_spec("astra") is None
assert importlib.util.find_spec("pyhmmer") is None
assert importlib.util.find_spec("plan7_gpu._native") is None
assert "plan7_gpu.adapter" not in sys.modules
assert _pipeline._ASTRA_TSV_RENDERER_ABI == 2
assert _pipeline.PYHMMER_PRIVATE_ABI_SHA256 == _abi.pyhmmer_abi_fingerprint()
for module in (aksha, astra_pyhmmer, _pipeline):
    assert pathlib.Path(module.__file__).is_relative_to(sys.prefix)
assert md.version("aksha") == "0.2.0"
assert md.version("aksha-runtime") == "0.1.0"
print(json.dumps({"private_abi": _pipeline.PYHMMER_PRIVATE_ABI_SHA256,
    "fixture_dir": str(pathlib.Path(astra_pyhmmer.__file__).parent / "tests/data"),
    "prefix": sys.prefix, "python": sys.version}))
'''])
    imported = json.loads((output / "imports.log").read_text())
    assert imported["private_abi"] == report["private_abi"]
    run("cli-help", [cli, "--help"])
    assert "aksha" in (output / "cli-help.log").read_text().lower()
    fixtures = Path(imported["fixture_dir"])
    hmm = fixtures / "hmms/txt/Thioesterase.hmm"
    sequences = fixtures / "seqs/938293.PRJEB85.HG003687.faa"
    old_build = json.loads((baseline / "build-result.json").read_text())
    reference, = [row for row in old_build["wheels"] if Path(row["path"]).name.startswith("astra_hmm-")]
    assert record(reference["path"]) == reference
    with zipfile.ZipFile(reference["path"]) as archive:
        archive.extractall(output / "reference-app")
    commands = {
        "reference": [python, "-I", "-c", "import sys; sys.path.insert(0, " + repr(str(output / "reference-app")) + "); from astra._launcher import main; main()"],
        "candidate": [cli],
    }
    requests = {}
    for name, command in commands.items():
        command += ["search", "--hmm_in", hmm, "--prot_in", sequences,
                    "--outdir", output / name, "--threads", "2", "--evalue", "1e-3"]
        run(name, ["/usr/bin/time", "-f", "%e %M", "-o", output / (name + ".time"), *command])
        seconds, rss = (output / (name + ".time")).read_text().split()
        result = output / name / "user_hmms_hits_df.tsv"
        assert result.stat().st_size > 0
        requests[name] = {"command": list(map(str, command)), "output": record(result),
                          "wall_seconds": float(seconds), "peak_rss_kib": int(rss)}
    assert requests["reference"]["output"]["sha256"] == requests["candidate"]["output"]["sha256"]
    run("install-gpu", install + ["--report", output / "pip-gpu.json", "aksha[gpu]==0.2.0"])
    run("pip-check-gpu", pip + ["check"])
    run("gpu-extra", [python, "-I", "-c", "import importlib.metadata as m, importlib.util as u; assert m.version('aksha-cuda12') == '0.1.0'; assert u.find_spec('plan7_gpu._native') is not None; print('PASS_EXTRA_INSTALL')"])
    run("freeze", pip + ["freeze"])
    save(output / "result.json", {
        "schema": "aksha-rename-install-v1", "passed": True,
        "job": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
        "artifacts": [row for row in report["files"] if row["path"].endswith((".whl", ".tar.gz"))],
        "private_abi": imported["private_abi"], "environment": imported,
        "fixtures": [record(hmm), record(sequences)], "requests": requests,
        "exact_output": True, "cpu_install": True, "gpu_extra_install": True,
        "gpu_execution_repeated": False, "benchmark": False,
        "scope": "Fresh pip dependency resolution/imports/ABI/CLI; one tiny real-fixture reference/candidate pair. GPU execution evidence carried from unchanged binary payloads.",
        "evidence": [record(p) for p in sorted(output.iterdir()) if p.is_file()],
    })
    print("PASS_AKSHA_RENAME_INSTALL", output / "result.json", flush=True)


def finalize(bundle, result):
    report = verified_bundle(bundle)
    review = json.loads(result.read_text())
    assert review["passed"] and review["exact_output"] and review["cpu_install"] and review["gpu_extra_install"]
    assert review["private_abi"] == report["private_abi"]
    assert review["artifacts"] == [row for row in report["files"] if row["path"].endswith((".whl", ".tar.gz"))]
    for row in review["evidence"]:
        assert record(row["path"]) == row
    for row in review["requests"].values():
        assert record(row["output"]["path"]) == row["output"]
    report.update(status="READY_FOR_USER_UPLOAD", rename_install_review=review,
                  rename_install_review_file=record(result))
    save(bundle / "release-manifest.json", report)
    (bundle / "SHA256SUMS").write_text("".join(record(p)["sha256"] + "  " + p.name + "\n"
        for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
    print("READY_FOR_USER_UPLOAD; not published")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize", type=Path)
    args = parser.parse_args()
    if args.finalize:
        finalize(args.bundle.resolve(), args.finalize.resolve())
    else:
        assert args.baseline and args.output
        smoke(args.bundle.resolve(), args.baseline.resolve(), args.output.resolve())
