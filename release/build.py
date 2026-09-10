"""Build coordinated CPU/CUDA wheels from a fresh prepared tree; never publish."""
from __future__ import annotations

import argparse
import email
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import zipfile

from prepare import record, save


def add_native_license(info, native):
    text = "MIT License\n" + (native / "README.md").read_text().split("\nMIT License\n", 1)[1]
    licenses = info / "licenses"
    licenses.mkdir(exist_ok=True)
    (licenses / "AKSHA-NATIVE-LICENSE.txt").write_text(text)
    metadata = email.message_from_bytes((info / "METADATA").read_bytes())
    metadata["License-File"] = "AKSHA-NATIVE-LICENSE.txt"
    (info / "METADATA").write_bytes(metadata.as_bytes())


def run(args, log, env, cwd=None):
    args = list(map(str, args))
    print("running", log.name, flush=True)
    with log.with_suffix(".log").open("w") as stream:
        result = subprocess.run(args, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    save(log, {"command": args, "cwd": str(cwd) if cwd else None,
               "returncode": result.returncode,
               "build_environment": {k: env[k] for k in ("CC", "CXX", "CMAKE_BUILD_PARALLEL_LEVEL",
                   "CFLAGS", "CXXFLAGS", "PYTHONPATH", "LD_LIBRARY_PATH") if k in env}})
    result.check_returncode()


def pack(source, destination):
    from wheel.wheelfile import WheelFile
    destination.parent.mkdir(parents=True, exist_ok=True)
    with WheelFile(destination, "w") as wheel:
        wheel.write_files(source)
    return destination


def repair(wheel, directory, policy, log, env, exclude=()):
    if not policy:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / wheel.name
        shutil.copyfile(wheel, target)
        return target
    run([sys.executable, "-m", "auditwheel", "repair", "--plat", policy,
         *[arg for name in exclude for arg in ("--exclude", name)],
         "--wheel-dir", directory, wheel], log, env)
    candidates = list(directory.glob("*.whl"))
    if len(candidates) != 1:
        raise RuntimeError("expected one repaired wheel")
    return candidates[0]


def build_gpu(output, prep, site, abi, env, logs, cuda_home, policy, work_name="gpu-build", reuse_host=None):
    native, work = output / "native", output / work_name
    work.mkdir()
    libs = site / "astra_pyhmmer.libs"
    nvcc = cuda_home / "bin/nvcc"
    run([nvcc, "--version"], logs / "20-cuda-toolchain.json", env)
    includes = ["-I" + str(p) for p in (sysconfig.get_path("include"), native / "cuda",
        libs / "include", libs / "include/libeasel", libs / "include/libhmmer")]
    generated = work / "_native.cpp"
    strict = ["-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-std=c++17"]
    objects = [work / "_native.o", work / "f3_threshold.o"]
    if reuse_host is None:
        run([sys.executable, "-m", "cython", "--cplus", "-3", "-I" + str(libs / "cython/include"),
            "-E", "HMMER_IMPL=SSE", "-E", "TARGET_SYSTEM=Linux", "-E", "PYHMMER_ABI_SHA256=" + abi,
            "-o", generated, native / "python/plan7_gpu/_native.pyx"], logs / "21-gpu-cython.json", env)
        run(["g++", *strict, "-msse4.1", *includes, "-c", generated, "-o", objects[0]],
            logs / "22-gpu-host.json", env)
        run(["g++", *strict, *includes, "-c", native / "cuda/f3_threshold.cc", "-o", objects[1]],
            logs / "23-gpu-f3.json", env)
    else:
        reused = []
        for obj in objects:
            source = reuse_host / obj.name
            shutil.copyfile(source, obj)
            reused.append(record(source))
        save(logs / "21-reused-host-objects.json", {"objects": reused, "private_abi": abi})
    # Preserve the qualified GPU architecture and CUDA math policy. NVCC's
    # ordinary precise div/sqrt, ftz=false and fmad=true defaults are unchanged.
    arch = ["--generate-code=arch=compute_75,code=sm_75",
        "--generate-code=arch=compute_75,code=compute_75",
        "--generate-code=arch=compute_90,code=sm_90"]
    for index, name in enumerate(("ssv_cuda", "bias_cuda", "postfilter_cuda", "forward_cuda",
                                  "backward_domain_cuda", "domain_rescore_cuda")):
        obj = work / (name + ".o")
        run([nvcc, "-O3", "-lineinfo", "-std=c++17", "-Xcompiler=-fPIC,-pthread",
            *arch, *includes, "-c", native / "cuda" / (name + ".cu"), "-o", obj],
            logs / f"{24 + index}-{name}.json", env)
        objects.append(obj)
    extension = work / ("_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    run([nvcc, "-shared", "--cudart=static", *arch, "-o", extension, *objects,
        "-L" + str(libs), "-Xlinker", "--no-as-needed", "-lastra_hmmer", "-lastra_easel",
        "-ldl", "-lpthread", "-Xlinker", "-rpath", "-Xlinker", "$ORIGIN/../astra_pyhmmer.libs"],
        logs / "30-gpu-link.json", env)
    run([cuda_home / "bin/cuobjdump", "--dump-resource-usage", extension],
        logs / "31-gpu-resources.json", env)
    assembled = work / "assembled"
    (assembled / "plan7_gpu").mkdir(parents=True)
    shutil.copyfile(extension, assembled / "plan7_gpu" / extension.name)
    lock = prep["source_lock"]
    name = "aksha_cuda12"
    info = assembled / f'{name}-{lock["gpu_version"]}.dist-info'
    info.mkdir()
    (info / "licenses").mkdir()
    shutil.copyfile(cuda_home / "EULA.txt", info / "licenses/NVIDIA-CUDA-EULA.txt")
    (info / "METADATA").write_text(
        f'Metadata-Version: 2.4\nName: aksha-cuda12\nVersion: {lock["gpu_version"]}\n'
        'Summary: Coordinated optional CUDA backend for Aksha\nRequires-Python: >=3.12,<3.13\n'
        f'Requires-Dist: aksha-runtime=={lock["runtime_version"]}\n'
        'License-File: NVIDIA-CUDA-EULA.txt\n\n'
        'CUDA extension for Aksha. Original native additions use MIT; NVIDIA terms apply to CUDA.\n')
    add_native_license(info, native)
    (info / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: astra-release\n"
        "Root-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n")
    save(info / "astra-build.json", {"private_abi": abi, "cuda": str(cuda_home),
        "architectures": arch, "cudart": "static", "runtime_dependency": lock["runtime_version"],
        "external_pair_libraries": ["libastra_hmmer.so", "libastra_easel.so"],
        "publish_allowed": False, "production_promoted": False})
    raw = pack(assembled, output / "gpu-raw" / f'{name}-{lock["gpu_version"]}-cp312-cp312-linux_x86_64.whl')
    # These two DSOs belong to the exact runtime dependency, not the GPU wheel.
    # Never duplicate them: doing so could create incompatible live C objects.
    return repair(raw, output / "gpu-repaired", policy, logs / "32-gpu-repair.json", env,
                  exclude=("libastra_hmmer.so", "libastra_easel.so"))


def main(output, policy=None, cuda_home=None):
    output = output.resolve(strict=True)
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("initial packaging proof requires Linux x86-64 CPython 3.12")
    prep = json.loads((output / "preparation.json").read_text())
    if prep["status"] != "PREPARED_NOT_VALIDATED":
        raise RuntimeError("source preparation did not finish")
    for entry in prep["staged_files"]:
        if record(entry["path"]) != entry:
            raise RuntimeError("staged source changed: " + entry["path"])
    logs, dist = output / "build-logs", output / "dist"
    logs.mkdir(exist_ok=False)
    dist.mkdir(exist_ok=False)
    shutil.copyfile(__file__, logs / "builder.py")
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "USER", "TMPDIR", "LANG"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", LC_ALL="C",
        CC="gcc", CXX="g++", CMAKE_BUILD_PARALLEL_LEVEL="2", CMAKE_GENERATOR="Ninja",
        CFLAGS="-fno-fast-math -ffp-contract=off", CXXFLAGS="-fno-fast-math -ffp-contract=off",
        SOURCE_DATE_EPOCH="1769106080")
    run(["gcc", "--version"], logs / "00-compiler.json", env)
    run([sys.executable, "-m", "pip", "freeze", "--all"], logs / "00-tools.json", env)
    upstream, native = output / "pyhmmer-0.12.0", output / "native"
    run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", output / "base-wheel", upstream],
        logs / "01-bindings.json", env)
    base = list((output / "base-wheel").glob("*.whl"))
    if len(base) != 1:
        raise RuntimeError("expected one base runtime wheel")
    guarded = output / "guarded-base"
    guarded.mkdir()
    with zipfile.ZipFile(base[0]) as wheel:
        wheel.extractall(guarded)
    guard = guarded / "astra_pyhmmer" / ("_cpu_check" + sysconfig.get_config_var("EXT_SUFFIX"))
    run(["gcc", "-O2", "-fPIC", "-shared", "-march=x86-64", "-mtune=generic",
        "-I" + sysconfig.get_path("include"), output / "recipes/cpu_check.c", "-o", guard],
        logs / "01-cpu-guard.json", env)
    guarded_wheel = pack(guarded, output / "guarded-wheel" / base[0].name)
    base_wheel = repair(guarded_wheel, output / "base-repaired", policy,
        logs / "01-base-repair.json", env)
    site = output / "bootstrap-site"
    run([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--target", site, base_wheel],
        logs / "02-bootstrap.json", env)
    env["PYTHONPATH"] = str(site)
    libs = site / "astra_pyhmmer.libs"
    module_source = native / "python/plan7_gpu"
    abi = subprocess.check_output([sys.executable, module_source / "_abi.py"], env=env, text=True).strip()
    if len(abi) != 64:
        raise RuntimeError("private ABI fingerprint missing")
    build = output / "native-build"
    build.mkdir()
    generated, obj, tail = build / "_pipeline.c", build / "_pipeline.o", build / "tail.o"
    run([sys.executable, "-m", "cython", "-3", "-I" + str(libs / "cython/include"),
        "-E", "HMMER_IMPL=SSE", "-E", "TARGET_SYSTEM=Linux", "-E", "PYHMMER_ABI_SHA256=" + abi,
        "-o", generated, module_source / "_pipeline.pyx"], logs / "03-cython.json", env)
    includes = ["-I" + str(p) for p in (sysconfig.get_path("include"), native / "cuda", native / "cpu",
        libs / "include", libs / "include/libeasel", libs / "include/libhmmer")]
    strict = ["-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off"]
    run(["gcc", *strict, "-std=c11", "-msse4.1", *includes, "-c", generated, "-o", obj],
        logs / "04-pipeline.json", env)
    run(["g++", *strict, "-std=c++17", "-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl",
        "-DPLAN7_AVX512_TAIL_LIBRARY", *includes, "-c", native / "cpu/forward4_avx512.cpp",
        "-o", tail], logs / "05-tail.json", env)
    extension = build / ("_pipeline" + sysconfig.get_config_var("EXT_SUFFIX"))
    run(["g++", "-shared", "-o", extension, obj, tail, "-L" + str(libs), "-Wl,--no-as-needed",
        "-lastra_hmmer", "-lastra_easel", "-Wl,-rpath,$ORIGIN/../astra_pyhmmer.libs"],
        logs / "06-link.json", env)

    # Assemble one distribution with one owner for the patched libraries and
    # bindings. WheelFile recomputes RECORD for every installed file.
    assembled = output / "assembled"
    assembled.mkdir()
    with zipfile.ZipFile(base_wheel) as wheel:
        wheel.extractall(assembled)
    info, = assembled.glob("*.dist-info")
    add_native_license(info, native)
    package = assembled / "plan7_gpu"
    package.mkdir()
    for path in module_source.glob("*.py"):
        shutil.copyfile(path, package / path.name)
    shutil.copyfile(extension, package / extension.name)
    save(package / "release-build.json", {"status": "LOCAL_PACKAGING_CANDIDATE", "private_abi": abi,
        "source_lock": prep["source_lock"], "production_promoted": False,
        "preparation_sha256": record(output / "preparation.json")["sha256"]})
    raw = pack(assembled, output / "runtime-raw" / base_wheel.name)
    destination = repair(raw, output / "runtime-repaired", policy,
        logs / "07-runtime-repair.json", env)
    final_site = output / "final-abi-site"
    final_site.mkdir()
    with zipfile.ZipFile(destination) as wheel:
        wheel.extractall(final_site)
    final_env = dict(env, PYTHONPATH=str(final_site) + os.pathsep + str(site))
    final_abi = subprocess.check_output([sys.executable, final_site / "plan7_gpu/_abi.py"],
                                      env=final_env, text=True).strip()
    if final_abi != abi:
        raise RuntimeError("repair changed private ABI: rebuild companions; do not bypass guard")
    run([sys.executable, "-c", "from plan7_gpu import _pipeline; print(_pipeline.PYHMMER_PRIVATE_ABI_SHA256)"],
        logs / "08-final-abi.json", final_env)
    shutil.copyfile(destination, dist / destination.name)
    if cuda_home:
        gpu = build_gpu(output, prep, final_site, abi, final_env, logs, cuda_home.resolve(), policy)
        shutil.copyfile(gpu, dist / gpu.name)
    run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist, output / "app"],
        logs / "07-astra.json", env)
    report = {"schema": "astra-pip-build-v1", "status": "BUILT_NOT_VALIDATED",
        "publish_allowed": False, "private_abi": abi, "preparation": record(output / "preparation.json"),
        "builder": record(logs / "builder.py"), "wheels": [record(p) for p in sorted(dist.glob("*.whl"))],
        "commands": [record(p) for p in sorted(logs.glob("*.json"))],
        "manylinux_policy": policy, "cuda_built": bool(cuda_home), "post_repair_abi_equal": final_abi == abi,
        "python": sys.version, "platform": platform.platform(),
        "remaining": ["Clean relocated installation and loader checks", "Installed artifact exactness",
            "manylinux build and post-repair ABI qualification", "GPU wheel", "Full request/RAM qualification",
            "Maintainer review before publication"]}
    save(output / "build-result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manylinux", choices=("manylinux_2_17_x86_64", "manylinux_2_28_x86_64", "manylinux_2_34_x86_64"))
    parser.add_argument("--cuda-home", type=Path)
    args = parser.parse_args()
    main(args.output, args.manylinux, args.cuda_home)
