"""CPU-only tests of GPU bias-math trust, its fallback route and `aksha gpu-check`.

The plan7_gpu CUDA extension is replaced by a small fake that implements the
attestation API: built-in attestation, self-attestation (live and cached),
registration checks against the live identity, and device listing.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from aksha import gpu_attestation, gpu_check, search
from tests.test_gpu_search import make_pressed_members, synthetic_plan7_gpu

ALGORITHM = "plan7-gpu-bias-log-exhaustive-v1"
INPUTS = 2139095039
HOST = {
    "vendor_words": [1970169159, 1231384169, 1818588270],
    "family": 6, "model": 143, "stepping": 8,
    "leaf1_ecx": 2147417087, "leaf1_edx": 3219913727,
    "leaf7_ebx": 4089427951, "xcr0_low": 393959, "xcr0_high": 0,
}
CUDA = {
    "device_ordinal": 0, "runtime_version": 12050, "driver_version": 13000,
    "cudart_version": 12050, "nvcc": {"major": 12, "minor": 5, "build": 82},
    "compute_capability": [9, 0], "multiprocessor_count": 132,
    "pci_domain_id": 0, "pci_bus_id": 25, "pci_device_id": 0,
    "total_global_memory": 150109880320,
    "uuid": "GPU-4d1f8223-38cd-6a1c-42c2-a6338ad6cb95",
    "pci_bus_address": "0000:19:00.0", "name": "NVIDIA H200",
}
NEW_LIBM = "d53fcbff9f855463606b25cfbe2a2e10856ea3a7"


class FakeNative:
    """Attestation API of plan7_gpu._native with a controllable environment."""

    BIAS_ATTESTATION_ALGORITHM = ALGORITHM
    BIAS_ATTESTATION_INPUT_COUNT = INPUTS

    def __init__(self, extension_file, *, state="not-attested", libm=NEW_LIBM,
                 audit_mismatches=0, audit_error=None, devices=1):
        self.__file__ = extension_file
        self.state = state
        self.libm = libm
        self.audit_mismatches = audit_mismatches
        self.audit_error = audit_error
        self.registered = None
        self.audit_calls = []
        self.register_calls = []
        self.device_total = devices

    # -- classification -------------------------------------------------
    def _code(self):
        if self.state == "built-in":
            return 1, "attested-builtin", ""
        if self.state == "cuda-device":
            return -3, "cuda-device", "bias math is not attested for this CUDA product/architecture"
        if self.registered == (self.libm, json.dumps(HOST, sort_keys=True), 2):
            return 2, "attested-self", ""
        return -6, "not-attested", (
            f"bias math is not attested for this host math library "
            f"(glibc 2.39, libm GNU build ID {self.libm})"
        )

    def bias_environment_attested(self):
        code, _, reason = self._code()
        return code > 0, reason

    def bias_environment_status(self):
        code, name, reason = self._code()
        return {
            "code": code, "name": name, "attested": code > 0,
            "source": {1: "built-in", 2: "self-attested"}.get(code),
            "self_attestation_applicable": code == -6, "reason": reason,
        }

    def bias_environment_provenance(self):
        code, _, reason = self._code()
        return {
            "attested": code > 0, "reason": reason, "target": "sm90_h200",
            "target_code": 2, "cuda": dict(CUDA), "cuda_identity_reason": "",
            "host": dict(HOST), "host_identity_reason": "",
            "libm": {"gnu_build_id": self.libm, "attested_for_target": False},
        }

    # -- audit and registration -------------------------------------------
    def bias_self_attest(self, host_threads):
        self.audit_calls.append(host_threads)
        if self.audit_error is not None:
            raise RuntimeError(self.audit_error)
        attested = self.audit_mismatches == 0
        if attested:
            self.registered = (self.libm, json.dumps(HOST, sort_keys=True), 2)
        return {
            "algorithm": ALGORITHM, "algorithm_version": 1, "attested": attested,
            "reason": "" if attested else "device and host log differ on at least one input",
            "cuda_target": 2, "target": "sm90_h200", "host_threads": host_threads,
            "inputs_checked": INPUTS, "float_mismatches": self.audit_mismatches,
            "double_mismatches": 215332, "maximum_float_ulp": 1 if self.audit_mismatches else 0,
            "maximum_double_ulp": 1,
            "first_float_mismatch": (
                {"input": "0x3f800001", "host": "0x33ffffff", "device": "0x34000000"}
                if self.audit_mismatches else None
            ),
            "first_double_mismatch_input": "0x0000bfb2", "seconds": 1.9,
            "libm_build_id": self.libm, "host": dict(HOST), "cuda": dict(CUDA),
        }

    def bias_register_attestation(self, build_id, host, target, algorithm,
                                  inputs, mismatches):
        self.register_calls.append((build_id, target, algorithm, inputs, mismatches))
        if algorithm != 1:
            return False, "bias attestation record uses a different audit algorithm"
        if inputs != INPUTS or mismatches != 0:
            return False, "bias attestation record is not a complete zero-mismatch audit"
        if build_id != self.libm:
            return False, "bias attestation record is for a different libm build"
        if host != HOST:
            return False, "bias attestation record is for a different CPU identity"
        if target != 2:
            return False, "bias attestation record is for a different CUDA target"
        self.registered = (build_id, json.dumps(host, sort_keys=True), target)
        return True, ""

    def bias_clear_attestation(self):
        self.registered = None

    def bias_cuda_devices(self):
        devices = []
        for ordinal in range(self.device_total):
            identity = dict(CUDA, device_ordinal=ordinal)
            devices.append({"device_ordinal": ordinal, "identity": identity,
                            "target_code": 2, "target": "sm90_h200", "reason": ""})
        return {"devices": devices, "reason": "" if devices else "no CUDA-capable device is detected"}


class AttestationTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aksha-attest-")
        self.root = Path(self.temporary.name)
        self.extension = self.root / "_native.so"
        self.extension.write_bytes(b"fake extension v1")
        self.environ = {gpu_attestation.CACHE_DIR_ENV: os.fspath(self.root / "cache")}
        self.addCleanup(self.temporary.cleanup)

    def native(self, **kwargs):
        native = FakeNative(os.fspath(self.extension), **kwargs)
        patcher = mock.patch.object(gpu_attestation, "native_module", return_value=native)
        patcher.start()
        self.addCleanup(patcher.stop)
        return native

    def evaluate(self, attest="auto", threads=4):
        lines = []
        trust = gpu_attestation.evaluate_bias_trust(
            attest=attest, threads=threads, announce=lines.append, environ=self.environ,
        )
        return trust, lines


class ModeAndCacheTests(unittest.TestCase):
    def test_self_attest_mode_values(self):
        self.assertEqual(gpu_attestation.self_attest_mode({}), "auto")
        for raw, mode in (("never", "never"), ("0", "never"), ("OFF", "never"),
                          ("force", "force"), ("auto", "auto"), ("1", "auto")):
            self.assertEqual(
                gpu_attestation.self_attest_mode({gpu_attestation.SELF_ATTEST_ENV: raw}), mode
            )
        with self.assertRaises(ValueError):
            gpu_attestation.self_attest_mode({gpu_attestation.SELF_ATTEST_ENV: "sometimes"})

    def test_cache_directory_precedence(self):
        self.assertEqual(
            gpu_attestation.cache_directory({gpu_attestation.CACHE_DIR_ENV: "/x/y"}), Path("/x/y")
        )
        self.assertEqual(
            gpu_attestation.cache_directory({"XDG_CACHE_HOME": "/c"}),
            Path("/c/aksha/gpu-attestation"),
        )


class TrustTests(AttestationTestCase):
    def test_no_extension_leaves_routing_unchanged(self):
        with mock.patch.object(gpu_attestation, "native_module", return_value=None):
            trust, lines = self.evaluate()
        self.assertEqual(trust.state, "unavailable")
        self.assertIsNone(trust.trusted)
        self.assertEqual(trust.search_line(), "")
        self.assertEqual(lines, [])

    def test_extension_without_self_attestation(self):
        native = SimpleNamespace(bias_environment_attested=lambda: (False, "not attested for this glibc version"))
        with mock.patch.object(gpu_attestation, "native_module", return_value=native):
            trust, _ = self.evaluate()
        self.assertFalse(trust.trusted)
        self.assertIn("cannot self-attest", trust.hint)
        native = SimpleNamespace(bias_environment_attested=lambda: (True, ""))
        with mock.patch.object(gpu_attestation, "native_module", return_value=native):
            trust, _ = self.evaluate()
        self.assertTrue(trust.trusted)

    def test_builtin_environment_needs_no_audit(self):
        native = self.native(state="built-in")
        trust, lines = self.evaluate()
        self.assertTrue(trust.trusted)
        self.assertEqual(trust.source, "built-in attestation")
        self.assertEqual(native.audit_calls, [])
        self.assertEqual(lines, [])

    def test_inadmissible_device_is_untrusted_without_audit(self):
        native = self.native(state="cuda-device")
        trust, _ = self.evaluate()
        self.assertFalse(trust.trusted)
        self.assertEqual(native.audit_calls, [])
        self.assertIn("admitted device", trust.hint)

    def test_first_use_audits_caches_then_reuses(self):
        native = self.native()
        trust, lines = self.evaluate(threads=7)
        self.assertTrue(trust.trusted)
        self.assertEqual(native.audit_calls, [min(7, os.cpu_count() or 1)])
        self.assertEqual(len(lines), 1)
        self.assertIn("one-time exhaustive self-attestation", lines[0])
        record = json.loads(Path(trust.record_path).read_text())
        self.assertEqual(record["schema"], gpu_attestation.RECORD_SCHEMA)
        self.assertEqual(record["key"]["libm_build_id"], NEW_LIBM)
        self.assertEqual(record["key"]["algorithm"], ALGORITHM)
        self.assertEqual(record["audit"]["float_mismatches"], 0)

        # A new process: nothing registered, the cached record is re-bound.
        native.bias_clear_attestation()
        trust, lines = self.evaluate()
        self.assertTrue(trust.trusted)
        self.assertIn("cached at", trust.source)
        self.assertEqual(len(native.audit_calls), 1)
        self.assertEqual(native.register_calls[-1], (NEW_LIBM, 2, 1, INPUTS, 0))
        self.assertEqual(lines, [])

    def test_os_update_selects_a_new_record_and_audit(self):
        native = self.native()
        first, _ = self.evaluate()
        native.bias_clear_attestation()
        native.libm = "0" * 40  # the next glibc package
        second, lines = self.evaluate()
        self.assertTrue(second.trusted)
        self.assertNotEqual(first.record_path, second.record_path)
        self.assertEqual(len(native.audit_calls), 2)
        self.assertEqual(len(lines), 1)

    def test_rebuilt_extension_selects_a_new_record(self):
        native = self.native()
        first, _ = self.evaluate()
        native.bias_clear_attestation()
        self.extension.write_bytes(b"fake extension v2")
        second, _ = self.evaluate()
        self.assertNotEqual(first.record_path, second.record_path)
        self.assertEqual(len(native.audit_calls), 2)

    def test_tampered_record_is_never_accepted(self):
        native = self.native()
        trust, _ = self.evaluate()
        native.bias_clear_attestation()
        path = Path(trust.record_path)
        record = json.loads(path.read_text())
        record["audit"]["libm_build_id"] = "f" * 40  # key still matches, identity does not
        path.write_text(json.dumps(record))
        report, _ = self.evaluate(attest="report")
        self.assertEqual(report.state, "pending")
        self.assertIn("different libm build", report.reason)
        record["key"]["glibc"] = "glibc 9.99"  # key mismatch: ignored entirely
        path.write_text(json.dumps(record))
        report, _ = self.evaluate(attest="report")
        self.assertEqual(report.state, "pending")
        self.assertEqual(len(native.audit_calls), 1)

    def test_failed_audit_is_untrusted_cached_and_forced_rerun(self):
        native = self.native(audit_mismatches=3)
        trust, _ = self.evaluate()
        self.assertFalse(trust.trusted)
        self.assertIn("3 binary32 mismatches", trust.reason)
        self.assertIn("first at input 0x3f800001", trust.reason)
        again, _ = self.evaluate()
        self.assertFalse(again.trusted)
        self.assertEqual(len(native.audit_calls), 1)  # the failure is cached
        native.audit_mismatches = 0
        forced, _ = self.evaluate(attest="force")
        self.assertTrue(forced.trusted)
        self.assertEqual(len(native.audit_calls), 2)

    def test_never_and_report_modes_do_not_audit(self):
        native = self.native()
        never, _ = self.evaluate(attest="never")
        self.assertFalse(never.trusted)
        self.assertIn("AKSHA_GPU_SELF_ATTEST=never", never.reason)
        report, _ = self.evaluate(attest="report")
        self.assertEqual(report.state, "pending")
        self.assertIsNone(report.trusted)
        self.assertEqual(native.audit_calls, [])

    def test_audit_that_cannot_run_falls_back(self):
        self.native(audit_error="bias self-attestation cudaMalloc: out of memory")
        trust, _ = self.evaluate()
        self.assertFalse(trust.trusted)
        self.assertIn("could not run", trust.reason)

    def test_untrusted_search_line_is_one_warning_line(self):
        self.native()
        trust, _ = self.evaluate(attest="never")
        line = trust.search_line()
        self.assertTrue(line.startswith("WARNING: GPU bias math is untrusted"))
        self.assertNotIn("\n", line)
        self.assertIn("libm GNU build ID", line)
        self.assertIn("exact CPU path", line)
        self.assertIn("aksha gpu-check --attest", line)


def single_database_config(root):
    db_dir = Path(tempfile.mkdtemp(prefix="db-", dir=root)) / "GPUDB"
    db_dir.mkdir()
    pressed_base = make_pressed_members(db_dir, "profiles")
    config = {"db_urls": [{
        "name": "GPUDB", "installed": True,
        "installation_dir": os.fspath(db_dir), "molecule_type": "protein",
    }]}
    return config, pressed_base


class FallbackRouteTests(unittest.TestCase):
    def preflight(self, bias_trusted, environ=None):
        with tempfile.TemporaryDirectory(prefix="aksha-route-") as temporary:
            config, _ = single_database_config(Path(temporary))
            modules, api = synthetic_plan7_gpu(True, True)
            pair = object()
            batch = mock.Mock(name="sequence_batch")
            batch.memory_snapshot = {"device_ordinal": 0}
            api.load_pressed_profiles.return_value = (pair,)
            api.SequenceBatch.return_value = batch
            with mock.patch.dict(sys.modules, modules), \
                    mock.patch.dict(os.environ, environ or {}, clear=False):
                databases, _, postfilter = search.preflight_gpu_databases(
                    {"GPUDB": "manifest.json"}, ["GPUDB"], config, [object()], 1,
                    bias_trusted=bias_trusted,
                )
            return databases, postfilter, api

    def test_untrusted_bias_math_keeps_only_gpu_ssv(self):
        databases, postfilter, api = self.preflight(False)
        self.assertFalse(postfilter)
        self.assertIsNone(databases["GPUDB"][2])
        api.ProfileSession.assert_not_called()

    def test_trusted_and_unevaluated_keep_the_exact_gpu_route(self):
        for trusted in (True, None):
            with self.subTest(trusted=trusted):
                databases, postfilter, api = self.preflight(trusted)
                self.assertTrue(postfilter)
                api.ProfileSession.assert_called_once()

    def test_native_fallback_audit_control_keeps_session_route(self):
        databases, postfilter, api = self.preflight(
            False, {search.GPU_UNTRUSTED_NATIVE_FALLBACK_ENV: "1"}
        )
        self.assertTrue(postfilter)
        api.ProfileSession.assert_called_once()


class SearchStartTests(unittest.TestCase):
    def test_search_prints_one_warning_and_routes_to_ssv(self):
        class Stop(Exception):
            pass

        captured = {}

        def fake_preflight(*args, **kwargs):
            captured.update(kwargs)
            raise Stop

        untrusted = gpu_attestation.BiasTrust(
            "untrusted", None,
            "bias math is not attested for this host math library (glibc 2.39, "
            "libm GNU build ID d53f...)",
        )
        with tempfile.TemporaryDirectory(prefix="aksha-start-") as temporary:
            root = Path(temporary)
            config, _ = single_database_config(root)
            args = SimpleNamespace(
                hmm_in=None, prot_in="proteins.faa", outdir=os.fspath(root / "out"),
                installed_hmms="GPUDB", gpu_manifest=["GPUDB=manifest.json"],
                synteny=None, evalue=None, bitscore=None, incE=None, incT=None,
                domE=None, domT=None, incdomE=None, incdomT=None, cut_ga=True,
                cut_nc=False, cut_tc=False, cascade=False, meta=False,
                individual_results=False, write_seqs=False, threads=4,
                write_macsyfinder=False, cpu_parallel="auto",
            )
            setattr(args, "16rp", False)
            setattr(args, "15rp", False)
            output = io.StringIO()
            with mock.patch.object(search.initialize, "load_config", return_value=config), \
                    mock.patch.object(search, "parse_protein_input", return_value={"p": [object()]}), \
                    mock.patch.object(search, "gpu_hmm_chunk_size", return_value=1), \
                    mock.patch.object(gpu_attestation, "evaluate_bias_trust", return_value=untrusted) as evaluate, \
                    mock.patch.object(search, "preflight_gpu_databases", side_effect=fake_preflight), \
                    redirect_stdout(output):
                with self.assertRaises(Stop):
                    search.main(args)
        evaluate.assert_called_once_with(attest="auto", threads=4)
        self.assertIs(captured["bias_trusted"], False)
        warnings = [line for line in output.getvalue().splitlines() if line.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("GPU runs only the SSV filter", warnings[0])


class ReadinessCheckTests(AttestationTestCase):
    def readiness(self, native_kwargs, manifests=("GPUDB",), attest=False, valid=True):
        native = self.native(**native_kwargs)
        config, pressed_base = single_database_config(self.root)
        modules, api = synthetic_plan7_gpu(True, True)
        if valid:
            api.validate_pressed_manifest.return_value = SimpleNamespace(model_count=3)
        else:
            api.validate_pressed_manifest.side_effect = ValueError("h3m sha256 differs")
        mappings = {name: "manifest.json" for name in manifests}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, self.environ, clear=False), \
                mock.patch("aksha.initialize.load_config", return_value=config):
            document = gpu_check.readiness(mappings, attest=attest, threads=2, announce=lambda _: None)
        return document, native

    def test_ready_when_trusted(self):
        document, _ = self.readiness({"state": "built-in", "devices": 2})
        self.assertEqual(document["overall"], gpu_check.READY)
        verdicts = {v["device_ordinal"]: v for v in document["verdicts"]}
        self.assertEqual(verdicts[0]["route"], "GPU")
        self.assertEqual(verdicts[0]["why"], "exact GPU-through-Forward")
        self.assertEqual(verdicts[1]["route"], "not used")
        text = gpu_check.format_report(document)
        self.assertIn("GPUDB on GPU 0: GPU -- exact GPU-through-Forward", text)
        self.assertIn("overall: READY", text)

    def test_pending_then_attest(self):
        document, native = self.readiness({})
        self.assertEqual(document["overall"], gpu_check.PENDING)
        self.assertEqual(document["verdicts"][0]["route"], "GPU after self-attestation")
        self.assertEqual(native.audit_calls, [])
        document, native = self.readiness({}, attest=True)
        self.assertEqual(document["overall"], gpu_check.READY)
        self.assertEqual(len(native.audit_calls), 1)
        self.assertEqual(document["bias_math"]["audit"]["float_mismatches"], 0)

    def test_disabled_self_attestation_is_reported_as_fallback(self):
        with mock.patch.dict(os.environ, {gpu_attestation.SELF_ATTEST_ENV: "never"}):
            document, native = self.readiness({})
        self.assertEqual(document["overall"], gpu_check.FALLBACK)
        verdict = document["verdicts"][0]
        self.assertEqual(verdict["route"], "SSV only")
        self.assertIn("AKSHA_GPU_SELF_ATTEST=never", verdict["why"])
        self.assertEqual(native.audit_calls, [])

    def test_fallback_reports_reason_and_restore_hint(self):
        with mock.patch.dict(os.environ, {gpu_attestation.SELF_ATTEST_ENV: "never"}):
            document, _ = self.readiness({"audit_mismatches": 2}, attest=True)
        self.assertEqual(document["overall"], gpu_check.FALLBACK)
        verdict = document["verdicts"][0]
        self.assertEqual(verdict["route"], "SSV only")
        self.assertIn("2 binary32 mismatches", verdict["why"])

    def test_invalid_manifest_and_missing_device_are_refusals(self):
        document, _ = self.readiness({"state": "built-in"}, valid=False)
        self.assertEqual(document["overall"], gpu_check.FALLBACK)
        self.assertEqual(document["verdicts"][0]["route"], "refused")
        self.assertIn("h3m sha256 differs", document["verdicts"][0]["why"])
        document, _ = self.readiness({"state": "built-in", "devices": 0})
        self.assertEqual(document["verdicts"][0]["route"], "refused")
        self.assertIn("no CUDA device visible", document["verdicts"][0]["why"])

    def test_exit_status_and_json(self):
        native = self.native(state="built-in")
        config, _ = single_database_config(self.root)
        modules, api = synthetic_plan7_gpu(True, True)
        api.validate_pressed_manifest.return_value = SimpleNamespace(model_count=3)
        args = SimpleNamespace(gpu_manifest=["GPUDB=manifest.json"], attest=False,
                               threads=1, json=True)
        output = io.StringIO()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, self.environ, clear=False), \
                mock.patch("aksha.initialize.load_config", return_value=config), \
                redirect_stdout(output):
            status = gpu_check.main(args)
        self.assertEqual(status, 0)
        document = json.loads(output.getvalue())
        self.assertEqual(document["overall"], "READY")
        self.assertEqual(document["bias_math"]["state"], "trusted")


if __name__ == "__main__":
    unittest.main()
