"""GPU tests of the native bias-math attestation (skipped without a CUDA device).

They run the real exhaustive audit (every positive finite binary32 input, a
few seconds on an H200) and check that registration re-binds only a record of
the live libm build, CPU identity and device target.
"""
import copy
import os
import unittest

try:
    from plan7_gpu import _native
except ImportError:  # CPU-only install
    _native = None


def _cuda_available():
    if _native is None or not hasattr(_native, "bias_cuda_devices"):
        return False
    return bool(_native.bias_cuda_devices()["devices"])


@unittest.skipUnless(_cuda_available(), "needs plan7_gpu with a CUDA device")
class NativeAttestationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _native.bias_clear_attestation()
        cls.initial = _native.bias_environment_status()
        try:
            threads = len(os.sched_getaffinity(0))
        except AttributeError:
            threads = os.cpu_count() or 1
        cls.report = (
            _native.bias_self_attest(threads)
            if cls.initial["code"] in (1, -6) else None
        )

    def tearDown(self):
        _native.bias_clear_attestation()

    def require_audit(self):
        if self.report is None:
            self.skipTest(f"environment not admitted: {self.initial['reason']}")

    def test_status_names_the_failing_requirement(self):
        self.assertIn(self.initial["name"], {
            "attested-builtin", "not-attested", "cuda-device", "host-features",
            "float-environment", "libm-unavailable",
        })
        if self.initial["name"] == "not-attested":
            self.assertTrue(self.initial["self_attestation_applicable"])
            self.assertIn("libm GNU build ID", self.initial["reason"])

    def test_exhaustive_audit_covers_every_input(self):
        self.require_audit()
        report = self.report
        self.assertEqual(report["algorithm"], _native.BIAS_ATTESTATION_ALGORITHM)
        self.assertEqual(report["inputs_checked"], _native.BIAS_ATTESTATION_INPUT_COUNT)
        self.assertEqual(report["inputs_checked"], 2139095039)
        self.assertEqual(report["float_mismatches"], 0, report)
        self.assertTrue(report["attested"])
        self.assertEqual(
            report["libm_build_id"],
            _native.bias_environment_provenance()["libm"]["gnu_build_id"],
        )

    def register(self, **changes):
        record = {
            "libm_build_id": self.report["libm_build_id"],
            "host": copy.deepcopy(self.report["host"]),
            "cuda_target": self.report["cuda_target"],
            "algorithm_version": self.report["algorithm_version"],
            "inputs_checked": self.report["inputs_checked"],
            "float_mismatches": self.report["float_mismatches"],
        }
        record.update(changes)
        return _native.bias_register_attestation(
            record["libm_build_id"], record["host"], record["cuda_target"],
            record["algorithm_version"], record["inputs_checked"],
            record["float_mismatches"],
        )

    def test_registration_rebinds_only_the_live_identity(self):
        self.require_audit()
        if self.initial["name"] == "attested-builtin":
            self.skipTest("built-in environment: registration is not consulted")
        _native.bias_clear_attestation()
        self.assertFalse(_native.bias_environment_attested()[0])
        host = copy.deepcopy(self.report["host"])
        host["model"] += 1
        for changes, reason in (
            ({"libm_build_id": "00" * 20}, "different libm build"),
            ({"host": host}, "different CPU identity"),
            ({"cuda_target": 1 if self.report["cuda_target"] == 2 else 2}, "different CUDA target"),
            ({"float_mismatches": 1}, "zero-mismatch"),
            ({"inputs_checked": 1000}, "zero-mismatch"),
            ({"algorithm_version": 99}, "different audit algorithm"),
        ):
            with self.subTest(changes=list(changes)):
                registered, why = self.register(**changes)
                self.assertFalse(registered)
                self.assertIn(reason, why)
                self.assertFalse(_native.bias_environment_attested()[0])
        registered, why = self.register()
        self.assertTrue(registered, why)
        status = _native.bias_environment_status()
        self.assertEqual(status["name"], "attested-self")
        self.assertTrue(_native.bias_environment_attested()[0])

    def test_device_listing(self):
        listing = _native.bias_cuda_devices()
        self.assertGreaterEqual(len(listing["devices"]), 1)
        first = listing["devices"][0]
        self.assertEqual(first["device_ordinal"], 0)
        self.assertIsNotNone(first["identity"])


if __name__ == "__main__":
    unittest.main()
