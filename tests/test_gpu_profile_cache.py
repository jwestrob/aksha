import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from astra.gpu_profile_cache import (
    GPUProfileCacheBusyError,
    GPUProfileSessionCache,
    PROFILE_SEMANTICS,
    Plan7RuntimeIdentity,
)


RUNTIME = Plan7RuntimeIdentity(
    pyhmmer_version="test-pyhmmer",
    pyhmmer_private_abi_sha256="1" * 64,
    adapter_sha256="2" * 64,
    native_extension_sha256="3" * 64,
    pipeline_extension_sha256="4" * 64,
)


class FakeSession:
    def __init__(self, pairs, **options):
        self.pairs = tuple(pairs)
        self.options = options
        self.closed = False
        self.close_count = 0
        self.selected = []

    def __len__(self):
        return len(self.pairs)

    @property
    def statistics(self):
        if self.closed:
            raise RuntimeError("session is closed")
        return {"profile_count": len(self.pairs), "host_bytes": 123}

    def select(self, indices):
        if self.closed:
            raise RuntimeError("session is closed")
        result = tuple(indices)
        self.selected.append(result)
        return result

    def close(self):
        self.close_count += 1
        self.closed = True


class CacheFixture:
    def make_cache(self, base):
        state = {
            "manifest_sha256": "a" * 64,
            "stat_token": ("stable", 1),
            "model_count": 2,
        }
        pairs = (object(), object())
        sessions = []

        def validate(observed_base, manifest):
            self.assertEqual(observed_base, base.resolve())
            self.assertEqual(manifest, "manifest.json")
            return SimpleNamespace(
                canonical_base=base.resolve(),
                manifest_sha256=state["manifest_sha256"],
                stat_token=state["stat_token"],
                model_count=state["model_count"],
            )

        validator = mock.Mock(side_effect=validate)
        loader = mock.Mock(return_value=pairs)

        def build(observed_pairs, **options):
            session = FakeSession(observed_pairs, **options)
            sessions.append(session)
            return session

        factory = mock.Mock(side_effect=build)
        cache = GPUProfileSessionCache(
            runtime_identity=RUNTIME,
            validator=validator,
            loader=loader,
            session_factory=factory,
        )
        return cache, state, pairs, sessions, validator, loader, factory


class GPUProfileSessionCacheTests(CacheFixture, unittest.TestCase):
    def test_warm_acquire_reuses_pairs_and_session_until_owner_close(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, _, pairs, sessions, validator, loader, factory = (
                self.make_cache(base)
            )

            cold = cache.acquire(
                base,
                "manifest.json",
                device_key=("cuda-ordinal", 0),
                build_workers=16,
            )
            self.assertFalse(cold.reused)
            self.assertIs(cold.profile_pairs, pairs)
            self.assertEqual(cold.select([1, 0]), (1, 0))
            cold.close()
            self.assertFalse(sessions[0].closed)

            warm = cache.acquire(
                base,
                "manifest.json",
                device_key=("cuda-ordinal", 0),
                build_workers=16,
            )
            self.assertTrue(warm.reused)
            self.assertIs(warm.profile_pairs, pairs)
            self.assertIs(warm._entry.session, sessions[0])
            self.assertEqual(warm.profile_load_seconds, 0.0)
            self.assertEqual(warm.session_build_seconds, 0.0)
            warm.close()

            self.assertEqual(validator.call_count, 2)
            loader.assert_called_once_with(base.resolve(), manifest="manifest.json")
            factory.assert_called_once_with(
                pairs, build_workers=16, selection_workers=0
            )
            cache.close()
            self.assertTrue(sessions[0].closed)
            self.assertEqual(sessions[0].close_count, 1)
            self.assertFalse(cache.resident)

    def test_every_compatibility_key_component_invalidates_idle_entry(self):
        cases = (
            ("manifest", {"manifest_sha256": "b" * 64}, {}),
            ("pressed-stat", {"stat_token": ("changed", 2)}, {}),
            ("device", {}, {"device_key": ("cuda-ordinal", 1)}),
            ("build-workers", {}, {"build_workers": 8}),
            ("selection-workers", {}, {"selection_workers": 1}),
            ("profile-semantics", {}, {"profile_semantics": "future-v2"}),
        )
        for label, state_change, option_change in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix="astra-profile-cache-key-"
            ) as temporary:
                base = Path(temporary) / "PFAM"
                cache, state, _, sessions, _, loader, _ = self.make_cache(base)
                options = {
                    "device_key": ("cuda-ordinal", 0),
                    "build_workers": 16,
                    "selection_workers": 0,
                    "profile_semantics": PROFILE_SEMANTICS,
                }
                first = cache.acquire(base, "manifest.json", **options)
                first.close()
                state.update(state_change)
                options.update(option_change)
                second = cache.acquire(base, "manifest.json", **options)

                self.assertFalse(second.reused)
                self.assertTrue(sessions[0].closed)
                self.assertEqual(loader.call_count, 2)
                second.close()
                cache.close()

    def test_active_lease_excludes_overlap_and_owner_close_is_deferred(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, _, _, sessions, validator, _, _ = self.make_cache(base)
            lease = cache.acquire(
                base,
                "manifest.json",
                device_key=0,
                build_workers=4,
            )
            with self.assertRaisesRegex(
                GPUProfileCacheBusyError, "active search lease"
            ):
                cache.acquire(
                    base,
                    "manifest.json",
                    device_key=0,
                    build_workers=4,
                )
            self.assertEqual(validator.call_count, 1)

            cache.close()
            self.assertTrue(cache.closed)
            self.assertFalse(sessions[0].closed)
            lease.close()
            self.assertTrue(sessions[0].closed)
            self.assertFalse(cache.resident)
            with self.assertRaisesRegex(RuntimeError, "cache is closed"):
                cache.acquire(
                    base,
                    "manifest.json",
                    device_key=0,
                    build_workers=4,
                )

    def test_reservation_excludes_work_before_lease_and_can_be_cancelled(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, _, _, _, validator, _, _ = self.make_cache(base)
            reservation = cache.reserve()
            self.assertTrue(reservation.active)
            with self.assertRaisesRegex(
                GPUProfileCacheBusyError, "active search lease"
            ):
                cache.acquire(
                    base,
                    "manifest.json",
                    device_key=0,
                    build_workers=1,
                )
            validator.assert_not_called()
            reservation.close()
            self.assertFalse(reservation.active)

            lease = cache.acquire(
                base,
                "manifest.json",
                device_key=0,
                build_workers=1,
            )
            lease.close()
            cache.close()

    def test_reservation_is_consumed_only_after_successful_acquire(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, _, _, _, _, loader, _ = self.make_cache(base)
            reservation = cache.reserve()
            loader.side_effect = RuntimeError("load failed")
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                cache.acquire(
                    base,
                    "manifest.json",
                    device_key=0,
                    build_workers=1,
                    reservation=reservation,
                )
            self.assertTrue(reservation.active)
            reservation.close()

            loader.side_effect = None
            loader.return_value = (object(), object())
            reservation = cache.reserve()
            lease = cache.acquire(
                base,
                "manifest.json",
                device_key=0,
                build_workers=1,
                reservation=reservation,
            )
            self.assertFalse(reservation.active)
            lease.close()
            cache.close()

    def test_cache_close_cancels_an_unconsumed_reservation(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, *_ = self.make_cache(base)
            reservation = cache.reserve()
            cache.close()
            self.assertFalse(reservation.active)
            reservation.close()

    def test_released_lease_cannot_access_reused_session(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, *_ = self.make_cache(base)
            lease = cache.acquire(
                base,
                "manifest.json",
                device_key=0,
                build_workers=1,
            )
            lease.close()
            self.assertTrue(lease.closed)
            with self.assertRaisesRegex(RuntimeError, "lease is closed"):
                lease.select([0])
            cache.close()

    def test_externally_closed_idle_session_is_rebuilt(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, _, _, sessions, _, loader, _ = self.make_cache(base)
            first = cache.acquire(
                base,
                "manifest.json",
                device_key=0,
                build_workers=1,
            )
            first.close()
            sessions[0].close()

            second = cache.acquire(
                base,
                "manifest.json",
                device_key=0,
                build_workers=1,
            )
            self.assertFalse(second.reused)
            self.assertEqual(loader.call_count, 2)
            second.close()
            cache.close()

    def test_new_session_contract_failure_closes_it_and_caches_nothing(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            validation = SimpleNamespace(
                canonical_base=base.resolve(),
                manifest_sha256="a" * 64,
                stat_token=(1,),
                model_count=2,
            )
            broken = FakeSession((object(),))
            cache = GPUProfileSessionCache(
                runtime_identity=RUNTIME,
                validator=mock.Mock(return_value=validation),
                loader=mock.Mock(return_value=(object(), object())),
                session_factory=mock.Mock(return_value=broken),
            )
            with self.assertRaisesRegex(RuntimeError, "does not cover"):
                cache.acquire(
                    base,
                    "manifest.json",
                    device_key=0,
                    build_workers=1,
                )
            self.assertTrue(broken.closed)
            self.assertFalse(cache.resident)
            cache.close()

    def test_rejects_invalid_inputs_before_loading(self):
        with tempfile.TemporaryDirectory(prefix="astra-profile-cache-") as temporary:
            base = Path(temporary) / "PFAM"
            cache, _, _, _, validator, loader, factory = self.make_cache(base)
            for options, exception in (
                ({"device_key": 0, "build_workers": 0}, ValueError),
                ({"device_key": 0, "build_workers": True}, TypeError),
                (
                    {
                        "device_key": 0,
                        "build_workers": 1,
                        "selection_workers": -1,
                    },
                    ValueError,
                ),
                ({"device_key": [], "build_workers": 1}, TypeError),
            ):
                with self.subTest(options=options), self.assertRaises(exception):
                    cache.acquire(base, "manifest.json", **options)
            validator.assert_not_called()
            loader.assert_not_called()
            factory.assert_not_called()
            cache.close()


if __name__ == "__main__":
    unittest.main()
