import argparse
import builtins
import gc
import importlib
import inspect
import io
import math
import os
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from queue import Empty, Full
from types import ModuleType, SimpleNamespace
from unittest import mock

import pyhmmer

from astra import search
from astra.gpu_profile_cache import (
    GPUProfileCacheBusyError,
    GPUProfileSessionCache,
    Plan7RuntimeIdentity,
)

astra_main = importlib.import_module("astra.main")


def search_options(outdir, **overrides):
    options = {
        "cascade": False,
        "cut_ga": False,
        "cut_nc": False,
        "cut_tc": False,
        "evalue": None,
        "bitscore": None,
        "domE": None,
        "domT": None,
        "incE": None,
        "incT": None,
        "incdomE": None,
        "incdomT": None,
        "outdir": os.fspath(outdir),
        "meta": False,
    }
    options.update(overrides)
    return options


def search_args(outdir, **overrides):
    values = {
        "hmm_in": None,
        "prot_in": "proteins.faa",
        "outdir": os.fspath(outdir),
        "installed_hmms": None,
        "gpu_manifest": [],
        "16rp": False,
        "15rp": False,
        "synteny": None,
        "evalue": None,
        "bitscore": None,
        "incE": None,
        "incT": None,
        "domE": None,
        "domT": None,
        "incdomE": None,
        "incdomT": None,
        "cut_ga": False,
        "cut_nc": False,
        "cut_tc": False,
        "cascade": False,
        "meta": False,
        "individual_results": False,
        "write_seqs": False,
        "threads": 1,
        "write_macsyfinder": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_pressed_members(directory, base_name):
    base = directory / base_name
    for suffix in search.PRESSED_SUFFIXES:
        Path(f"{base}.{suffix}").touch()
    return base


def synthetic_plan7_gpu(postfilter_available=None, forward_available=None,
                        simple_available=None, domain_adapter_available=True,
                        domain_method_available=None,
                        domain_native_available=None,
                        compact_adapter_available=False,
                        compact_native_available=False,
                        compact_seam_available=None,
                        compact_tail_available=None):
    """Return optional-package modules suitable for CPU-only wiring tests."""
    package = ModuleType("plan7_gpu")
    package.__path__ = []
    astra_search_module = ModuleType("plan7_gpu.astra_search")
    manifest_module = ModuleType("plan7_gpu.pressed_manifest")
    pipeline_module = ModuleType("plan7_gpu._pipeline")
    native_module = ModuleType("plan7_gpu._native")

    if domain_method_available is None:
        domain_method_available = domain_adapter_available
    if domain_native_available is None:
        domain_native_available = domain_adapter_available

    class LegacySequenceBatch:
        def _postfilter_forward_selection(
            self, selection, F1, F2, F3, bias_filter, *,
            pipeline=None, domain_guard=2.0e-4,
        ):
            pass

    class DomainSequenceBatch(LegacySequenceBatch):
        def _postfilter_forward_domain_selection(
            self, selection, F1, f2, f3, bias_filter,
            pipeline, domain_guard,
        ):
            pass

    class CompactSequenceBatch:
        def _postfilter_forward_selection(
            self, selection, F1, F2, F3, bias_filter, *,
            pipeline=None, domain_guard=2.0e-4,
            _rescore_compact_byte_budget=0,
            _rescore_matrix_byte_budget=0,
            _rescore_trace_byte_budget=0,
            _rescore_test_fault=0,
        ):
            pass

    class CompactDomainSequenceBatch(CompactSequenceBatch):
        def _postfilter_forward_domain_selection(
            self, selection, F1, f2, f3, bias_filter,
            pipeline, domain_guard, rescore_compact_byte_budget,
            rescore_matrix_byte_budget, rescore_trace_byte_budget,
            rescore_test_fault,
        ):
            pass

    class LegacyNativeSequenceBatch:
        def _postfilter_forward_domain_selection_sealed(
            self, selection, f1, f2, f3, guard_band=2.0e-4,
            gathered_byte_budget=0,
        ):
            pass

    class CompactNativeSequenceBatch:
        def _postfilter_forward_domain_selection_sealed(
            self, selection, f1, f2, f3, guard_band=2.0e-4,
            gathered_byte_budget=0, rescore_simple_diagnostic=False,
            rescore_matrix_byte_budget=0, rescore_trace_byte_budget=0,
            rescore_compact_byte_budget=0, _rescore_test_fault=0,
            generation_tail_fingerprint=0,
        ):
            pass

    class NoDomainNativeSequenceBatch:
        pass

    if compact_adapter_available:
        sequence_batch_spec = (
            CompactDomainSequenceBatch
            if domain_method_available
            else CompactSequenceBatch
        )
    else:
        sequence_batch_spec = (
            DomainSequenceBatch
            if domain_method_available
            else LegacySequenceBatch
        )

    default_batch = mock.Mock(name="sequence_batch")
    default_batch.memory_snapshot = {"device_ordinal": 0}
    api = SimpleNamespace(
        ProfileSession=mock.Mock(name="ProfileSession"),
        SequenceBatch=mock.Mock(
            name="SequenceBatch",
            spec=sequence_batch_spec,
            return_value=default_batch,
        ),
        load_pressed_profiles=mock.Mock(name="load_pressed_profiles"),
        validate_pressed_manifest=mock.Mock(name="validate_pressed_manifest"),
        gpu_hmmsearch=mock.Mock(name="gpu_hmmsearch"),
        filter_scores_seam_available=None,
        forward_scores_seam_available=None,
        simple_regions_seam_available=None,
        compact_domains_seam_available=None,
        compact_tail_fingerprint=None,
        seal_profile_selection_continuation=None,
    )
    api.SequenceBatch._postfilter_forward_selection = (
        sequence_batch_spec._postfilter_forward_selection
    )
    if domain_method_available:
        api.SequenceBatch._postfilter_forward_domain_selection = (
            sequence_batch_spec._postfilter_forward_domain_selection
        )
    if postfilter_available is not None:
        api.filter_scores_seam_available = mock.Mock(
            name="filter_scores_seam_available",
            return_value=postfilter_available,
        )
        pipeline_module._filter_scores_seam_available = (
            api.filter_scores_seam_available
        )
    if forward_available is None:
        forward_available = postfilter_available
    if forward_available is not None:
        api.forward_scores_seam_available = mock.Mock(
            name="forward_scores_seam_available",
            return_value=forward_available,
        )
        pipeline_module._filter_and_forward_scores_seam_available = (
            api.forward_scores_seam_available
        )
    if simple_available is not None:
        api.simple_regions_seam_available = mock.Mock(
            name="simple_regions_seam_available",
            return_value=simple_available,
        )
        pipeline_module._simple_regions_seam_available = (
            api.simple_regions_seam_available
        )
        if domain_adapter_available:
            api.seal_profile_selection_continuation = mock.Mock(
                name="seal_profile_selection_continuation"
            )
            pipeline_module._seal_profile_selection_continuation_bound = (
                api.seal_profile_selection_continuation
            )
    if compact_seam_available is not None:
        api.compact_domains_seam_available = mock.Mock(
            name="compact_domains_seam_available",
            return_value=compact_seam_available,
        )
        pipeline_module._compact_domains_seam_available = (
            api.compact_domains_seam_available
        )
    if compact_tail_available is None:
        compact_tail_available = compact_seam_available is not None
    if compact_tail_available:
        api.compact_tail_fingerprint = mock.Mock(
            name="compact_tail_fingerprint"
        )
        pipeline_module._compact_tail_fingerprint_bound = (
            api.compact_tail_fingerprint
        )
    package.SequenceBatch = api.SequenceBatch
    package.ProfileSession = api.ProfileSession
    package.load_pressed_profiles = api.load_pressed_profiles
    package.astra_search = astra_search_module
    package._native = native_module
    package._pipeline = pipeline_module
    if not domain_native_available:
        native_module.SequenceBatch = NoDomainNativeSequenceBatch
    elif compact_native_available:
        native_module.SequenceBatch = CompactNativeSequenceBatch
    else:
        native_module.SequenceBatch = LegacyNativeSequenceBatch
    astra_search_module.hmmsearch = api.gpu_hmmsearch
    manifest_module.validate_pressed_manifest = api.validate_pressed_manifest
    return (
        {
            "plan7_gpu": package,
            "plan7_gpu.astra_search": astra_search_module,
            "plan7_gpu.pressed_manifest": manifest_module,
            "plan7_gpu._native": native_module,
            "plan7_gpu._pipeline": pipeline_module,
        },
        api,
    )


class GPUConfigurationTests(unittest.TestCase):
    def test_search_cli_collects_repeatable_gpu_manifest_values(self):
        argv = [
            "astra",
            "search",
            "--prot_in",
            "proteins.faa",
            "--outdir",
            "out",
            "--installed_hmms",
            "HydDB,PFAM",
            "--gpu-manifest",
            "HydDB=hyd.json",
            "--gpu-manifest",
            "PFAM=pfam.json",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(astra_main.search, "main") as run_search,
        ):
            astra_main.main()
        run_search.assert_called_once()
        self.assertEqual(
            run_search.call_args.args[0].gpu_manifest,
            ["HydDB=hyd.json", "PFAM=pfam.json"],
        )

    def test_mapping_parser_accepts_repeatable_values_and_path_equals(self):
        mappings = search.parse_gpu_manifest_mappings(
            [
                "HydDB=~/manifests/hyddb.json",
                "PFAM=/tmp/name=release.json",
            ]
        )
        self.assertEqual(
            mappings,
            {
                "HydDB": os.path.expanduser("~/manifests/hyddb.json"),
                "PFAM": "/tmp/name=release.json",
            },
        )

    def test_mapping_parser_rejects_malformed_and_duplicate_values(self):
        for value in ("HydDB", "=manifest.json", "HydDB=", "  =  "):
            with self.subTest(value=value):
                with self.assertRaisesRegex(search.GPUConfigurationError, "malformed"):
                    search.parse_gpu_manifest_mappings([value])
        with self.assertRaisesRegex(search.GPUConfigurationError, "duplicate"):
            search.parse_gpu_manifest_mappings(["HydDB=one", "HydDB=two"])

    def test_configuration_rejects_nonpositive_threads_macsyfinder_and_unused(self):
        config = {
            "db_urls": [
                {
                    "name": "HydDB",
                    "installed": True,
                    "installation_dir": "/db/HydDB",
                    "molecule_type": "protein",
                },
                {
                    "name": "NotInstalled",
                    "installed": False,
                    "installation_dir": "/db/NotInstalled",
                    "molecule_type": "protein",
                },
            ]
        }
        search.validate_gpu_configuration(
            {"HydDB": "manifest"}, ["HydDB"], config, 1, False
        )
        for threads in (0, -1, True):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(
                    search.GPUConfigurationError, "positive integer"
                ):
                    search.validate_gpu_configuration(
                        {"HydDB": "manifest"}, ["HydDB"], config, threads, False
                    )
        with self.assertRaisesRegex(search.GPUConfigurationError, "write_macsyfinder"):
            search.validate_gpu_configuration(
                {"HydDB": "manifest"}, ["HydDB"], config, 1, True
            )
        for name in ("Unknown", "NotInstalled"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(search.GPUConfigurationError, "unused"):
                    search.validate_gpu_configuration(
                        {name: "manifest"}, [name], config, 1, False
                    )
        with self.assertRaisesRegex(search.GPUConfigurationError, "duplicate"):
            search.validate_gpu_configuration(
                {"HydDB": "manifest"}, ["HydDB", "HydDB"], config, 1, False
            )

    def test_pressed_base_accepts_conventional_or_unique_source_named_set(self):
        with tempfile.TemporaryDirectory(prefix="astra-pressed-base-") as temporary:
            root = Path(temporary)
            conventional_dir = root / "PFAM"
            conventional_dir.mkdir()
            conventional = make_pressed_members(conventional_dir, "PFAM")
            self.assertEqual(
                search.discover_pressed_base(conventional_dir), conventional.resolve()
            )

            hyd_dir = root / "HydDB"
            hyd_dir.mkdir()
            hyd_base = make_pressed_members(hyd_dir, "HydDB_all_MM2022.hmm")
            self.assertEqual(search.discover_pressed_base(hyd_dir), hyd_base.resolve())

    def test_pressed_base_rejects_missing_and_ambiguous_sets(self):
        with tempfile.TemporaryDirectory(prefix="astra-pressed-errors-") as temporary:
            root = Path(temporary)
            missing = root / "missing"
            missing.mkdir()
            (missing / "partial.h3m").touch()
            with self.assertRaisesRegex(search.GPUConfigurationError, "no complete"):
                search.discover_pressed_base(missing)

            ambiguous = root / "ambiguous"
            ambiguous.mkdir()
            make_pressed_members(ambiguous, "ambiguous")
            make_pressed_members(ambiguous, "another.hmm")
            with self.assertRaisesRegex(search.GPUConfigurationError, "ambiguous"):
                search.discover_pressed_base(ambiguous)

    def test_gpu_chunk_sizing_boundaries(self):
        self.assertEqual(search.gpu_hmm_chunk_size(0), 2000)
        self.assertEqual(search.gpu_hmm_chunk_size(50_000), 2000)
        self.assertEqual(search.gpu_hmm_chunk_size(50_001), 1999)
        self.assertEqual(search.gpu_hmm_chunk_size(1_800_000), 55)
        self.assertEqual(search.gpu_hmm_chunk_size(100_000_000), 1)
        with self.assertRaisesRegex(search.GPUConfigurationError, "at most"):
            search.gpu_hmm_chunk_size(100_000_001)

    def test_profile_worker_allocation_reserves_one_control_slot(self):
        self.assertEqual(
            search.gpu_profile_worker_allocation(1, True),
            (False, 0, 1),
        )
        self.assertEqual(
            search.gpu_profile_worker_allocation(2, True),
            (True, 1, 1),
        )
        self.assertEqual(
            search.gpu_profile_worker_allocation(3, False),
            (False, 1, 2),
        )
        for threads in (0, -1, True):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(
                    search.GPUConfigurationError, "positive integer"
                ):
                    search.gpu_profile_worker_allocation(threads, True)
        with self.assertRaisesRegex(TypeError, "overlap_requested"):
            search.gpu_profile_worker_allocation(2, 1)

    def test_legacy_scheduler_selector_is_exact_and_overlap_only(self):
        environment = search.GPU_LEGACY_OVERLAP_ENV
        previous = os.environ.pop(environment, None)
        try:
            self.assertEqual(
                search.gpu_profile_scheduler_mode(None, False), "serial"
            )
            self.assertEqual(
                search.gpu_profile_scheduler_mode(object(), True),
                "bounded-ready-queue",
            )
            with mock.patch.dict(os.environ, {environment: "1"}):
                with self.assertRaisesRegex(
                    search.GPUConfigurationError, "accepts only"
                ):
                    search.gpu_profile_scheduler_mode(object(), True)
            with mock.patch.dict(
                os.environ,
                {environment: search.GPU_LEGACY_OVERLAP_VALUE},
            ):
                self.assertEqual(
                    search.gpu_profile_scheduler_mode(object(), True),
                    "single-prefetch",
                )
                for session, overlap in ((None, False), (object(), False)):
                    with self.subTest(session=session, overlap=overlap):
                        with self.assertRaisesRegex(
                            search.GPUConfigurationError,
                            "requires an active overlapping",
                        ):
                            search.gpu_profile_scheduler_mode(session, overlap)
        finally:
            if previous is not None:
                os.environ[environment] = previous
            else:
                os.environ.pop(environment, None)

    def test_ready_queue_configuration_is_exact_and_fail_closed(self):
        depth_environment = search.GPU_READY_QUEUE_DEPTH_ENV
        byte_environment = search.GPU_READY_QUEUE_BYTES_ENV
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                search.gpu_ready_queue_configuration(), (1, None)
            )

        accepted = {
            ("1", None): (1, None),
            (None, "4096"): (1, 4096),
            ("1", "4096"): (1, 4096),
            ("2", "4096"): (2, 4096),
            ("4", "4096"): (4, 4096),
        }
        for (depth, byte_count), expected in accepted.items():
            environment = {}
            if depth is not None:
                environment[depth_environment] = depth
            if byte_count is not None:
                environment[byte_environment] = byte_count
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(
                        search.gpu_ready_queue_configuration(), expected
                    )

        for depth in ("", "0", "01", "3", "8", "+2", " 2", "2 "):
            with self.subTest(depth=depth):
                with mock.patch.dict(
                    os.environ, {depth_environment: depth}, clear=True
                ):
                    with self.assertRaisesRegex(
                        search.GPUConfigurationError, "accepts only"
                    ):
                        search.gpu_ready_queue_configuration()

        for byte_count in (
            "", "0", "00", "01", "-1", "+1", " 1", "1 ", "1_000", "١",
        ):
            with self.subTest(byte_count=byte_count):
                with mock.patch.dict(
                    os.environ, {byte_environment: byte_count}, clear=True
                ):
                    with self.assertRaisesRegex(
                        search.GPUConfigurationError,
                        "canonical positive decimal",
                    ):
                        search.gpu_ready_queue_configuration()

        with mock.patch.dict(
            os.environ, {depth_environment: "2"}, clear=True
        ):
            with self.assertRaisesRegex(
                search.GPUConfigurationError, "requires an explicit"
            ):
                search.gpu_ready_queue_configuration()
        with mock.patch.dict(
            os.environ,
            {byte_environment: str(search.GPU_READY_QUEUE_MAX_BYTES + 1)},
            clear=True,
        ):
            with self.assertRaisesRegex(
                search.GPUConfigurationError, "must be at most"
            ):
                search.gpu_ready_queue_configuration()

    def test_invalid_ready_queue_configuration_precedes_output_creation(self):
        pair = SimpleNamespace(cutoffs=SimpleNamespace(
            gathering=None, noise=None, trusted=None
        ))

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 0}

            def __len__(self):
                return 1

        with tempfile.TemporaryDirectory(
            prefix="astra-gpu-ready-config-"
        ) as temporary:
            outdir = Path(temporary) / "output"
            with mock.patch.dict(
                os.environ,
                {search.GPU_READY_QUEUE_DEPTH_ENV: "2"},
                clear=True,
            ):
                with mock.patch.object(
                    search, "gpu_profile_domain_available", return_value=False
                ):
                    with self.assertRaisesRegex(
                        search.GPUConfigurationError, "requires an explicit"
                    ):
                        search.hmmsearch(
                            {},
                            [pair],
                            2,
                            search_options(outdir),
                            all_sequences=[object()],
                            gpu_sequence_batch=object(),
                            gpu_postfilter=True,
                            gpu_profile_session=Session(),
                        )
            self.assertFalse(outdir.exists())

            serial_outdir = Path(temporary) / "serial-output"
            with mock.patch.dict(
                os.environ,
                {search.GPU_READY_QUEUE_BYTES_ENV: "10"},
                clear=True,
            ):
                with mock.patch.object(
                    search, "gpu_profile_domain_available", return_value=False
                ):
                    with self.assertRaisesRegex(
                        search.GPUConfigurationError,
                        "require the active bounded-ready-queue",
                    ):
                        search.hmmsearch(
                            {},
                            [pair],
                            1,
                            search_options(serial_outdir),
                            all_sequences=[object()],
                            gpu_sequence_batch=object(),
                            gpu_postfilter=True,
                            gpu_profile_session=Session(),
                        )
            self.assertFalse(serial_outdir.exists())


class GPUByteBoundedQueueTests(unittest.TestCase):
    def test_fifo_enforces_both_caps_and_releases_bytes_on_pop(self):
        ready = search._GPUByteBoundedReadyQueue(2, 10)
        ready.put_nowait("first", 6)
        with self.assertRaises(Full):
            ready.put_nowait("byte-blocked", 5)
        ready.put_nowait("second", 4)
        with self.assertRaises(Full):
            ready.put_nowait("item-blocked", 0)
        self.assertEqual(ready.get_nowait(), "first")
        ready.put_nowait("third", 6)
        self.assertEqual(ready.get_nowait(), "second")
        self.assertEqual(ready.get_nowait(), "third")
        with self.assertRaises(Empty):
            ready.get_nowait()
        self.assertEqual(ready.high_water(), (2, 10))
        self.assertEqual(ready.state(), (0, 0, 2, 10))

        # The popped six bytes were subtracted before the third item entered;
        # otherwise this high-water would incorrectly be sixteen.
        one_at_a_time = search._GPUByteBoundedReadyQueue(2, 6)
        one_at_a_time.put_nowait("a", 6)
        self.assertEqual(one_at_a_time.get_nowait(), "a")
        one_at_a_time.put_nowait("b", 6)
        self.assertEqual(one_at_a_time.get_nowait(), "b")
        self.assertEqual(one_at_a_time.high_water(), (1, 6))

    def test_rejects_oversize_and_nonexact_byte_weights(self):
        ready = search._GPUByteBoundedReadyQueue(1, 10)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            ready.put_nowait("large", 11)
        for invalid in (True, -1, 1.0, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    ready.put_nowait("invalid", invalid)

    def test_candidate_byte_api_is_exact_or_fails_closed(self):
        self.assertEqual(
            search._gpu_candidate_resident_bytes(
                SimpleNamespace(resident_bytes=17)
            ),
            17,
        )
        with self.assertRaisesRegex(
            search.GPUConfigurationError, "exact resident_bytes support"
        ):
            search._gpu_candidate_resident_bytes(SimpleNamespace())
        for invalid in (True, -1, 1.0, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    RuntimeError, "exact nonnegative integer"
                ):
                    search._gpu_candidate_resident_bytes(
                        SimpleNamespace(resident_bytes=invalid)
                    )

    def test_oversized_gpu_target_set_is_rejected_before_preflight_or_output(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-cap-") as temporary:
            root = Path(temporary)
            outdir = root / "out"
            config = {
                "db_urls": [
                    {
                        "name": "GPUDB",
                        "installed": True,
                        "installation_dir": os.fspath(root / "GPUDB"),
                        "molecule_type": "protein",
                    }
                ]
            }
            args = search_args(
                outdir,
                installed_hmms="GPUDB",
                gpu_manifest=["GPUDB=manifest.json"],
            )

            with (
                mock.patch.object(search, "GPU_CELL_CAP", 1),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object(), object()]},
                ),
                mock.patch.object(search, "preflight_gpu_databases") as preflight,
            ):
                with self.assertRaisesRegex(search.GPUConfigurationError, "at most"):
                    search.main(args)

            preflight.assert_not_called()
            self.assertFalse(outdir.exists())


class CPUPathTests(unittest.TestCase):
    def test_cpu_collection_boundaries_are_unchanged(self):
        hmms = [SimpleNamespace(cutoffs=SimpleNamespace()) for _ in range(3)]
        with tempfile.TemporaryDirectory(prefix="astra-cpu-gc-") as temporary:
            with (
                mock.patch.object(search, "HMM_CHUNK_SIZE", 2),
                mock.patch.object(
                    search.pyhmmer,
                    "hmmsearch",
                    side_effect=lambda *_args, **_kwargs: iter(()),
                ),
                mock.patch.object(search.gc, "collect") as collect,
            ):
                search.hmmsearch(
                    {},
                    hmms,
                    1,
                    search_options(temporary),
                    all_sequences=[object()],
                )
                self.assertEqual(collect.call_count, 3)

                collect.reset_mock()
                search.hmmsearch(
                    {"one.faa": [], "two.faa": []},
                    hmms,
                    1,
                    search_options(temporary),
                    macsyfinder_dir="macsyfinder",
                )
                self.assertEqual(collect.call_count, 2)

    def test_bulk_cpu_path_still_calls_only_pyhmmer_with_same_arguments(self):
        alphabet = pyhmmer.easel.Alphabet.amino()
        background = pyhmmer.plan7.Background(alphabet)
        builder = pyhmmer.plan7.Builder(alphabet, seed=7)
        query = pyhmmer.easel.TextSequence(
            name=b"cpu-model", sequence="ACDEFGHIKLMNPQRSTVWY" * 3
        ).digitize(alphabet)
        hmm, _, _ = builder.build(query, background)
        targets = pyhmmer.easel.DigitalSequenceBlock(alphabet, [query])

        original_import = builtins.__import__

        def reject_plan7_gpu(name, *args, **kwargs):
            if name == "plan7_gpu" or name.startswith("plan7_gpu."):
                raise AssertionError("CPU path imported plan7_gpu")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="astra-cpu-path-") as temporary:
            options = search_options(temporary, evalue="1.5")
            with (
                mock.patch.object(
                    search.pyhmmer, "hmmsearch", return_value=iter(())
                ) as cpu_search,
                mock.patch("builtins.__import__", side_effect=reject_plan7_gpu),
            ):
                search.hmmsearch({}, [hmm], 3, options, all_sequences=targets)

        cpu_search.assert_called_once()
        positional, keywords = cpu_search.call_args
        self.assertEqual(positional[0], [hmm])
        self.assertIs(positional[1], targets)
        self.assertEqual(keywords, {"cpus": 3, "E": 1.5})


class BulkDispatchTests(unittest.TestCase):
    def test_gpu_chunks_release_results_without_collecting_between_chunks(self):
        class Result:
            pass

        pairs = [
            SimpleNamespace(
                cutoffs=SimpleNamespace(
                    gathering=None, noise=None, trusted=None
                )
            )
            for _ in range(3)
        ]
        modules, api = synthetic_plan7_gpu()
        iterator_refs = []
        result_refs = []
        released_before_next = []

        def gpu_search(*_args, **_kwargs):
            if iterator_refs:
                released_before_next.append(
                    (iterator_refs[-1]() is None, result_refs[-1]() is None)
                )

            def results():
                result = Result()
                result_refs.append(weakref.ref(result))
                yield result

            iterator = results()
            iterator_refs.append(weakref.ref(iterator))
            return iterator

        api.gpu_hmmsearch.side_effect = gpu_search
        with tempfile.TemporaryDirectory(prefix="astra-gpu-gc-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 2),
                mock.patch.object(search.gc, "collect") as collect,
                mock.patch.object(
                    search, "process_hits_to_file", new=lambda *_args: None
                ),
            ):
                search.hmmsearch(
                    {},
                    pairs,
                    1,
                    search_options(temporary),
                    all_sequences=[object(), object()],
                    gpu_sequence_batch=object(),
                    gpu_postfilter=False,
                )

        self.assertEqual(released_before_next, [(True, True), (True, True)])
        self.assertTrue(all(reference() is None for reference in iterator_refs))
        self.assertTrue(all(reference() is None for reference in result_refs))
        collect.assert_called_once_with()

    def test_gpu_chunk_failure_closes_the_active_iterator(self):
        pair = SimpleNamespace(
            cutoffs=SimpleNamespace(gathering=None, noise=None, trusted=None)
        )
        modules, api = synthetic_plan7_gpu()
        closed = []

        def results():
            try:
                yield object()
            finally:
                closed.append(True)

        api.gpu_hmmsearch.side_effect = lambda *_args, **_kwargs: results()
        with tempfile.TemporaryDirectory(prefix="astra-gpu-gc-error-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search.gc, "collect") as collect,
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=RuntimeError("write failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    search.hmmsearch(
                        {},
                        [pair],
                        1,
                        search_options(temporary),
                        all_sequences=[object()],
                        gpu_sequence_batch=object(),
                        gpu_postfilter=False,
                    )

        self.assertEqual(closed, [True])
        collect.assert_not_called()

    def test_gpu_groups_whole_pairs_and_applies_cell_capped_chunks_in_order(self):
        def pair(cutoff):
            cutoffs = {"gathering": None, "noise": None, "trusted": None}
            if cutoff is not None:
                cutoffs[cutoff] = (0.0, 0.0)
            return SimpleNamespace(cutoffs=SimpleNamespace(**cutoffs))

        pairs = [
            pair("gathering"),
            pair(None),
            pair("gathering"),
            pair("trusted"),
            pair("gathering"),
            pair(None),
            pair("gathering"),
        ]
        targets = [object(), object()]
        modules, api = synthetic_plan7_gpu()
        api.gpu_hmmsearch.side_effect = lambda *args, **kwargs: iter(())
        with tempfile.TemporaryDirectory(prefix="astra-gpu-dispatch-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 5),
            ):
                search.hmmsearch(
                    {},
                    pairs,
                    2,
                    search_options(temporary, cascade=True),
                    all_sequences=targets,
                    gpu_sequence_batch=object(),
                )

        self.assertEqual(
            [call.args[0] for call in api.gpu_hmmsearch.call_args_list],
            [
                [pairs[0], pairs[2]],
                [pairs[4], pairs[6]],
                [pairs[1], pairs[5]],
                [pairs[3]],
            ],
        )
        self.assertEqual(
            [call.kwargs for call in api.gpu_hmmsearch.call_args_list],
            [
                {"cpus": 2, "bit_cutoffs": "gathering"},
                {"cpus": 2, "bit_cutoffs": "gathering"},
                {"cpus": 2, "E": 1e-15},
                {"cpus": 2, "bit_cutoffs": "trusted"},
            ],
        )

    def test_exact_postfilter_mode_reaches_a_nonempty_gpu_chunk(self):
        pair = SimpleNamespace(
            cutoffs=SimpleNamespace(gathering=None, noise=None, trusted=None)
        )
        batch = object()
        modules, api = synthetic_plan7_gpu()
        api.gpu_hmmsearch.return_value = iter(())
        with tempfile.TemporaryDirectory(prefix="astra-gpu-postfilter-") as temporary:
            with mock.patch.dict(sys.modules, modules):
                search.hmmsearch(
                    {},
                    [pair],
                    1,
                    search_options(temporary),
                    all_sequences=[object()],
                    gpu_sequence_batch=batch,
                    gpu_postfilter=True,
                )

        api.gpu_hmmsearch.assert_called_once_with(
            [pair], batch, cpus=1, postfilter=True
        )


class GPUProfileOverlapTests(unittest.TestCase):
    @staticmethod
    def pair(cutoff=None):
        cutoffs = {"gathering": None, "noise": None, "trusted": None}
        if cutoff is not None:
            cutoffs[cutoff] = (0.0, 0.0)
        return SimpleNamespace(cutoffs=SimpleNamespace(**cutoffs))

    def test_complete_domain_abi_passes_an_isolated_generation_pipeline(self):
        pair = self.pair()
        pipeline = object()
        generation_calls = []

        class Selection:
            indices = (0,)

            def __init__(self):
                self.close_count = 0

            def close(self):
                self.close_count += 1

        selection = Selection()

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 1

            def select(self, indices):
                self_outer.assertEqual(tuple(indices), (0,))
                return selection

        class Batch:
            alphabet = object()

            def _postfilter_forward_selection(self, *args, **kwargs):
                generation_calls.append((args, kwargs))
                return SimpleNamespace(sealed=True)

        self_outer = self
        modules, api = synthetic_plan7_gpu(True, True, True)
        api.gpu_hmmsearch.return_value = iter(("row",))
        pipeline_options = {
            "F1": 0.03,
            "F2": 0.004,
            "F3": 0.00005,
            "bias_filter": True,
            "E": 7.0,
        }
        batch = Batch()
        pipeline_factory = mock.Mock(return_value=pipeline)
        with tempfile.TemporaryDirectory(prefix="astra-gpu-domain-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search, "define_kwargs", return_value=pipeline_options
                ),
                mock.patch.object(
                    pyhmmer.plan7, "Pipeline", pipeline_factory
                ),
                mock.patch.object(search, "process_hits_to_file"),
            ):
                search.hmmsearch(
                    {},
                    [pair],
                    1,
                    search_options(temporary),
                    all_sequences=[object()],
                    gpu_sequence_batch=batch,
                    gpu_postfilter=True,
                    gpu_profile_session=Session(),
                )

        pipeline_factory.assert_called_once_with(
            batch.alphabet, **pipeline_options
        )
        self.assertEqual(len(generation_calls), 1)
        args, kwargs = generation_calls[0]
        self.assertIs(args[0], selection)
        self.assertEqual(args[1:], (0.03, 0.004, 0.00005, True))
        self.assertIs(kwargs["pipeline"], pipeline)
        self.assertEqual(kwargs["domain_guard"], search.GPU_DOMAIN_GUARD)
        self.assertEqual(selection.close_count, 1)
        api.gpu_hmmsearch.assert_called_once_with(
            [pair],
            mock.ANY,
            cpus=1,
            postfilter=True,
            **pipeline_options,
        )

    def test_disabled_bias_uses_the_existing_forward_selection_abi(self):
        calls = []

        class Batch:
            def _postfilter_forward_selection(self, *args, **kwargs):
                calls.append((args, kwargs))
                return object()

        with mock.patch.object(
            pyhmmer.plan7,
            "Pipeline",
            side_effect=AssertionError("domain pipeline constructed"),
        ):
            result = search._generate_gpu_profile_candidates(
                Batch(),
                "selection",
                {"F1": 0.03, "F2": 0.004, "F3": 0.00005,
                 "bias_filter": False},
                True,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            calls,
            [(("selection", 0.03, 0.004, 0.00005, False), {})],
        )

    def test_bounded_queue_runs_two_ahead_in_canonical_order(self):
        pairs = [
            self.pair("gathering"),
            self.pair(),
            self.pair("gathering"),
            self.pair(),
        ]
        first_consumption_started = threading.Event()
        third_generation_started = threading.Event()
        release_third_generation = threading.Event()
        third_generation_finished = threading.Event()
        selection_calls = []
        selections = []
        selection_threads = []
        generation_threads = []
        generation_calls = []
        search_calls = []
        candidate_refs = []
        live_candidate_counts = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.closed = False
                self.close_count = 0

            def close(self):
                self.closed = True
                self.close_count += 1

        class Session:
            closed = False
            statistics = {
                "worker_count": 0,
                "build_worker_count": 3,
                "selection_worker_count": 0,
                "host_bytes": 1234,
            }

            def __len__(self):
                return len(pairs)

            def select(self, indices):
                selection_threads.append(threading.current_thread().name)
                selection_calls.append(tuple(indices))
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                generation_threads.append(threading.current_thread().name)
                generation_calls.append(
                    (selection.indices, F1, F2, F3, bias_filter)
                )
                if selection.indices == (1,):
                    self_outer.assertTrue(first_consumption_started.wait(2))
                    third_generation_started.set()
                    self_outer.assertTrue(release_third_generation.wait(2))
                    time.sleep(0.03)
                    third_generation_finished.set()
                candidates = SimpleNamespace(
                    indices=selection.indices,
                    F1=F1,
                    F2=F2,
                    F3=F3,
                    bias_filter=bias_filter,
                    sealed=True,
                )
                # SimpleNamespace is not weak-referenceable; use a tiny bound
                # wrapper so the test can prove the three-live-batch ceiling.
                class Candidate:
                    pass

                candidate = Candidate()
                candidate.__dict__.update(vars(candidates))
                candidate_refs.append(weakref.ref(candidate))
                live_candidate_counts.append(sum(
                    reference() is not None for reference in candidate_refs
                ))
                return candidate

        self_outer = self
        modules, api = synthetic_plan7_gpu(True)
        observed = []

        def gpu_search(chunk, candidates, **kwargs):
            search_calls.append((list(chunk), candidates.indices, kwargs))

            def results():
                if candidates.indices == (0,):
                    first_consumption_started.set()
                    self.assertTrue(third_generation_started.wait(2))
                    release_third_generation.set()
                    self.assertTrue(third_generation_finished.wait(2))
                    # Let the producer reach its bounded put while chunk 2 is
                    # still resident in the sole ready slot.
                    time.sleep(0.03)
                for pair in chunk:
                    yield (candidates.indices, pair, kwargs)

            return results()

        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        metrics = search.GPUOverlapMetrics()
        pipeline_options = {
            "bit_cutoffs": "gathering",
            "F1": 0.03,
            "F2": 0.004,
            "F3": 0.00005,
            "bias_filter": False,
        }
        with tempfile.TemporaryDirectory(prefix="astra-gpu-overlap-") as temporary:
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 2),
                mock.patch.object(
                    search, "define_kwargs", return_value=pipeline_options
                ),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=lambda hits, _fh: observed.append(hits),
                ),
            ):
                search.hmmsearch(
                    {},
                    pairs,
                    3,
                    search_options(temporary),
                    all_sequences=[object(), object()],
                    gpu_sequence_batch=Batch(),
                    gpu_postfilter=True,
                    gpu_profile_session=Session(),
                    gpu_metrics=metrics,
                )

        self.assertEqual(selection_calls, [(0,), (2,), (1,), (3,)])
        self.assertTrue(all(selection.closed for selection in selections))
        self.assertTrue(all(selection.close_count == 1 for selection in selections))
        self.assertTrue(all(
            name.startswith("astra-gpu-generate")
            for name in selection_threads
        ))
        self.assertEqual(
            generation_calls,
            [
                ((0,), 0.03, 0.004, 0.00005, False),
                ((2,), 0.03, 0.004, 0.00005, False),
                ((1,), 0.03, 0.004, 0.00005, False),
                ((3,), 0.03, 0.004, 0.00005, False),
            ],
        )
        self.assertEqual(
            [item[0] for item in observed],
            [(0,), (2,), (1,), (3,)],
        )
        self.assertTrue(
            all(name.startswith("astra-gpu-generate") for name in generation_threads)
        )
        self.assertEqual(
            [call[0] for call in search_calls],
            [[pairs[0]], [pairs[2]], [pairs[1]], [pairs[3]]],
        )
        for _, _, kwargs in search_calls:
            self.assertEqual(kwargs["F1"], 0.03)
            self.assertEqual(kwargs["F2"], 0.004)
            self.assertEqual(kwargs["F3"], 0.00005)
            self.assertIs(kwargs["bias_filter"], False)
            self.assertEqual(kwargs["cpus"], 2)
            self.assertIs(kwargs["postfilter"], True)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requested_thread_count"], 3)
        self.assertEqual(snapshot["profile_worker_count"], 0)
        self.assertEqual(snapshot["profile_build_worker_count"], 3)
        self.assertEqual(snapshot["profile_selection_worker_count"], 0)
        self.assertEqual(snapshot["producer_slot_count"], 1)
        self.assertEqual(snapshot["continuation_worker_count"], 2)
        self.assertTrue(snapshot["profile_overlap_enabled"])
        self.assertEqual(snapshot["scheduler_mode"], "bounded-ready-queue")
        self.assertEqual(snapshot["chunk_count"], 4)
        self.assertEqual(snapshot["generated_chunk_count"], 4)
        self.assertEqual(snapshot["consumed_chunk_count"], 4)
        self.assertEqual(snapshot["ready_queue_capacity"], 1)
        self.assertEqual(snapshot["ready_queue_high_water"], 1)
        self.assertEqual(snapshot["ready_queue_byte_capacity"], 0)
        self.assertEqual(snapshot["ready_queue_byte_high_water"], 0)
        self.assertEqual(snapshot["ready_queue_final_count"], 0)
        self.assertEqual(snapshot["ready_queue_final_bytes"], 0)
        self.assertEqual(snapshot["generated_candidate_bytes"], 0)
        self.assertEqual(snapshot["consumed_candidate_bytes"], 0)
        self.assertEqual(snapshot["maximum_candidate_bytes"], 0)
        self.assertEqual(len(snapshot["generation_records"]), 4)
        self.assertTrue(all(
            record["candidate_resident_bytes"] is None
            for record in snapshot["generation_records"]
        ))
        self.assertEqual(len(snapshot["continuation_records"]), 4)
        self.assertEqual(snapshot["producer_lookahead_capacity"], 2)
        self.assertEqual(snapshot["producer_lookahead_high_water"], 2)
        self.assertGreaterEqual(snapshot["producer_lookahead_start_count"], 1)
        self.assertEqual(snapshot["live_candidate_capacity"], 3)
        self.assertGreaterEqual(snapshot["producer_idle_count"], 1)
        self.assertGreater(snapshot["producer_idle_seconds"], 0.01)
        self.assertGreater(snapshot["selection_seconds"], 0.0)
        self.assertGreater(snapshot["overlap_seconds"], 0.01)
        self.assertEqual(max(live_candidate_counts), 3)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in candidate_refs))
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

    def test_byte_bounded_depths_reach_exact_fifo_and_memory_limits(self):
        for depth in (1, 2, 4):
            with self.subTest(depth=depth):
                pair_count = depth + 2
                pairs = [self.pair() for _ in range(pair_count)]
                first_consumption_started = threading.Event()
                last_generation_started = threading.Event()
                selection_calls = []
                selections = []
                generation_calls = []
                candidate_refs = []
                live_candidate_counts = []
                observed = []

                class Selection:
                    def __init__(self, indices):
                        self.indices = tuple(indices)
                        self.close_count = 0

                    def close(self):
                        self.close_count += 1

                class Session:
                    statistics = {
                        "worker_count": 0,
                        "build_worker_count": 3,
                        "selection_worker_count": 0,
                        "host_bytes": 1234,
                    }

                    def select(self, indices):
                        selection_calls.append(tuple(indices))
                        selection = Selection(indices)
                        selections.append(selection)
                        return selection

                class Candidate:
                    resident_bytes = 10

                    def __init__(self, indices):
                        self.indices = tuple(indices)

                class Batch:
                    def _postfilter_forward_selection(
                        self, selection, F1, F2, F3, bias_filter
                    ):
                        if selection.indices == (1,):
                            self_outer.assertTrue(
                                first_consumption_started.wait(2)
                            )
                        generation_calls.append(selection.indices)
                        candidate = Candidate(selection.indices)
                        candidate_refs.append(weakref.ref(candidate))
                        live_candidate_counts.append(sum(
                            reference() is not None
                            for reference in candidate_refs
                        ))
                        if len(generation_calls) == pair_count:
                            last_generation_started.set()
                        return candidate

                def gpu_search(_chunk, candidates, **_kwargs):
                    def results():
                        if candidates.indices == (0,):
                            first_consumption_started.set()
                            self.assertTrue(last_generation_started.wait(2))
                        yield candidates.indices

                    return results()

                chunks = [
                    (index + 1, [pair], (index,), {})
                    for index, pair in enumerate(pairs)
                ]
                self_outer = self
                modules, _ = synthetic_plan7_gpu(True)
                modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
                metrics = search.GPUOverlapMetrics()
                with (
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(
                        search,
                        "process_hits_to_file",
                        side_effect=lambda hits, _fh: observed.append(hits),
                    ),
                ):
                    search._run_gpu_profile_pipeline(
                        chunks,
                        Session(),
                        Batch(),
                        2,
                        io.StringIO(),
                        metrics,
                        ready_queue_configuration=(depth, depth * 10),
                    )

                expected_order = [(index,) for index in range(pair_count)]
                self.assertEqual(selection_calls, expected_order)
                self.assertEqual(generation_calls, expected_order)
                self.assertEqual(observed, expected_order)
                self.assertTrue(all(
                    selection.close_count == 1 for selection in selections
                ))
                snapshot = metrics.snapshot()
                self.assertEqual(snapshot["ready_queue_capacity"], depth)
                self.assertEqual(
                    snapshot["ready_queue_byte_capacity"], depth * 10
                )
                self.assertEqual(snapshot["ready_queue_high_water"], depth)
                self.assertEqual(
                    snapshot["ready_queue_byte_high_water"], depth * 10
                )
                self.assertEqual(snapshot["ready_queue_final_count"], 0)
                self.assertEqual(snapshot["ready_queue_final_bytes"], 0)
                self.assertEqual(
                    snapshot["producer_lookahead_capacity"], depth + 1
                )
                self.assertEqual(
                    snapshot["live_candidate_capacity"], depth + 2
                )
                self.assertEqual(
                    snapshot["generated_candidate_bytes"], pair_count * 10
                )
                self.assertEqual(
                    snapshot["consumed_candidate_bytes"], pair_count * 10
                )
                self.assertEqual(snapshot["maximum_candidate_bytes"], 10)
                self.assertEqual(
                    [record["position"] for record in snapshot["generation_records"]],
                    list(range(pair_count)),
                )
                self.assertTrue(all(
                    record["candidate_resident_bytes"] == 10
                    and math.isfinite(
                        record["generation_started_monotonic_seconds"]
                    )
                    and math.isfinite(
                        record["generation_finished_monotonic_seconds"]
                    )
                    and math.isfinite(record["generation_duration_seconds"])
                    and record["generation_finished_monotonic_seconds"]
                    >= record["generation_started_monotonic_seconds"]
                    and record["generation_duration_seconds"]
                    == record["generation_finished_monotonic_seconds"]
                    - record["generation_started_monotonic_seconds"]
                    for record in snapshot["generation_records"]
                ))
                for record in snapshot["generation_records"]:
                    for phase in ("selection", "generation"):
                        started = record[
                            f"{phase}_started_monotonic_seconds"
                        ]
                        finished = record[
                            f"{phase}_finished_monotonic_seconds"
                        ]
                        duration = record[f"{phase}_duration_seconds"]
                        self.assertTrue(all(map(
                            math.isfinite, (started, finished, duration)
                        )))
                        self.assertGreaterEqual(finished, started)
                        self.assertEqual(duration, finished - started)
                    wait_started = record[
                        "ready_wait_started_monotonic_seconds"
                    ]
                    dequeued = record["ready_dequeued_monotonic_seconds"]
                    wait_duration = record["ready_wait_duration_seconds"]
                    self.assertTrue(all(map(
                        math.isfinite,
                        (wait_started, dequeued, wait_duration),
                    )))
                    self.assertGreaterEqual(dequeued, wait_started)
                    self.assertEqual(
                        wait_duration, dequeued - wait_started
                    )
                self.assertEqual(
                    [
                        record["chunk_index"]
                        for record in snapshot["continuation_records"]
                    ],
                    list(range(1, pair_count + 1)),
                )
                self.assertTrue(all(
                    record["completed"]
                    and math.isfinite(record["started_monotonic_seconds"])
                    and math.isfinite(record["finished_monotonic_seconds"])
                    and math.isfinite(record["duration_seconds"])
                    and record["finished_monotonic_seconds"]
                    >= record["started_monotonic_seconds"]
                    and record["duration_seconds"]
                    == record["finished_monotonic_seconds"]
                    - record["started_monotonic_seconds"]
                    for record in snapshot["continuation_records"]
                ))
                self.assertGreaterEqual(snapshot["producer_idle_count"], 1)
                self.assertEqual(max(live_candidate_counts), depth + 2)
                gc.collect()
                self.assertTrue(all(
                    reference() is None for reference in candidate_refs
                ))
                self.assertFalse(any(
                    thread.name.startswith("astra-gpu-generate")
                    for thread in threading.enumerate()
                ))

    def test_byte_ceiling_can_bind_before_configured_item_depth(self):
        pairs = [self.pair() for _ in range(4)]
        first_consumption_started = threading.Event()
        fourth_generation_started = threading.Event()
        generation_count = 0

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)

            def close(self):
                pass

        class Session:
            statistics = {"worker_count": 0, "host_bytes": 0}

            def select(self, indices):
                return Selection(indices)

        class Candidate:
            resident_bytes = 10

            def __init__(self, indices):
                self.indices = tuple(indices)

        class Batch:
            def _postfilter_forward_selection(self, selection, *_args):
                nonlocal generation_count
                if selection.indices == (1,):
                    self_outer.assertTrue(first_consumption_started.wait(2))
                generation_count += 1
                if generation_count == 4:
                    fourth_generation_started.set()
                return Candidate(selection.indices)

        def gpu_search(_chunk, candidates, **_kwargs):
            def results():
                if candidates.indices == (0,):
                    first_consumption_started.set()
                    self.assertTrue(fourth_generation_started.wait(2))
                yield candidates.indices

            return results()

        self_outer = self
        modules, _ = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        metrics = search.GPUOverlapMetrics()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(search, "process_hits_to_file"),
        ):
            search._run_gpu_profile_pipeline(
                [
                    (index + 1, [pair], (index,), {})
                    for index, pair in enumerate(pairs)
                ],
                Session(),
                Batch(),
                2,
                io.StringIO(),
                metrics,
                ready_queue_configuration=(4, 20),
            )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["ready_queue_capacity"], 4)
        self.assertEqual(snapshot["ready_queue_high_water"], 2)
        self.assertEqual(snapshot["ready_queue_byte_high_water"], 20)
        self.assertGreaterEqual(snapshot["producer_idle_count"], 1)

    def test_overweight_candidate_error_remains_behind_earlier_output(self):
        pairs = [self.pair() for _ in range(3)]
        selections = []
        observed = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.close_count = 0

            def close(self):
                self.close_count += 1

        class Session:
            statistics = {"worker_count": 0, "host_bytes": 0}

            def select(self, indices):
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Candidate:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.resident_bytes = 11 if self.indices == (1,) else 5

        class Batch:
            def _postfilter_forward_selection(self, selection, *_args):
                return Candidate(selection.indices)

        def gpu_search(_chunk, candidates, **_kwargs):
            return iter((candidates.indices,))

        modules, _ = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                search,
                "process_hits_to_file",
                side_effect=lambda hits, _fh: observed.append(hits),
            ),
        ):
            with self.assertRaisesRegex(
                search.GPUConfigurationError,
                "requires 11 bytes, exceeding",
            ):
                search._run_gpu_profile_pipeline(
                    [
                        (index + 1, [pair], (index,), {})
                        for index, pair in enumerate(pairs)
                    ],
                    Session(),
                    Batch(),
                    2,
                    io.StringIO(),
                    ready_queue_configuration=(2, 10),
                )

        self.assertEqual(observed, [(0,)])
        self.assertEqual(
            [selection.indices for selection in selections], [(0,), (1,)]
        )
        self.assertTrue(all(
            selection.close_count == 1 for selection in selections
        ))

    def test_mutating_candidate_byte_charge_is_rejected_before_search(self):
        pair = self.pair()
        search_called = []

        class Selection:
            indices = (0,)

            def close(self):
                pass

        class Session:
            statistics = {"worker_count": 0, "host_bytes": 0}

            def select(self, _indices):
                return Selection()

        class Candidate:
            def __init__(self):
                self.read_count = 0

            @property
            def resident_bytes(self):
                self.read_count += 1
                return 5 if self.read_count == 1 else 6

        class Batch:
            def _postfilter_forward_selection(self, *_args):
                return Candidate()

        def gpu_search(*_args, **_kwargs):
            search_called.append(True)
            return iter(())

        modules, _ = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                RuntimeError, "changed while queued: 5 -> 6"
            ):
                search._run_gpu_profile_pipeline(
                    [(1, [pair], (0,), {})],
                    Session(),
                    Batch(),
                    2,
                    io.StringIO(),
                    ready_queue_configuration=(1, 10),
                )
        self.assertEqual(search_called, [])

    def test_missing_candidate_byte_api_is_rejected_before_search(self):
        pair = self.pair()
        search_called = []

        class Selection:
            def close(self):
                pass

        class Session:
            statistics = {"worker_count": 0, "host_bytes": 0}

            def select(self, _indices):
                return Selection()

        class Batch:
            def _postfilter_forward_selection(self, *_args):
                return SimpleNamespace()

        def gpu_search(*_args, **_kwargs):
            search_called.append(True)
            return iter(())

        modules, _ = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                search.GPUConfigurationError, "exact resident_bytes support"
            ):
                search._run_gpu_profile_pipeline(
                    [(1, [pair], (0,), {})],
                    Session(),
                    Batch(),
                    2,
                    io.StringIO(),
                    ready_queue_configuration=(1, 10),
                )
        self.assertEqual(search_called, [])

    def test_current_error_stops_producer_blocked_by_byte_ceiling(self):
        pairs = [self.pair() for _ in range(3)]
        third_generated = threading.Event()
        selections = []
        candidate_refs = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.close_count = 0

            def close(self):
                self.close_count += 1

        class Session:
            statistics = {"worker_count": 0, "host_bytes": 0}

            def select(self, indices):
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Candidate:
            resident_bytes = 10

            def __init__(self, indices):
                self.indices = tuple(indices)

        class Batch:
            def _postfilter_forward_selection(self, selection, *_args):
                candidate = Candidate(selection.indices)
                candidate_refs.append(weakref.ref(candidate))
                if selection.indices == (2,):
                    third_generated.set()
                return candidate

        def gpu_search(_chunk, candidates, **_kwargs):
            def results():
                if candidates.indices == (0,):
                    self.assertTrue(third_generated.wait(2))
                yield candidates.indices

            return results()

        modules, _ = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        metrics = search.GPUOverlapMetrics()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                search,
                "process_hits_to_file",
                side_effect=ValueError("current write failed"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "current write failed"):
                search._run_gpu_profile_pipeline(
                    [
                        (index + 1, [pair], (index,), {})
                        for index, pair in enumerate(pairs)
                    ],
                    Session(),
                    Batch(),
                    2,
                    io.StringIO(),
                    metrics,
                    ready_queue_configuration=(4, 10),
                )

        self.assertEqual(
            [selection.close_count for selection in selections], [1, 1, 1]
        )
        self.assertGreaterEqual(metrics.producer_idle_count, 1)
        gc.collect()
        self.assertEqual(len(candidate_refs), 3)
        self.assertTrue(all(
            reference() is None for reference in candidate_refs
        ))
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

    def test_legacy_single_prefetch_control_retains_old_scheduler(self):
        pairs = [self.pair(), self.pair()]
        selection_threads = []
        generation_threads = []
        emitted = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)

            def close(self):
                pass

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 2

            def select(self, indices):
                selection_threads.append(threading.current_thread().name)
                return Selection(indices)

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                generation_threads.append(threading.current_thread().name)
                return SimpleNamespace(indices=selection.indices)

        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.side_effect = (
            lambda _chunk, candidates, **_kwargs: iter((candidates.indices,))
        )
        metrics = search.GPUOverlapMetrics()
        with tempfile.TemporaryDirectory(
            prefix="astra-gpu-legacy-overlap-"
        ) as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.dict(
                    os.environ,
                    {
                        search.GPU_LEGACY_OVERLAP_ENV:
                            search.GPU_LEGACY_OVERLAP_VALUE,
                    },
                ),
                mock.patch.object(search, "GPU_CELL_CAP", 1),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=lambda hits, _fh: emitted.append(hits),
                ),
            ):
                search.hmmsearch(
                    {}, pairs, 2, search_options(temporary),
                    all_sequences=[object()],
                    gpu_sequence_batch=Batch(),
                    gpu_postfilter=True,
                    gpu_profile_session=Session(),
                    gpu_metrics=metrics,
                )

        self.assertEqual(emitted, [(0,), (1,)])
        self.assertTrue(all(
            name == threading.main_thread().name
            for name in selection_threads
        ))
        self.assertTrue(all(
            name.startswith("astra-gpu-generate")
            for name in generation_threads
        ))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["scheduler_mode"], "single-prefetch")
        self.assertEqual(snapshot["ready_queue_capacity"], 0)
        self.assertEqual(snapshot["producer_lookahead_high_water"], 0)

    def test_legacy_selector_rejects_serial_and_sessionless_fallback(self):
        pair = self.pair()

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}
            select_count = 0

            def __len__(self):
                return 1

            def select(self, _indices):
                self.select_count += 1
                raise AssertionError("selection reached")

        class Batch:
            def _postfilter_forward_selection(self, *_args):
                raise AssertionError("generation reached")

        modules, _ = synthetic_plan7_gpu(True)
        with tempfile.TemporaryDirectory(
            prefix="astra-gpu-legacy-selector-"
        ) as temporary:
            root = Path(temporary)
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.dict(
                    os.environ,
                    {
                        search.GPU_LEGACY_OVERLAP_ENV:
                            search.GPU_LEGACY_OVERLAP_VALUE,
                    },
                ),
            ):
                for label, session in (
                    ("serial", Session()),
                    ("sessionless", None),
                ):
                    outdir = root / label
                    with self.subTest(label=label):
                        with self.assertRaisesRegex(
                            search.GPUConfigurationError,
                            "requires an active overlapping",
                        ):
                            search.hmmsearch(
                                {}, [pair], 1, search_options(outdir),
                                all_sequences=[object()],
                                gpu_sequence_batch=Batch(),
                                gpu_postfilter=True,
                                gpu_profile_session=session,
                            )
                        self.assertFalse(outdir.exists())
                        if session is not None:
                            self.assertEqual(session.select_count, 0)

    def test_one_thread_budget_uses_serial_full_forward_session(self):
        pair = self.pair()
        generated = []

        class Selection:
            indices = (0,)

            def close(self):
                pass

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 1

            def select(self, indices):
                self_outer.assertEqual(tuple(indices), (0,))
                return Selection()

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                self_outer.assertIs(
                    threading.current_thread(), threading.main_thread()
                )
                generated.append((F1, F2, F3, bias_filter))
                return SimpleNamespace(indices=selection.indices, sealed=True)

        self_outer = self
        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.return_value = iter(("row",))
        metrics = search.GPUOverlapMetrics()
        with tempfile.TemporaryDirectory(prefix="astra-gpu-one-thread-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search,
                    "ThreadPoolExecutor",
                    side_effect=AssertionError("overlap executor created"),
                ),
                mock.patch.object(search, "process_hits_to_file"),
            ):
                search.hmmsearch(
                    {},
                    [pair],
                    1,
                    search_options(temporary),
                    all_sequences=[object()],
                    gpu_sequence_batch=Batch(),
                    gpu_postfilter=True,
                    gpu_profile_session=Session(),
                    gpu_metrics=metrics,
                )

        self.assertEqual(generated, [(0.02, 0.001, 0.00001, True)])
        self.assertEqual(api.gpu_hmmsearch.call_args.kwargs["cpus"], 1)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requested_thread_count"], 1)
        self.assertEqual(snapshot["producer_slot_count"], 0)
        self.assertEqual(snapshot["continuation_worker_count"], 1)
        self.assertFalse(snapshot["profile_overlap_enabled"])

    def test_profile_selection_workers_must_be_zero_during_search(self):
        pair = self.pair()

        class Session:
            closed = False
            statistics = {
                "worker_count": 1,
                "build_worker_count": 3,
                "selection_worker_count": 1,
                "host_bytes": 1234,
            }

            def __len__(self):
                return 1

        class Batch:
            def _postfilter_forward_selection(self, *_args):
                raise AssertionError("generation reached")

        with tempfile.TemporaryDirectory(
            prefix="astra-gpu-selection-budget-"
        ) as temporary:
            with self.assertRaisesRegex(
                search.GPUConfigurationError,
                "require zero persistent selection workers",
            ):
                search.hmmsearch(
                    {},
                    [pair],
                    3,
                    search_options(temporary),
                    all_sequences=[object()],
                    gpu_sequence_batch=Batch(),
                    gpu_postfilter=True,
                    gpu_profile_session=Session(),
                )

    def test_prefetch_selection_failure_is_deferred_until_ready_output(self):
        pairs = [self.pair(), self.pair()]
        emitted = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)

            def close(self):
                pass

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 2

            def select(self, indices):
                if tuple(indices) == (1,):
                    raise RuntimeError("prefetch selection failed")
                return Selection(indices)

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                return SimpleNamespace(indices=selection.indices)

        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.side_effect = lambda *_args, **_kwargs: iter(("row-0",))
        with tempfile.TemporaryDirectory(prefix="astra-gpu-order-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 2),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=lambda hits, _fh: emitted.append(hits),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "prefetch selection failed"
                ):
                    search.hmmsearch(
                        {}, pairs, 2, search_options(temporary),
                        all_sequences=[object(), object()],
                        gpu_sequence_batch=Batch(),
                        gpu_postfilter=True,
                        gpu_profile_session=Session(),
                    )

        self.assertEqual(emitted, ["row-0"])
        self.assertEqual(api.gpu_hmmsearch.call_count, 1)
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

    def test_generation_failure_is_fifo_and_stops_later_selection(self):
        pairs = [self.pair(), self.pair(), self.pair()]
        selection_calls = []
        generation_calls = []
        selections = []
        emitted = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.close_count = 0

            def close(self):
                self.close_count += 1

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 3

            def select(self, indices):
                selection_calls.append(tuple(indices))
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                generation_calls.append(selection.indices)
                if selection.indices == (1,):
                    raise RuntimeError("future generation failed")
                return SimpleNamespace(indices=selection.indices)

        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.side_effect = (
            lambda _chunk, candidates, **_kwargs: iter((candidates.indices,))
        )
        with tempfile.TemporaryDirectory(
            prefix="astra-gpu-generation-order-"
        ) as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 1),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=lambda hits, _fh: emitted.append(hits),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "future generation failed"
                ):
                    search.hmmsearch(
                        {}, pairs, 2, search_options(temporary),
                        all_sequences=[object()],
                        gpu_sequence_batch=Batch(),
                        gpu_postfilter=True,
                        gpu_profile_session=Session(),
                    )

        self.assertEqual(emitted, [(0,)])
        self.assertEqual(selection_calls, [(0,), (1,)])
        self.assertEqual(generation_calls, [(0,), (1,)])
        self.assertEqual(
            [selection.close_count for selection in selections], [1, 1]
        )
        self.assertEqual(api.gpu_hmmsearch.call_count, 1)
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

    def test_serial_control_uses_identical_gpu_through_forward_stages(self):
        pairs = [self.pair("gathering"), self.pair(), self.pair("gathering")]
        selection_calls = []
        generation_calls = []
        emitted = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)

            def close(self):
                pass

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 3

            def select(self, indices):
                selection_calls.append(tuple(indices))
                return Selection(indices)

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                generation_calls.append(
                    (selection.indices, F1, F2, F3, bias_filter)
                )
                return SimpleNamespace(indices=selection.indices, sealed=True)

        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.side_effect = (
            lambda chunk, candidates, **kwargs: iter(
                (candidates.indices, pair, kwargs) for pair in chunk
            )
        )
        pipeline_options = {
            "bit_cutoffs": "gathering",
            "F1": 0.03,
            "F2": 0.004,
            "F3": 0.00005,
            "bias_filter": False,
        }
        metrics = search.GPUOverlapMetrics()
        with tempfile.TemporaryDirectory(prefix="astra-gpu-serial-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 4),
                mock.patch.object(
                    search, "define_kwargs", return_value=pipeline_options
                ),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=lambda hits, _fh: emitted.append(hits),
                ),
            ):
                search.hmmsearch(
                    {}, pairs, 3, search_options(temporary),
                    all_sequences=[object(), object()],
                    gpu_sequence_batch=Batch(),
                    gpu_postfilter=True,
                    gpu_profile_session=Session(),
                    gpu_metrics=metrics,
                    gpu_profile_overlap=False,
                )

        self.assertEqual(selection_calls, [(0, 2), (1,)])
        self.assertEqual(
            generation_calls,
            [
                ((0, 2), 0.03, 0.004, 0.00005, False),
                ((1,), 0.03, 0.004, 0.00005, False),
            ],
        )
        self.assertEqual([item[0] for item in emitted], [(0, 2), (0, 2), (1,)])
        for call in api.gpu_hmmsearch.call_args_list:
            self.assertEqual(call.kwargs["cpus"], 2)
            self.assertEqual(call.kwargs["F2"], 0.004)
            self.assertEqual(call.kwargs["F3"], 0.00005)
            self.assertIs(call.kwargs["bias_filter"], False)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requested_thread_count"], 3)
        self.assertEqual(snapshot["profile_worker_count"], 0)
        self.assertEqual(snapshot["profile_build_worker_count"], 0)
        self.assertEqual(snapshot["profile_selection_worker_count"], 0)
        self.assertEqual(snapshot["producer_slot_count"], 1)
        self.assertEqual(snapshot["continuation_worker_count"], 2)
        self.assertFalse(snapshot["profile_overlap_enabled"])
        self.assertEqual(snapshot["generated_chunk_count"], 2)
        self.assertEqual(snapshot["consumed_chunk_count"], 2)
        self.assertEqual(snapshot["overlap_seconds"], 0.0)
        self.assertEqual(snapshot["generation_wait_seconds"], 0.0)

    def test_current_output_error_wins_and_closes_while_prefetch_fails(self):
        pairs = [self.pair(), self.pair()]
        consumption_started = threading.Event()
        iterator_closed = []
        selections = []

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.closed = False
                self.close_count = 0

            def close(self):
                self.closed = True
                self.close_count += 1

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 2

            def select(self, indices):
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                if selection.indices == (1,):
                    if not consumption_started.wait(2):
                        raise AssertionError("continuation never started")
                    raise RuntimeError("prefetched generation failed")
                return SimpleNamespace(indices=selection.indices)

        def gpu_search(*_args, **_kwargs):
            def results():
                try:
                    consumption_started.set()
                    yield "current-row"
                finally:
                    iterator_closed.append(True)

            return results()

        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.side_effect = gpu_search
        with tempfile.TemporaryDirectory(prefix="astra-gpu-cleanup-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 2),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=ValueError("current write failed"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "current write failed"):
                    search.hmmsearch(
                        {}, pairs, 2, search_options(temporary),
                        all_sequences=[object(), object()],
                        gpu_sequence_batch=Batch(),
                        gpu_postfilter=True,
                        gpu_profile_session=Session(),
                    )

        self.assertEqual(iterator_closed, [True])
        self.assertTrue(all(selection.closed for selection in selections))
        self.assertTrue(all(selection.close_count == 1 for selection in selections))
        self.assertEqual(api.gpu_hmmsearch.call_count, 1)
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

    def test_current_error_drains_ready_and_inflight_candidates(self):
        pairs = [self.pair(), self.pair(), self.pair()]
        selections = []
        candidate_refs = []
        third_generation_started = threading.Event()
        release_third_generation = threading.Event()

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.close_count = 0

            def close(self):
                self.close_count += 1

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 3

            def select(self, indices):
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Candidate:
            def __init__(self, indices):
                self.indices = tuple(indices)

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                if selection.indices == (2,):
                    third_generation_started.set()
                    self_outer.assertTrue(release_third_generation.wait(2))
                candidate = Candidate(selection.indices)
                candidate_refs.append(weakref.ref(candidate))
                return candidate

        def gpu_search(_chunk, candidates, **_kwargs):
            def results():
                yield (candidates.indices, "current-row")

            return results()

        def fail_current_output(_hits, _fh):
            self.assertTrue(third_generation_started.wait(2))
            release_third_generation.set()
            raise ValueError("current write failed")

        self_outer = self
        modules, api = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        with tempfile.TemporaryDirectory(prefix="astra-gpu-cancel-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 1),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=fail_current_output,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "current write failed"):
                    search.hmmsearch(
                        {}, pairs, 2, search_options(temporary),
                        all_sequences=[object()],
                        gpu_sequence_batch=Batch(),
                        gpu_postfilter=True,
                        gpu_profile_session=Session(),
                    )

        self.assertEqual(
            [selection.close_count for selection in selections],
            [1, 1, 1],
        )
        gc.collect()
        self.assertEqual(len(candidate_refs), 3)
        self.assertTrue(all(reference() is None for reference in candidate_refs))
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

    def test_current_error_wins_join_interrupt_and_join_retries_to_exit(self):
        pairs = [self.pair(), self.pair()]
        second_generation_started = threading.Event()
        second_generation_finished = threading.Event()
        selections = []
        outer_resource_closed = threading.Event()

        class Selection:
            def __init__(self, indices):
                self.indices = tuple(indices)
                self.close_count = 0

            def close(self):
                self.close_count += 1

        class Session:
            closed = False
            statistics = {"worker_count": 0, "host_bytes": 1234}

            def __len__(self):
                return 2

            def select(self, indices):
                selection = Selection(indices)
                selections.append(selection)
                return selection

        class Batch:
            def _postfilter_forward_selection(
                self, selection, F1, F2, F3, bias_filter
            ):
                if selection.indices == (1,):
                    second_generation_started.set()
                    time.sleep(0.05)
                    self_outer.assertFalse(outer_resource_closed.is_set())
                    second_generation_finished.set()
                return SimpleNamespace(indices=selection.indices)

        class InterruptingJoinThread(threading.Thread):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.join_calls = 0
                self._misreport_dead = False
                self.premature_join = None
                self.instances.append(self)

            def join(self, *args, **kwargs):
                self.join_calls += 1
                if self.join_calls == 1:
                    self.premature_join = not second_generation_finished.is_set()
                    self._misreport_dead = True
                    raise RuntimeError("join interrupted")
                result = super().join(*args, **kwargs)
                self._misreport_dead = False
                return result

            def is_alive(self):
                if self._misreport_dead:
                    return False
                return super().is_alive()

        def gpu_search(_chunk, candidates, **_kwargs):
            return iter(((candidates.indices, "current-row"),))

        def fail_current_output(_hits, _fh):
            self.assertTrue(second_generation_started.wait(2))
            raise ValueError("current write failed")

        self_outer = self
        modules, _ = synthetic_plan7_gpu(True)
        modules["plan7_gpu.astra_search"].hmmsearch = gpu_search
        with tempfile.TemporaryDirectory(
            prefix="astra-gpu-join-interrupt-"
        ) as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 1),
                mock.patch.object(search, "Thread", InterruptingJoinThread),
                mock.patch.object(
                    search,
                    "process_hits_to_file",
                    side_effect=fail_current_output,
                ),
            ):
                try:
                    with self.assertRaisesRegex(
                        ValueError, "current write failed"
                    ):
                        search.hmmsearch(
                            {}, pairs, 2, search_options(temporary),
                            all_sequences=[object()],
                            gpu_sequence_batch=Batch(),
                            gpu_postfilter=True,
                            gpu_profile_session=Session(),
                        )
                finally:
                    outer_resource_closed.set()

        self.assertTrue(second_generation_finished.is_set())
        self.assertEqual(len(InterruptingJoinThread.instances), 1)
        producer = InterruptingJoinThread.instances[0]
        self.assertIs(producer.premature_join, False)
        self.assertEqual(producer.join_calls, 2)
        self.assertFalse(producer.is_alive())
        self.assertEqual(
            [selection.close_count for selection in selections], [1, 1]
        )


class GPUPostfilterSelectionTests(unittest.TestCase):
    def test_domain_probe_requires_the_complete_adapter_and_all_three_seams(self):
        cases = (
            (True, True, True, True, True, True, True),
            (True, True, True, False, True, True, False),
            (True, True, True, True, False, True, False),
            (True, True, True, True, True, False, False),
            (True, True, False, True, True, True, False),
            (True, False, True, True, True, True, False),
            (False, True, True, True, True, True, False),
        )
        for (
            postfilter,
            forward,
            simple,
            adapter,
            method,
            native,
            expected,
        ) in cases:
            with self.subTest(
                postfilter=postfilter,
                forward=forward,
                simple=simple,
                adapter=adapter,
                method=method,
                native=native,
            ):
                modules, _ = synthetic_plan7_gpu(
                    postfilter,
                    forward,
                    simple,
                    domain_adapter_available=adapter,
                    domain_method_available=method,
                    domain_native_available=native,
                )
                with mock.patch.dict(sys.modules, modules):
                    self.assertIs(
                        search.gpu_profile_domain_available(), expected
                    )

    def test_compact_probe_requires_a_coherent_v2_journal_abi(self):
        complete_v2 = {
            "postfilter_available": True,
            "forward_available": True,
            "simple_available": True,
            "compact_adapter_available": True,
            "compact_native_available": True,
            "compact_seam_available": True,
        }
        cases = (
            ("guarded-v1", {
                "postfilter_available": True,
                "forward_available": True,
                "simple_available": True,
            }, True, False),
            ("compact-v2", complete_v2, True, True),
            ("v2-seam-absent", {
                **complete_v2, "compact_seam_available": False,
            }, True, False),
            ("v1-adapter-v2-journal", {
                **complete_v2, "compact_adapter_available": False,
            }, True, False),
            ("v2-adapter-v1-native", {
                **complete_v2, "compact_native_available": False,
            }, False, False),
            ("v2-adapter-native-v1-pipeline", {
                **complete_v2, "compact_seam_available": None,
            }, False, False),
            ("v2-probe-without-tail", {
                **complete_v2, "compact_tail_available": False,
            }, False, False),
            ("v2-tail-without-probe", {
                **complete_v2,
                "compact_seam_available": None,
                "compact_tail_available": True,
            }, False, False),
            ("non-bool-v2-probe", {
                **complete_v2, "compact_seam_available": 1,
            }, False, False),
            ("v1-journal-v2-pipeline", {
                "postfilter_available": True,
                "forward_available": True,
                "simple_available": True,
                "compact_seam_available": True,
            }, False, False),
            ("v2-journal-v1-pipeline", {
                "postfilter_available": True,
                "forward_available": True,
                "simple_available": True,
                "compact_native_available": True,
            }, False, False),
        )
        for label, arguments, domain_expected, compact_expected in cases:
            with self.subTest(label=label):
                modules, _ = synthetic_plan7_gpu(**arguments)
                with mock.patch.dict(sys.modules, modules):
                    self.assertIs(
                        search.gpu_profile_domain_available(),
                        domain_expected,
                    )
                    self.assertIs(
                        search.gpu_profile_compact_available(),
                        compact_expected,
                    )

    def test_preflight_selects_live_seam_and_safely_falls_back_when_absent(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-mode-") as temporary:
            root = Path(temporary)
            db_dir = root / "GPUDB"
            db_dir.mkdir()
            pressed_base = make_pressed_members(db_dir, "profiles")
            config = {
                "db_urls": [
                    {
                        "name": "GPUDB",
                        "installed": True,
                        "installation_dir": os.fspath(db_dir),
                        "molecule_type": "protein",
                    }
                ]
            }

            for available, forward_available, session_expected in (
                (True, True, True),
                (True, False, False),
                (False, False, False),
                (None, None, False),
            ):
                with self.subTest(
                    seam_available=available,
                    forward_available=forward_available,
                ):
                    modules, api = synthetic_plan7_gpu(
                        available, forward_available
                    )
                    pair = object()
                    batch = mock.Mock(name="sequence_batch")
                    batch.memory_snapshot = {"device_ordinal": 0}
                    api.load_pressed_profiles.return_value = (pair,)
                    api.SequenceBatch.return_value = batch
                    with mock.patch.dict(sys.modules, modules):
                        databases, observed_batch, postfilter = (
                            search.preflight_gpu_databases(
                                {"GPUDB": "manifest.json"},
                                ["GPUDB"],
                                config,
                                [object()],
                                1,
                            )
                        )

                    self.assertEqual(
                        databases,
                        {
                            "GPUDB": (
                                pressed_base.resolve(),
                                (pair,),
                                (
                                    api.ProfileSession.return_value
                                    if session_expected
                                    else None
                                ),
                            )
                        },
                    )
                    self.assertIs(observed_batch, batch)
                    self.assertIs(postfilter, bool(available))
                    if available is not None:
                        api.filter_scores_seam_available.assert_called_once_with()
                    if available:
                        api.forward_scores_seam_available.assert_called_once_with()
                    if session_expected:
                        api.ProfileSession.assert_called_once_with(
                            (pair,), build_workers=1, selection_workers=0
                        )
                    else:
                        api.ProfileSession.assert_not_called()

    def test_nonzero_cuda_device_retains_same_thread_forward_path(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-device-") as temporary:
            root = Path(temporary)
            db_dir = root / "GPUDB"
            db_dir.mkdir()
            pressed_base = make_pressed_members(db_dir, "profiles")
            config = {
                "db_urls": [{
                    "name": "GPUDB",
                    "installed": True,
                    "installation_dir": os.fspath(db_dir),
                    "molecule_type": "protein",
                }]
            }
            modules, api = synthetic_plan7_gpu(True)
            pair = object()
            batch = mock.Mock(name="sequence_batch")
            batch.memory_snapshot = {"device_ordinal": 2}
            api.load_pressed_profiles.return_value = (pair,)
            api.SequenceBatch.return_value = batch

            with mock.patch.dict(sys.modules, modules):
                databases, observed_batch, postfilter = (
                    search.preflight_gpu_databases(
                        {"GPUDB": "manifest.json"},
                        ["GPUDB"],
                        config,
                        [object()],
                        1,
                    )
                )

            self.assertEqual(
                databases,
                {"GPUDB": (pressed_base.resolve(), (pair,), None)},
            )
            self.assertIs(observed_batch, batch)
            self.assertTrue(postfilter)
            api.ProfileSession.assert_not_called()

    def test_session_path_splits_build_and_selection_workers(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-budget-") as temporary:
            root = Path(temporary)
            db_dir = root / "GPUDB"
            db_dir.mkdir()
            pressed_base = make_pressed_members(db_dir, "profiles")
            config = {
                "db_urls": [{
                    "name": "GPUDB",
                    "installed": True,
                    "installation_dir": os.fspath(db_dir),
                    "molecule_type": "protein",
                }]
            }
            modules, api = synthetic_plan7_gpu(True)
            pairs = (object(), object(), object())
            batch = mock.Mock(name="sequence_batch")
            batch.memory_snapshot = {"device_ordinal": 0}
            api.load_pressed_profiles.return_value = pairs
            api.SequenceBatch.return_value = batch

            with mock.patch.dict(sys.modules, modules):
                databases, _, postfilter = search.preflight_gpu_databases(
                    {"GPUDB": "manifest.json"},
                    ["GPUDB"],
                    config,
                    [object()],
                    2,
                )

            self.assertEqual(
                databases,
                {
                    "GPUDB": (
                        pressed_base.resolve(),
                        pairs,
                        api.ProfileSession.return_value,
                    )
                },
            )
            self.assertTrue(postfilter)
            api.ProfileSession.assert_called_once_with(
                pairs, build_workers=2, selection_workers=0
            )

    def test_profile_cache_reuses_one_attested_session_across_preflights(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-cache-") as temporary:
            root = Path(temporary)
            db_dir = root / "GPUDB"
            db_dir.mkdir()
            pressed_base = make_pressed_members(db_dir, "profiles")
            config = {
                "db_urls": [{
                    "name": "GPUDB",
                    "installed": True,
                    "installation_dir": os.fspath(db_dir),
                    "molecule_type": "protein",
                }]
            }
            modules, api = synthetic_plan7_gpu(True)
            pairs = (object(), object())
            underlying = mock.MagicMock(name="cached_profile_session")
            underlying.closed = False
            underlying.__len__.return_value = len(pairs)
            underlying.statistics = {
                "session_id": 41,
                "profile_count": len(pairs),
                "worker_count": 0,
                "build_worker_count": 2,
                "selection_worker_count": 0,
                "selection_count": 0,
                "host_bytes": 123,
            }
            api.load_pressed_profiles.return_value = pairs
            api.ProfileSession.return_value = underlying
            api.validate_pressed_manifest.return_value = SimpleNamespace(
                canonical_base=pressed_base.resolve(),
                manifest_sha256="a" * 64,
                stat_token=("stable",),
                model_count=len(pairs),
            )
            first_batch = mock.Mock(name="first_sequence_batch")
            first_batch.memory_snapshot = {"device_ordinal": 0}
            second_batch = mock.Mock(name="second_sequence_batch")
            second_batch.memory_snapshot = {"device_ordinal": 0}
            api.SequenceBatch.side_effect = [first_batch, second_batch]
            runtime = Plan7RuntimeIdentity(
                pyhmmer_version="test",
                pyhmmer_private_abi_sha256="1" * 64,
                adapter_sha256="2" * 64,
                native_extension_sha256="3" * 64,
                pipeline_extension_sha256="4" * 64,
            )
            cache = GPUProfileSessionCache(
                runtime_identity=runtime,
                validator=api.validate_pressed_manifest,
                loader=api.load_pressed_profiles,
                session_factory=api.ProfileSession,
            )

            observed = []
            with mock.patch.dict(sys.modules, modules):
                for expected_batch, expected_hit in (
                    (first_batch, False),
                    (second_batch, True),
                ):
                    metrics = {"GPUDB": search.GPUOverlapMetrics()}
                    databases, batch, postfilter = search.preflight_gpu_databases(
                        {"GPUDB": "manifest.json"},
                        ["GPUDB"],
                        config,
                        [object()],
                        2,
                        metrics,
                        cache,
                    )
                    _, observed_pairs, lease = databases["GPUDB"]
                    observed.append(lease)
                    self.assertIs(batch, expected_batch)
                    self.assertTrue(postfilter)
                    self.assertIs(observed_pairs, pairs)
                    self.assertIs(lease._entry.session, underlying)
                    self.assertIs(lease.reused, expected_hit)
                    self.assertTrue(metrics["GPUDB"].profile_cache_enabled)
                    self.assertIs(
                        metrics["GPUDB"].profile_cache_hit, expected_hit
                    )
                    self.assertEqual(metrics["GPUDB"].profile_session_id, 41)
                    self.assertEqual(
                        metrics["GPUDB"].profile_session_selection_count_start,
                        0,
                    )
                    if expected_hit:
                        self.assertEqual(
                            metrics["GPUDB"].profile_load_seconds, 0.0
                        )
                        self.assertEqual(
                            metrics["GPUDB"].session_build_seconds, 0.0
                        )
                    lease.close()
                    batch.close()

            self.assertEqual(api.validate_pressed_manifest.call_count, 4)
            api.load_pressed_profiles.assert_called_once_with(
                pressed_base.resolve(), manifest="manifest.json"
            )
            api.ProfileSession.assert_called_once_with(
                pairs, build_workers=2, selection_workers=0
            )
            self.assertIsNot(observed[0], observed[1])
            underlying.close.assert_not_called()
            cache.close()
            underlying.close.assert_called_once_with()

    def test_profile_cache_rejects_multi_database_preflight_before_batch(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-cache-multi-") as temporary:
            root = Path(temporary)
            config = {"db_urls": []}
            mappings = {}
            for name in ("GPU1", "GPU2"):
                db_dir = root / name
                db_dir.mkdir()
                make_pressed_members(db_dir, name)
                config["db_urls"].append({
                    "name": name,
                    "installed": True,
                    "installation_dir": os.fspath(db_dir),
                    "molecule_type": "protein",
                })
                mappings[name] = f"{name}.json"
            modules, api = synthetic_plan7_gpu(True)
            runtime = Plan7RuntimeIdentity(
                pyhmmer_version="test",
                pyhmmer_private_abi_sha256="1" * 64,
                adapter_sha256="2" * 64,
                native_extension_sha256="3" * 64,
                pipeline_extension_sha256="4" * 64,
            )
            cache = GPUProfileSessionCache(
                runtime_identity=runtime,
                validator=mock.Mock(),
                loader=mock.Mock(),
                session_factory=mock.Mock(),
            )
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(
                    search.GPUConfigurationError, "exactly one mapped database"
                ):
                    search.preflight_gpu_databases(
                        mappings,
                        ["GPU1", "GPU2"],
                        config,
                        [object()],
                        2,
                        profile_session_cache=cache,
                    )
            api.SequenceBatch.assert_not_called()
            cache.close()

    def test_later_session_failure_closes_prior_session_and_target_batch(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-session-") as temporary:
            root = Path(temporary)
            config = {"db_urls": []}
            for name in ("GPU1", "GPU2"):
                db_dir = root / name
                db_dir.mkdir()
                make_pressed_members(db_dir, name)
                config["db_urls"].append({
                    "name": name,
                    "installed": True,
                    "installation_dir": os.fspath(db_dir),
                    "molecule_type": "protein",
                })
            modules, api = synthetic_plan7_gpu(True)
            batch = mock.Mock(name="sequence_batch")
            batch.memory_snapshot = {"device_ordinal": 0}
            first_session = mock.Mock(name="first_session")
            first_pair = object()
            second_pair = object()
            api.SequenceBatch.return_value = batch
            api.load_pressed_profiles.side_effect = [
                (first_pair,),
                (second_pair,),
            ]
            api.ProfileSession.side_effect = [
                first_session,
                RuntimeError("session build failed"),
            ]

            with mock.patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "session build failed"):
                    search.preflight_gpu_databases(
                        {"GPU1": "one.json", "GPU2": "two.json"},
                        ["GPU1", "GPU2"],
                        config,
                        [object()],
                        1,
                    )

            first_session.close.assert_called_once_with()
            batch.close.assert_called_once_with()
            self.assertEqual(
                api.ProfileSession.call_args_list,
                [
                    mock.call(
                        (first_pair,), build_workers=1, selection_workers=0
                    ),
                    mock.call(
                        (second_pair,), build_workers=1, selection_workers=0
                    ),
                ],
            )


class InstalledGPUSelectionTests(unittest.TestCase):
    def test_cached_lease_excludes_until_batch_close_on_success_and_failure(self):
        for combine_failure, batch_close_failure in (
            (False, False),
            (True, False),
            (False, True),
        ):
            with self.subTest(
                combine_failure=combine_failure,
                batch_close_failure=batch_close_failure,
            ), tempfile.TemporaryDirectory(
                prefix="astra-cache-request-lifetime-"
            ) as temporary:
                root = Path(temporary)
                gpu_dir = root / "GPUDB"
                gpu_dir.mkdir()
                pressed_base = make_pressed_members(gpu_dir, "profiles")
                config = {
                    "db_urls": [{
                        "name": "GPUDB",
                        "installed": True,
                        "installation_dir": os.fspath(gpu_dir),
                        "molecule_type": "protein",
                    }]
                }
                args = search_args(
                    root / "out",
                    installed_hmms="GPUDB",
                    gpu_manifest=["GPUDB=manifest.json"],
                    threads=2,
                )
                modules, api = synthetic_plan7_gpu(True)
                pairs = (object(), object())
                underlying = mock.MagicMock(name="cached_profile_session")
                underlying.closed = False
                underlying.__len__.return_value = len(pairs)
                underlying.statistics = {
                    "session_id": 17,
                    "selection_count": 0,
                }
                api.ProfileSession.return_value = underlying
                api.load_pressed_profiles.return_value = pairs
                api.validate_pressed_manifest.return_value = SimpleNamespace(
                    canonical_base=pressed_base.resolve(),
                    manifest_sha256="a" * 64,
                    stat_token=("stable",),
                    model_count=len(pairs),
                )
                batch = mock.Mock(name="sequence_batch")
                batch.memory_snapshot = {"device_ordinal": 0}
                api.SequenceBatch.return_value = batch
                cache = GPUProfileSessionCache(
                    runtime_identity=Plan7RuntimeIdentity(
                        pyhmmer_version="test",
                        pyhmmer_private_abi_sha256="1" * 64,
                        adapter_sha256="2" * 64,
                        native_extension_sha256="3" * 64,
                        pipeline_extension_sha256="4" * 64,
                    ),
                    validator=api.validate_pressed_manifest,
                    loader=api.load_pressed_profiles,
                    session_factory=api.ProfileSession,
                )
                events = []

                def close_batch():
                    events.append("batch-close")
                    if batch_close_failure:
                        raise RuntimeError("batch close failed")

                batch.close.side_effect = close_batch
                original_release = cache._release

                def release_after_batch(entry):
                    self.assertTrue(batch.close.called)
                    events.append("lease-release")
                    return original_release(entry)

                def combine_while_exclusive(*_args):
                    self.assertFalse(batch.close.called)
                    events.append("combine")
                    with self.assertRaises(GPUProfileCacheBusyError):
                        cache.reserve()
                    if combine_failure:
                        raise RuntimeError("combine failed")

                with (
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(
                        search.initialize, "load_config", return_value=config
                    ),
                    mock.patch.object(
                        search,
                        "parse_protein_input",
                        return_value={"proteins.faa": [object()]},
                    ),
                    mock.patch.object(
                        search, "hmmsearch", return_value=root / "tmp"
                    ),
                    mock.patch.object(
                        search, "combine_results", new=combine_while_exclusive
                    ),
                    mock.patch.object(search, "cleanup_temp_files"),
                    mock.patch.object(
                        cache, "_release", side_effect=release_after_batch
                    ),
                ):
                    if combine_failure:
                        with self.assertRaisesRegex(
                            RuntimeError, "combine failed"
                        ):
                            search.main(
                                args, gpu_profile_session_cache=cache
                            )
                    elif batch_close_failure:
                        with self.assertRaisesRegex(
                            RuntimeError, "batch close failed"
                        ):
                            search.main(
                                args, gpu_profile_session_cache=cache
                            )
                    else:
                        search.main(args, gpu_profile_session_cache=cache)

                self.assertEqual(
                    events, ["combine", "batch-close", "lease-release"]
                )
                if batch_close_failure:
                    self.assertTrue(cache.closed)
                    self.assertFalse(cache.resident)
                    with self.assertRaisesRegex(
                        RuntimeError, "cache is closed"
                    ):
                        cache.reserve()
                    underlying.close.assert_called_once_with()
                else:
                    next_request = cache.reserve()
                    next_request.close()
                    self.assertTrue(cache.resident)
                    underlying.close.assert_not_called()
                    cache.close()
                    underlying.close.assert_called_once_with()

    def test_only_mapped_installed_databases_use_one_reused_gpu_batch(self):
        with tempfile.TemporaryDirectory(prefix="astra-selection-") as temporary:
            root = Path(temporary)
            cpu_dir = root / "CPUDB"
            gpu_one_dir = root / "GPU1"
            gpu_two_dir = root / "GPU2"
            for directory in (cpu_dir, gpu_one_dir, gpu_two_dir):
                directory.mkdir()
            gpu_one_base = make_pressed_members(gpu_one_dir, "profiles-one.hmm")
            gpu_two_base = make_pressed_members(gpu_two_dir, "profiles-two")
            config = {
                "db_urls": [
                    {
                        "name": name,
                        "installed": True,
                        "installation_dir": os.fspath(directory),
                        "molecule_type": "protein",
                    }
                    for name, directory in (
                        ("CPUDB", cpu_dir),
                        ("GPU1", gpu_one_dir),
                        ("GPU2", gpu_two_dir),
                    )
                ]
            }
            args = search_args(
                root / "out",
                hmm_in="user.hmm",
                installed_hmms="CPUDB,GPU1,GPU2",
                gpu_manifest=["GPU1=one.json", "GPU2=two.json"],
            )
            fake_batch = mock.Mock(name="shared_sequence_batch")
            fake_batch.memory_snapshot = {"device_ordinal": 0}
            pair_one = object()
            pair_two = object()
            modules, api = synthetic_plan7_gpu(True)
            api.SequenceBatch.return_value = fake_batch
            api.load_pressed_profiles.side_effect = [(pair_one,), (pair_two,)]

            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object()]},
                ),
                mock.patch.object(
                    search,
                    "parse_hmms",
                    side_effect=lambda path: ([f"cpu:{path}"], {}),
                ) as parse_hmms,
                mock.patch.object(
                    search, "hmmsearch", return_value=root / "tmp"
                ) as run,
                mock.patch.object(search, "combine_results"),
                mock.patch.object(search, "cleanup_temp_files"),
            ):
                search.main(args)

            self.assertEqual(
                [call.args[0] for call in parse_hmms.call_args_list],
                ["user.hmm", os.fspath(cpu_dir)],
            )
            api.SequenceBatch.assert_called_once()
            fake_batch.close.assert_called_once_with()
            self.assertEqual(
                api.validate_pressed_manifest.call_args_list,
                [
                    mock.call(gpu_one_base.resolve(), "one.json"),
                    mock.call(gpu_two_base.resolve(), "two.json"),
                ],
            )
            self.assertEqual(
                api.load_pressed_profiles.call_args_list,
                [
                    mock.call(gpu_one_base.resolve(), manifest="one.json"),
                    mock.call(gpu_two_base.resolve(), manifest="two.json"),
                ],
            )
            self.assertEqual(len(run.call_args_list), 4)
            self.assertNotIn("gpu_sequence_batch", run.call_args_list[0].kwargs)
            self.assertNotIn("gpu_sequence_batch", run.call_args_list[1].kwargs)
            self.assertIs(
                run.call_args_list[2].kwargs["gpu_sequence_batch"], fake_batch
            )
            self.assertIs(run.call_args_list[2].kwargs["gpu_postfilter"], True)
            self.assertIs(
                run.call_args_list[3].kwargs["gpu_sequence_batch"], fake_batch
            )
            self.assertIs(run.call_args_list[3].kwargs["gpu_postfilter"], True)
            api.filter_scores_seam_available.assert_called_once_with()

    def test_explicit_gpu_error_propagates_without_cpu_fallback_and_closes_batch(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-error-") as temporary:
            root = Path(temporary)
            gpu_dir = root / "GPUDB"
            gpu_dir.mkdir()
            make_pressed_members(gpu_dir, "only")
            config = {
                "db_urls": [
                    {
                        "name": "GPUDB",
                        "installed": True,
                        "installation_dir": os.fspath(gpu_dir),
                        "molecule_type": "protein",
                    }
                ]
            }
            args = search_args(
                root / "out",
                installed_hmms="GPUDB",
                gpu_manifest=["GPUDB=manifest.json"],
            )
            fake_batch = mock.Mock(name="sequence_batch")
            modules, api = synthetic_plan7_gpu()
            api.SequenceBatch.return_value = fake_batch
            api.load_pressed_profiles.return_value = (object(),)

            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object()]},
                ),
                mock.patch.object(search, "parse_hmms") as cpu_parser,
                mock.patch.object(
                    search, "hmmsearch", side_effect=RuntimeError("GPU failed")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                    search.main(args)

            cpu_parser.assert_not_called()
            fake_batch.close.assert_called_once_with()

    def test_presearch_failure_closes_popped_profile_session(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-presearch-") as temporary:
            root = Path(temporary)
            gpu_dir = root / "GPUDB"
            gpu_dir.mkdir()
            make_pressed_members(gpu_dir, "only")
            config = {
                "db_urls": [{
                    "name": "GPUDB",
                    "installed": True,
                    "installation_dir": os.fspath(gpu_dir),
                    "molecule_type": "protein",
                }]
            }
            args = search_args(
                root / "out",
                installed_hmms="GPUDB",
                gpu_manifest=["GPUDB=manifest.json"],
            )
            modules, api = synthetic_plan7_gpu(True)
            batch = mock.Mock(name="sequence_batch")
            batch.memory_snapshot = {"device_ordinal": 0}
            session = mock.Mock(name="profile_session")
            api.SequenceBatch.return_value = batch
            api.ProfileSession.return_value = session
            api.load_pressed_profiles.return_value = (object(),)

            def log(message):
                if str(message).startswith("GPU search for GPUDB"):
                    raise BrokenPipeError("log sink failed")

            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object()]},
                ),
                mock.patch.object(search.logging, "info", side_effect=log),
                mock.patch.object(search, "hmmsearch") as run_search,
            ):
                with self.assertRaisesRegex(BrokenPipeError, "log sink failed"):
                    search.main(args)

            run_search.assert_not_called()
            session.close.assert_called_once_with()
            batch.close.assert_called_once_with()

    def test_consumed_profile_tuple_is_released_before_next_database(self):
        class Profile:
            pass

        with tempfile.TemporaryDirectory(prefix="astra-gpu-lifetime-") as temporary:
            root = Path(temporary)
            config = {"db_urls": []}
            for name in ("GPU1", "GPU2"):
                directory = root / name
                directory.mkdir()
                make_pressed_members(directory, name)
                config["db_urls"].append(
                    {
                        "name": name,
                        "installed": True,
                        "installation_dir": os.fspath(directory),
                        "molecule_type": "protein",
                    }
                )
            args = search_args(
                root / "out",
                installed_hmms="GPU1,GPU2",
                gpu_manifest=["GPU1=one.json", "GPU2=two.json"],
            )
            modules, api = synthetic_plan7_gpu()
            api.SequenceBatch.return_value = mock.Mock(name="sequence_batch")
            profile_refs = []
            alive_at_search = {}

            def load_profiles(*_args, **_kwargs):
                profile = Profile()
                profile_refs.append(weakref.ref(profile))
                return (profile,)

            def run_search(
                _protein_dict, _hmms, _threads, _options, db_name, **_kwargs
            ):
                gc.collect()
                alive_at_search[db_name] = tuple(
                    reference() is not None for reference in profile_refs
                )
                return root / "tmp"

            api.load_pressed_profiles.side_effect = load_profiles
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object()]},
                ),
                mock.patch.object(search, "hmmsearch", new=run_search),
                mock.patch.object(search, "combine_results"),
                mock.patch.object(search, "cleanup_temp_files"),
            ):
                search.main(args)

            self.assertEqual(alive_at_search["GPU1"], (True, True))
            self.assertEqual(alive_at_search["GPU2"], (False, True))

    def test_second_profile_load_fails_before_first_search_or_publish(self):
        with tempfile.TemporaryDirectory(prefix="astra-gpu-preload-") as temporary:
            root = Path(temporary)
            config = {"db_urls": []}
            bases = {}
            for name in ("GPU1", "GPU2"):
                directory = root / name
                directory.mkdir()
                bases[name] = make_pressed_members(directory, name)
                config["db_urls"].append(
                    {
                        "name": name,
                        "installed": True,
                        "installation_dir": os.fspath(directory),
                        "molecule_type": "protein",
                    }
                )
            outdir = root / "out"
            args = search_args(
                outdir,
                hmm_in="user.hmm",
                installed_hmms="GPU1,GPU2",
                gpu_manifest=["GPU1=one.json", "GPU2=two.json"],
            )
            modules, api = synthetic_plan7_gpu()
            api.load_pressed_profiles.side_effect = [
                (object(),),
                RuntimeError("second profile load failed"),
            ]

            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object()]},
                ),
                mock.patch.object(search, "parse_hmms") as cpu_parser,
                mock.patch.object(search, "hmmsearch") as run_search,
                mock.patch.object(search, "combine_results") as publish,
            ):
                with self.assertRaisesRegex(RuntimeError, "second profile load failed"):
                    search.main(args)

            self.assertEqual(
                api.validate_pressed_manifest.call_args_list,
                [
                    mock.call(bases["GPU1"].resolve(), "one.json"),
                    mock.call(bases["GPU2"].resolve(), "two.json"),
                ],
            )
            self.assertEqual(
                api.load_pressed_profiles.call_args_list,
                [
                    mock.call(bases["GPU1"].resolve(), manifest="one.json"),
                    mock.call(bases["GPU2"].resolve(), manifest="two.json"),
                ],
            )
            api.SequenceBatch.assert_not_called()
            cpu_parser.assert_not_called()
            run_search.assert_not_called()
            publish.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_manifest_error_propagates_without_batch_creation_or_cpu_fallback(self):
        with tempfile.TemporaryDirectory(prefix="astra-manifest-error-") as temporary:
            root = Path(temporary)
            gpu_dir = root / "GPUDB"
            gpu_dir.mkdir()
            make_pressed_members(gpu_dir, "only")
            config = {
                "db_urls": [
                    {
                        "name": "GPUDB",
                        "installed": True,
                        "installation_dir": os.fspath(gpu_dir),
                        "molecule_type": "protein",
                    }
                ]
            }
            args = search_args(
                root / "out",
                hmm_in="user.hmm",
                installed_hmms="GPUDB",
                gpu_manifest=["GPUDB=manifest.json"],
            )
            modules, api = synthetic_plan7_gpu()
            api.validate_pressed_manifest.side_effect = ValueError("manifest invalid")

            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(
                    search.initialize, "load_config", return_value=config
                ),
                mock.patch.object(
                    search,
                    "parse_protein_input",
                    return_value={"proteins.faa": [object()]},
                ),
                mock.patch.object(search, "parse_hmms") as cpu_parser,
                mock.patch.object(search, "hmmsearch") as run_search,
            ):
                with self.assertRaisesRegex(ValueError, "manifest invalid"):
                    search.main(args)

            cpu_parser.assert_not_called()
            api.SequenceBatch.assert_not_called()
            api.load_pressed_profiles.assert_not_called()
            run_search.assert_not_called()


def gpu_available():
    try:
        from plan7_gpu import _native

        return _native.device_count() > 0
    except (ImportError, RuntimeError):
        return False


@unittest.skipUnless(gpu_available(), "plan7_gpu CUDA backend unavailable")
class GPUParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plan7_gpu import load_pressed_profiles
        from plan7_gpu.pressed_manifest import create_pressed_manifest

        cls.temporary = tempfile.TemporaryDirectory(prefix="astra-gpu-parity-")
        root = Path(cls.temporary.name)
        cls.alphabet = pyhmmer.easel.Alphabet.amino()
        background = pyhmmer.plan7.Background(cls.alphabet)
        builder = pyhmmer.plan7.Builder(cls.alphabet, seed=19)
        patterns = (
            "ACDEFGHIKLMNPQRSTVWY" * 4,
            "CDEFGHIKLMNPQRSTVWYA" * 4,
            "DEFGHIKLMNPQRSTVWYAC" * 4,
            "EFGHIKLMNPQRSTVWYACD" * 4,
        )
        hmms = []
        targets = []
        for index, pattern in enumerate(patterns):
            sequence = pyhmmer.easel.TextSequence(
                name=f"target-{index}".encode(), sequence=pattern
            ).digitize(cls.alphabet)
            hmm, _, _ = builder.build(sequence, background)
            hmm.name = f"model-{index}".encode()
            if index in (0, 3):
                hmm.cutoffs.gathering = (0.0, 0.0)
            elif index == 2:
                hmm.cutoffs.trusted = (0.0, 0.0)
            # index 1 deliberately has no model-specific cutoff.
            hmms.append(hmm)
            targets.append(sequence)
        targets.append(
            pyhmmer.easel.TextSequence(
                name=b"decoy", sequence="YYYYVVVVAAAAGGGGSSSSTTTT" * 3
            ).digitize(cls.alphabet)
        )
        cls.targets = pyhmmer.easel.DigitalSequenceBlock(cls.alphabet, targets)

        cls.pressed_base = root / "mixed-cutoffs.hmm"
        pyhmmer.hmmer.hmmpress(hmms, cls.pressed_base)
        cls.manifest = root / "mixed-cutoffs.manifest.json"
        create_pressed_manifest(cls.pressed_base, cls.manifest)
        with pyhmmer.plan7.HMMFile(cls.pressed_base) as hmm_file:
            cls.cpu_hmms = tuple(hmm_file)
        cls.gpu_pairs = load_pressed_profiles(cls.pressed_base, manifest=cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_bulk_tsv_is_byte_exact_for_mixed_cutoff_groups_threads_1_and_2(self):
        from plan7_gpu import SequenceBatch

        root = Path(self.temporary.name)
        for threads in (1, 2):
            with self.subTest(threads=threads):
                cpu_out = root / f"cpu-{threads}"
                gpu_out = root / f"gpu-{threads}"
                search.hmmsearch(
                    {},
                    self.cpu_hmms,
                    threads,
                    search_options(cpu_out, cascade=True),
                    all_sequences=self.targets,
                )
                with SequenceBatch(self.targets) as batch:
                    search.hmmsearch(
                        {},
                        self.gpu_pairs,
                        threads,
                        search_options(gpu_out, cascade=True),
                        all_sequences=self.targets,
                        gpu_sequence_batch=batch,
                    )
                cpu_tsv = cpu_out / "tmp_results" / "bulk_results.tsv"
                gpu_tsv = gpu_out / "tmp_results" / "bulk_results.tsv"
                self.assertEqual(gpu_tsv.read_bytes(), cpu_tsv.read_bytes())

    def test_profile_session_pipeline_is_byte_exact_across_ordered_chunks(self):
        if not search.gpu_postfilter_available():
            self.skipTest("private post-filter continuation seam unavailable")

        from plan7_gpu import ProfileSession, SequenceBatch, _pipeline
        from plan7_gpu.adapter import _candidate_state, _sequence_native

        forward_probe = getattr(
            _pipeline, "_filter_and_forward_scores_seam_available", None
        )
        if not callable(forward_probe) or not forward_probe():
            self.skipTest("private Forward continuation seam unavailable")
        domain_available = search.gpu_profile_domain_available()

        class RecordingSession:
            def __init__(self, session):
                self.session = session
                self.selections = []

            def __len__(self):
                return len(self.session)

            @property
            def closed(self):
                return self.session.closed

            @property
            def statistics(self):
                return self.session.statistics

            def select(self, indices):
                self.selections.append(tuple(indices))
                return self.session.select(indices)

        root = Path(self.temporary.name)
        expected_selections = [(0,), (3,), (1,), (2,)]
        generation_options = {
            "preferred_cutoff": "trusted",
            "F1": 0.99,
            "F2": 1.0,
            "F3": 1.0,
            "bias_filter": True,
        }
        for threads in (1, 2):
            with self.subTest(threads=threads):
                cpu_out = root / f"session-cpu-{threads}"
                gpu_out = root / f"session-gpu-{threads}"
                generated = []
                original_generate = (
                    SequenceBatch._postfilter_forward_selection
                )

                def record_generate(
                    batch, selection, F1, F2, F3, bias_filter, **domain_options
                ):
                    candidates = original_generate(
                        batch,
                        selection,
                        F1,
                        F2,
                        F3,
                        bias_filter,
                        **domain_options,
                    )
                    self.assertIs(
                        "pipeline" in domain_options, domain_available
                    )
                    state = _candidate_state(candidates)
                    self.assertIsNotNone(state.sealed_postfilter)
                    self.assertIsNone(state.forward)
                    generated.append(
                        (selection.indices, F1, F2, F3, bias_filter)
                    )
                    return candidates

                record_generate.__signature__ = inspect.signature(
                    original_generate
                )

                with mock.patch.object(
                    search, "define_kwargs", return_value=generation_options
                ):
                    search.hmmsearch(
                        {},
                        self.cpu_hmms,
                        threads,
                        search_options(cpu_out, cascade=True),
                        all_sequences=self.targets,
                    )
                metrics = search.GPUOverlapMetrics()
                with ProfileSession(
                    self.gpu_pairs,
                    build_workers=threads,
                    selection_workers=0,
                ) as raw_session:
                    session = RecordingSession(raw_session)
                    with SequenceBatch(self.targets) as batch:
                        with (
                            mock.patch.object(
                                search, "GPU_CELL_CAP", len(self.targets)
                            ),
                            mock.patch.object(
                                search,
                                "define_kwargs",
                                return_value=generation_options,
                            ),
                            mock.patch.object(
                                SequenceBatch,
                                "_postfilter_forward_selection",
                                new=record_generate,
                            ),
                        ):
                            search.hmmsearch(
                                {},
                                self.gpu_pairs,
                                threads,
                                search_options(gpu_out, cascade=True),
                                all_sequences=self.targets,
                                gpu_sequence_batch=batch,
                                gpu_postfilter=True,
                                gpu_profile_session=session,
                                gpu_metrics=metrics,
                            )
                        self.assertGreater(
                            _sequence_native(batch).workspace_statistics[
                                "forward_run_count"
                            ],
                            0,
                        )
                self.assertEqual(session.selections, expected_selections)
                self.assertEqual(
                    [call[0] for call in generated], expected_selections
                )
                self.assertTrue(all(
                    call[1:] == (0.99, 1.0, 1.0, True)
                    for call in generated
                ))
                snapshot = metrics.snapshot()
                self.assertEqual(snapshot["profile_worker_count"], 0)
                self.assertEqual(
                    snapshot["profile_build_worker_count"], threads
                )
                self.assertEqual(
                    snapshot["profile_selection_worker_count"], 0
                )
                self.assertEqual(snapshot["requested_thread_count"], threads)
                self.assertEqual(
                    snapshot["continuation_worker_count"], 1
                )
                self.assertEqual(
                    snapshot["producer_slot_count"], int(threads == 2)
                )
                self.assertIs(
                    snapshot["profile_overlap_enabled"], threads == 2
                )
                cpu_tsv = cpu_out / "tmp_results" / "bulk_results.tsv"
                gpu_tsv = gpu_out / "tmp_results" / "bulk_results.tsv"
                self.assertEqual(gpu_tsv.read_bytes(), cpu_tsv.read_bytes())

    def test_profile_session_domain_journal_has_both_routes_and_exact_counters(self):
        if not search.gpu_profile_domain_available():
            self.skipTest("opaque simple-region continuation ABI unavailable")
        compact_available = search.gpu_profile_compact_available()

        from plan7_gpu import ProfileSession, SequenceBatch, _pipeline
        from plan7_gpu import load_pressed_profiles
        from plan7_gpu.adapter import _candidate_state
        from plan7_gpu.pressed_manifest import create_pressed_manifest

        data = (
            Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
        )
        names = ("RREFam.hmm", "LuxC.hmm", "Thioesterase.hmm")
        if not all((data / name).is_file() for name in names):
            self.skipTest("PyHMMER HMM test fixtures unavailable")

        root = Path(self.temporary.name)
        base = root / "domain-routes.hmm"
        hmms = []
        for name in names:
            with pyhmmer.plan7.HMMFile(data / name) as hmm_file:
                hmms.append(hmm_file.read())
        pyhmmer.hmmer.hmmpress(hmms, base)
        manifest = root / "domain-routes.manifest.json"
        create_pressed_manifest(base, manifest)
        with pyhmmer.plan7.HMMFile(base) as hmm_file:
            cpu_hmms = tuple(hmm_file)
        pairs = load_pressed_profiles(base, manifest=manifest)

        target_sequences = [
            pyhmmer.easel.TextSequence(
                name=b"mixed",
                sequence="ACDEFGHIKLMNPQRSTVWY" * 5,
            ).digitize(self.alphabet),
            pyhmmer.easel.TextSequence(
                name=b"low-complexity", sequence="A" * 91
            ).digitize(self.alphabet),
            pyhmmer.easel.TextSequence(
                name=b"short", sequence="MTEYKLVVVGAGGVGKSALTIQLIQ"
            ).digitize(self.alphabet),
        ]
        target_sequences.extend(
            pyhmmer.easel.TextSequence(
                name=f"consensus-{index}".encode(),
                sequence=hmm.consensus,
            ).digitize(self.alphabet)
            for index, hmm in enumerate(hmms)
        )
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, target_sequences
        )
        options = {
            "F1": 0.99,
            "F2": 1.0,
            "F3": 1.0,
            "bias_filter": True,
            "E": 10.0,
            "domE": 10.0,
            "incE": 10.0,
            "incdomE": 10.0,
        }

        def semantic_state(hits):
            pipeline_fields = (
                "Z_setby",
                "domZ_setby",
                "n_past_msv",
                "n_past_bias",
                "n_past_vit",
                "n_past_fwd",
                "pos_past_msv",
                "pos_past_bias",
                "pos_past_vit",
                "pos_past_fwd",
                "mode",
                "W",
            )
            top_fields = (
                "Z",
                "domZ",
                "searched_models",
                "searched_nodes",
                "searched_residues",
                "searched_sequences",
            )
            pipeline = hits.__getstate__()["pipeline"]
            tables = io.BytesIO()
            hits.write(tables, format="targets", header=True)
            hits.write(tables, format="domains", header=True)
            return (
                tuple(pipeline[field] for field in pipeline_fields),
                tuple(getattr(hits, field) for field in top_fields),
                tables.getvalue(),
            )

        original_process = search.process_hits_to_file
        cpu_states = []
        gpu_states = []

        def record_cpu(hits, fh):
            cpu_states.append(semantic_state(hits))
            return original_process(hits, fh)

        def record_gpu(hits, fh):
            gpu_states.append(semantic_state(hits))
            return original_process(hits, fh)

        class RecordingSession:
            def __init__(self, session):
                self.session = session
                self.selections = []

            def __len__(self):
                return len(self.session)

            @property
            def closed(self):
                return self.session.closed

            @property
            def statistics(self):
                return self.session.statistics

            def select(self, indices):
                self.selections.append(tuple(indices))
                return self.session.select(indices)

        cpu_out = root / "domain-cpu"
        with (
            mock.patch.object(search, "define_kwargs", return_value=options),
            mock.patch.object(search, "process_hits_to_file", new=record_cpu),
        ):
            search.hmmsearch(
                {},
                cpu_hmms,
                2,
                search_options(cpu_out),
                all_sequences=targets,
            )

        route_statistics = []
        original_generate = SequenceBatch._postfilter_forward_selection

        def record_generate(batch, selection, F1, F2, F3, bias_filter,
                            *, pipeline=None, domain_guard=2.0e-4):
            self.assertIsNotNone(pipeline)
            candidates = original_generate(
                batch,
                selection,
                F1,
                F2,
                F3,
                bias_filter,
                pipeline=pipeline,
                domain_guard=domain_guard,
            )
            route_statistics.append(
                _pipeline._sealed_continuation_statistics_bound(
                    _candidate_state(candidates).sealed_postfilter
                )
            )
            return candidates

        record_generate.__signature__ = inspect.signature(original_generate)

        gpu_out = root / "domain-gpu"
        metrics = search.GPUOverlapMetrics()
        with ProfileSession(
            pairs, build_workers=2, selection_workers=0
        ) as raw_session:
            session = RecordingSession(raw_session)
            with SequenceBatch(targets) as batch:
                with (
                    mock.patch.object(search, "GPU_CELL_CAP", 12),
                    mock.patch.object(
                        search, "define_kwargs", return_value=options
                    ),
                    mock.patch.object(
                        search, "process_hits_to_file", new=record_gpu
                    ),
                    mock.patch.object(
                        SequenceBatch,
                        "_postfilter_forward_selection",
                        new=record_generate,
                    ),
                ):
                    search.hmmsearch(
                        {},
                        pairs,
                        2,
                        search_options(gpu_out),
                        all_sequences=targets,
                        gpu_sequence_batch=batch,
                        gpu_postfilter=True,
                        gpu_profile_session=session,
                        gpu_metrics=metrics,
                    )

        self.assertEqual(session.selections, [(0, 1), (2,)])
        totals = {
            key: sum(item[key] for item in route_statistics)
            for key in (
                "row_count",
                "cpu_required_count",
                "no_region_count",
                "simple_count",
            )
        }
        self.assertEqual(
            totals["row_count"],
            totals["cpu_required_count"]
            + totals["no_region_count"]
            + totals["simple_count"],
        )
        self.assertGreater(totals["cpu_required_count"], 0)
        self.assertGreater(totals["simple_count"], 0)
        if compact_available:
            self.assertTrue(all(
                item["compact_enabled"] for item in route_statistics
            ))
            self.assertGreater(sum(
                item["compact_simple_row_count"]
                for item in route_statistics
            ), 0)
            self.assertGreater(sum(
                item["compact_device_result_count"]
                for item in route_statistics
            ), 0)
        self.assertEqual(gpu_states, cpu_states)
        gpu_tsv = gpu_out / "tmp_results" / "bulk_results.tsv"
        cpu_tsv = cpu_out / "tmp_results" / "bulk_results.tsv"
        self.assertEqual(gpu_tsv.read_bytes(), cpu_tsv.read_bytes())
        data_rows = cpu_tsv.read_text().splitlines()[1:]
        self.assertGreater(len(data_rows), 0)
        self.assertTrue(all(len(row.split("\t")) == 13 for row in data_rows))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["generated_chunk_count"], 2)
        self.assertEqual(snapshot["consumed_chunk_count"], 2)
        self.assertEqual(snapshot["continuation_worker_count"], 1)
        self.assertTrue(snapshot["profile_overlap_enabled"])

    def test_profile_session_compact_threshold_fallback_is_byte_exact(self):
        if not search.gpu_profile_compact_available():
            self.skipTest("compact-domain continuation ABI unavailable")

        from plan7_gpu import ProfileSession, SequenceBatch, _pipeline
        from plan7_gpu import load_pressed_profiles
        from plan7_gpu.adapter import _candidate_state
        from plan7_gpu.pressed_manifest import create_pressed_manifest

        data = (
            Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
        )
        names = ("RREFam.hmm", "Thioesterase.hmm")
        if not all((data / name).is_file() for name in names):
            self.skipTest("PyHMMER compact-domain fixtures unavailable")

        root = Path(self.temporary.name)
        base = root / "compact-threshold.hmm"
        hmms = []
        for name in names:
            with pyhmmer.plan7.HMMFile(data / name) as hmm_file:
                hmms.append(hmm_file.read())
        pyhmmer.hmmer.hmmpress(hmms, base)
        manifest = root / "compact-threshold.manifest.json"
        create_pressed_manifest(base, manifest)
        with pyhmmer.plan7.HMMFile(base) as hmm_file:
            pressed_hmms = tuple(hmm_file)
        pressed_pairs = load_pressed_profiles(base, manifest=manifest)

        consensus = hmms[1].consensus
        if isinstance(consensus, bytes):
            consensus = consensus.decode()
        consensus = consensus.replace("-", "")
        target = pyhmmer.easel.TextSequence(
            name=b"two-domain-probe",
            sequence=consensus + "X" * 100 + consensus,
        ).digitize(self.alphabet)
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [target]
        )
        base_options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        baseline = pyhmmer.plan7.Pipeline(
            self.alphabet, **base_options
        ).search_hmm(pressed_hmms[0], targets)
        self.assertEqual([hit.name for hit in baseline], ["two-domain-probe"])
        self.assertGreaterEqual(len(baseline[0].domains), 2)
        options = {
            **base_options,
            "bias_filter": True,
            "T": baseline[0].score,
            "incT": -1000.0,
            "incdomT": -1000.0,
        }

        def semantic_state(hits):
            pipeline_fields = (
                "Z_setby",
                "domZ_setby",
                "n_past_msv",
                "n_past_bias",
                "n_past_vit",
                "n_past_fwd",
                "pos_past_msv",
                "pos_past_bias",
                "pos_past_vit",
                "pos_past_fwd",
                "mode",
                "W",
            )
            top_fields = (
                "Z",
                "domZ",
                "searched_models",
                "searched_nodes",
                "searched_residues",
                "searched_sequences",
            )
            pipeline = hits.__getstate__()["pipeline"]
            tables = io.BytesIO()
            hits.write(tables, format="targets", header=True)
            hits.write(tables, format="domains", header=True)
            return (
                tuple(pipeline[field] for field in pipeline_fields),
                tuple(getattr(hits, field) for field in top_fields),
                tables.getvalue(),
            )

        original_process = search.process_hits_to_file
        cpu_states = []
        gpu_states = []

        def record_cpu(hits, fh):
            cpu_states.append(semantic_state(hits))
            return original_process(hits, fh)

        def record_gpu(hits, fh):
            gpu_states.append(semantic_state(hits))
            return original_process(hits, fh)

        cpu_out = root / "compact-threshold-cpu"
        with (
            mock.patch.object(search, "define_kwargs", return_value=options),
            mock.patch.object(search, "process_hits_to_file", new=record_cpu),
        ):
            search.hmmsearch(
                {},
                (pressed_hmms[0],),
                2,
                search_options(cpu_out),
                all_sequences=targets,
            )

        continuation_statistics = []
        original_generate = SequenceBatch._postfilter_forward_selection

        def record_generate(
            batch, selection, F1, F2, F3, bias_filter, **domain_options
        ):
            self.assertIsNotNone(domain_options.get("pipeline"))
            candidates = original_generate(
                batch,
                selection,
                F1,
                F2,
                F3,
                bias_filter,
                **domain_options,
            )
            continuation_statistics.append(
                _pipeline._sealed_continuation_statistics_bound(
                    _candidate_state(candidates).sealed_postfilter
                )
            )
            return candidates

        record_generate.__signature__ = inspect.signature(original_generate)

        gpu_out = root / "compact-threshold-gpu"
        metrics = search.GPUOverlapMetrics()
        with ProfileSession(
            (pressed_pairs[0],), build_workers=2, selection_workers=0
        ) as session:
            with SequenceBatch(targets) as batch:
                with (
                    mock.patch.object(
                        search, "define_kwargs", return_value=options
                    ),
                    mock.patch.object(
                        search, "process_hits_to_file", new=record_gpu
                    ),
                    mock.patch.object(
                        SequenceBatch,
                        "_postfilter_forward_selection",
                        new=record_generate,
                    ),
                ):
                    search.hmmsearch(
                        {},
                        (pressed_pairs[0],),
                        2,
                        search_options(gpu_out),
                        all_sequences=targets,
                        gpu_sequence_batch=batch,
                        gpu_postfilter=True,
                        gpu_profile_session=session,
                        gpu_metrics=metrics,
                    )

        self.assertEqual(len(continuation_statistics), 1)
        statistics = continuation_statistics[0]
        self.assertTrue(statistics["compact_enabled"])
        self.assertGreater(statistics["compact_device_result_count"], 0)
        self.assertEqual(gpu_states, cpu_states)
        self.assertEqual(len(cpu_states), 1)
        cpu_tsv = cpu_out / "tmp_results" / "bulk_results.tsv"
        gpu_tsv = gpu_out / "tmp_results" / "bulk_results.tsv"
        self.assertEqual(gpu_tsv.read_bytes(), cpu_tsv.read_bytes())
        data_rows = cpu_tsv.read_text().splitlines()[1:]
        self.assertGreater(len(data_rows), 0)
        self.assertTrue(all(len(row.split("\t")) == 13 for row in data_rows))
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["generated_chunk_count"], 1)
        self.assertEqual(snapshot["consumed_chunk_count"], 1)


if __name__ == "__main__":
    unittest.main()
