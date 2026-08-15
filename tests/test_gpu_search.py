import argparse
import builtins
import gc
import importlib
import os
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pyhmmer

from astra import search

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


def synthetic_plan7_gpu(postfilter_available=None, forward_available=None):
    """Return optional-package modules suitable for CPU-only wiring tests."""
    package = ModuleType("plan7_gpu")
    package.__path__ = []
    astra_search_module = ModuleType("plan7_gpu.astra_search")
    manifest_module = ModuleType("plan7_gpu.pressed_manifest")
    pipeline_module = ModuleType("plan7_gpu._pipeline")

    default_batch = mock.Mock(name="sequence_batch")
    default_batch.memory_snapshot = {"device_ordinal": 0}
    api = SimpleNamespace(
        ProfileSession=mock.Mock(name="ProfileSession"),
        SequenceBatch=mock.Mock(name="SequenceBatch", return_value=default_batch),
        load_pressed_profiles=mock.Mock(name="load_pressed_profiles"),
        validate_pressed_manifest=mock.Mock(name="validate_pressed_manifest"),
        gpu_hmmsearch=mock.Mock(name="gpu_hmmsearch"),
        filter_scores_seam_available=None,
        forward_scores_seam_available=None,
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
    package.SequenceBatch = api.SequenceBatch
    package.ProfileSession = api.ProfileSession
    package.load_pressed_profiles = api.load_pressed_profiles
    package.astra_search = astra_search_module
    package._pipeline = pipeline_module
    astra_search_module.hmmsearch = api.gpu_hmmsearch
    manifest_module.validate_pressed_manifest = api.validate_pressed_manifest
    return (
        {
            "plan7_gpu": package,
            "plan7_gpu.astra_search": astra_search_module,
            "plan7_gpu.pressed_manifest": manifest_module,
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

    def test_two_slots_overlap_in_order_with_noncontiguous_selection(self):
        pairs = [self.pair("gathering"), self.pair(), self.pair("gathering")]
        consumption_started = threading.Event()
        second_generation_finished = threading.Event()
        selection_calls = []
        selections = []
        generation_threads = []
        generation_calls = []

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
                return len(pairs)

            def select(self, indices):
                self_outer.assertIs(
                    threading.current_thread(), threading.main_thread()
                )
                self_outer.assertFalse(consumption_started.is_set())
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
                    self_outer.assertTrue(consumption_started.wait(2))
                    time.sleep(0.03)
                    second_generation_finished.set()
                return SimpleNamespace(
                    indices=selection.indices,
                    F1=F1,
                    F2=F2,
                    F3=F3,
                    bias_filter=bias_filter,
                    sealed=True,
                )

        self_outer = self
        modules, api = synthetic_plan7_gpu(True)
        observed = []

        def gpu_search(chunk, candidates, **kwargs):
            def results():
                if candidates.indices == (0, 2):
                    consumption_started.set()
                    self.assertTrue(second_generation_finished.wait(2))
                for pair in chunk:
                    yield (candidates.indices, pair, kwargs)

            return results()

        api.gpu_hmmsearch.side_effect = gpu_search
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
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 4),
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

        self.assertEqual(selection_calls, [(0, 2), (1,)])
        self.assertTrue(all(selection.closed for selection in selections))
        self.assertTrue(all(selection.close_count == 1 for selection in selections))
        self.assertEqual(
            generation_calls,
            [
                ((0, 2), 0.03, 0.004, 0.00005, False),
                ((1,), 0.03, 0.004, 0.00005, False),
            ],
        )
        self.assertEqual([item[0] for item in observed], [(0, 2), (0, 2), (1,)])
        self.assertTrue(
            all(name.startswith("astra-gpu-generate") for name in generation_threads)
        )
        self.assertEqual(
            [call.args[0] for call in api.gpu_hmmsearch.call_args_list],
            [[pairs[0], pairs[2]], [pairs[1]]],
        )
        for call in api.gpu_hmmsearch.call_args_list:
            self.assertEqual(call.kwargs["F1"], 0.03)
            self.assertEqual(call.kwargs["F2"], 0.004)
            self.assertEqual(call.kwargs["F3"], 0.00005)
            self.assertIs(call.kwargs["bias_filter"], False)
            self.assertEqual(call.kwargs["cpus"], 2)
            self.assertIs(call.kwargs["postfilter"], True)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requested_thread_count"], 3)
        self.assertEqual(snapshot["profile_worker_count"], 0)
        self.assertEqual(snapshot["producer_slot_count"], 1)
        self.assertEqual(snapshot["continuation_worker_count"], 2)
        self.assertTrue(snapshot["profile_overlap_enabled"])
        self.assertEqual(snapshot["chunk_count"], 2)
        self.assertEqual(snapshot["generated_chunk_count"], 2)
        self.assertEqual(snapshot["consumed_chunk_count"], 2)
        self.assertGreater(snapshot["selection_seconds"], 0.0)
        self.assertGreater(snapshot["overlap_seconds"], 0.01)
        self.assertFalse(any(
            thread.name.startswith("astra-gpu-generate")
            for thread in threading.enumerate()
        ))

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

    def test_profile_pack_workers_cannot_exceed_cli_budget(self):
        pair = self.pair()

        class Session:
            closed = False
            statistics = {"worker_count": 2, "host_bytes": 1234}

            def __len__(self):
                return 1

        class Batch:
            def _postfilter_forward_selection(self, *_args):
                raise AssertionError("generation reached")

        with tempfile.TemporaryDirectory(prefix="astra-gpu-pack-budget-") as temporary:
            with self.assertRaisesRegex(
                search.GPUConfigurationError, "pack workers exceed"
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

    def test_cancelled_queued_generation_closes_its_selection_once(self):
        pairs = [self.pair(), self.pair()]
        selections = []
        queued_future = None

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
                return SimpleNamespace(indices=selection.indices)

        class ImmediateFuture:
            def __init__(self, function, args):
                self.function = function
                self.args = args

            def done(self):
                return False

            def result(self):
                return self.function(*self.args)

            def cancel(self):
                return False

        class QueuedFuture:
            cancelled = False

            def done(self):
                return False

            def cancel(self):
                self.cancelled = True
                return True

        class Executor:
            def __init__(self, **_kwargs):
                self.submit_count = 0

            def submit(self, function, *args):
                nonlocal queued_future
                self.submit_count += 1
                if self.submit_count == 1:
                    return ImmediateFuture(function, args)
                queued_future = QueuedFuture()
                return queued_future

            def shutdown(self, **_kwargs):
                pass

        modules, api = synthetic_plan7_gpu(True)
        api.gpu_hmmsearch.return_value = iter(("current-row",))
        with tempfile.TemporaryDirectory(prefix="astra-gpu-cancel-") as temporary:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(search, "GPU_CELL_CAP", 2),
                mock.patch.object(search, "ThreadPoolExecutor", Executor),
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

        self.assertTrue(queued_future.cancelled)
        self.assertEqual(
            [selection.close_count for selection in selections],
            [1, 1],
        )


class GPUPostfilterSelectionTests(unittest.TestCase):
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
                            (pair,), pack_workers=0
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

    def test_session_path_uses_zero_pack_workers_with_small_budget(self):
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
                pairs, pack_workers=0
            )

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
                    mock.call((first_pair,), pack_workers=0),
                    mock.call((second_pair,), pack_workers=0),
                ],
            )


class InstalledGPUSelectionTests(unittest.TestCase):
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
                    batch, selection, F1, F2, F3, bias_filter
                ):
                    candidates = original_generate(
                        batch, selection, F1, F2, F3, bias_filter
                    )
                    state = _candidate_state(candidates)
                    self.assertIsNotNone(state.sealed_postfilter)
                    self.assertIsNone(state.forward)
                    generated.append(
                        (selection.indices, F1, F2, F3, bias_filter)
                    )
                    return candidates

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
                    self.gpu_pairs, pack_workers=0
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


if __name__ == "__main__":
    unittest.main()
