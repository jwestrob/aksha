import argparse
import builtins
import importlib
import os
import sys
import tempfile
import unittest
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


def synthetic_plan7_gpu():
    """Return optional-package modules suitable for CPU-only wiring tests."""
    package = ModuleType("plan7_gpu")
    package.__path__ = []
    astra_search_module = ModuleType("plan7_gpu.astra_search")
    manifest_module = ModuleType("plan7_gpu.pressed_manifest")

    api = SimpleNamespace(
        SequenceBatch=mock.Mock(name="SequenceBatch"),
        load_pressed_profiles=mock.Mock(name="load_pressed_profiles"),
        validate_pressed_manifest=mock.Mock(name="validate_pressed_manifest"),
        gpu_hmmsearch=mock.Mock(name="gpu_hmmsearch"),
    )
    package.SequenceBatch = api.SequenceBatch
    package.load_pressed_profiles = api.load_pressed_profiles
    package.astra_search = astra_search_module
    astra_search_module.hmmsearch = api.gpu_hmmsearch
    manifest_module.validate_pressed_manifest = api.validate_pressed_manifest
    return (
        {
            "plan7_gpu": package,
            "plan7_gpu.astra_search": astra_search_module,
            "plan7_gpu.pressed_manifest": manifest_module,
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
            pair_one = object()
            pair_two = object()
            modules, api = synthetic_plan7_gpu()
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
            self.assertIs(
                run.call_args_list[3].kwargs["gpu_sequence_batch"], fake_batch
            )

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


if __name__ == "__main__":
    unittest.main()
