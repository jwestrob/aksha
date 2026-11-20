"""CPU scheduling strategy (--cpu-parallel): validation, CLI wiring and result equivalence."""

import csv
import importlib
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import astra_pyhmmer as h

from aksha import search
from tests.test_cpu_stream_profiles import _options

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class CPUParallelTests(unittest.TestCase):
    def test_strategy_validation(self):
        self.assertEqual(search.cpu_parallel_options({}), {})
        self.assertEqual(search.cpu_parallel_options({"cpu_parallel": "auto"}), {})
        self.assertEqual(search.cpu_parallel_options({"cpu_parallel": "targets"}), {"parallel": "targets"})
        with self.assertRaises(ValueError):
            search.cpu_parallel_options({"cpu_parallel": "invalid"})

    def test_queries_and_targets_give_identical_results(self):
        # Heterogeneous model lengths, a mix of profiles with and without GA cutoffs
        # (cascade grouping), and a target without hits.
        alphabet = h.easel.Alphabet.amino()
        background = h.plan7.Background(alphabet)
        builder = h.plan7.Builder(alphabet, seed=31)
        rng = random.Random(33)
        texts = ["".join(rng.choices(AMINO_ACIDS, k=n)) for n in (80, 110, 140, 210)]
        models, targets = [], []
        for i, text in enumerate(texts):
            sequence = h.easel.TextSequence(name=f"model-{i}".encode(), sequence=text).digitize(alphabet)
            model, _, _ = builder.build(sequence, background)
            if i < 3:
                model.cutoffs.gathering = (10.0, 10.0)
            models.append(model)
            for j in range(7):
                residues = list(text)
                for _ in range(j * 3):
                    residues[rng.randrange(len(residues))] = rng.choice(AMINO_ACIDS)
                targets.append(h.easel.TextSequence(
                    name=f"target-{i}-{j}".encode(), sequence="".join(residues)).digitize(alphabet))
        targets.append(h.easel.TextSequence(name=b"no-hit", sequence="W" * 100).digitize(alphabet))

        with tempfile.TemporaryDirectory() as tmp:
            results = {}
            for mode in ("queries", "targets"):
                out = Path(tmp) / mode
                options = _options(out)
                options.update(cascade=True, cpu_parallel=mode)
                search.hmmsearch({}, models, 4, options, all_sequences=targets)
                with (out / "tmp_results" / "bulk_results.tsv").open() as handle:
                    results[mode] = list(csv.DictReader(handle, delimiter="\t"))

        def key(row):
            return row["hmm_name"], row["sequence_id"], row["ali_from"], row["ali_to"]

        left, right = sorted(results["queries"], key=key), sorted(results["targets"], key=key)
        self.assertGreater(len(left), 20)
        self.assertEqual(left, right)  # full/domain scores, c/i E-values and coordinates

    def test_cli_passes_requested_strategy(self):
        main = importlib.import_module("aksha.main")
        argv = ["aksha", "search", "--prot_in", "input.faa", "--hmm_in", "profiles.hmm",
                "--outdir", "output", "--cpu-parallel", "targets"]
        with mock.patch("sys.argv", argv), mock.patch.object(main.search, "main") as call:
            main.main()
        self.assertEqual(call.call_args.args[0].cpu_parallel, "targets")

    def test_gpu_search_rejects_a_cpu_strategy_before_any_work(self):
        args = SimpleNamespace(hmm_in=None, prot_in="input.faa", outdir="output",
                               gpu_manifest=["DB=manifest.json"], cpu_parallel="targets")
        with mock.patch.object(search, "parse_gpu_manifest_mappings", return_value={"DB": "manifest.json"}):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                search.main(args)


if __name__ == "__main__":
    unittest.main()
