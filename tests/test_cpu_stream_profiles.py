import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyhmmer

from astra import search


def _options(outdir):
    return {
        "cascade": False,
        "cut_ga": True,
        "cut_nc": False,
        "cut_tc": False,
        "evalue": None,
        "bitscore": None,
        "domE": 10.0,
        "domT": None,
        "incE": 10.0,
        "incT": None,
        "incdomE": 10.0,
        "incdomT": None,
        "outdir": os.fspath(outdir),
        "meta": False,
    }


class PressedHMMStreamTests(unittest.TestCase):
    def test_streamed_chunks_preserve_profile_and_output_order(self):
        with tempfile.TemporaryDirectory(prefix="astra-cpu-stream-") as tmp:
            root = Path(tmp)
            alphabet = pyhmmer.easel.Alphabet.amino()
            background = pyhmmer.plan7.Background(alphabet)
            builder = pyhmmer.plan7.Builder(alphabet, seed=31)
            hmms = []
            targets = []
            for index in range(5):
                pattern = "ACDEFGHIKLMNPQRSTVWY"[index:] + "ACDEFGHIKLMNPQRSTVWY"[:index]
                sequence = pyhmmer.easel.TextSequence(
                    name=f"target-{index}".encode(), sequence=pattern * 4
                ).digitize(alphabet)
                hmm, _, _ = builder.build(sequence, background)
                hmm.name = f"model-{index}".encode()
                hmm.cutoffs.gathering = (0.0, 0.0)
                hmms.append(hmm)
                targets.append(sequence)

            pressed = root / "profiles"
            pyhmmer.hmmer.hmmpress(hmms, pressed)
            stream = search._PressedHMMStream(pressed)
            self.assertTrue(stream.all_have_cutoff("gathering"))
            chunks = list(stream.chunks(2))
            self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
            self.assertEqual(
                [hmm.name for chunk in chunks for hmm in chunk],
                [hmm.name for hmm in hmms],
            )

            target_block = pyhmmer.easel.DigitalSequenceBlock(alphabet, targets)
            eager_dir = root / "eager"
            stream_dir = root / "stream"
            with pyhmmer.plan7.HMMFile(pressed) as hmm_file:
                eager_hmms = tuple(hmm_file)
            with mock.patch.object(search, "HMM_CHUNK_SIZE", 2):
                search.hmmsearch(
                    {}, eager_hmms, 2, _options(eager_dir),
                    all_sequences=target_block,
                )
                search.hmmsearch(
                    {}, stream, 2, _options(stream_dir),
                    all_sequences=target_block,
                )
            eager_tsv = eager_dir / "tmp_results" / "bulk_results.tsv"
            stream_tsv = stream_dir / "tmp_results" / "bulk_results.tsv"
            self.assertEqual(stream_tsv.read_bytes(), eager_tsv.read_bytes())

            # Mixed cutoff availability changes the eager path's global
            # profile grouping.  The stream must detect that case and fall
            # back before executing any profiles so ordering stays exact.
            hmms[-1].cutoffs.gathering = None
            mixed_pressed = root / "mixed-profiles"
            pyhmmer.hmmer.hmmpress(hmms, mixed_pressed)
            mixed_stream = search._PressedHMMStream(mixed_pressed)
            self.assertFalse(mixed_stream.all_have_cutoff("gathering"))
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(search, "HMM_CHUNK_SIZE", 2),
            ):
                selected, decision = search._select_pressed_cpu_stream(
                    mixed_pressed, _options(root / "mixed-selection")
                )
            self.assertIsNone(selected)
            self.assertEqual(decision.reason, "mixed-gathering-availability")
            mixed_eager_dir = root / "mixed-eager"
            mixed_stream_dir = root / "mixed-stream"
            with pyhmmer.plan7.HMMFile(mixed_pressed) as hmm_file:
                mixed_eager_hmms = tuple(hmm_file)
            with mock.patch.object(search, "HMM_CHUNK_SIZE", 2):
                search.hmmsearch(
                    {}, mixed_eager_hmms, 2, _options(mixed_eager_dir),
                    all_sequences=target_block,
                )
                search.hmmsearch(
                    {}, mixed_stream, 2, _options(mixed_stream_dir),
                    all_sequences=target_block,
                )
            mixed_eager_tsv = (
                mixed_eager_dir / "tmp_results" / "bulk_results.tsv"
            )
            mixed_stream_tsv = (
                mixed_stream_dir / "tmp_results" / "bulk_results.tsv"
            )
            self.assertEqual(
                mixed_stream_tsv.read_bytes(), mixed_eager_tsv.read_bytes()
            )

    def test_policy_is_auto_by_default_and_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(search._stream_pressed_cpu_policy(), "auto")
            self.assertTrue(search._stream_pressed_cpu_enabled())
        for value, enabled in (("auto", True), ("0", False), ("1", True)):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {search.CPU_STREAM_PRESSED_ENV: value}
            ):
                self.assertEqual(search._stream_pressed_cpu_enabled(), enabled)
        with mock.patch.dict(
            os.environ, {search.CPU_STREAM_PRESSED_ENV: "yes"}
        ):
            with self.assertRaises(ValueError):
                search._stream_pressed_cpu_enabled()

    def test_automatic_selection_is_large_installed_fixed_cpu_only(self):
        with tempfile.TemporaryDirectory(prefix="astra-cpu-select-") as tmp:
            root = Path(tmp)
            alphabet = pyhmmer.easel.Alphabet.amino()
            background = pyhmmer.plan7.Background(alphabet)
            builder = pyhmmer.plan7.Builder(alphabet, seed=37)
            hmms = []
            for index in range(3):
                sequence = pyhmmer.easel.TextSequence(
                    name=f"target-{index}".encode(),
                    sequence="ACDEFGHIKLMNPQRSTVWY" * 2,
                ).digitize(alphabet)
                hmm, _, _ = builder.build(sequence, background)
                hmm.name = f"model-{index}".encode()
                hmm.cutoffs.gathering = (0.0, 0.0)
                hmms.append(hmm)
            pressed = root / "profiles"
            pyhmmer.hmmer.hmmpress(hmms, pressed)
            options = _options(root / "output")

            for value in (None, "auto", "1"):
                environment = (
                    {} if value is None
                    else {search.CPU_STREAM_PRESSED_ENV: value}
                )
                with (
                    self.subTest(policy=value),
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.object(search, "HMM_CHUNK_SIZE", 2),
                ):
                    stream, decision = search._select_pressed_cpu_stream(
                        pressed, options
                    )
                self.assertIsInstance(stream, search._PressedHMMStream)
                self.assertEqual(stream.pressed_base, os.fspath(pressed))
                self.assertTrue(decision.enabled)
                self.assertEqual(decision.policy, value or "auto")
                self.assertEqual(decision.reason, "eligible-fixed-profile-cutoff")
                self.assertEqual(decision.cutoff, "gathering")
                self.assertEqual(decision.profiles_inspected, 3)

            with (
                mock.patch.dict(
                    os.environ, {search.CPU_STREAM_PRESSED_ENV: "0"},
                    clear=True,
                ),
                mock.patch.object(search, "HMM_CHUNK_SIZE", 2),
            ):
                stream, decision = search._select_pressed_cpu_stream(
                    pressed, options
                )
            self.assertIsNone(stream)
            self.assertEqual(decision.reason, "disabled-by-environment")

            # Explicit enable is a request, not permission to penalize a tiny
            # database or cross one of the unproven semantic boundaries.
            fallback_cases = (
                ({}, {"installed": False}, "custom-source"),
                ({}, {"gpu": True}, "gpu-search"),
                ({}, {"macsyfinder": True}, "macsyfinder-output"),
                ({"cascade": True}, {}, "cascade-thresholds"),
                ({"cut_nc": True}, {}, "multiple-cutoff-families"),
            )
            for overrides, keywords, reason in fallback_cases:
                candidate_options = dict(options)
                candidate_options.update(overrides)
                with (
                    self.subTest(reason=reason),
                    mock.patch.dict(
                        os.environ,
                        {search.CPU_STREAM_PRESSED_ENV: "1"},
                        clear=True,
                    ),
                    mock.patch.object(search, "HMM_CHUNK_SIZE", 2),
                ):
                    stream, decision = search._select_pressed_cpu_stream(
                        pressed, candidate_options, **keywords
                    )
                self.assertIsNone(stream)
                self.assertEqual(decision.reason, reason)

            with (
                mock.patch.dict(
                    os.environ, {search.CPU_STREAM_PRESSED_ENV: "1"},
                    clear=True,
                ),
                mock.patch.object(search, "HMM_CHUNK_SIZE", len(hmms)),
            ):
                stream, decision = search._select_pressed_cpu_stream(
                    pressed, options
                )
            self.assertIsNone(stream)
            self.assertEqual(decision.reason, "small-profile-database")
            self.assertEqual(decision.profiles_inspected, len(hmms))

            stream, decision = search._select_pressed_cpu_stream(None, options)
            self.assertIsNone(stream)
            self.assertEqual(decision.reason, "unpressed-source")

    def test_selection_telemetry_is_explicit_and_machine_readable(self):
        decision = search._PressedCPUStreamDecision(
            "auto", True, "eligible-fixed-global", None, 2001
        )
        with (
            mock.patch("builtins.print") as emit,
            mock.patch.object(search.logging, "info") as log,
        ):
            search._report_pressed_cpu_stream_decision(decision)
        message = (
            "CPU pressed-profile stream policy=auto status=enabled "
            "reason=eligible-fixed-global cutoff=global "
            "profiles_inspected=2001"
        )
        emit.assert_called_once_with(f"  {message}")
        log.assert_called_once_with(message)


if __name__ == "__main__":
    unittest.main()
