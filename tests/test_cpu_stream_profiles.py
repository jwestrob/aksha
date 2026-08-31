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

    def test_opt_in_is_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(search._stream_pressed_cpu_enabled())
        with mock.patch.dict(os.environ, {search.CPU_STREAM_PRESSED_ENV: "1"}):
            self.assertTrue(search._stream_pressed_cpu_enabled())
        with mock.patch.dict(os.environ, {search.CPU_STREAM_PRESSED_ENV: "yes"}):
            with self.assertRaises(ValueError):
                search._stream_pressed_cpu_enabled()


if __name__ == "__main__":
    unittest.main()
