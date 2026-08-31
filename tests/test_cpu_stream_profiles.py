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
        "cut_ga": False,
        "cut_nc": False,
        "cut_tc": False,
        "evalue": 10.0,
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
                hmms.append(hmm)
                targets.append(sequence)

            pressed = root / "profiles"
            pyhmmer.hmmer.hmmpress(hmms, pressed)
            stream = search._PressedHMMStream(pressed)
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
