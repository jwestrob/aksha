"""Threshold behavior tests (cutoffs and cascade)."""

import importlib

from aksha import search, ThresholdOptions
from aksha.types import BitscoreCutoff

search_module = importlib.import_module("aksha.search")


def test_cut_ga_sets_bit_cutoffs(monkeypatch, small_fasta, cutoff_ga_hmm, temp_dir):
    """Explicit --cut_ga should pass gathering cutoff to hmmsearch."""
    captured = {}

    def fake_hmmsearch(hmms, seqs, cpus=1, **kwargs):
        captured["bit_cutoffs"] = kwargs.get("bit_cutoffs")
        return []

    monkeypatch.setattr(search_module.pyhmmer, "hmmsearch", fake_hmmsearch)

    search(
        sequences=small_fasta,
        hmms=cutoff_ga_hmm,
        thresholds=ThresholdOptions(cut_ga=True),
        output_dir=temp_dir,
        show_progress=False,
    )

    assert captured["bit_cutoffs"] == BitscoreCutoff.GATHERING.value


def test_cascade_prefers_trusted(monkeypatch, small_fasta, cutoff_ga_hmm, temp_dir):
    """Cascade mode should choose the best available cutoff (trusted first)."""
    captured = {}

    def fake_hmmsearch(hmms, seqs, cpus=1, **kwargs):
        captured["bit_cutoffs"] = kwargs.get("bit_cutoffs")
        return []

    monkeypatch.setattr(search_module.pyhmmer, "hmmsearch", fake_hmmsearch)

    search(
        sequences=small_fasta,
        hmms=cutoff_ga_hmm,
        thresholds=ThresholdOptions(cascade=True),
        output_dir=temp_dir,
        show_progress=False,
    )

    # HMM has TC, GA, NC; cascade default prefers trusted
    assert captured["bit_cutoffs"] == BitscoreCutoff.TRUSTED.value
