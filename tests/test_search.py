"""Tests for search functionality."""

import pytest

from aksha import search, ThresholdOptions


def test_search_basic(small_fasta, small_hmm, temp_dir):
    """Test basic search functionality."""
    result = search(
        sequences=small_fasta,
        hmms=small_hmm,
        output_dir=temp_dir,
    )

    assert result is not None
    df = result.to_dataframe()
    assert "sequence_id" in df.columns
    assert "hmm_name" in df.columns


def test_search_with_thresholds(small_fasta, small_hmm, temp_dir):
    """Test search with threshold options."""
    result = search(
        sequences=small_fasta,
        hmms=small_hmm,
        thresholds=ThresholdOptions(evalue=0.001),
        output_dir=temp_dir,
    )

    df = result.to_dataframe()
    assert all(df["evalue"] <= 0.001)
