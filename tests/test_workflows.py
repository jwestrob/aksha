"""End-to-end smoke tests for CLI-equivalent workflows on protein data."""

from aksha import ThresholdOptions, search, scan, nhmmer
from aksha.phmmer import phmmer
from aksha.jackhmmer import jackhmmer


REQUIRED_COLS = {
    "sequence_id",
    "hmm_name",
    "evalue",
    "bitscore",
    "c_evalue",
    "i_evalue",
    "dom_bitscore",
    "env_from",
    "env_to",
    "ali_from",
    "ali_to",
}


def test_scan_small_has_hits(small_fasta, small_hmm, temp_dir):
    """hmmscan-style run should produce hits on the toy fixtures."""
    result = scan(
        sequences=small_fasta,
        hmms=small_hmm,
        output_dir=temp_dir,
        show_progress=False,
    )

    df = result.to_dataframe()
    assert len(df) >= 1
    assert REQUIRED_COLS.issubset(df.columns)


def test_phmmer_self_search(small_fasta, temp_dir):
    """phmmer should recover self-hits within the same file."""
    result = phmmer(
        query=small_fasta,
        target=small_fasta,
        output_dir=temp_dir,
        show_progress=False,
    )

    # At least one hit per query is expected when searching against itself
    assert len(result) >= 4
    df = result.to_dataframe()
    assert REQUIRED_COLS.issubset(df.columns)


def test_jackhmmer_iterative_small(small_fasta, temp_dir):
    """jackhmmer should run a couple iterations and yield hits."""
    result = jackhmmer(
        query=small_fasta,
        target=small_fasta,
        max_iterations=2,
        output_dir=temp_dir,
        show_progress=False,
    )

    assert len(result) >= 1
    df = result.to_dataframe()
    assert REQUIRED_COLS.issubset(df.columns)


def test_search_dummy_dataset_runs(dummy_dataset_dir, small_hmm, temp_dir):
    """search should handle a directory of FASTA files (dummy dataset)."""
    result = search(
        sequences=dummy_dataset_dir,
        hmms=small_hmm,
        thresholds=ThresholdOptions(),  # defaults, no strict cutoffs
        output_dir=temp_dir,
        show_progress=False,
    )

    out_path = result.to_csv(temp_dir / "search_dummy.tsv")
    assert out_path.exists()
    df = result.to_dataframe()
    assert REQUIRED_COLS.issubset(df.columns)


def test_scan_dummy_dataset_runs(dummy_dataset_dir, small_hmm, temp_dir):
    """scan should also handle directory input for sequences."""
    result = scan(
        sequences=dummy_dataset_dir,
        hmms=small_hmm,
        thresholds=ThresholdOptions(),
        output_dir=temp_dir,
        show_progress=False,
    )

    out_path = result.to_csv(temp_dir / "scan_dummy.tsv")
    assert out_path.exists()
    df = result.to_dataframe()
    assert REQUIRED_COLS.issubset(df.columns)


def test_nhmmer_small_has_hits(small_fna, small_nuc_hmm, temp_dir):
    """nhmmer should yield hits for the toy nucleotide fixtures."""
    result = nhmmer(
        sequences=small_fna,
        hmms=small_nuc_hmm,
        thresholds=ThresholdOptions(),
        output_dir=temp_dir,
        show_progress=False,
    )

    df = result.to_dataframe()
    assert len(df) >= 1
    assert REQUIRED_COLS.issubset(df.columns)


def test_nhmmer_dummy_genomes_runs(dummy_genomes_dir, small_nuc_hmm, temp_dir):
    """nhmmer should run on the bundled dummy nucleotide genomes directory."""
    result = nhmmer(
        sequences=dummy_genomes_dir,
        hmms=small_nuc_hmm,
        thresholds=ThresholdOptions(),
        output_dir=temp_dir,
        show_progress=False,
    )

    out_path = result.to_csv(temp_dir / "nhmmer_dummy.tsv")
    assert out_path.exists()
    df = result.to_dataframe()
    assert REQUIRED_COLS.issubset(df.columns)
