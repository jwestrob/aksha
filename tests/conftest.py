"""Pytest fixtures for Aksha tests."""

from pathlib import Path
import shutil
import tempfile

import pytest

from aksha.config import AkshaConfig, DatabaseRegistry, reset_globals


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def test_config(temp_dir, monkeypatch):
    """Create test configuration with isolated paths."""
    config_dir = temp_dir / "config"
    data_dir = temp_dir / "data"
    config_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setenv("AKSHA_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AKSHA_DATA_DIR", str(data_dir))

    reset_globals()

    yield AkshaConfig.load()

    reset_globals()


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def small_fasta(fixtures_dir):
    """Path to small test FASTA."""
    return fixtures_dir / "small.faa"


@pytest.fixture
def small_hmm(fixtures_dir):
    """Path to small test HMM."""
    return fixtures_dir / "small.hmm"


@pytest.fixture
def dummy_dataset_dir():
    """Path to bundled dummy dataset (protein FASTA files)."""
    return Path(__file__).resolve().parents[1] / "dummy_dataset"


@pytest.fixture
def small_fna(fixtures_dir):
    """Path to small nucleotide test FASTA."""
    return fixtures_dir / "small.fna"


@pytest.fixture
def small_nuc_hmm(fixtures_dir):
    """Path to small nucleotide test HMM."""
    return fixtures_dir / "small_nuc.hmm"


@pytest.fixture
def dummy_genomes_dir():
    """Path to bundled dummy nucleotide genomes."""
    return Path(__file__).resolve().parents[1] / "dummy_genomes"


@pytest.fixture
def cutoff_ga_hmm(fixtures_dir):
    """Path to toy HMM with GA/TC/NC cutoffs embedded."""
    return fixtures_dir / "cutoff_ga.hmm"


@pytest.fixture
def rrna_16s_fasta(fixtures_dir):
    """16S rRNA sequence for barrnap integration test."""
    return fixtures_dir / "rrna_16S.fa"
