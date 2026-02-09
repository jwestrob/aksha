"""Edge case and validation tests for Aksha."""

import logging
import shutil
from pathlib import Path

import pytest

from aksha.types import (
    ThresholdOptions,
    SearchResult,
    SequenceHit,
    DomainHit,
    MoleculeType,
)
from aksha.results import ResultCollector
from aksha.parsers import iter_sequences, parse_hmms
from aksha.config import DatabaseRegistry, reset_globals


# ---------------------------------------------------------------------------
# ThresholdOptions validation
# ---------------------------------------------------------------------------


def test_threshold_rejects_multiple_cutoffs():
    """Only one of cut_ga, cut_tc, cut_nc can be True."""
    with pytest.raises(ValueError, match="Only one of"):
        ThresholdOptions(cut_ga=True, cut_tc=True)


def test_threshold_rejects_cascade_with_explicit_cutoff():
    """cascade=True cannot be combined with an explicit cutoff."""
    with pytest.raises(ValueError, match="Cannot combine cascade"):
        ThresholdOptions(cascade=True, cut_ga=True)

    with pytest.raises(ValueError, match="Cannot combine cascade"):
        ThresholdOptions(cascade=True, cut_nc=True)


def test_threshold_allows_cascade_alone():
    """cascade=True without explicit cutoff should be fine."""
    opts = ThresholdOptions(cascade=True)
    assert opts.cascade is True


def test_threshold_allows_single_cutoff():
    """A single cutoff flag should be fine."""
    opts = ThresholdOptions(cut_ga=True)
    assert opts.cut_ga is True


# ---------------------------------------------------------------------------
# Disk-backed SearchResult: __len__, __iter__, to_csv
# ---------------------------------------------------------------------------


def _make_disk_result(temp_dir: Path, n_hits: int = 3) -> SearchResult:
    """Create a disk-backed SearchResult with n_hits via ResultCollector."""
    collector = ResultCollector(output_dir=temp_dir, force_disk=True)
    for i in range(n_hits):
        domain = DomainHit(
            c_evalue=1e-5 * (i + 1),
            i_evalue=1e-5 * (i + 1),
            bitscore=50.0 - i,
            env_from=1 + i * 10,
            env_to=10 + i * 10,
            ali_from=1 + i * 10,
            ali_to=10 + i * 10,
        )
        hit = SequenceHit(
            sequence_id=f"seq_{i}",
            hmm_name=f"model_{i % 2}",
            evalue=1e-6 * (i + 1),
            bitscore=60.0 - i,
            domains=(domain,),
        )
        collector.add(hit)
    return collector.finalize()


def test_disk_result_len(temp_dir):
    """len() on a disk-backed result should return the correct count."""
    result = _make_disk_result(temp_dir, n_hits=5)
    assert len(result) == 5


def test_disk_result_iter(temp_dir):
    """Iterating a disk-backed result should yield SequenceHit objects."""
    result = _make_disk_result(temp_dir, n_hits=3)
    hits = list(result)
    assert len(hits) == 3
    assert all(isinstance(h, SequenceHit) for h in hits)
    assert hits[0].sequence_id == "seq_0"
    assert hits[2].sequence_id == "seq_2"


def test_disk_result_iter_domains(temp_dir):
    """Domain data should survive the disk round-trip."""
    result = _make_disk_result(temp_dir, n_hits=1)
    hit = next(iter(result))
    assert len(hit.domains) == 1
    assert hit.domains[0].env_from == 1
    assert hit.domains[0].env_to == 10


def test_disk_result_to_csv_copies(temp_dir):
    """to_csv on a disk-backed result should copy the file, not rebuild via pandas."""
    result = _make_disk_result(temp_dir, n_hits=2)
    out_path = temp_dir / "exported.tsv"
    result.to_csv(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    # Read it back — should still parse
    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 data rows


def test_disk_result_to_csv_same_path(temp_dir):
    """to_csv with same path as the backing file should be a no-op."""
    result = _make_disk_result(temp_dir, n_hits=1)
    backing_path = result._output_path
    returned = result.to_csv(backing_path)
    assert returned == backing_path


# ---------------------------------------------------------------------------
# Multi-domain hits through disk round-trip
# ---------------------------------------------------------------------------


def test_disk_result_multi_domain(temp_dir):
    """A hit with multiple domains should group correctly on disk round-trip."""
    collector = ResultCollector(output_dir=temp_dir, force_disk=True)
    domains = tuple(
        DomainHit(
            c_evalue=1e-5,
            i_evalue=1e-5,
            bitscore=50.0 + j,
            env_from=1 + j * 20,
            env_to=20 + j * 20,
            ali_from=1 + j * 20,
            ali_to=20 + j * 20,
        )
        for j in range(3)
    )
    hit = SequenceHit(
        sequence_id="multi_dom_seq",
        hmm_name="hmm_X",
        evalue=1e-10,
        bitscore=100.0,
        domains=domains,
    )
    collector.add(hit)
    result = collector.finalize()

    hits = list(result)
    assert len(hits) == 1
    assert len(hits[0].domains) == 3
    assert hits[0].domains[0].bitscore == pytest.approx(50.0, abs=0.1)
    assert hits[0].domains[2].bitscore == pytest.approx(52.0, abs=0.1)


# ---------------------------------------------------------------------------
# iter_sequences edge cases
# ---------------------------------------------------------------------------


def test_iter_sequences_empty_dir(temp_dir):
    """iter_sequences on an empty directory should raise ValueError."""
    empty = temp_dir / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No sequence files found"):
        list(iter_sequences(empty, molecule_type=MoleculeType.PROTEIN, show_progress=False))


def test_iter_sequences_nonexistent_path():
    """iter_sequences on a missing path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        list(iter_sequences("/nonexistent/path.faa", molecule_type=MoleculeType.PROTEIN))


def test_iter_sequences_single_file(small_fasta):
    """iter_sequences on a single file should yield exactly one pair."""
    pairs = list(iter_sequences(small_fasta, molecule_type=MoleculeType.PROTEIN, show_progress=False))
    assert len(pairs) == 1
    path, block = pairs[0]
    assert path == small_fasta
    assert len(block) > 0


def test_iter_sequences_directory(dummy_dataset_dir):
    """iter_sequences on a directory should yield one pair per .faa file."""
    pairs = list(
        iter_sequences(dummy_dataset_dir, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    )
    assert len(pairs) >= 2
    for path, block in pairs:
        assert path.suffix.lower() in (".faa", ".fna", ".fa", ".fasta", ".fas")
        assert len(block) > 0


# ---------------------------------------------------------------------------
# phmmer/jackhmmer single-file validation
# ---------------------------------------------------------------------------


def test_phmmer_rejects_directory_input(dummy_dataset_dir, temp_dir):
    """phmmer should reject directory input for query."""
    from aksha.phmmer import phmmer

    with pytest.raises(ValueError, match="single query file"):
        phmmer(
            query=dummy_dataset_dir,
            target=dummy_dataset_dir,
            output_dir=temp_dir,
            show_progress=False,
        )


def test_jackhmmer_rejects_directory_input(dummy_dataset_dir, temp_dir):
    """jackhmmer should reject directory input for query."""
    from aksha.jackhmmer import jackhmmer

    with pytest.raises(ValueError, match="single query file"):
        jackhmmer(
            query=dummy_dataset_dir,
            target=dummy_dataset_dir,
            output_dir=temp_dir,
            show_progress=False,
        )


# ---------------------------------------------------------------------------
# Empty HMM validation
# ---------------------------------------------------------------------------


def test_search_rejects_empty_hmm_dir(temp_dir, small_fasta):
    """search should raise ValueError if the HMM directory has no .hmm files."""
    from aksha.search import search

    empty_hmm = temp_dir / "empty_hmm"
    empty_hmm.mkdir()

    with pytest.raises((ValueError, FileNotFoundError)):
        search(
            sequences=small_fasta,
            hmms=str(empty_hmm),
            output_dir=temp_dir,
            show_progress=False,
        )


# ---------------------------------------------------------------------------
# Custom database registration and persistence
# ---------------------------------------------------------------------------


def test_register_custom_database(test_config):
    """Registering a custom DB should persist across registry reloads."""
    from aksha.config import DatabaseRegistry

    registry = DatabaseRegistry(test_config)
    registry.register_custom("my_custom", "/some/path/hmms", MoleculeType.PROTEIN, "My citation")

    entry = registry.get("my_custom")
    assert entry is not None
    assert entry.installed is True
    assert entry.molecule_type == MoleculeType.PROTEIN
    assert entry.citation == "My citation"

    # Reload the registry from disk — custom DB should survive
    registry2 = DatabaseRegistry(test_config)
    entry2 = registry2.get("my_custom")
    assert entry2 is not None
    assert entry2.installed is True
    assert entry2.molecule_type == MoleculeType.PROTEIN
    assert entry2.citation == "My citation"


def test_register_duplicate_database_raises(test_config):
    """Registering a DB with a name that already exists should raise."""
    from aksha.config import DatabaseRegistry

    registry = DatabaseRegistry(test_config)
    # PFAM is bundled
    with pytest.raises(ValueError, match="already exists"):
        registry.register_custom("PFAM", "/some/path", MoleculeType.PROTEIN)


def test_list_databases_filters(test_config):
    """list_available should filter by molecule type and installed status."""
    from aksha.config import DatabaseRegistry

    registry = DatabaseRegistry(test_config)
    all_dbs = registry.list_available()
    assert len(all_dbs) > 0

    protein_dbs = registry.list_available(molecule_type=MoleculeType.PROTEIN)
    nuc_dbs = registry.list_available(molecule_type=MoleculeType.NUCLEOTIDE)
    assert len(protein_dbs) + len(nuc_dbs) == len(all_dbs)

    installed_dbs = registry.list_available(installed_only=True)
    assert all(db.installed for db in installed_dbs)


# ---------------------------------------------------------------------------
# nhmmer cutoff fallback logging
# ---------------------------------------------------------------------------


def test_nhmmer_logs_cutoff_fallback(caplog, small_fna, small_nuc_hmm, temp_dir):
    """nhmmer should log a warning when bit_cutoffs are dropped."""
    from aksha import nhmmer

    with caplog.at_level(logging.WARNING, logger="aksha.nhmmer"):
        nhmmer(
            sequences=small_fna,
            hmms=small_nuc_hmm,
            thresholds=ThresholdOptions(cut_ga=True),
            output_dir=temp_dir,
            show_progress=False,
        )

    # The small_nuc_hmm likely lacks gathering cutoffs, so a warning should appear
    # If it does have cutoffs, the test still passes (just no warning)
    has_cutoffs = all(
        h.cutoffs.gathering_available()
        for h in parse_hmms(small_nuc_hmm, show_progress=False)
    )
    if not has_cutoffs:
        assert any("falling back" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SearchResult.best_domain property
# ---------------------------------------------------------------------------


def test_best_domain_property():
    """best_domain should return the highest-scoring domain."""
    d1 = DomainHit(c_evalue=1e-3, i_evalue=1e-3, bitscore=30.0, env_from=1, env_to=10, ali_from=1, ali_to=10)
    d2 = DomainHit(c_evalue=1e-5, i_evalue=1e-5, bitscore=80.0, env_from=20, env_to=30, ali_from=20, ali_to=30)
    hit = SequenceHit(sequence_id="s", hmm_name="h", evalue=1e-5, bitscore=80.0, domains=(d1, d2))
    assert hit.best_domain is d2


def test_best_domain_empty():
    """best_domain with no domains should return None."""
    hit = SequenceHit(sequence_id="s", hmm_name="h", evalue=1e-5, bitscore=0.0, domains=())
    assert hit.best_domain is None
