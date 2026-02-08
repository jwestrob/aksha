"""Tests for ResultCollector edge cases."""

from aksha.results import ResultCollector
from aksha.types import SequenceHit, DomainHit


def test_resultcollector_force_disk(temp_dir):
    """Forcing disk storage should write a temp TSV and preserve rows."""
    collector = ResultCollector(output_dir=temp_dir, force_disk=True)

    domain = DomainHit(
        c_evalue=1e-5,
        i_evalue=1e-5,
        bitscore=50.0,
        env_from=1,
        env_to=20,
        ali_from=1,
        ali_to=20,
    )
    hit = SequenceHit(
        sequence_id="seqA",
        hmm_name="modelA",
        evalue=1e-6,
        bitscore=60.0,
        domains=(domain,),
    )

    collector.add(hit)
    result = collector.finalize()

    # Should have switched to disk and produced the temp file
    temp_tsv = temp_dir / ".aksha_temp_results.tsv"
    assert temp_tsv.exists() and temp_tsv.stat().st_size > 0

    df = result.to_dataframe()
    assert len(df) == 1
    assert set(df.columns) >= {
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
