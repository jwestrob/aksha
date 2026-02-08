"""Integration smoke test using installed barrnap HMMs via Aksha nhmmer."""

from pathlib import Path

import platformdirs
import pytest

from aksha import nhmmer, ThresholdOptions


@pytest.mark.integration
def test_barrnap_db_nhmmer_hits(temp_dir, rrna_16s_fasta):
    """nhmmer against installed barrnap DB should find the 16S sequence."""
    data_dir = Path(platformdirs.user_data_dir("aksha"))
    if not (data_dir / "barrnap").exists():
        pytest.skip(f"barrnap database not installed under {data_dir / 'barrnap'}")

    result = nhmmer(
        sequences=rrna_16s_fasta,
        hmms="barrnap",
        thresholds=ThresholdOptions(),
        output_dir=temp_dir,
        show_progress=False,
    )

    df = result.to_dataframe()
    assert not df.empty
    assert {"sequence_id", "hmm_name", "bitscore"}.issubset(df.columns)
    # Ensure the 16S entry is detected
    assert df["sequence_id"].str.contains("NR_074891").any()
