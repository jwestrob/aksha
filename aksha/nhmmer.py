"""nhmmer: nucleotide HMM search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pyhmmer

from aksha.types import (
    PathLike,
    MoleculeType,
    ThresholdOptions,
    SearchResult,
)
from aksha.parsers import iter_sequences, HMMInput, SequenceInput
from aksha.thresholds import build_search_kwargs
from aksha.results import ResultCollector, hits_from_pyhmmer
from aksha.search import _resolve_hmms, _resolve_cpus

logger = logging.getLogger(__name__)


def nhmmer(
    sequences: SequenceInput,
    hmms: Union[HMMInput, str],
    *,
    thresholds: Optional[ThresholdOptions] = None,
    threads: int = 0,
    output_dir: Optional[PathLike] = None,
    show_progress: bool = True,
) -> SearchResult:
    """Search nucleotide sequences against nucleotide HMMs using nhmmer."""
    if thresholds is None:
        thresholds = ThresholdOptions()

    hmm_list = _resolve_hmms(hmms)
    if not hmm_list:
        raise ValueError("No HMM profiles found. Check your --hmms path or database name.")

    # Use the HMM alphabet (barrnap models are RNA) to digitize sequences correctly
    alphabet = hmm_list[0].alphabet if hmm_list else None

    sequence_iter = iter_sequences(
        sequences,
        molecule_type=MoleculeType.NUCLEOTIDE,
        alphabet=alphabet,
        show_progress=show_progress,
    )

    kwargs = build_search_kwargs(thresholds)
    # If bit cutoffs requested but HMMs lack them, fall back gracefully
    if kwargs.get("bit_cutoffs") and not all(h.cutoffs.gathering_available() for h in hmm_list):
        logger.warning(
            "Requested bit_cutoffs='%s' but not all HMMs have cutoffs available; "
            "falling back to default thresholds",
            kwargs["bit_cutoffs"],
        )
        kwargs.pop("bit_cutoffs", None)
    output_path = Path(output_dir) if output_dir else None
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        for seq_path, sequences_block in sequence_iter:
            for hits in pyhmmer.nhmmer(hmm_list, sequences_block, cpus=cpus, **kwargs):
                hmm_name = hits.query.name.decode()
                for hit in hits_from_pyhmmer(hits, hmm_name):
                    collector.add(hit)

        return collector.finalize()
