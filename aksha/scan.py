"""HMM scan implementation using PyHMMER.

Unlike search (HMMs as queries), scan uses sequences as queries
against an HMM database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Union

import pyhmmer

from aksha.types import (
    PathLike,
    MoleculeType,
    ThresholdOptions,
    SearchResult,
    SequenceBlock,
    HMM,
)
from aksha.parsers import iter_sequences, HMMInput, SequenceInput
from aksha.thresholds import build_search_kwargs, group_hmms_by_cutoff
from aksha.results import ResultCollector, hits_from_pyhmmer
from aksha.search import _resolve_hmms, _resolve_cpus

logger = logging.getLogger(__name__)


def scan(
    sequences: SequenceInput,
    hmms: Union[HMMInput, str],
    *,
    thresholds: Optional[ThresholdOptions] = None,
    threads: int = 0,
    output_dir: Optional[PathLike] = None,
    show_progress: bool = True,
) -> SearchResult:
    """Scan sequences against HMM database using hmmscan.

    Unlike search(), this uses sequences as queries against the HMM database.
    Better when you have few sequences and many HMMs.
    """
    if thresholds is None:
        thresholds = ThresholdOptions()

    hmm_list = _resolve_hmms(hmms)
    if not hmm_list:
        raise ValueError("No HMM profiles found. Check your --hmms path or database name.")

    sequence_iter = iter_sequences(
        sequences,
        molecule_type=MoleculeType.PROTEIN,
        show_progress=show_progress,
    )

    output_path = Path(output_dir) if output_dir else None

    with ResultCollector(output_dir=output_path) as collector:
        _run_hmmscan(
            sequence_iter=sequence_iter,
            hmms=hmm_list,
            thresholds=thresholds,
            threads=threads,
            collector=collector,
            show_progress=show_progress,
        )
        return collector.finalize()


def _run_hmmscan(
    sequence_iter: Iterator[tuple[Path, SequenceBlock]],
    hmms: list[HMM],
    thresholds: ThresholdOptions,
    threads: int,
    collector: ResultCollector,
    show_progress: bool,
) -> None:
    """Execute hmmscan across all sequence files."""
    hmm_groups = group_hmms_by_cutoff(hmms, thresholds)
    base_kwargs = build_search_kwargs(thresholds)
    cpus = _resolve_cpus(threads)

    for seq_path, sequences in sequence_iter:
        logger.debug("Scanning %s (%d sequences)", seq_path, len(sequences))

        for cutoff_type, hmm_group in hmm_groups.items():
            kwargs = base_kwargs.copy()
            if cutoff_type:
                kwargs["bit_cutoffs"] = cutoff_type

            # hmmscan takes (sequences, hmms); each TopHits corresponds to one query sequence
            for hits in pyhmmer.hmmscan(sequences, hmm_group, cpus=cpus, **kwargs):
                seq_name = hits.query.name.decode()
                for hit in hits_from_pyhmmer(hits, hmm_name="", sequence_id=seq_name):
                    collector.add(hit)
