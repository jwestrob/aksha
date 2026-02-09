"""HMM search implementation using PyHMMER."""

from __future__ import annotations

import logging
import os
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
from aksha.parsers import parse_hmms, iter_sequences, HMMInput, SequenceInput
from aksha.thresholds import build_search_kwargs, group_hmms_by_cutoff
from aksha.results import ResultCollector, hits_from_pyhmmer
from aksha.config import get_registry

logger = logging.getLogger(__name__)


def _resolve_cpus(threads: int) -> int:
    """Normalize thread count (0 -> all available)."""
    return threads if threads > 0 else (os.cpu_count() or 1)


def search(
    sequences: SequenceInput,
    hmms: Union[HMMInput, str],  # str = database name
    *,
    thresholds: Optional[ThresholdOptions] = None,
    threads: int = 0,
    output_dir: Optional[PathLike] = None,
    show_progress: bool = True,
) -> SearchResult:
    """Search sequences against HMM profiles using hmmsearch.

    Args:
        sequences: Protein sequences (path, directory, or pre-loaded)
        hmms: HMM profiles (path, directory, database name, or pre-loaded)
        thresholds: Threshold configuration (default: no cutoffs)
        threads: CPU threads (0 = auto-detect)
        output_dir: Output directory for large results
        show_progress: Show progress bar

    Returns:
        SearchResult containing hits
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
        _run_hmmsearch(
            sequence_iter=sequence_iter,
            hmms=hmm_list,
            thresholds=thresholds,
            threads=threads,
            collector=collector,
        )
        return collector.finalize()


def _resolve_hmms(source: Union[HMMInput, str]) -> list[HMM]:
    """Resolve HMM source to list of HMM objects."""
    if isinstance(source, str) and not Path(source).exists():
        registry = get_registry()
        db_path = registry.get_path(source)
        if db_path:
            logger.info("Using installed database: %s at %s", source, db_path)
            return parse_hmms(db_path)
        # Check if it's a known but uninstalled database
        if registry.get(source) is not None:
            raise FileNotFoundError(
                f"Database '{source}' is not installed. "
                f"Run: aksha database install {source}"
            )

    return parse_hmms(source)


def _run_hmmsearch(
    sequence_iter: Iterator[tuple[Path, SequenceBlock]],
    hmms: list[HMM],
    thresholds: ThresholdOptions,
    threads: int,
    collector: ResultCollector,
) -> None:
    """Execute hmmsearch across all sequence files."""
    hmm_groups = group_hmms_by_cutoff(hmms, thresholds)

    base_kwargs = build_search_kwargs(thresholds)
    cpus = _resolve_cpus(threads)

    for seq_path, sequences in sequence_iter:
        logger.debug("Searching %s (%d sequences)", seq_path, len(sequences))

        for cutoff_type, hmm_group in hmm_groups.items():
            kwargs = base_kwargs.copy()
            if cutoff_type:
                kwargs["bit_cutoffs"] = cutoff_type

            for hits in pyhmmer.hmmsearch(hmm_group, sequences, cpus=cpus, **kwargs):
                hmm_name = hits.query.name.decode()
                for hit in hits_from_pyhmmer(hits, hmm_name):
                    collector.add(hit)
