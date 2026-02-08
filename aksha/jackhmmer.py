"""jackhmmer: iterative protein sequence search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyhmmer

from aksha.types import (
    PathLike,
    MoleculeType,
    ThresholdOptions,
    SearchResult,
)
from aksha.parsers import iter_sequences, SequenceInput
from aksha.thresholds import build_search_kwargs
from aksha.results import ResultCollector, hits_from_pyhmmer
from aksha.search import _resolve_cpus

logger = logging.getLogger(__name__)


def jackhmmer(
    query: SequenceInput,
    target: SequenceInput,
    *,
    thresholds: Optional[ThresholdOptions] = None,
    threads: int = 0,
    max_iterations: int = 5,
    output_dir: Optional[PathLike] = None,
    show_progress: bool = True,
) -> SearchResult:
    """Iteratively search query sequences against targets using jackhmmer.

    jackhmmer performs iterative searches, building an HMM from hits
    at each iteration to find more distant homologs.
    """
    if thresholds is None:
        thresholds = ThresholdOptions()

    query_iter = iter_sequences(query, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, query_seqs = next(query_iter)
    if next(query_iter, None) is not None:
        raise ValueError("jackhmmer expects a single query file, got a directory with multiple files")

    target_iter = iter_sequences(target, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, target_seqs = next(target_iter)
    if next(target_iter, None) is not None:
        raise ValueError("jackhmmer expects a single target file, got a directory with multiple files")

    kwargs = build_search_kwargs(thresholds)
    output_path = Path(output_dir) if output_dir else None
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        search_iter = pyhmmer.jackhmmer(
            query_seqs,
            target_seqs,
            cpus=cpus,
            max_iterations=max_iterations,
            **kwargs,
        )

        final_results = None
        for iteration in search_iter:
            final_results = iteration

        if final_results is None:
            return collector.finalize()

        for hit in hits_from_pyhmmer(
            final_results.hits, "jackhmmer_query", skip_duplicates=True
        ):
            collector.add(hit)

        return collector.finalize()
