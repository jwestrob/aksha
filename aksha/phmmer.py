"""phmmer: protein sequence vs protein sequence search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyhmmer
from tqdm import tqdm

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


def phmmer(
    query: SequenceInput,
    target: SequenceInput,
    *,
    thresholds: Optional[ThresholdOptions] = None,
    threads: int = 0,
    output_dir: Optional[PathLike] = None,
    show_progress: bool = True,
) -> SearchResult:
    """Search query sequences against target sequences using phmmer.

    phmmer builds a profile from each query sequence and searches
    against the target database.
    """
    if thresholds is None:
        thresholds = ThresholdOptions()

    query_iter = iter_sequences(query, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, query_seqs = next(query_iter)
    if next(query_iter, None) is not None:
        raise ValueError("phmmer expects a single query file, got a directory with multiple files")

    target_iter = iter_sequences(target, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, target_seqs = next(target_iter)
    if next(target_iter, None) is not None:
        raise ValueError("phmmer expects a single target file, got a directory with multiple files")

    kwargs = build_search_kwargs(thresholds)
    output_path = Path(output_dir) if output_dir else None
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        results = pyhmmer.phmmer(query_seqs, target_seqs, cpus=cpus, **kwargs)
        iterator = tqdm(results, desc="phmmer") if show_progress else results

        for hits in iterator:
            query_name = hits.query.name.decode()
            for hit in hits_from_pyhmmer(hits, query_name):
                collector.add(hit)

        return collector.finalize()
