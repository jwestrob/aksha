"""jackhmmer: iterative protein sequence search."""

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
    SequenceHit,
    DomainHit,
)
from aksha.parsers import parse_sequences, SequenceInput
from aksha.thresholds import build_search_kwargs
from aksha.results import ResultCollector
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

    query_dict = parse_sequences(query, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    target_dict = parse_sequences(target, molecule_type=MoleculeType.PROTEIN, show_progress=False)

    query_seqs = list(query_dict.values())[0]
    target_seqs = list(target_dict.values())[0]

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

        for hit in final_results.hits:
            if not hit.included or hit.duplicate:
                continue

            domains = tuple(
                DomainHit(
                    c_evalue=d.c_evalue,
                    i_evalue=d.i_evalue,
                    bitscore=d.score,
                    env_from=d.env_from,
                    env_to=d.env_to,
                    ali_from=d.alignment.target_from,
                    ali_to=d.alignment.target_to,
                )
                for d in hit.domains.reported
            )

            collector.add(
                SequenceHit(
                    sequence_id=hit.name.decode(),
                    hmm_name="jackhmmer_query",
                    evalue=hit.evalue,
                    bitscore=hit.score,
                    domains=domains,
                )
            )

        return collector.finalize()
