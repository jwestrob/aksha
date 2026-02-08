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
    SequenceHit,
    DomainHit,
)
from aksha.parsers import parse_hmms, iter_sequences, HMMInput, SequenceInput
from aksha.thresholds import build_search_kwargs
from aksha.results import ResultCollector
from aksha.search import _resolve_cpus
from aksha.config import get_registry

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

    if isinstance(hmms, str) and not Path(hmms).exists():
        registry = get_registry()
        db_path = registry.get_path(hmms)
        if db_path:
            hmm_list = parse_hmms(db_path)
        else:
            raise FileNotFoundError(f"HMM source not found: {hmms}")
    else:
        hmm_list = parse_hmms(hmms)

    # Use the HMM alphabet (barrnap models are RNA) to digitize sequences correctly
    alphabet = hmm_list[0].alphabet if hmm_list else None

    sequence_iter = iter_sequences(
        sequences,
        molecule_type=MoleculeType.NUCLEOTIDE,
        alphabet=alphabet,
        show_progress=show_progress,
    )

    kwargs = build_search_kwargs(thresholds)
    # If cut_ga requested but HMM lacks gathering cutoffs, drop bit_cutoffs to avoid MissingCutoffs
    if kwargs.get("bit_cutoffs") and not all(h.cutoffs.gathering_available() for h in hmm_list):
        kwargs.pop("bit_cutoffs", None)
    output_path = Path(output_dir) if output_dir else None
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        for seq_path, sequences_block in sequence_iter:
            for hits in pyhmmer.nhmmer(hmm_list, sequences_block, cpus=cpus, **kwargs):
                # TopHits.query holds the HMM used as query
                hmm_name = hits.query.name.decode()

                for hit in hits:
                    if not hit.included:
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
                            hmm_name=hmm_name,
                            evalue=hit.evalue,
                            bitscore=hit.score,
                            domains=domains,
                        )
                    )

        return collector.finalize()
