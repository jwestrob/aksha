"""Threshold handling for HMM searches.

Implements the cascade system that tries multiple threshold types
in order of preference.
"""

from __future__ import annotations

from typing import Any, Optional

from aksha.types import ThresholdOptions, BitscoreCutoff, HMM
from aksha.parsers import get_best_available_cutoff


def build_search_kwargs(
    options: ThresholdOptions,
    hmms: Optional[list[HMM]] = None,
) -> dict[str, Any]:
    """Build kwargs dict for PyHMMER search functions.

    Args:
        options: Threshold configuration
        hmms: Optional list of HMMs (needed for cascade mode)

    Returns:
        Dict of kwargs to pass to pyhmmer.hmmsearch etc.
    """
    kwargs: dict[str, Any] = {}

    if cutoff := options.get_explicit_cutoff():
        kwargs["bit_cutoffs"] = cutoff.value
        return kwargs

    if options.cascade:
        # Caller handles per-HMM cascade; no shared kwargs
        pass

    if options.evalue is not None:
        kwargs["E"] = options.evalue
    if options.bitscore is not None:
        kwargs["T"] = options.bitscore
    if options.dom_evalue is not None:
        kwargs["domE"] = options.dom_evalue
    if options.dom_bitscore is not None:
        kwargs["domT"] = options.dom_bitscore
    if options.inc_evalue is not None:
        kwargs["incE"] = options.inc_evalue
    if options.inc_bitscore is not None:
        kwargs["incT"] = options.inc_bitscore
    if options.inc_dom_evalue is not None:
        kwargs["incdomE"] = options.inc_dom_evalue
    if options.inc_dom_bitscore is not None:
        kwargs["incdomT"] = options.inc_dom_bitscore

    return kwargs


def group_hmms_by_cutoff(
    hmms: list[HMM],
    options: ThresholdOptions,
) -> dict[Optional[str], list[HMM]]:
    """Group HMMs by their best available cutoff.

    Used for cascade mode where different HMMs may have different
    threshold types available.

    Args:
        hmms: List of HMMs to group
        options: Threshold options (determines preferred order)

    Returns:
        Dict mapping cutoff type (or None) to list of HMMs
    """
    if not options.cascade:
        cutoff = options.get_explicit_cutoff()
        return {cutoff.value if cutoff else None: hmms}

    if options.cut_tc:
        preferred = ("trusted", "gathering", "noise")
    elif options.cut_ga:
        preferred = ("gathering", "trusted", "noise")
    elif options.cut_nc:
        preferred = ("noise", "trusted", "gathering")
    else:
        preferred = ("trusted", "gathering", "noise")

    groups: dict[Optional[str], list[HMM]] = {}

    for hmm in hmms:
        best = get_best_available_cutoff(hmm, preferred)
        if best not in groups:
            groups[best] = []
        groups[best].append(hmm)

    return groups
