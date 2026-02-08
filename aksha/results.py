"""Search result handling and output formatting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Iterator, TextIO

from aksha.types import (
    PathLike,
    SearchResult,
    SequenceHit,
    DomainHit,
)

logger = logging.getLogger(__name__)

# Threshold for switching to disk-based storage
MAX_IN_MEMORY_RESULTS = 100_000


class ResultCollector:
    """Collects search results, switching to disk for large datasets.

    Usage:
        collector = ResultCollector(output_dir)
        for hit in search_results:
            collector.add(hit)
        result = collector.finalize()
    """

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        force_disk: bool = False,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.force_disk = force_disk

        self._hits: list[SequenceHit] = []
        self._temp_file: Optional[TextIO] = None
        self._temp_path: Optional[Path] = None
        self._count = 0
        self._using_disk = force_disk

    def add(self, hit: SequenceHit) -> None:
        """Add a hit to the collection."""
        self._count += 1

        if self._using_disk:
            self._write_to_disk(hit)
        elif self._count > MAX_IN_MEMORY_RESULTS:
            self._switch_to_disk()
            self._write_to_disk(hit)
        else:
            self._hits.append(hit)

    def _switch_to_disk(self) -> None:
        """Switch from memory to disk storage."""
        logger.info("Switching to disk storage after %d results", self._count)
        self._using_disk = True

        if self.output_dir is None:
            self.output_dir = Path.cwd()

        self._temp_path = self.output_dir / ".aksha_temp_results.tsv"
        self._temp_file = open(self._temp_path, "w", encoding="utf-8")
        self._write_header()

        for hit in self._hits:
            self._write_to_disk(hit)
        self._hits.clear()

    def _write_header(self) -> None:
        """Write TSV header."""
        if self._temp_file:
            self._temp_file.write(
                "sequence_id\thmm_name\tevalue\tbitscore\t"
                "c_evalue\ti_evalue\tdom_bitscore\t"
                "env_from\tenv_to\tali_from\tali_to\n"
            )

    def _write_to_disk(self, hit: SequenceHit) -> None:
        """Write a single hit to disk."""
        if self._temp_file is None:
            self._temp_path = self.output_dir / ".aksha_temp_results.tsv"
            self._temp_file = open(self._temp_path, "w", encoding="utf-8")
            self._write_header()

        for domain in hit.domains:
            self._temp_file.write(
                f"{hit.sequence_id}\t{hit.hmm_name}\t{hit.evalue:.2e}\t{hit.bitscore:.2f}\t"
                f"{domain.c_evalue:.2e}\t{domain.i_evalue:.2e}\t{domain.bitscore:.2f}\t"
                f"{domain.env_from}\t{domain.env_to}\t{domain.ali_from}\t{domain.ali_to}\n"
            )

    def finalize(self) -> SearchResult:
        """Finalize collection and return SearchResult."""
        if self._temp_file:
            self._temp_file.close()
            self._temp_file = None

        if self._using_disk and self._temp_path:
            return SearchResult.from_file(self._temp_path, count=self._count)

        return SearchResult(hits=self._hits)

    def __enter__(self) -> "ResultCollector":
        return self

    def __exit__(self, *args) -> None:
        if self._temp_file:
            self._temp_file.close()


def hits_from_pyhmmer(
    pyhmmer_hits,
    hmm_name: str,
    *,
    sequence_id: Optional[str] = None,
    skip_duplicates: bool = False,
) -> Iterator[SequenceHit]:
    """Convert PyHMMER hits to SequenceHit objects.

    Args:
        pyhmmer_hits: TopHits object from PyHMMER
        hmm_name: Name of the query HMM (or query sequence for phmmer)
        sequence_id: If set, use this as sequence_id and hit.name as hmm_name
            (for hmmscan where query is a sequence and hits are HMMs)
        skip_duplicates: Skip hits flagged as duplicates (for jackhmmer)

    Yields:
        SequenceHit objects for included hits
    """
    for hit in pyhmmer_hits:
        if not hit.included:
            continue
        if skip_duplicates and hit.duplicate:
            continue

        domains = tuple(
            DomainHit(
                c_evalue=domain.c_evalue,
                i_evalue=domain.i_evalue,
                bitscore=domain.score,
                env_from=domain.env_from,
                env_to=domain.env_to,
                ali_from=domain.alignment.target_from,
                ali_to=domain.alignment.target_to,
            )
            for domain in hit.domains.reported
        )

        if sequence_id is not None:
            # scan mode: query is a sequence, hit is an HMM
            yield SequenceHit(
                sequence_id=sequence_id,
                hmm_name=hit.name.decode(),
                evalue=hit.evalue,
                bitscore=hit.score,
                domains=domains,
            )
        else:
            yield SequenceHit(
                sequence_id=hit.name.decode(),
                hmm_name=hmm_name,
                evalue=hit.evalue,
                bitscore=hit.score,
                domains=domains,
            )
