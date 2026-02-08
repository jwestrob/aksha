"""Core type definitions for Aksha."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Optional, Union, Sequence, Iterator, TypeAlias

import pandas as pd
import pyhmmer.easel
import pyhmmer.plan7

# Type aliases
PathLike: TypeAlias = Union[str, Path]
HMM: TypeAlias = pyhmmer.plan7.HMM
SequenceBlock: TypeAlias = pyhmmer.easel.DigitalSequenceBlock
Alphabet: TypeAlias = pyhmmer.easel.Alphabet


class MoleculeType(Enum):
    """Sequence molecule type."""

    PROTEIN = auto()
    NUCLEOTIDE = auto()


class BitscoreCutoff(Enum):
    """HMM bitscore cutoff types."""

    GATHERING = "gathering"
    TRUSTED = "trusted"
    NOISE = "noise"


@dataclass(frozen=True)
class ThresholdOptions:
    """Search threshold configuration.

    Precedence (highest to lowest):
    1. Explicit cutoff type (cut_ga, cut_tc, cut_nc)
    2. Cascade mode (try available cutoffs in order)
    3. Numeric thresholds (evalue, bitscore, etc.)
    4. PyHMMER defaults

    Args:
        cut_ga: Use gathering thresholds from HMM file
        cut_tc: Use trusted cutoffs from HMM file
        cut_nc: Use noise cutoffs from HMM file
        cascade: Try cutoffs in order (trusted -> gathering -> noise -> none)
        evalue: E-value threshold for full sequence
        bitscore: Bitscore threshold for full sequence
        dom_evalue: Domain E-value threshold
        dom_bitscore: Domain bitscore threshold
        inc_evalue: Inclusion E-value threshold
        inc_bitscore: Inclusion bitscore threshold
        inc_dom_evalue: Domain inclusion E-value threshold
        inc_dom_bitscore: Domain inclusion bitscore threshold
    """

    cut_ga: bool = False
    cut_tc: bool = False
    cut_nc: bool = False
    cascade: bool = False
    evalue: Optional[float] = None
    bitscore: Optional[float] = None
    dom_evalue: Optional[float] = None
    dom_bitscore: Optional[float] = None
    inc_evalue: Optional[float] = None
    inc_bitscore: Optional[float] = None
    inc_dom_evalue: Optional[float] = None
    inc_dom_bitscore: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate threshold options."""
        cutoff_count = sum([self.cut_ga, self.cut_tc, self.cut_nc])
        if cutoff_count > 1:
            raise ValueError("Only one of cut_ga, cut_tc, cut_nc can be True")

    def get_explicit_cutoff(self) -> Optional[BitscoreCutoff]:
        """Return explicitly requested cutoff type, if any."""
        if self.cut_ga:
            return BitscoreCutoff.GATHERING
        if self.cut_tc:
            return BitscoreCutoff.TRUSTED
        if self.cut_nc:
            return BitscoreCutoff.NOISE
        return None


@dataclass(frozen=True, slots=True)
class DomainHit:
    """A single domain hit within a sequence."""

    c_evalue: float
    i_evalue: float
    bitscore: float
    env_from: int
    env_to: int
    ali_from: int
    ali_to: int


@dataclass(frozen=True, slots=True)
class SequenceHit:
    """A sequence hit from HMM search."""

    sequence_id: str
    hmm_name: str
    evalue: float
    bitscore: float
    domains: tuple[DomainHit, ...]

    @property
    def best_domain(self) -> Optional[DomainHit]:
        """Return highest-scoring domain."""
        if not self.domains:
            return None
        return max(self.domains, key=lambda d: d.bitscore)


@dataclass(slots=True)
class SearchResult:
    """Container for search results with uniform interface.

    Handles both in-memory and on-disk storage transparently.
    """

    hits: list[SequenceHit] = field(default_factory=list)
    _dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)
    _output_path: Optional[Path] = field(default=None, repr=False)
    _count: int = field(default=0, repr=False)

    def __len__(self) -> int:
        if self._output_path is not None and self._count:
            return self._count
        if self._dataframe is not None:
            return len(self._dataframe)
        return len(self.hits)

    def __iter__(self) -> Iterator[SequenceHit]:
        if self._output_path is not None:
            return self._iter_from_disk()
        return iter(self.hits)

    def _iter_from_disk(self) -> Iterator[SequenceHit]:
        """Stream SequenceHit objects from the on-disk TSV."""
        with open(self._output_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            next(reader)  # skip header
            for key, group in groupby(reader, key=itemgetter(0, 1)):
                sequence_id, hmm_name = key
                domains: list[DomainHit] = []
                seq_evalue = 0.0
                seq_bitscore = 0.0
                for row in group:
                    seq_evalue = float(row[2])
                    seq_bitscore = float(row[3])
                    domains.append(
                        DomainHit(
                            c_evalue=float(row[4]),
                            i_evalue=float(row[5]),
                            bitscore=float(row[6]),
                            env_from=int(row[7]),
                            env_to=int(row[8]),
                            ali_from=int(row[9]),
                            ali_to=int(row[10]),
                        )
                    )
                yield SequenceHit(
                    sequence_id=sequence_id,
                    hmm_name=hmm_name,
                    evalue=seq_evalue,
                    bitscore=seq_bitscore,
                    domains=tuple(domains),
                )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame.

        Expands domain hits into separate rows.
        """
        columns = [
            "sequence_id",
            "hmm_name",
            "evalue",
            "bitscore",
            "c_evalue",
            "i_evalue",
            "dom_bitscore",
            "env_from",
            "env_to",
            "ali_from",
            "ali_to",
        ]

        if self._dataframe is not None:
            return self._dataframe

        if self._output_path is not None:
            return pd.read_csv(self._output_path, sep="\t")

        rows = []
        for hit in self.hits:
            for domain in hit.domains:
                rows.append(
                    {
                        "sequence_id": hit.sequence_id,
                        "hmm_name": hit.hmm_name,
                        "evalue": hit.evalue,
                        "bitscore": hit.bitscore,
                        "c_evalue": domain.c_evalue,
                        "i_evalue": domain.i_evalue,
                        "dom_bitscore": domain.bitscore,
                        "env_from": domain.env_from,
                        "env_to": domain.env_to,
                        "ali_from": domain.ali_from,
                        "ali_to": domain.ali_to,
                    }
                )

        return pd.DataFrame(rows, columns=columns)

    def to_csv(self, path: PathLike, **kwargs) -> Path:
        """Write results to TSV file."""
        path = Path(path)

        if self._output_path is not None:
            src = self._output_path.resolve()
            dst = path.resolve()
            if src == dst:
                return path
            shutil.copy2(src, dst)
            return path

        df = self.to_dataframe()
        df.to_csv(
            path,
            sep="\t",
            index=False,
            lineterminator="\n",
            float_format="%.3g",
            **kwargs,
        )
        return path

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SearchResult":
        """Create SearchResult from DataFrame."""
        result = cls()
        result._dataframe = df
        return result

    @classmethod
    def from_file(cls, path: PathLike, count: int = 0) -> "SearchResult":
        """Create SearchResult referencing an output file."""
        result = cls()
        result._output_path = Path(path)
        result._count = count
        return result


@dataclass
class DatabaseInfo:
    """Metadata for an HMM database."""

    name: str
    url: str
    molecule_type: MoleculeType
    citation: str
    notes: Optional[str] = None
    domain: bool = False  # True if contains domain-level models
    has_thresholds: bool = False  # True if models include GA/TC/NC cutoffs
    installed: bool = False
    path: Optional[str] = None  # Relative to data_dir, or absolute
    version: Optional[str] = None

    def resolve_path(self, data_dir: Path) -> Optional[Path]:
        """Resolve path relative to data directory."""
        if self.path is None:
            return None
        p = Path(self.path)
        if p.is_absolute():
            return p
        return data_dir / p
