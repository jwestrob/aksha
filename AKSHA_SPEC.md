# Aksha: Complete Rewrite Specification

## Overview

Aksha is a Python library and CLI tool for scalable HMM-based sequence searching. It wraps PyHMMER to provide a user-friendly interface for bioinformatics workflows, with built-in support for downloading and managing HMM databases.

**Name etymology:** Sanskrit अक्ष (akṣa) — axis, eye, dice, wheel. The axis around which sequence analysis turns.

**Goals:**
- Clean library API for programmatic use
- CLI for interactive/pipeline use
- Unified configuration and database management
- Type hints throughout
- No global state
- Minimal code duplication

---

## Post-Implementation Changes

The following architectural changes were made during implementation:

1. **`_compat.py` removed** — Python version compat (tomli/tomllib) lives in `config.py`; `tomli` added as conditional dependency.
2. **`platformdirs` replaces hand-rolled XDG** — `config.py` uses `platformdirs.user_config_dir("aksha")` / `user_data_dir("aksha")` instead of manual `XDG_*` fallbacks. Env var overrides (`AKSHA_CONFIG_DIR`, `AKSHA_DATA_DIR`) still work.
3. **Lazy sequence iteration** — `iter_sequences()` added in `parsers.py`; all five search pipelines use it instead of `parse_sequences()` so only one file's sequences are in memory at a time.
4. **`hits_from_pyhmmer` consolidated** — `results.py` now has `sequence_id` and `skip_duplicates` keyword args so all five pipelines share a single hit-extraction path.
5. **`SearchResult` memory fixes** — `_count` field tracks hit count for disk-backed results; `__len__` and `__iter__` work correctly when `_output_path` is set; `to_csv()` copies the file directly via `shutil.copy2` instead of round-tripping through pandas.
6. **`_save_state` custom DB fix** — Custom databases now persist `custom: True`, `molecule_type`, and `citation` so they survive process restarts.
7. **Tarfile security** — `databases.py` filters tar members through `_safe_tar_members()` to prevent path traversal and symlink attacks.
8. **CLI helpers** — `_resolve_output()` and `_write_result()` deduplicate output logic across all five `_cmd_*` functions; `--incE`, `--incT`, `--incdomE`, `--incdomT` flags added.
9. **Error messages** — `_resolve_hmms` now distinguishes "unknown database" from "known but not installed" with an actionable message.

---

## File Structure

```
aksha/
├── __init__.py           # Public API exports
├── types.py              # Dataclasses, TypedDicts, type aliases
├── config.py             # Configuration management (platformdirs)
├── parsers.py            # HMM and sequence file parsing + iter_sequences
├── thresholds.py         # Threshold logic (cascade, cutoffs)
├── results.py            # ResultCollector, hits_from_pyhmmer
├── search.py             # hmmsearch implementation
├── scan.py               # hmmscan implementation
├── phmmer.py             # phmmer implementation
├── jackhmmer.py          # jackhmmer implementation
├── nhmmer.py             # nhmmer (nucleotide search)
├── databases.py          # Database installation and management
├── cli.py                # Argparse CLI, all subcommands
├── data/
│   └── databases.json    # Bundled database definitions (URLs, citations)
├── py.typed              # PEP 561 marker
tests/
├── conftest.py           # Pytest fixtures
├── test_search.py        # Search pipeline tests
├── test_barrnap.py       # barrnap nhmmer integration test
└── fixtures/
    ├── small.faa         # 10 protein sequences
    ├── small.fna         # 10 nucleotide sequences
    ├── small.hmm         # 2-3 HMM profiles
    ├── small_nuc.hmm     # Nucleotide HMM profile
    ├── cutoff_ga.hmm     # HMM with gathering cutoffs
    └── rrna_16S.fa       # 16S rRNA test sequences
```

---

## Dependencies

```toml
[project]
name = "aksha"
version = "0.1.0"
description = "Scalable HMM-based sequence search and retrieval"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Jacob West-Roberts" }]
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
dependencies = [
    "pyhmmer>=0.10.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "requests>=2.28.0",
    "platformdirs>=3.0.0",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
aksha = "aksha.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["aksha"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true

[tool.ruff]
line-length = 100
target-version = "py310"
```

---

## Module Specifications

### `types.py`

Define all shared types. No business logic here.

```python
"""Core type definitions for Aksha."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
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


@dataclass(frozen=True)
class DomainHit:
    """A single domain hit within a sequence."""
    c_evalue: float
    i_evalue: float
    bitscore: float
    env_from: int
    env_to: int
    ali_from: int
    ali_to: int


@dataclass(frozen=True)
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


@dataclass
class SearchResult:
    """Container for search results with uniform interface.

    Handles both in-memory and on-disk storage transparently.
    Disk-backed results stream from TSV without loading into memory.
    """
    hits: list[SequenceHit] = field(default_factory=list)
    _dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)
    _output_path: Optional[Path] = field(default=None, repr=False)
    _count: int = field(default=0, repr=False)

    def __len__(self) -> int:
        if self._count:
            return self._count
        if self._dataframe is not None:
            return len(self._dataframe)
        return len(self.hits)

    def __iter__(self) -> Iterator[SequenceHit]:
        if self._output_path is not None:
            return self._iter_from_disk()
        return iter(self.hits)

    def _iter_from_disk(self) -> Iterator[SequenceHit]:
        """Stream SequenceHit objects from disk TSV without loading all into memory."""
        import csv
        from itertools import groupby
        from operator import itemgetter

        with open(self._output_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            next(reader)  # skip header
            for key, group in groupby(reader, key=itemgetter(0, 1)):
                sequence_id, hmm_name = key
                domains: list[DomainHit] = []
                seq_evalue = seq_bitscore = 0.0
                for row in group:
                    seq_evalue = float(row[2])
                    seq_bitscore = float(row[3])
                    domains.append(DomainHit(
                        c_evalue=float(row[4]), i_evalue=float(row[5]),
                        bitscore=float(row[6]), env_from=int(row[7]),
                        env_to=int(row[8]), ali_from=int(row[9]), ali_to=int(row[10]),
                    ))
                yield SequenceHit(
                    sequence_id=sequence_id, hmm_name=hmm_name,
                    evalue=seq_evalue, bitscore=seq_bitscore, domains=tuple(domains),
                )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame.  Expands domain hits into separate rows."""
        if self._dataframe is not None:
            return self._dataframe
        if self._output_path is not None:
            return pd.read_csv(self._output_path, sep="\t")
        rows = []
        for hit in self.hits:
            for domain in hit.domains:
                rows.append({
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
                })
        return pd.DataFrame(rows)

    def to_csv(self, path: PathLike, **kwargs) -> Path:
        """Write results to TSV file.  Copies directly for disk-backed results."""
        import shutil
        path = Path(path)
        if self._output_path is not None and self._output_path != path:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._output_path, path)
            return path
        df = self.to_dataframe()
        df.to_csv(path, sep="\t", index=False, **kwargs)
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
```

---

### `config.py`

Configuration management using `platformdirs` for cross-platform path resolution.

```python
"""Configuration management for Aksha.

Path resolution precedence:
1. Environment variables (AKSHA_DATA_DIR, AKSHA_CONFIG_DIR)
2. User config file (<platformdirs config>/config.toml)
3. platformdirs defaults (platform-appropriate locations)

Example config.toml:
    data_dir = "/scratch/shared/hmm_databases"

    [defaults]
    threads = 8
    cut_ga = true
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import platformdirs

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from aksha.types import DatabaseInfo, MoleculeType


@dataclass
class AkshaConfig:
    """Global configuration for Aksha."""
    
    data_dir: Path
    config_dir: Path
    defaults: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load(cls) -> "AkshaConfig":
        """Load configuration from files and environment."""
        config_dir = cls._resolve_config_dir()
        config_file = config_dir / "config.toml"
        
        user_config: dict[str, Any] = {}
        if config_file.exists():
            with open(config_file, "rb") as f:
                user_config = tomllib.load(f)
        
        data_dir = cls._resolve_data_dir(user_config)
        
        return cls(
            data_dir=data_dir,
            config_dir=config_dir,
            defaults=user_config.get("defaults", {}),
        )
    
    @staticmethod
    def _resolve_config_dir() -> Path:
        """Resolve config directory path."""
        if env := os.environ.get("AKSHA_CONFIG_DIR"):
            return Path(env).expanduser()
        return Path(platformdirs.user_config_dir("aksha"))

    @staticmethod
    def _resolve_data_dir(user_config: dict[str, Any]) -> Path:
        """Resolve data directory path."""
        if env := os.environ.get("AKSHA_DATA_DIR"):
            return Path(env).expanduser()
        if config_path := user_config.get("data_dir"):
            return Path(config_path).expanduser()
        return Path(platformdirs.user_data_dir("aksha"))
    
    def ensure_dirs(self) -> None:
        """Create config and data directories if needed."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def resolve_database_path(self, db_path: str) -> Path:
        """Resolve database path (absolute or relative to data_dir)."""
        p = Path(db_path)
        if p.is_absolute():
            return p
        return self.data_dir / p


class DatabaseRegistry:
    """Manages HMM database metadata and installation state.
    
    Separates bundled database definitions (immutable) from local state
    (installed status, paths, versions).
    """
    
    def __init__(self, config: AkshaConfig):
        self.config = config
        self._state_file = config.data_dir / "databases.json"
        self._entries: dict[str, DatabaseInfo] = {}
        self._load()
    
    def _load(self) -> None:
        """Load database definitions and local state."""
        # Load bundled definitions
        bundled_path = Path(__file__).parent / "data" / "databases.json"
        with open(bundled_path) as f:
            bundled = json.load(f)
        
        # Load local state
        local_state: dict[str, dict] = {}
        if self._state_file.exists():
            with open(self._state_file) as f:
                local_state = json.load(f)
        
        # Merge bundled + local
        for db in bundled["databases"]:
            name = db["name"]
            state = local_state.get(name, {})
            
            self._entries[name] = DatabaseInfo(
                name=name,
                url=db["url"],
                molecule_type=MoleculeType[db["molecule_type"].upper()],
                citation=db.get("citation", ""),
                notes=db.get("notes"),
                domain=db.get("domain", False),
                has_thresholds=db.get("has_thresholds", False),
                installed=state.get("installed", False),
                path=state.get("path"),
                version=state.get("version"),
            )
        
        # Load custom databases
        for name, state in local_state.items():
            if name not in self._entries and state.get("custom"):
                self._entries[name] = DatabaseInfo(
                    name=name,
                    url=state.get("url", ""),
                    molecule_type=MoleculeType[state["molecule_type"].upper()],
                    citation=state.get("citation", ""),
                    installed=state.get("installed", False),
                    path=state.get("path"),
                )
    
    def _save_state(self) -> None:
        """Persist installation state to disk."""
        state = {}
        for name, entry in self._entries.items():
            if entry.installed or entry.path:
                rec: dict[str, Any] = {
                    "installed": entry.installed,
                    "path": entry.path,
                    "version": entry.version,
                }
                if name not in self._bundled_names:
                    rec["custom"] = True
                    rec["molecule_type"] = entry.molecule_type.name.lower()
                    rec["citation"] = entry.citation
                state[name] = rec
        
        self.config.ensure_dirs()
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def get(self, name: str) -> Optional[DatabaseInfo]:
        """Get database by name."""
        return self._entries.get(name)
    
    def list_available(
        self, 
        molecule_type: Optional[MoleculeType] = None,
        installed_only: bool = False,
    ) -> list[DatabaseInfo]:
        """List databases, optionally filtered."""
        entries = list(self._entries.values())
        
        if molecule_type:
            entries = [e for e in entries if e.molecule_type == molecule_type]
        if installed_only:
            entries = [e for e in entries if e.installed]
        
        return sorted(entries, key=lambda e: e.name)
    
    def mark_installed(
        self, 
        name: str, 
        path: str, 
        version: Optional[str] = None,
    ) -> None:
        """Mark database as installed."""
        if name not in self._entries:
            raise ValueError(f"Unknown database: {name}")
        
        self._entries[name].installed = True
        self._entries[name].path = path
        self._entries[name].version = version
        self._save_state()
    
    def mark_uninstalled(self, name: str) -> None:
        """Mark database as uninstalled."""
        if name in self._entries:
            self._entries[name].installed = False
            self._entries[name].path = None
            self._save_state()
    
    def get_path(self, name: str) -> Optional[Path]:
        """Get resolved filesystem path for installed database."""
        entry = self.get(name)
        if entry and entry.installed and entry.path:
            return entry.resolve_path(self.config.data_dir)
        return None
    
    def register_custom(
        self,
        name: str,
        path: str,
        molecule_type: MoleculeType,
        citation: str = "",
    ) -> None:
        """Register a custom database not in bundled list."""
        if name in self._entries:
            raise ValueError(f"Database already exists: {name}")
        
        self._entries[name] = DatabaseInfo(
            name=name,
            url="",
            molecule_type=molecule_type,
            citation=citation,
            installed=True,
            path=path,
        )
        self._save_state()


# Module-level convenience functions
_config: Optional[AkshaConfig] = None
_registry: Optional[DatabaseRegistry] = None


def get_config() -> AkshaConfig:
    """Get global config instance (lazy loaded)."""
    global _config
    if _config is None:
        _config = AkshaConfig.load()
    return _config


def get_registry() -> DatabaseRegistry:
    """Get global database registry (lazy loaded)."""
    global _registry
    if _registry is None:
        _registry = DatabaseRegistry(get_config())
    return _registry


def reset_globals() -> None:
    """Reset global instances (useful for testing)."""
    global _config, _registry
    _config = None
    _registry = None
```

---

### `parsers.py`

Unified parsing for HMMs and sequences.

```python
"""Parsing utilities for HMM and sequence files.

Supports multiple input types for flexibility:
- File paths (str or Path)
- Directories of files
- Pre-loaded PyHMMER objects
- Lists of sequences
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union, Sequence, Optional, Iterator, overload

import pyhmmer.easel
import pyhmmer.plan7
from tqdm import tqdm

from aksha.types import (
    PathLike, 
    HMM, 
    SequenceBlock, 
    Alphabet,
    MoleculeType,
)

logger = logging.getLogger(__name__)


# Type for flexible HMM input
HMMInput = Union[
    PathLike,                      # Single file or directory
    HMM,                           # Single HMM object
    Sequence[HMM],                 # List of HMM objects
    pyhmmer.plan7.HMMFile,         # Open HMM file
]

# Type for flexible sequence input
SequenceInput = Union[
    PathLike,                      # Single file or directory
    SequenceBlock,                 # Pre-loaded sequences
    Sequence[pyhmmer.easel.DigitalSequence],  # List of sequences
]


def parse_hmms(
    source: HMMInput,
    *,
    show_progress: bool = True,
) -> list[HMM]:
    """Parse HMMs from various input types.
    
    Args:
        source: HMM file path, directory, or pre-loaded HMMs
        show_progress: Show progress bar for directory parsing
        
    Returns:
        List of HMM objects
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If input is empty or invalid
    """
    # Already HMM objects
    if isinstance(source, pyhmmer.plan7.HMM):
        return [source]
    
    if isinstance(source, (list, tuple)):
        if all(isinstance(h, pyhmmer.plan7.HMM) for h in source):
            return list(source)
    
    if isinstance(source, pyhmmer.plan7.HMMFile):
        return list(source)
    
    # Path-based input
    path = Path(source)
    
    if not path.exists():
        raise FileNotFoundError(f"HMM path not found: {path}")
    
    if path.is_file():
        return _parse_hmm_file(path)
    
    if path.is_dir():
        return _parse_hmm_directory(path, show_progress=show_progress)
    
    raise ValueError(f"Invalid HMM source: {source}")


def _parse_hmm_file(path: Path) -> list[HMM]:
    """Parse single HMM file (may contain multiple models)."""
    if path.stat().st_size == 0:
        raise ValueError(f"HMM file is empty: {path}")
    
    with pyhmmer.plan7.HMMFile(path) as f:
        hmms = list(f)
    
    logger.info(f"Parsed {len(hmms)} HMMs from {path}")
    return hmms


def _parse_hmm_directory(path: Path, show_progress: bool = True) -> list[HMM]:
    """Parse directory of HMM files."""
    hmm_files = [
        f for f in path.iterdir()
        if f.suffix.lower() in (".hmm",)
    ]
    
    if not hmm_files:
        raise ValueError(f"No HMM files found in directory: {path}")
    
    hmms = []
    iterator = tqdm(hmm_files, desc="Parsing HMMs") if show_progress else hmm_files
    
    for hmm_path in iterator:
        try:
            with pyhmmer.plan7.HMMFile(hmm_path) as f:
                hmms.append(f.read())
        except Exception as e:
            logger.warning(f"Failed to parse {hmm_path}: {e}")
    
    logger.info(f"Parsed {len(hmms)} HMMs from {path}")
    return hmms


def parse_sequences(
    source: SequenceInput,
    *,
    molecule_type: MoleculeType = MoleculeType.PROTEIN,
    show_progress: bool = True,
) -> dict[Path, SequenceBlock]:
    """Parse sequences from various input types.
    
    Args:
        source: Sequence file path, directory, or pre-loaded sequences
        molecule_type: PROTEIN or NUCLEOTIDE
        show_progress: Show progress bar for directory parsing
        
    Returns:
        Dict mapping source path to sequence block.
        For pre-loaded sequences, uses a synthetic path.
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If input is empty or invalid
    """
    alphabet = _get_alphabet(molecule_type)
    
    # Already loaded sequences
    if isinstance(source, pyhmmer.easel.DigitalSequenceBlock):
        return {Path("<memory>"): source}
    
    if isinstance(source, (list, tuple)):
        if all(isinstance(s, pyhmmer.easel.DigitalSequence) for s in source):
            # Convert list to block
            block = pyhmmer.easel.DigitalSequenceBlock(alphabet, source)
            return {Path("<memory>"): block}
    
    # Path-based input
    path = Path(source)
    
    if not path.exists():
        raise FileNotFoundError(f"Sequence path not found: {path}")
    
    if path.is_file():
        return {path: _parse_sequence_file(path, alphabet)}
    
    if path.is_dir():
        return _parse_sequence_directory(path, alphabet, show_progress=show_progress)
    
    raise ValueError(f"Invalid sequence source: {source}")


def _get_alphabet(molecule_type: MoleculeType) -> Alphabet:
    """Get PyHMMER alphabet for molecule type."""
    if molecule_type == MoleculeType.PROTEIN:
        return pyhmmer.easel.Alphabet.amino()
    return pyhmmer.easel.Alphabet.dna()


def _parse_sequence_file(path: Path, alphabet: Alphabet) -> SequenceBlock:
    """Parse single sequence file."""
    if path.stat().st_size == 0:
        raise ValueError(f"Sequence file is empty: {path}")
    
    with pyhmmer.easel.SequenceFile(path, digital=True, alphabet=alphabet) as f:
        sequences = f.read_block()
    
    logger.info(f"Parsed {len(sequences)} sequences from {path}")
    return sequences


def _parse_sequence_directory(
    path: Path, 
    alphabet: Alphabet,
    show_progress: bool = True,
) -> dict[Path, SequenceBlock]:
    """Parse directory of sequence files."""
    seq_files = [
        f for f in path.iterdir()
        if f.suffix.lower() in (".faa", ".fna", ".fa", ".fasta", ".fas")
    ]
    
    if not seq_files:
        raise ValueError(f"No sequence files found in directory: {path}")
    
    result = {}
    iterator = tqdm(seq_files, desc="Parsing sequences") if show_progress else seq_files
    
    for seq_path in iterator:
        try:
            result[seq_path] = _parse_sequence_file(seq_path, alphabet)
        except Exception as e:
            logger.warning(f"Failed to parse {seq_path}: {e}")
    
    logger.info(f"Parsed sequences from {len(result)} files in {path}")
    return result


def iter_sequences(
    source: SequenceInput,
    *,
    molecule_type: MoleculeType = MoleculeType.PROTEIN,
    alphabet: Optional[Alphabet] = None,
    show_progress: bool = True,
) -> Iterator[tuple[Path, SequenceBlock]]:
    """Lazily yield (path, SequenceBlock) pairs — one file at a time.

    For directory input, only one file's sequences live in memory at a time.
    For single-file or pre-loaded input, yields the single entry.
    All five search pipelines use this instead of parse_sequences().
    """
    if alphabet is None:
        alphabet = _get_alphabet(molecule_type)

    if isinstance(source, pyhmmer.easel.DigitalSequenceBlock):
        yield Path("<memory>"), source
        return

    if isinstance(source, (list, tuple)):
        if all(isinstance(s, pyhmmer.easel.DigitalSequence) for s in source):
            block = pyhmmer.easel.DigitalSequenceBlock(alphabet, source)
            yield Path("<memory>"), block
            return

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Sequence path not found: {path}")

    if path.is_file():
        yield path, _parse_sequence_file(path, alphabet)
        return

    if path.is_dir():
        seq_files = sorted(
            f for f in path.iterdir()
            if f.suffix.lower() in (".faa", ".fna", ".fa", ".fasta", ".fas")
        )
        if not seq_files:
            raise ValueError(f"No sequence files found in directory: {path}")

        iterator = tqdm(seq_files, desc="Parsing sequences") if show_progress else seq_files
        for seq_path in iterator:
            try:
                yield seq_path, _parse_sequence_file(seq_path, alphabet)
            except Exception as e:
                logger.warning(f"Failed to parse {seq_path}: {e}")
        return

    raise ValueError(f"Invalid sequence source: {source}")


def check_hmm_thresholds(hmm: HMM) -> dict[str, bool]:
    """Check which threshold types are available for an HMM.
    
    Returns:
        Dict with keys 'gathering', 'trusted', 'noise' and bool values
    """
    return {
        "gathering": hmm.cutoffs.gathering_available(),
        "trusted": hmm.cutoffs.trusted_available(),
        "noise": hmm.cutoffs.noise_available(),
    }


def get_best_available_cutoff(
    hmm: HMM, 
    preferred_order: Sequence[str] = ("trusted", "gathering", "noise"),
) -> Optional[str]:
    """Get best available cutoff type for an HMM.
    
    Args:
        hmm: HMM to check
        preferred_order: Order of preference for cutoff types
        
    Returns:
        Cutoff type name or None if no cutoffs available
    """
    available = check_hmm_thresholds(hmm)
    
    for cutoff in preferred_order:
        if available.get(cutoff):
            return cutoff
    
    return None
```

---

### `thresholds.py`

Threshold handling logic, including the cascade system.

```python
"""Threshold handling for HMM searches.

Implements the cascade system that tries multiple threshold types
in order of preference.
"""

from __future__ import annotations

from typing import Any, Optional

import pyhmmer.plan7

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
    
    # Handle explicit cutoff
    if cutoff := options.get_explicit_cutoff():
        kwargs["bit_cutoffs"] = cutoff.value
        return kwargs
    
    # Cascade mode is handled separately per-HMM
    if options.cascade:
        # Return empty - caller handles grouping
        pass
    
    # Numeric thresholds
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
        # No cascade - use explicit cutoff for all
        cutoff = options.get_explicit_cutoff()
        return {cutoff.value if cutoff else None: hmms}
    
    # Determine preference order
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
```

---

### `results.py`

Result handling, including disk-based storage for large datasets.

```python
"""Search result handling and output formatting."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Iterator, TextIO

import pandas as pd

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
        logger.info(f"Switching to disk storage after {self._count} results")
        self._using_disk = True
        
        if self.output_dir is None:
            self.output_dir = Path.cwd()
        
        self._temp_path = self.output_dir / ".aksha_temp_results.tsv"
        self._temp_file = open(self._temp_path, "w")
        self._write_header()
        
        # Flush existing hits
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
            self._temp_file = open(self._temp_path, "w")
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
        hmm_name: Name of the query HMM
        sequence_id: Override sequence_id (used by scan where query is the sequence)
        skip_duplicates: Skip hits marked as duplicates (used by jackhmmer)

    Yields:
        SequenceHit objects for included hits
    """
    for hit in pyhmmer_hits:
        if not hit.included:
            continue
        if skip_duplicates and getattr(hit, "duplicate", False):
            continue

        domains = tuple(
            DomainHit(
                c_evalue=domain.c_evalue,
                i_evalue=domain.i_evalue,
                bitscore=domain.score,
                env_from=domain.env_from,
                env_to=domain.env_to,
                ali_from=domain.ali_from,
                ali_to=domain.ali_to,
            )
            for domain in hit.domains.reported
        )

        yield SequenceHit(
            sequence_id=sequence_id or hit.name.decode(),
            hmm_name=hmm_name if not sequence_id else hit.name.decode(),
            evalue=hit.evalue,
            bitscore=hit.score,
            domains=domains,
        )
```

---

### `search.py`

Core hmmsearch implementation.

```python
"""HMM search implementation using PyHMMER."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pyhmmer
from tqdm import tqdm

from aksha.types import (
    PathLike,
    MoleculeType,
    ThresholdOptions,
    SearchResult,
    HMM,
    SequenceBlock,
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
    """Search sequences against HMM profiles using hmmsearch."""
    if thresholds is None:
        thresholds = ThresholdOptions()

    hmm_list = _resolve_hmms(hmms)

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
            show_progress=show_progress,
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
        if registry.get(source) is not None:
            raise FileNotFoundError(
                f"Database '{source}' is not installed. "
                f"Run: aksha database install {source}"
            )
    return parse_hmms(source)


def _run_hmmsearch(
    sequence_iter,
    hmms: list[HMM],
    thresholds: ThresholdOptions,
    threads: int,
    collector: ResultCollector,
    show_progress: bool,
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
```

---

### `scan.py`

hmmscan implementation (sequences as queries, HMMs as database). Uses `hits_from_pyhmmer`
with `sequence_id` override to flip the query/target perspective.

```python
"""HMM scan implementation using PyHMMER."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pyhmmer

from aksha.types import PathLike, MoleculeType, ThresholdOptions, SearchResult
from aksha.parsers import parse_hmms, iter_sequences, HMMInput, SequenceInput
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
    """Scan sequences against HMM database using hmmscan."""
    if thresholds is None:
        thresholds = ThresholdOptions()

    hmm_list = _resolve_hmms(hmms)
    sequence_iter = iter_sequences(
        sequences, molecule_type=MoleculeType.PROTEIN, show_progress=show_progress,
    )

    output_path = Path(output_dir) if output_dir else None
    hmm_groups = group_hmms_by_cutoff(hmm_list, thresholds)
    base_kwargs = build_search_kwargs(thresholds)
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        for seq_path, sequences_block in sequence_iter:
            for cutoff_type, hmm_group in hmm_groups.items():
                kwargs = base_kwargs.copy()
                if cutoff_type:
                    kwargs["bit_cutoffs"] = cutoff_type
                # hmmscan: sequences as queries, HMMs as database
                for hits in pyhmmer.hmmscan(sequences_block, hmm_group, cpus=cpus, **kwargs):
                    seq_name = hits.query_name.decode()
                    for hit in hits_from_pyhmmer(hits, hmm_name="", sequence_id=seq_name):
                        collector.add(hit)
        return collector.finalize()
```

---

### `phmmer.py`

Uses `iter_sequences` with a single-file guard.

```python
"""phmmer: protein sequence vs protein sequence search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyhmmer

from aksha.types import PathLike, MoleculeType, ThresholdOptions, SearchResult
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
    """Search query sequences against target sequences using phmmer."""
    if thresholds is None:
        thresholds = ThresholdOptions()

    query_iter = iter_sequences(query, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, query_seqs = next(query_iter)
    if next(query_iter, None) is not None:
        raise ValueError("phmmer expects a single query file, got a directory with multiple files")

    target_iter = iter_sequences(target, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, target_seqs = next(target_iter)

    kwargs = build_search_kwargs(thresholds)
    output_path = Path(output_dir) if output_dir else None
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        for hits in pyhmmer.phmmer(query_seqs, target_seqs, cpus=cpus, **kwargs):
            query_name = hits.query_name.decode()
            for hit in hits_from_pyhmmer(hits, query_name):
                collector.add(hit)
        return collector.finalize()
```

---

### `jackhmmer.py`

Uses `iter_sequences` with single-file guard. Uses `hits_from_pyhmmer` with `skip_duplicates=True`.
Note: `final_results.hits` (not `final_results` itself) must be passed to `hits_from_pyhmmer`
because `IterationResult` iterates over HMMs, not hits.

```python
"""jackhmmer: iterative protein sequence search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyhmmer

from aksha.types import PathLike, MoleculeType, ThresholdOptions, SearchResult
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
    """Iteratively search query sequences against targets using jackhmmer."""
    if thresholds is None:
        thresholds = ThresholdOptions()

    query_iter = iter_sequences(query, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, query_seqs = next(query_iter)
    if next(query_iter, None) is not None:
        raise ValueError("jackhmmer expects a single query file, got a directory with multiple files")

    target_iter = iter_sequences(target, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    _, target_seqs = next(target_iter)

    kwargs = build_search_kwargs(thresholds)
    output_path = Path(output_dir) if output_dir else None
    cpus = _resolve_cpus(threads)

    with ResultCollector(output_dir=output_path) as collector:
        search_iter = pyhmmer.jackhmmer(
            query_seqs, target_seqs,
            cpus=cpus, max_iterations=max_iterations, **kwargs,
        )

        final_results = None
        for iteration in search_iter:
            final_results = iteration

        if final_results is None:
            return collector.finalize()

        # .hits gives TopHits; iterating IterationResult directly yields HMMs
        for hit in hits_from_pyhmmer(final_results.hits, "jackhmmer_query", skip_duplicates=True):
            collector.add(hit)

        return collector.finalize()
```

---

### `nhmmer.py`

Has its own `_resolve_nhmmer_hmms` (separate from search's) because nhmmer needs
to probe the HMM alphabet for correct sequence digitization.

```python
"""nhmmer: nucleotide HMM search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pyhmmer

from aksha.types import PathLike, MoleculeType, ThresholdOptions, SearchResult
from aksha.parsers import parse_hmms, iter_sequences, HMMInput, SequenceInput
from aksha.thresholds import build_search_kwargs
from aksha.results import ResultCollector, hits_from_pyhmmer
from aksha.search import _resolve_cpus
from aksha.config import get_registry

logger = logging.getLogger(__name__)


def _resolve_nhmmer_hmms(source: Union[HMMInput, str]) -> list:
    """Resolve HMM source for nhmmer (nucleotide models)."""
    if isinstance(source, str) and not Path(source).exists():
        registry = get_registry()
        db_path = registry.get_path(source)
        if db_path:
            return parse_hmms(db_path)
        if registry.get(source) is not None:
            raise FileNotFoundError(
                f"Database '{source}' is not installed. "
                f"Run: aksha database install {source}"
            )
    return parse_hmms(source)


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

    hmm_list = _resolve_nhmmer_hmms(hmms)
    alphabet = hmm_list[0].alphabet if hmm_list else None

    sequence_iter = iter_sequences(
        sequences, molecule_type=MoleculeType.NUCLEOTIDE,
        alphabet=alphabet, show_progress=show_progress,
    )

    kwargs = build_search_kwargs(thresholds)
    if kwargs.get("bit_cutoffs") and not all(h.cutoffs.gathering_available() for h in hmm_list):
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
```

---

### `databases.py`

Database installation and management.

```python
"""Database installation and management."""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Callable
from urllib.request import urlretrieve

import pandas as pd
import pyhmmer.plan7
import requests
from tqdm import tqdm

from aksha.types import MoleculeType, DatabaseInfo
from aksha.config import get_config, get_registry, AkshaConfig, DatabaseRegistry

logger = logging.getLogger(__name__)


class DownloadProgress(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def list_databases(
    molecule_type: Optional[MoleculeType] = None,
    installed_only: bool = False,
) -> list[DatabaseInfo]:
    """List available databases.
    
    Args:
        molecule_type: Filter by molecule type
        installed_only: Only show installed databases
        
    Returns:
        List of DatabaseInfo objects
    """
    registry = get_registry()
    return registry.list_available(molecule_type=molecule_type, installed_only=installed_only)


def install_database(
    name: str,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Install a database from the registry.
    
    Args:
        name: Database name (e.g., "PFAM", "KOFAM")
        force: Reinstall even if already installed
        show_progress: Show download progress
        
    Returns:
        Path to installed database
        
    Raises:
        ValueError: If database not found in registry
    """
    config = get_config()
    registry = get_registry()
    
    db_info = registry.get(name)
    if db_info is None:
        raise ValueError(f"Unknown database: {name}")
    
    if db_info.installed and not force:
        logger.info(f"Database {name} already installed")
        return db_info.resolve_path(config.data_dir)
    
    # Special handling for KOFAM (needs threshold injection)
    if name == "KOFAM":
        return _install_kofam(config, registry, show_progress)
    
    return _install_standard(name, db_info, config, registry, show_progress)


def _install_standard(
    name: str,
    db_info: DatabaseInfo,
    config: AkshaConfig,
    registry: DatabaseRegistry,
    show_progress: bool,
) -> Path:
    """Standard database installation."""
    config.ensure_dirs()
    target_dir = config.data_dir / name
    target_dir.mkdir(exist_ok=True)
    
    url = db_info.url
    logger.info(f"Downloading {name} from {url}")
    
    # Handle different URL types
    if "github.com" in url:
        _download_github(url, target_dir, show_progress)
    else:
        _download_and_extract(url, target_dir, show_progress)
    
    # Mark as installed (store relative path)
    registry.mark_installed(name, name)
    logger.info(f"Installed {name} to {target_dir}")
    
    return target_dir


def _safe_tar_members(tar: tarfile.TarFile, target_dir: Path):
    """Filter tar members to prevent path traversal attacks."""
    resolved = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if not str(member_path).startswith(str(resolved)):
            logger.warning("Skipping unsafe tar member: %s", member.name)
            continue
        if member.issym() or member.islnk():
            logger.warning("Skipping symbolic link in tar: %s", member.name)
            continue
        yield member


def _download_and_extract(url: str, target_dir: Path, show_progress: bool) -> None:
    """Download and extract a file."""
    filename = url.split("/")[-1]
    download_path = target_dir / filename
    
    # Download
    if show_progress:
        with DownloadProgress(unit="B", unit_scale=True, desc=filename) as pbar:
            urlretrieve(url, download_path, reporthook=pbar.update_to)
    else:
        urlretrieve(url, download_path)
    
    # Extract based on file type
    if tarfile.is_tarfile(download_path):
        logger.info(f"Extracting {filename}")
        with tarfile.open(download_path, "r:*") as tar:
            tar.extractall(target_dir, members=_safe_tar_members(tar, target_dir))
        download_path.unlink()
        _flatten_single_subdir(target_dir)
    
    elif filename.endswith(".gz") and not filename.endswith(".tar.gz"):
        logger.info(f"Decompressing {filename}")
        output_path = download_path.with_suffix("")
        with gzip.open(download_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        download_path.unlink()


def _flatten_single_subdir(target_dir: Path) -> None:
    """If extraction created a single subdirectory, flatten it."""
    contents = list(target_dir.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        subdir = contents[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(target_dir))
        subdir.rmdir()


def _download_github(url: str, target_dir: Path, show_progress: bool) -> None:
    """Download files from a GitHub directory."""
    # Convert to API URL
    url = url.rstrip("/")
    if "/blob/" in url:
        # Single file
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        filename = raw_url.split("/")[-1]
        download_path = target_dir / filename
        urlretrieve(raw_url, download_path)
    else:
        # Directory
        api_url = url.replace("github.com", "api.github.com/repos").replace("/tree/master", "/contents")
        api_url = api_url.replace("/tree/main", "/contents")
        
        response = requests.get(api_url)
        response.raise_for_status()
        
        files = response.json()
        iterator = tqdm(files, desc="Downloading") if show_progress else files
        
        for file_info in iterator:
            if file_info["name"].lower().endswith(".hmm"):
                download_url = file_info["download_url"]
                download_path = target_dir / file_info["name"]
                urlretrieve(download_url, download_path)


def _install_kofam(
    config: AkshaConfig,
    registry: DatabaseRegistry,
    show_progress: bool,
) -> Path:
    """Install KOFAM with threshold injection.
    
    KOFAM requires special handling because:
    1. Thresholds are in a separate file (ko_list)
    2. We inject thresholds into individual HMM files for PyHMMER
    """
    name = "KOFAM"
    db_info = registry.get(name)
    
    config.ensure_dirs()
    target_dir = config.data_dir / name
    target_dir.mkdir(exist_ok=True)
    
    # Download profiles
    logger.info(f"Downloading KOFAM profiles")
    _download_and_extract(db_info.url, target_dir, show_progress)
    
    # Download ko_list (thresholds)
    ko_list_url = "https://www.genome.jp/ftp/db/kofam/ko_list.gz"
    ko_list_path = config.data_dir / "ko_list.gz"
    
    logger.info("Downloading KOFAM thresholds")
    if show_progress:
        with DownloadProgress(unit="B", unit_scale=True, desc="ko_list.gz") as pbar:
            urlretrieve(ko_list_url, ko_list_path, reporthook=pbar.update_to)
    else:
        urlretrieve(ko_list_url, ko_list_path)
    
    # Decompress
    with gzip.open(ko_list_path, "rb") as f_in:
        with open(ko_list_path.with_suffix(""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    ko_list_path.unlink()
    
    # Parse thresholds
    logger.info("Injecting thresholds into HMM files")
    ko_list = pd.read_csv(ko_list_path.with_suffix(""), sep="\t")
    
    # Inject thresholds
    for _, row in tqdm(ko_list.iterrows(), total=len(ko_list), desc="Injecting thresholds"):
        if row["threshold"] == "-":
            continue
        
        try:
            threshold = float(row["threshold"])
        except (ValueError, TypeError):
            continue
        
        hmm_path = target_dir / f"{row['knum']}.hmm"
        if hmm_path.exists():
            _inject_threshold(hmm_path, threshold)
    
    registry.mark_installed(name, name)
    logger.info(f"Installed KOFAM to {target_dir}")
    
    return target_dir


def _inject_threshold(hmm_path: Path, threshold: float) -> None:
    """Inject threshold into HMM file."""
    try:
        with pyhmmer.plan7.HMMFile(hmm_path) as f:
            hmm = f.read()
        
        hmm.cutoffs.gathering = (threshold, threshold)
        hmm.cutoffs.trusted = (threshold, threshold)
        hmm.cutoffs.noise = (threshold, threshold)
        
        with open(hmm_path, "wb") as f:
            hmm.write(f)
    except Exception as e:
        logger.warning(f"Failed to inject threshold into {hmm_path}: {e}")


def uninstall_database(name: str) -> None:
    """Uninstall a database.
    
    Args:
        name: Database name
    """
    config = get_config()
    registry = get_registry()
    
    db_info = registry.get(name)
    if db_info is None or not db_info.installed:
        logger.warning(f"Database {name} not installed")
        return
    
    # Remove files
    db_path = db_info.resolve_path(config.data_dir)
    if db_path and db_path.exists():
        shutil.rmtree(db_path)
        logger.info(f"Removed {db_path}")
    
    registry.mark_uninstalled(name)


def register_custom_database(
    name: str,
    path: str,
    molecule_type: MoleculeType,
    citation: str = "",
) -> None:
    """Register a custom database.
    
    Args:
        name: Database name
        path: Path to HMM file or directory
        molecule_type: PROTEIN or NUCLEOTIDE
        citation: Optional citation
    """
    registry = get_registry()
    registry.register_custom(name, path, molecule_type, citation)
    logger.info(f"Registered custom database: {name}")
```

---

### `cli.py`

Command-line interface.

```python
"""Command-line interface for Aksha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from aksha.types import ThresholdOptions, MoleculeType


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        logging.error(str(e))
        if getattr(args, "verbose", False):
            raise
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="aksha",
        description="Aksha: Scalable HMM-based sequence search",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # search
    _add_search_parser(subparsers)
    
    # scan
    _add_scan_parser(subparsers)
    
    # phmmer
    _add_phmmer_parser(subparsers)
    
    # jackhmmer
    _add_jackhmmer_parser(subparsers)
    
    # nhmmer
    _add_nhmmer_parser(subparsers)
    
    # database management
    _add_database_parser(subparsers)
    
    return parser


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add threshold arguments to parser."""
    group = parser.add_argument_group("thresholds")
    group.add_argument("--cut_ga", action="store_true", help="Use gathering thresholds")
    group.add_argument("--cut_tc", action="store_true", help="Use trusted cutoffs")
    group.add_argument("--cut_nc", action="store_true", help="Use noise cutoffs")
    group.add_argument("--cascade", action="store_true", help="Try available cutoffs in order")
    group.add_argument("-E", "--evalue", type=float, help="E-value threshold")
    group.add_argument("-T", "--bitscore", type=float, help="Bitscore threshold")
    group.add_argument("--domE", type=float, help="Domain E-value threshold")
    group.add_argument("--domT", type=float, help="Domain bitscore threshold")
    group.add_argument("--incE", type=float, help="Inclusion E-value threshold")
    group.add_argument("--incT", type=float, help="Inclusion bitscore threshold")
    group.add_argument("--incdomE", type=float, help="Domain inclusion E-value threshold")
    group.add_argument("--incdomT", type=float, help="Domain inclusion bitscore threshold")


def _threshold_opts_from_args(args: argparse.Namespace) -> ThresholdOptions:
    """Build ThresholdOptions from parsed args."""
    return ThresholdOptions(
        cut_ga=getattr(args, "cut_ga", False),
        cut_tc=getattr(args, "cut_tc", False),
        cut_nc=getattr(args, "cut_nc", False),
        cascade=getattr(args, "cascade", False),
        evalue=getattr(args, "evalue", None),
        bitscore=getattr(args, "bitscore", None),
        dom_evalue=getattr(args, "domE", None),
        dom_bitscore=getattr(args, "domT", None),
        inc_evalue=getattr(args, "incE", None),
        inc_bitscore=getattr(args, "incT", None),
        inc_dom_evalue=getattr(args, "incdomE", None),
        inc_dom_bitscore=getattr(args, "incdomT", None),
    )


def _resolve_output(output: str) -> tuple[Path, Optional[Path]]:
    """Parse -o into (output_dir, final_csv_path or None)."""
    p = Path(output)
    if p.suffix:
        return p.parent, p
    return p, None


def _write_result(result, output_path: Path, csv_path: Optional[Path], default_name: str) -> None:
    """Write search result to CSV and print summary."""
    if csv_path:
        result.to_csv(csv_path)
    else:
        output_path.mkdir(exist_ok=True)
        result.to_csv(output_path / default_name)
    print(f"Found {len(result)} hits")


def _add_search_parser(subparsers) -> None:
    """Add search subcommand."""
    parser = subparsers.add_parser("search", help="Search sequences with HMM profiles")
    parser.add_argument("--sequences", "-s", required=True, help="Protein sequences (file or directory)")
    parser.add_argument("--hmms", "-H", required=True, help="HMM profiles (file, directory, or database name)")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads (0=auto)")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_search)


def _cmd_search(args: argparse.Namespace) -> int:
    """Execute search command."""
    from aksha.search import search

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = search(
        sequences=args.sequences,
        hmms=args.hmms,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "search_results.tsv")
    return 0


def _add_scan_parser(subparsers) -> None:
    """Add scan subcommand."""
    parser = subparsers.add_parser("scan", help="Scan sequences against HMM database")
    parser.add_argument("--sequences", "-s", required=True, help="Protein sequences")
    parser.add_argument("--hmms", "-H", required=True, help="HMM database")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_scan)


def _cmd_scan(args: argparse.Namespace) -> int:
    """Execute scan command."""
    from aksha.scan import scan

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = scan(
        sequences=args.sequences,
        hmms=args.hmms,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "scan_results.tsv")
    return 0


def _add_phmmer_parser(subparsers) -> None:
    """Add phmmer subcommand."""
    parser = subparsers.add_parser("phmmer", help="Protein vs protein search")
    parser.add_argument("--query", "-q", required=True, help="Query sequences")
    parser.add_argument("--target", "-d", required=True, help="Target database")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_phmmer)


def _cmd_phmmer(args: argparse.Namespace) -> int:
    """Execute phmmer command."""
    from aksha.phmmer import phmmer

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = phmmer(
        query=args.query,
        target=args.target,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "phmmer_results.tsv")
    return 0


def _add_jackhmmer_parser(subparsers) -> None:
    """Add jackhmmer subcommand."""
    parser = subparsers.add_parser("jackhmmer", help="Iterative protein search")
    parser.add_argument("--query", "-q", required=True, help="Query sequences")
    parser.add_argument("--target", "-d", required=True, help="Target database")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    parser.add_argument("--iterations", "-N", type=int, default=5, help="Max iterations")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_jackhmmer)


def _cmd_jackhmmer(args: argparse.Namespace) -> int:
    """Execute jackhmmer command."""
    from aksha.jackhmmer import jackhmmer

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = jackhmmer(
        query=args.query,
        target=args.target,
        thresholds=thresholds,
        threads=args.threads,
        max_iterations=args.iterations,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "jackhmmer_results.tsv")
    return 0


def _add_nhmmer_parser(subparsers) -> None:
    """Add nhmmer subcommand."""
    parser = subparsers.add_parser("nhmmer", help="Nucleotide HMM search")
    parser.add_argument("--sequences", "-s", required=True, help="Nucleotide sequences")
    parser.add_argument("--hmms", "-H", required=True, help="Nucleotide HMMs")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_nhmmer)


def _cmd_nhmmer(args: argparse.Namespace) -> int:
    """Execute nhmmer command."""
    from aksha.nhmmer import nhmmer

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = nhmmer(
        sequences=args.sequences,
        hmms=args.hmms,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "nhmmer_results.tsv")
    return 0


def _add_database_parser(subparsers) -> None:
    """Add database management subcommand."""
    parser = subparsers.add_parser("database", aliases=["db"], help="Database management")
    db_subparsers = parser.add_subparsers(dest="db_command")
    
    # list
    list_parser = db_subparsers.add_parser("list", help="List databases")
    list_parser.add_argument("--installed", action="store_true", help="Show only installed")
    list_parser.add_argument("--protein", action="store_true", help="Show only protein databases")
    list_parser.add_argument("--nucleotide", action="store_true", help="Show only nucleotide databases")
    list_parser.set_defaults(func=_cmd_db_list)
    
    # install
    install_parser = db_subparsers.add_parser("install", help="Install database")
    install_parser.add_argument("names", nargs="+", help="Database names to install")
    install_parser.add_argument("--force", action="store_true", help="Reinstall if exists")
    install_parser.set_defaults(func=_cmd_db_install)
    
    # uninstall
    uninstall_parser = db_subparsers.add_parser("uninstall", help="Uninstall database")
    uninstall_parser.add_argument("names", nargs="+", help="Database names to uninstall")
    uninstall_parser.set_defaults(func=_cmd_db_uninstall)
    
    # register
    register_parser = db_subparsers.add_parser("register", help="Register custom database")
    register_parser.add_argument("name", help="Database name")
    register_parser.add_argument("path", help="Path to HMM file or directory")
    register_parser.add_argument("--type", choices=["protein", "nucleotide"], default="protein")
    register_parser.add_argument("--citation", default="", help="Citation")
    register_parser.set_defaults(func=_cmd_db_register)
    
    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def _cmd_db_list(args: argparse.Namespace) -> int:
    """List databases."""
    from aksha.databases import list_databases
    
    mol_type = None
    if args.protein:
        mol_type = MoleculeType.PROTEIN
    elif args.nucleotide:
        mol_type = MoleculeType.NUCLEOTIDE
    
    databases = list_databases(molecule_type=mol_type, installed_only=args.installed)
    
    if not databases:
        print("No databases found")
        return 0
    
    print(f"{'Name':<20} {'Type':<12} {'Installed':<10}")
    print("-" * 45)
    for db in databases:
        installed = "Yes" if db.installed else "No"
        print(f"{db.name:<20} {db.molecule_type.name.lower():<12} {installed:<10}")
    
    return 0


def _cmd_db_install(args: argparse.Namespace) -> int:
    """Install databases."""
    from aksha.databases import install_database
    
    for name in args.names:
        print(f"Installing {name}...")
        install_database(name, force=args.force)
        print(f"Installed {name}")
    
    return 0


def _cmd_db_uninstall(args: argparse.Namespace) -> int:
    """Uninstall databases."""
    from aksha.databases import uninstall_database
    
    for name in args.names:
        uninstall_database(name)
        print(f"Uninstalled {name}")
    
    return 0


def _cmd_db_register(args: argparse.Namespace) -> int:
    """Register custom database."""
    from aksha.databases import register_custom_database
    
    mol_type = MoleculeType.PROTEIN if args.type == "protein" else MoleculeType.NUCLEOTIDE
    register_custom_database(args.name, args.path, mol_type, args.citation)
    print(f"Registered {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### `__init__.py`

Public API exports.

```python
"""Aksha: Scalable HMM-based sequence search and retrieval.

Example usage:
    from aksha import search, scan, install_database
    
    # Install a database
    install_database("PFAM")
    
    # Search sequences
    result = search("proteins.faa", "PFAM", thresholds=ThresholdOptions(cut_ga=True))
    
    # Get results as DataFrame
    df = result.to_dataframe()
"""

from aksha.types import (
    ThresholdOptions,
    SearchResult,
    SequenceHit,
    DomainHit,
    MoleculeType,
    DatabaseInfo,
)
from aksha.search import search
from aksha.scan import scan
from aksha.phmmer import phmmer
from aksha.jackhmmer import jackhmmer
from aksha.nhmmer import nhmmer
from aksha.databases import (
    install_database,
    uninstall_database,
    list_databases,
    register_custom_database,
)
from aksha.config import get_config, get_registry

__version__ = "0.1.0"
__all__ = [
    # Core functions
    "search",
    "scan",
    "phmmer",
    "jackhmmer",
    "nhmmer",
    # Database management
    "install_database",
    "uninstall_database",
    "list_databases",
    "register_custom_database",
    # Types
    "ThresholdOptions",
    "SearchResult",
    "SequenceHit",
    "DomainHit",
    "MoleculeType",
    "DatabaseInfo",
    # Config
    "get_config",
    "get_registry",
]
```

---

### `data/databases.json`

```json
{
  "databases": [
    {
      "name": "PFAM",
      "url": "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz",
      "molecule_type": "protein",
      "domain": true,
      "has_thresholds": true,
      "citation": "Mistry J et al. Pfam: The protein families database in 2021. Nucleic Acids Res. 2021;49(D1):D412-D419.",
      "notes": "Contains GA, TC, NC bitscore cutoffs"
    },
    {
      "name": "KOFAM",
      "url": "https://www.genome.jp/ftp/db/kofam/profiles.tar.gz",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": true,
      "citation": "Aramaki T et al. KofamKOALA: KEGG Ortholog assignment based on profile HMM. Bioinformatics. 2020;36(7):2251-2252.",
      "notes": "Thresholds injected during installation"
    },
    {
      "name": "DFAM",
      "url": "https://dfam.org/releases/Dfam_3.7/families/Dfam.hmm.gz",
      "molecule_type": "nucleotide",
      "domain": false,
      "has_thresholds": false,
      "citation": "Storer J et al. The Dfam community resource of transposable element families. Mobile DNA. 2021;12:2."
    },
    {
      "name": "NCBI",
      "url": "https://ftp.ncbi.nlm.nih.gov/hmm/current/hmm_PGAP.HMM.tgz",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Lu S et al. CDD/SPARCLE: the conserved domain database in 2020. Nucleic Acids Res. 2020;48(D1):D265-D268.",
      "notes": "Contains TIGRFAM database"
    },
    {
      "name": "dbCAN",
      "url": "https://bcb.unl.edu/dbCAN2/download/Databases/dbCAN-old@UGA/dbCAN-fam-HMMs.txt.v11",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Yin Y et al. dbCAN: a web resource for automated carbohydrate-active enzyme annotation. Nucleic Acids Res. 2012;40(W1):W445-W451.",
      "notes": "Default evalue 1e-3"
    },
    {
      "name": "VOGdb",
      "url": "https://fileshare.csb.univie.ac.at/vog/latest/vog.hmm.tar.gz",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "http://vogdb.org",
      "notes": "Viral protein orthologs"
    },
    {
      "name": "Resfam",
      "url": "http://dantaslab.wustl.edu/resfams/Resfams-full.hmm.gz",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Gibson MK et al. Improved annotation of antibiotic resistance functions. ISME J. 2015;9(1):207-216.",
      "notes": "Antibiotic resistance gene families"
    },
    {
      "name": "AntiFam",
      "url": "https://ftp.ebi.ac.uk/pub/databases/Pfam/AntiFam/current/Antifam.tar.gz",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": true,
      "citation": "Eberhardt RY et al. AntiFam: a tool to help identify spurious ORFs. Database. 2012;2012:bas003.",
      "notes": "Identifies spurious ORF predictions"
    },
    {
      "name": "FOAM",
      "url": "http://files.cqls.oregonstate.edu/David_Lab/FOAM_version1/FOAM-hmm_rel1.hmm",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Prestat E et al. FOAM: Functional Ontology Assignments for Metagenomes. Nucleic Acids Res. 2014;42(19):e145."
    },
    {
      "name": "DefenseFinder",
      "url": "https://github.com/mdmparis/defense-finder-models/tree/master/profiles/",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Tesson F et al. Systematic and quantitative view of the antiviral arsenal of prokaryotes. Nat Commun. 2022."
    },
    {
      "name": "PADLOC",
      "url": "https://github.com/padlocbio/padloc-db/tree/master/hmm/",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Payne LJ et al. Identification and classification of antiviral defence systems in bacteria and archaea with PADLOC. Nucleic Acids Res. 2021;49(19):10868-10878."
    },
    {
      "name": "CANT-HYD",
      "url": "https://github.com/dgittins/CANT-HYD-HydrocarbonBiodegradation/blob/main/HMMs/concatenated%20HMMs/CANT-HYD.hmm",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": true,
      "citation": "Khot V et al. CANT-HYD: A Curated Database for Annotation of Marker Genes Involved in Hydrocarbon Degradation. Front Microbiol. 2022;12:764058.",
      "notes": "Contains TC and NC cutoffs"
    },
    {
      "name": "NMPFamsDB",
      "url": "https://bib.fleming.gr/NMPFamsDB/download_files/pHMMs_all.tgz",
      "molecule_type": "protein",
      "domain": false,
      "has_thresholds": false,
      "citation": "Baltoumas FA et al. NMPFamsDB: a database of novel protein families from microbial metagenomes. Nucleic Acids Res. 2024;52(D1):D592-D599."
    },
    {
      "name": "barrnap",
      "url": "https://github.com/tseemann/barrnap/tree/master/db",
      "molecule_type": "nucleotide",
      "domain": false,
      "has_thresholds": false,
      "citation": "https://github.com/tseemann/barrnap",
      "notes": "16S/23S rRNA HMMs from Rfam seed alignments"
    }
  ]
}
```

---

## Testing

### `tests/conftest.py`

```python
"""Pytest fixtures for Aksha tests."""

import pytest
from pathlib import Path
import tempfile
import shutil

from aksha.config import AkshaConfig, DatabaseRegistry, reset_globals


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def test_config(temp_dir, monkeypatch):
    """Create test configuration with isolated paths."""
    config_dir = temp_dir / "config"
    data_dir = temp_dir / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    
    monkeypatch.setenv("AKSHA_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AKSHA_DATA_DIR", str(data_dir))
    
    reset_globals()
    
    yield AkshaConfig.load()
    
    reset_globals()


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def small_fasta(fixtures_dir):
    """Path to small test FASTA."""
    return fixtures_dir / "small.faa"


@pytest.fixture
def small_hmm(fixtures_dir):
    """Path to small test HMM."""
    return fixtures_dir / "small.hmm"
```

### `tests/test_search.py`

```python
"""Tests for search functionality."""

import pytest
from aksha import search, ThresholdOptions


def test_search_basic(small_fasta, small_hmm, temp_dir):
    """Test basic search functionality."""
    result = search(
        sequences=small_fasta,
        hmms=small_hmm,
        output_dir=temp_dir,
    )
    
    assert result is not None
    df = result.to_dataframe()
    assert "sequence_id" in df.columns
    assert "hmm_name" in df.columns


def test_search_with_thresholds(small_fasta, small_hmm, temp_dir):
    """Test search with threshold options."""
    result = search(
        sequences=small_fasta,
        hmms=small_hmm,
        thresholds=ThresholdOptions(evalue=0.001),
        output_dir=temp_dir,
    )
    
    df = result.to_dataframe()
    assert all(df["evalue"] <= 0.001)
```

---

## Implementation Notes for Codex

### Priority Order

1. **types.py** - All other modules depend on this
2. **config.py** - Configuration needed everywhere
3. **parsers.py** - Parsing needed for all search functions
4. **thresholds.py** - Small but important
5. **results.py** - Result handling
6. **search.py** - Core search, template for others
7. **scan.py, phmmer.py, jackhmmer.py, nhmmer.py** - Follow search.py pattern
8. **databases.py** - Can be done independently
9. **cli.py** - Thin wrapper, do last
10. **__init__.py** - Just exports

### Code Style

- Use `from __future__ import annotations` in all files
- Type hints on all public functions
- Docstrings in Google style
- f-strings for formatting
- Path objects over strings internally
- Logging over print statements
- Context managers for file handling

### Testing Strategy

- Unit tests for parsers, config, threshold logic
- Integration tests for full search pipeline
- Use small fixtures (10 sequences, 2-3 HMMs)
- Mock network calls in database tests

### Migration from Astra

- The old code is reference only
- Do not copy-paste — rewrite cleanly
- The JSON database format changes slightly (see databases.json spec)
- Config paths use `platformdirs` (not hand-rolled XDG)

---

## Checklist

- [x] Set up pyproject.toml
- [x] Create package structure
- [x] Implement types.py
- [x] Implement config.py (platformdirs)
- [x] Implement parsers.py (incl. iter_sequences)
- [x] Implement thresholds.py
- [x] Implement results.py (incl. consolidated hits_from_pyhmmer)
- [x] Implement search.py
- [x] Implement scan.py
- [x] Implement phmmer.py
- [x] Implement jackhmmer.py
- [x] Implement nhmmer.py
- [x] Implement databases.py (incl. tarfile security)
- [x] Implement cli.py (incl. _resolve_output, _write_result, full threshold flags)
- [x] Create __init__.py
- [x] Create data/databases.json
- [x] Create test fixtures
- [x] Write tests
- [ ] Test CLI end-to-end
- [x] Update README
- [ ] Build and test install
- [ ] Upload to PyPI
