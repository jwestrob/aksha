"""Parsing utilities for HMM and sequence files.

Supports multiple input types for flexibility:
- File paths (str or Path)
- Directories of files
- Pre-loaded PyHMMER objects
- Lists of sequences
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Union, Sequence, Optional

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
    PathLike,  # Single file or directory
    HMM,  # Single HMM object
    Sequence[HMM],  # List of HMM objects
    pyhmmer.plan7.HMMFile,  # Open HMM file
]

# Type for flexible sequence input
SequenceInput = Union[
    PathLike,  # Single file or directory
    SequenceBlock,  # Pre-loaded sequences
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
    if isinstance(source, pyhmmer.plan7.HMM):
        return [source]

    if isinstance(source, (list, tuple)):
        if all(isinstance(h, pyhmmer.plan7.HMM) for h in source):
            return list(source)

    if isinstance(source, pyhmmer.plan7.HMMFile):
        return list(source)

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

    logger.info("Parsed %d HMMs from %s", len(hmms), path)
    return hmms


def _parse_hmm_directory(path: Path, show_progress: bool = True) -> list[HMM]:
    """Parse directory of HMM files."""
    hmm_files = [f for f in path.iterdir() if f.suffix.lower() in (".hmm",)]

    if not hmm_files:
        raise ValueError(f"No HMM files found in directory: {path}")

    hmms = []
    iterator = tqdm(hmm_files, desc="Parsing HMMs") if show_progress else hmm_files

    for hmm_path in iterator:
        try:
            with pyhmmer.plan7.HMMFile(hmm_path) as f:
                # Consume all models in the file (many databases bundle thousands per file)
                hmms.extend(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", hmm_path, exc)

    logger.info("Parsed %d HMMs from %s", len(hmms), path)
    return hmms


def parse_sequences(
    source: SequenceInput,
    *,
    molecule_type: MoleculeType = MoleculeType.PROTEIN,
    alphabet: Alphabet | None = None,
    show_progress: bool = True,
) -> dict[Path, SequenceBlock]:
    """Parse sequences from various input types.

    Args:
        source: Sequence file path, directory, or pre-loaded sequences
        molecule_type: PROTEIN or NUCLEOTIDE
        alphabet: Optional explicit alphabet override (e.g., RNA HMMs)
        show_progress: Show progress bar for directory parsing

    Returns:
        Dict mapping source path to sequence block.
        For pre-loaded sequences, uses a synthetic path.

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If input is empty or invalid
    """
    alphabet = alphabet or _get_alphabet(molecule_type)

    if isinstance(source, pyhmmer.easel.DigitalSequenceBlock):
        return {Path("<memory>"): source}

    if isinstance(source, (list, tuple)):
        if all(isinstance(s, pyhmmer.easel.DigitalSequence) for s in source):
            block = pyhmmer.easel.DigitalSequenceBlock(alphabet, source)
            return {Path("<memory>"): block}

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Sequence path not found: {path}")

    if path.is_file():
        return {path: _parse_sequence_file(path, alphabet)}

    if path.is_dir():
        return _parse_sequence_directory(path, alphabet, show_progress=show_progress)

    raise ValueError(f"Invalid sequence source: {source}")


def iter_sequences(
    source: SequenceInput,
    *,
    molecule_type: MoleculeType = MoleculeType.PROTEIN,
    alphabet: Alphabet | None = None,
    show_progress: bool = True,
) -> Iterator[tuple[Path, SequenceBlock]]:
    """Lazily yield (path, sequence_block) pairs.

    For directory input, only one file's sequences are in memory at a time.
    For single-file or pre-loaded input, yields a single entry.
    """
    alphabet = alphabet or _get_alphabet(molecule_type)

    if isinstance(source, pyhmmer.easel.DigitalSequenceBlock):
        yield (Path("<memory>"), source)
        return

    if isinstance(source, (list, tuple)):
        if all(isinstance(s, pyhmmer.easel.DigitalSequence) for s in source):
            block = pyhmmer.easel.DigitalSequenceBlock(alphabet, source)
            yield (Path("<memory>"), block)
            return

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Sequence path not found: {path}")

    if path.is_file():
        yield (path, _parse_sequence_file(path, alphabet))
        return

    if path.is_dir():
        seq_files = [
            f
            for f in path.iterdir()
            if f.suffix.lower() in (".faa", ".fna", ".fa", ".fasta", ".fas")
        ]

        if not seq_files:
            raise ValueError(f"No sequence files found in directory: {path}")

        iterator = (
            tqdm(seq_files, desc="Parsing sequences") if show_progress else seq_files
        )

        for seq_path in iterator:
            try:
                yield (seq_path, _parse_sequence_file(seq_path, alphabet))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse %s: %s", seq_path, exc)
        return

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

    logger.info("Parsed %d sequences from %s", len(sequences), path)
    return sequences


def _parse_sequence_directory(
    path: Path,
    alphabet: Alphabet,
    show_progress: bool = True,
) -> dict[Path, SequenceBlock]:
    """Parse directory of sequence files."""
    seq_files = [
        f
        for f in path.iterdir()
        if f.suffix.lower() in (".faa", ".fna", ".fa", ".fasta", ".fas")
    ]

    if not seq_files:
        raise ValueError(f"No sequence files found in directory: {path}")

    result = {}
    iterator = tqdm(seq_files, desc="Parsing sequences") if show_progress else seq_files

    for seq_path in iterator:
        try:
            result[seq_path] = _parse_sequence_file(seq_path, alphabet)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", seq_path, exc)

    logger.info("Parsed sequences from %d files in %s", len(result), path)
    return result


def _check_hmm_thresholds(hmm: HMM) -> dict[str, bool]:
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
    available = _check_hmm_thresholds(hmm)

    for cutoff in preferred_order:
        if available.get(cutoff):
            return cutoff

    return None
