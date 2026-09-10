from __future__ import annotations

import hashlib
import io
from typing import Any


def optimized_profile_fingerprint(profile: Any) -> bytes:
    """Hash one OM after removing only its mutable target length.

    The local/glocal and uni/multihit configuration remains in the pressed
    bytes.  Those choices affect Forward and the downstream continuation and
    therefore must never be normalized away.
    """
    profile = profile.copy()
    profile.L = 400
    filter_output = io.BytesIO()
    profile_output = io.BytesIO()
    profile.write(filter_output, profile_output)
    filter_bytes = filter_output.getvalue()
    profile_bytes = profile_output.getvalue()
    digest = hashlib.sha256()
    digest.update(len(filter_bytes).to_bytes(8, "little"))
    digest.update(filter_bytes)
    digest.update(len(profile_bytes).to_bytes(8, "little"))
    digest.update(profile_bytes)
    return digest.digest()


def sequence_content_fingerprint(
    alphabet_size: int,
    residues: Any,
    offsets: Any,
) -> bytes:
    """Hash the exact ordered digital residues used to create a native batch."""
    residue_view = memoryview(residues).cast("B")
    offset_values = tuple(int(value) for value in offsets)
    if not offset_values or offset_values[0] != 0:
        raise ValueError("sequence offsets must start at zero")
    if offset_values[-1] != len(residue_view):
        raise ValueError("sequence offsets do not span residues")

    digest = hashlib.sha256()
    digest.update(b"plan7-gpu-sequence-content-v1\0")
    digest.update(int(alphabet_size).to_bytes(8, "little", signed=False))
    digest.update((len(offset_values) - 1).to_bytes(8, "little"))
    previous = 0
    for stop in offset_values[1:]:
        if stop < previous or stop > len(residue_view):
            raise ValueError("sequence offsets are not monotone")
        digest.update((stop - previous).to_bytes(8, "little"))
        digest.update(residue_view[previous:stop])
        previous = stop
    return digest.digest()


def sequence_block_content_fingerprint(sequences: Any) -> bytes:
    """Hash a copied ``DigitalSequenceBlock`` in native batch order."""
    residues = bytearray()
    offsets = [0]
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        offsets.append(len(residues))
    return sequence_content_fingerprint(sequences.alphabet.Kp, residues, offsets)
