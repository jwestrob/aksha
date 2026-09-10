from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import io
from itertools import zip_longest
import math
import operator
from pathlib import Path
import struct
from threading import Lock
from typing import Any, NamedTuple, cast
from weakref import WeakKeyDictionary

import astra_pyhmmer

from . import _native  # type: ignore[attr-defined]
from ._fingerprint import (
    optimized_profile_fingerprint,
    sequence_block_content_fingerprint,
    sequence_content_fingerprint,
)

_MISSING = object()
_PIPELINE_LEASES_LOCK = Lock()
_FORWARD_SPECIAL_BYTE_BUDGET = 384 << 20
_EXECUTION_POLICY_CODES = {
    "auto": _native.EXECUTION_POLICY_AUTO,
    "simple": _native.EXECUTION_POLICY_SIMPLE,
    "throughput": _native.EXECUTION_POLICY_THROUGHPUT,
}


def _normalize_execution_policy(value: Any) -> tuple[str, int]:
    if type(value) is not str or value not in _EXECUTION_POLICY_CODES:
        raise ValueError(
            "execution_policy must be 'auto', 'simple', or 'throughput'"
        )
    return value, _EXECUTION_POLICY_CODES[value]


class _PipelineLease:
    __slots__ = ("pipeline", "lock", "refcount")

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.lock = Lock()
        self.refcount = 0


_PIPELINE_LEASES: dict[int, _PipelineLease] = {}


@contextmanager
def _lease_pipeline(pipeline: Any):
    identity = id(pipeline)
    with _PIPELINE_LEASES_LOCK:
        lease = _PIPELINE_LEASES.get(identity)
        if lease is None:
            lease = _PipelineLease(pipeline)
            _PIPELINE_LEASES[identity] = lease
        elif lease.pipeline is not pipeline:
            raise RuntimeError("pipeline identity collision")
        lease.refcount += 1

    try:
        with lease.lock:
            yield
    finally:
        with _PIPELINE_LEASES_LOCK:
            lease.refcount -= 1
            if lease.refcount == 0:
                if _PIPELINE_LEASES.get(identity) is not lease:
                    raise RuntimeError("pipeline lease registry corruption")
                del _PIPELINE_LEASES[identity]


class _FileStat(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class _PressedStatToken(NamedTuple):
    h3m: _FileStat
    h3i: _FileStat
    h3f: _FileStat
    h3p: _FileStat


class _CutoffSnapshot(NamedTuple):
    gathering: tuple[float, float] | None
    noise: tuple[float, float] | None
    trusted: tuple[float, float] | None


class _PairState:
    __slots__ = (
        "hmm",
        "optimized_profile",
        "profile_fingerprint",
        "background_fingerprint",
        "lock",
    )

    def __init__(
        self,
        hmm: Any,
        optimized_profile: Any,
        background_fingerprint: bytes,
    ) -> None:
        self.hmm = hmm
        self.optimized_profile = optimized_profile
        self.profile_fingerprint = optimized_profile_fingerprint(
            optimized_profile
        )
        self.background_fingerprint = background_fingerprint
        self.lock = Lock()


def _background_fingerprint(background: Any) -> bytes:
    frequencies = memoryview(background.residue_frequencies)
    if frequencies.format != "f" or frequencies.ndim != 1:
        raise TypeError("background residue frequencies are not binary32")
    return frequencies.cast("B").tobytes() + struct.pack("=f", background.omega)


def _optimized_profile_bytes(profile: Any) -> tuple[bytes, bytes]:
    filter_output = io.BytesIO()
    profile_output = io.BytesIO()
    profile.write(filter_output, profile_output)
    return filter_output.getvalue(), profile_output.getvalue()


def _verify_pressed_profile(
    hmm: Any,
    optimized_profile: Any,
    canonical_background: Any,
    pipeline_module: Any,
) -> bool:
    reference_profile = hmm.to_profile(
        canonical_background, L=optimized_profile.L
    ).to_optimized()
    for offset_name in ("model", "filter", "profile"):
        setattr(
            reference_profile.offsets,
            offset_name,
            getattr(optimized_profile.offsets, offset_name),
        )
    return pipeline_module._oprofiles_equal_hmmer(
        reference_profile, optimized_profile
    ) and _optimized_profile_bytes(reference_profile) == _optimized_profile_bytes(
        optimized_profile
    )


@dataclass(
    frozen=True,
    slots=True,
    weakref_slot=True,
    init=False,
    eq=False,
    repr=False,
)
class PressedProfilePair:
    """One immutable HMM/optimized-profile pair read in pressed-file lockstep."""

    canonical_base: Path
    ordinal: int
    stat_token: _PressedStatToken
    cutoffs: _CutoffSnapshot

    def __init__(self) -> None:
        raise TypeError("PressedProfilePair objects come from load_pressed_profiles")

    @property
    def hmm(self) -> Any:
        return _pair_state(self).hmm.copy()

    def __repr__(self) -> str:
        return (
            f"PressedProfilePair(canonical_base={self.canonical_base!r}, "
            f"ordinal={self.ordinal})"
        )


_PAIR_STATES: WeakKeyDictionary[PressedProfilePair, _PairState] = WeakKeyDictionary()


def _pair_state(pair: PressedProfilePair) -> _PairState:
    try:
        return _PAIR_STATES[pair]
    except KeyError as error:
        raise TypeError(
            "candidate batches require pairs from load_pressed_profiles"
        ) from error


def _new_pressed_profile_pair(
    hmm: Any,
    optimized_profile: Any,
    canonical_base: Path,
    ordinal: int,
    stat_token: _PressedStatToken,
    cutoffs: _CutoffSnapshot,
    background_fingerprint: bytes,
) -> PressedProfilePair:
    pair = object.__new__(PressedProfilePair)
    object.__setattr__(pair, "canonical_base", canonical_base)
    object.__setattr__(pair, "ordinal", ordinal)
    object.__setattr__(pair, "stat_token", stat_token)
    object.__setattr__(pair, "cutoffs", cutoffs)
    _PAIR_STATES[pair] = _PairState(hmm, optimized_profile, background_fingerprint)
    return pair


class _ProfileSessionState:
    __slots__ = (
        "pairs",
        "alphabet",
        "native",
        "lock",
        "queries",
        "profiles",
        "profile_fingerprints",
        "background_fingerprint",
        "chunk_local_pack",
    )

    def __init__(
        self,
        pairs: tuple[PressedProfilePair, ...],
        alphabet: Any,
        native: Any,
        queries: tuple[Any, ...],
        profiles: tuple[Any, ...],
        profile_fingerprints: tuple[bytes, ...],
        background_fingerprint: bytes,
        chunk_local_pack: bool,
    ) -> None:
        self.pairs = pairs
        self.alphabet = alphabet
        self.native = native
        self.lock = Lock()
        self.queries = queries
        self.profiles = profiles
        self.profile_fingerprints = profile_fingerprints
        self.background_fingerprint = background_fingerprint
        self.chunk_local_pack = chunk_local_pack


class _ProfileSelectionState:
    __slots__ = (
        "owner",
        "pairs",
        "indices",
        "alphabet",
        "native",
        "lock",
        "queries",
        "profiles",
        "profile_fingerprints",
        "background_fingerprint",
    )

    def __init__(
        self,
        owner: ProfileSession,
        pairs: tuple[PressedProfilePair, ...],
        indices: tuple[int, ...],
        alphabet: Any,
        native: Any,
        queries: tuple[Any, ...],
        profiles: tuple[Any, ...],
        profile_fingerprints: tuple[bytes, ...],
        background_fingerprint: bytes,
    ) -> None:
        self.owner = owner
        self.pairs = pairs
        self.indices = indices
        self.alphabet = alphabet
        self.native = native
        self.lock = Lock()
        self.queries = queries
        self.profiles = profiles
        self.profile_fingerprints = profile_fingerprints
        self.background_fingerprint = background_fingerprint


_PROFILE_SESSION_STATES: WeakKeyDictionary[Any, _ProfileSessionState] = (
    WeakKeyDictionary()
)
_PROFILE_SELECTION_STATES: WeakKeyDictionary[Any, _ProfileSelectionState] = (
    WeakKeyDictionary()
)


def _profile_session_state(session: Any) -> _ProfileSessionState:
    try:
        return _PROFILE_SESSION_STATES[session]
    except KeyError as error:
        raise TypeError("invalid ProfileSession object") from error


def _profile_selection_state(selection: Any) -> _ProfileSelectionState:
    try:
        return _PROFILE_SELECTION_STATES[selection]
    except KeyError as error:
        raise TypeError("invalid ProfileSelection object") from error


def _profile_worker_budget(
    value: Any,
    default: int,
    profile_count: int,
    name: str,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        requested = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a nonnegative integer") from error
    if requested < 0:
        raise ValueError(f"{name} must be nonnegative")
    return min(requested, profile_count)


class ProfileSession:
    """An immutable host snapshot of one pressed profile database.

    ``pack_workers`` remains the shared default for construction and selection.
    Either phase may be overridden independently; construction workers are
    retired before this constructor returns.

    ``_chunk_local_pack`` is a private large-database experiment. It keeps the
    already-owned optimized profiles alive and creates only immutable selected
    packs, leaving normal and small-workload behavior unchanged.
    """

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        profile_pairs: Iterable[PressedProfilePair],
        *,
        pack_workers: int | None = None,
        build_workers: int | None = None,
        selection_workers: int | None = None,
        _chunk_local_pack: bool = False,
    ):
        pairs = tuple(profile_pairs)
        if not pairs:
            raise ValueError("a profile session requires at least one profile")
        if type(_chunk_local_pack) is not bool:
            raise TypeError("_chunk_local_pack must be bool")
        default_workers = min(16, len(pairs))
        shared_workers = _profile_worker_budget(
            pack_workers, default_workers, len(pairs), "pack_workers"
        )
        build_worker_count = _profile_worker_budget(
            build_workers, shared_workers, len(pairs), "build_workers"
        )
        selection_worker_count = _profile_worker_budget(
            selection_workers,
            shared_workers,
            len(pairs),
            "selection_workers",
        )
        if not _native.bias_host_environment_attested():
            raise RuntimeError(
                "profile sessions require the attested host floating-point environment"
            )
        if type(pairs[0]) is not PressedProfilePair:
            raise TypeError(
                "profile sessions require pairs from load_pressed_profiles"
            )

        states = []
        seen: set[int] = set()
        canonical_base = pairs[0].canonical_base
        stat_token = pairs[0].stat_token
        for pair in pairs:
            if type(pair) is not PressedProfilePair:
                raise TypeError(
                    "profile sessions require pairs from load_pressed_profiles"
                )
            if pair.ordinal in seen:
                raise ValueError("profile session pairs must be unique")
            seen.add(pair.ordinal)
            if pair.canonical_base != canonical_base or pair.stat_token != stat_token:
                raise ValueError("profile session pairs come from different databases")
            states.append(_pair_state(pair))

        unique_locks = {id(state.lock): state.lock for state in states}
        with ExitStack() as locks:
            for lock_id in sorted(unique_locks):
                locks.enter_context(unique_locks[lock_id])
            alphabet = states[0].hmm.alphabet
            background_fingerprint = states[0].background_fingerprint
            for state in states:
                if state.hmm.alphabet != alphabet:
                    raise ValueError("profile session alphabets differ")
                if state.background_fingerprint != background_fingerprint:
                    raise ValueError("profile session backgrounds differ")
            background = astra_pyhmmer.plan7.Background(alphabet)
            if _background_fingerprint(background) != background_fingerprint:
                raise ValueError(
                    "profile session background is not the canonical hmmpress background"
                )
            if any(
                not state.optimized_profile.local
                or not state.optimized_profile.multihit
                for state in states
            ):
                raise ValueError(
                    "profile sessions require local multihit optimized profiles"
                )
            queries = tuple(state.hmm for state in states)
            profiles = tuple(state.optimized_profile for state in states)
            profile_fingerprints = tuple(
                state.profile_fingerprint for state in states
            )
            native = _native.ProfileSession(
                profiles,
                memoryview(background.residue_frequencies),
                build_worker_count,
                selection_worker_count,
                profile_fingerprints,
                _chunk_local_pack,
            )
            if (
                native._fingerprints_for_seal()
                != b"".join(profile_fingerprints)
            ):
                native.close()
                raise RuntimeError(
                    "native optimized-profile fingerprints changed"
                )
        _PROFILE_SESSION_STATES[self] = _ProfileSessionState(
            pairs,
            alphabet,
            native,
            queries,
            profiles,
            profile_fingerprints,
            background_fingerprint,
            _chunk_local_pack,
        )

    def __len__(self) -> int:
        return len(_profile_session_state(self).pairs)

    @property
    def closed(self) -> bool:
        state = _profile_session_state(self)
        with state.lock:
            return bool(state.native.closed)

    @property
    def statistics(self) -> dict[str, int]:
        state = _profile_session_state(self)
        with state.lock:
            return cast(dict[str, int], state.native.statistics)

    def select(self, indices: Iterable[int]) -> ProfileSelection:
        state = _profile_session_state(self)
        requested = tuple(indices)
        normalized = []
        seen: set[int] = set()
        for value in requested:
            if isinstance(value, bool):
                raise TypeError("profile selection index must not be bool")
            try:
                index = operator.index(value)
            except TypeError as error:
                raise TypeError(
                    "profile selection index must be an integer"
                ) from error
            if not 0 <= index < len(state.pairs):
                raise IndexError("profile selection index is out of range")
            if index in seen:
                raise ValueError("profile selection indexes must be unique")
            seen.add(index)
            normalized.append(index)
        normalized_indices = tuple(normalized)
        with ExitStack() as locks:
            if state.chunk_local_pack:
                pair_states = tuple(
                    _pair_state(state.pairs[index])
                    for index in normalized_indices
                )
                unique_locks = {
                    id(pair_state.lock): pair_state.lock
                    for pair_state in pair_states
                }
                for lock_id in sorted(unique_locks):
                    locks.enter_context(unique_locks[lock_id])
            locks.enter_context(state.lock)
            if state.native.closed:
                raise RuntimeError("profile session is closed")
            native = state.native.select(normalized_indices)
        selection = object.__new__(ProfileSelection)
        selected_pairs = tuple(state.pairs[index] for index in normalized_indices)
        selected_queries = tuple(
            state.queries[index] for index in normalized_indices
        )
        selected_profiles = tuple(
            state.profiles[index] for index in normalized_indices
        )
        selected_fingerprints = tuple(
            state.profile_fingerprints[index] for index in normalized_indices
        )
        _PROFILE_SELECTION_STATES[selection] = _ProfileSelectionState(
            self,
            selected_pairs,
            normalized_indices,
            state.alphabet,
            native,
            selected_queries,
            selected_profiles,
            selected_fingerprints,
            state.background_fingerprint,
        )
        return selection

    def close(self) -> None:
        state = _profile_session_state(self)
        with state.lock:
            state.native.close()

    def __enter__(self) -> ProfileSession:
        if self.closed:
            raise RuntimeError("profile session is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(
    frozen=True,
    slots=True,
    weakref_slot=True,
    init=False,
    eq=False,
    repr=False,
)
class ProfileSelection:
    """One immutable ordered selection from a profile session."""

    def __init__(self) -> None:
        raise TypeError("ProfileSelection objects come from ProfileSession.select")

    def __len__(self) -> int:
        return len(_profile_selection_state(self).pairs)

    @property
    def indices(self) -> tuple[int, ...]:
        return _profile_selection_state(self).indices

    @property
    def closed(self) -> bool:
        state = _profile_selection_state(self)
        with state.lock:
            return bool(state.native.closed)

    @property
    def identity(self) -> tuple[int, int]:
        state = _profile_selection_state(self)
        with state.lock:
            return cast(tuple[int, int], state.native.identity)

    @property
    def host_bytes(self) -> int:
        state = _profile_selection_state(self)
        with state.lock:
            return int(state.native.host_bytes)

    def close(self) -> None:
        state = _profile_selection_state(self)
        with state.lock:
            state.native.close()

    def __enter__(self) -> ProfileSelection:
        if self.closed:
            raise RuntimeError("profile selection is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _iter_pressed_profile_pairs(
    path: str | Path,
    *,
    manifest: str | Path | None = None,
) -> Iterator[PressedProfilePair]:
    """Yield a pressed HMM database as provenance-bound lockstep pairs.

    ``manifest`` is an unsigned trust anchor and must come from a trusted
    source, such as this repository's reviewed ``results/pressed`` records.
    """
    from .pressed_manifest import (
        _pinned_pressed_database,
        validate_pressed_manifest,
    )

    base = Path(path).resolve(strict=False)
    amino_alphabet = astra_pyhmmer.easel.Alphabet.amino()
    manifest_validation = None
    if manifest is not None:
        manifest_validation = validate_pressed_manifest(base, manifest)
        if manifest_validation.canonical_base != base:
            raise ValueError("pressed manifest canonical database mismatch")
        pipeline_module = None
    else:
        from . import _pipeline  # type: ignore[attr-defined]

        pipeline_module = _pipeline
    model_count = 0
    with _pinned_pressed_database(base) as (pinned_base, pinned_token):
        before = _PressedStatToken(*(_FileStat(*member) for member in pinned_token))
        if (
            manifest_validation is not None
            and pinned_token != manifest_validation.stat_token
        ):
            raise RuntimeError(
                "pressed database changed after its manifest was validated"
            )
        with (
            astra_pyhmmer.plan7.HMMFile(pinned_base) as hmm_file,
            astra_pyhmmer.plan7.HMMPressedFile(pinned_base) as optimized_file,
        ):
            for ordinal, (hmm, optimized_profile) in enumerate(
                zip_longest(hmm_file, optimized_file, fillvalue=_MISSING)
            ):
                if hmm is _MISSING:
                    raise ValueError(
                        "pressed optimized-profile stream has more models than HMM stream"
                    )
                if optimized_profile is _MISSING:
                    raise ValueError(
                        "pressed HMM stream has more models than optimized-profile stream"
                    )
                hmm = cast(astra_pyhmmer.plan7.HMM, hmm)
                optimized_profile = cast(
                    astra_pyhmmer.plan7.OptimizedProfile, optimized_profile
                )
                if hmm.alphabet != amino_alphabet:
                    raise ValueError(
                        "bound candidate search only supports amino-acid profiles"
                    )
                if (
                    hmm.alphabet != optimized_profile.alphabet
                    or hmm.M != optimized_profile.M
                    or hmm.name != optimized_profile.name
                    or hmm.accession != optimized_profile.accession
                    or hmm.consensus != optimized_profile.consensus
                ):
                    raise ValueError(
                        f"pressed HMM/profile identity mismatch at ordinal {ordinal}"
                    )
                canonical_background = astra_pyhmmer.plan7.Background(hmm.alphabet)
                if manifest_validation is None and not _verify_pressed_profile(
                    hmm,
                    optimized_profile,
                    canonical_background,
                    pipeline_module,
                ):
                    raise ValueError(
                        f"pressed HMM/profile score mismatch at ordinal {ordinal}"
                    )
                model_count = ordinal + 1
                yield _new_pressed_profile_pair(
                    hmm,
                    optimized_profile,
                    base,
                    ordinal,
                    before,
                    _CutoffSnapshot(
                        hmm.cutoffs.gathering,
                        hmm.cutoffs.noise,
                        hmm.cutoffs.trusted,
                    ),
                    _background_fingerprint(canonical_background),
                )

    if (
        manifest_validation is not None
        and model_count != manifest_validation.model_count
    ):
        raise ValueError("pressed profile count does not match the validated manifest")


def _iter_pressed_profile_chunks(
    path: str | Path,
    chunk_size: int,
    *,
    manifest: str | Path | None = None,
) -> Iterator[tuple[PressedProfilePair, ...]]:
    """Yield bounded, canonical chunks from one pinned pressed database."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk: list[PressedProfilePair] = []
    pairs = _iter_pressed_profile_pairs(path, manifest=manifest)
    try:
        for pair in pairs:
            chunk.append(pair)
            if len(chunk) == chunk_size:
                yield tuple(chunk)
                chunk = []
        if chunk:
            yield tuple(chunk)
    finally:
        pairs.close()


def load_pressed_profiles(
    path: str | Path,
    *,
    manifest: str | Path | None = None,
) -> tuple[PressedProfilePair, ...]:
    """Load a pressed HMM database as provenance-bound lockstep pairs.

    ``manifest`` is an unsigned trust anchor and must come from a trusted
    source, such as this repository's reviewed ``results/pressed`` records.
    """
    return tuple(_iter_pressed_profile_pairs(path, manifest=manifest))


class _PackedProfiles(NamedTuple):
    scores: bytearray
    score_offsets: array[int]
    score_counts: array[int]
    score_strides: array[int]
    model_lengths: array[int]
    constants: bytearray
    scales: array[float]


def _pack_profiles(profiles: list[Any]) -> _PackedProfiles:
    striped_score_buffers = []
    striped_score_strides = array("i")
    score_offsets = array("Q")
    score_counts = array("Q")
    score_strides = array("i")
    model_lengths = array("i")
    constants = bytearray()
    scales = array("f")
    score_offset = 0

    for profile in profiles:
        alphabet_size = profile.alphabet.Kp
        score_count = profile.M * alphabet_size
        striped_score_buffers.append(memoryview(profile.sbv).cast("B"))
        striped_score_strides.append(profile.sbv.shape[1])
        score_offsets.append(score_offset)
        score_counts.append(score_count)
        score_strides.append(alphabet_size)
        model_lengths.append(profile.M)
        constants.extend((profile.tbm, profile.tec, profile.base_b, profile.bias_b))
        scales.append(profile.scale_b)
        score_offset += score_count

    scores = (
        _native.pack_striped_scores(
            striped_score_buffers,
            striped_score_strides,
            model_lengths,
            profiles[0].alphabet.Kp,
        )
        if profiles
        else bytearray()
    )

    return _PackedProfiles(
        scores,
        score_offsets,
        score_counts,
        score_strides,
        model_lengths,
        constants,
        scales,
    )


def _f1_parameters(profile: Any) -> tuple[float, float, float] | None:
    try:
        parameters = profile.evalue_parameters.as_vector()
        m_mu = float(parameters[0])
        m_lambda = float(parameters[1])
        scale = float(profile.scale_b)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(m_mu)
        or not math.isfinite(m_lambda)
        or not math.isfinite(scale)
        or m_mu == -99999.0
        or m_lambda == -99999.0
        or m_lambda <= 0.0
        or scale <= 0.0
    ):
        return None
    return m_mu, m_lambda, scale


def _pack_postfilter_inputs(
    background: Any,
    profiles: list[Any],
    threshold: float,
) -> tuple[bytearray, array[float], array[float]] | None:
    packed_bias = bytearray()
    m_mu = array("f")
    m_lambda = array("f")
    background_frequencies = memoryview(background.residue_frequencies)
    for profile in profiles:
        parameters = _f1_parameters(profile)
        if parameters is None:
            return None
        profile_mu, profile_lambda, scale = parameters
        cutoff_mode, cutoff = _native.f1_cutoff(profile_mu, profile_lambda, threshold)
        if cutoff_mode == _native.F1_CUTOFF_INVALID:
            return None
        packed_bias.extend(
            _native.pack_bias_profile_raw(
                background_frequencies,
                memoryview(profile.compositions),
                profile.M,
                scale,
                cutoff_mode,
                math.nan if cutoff is None else cutoff,
            )
        )
        m_mu.append(profile_mu)
        m_lambda.append(profile_lambda)
    return packed_bias, m_mu, m_lambda


def _format_results(raw_results: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    names = {
        _native.STATUS_OK: "eslOK",
        _native.STATUS_ERANGE: "eslERANGE",
        _native.STATUS_ENORESULT: "eslENORESULT",
        _native.STATUS_EMPTY: "empty",
    }
    actions = {
        _native.STATUS_OK: "threshold_score",
        _native.STATUS_ERANGE: "promote",
        _native.STATUS_ENORESULT: "cpu_msv",
        _native.STATUS_EMPTY: "skip_empty",
    }
    return [
        {
            "status": names[status],
            "action": actions[status],
            "xE_u8": xE,
            "tjb_u8": length_tjb,
            "numerator": numerator if status == _native.STATUS_OK else None,
            "score_bits": score_bits,
        }
        for status, xE, length_tjb, numerator, score_bits in raw_results
    ]


class _ForwardAugmentation:
    __slots__ = (
        "records",
        "row_offsets",
        "special_offsets",
        "specials",
        "f2_bits",
        "f3_bits",
        "bias_filter",
    )

    def __init__(
        self,
        records: bytes,
        row_offsets: Any,
        special_offsets: Any,
        specials: Any,
        f2_bits: int,
        f3_bits: int,
        bias_filter: bool,
    ) -> None:
        self.records = memoryview(records).cast("B").tobytes()
        self.row_offsets = memoryview(
            memoryview(row_offsets).cast("B").tobytes()
        ).cast("Q")
        self.special_offsets = memoryview(
            memoryview(special_offsets).cast("B").tobytes()
        ).cast("Q")
        self.specials = memoryview(
            memoryview(specials).cast("B").tobytes()
        ).cast("f")
        self.f2_bits = f2_bits
        self.f3_bits = f3_bits
        self.bias_filter = bias_filter


class _CandidateState:
    __slots__ = (
        "pairs",
        "targets",
        "residue_offsets",
        "indices",
        "offsets",
        "all_rows",
        "all_targets",
        "postfilter_records",
        "forward",
        "sealed_postfilter",
        "f1",
    )

    def __init__(
        self,
        pairs: tuple[PressedProfilePair, ...],
        targets: Any,
        residue_offsets: bytes,
        indices: bytes,
        offsets: bytes,
        all_rows: bytes,
        all_targets: bytes,
        postfilter_records: bytes | None,
        forward: _ForwardAugmentation | None,
        sealed_postfilter: Any | None,
        f1: float,
    ) -> None:
        self.pairs = pairs
        self.targets = targets
        self.residue_offsets = residue_offsets
        self.indices = indices
        self.offsets = offsets
        self.all_rows = all_rows
        self.all_targets = all_targets
        self.postfilter_records = postfilter_records
        self.forward = forward
        self.sealed_postfilter = sealed_postfilter
        self.f1 = f1


@dataclass(
    frozen=True,
    slots=True,
    weakref_slot=True,
    init=False,
    eq=False,
    repr=False,
)
class CandidateBatch:
    """Opaque candidate rows bound to their queries, profiles, and targets."""

    def __init__(self) -> None:
        raise TypeError(
            "CandidateBatch objects come from SequenceBatch.candidate_batch"
        )

    def __len__(self) -> int:
        return len(_candidate_state(self).pairs)

    @property
    def F1(self) -> float:
        return _candidate_state(self).f1

    def __repr__(self) -> str:
        return f"CandidateBatch(rows={len(self)}, F1={self.F1!r})"

    @property
    def resident_memory(self) -> dict[str, Any]:
        """Return exact incremental buffer payload bytes owned by this batch.

        This is retained backing-buffer payload accounting, not Python object
        overhead or process RSS. Each immutable backing allocation is charged
        once even when several sealed views address it.

        Shared targets, source residue storage, pressed profiles,
        selection/session storage, and persistent CUDA workspaces are excluded;
        defensive identity-buffer copies are charged. Native Forward,
        Backward/domain, and rescore outputs are downloaded and destroyed before
        construction, so a CandidateBatch owns no device allocation.
        """
        state = _candidate_state(self)
        state_buffers = {
            "indices_bytes": len(state.indices),
            "offsets_bytes": len(state.offsets),
            "all_rows_bytes": len(state.all_rows),
            "all_targets_bytes": len(state.all_targets),
            "postfilter_records_bytes": (
                len(state.postfilter_records)
                if state.sealed_postfilter is None
                and state.postfilter_records is not None
                else 0
            ),
            "forward_records_bytes": (
                len(state.forward.records) if state.forward is not None else 0
            ),
            "forward_offsets_bytes": (
                state.forward.row_offsets.nbytes
                if state.forward is not None
                else 0
            ),
            "forward_special_offsets_bytes": (
                state.forward.special_offsets.nbytes
                if state.forward is not None
                else 0
            ),
            "forward_specials_bytes": (
                state.forward.specials.nbytes
                if state.forward is not None
                else 0
            ),
        }
        state_owned_host_bytes = sum(state_buffers.values())
        sealed_memory = None
        sealed_owned_host_bytes = 0
        sealed_owned_device_bytes = 0
        if state.sealed_postfilter is not None:
            from . import _pipeline  # type: ignore[attr-defined]

            sealed_memory = _pipeline._sealed_resident_memory_bound(
                state.sealed_postfilter
            )
            if type(sealed_memory) is not dict:
                raise RuntimeError("sealed resident-memory record is not a dict")
            sealed_owned_host_bytes = sealed_memory.get("owned_host_bytes")
            sealed_owned_device_bytes = sealed_memory.get("owned_device_bytes")
            for name, value in (
                ("owned_host_bytes", sealed_owned_host_bytes),
                ("owned_device_bytes", sealed_owned_device_bytes),
            ):
                if type(value) is not int or value < 0:
                    raise RuntimeError(
                        f"sealed resident-memory {name} is not exact nonnegative int"
                    )
        owned_host_bytes = state_owned_host_bytes + sealed_owned_host_bytes
        owned_device_bytes = sealed_owned_device_bytes
        resident_bytes = owned_host_bytes + owned_device_bytes
        return {
            "schema_version": 1,
            "accounting_scope": "owned immutable backing-buffer payload",
            "resident_bytes": resident_bytes,
            "owned_host_bytes": owned_host_bytes,
            "owned_device_bytes": owned_device_bytes,
            "state_owned_host_bytes": state_owned_host_bytes,
            "state_buffers": state_buffers,
            "sealed": sealed_memory,
            "excluded_shared": (
                "pressed_profile_pairs",
                "query_profile_reference_tuples_and_wrappers",
                "targets",
                "residue_offsets",
                "profile_selection_session",
                "sequence_batch_device_storage",
                "generation_workspaces",
            ),
        }

    @property
    def resident_bytes(self) -> int:
        """Exact incremental host plus device buffer bytes owned by this batch."""
        value = self.resident_memory["resident_bytes"]
        if type(value) is not int or value < 0:
            raise RuntimeError("candidate resident bytes are not exact nonnegative int")
        return value

    @property
    def generation_statistics(self) -> dict[str, Any] | None:
        """Return defensive opt-in generation telemetry, if it was collected."""
        state = _candidate_state(self)
        if state.sealed_postfilter is None:
            return None
        from . import _pipeline  # type: ignore[attr-defined]

        value = _pipeline._sealed_generation_statistics_bound(
            state.sealed_postfilter
        )
        if value is not None and type(value) is not dict:
            raise RuntimeError("sealed generation statistics are not a dict")
        return value

    def evaluate_ga_pruning(
        self,
        row: int,
        pipeline: Any,
        *,
        include_indices: bool = False,
    ) -> dict[str, Any]:
        """Evaluate certified ``--cut_ga`` pruning without changing search.

        This Phase 9 diagnostic scans authenticated compact-domain results and
        applies conservative upper bounds to HMMER's final target and domain
        scores.  It never suppresses work or changes a continuation decision.
        """
        row_index = self._row_index(row)
        if type(include_indices) is not bool:
            raise TypeError("include_indices must be bool")
        if type(pipeline) is not astra_pyhmmer.plan7.Pipeline:
            raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
        if not _native.bias_host_environment_attested():
            raise RuntimeError(
                "GA pruning census requires the attested host floating-point environment"
            )
        candidate_state = _candidate_state(self)
        sealed_postfilter = candidate_state.sealed_postfilter
        if sealed_postfilter is None:
            raise ValueError("GA pruning census requires a sealed fused batch")
        pair = candidate_state.pairs[row_index]
        state = _pair_state(pair)
        from . import _pipeline  # type: ignore[attr-defined]

        with _lease_pipeline(pipeline):
            with state.lock:
                value = _pipeline._sealed_ga_cutoff_census_bound(
                    sealed_postfilter,
                    row_index,
                    pipeline,
                    include_indices=include_indices,
                )
        if type(value) is not dict:
            raise RuntimeError("GA pruning census did not return a dict")
        return cast(dict[str, Any], value)

    def _row_index(self, row: int) -> int:
        if isinstance(row, bool):
            raise TypeError("candidate row must be an integer, not bool")
        try:
            row_index = operator.index(row)
        except TypeError as error:
            raise TypeError("candidate row must be an integer") from error
        if not 0 <= row_index < len(self):
            raise IndexError("candidate row out of range")
        return row_index

    def candidate_count(self, row: int) -> int:
        """Return the number of targets retained in one bound row."""
        row_index = self._row_index(row)
        state = _candidate_state(self)
        if state.sealed_postfilter is not None:
            from . import _pipeline  # type: ignore[attr-defined]

            return int(
                _pipeline._sealed_postfilter_candidate_count_bound(
                    state.sealed_postfilter, row_index
                )
            )
        if state.postfilter_records is not None:
            offsets = memoryview(state.offsets).cast("Q")
            return offsets[row_index + 1] - offsets[row_index]
        if state.all_rows[row_index]:
            return len(state.all_targets) // 4
        offsets = memoryview(state.offsets).cast("Q")
        return offsets[row_index + 1] - offsets[row_index]

    def search(
        self,
        row: int,
        pipeline: Any,
        *,
        return_telemetry: bool = False,
    ) -> Any:
        """Search one bound row using an exclusively owned matching pipeline.

        The caller must not inspect, mutate, clear, or use ``pipeline`` from
        another thread until this call returns.
        """
        row_index = self._row_index(row)
        if type(return_telemetry) is not bool:
            raise TypeError("return_telemetry must be bool")
        if type(pipeline) is not astra_pyhmmer.plan7.Pipeline:
            raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
        if not _native.bias_host_environment_attested():
            raise RuntimeError(
                "candidate search requires the attested host floating-point environment"
            )
        candidate_state = _candidate_state(self)
        pair = candidate_state.pairs[row_index]
        sealed_postfilter = candidate_state.sealed_postfilter
        from . import _pipeline  # type: ignore[attr-defined]

        state = _pair_state(pair)
        if sealed_postfilter is not None:
            with _lease_pipeline(pipeline):
                with state.lock:
                    if _pipeline._sealed_sparse_journal_v3_enabled_bound(
                        sealed_postfilter
                    ):
                        return (
                            _pipeline._search_hmm_sealed_sparse_journal_v3_bound(
                                sealed_postfilter,
                                row_index,
                                pipeline,
                                _return_route_statistics=return_telemetry,
                            )
                        )
                    if return_telemetry:
                        return _pipeline._search_hmm_sealed_postfilter_bound(
                            sealed_postfilter,
                            row_index,
                            pipeline,
                            _return_route_statistics=True,
                        )
                    # Preserve the exact pre-telemetry call boundary in the
                    # default path, including monkeypatched/private callers.
                    return _pipeline._search_hmm_sealed_postfilter_bound(
                        sealed_postfilter, row_index, pipeline
                    )

        if return_telemetry:
            raise ValueError(
                "continuation telemetry requires a sealed fused batch"
            )

        postfilter_records = candidate_state.postfilter_records
        forward = candidate_state.forward
        forward_row = None
        forward_special_offsets = None
        forward_specials = None
        if postfilter_records is not None:
            offsets = memoryview(candidate_state.offsets).cast("Q")
            record_size = _native.POSTFILTER_RESULT_SIZE
            start = offsets[row_index] * record_size
            stop = offsets[row_index + 1] * record_size
            candidate_row = memoryview(postfilter_records)[start:stop]
            if forward is not None:
                forward_offsets = forward.row_offsets
                forward_start = forward_offsets[row_index]
                forward_stop = forward_offsets[row_index + 1]
                if forward_start != forward_stop:
                    start = forward_start * _native.FORWARD_RESULT_SIZE
                    stop = forward_stop * _native.FORWARD_RESULT_SIZE
                    forward_row = memoryview(forward.records)[start:stop]
                    forward_special_offsets = forward.special_offsets[
                        forward_start : forward_stop + 1
                    ]
                    forward_specials = forward.specials
        elif candidate_state.all_rows[row_index]:
            candidate_row = memoryview(candidate_state.all_targets).cast("I")
        else:
            offsets = memoryview(candidate_state.offsets).cast("Q")
            start = offsets[row_index]
            stop = offsets[row_index + 1]
            candidate_row = memoryview(candidate_state.indices).cast("I")[start:stop]
        residue_offsets = memoryview(candidate_state.residue_offsets).cast("Q")
        with _lease_pipeline(pipeline):
            if float(pipeline.F1) != candidate_state.f1:
                raise ValueError(
                    f"pipeline F1 {pipeline.F1!r} does not match "
                    f"candidate F1 {candidate_state.f1!r}"
                )
            if (
                _background_fingerprint(pipeline.background)
                != state.background_fingerprint
            ):
                raise ValueError(
                    "pipeline background does not match the canonical "
                    "hmmpress background"
                )
            with state.lock:
                if postfilter_records is not None:
                    if forward_row is not None:
                        if forward is None:
                            raise RuntimeError(
                                "Forward row is missing its batch provenance"
                            )
                        return _pipeline._search_hmm_postfilter_forward_bound(
                            pipeline,
                            state.hmm.copy(),
                            state.optimized_profile,
                            candidate_state.targets,
                            candidate_row,
                            forward_row,
                            forward_special_offsets,
                            forward_specials,
                            residue_offsets,
                            forward.f2_bits,
                            forward.f3_bits,
                            forward.bias_filter,
                        )
                    return _pipeline._search_hmm_postfilter_bound(
                        pipeline,
                        state.hmm.copy(),
                        state.optimized_profile,
                        candidate_state.targets,
                        candidate_row,
                        residue_offsets,
                    )
                return _pipeline._search_hmm_candidates_bound(
                    pipeline,
                    state.hmm.copy(),
                    state.optimized_profile,
                    candidate_state.targets,
                    candidate_row,
                    residue_offsets,
                )


_CANDIDATE_STATES: WeakKeyDictionary[CandidateBatch, _CandidateState] = (
    WeakKeyDictionary()
)


def _candidate_state(candidates: CandidateBatch) -> _CandidateState:
    try:
        return _CANDIDATE_STATES[candidates]
    except KeyError as error:
        raise TypeError(
            "CandidateBatch objects come from SequenceBatch.candidate_batch"
        ) from error


def _new_candidate_batch(
    pairs: tuple[PressedProfilePair, ...],
    targets: Any,
    residue_offsets: bytes,
    indices: array[int],
    offsets: array[int],
    all_rows: bytes,
    all_targets: array[int],
    f1: float,
    *,
    postfilter_records: bytes | None = None,
    forward: _ForwardAugmentation | None = None,
    sealed_postfilter: Any | None = None,
) -> CandidateBatch:
    candidates = object.__new__(CandidateBatch)
    frozen_all_rows = memoryview(all_rows).cast("B").tobytes()
    frozen_postfilter_records = (
        memoryview(postfilter_records).cast("B").tobytes()
        if sealed_postfilter is None and postfilter_records is not None
        else None
    )
    _CANDIDATE_STATES[candidates] = _CandidateState(
        pairs,
        targets,
        residue_offsets,
        indices.tobytes(),
        offsets.tobytes(),
        frozen_all_rows,
        all_targets.tobytes(),
        frozen_postfilter_records,
        forward if sealed_postfilter is None else None,
        sealed_postfilter,
        f1,
    )
    return candidates


class _SequenceState:
    __slots__ = (
        "alphabet",
        "native",
        "lock",
        "targets",
        "residue_offsets",
        "native_generation",
        "content_fingerprint",
        "execution_policy",
    )

    def __init__(
        self,
        alphabet: Any,
        native: Any,
        targets: Any,
        residue_offsets: bytes,
        native_generation: int,
        content_fingerprint: bytes,
        execution_policy: str,
    ) -> None:
        self.alphabet = alphabet
        self.native = native
        self.lock = Lock()
        self.targets = targets
        self.residue_offsets = residue_offsets
        self.native_generation = native_generation
        self.content_fingerprint = content_fingerprint
        self.execution_policy = execution_policy


_SEQUENCE_STATES: WeakKeyDictionary[Any, _SequenceState] = WeakKeyDictionary()


def _sequence_state(batch: Any) -> _SequenceState:
    try:
        return _SEQUENCE_STATES[batch]
    except KeyError as error:
        raise TypeError("invalid SequenceBatch object") from error


def _sequence_native(batch: Any) -> Any:
    """Return the concealed native batch for package diagnostics only."""
    return _sequence_state(batch).native


class SequenceBatch:
    """A packed target batch kept resident on one CUDA device."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        sequences: Iterable[Any],
        *,
        alphabet: Any | None = None,
        execution_policy: str = "auto",
        _forward_cpu_max_cells: int | None = None,
    ):
        policy_name, policy_code = _normalize_execution_policy(execution_policy)
        inherited_alphabet = getattr(sequences, "alphabet", None)
        sequence_list = list(sequences)
        if alphabet is None:
            alphabet = inherited_alphabet
        if alphabet is None and sequence_list:
            alphabet = sequence_list[0].alphabet
        if alphabet is None:
            raise ValueError("an alphabet is required for an empty sequence batch")

        if any(
            type(sequence) is not astra_pyhmmer.easel.DigitalSequence
            for sequence in sequence_list
        ):
            raise TypeError(
                "sequence batches require exact DigitalSequence objects"
            )
        target_sequences = [sequence.copy() for sequence in sequence_list]
        residues = bytearray()
        offsets = array("Q", [0])
        for sequence in target_sequences:
            if sequence.alphabet != alphabet:
                raise ValueError("sequence alphabets differ")
            encoded = memoryview(sequence.sequence).cast("B")
            if len(encoded) > 100_000:
                raise ValueError("sequence length exceeds HMMER's protein limit")
            residues.extend(encoded)
            offsets.append(len(residues))
        targets = astra_pyhmmer.easel.DigitalSequenceBlock(alphabet, target_sequences)

        content_fingerprint = sequence_content_fingerprint(
            alphabet.Kp, residues, offsets
        )
        if sequence_block_content_fingerprint(targets) != content_fingerprint:
            raise RuntimeError("copied target content fingerprint changed")
        native_arguments = (residues, offsets, alphabet.Kp, policy_code)
        native = (
            _native.SequenceBatch(*native_arguments)
            if _forward_cpu_max_cells is None
            else _native.SequenceBatch(
                *native_arguments, _forward_cpu_max_cells
            )
        )
        native_generation, native_content_fingerprint = (
            native._generation_and_content_for_seal()
        )
        if native_content_fingerprint != content_fingerprint:
            native.close()
            raise RuntimeError("native target content fingerprint changed")
        _SEQUENCE_STATES[self] = _SequenceState(
            alphabet,
            native,
            targets,
            offsets.tobytes(),
            native_generation,
            content_fingerprint,
            policy_name,
        )

    @property
    def alphabet(self) -> Any:
        return _sequence_state(self).alphabet

    def __len__(self) -> int:
        return len(_sequence_state(self).native)

    @property
    def closed(self) -> bool:
        state = _sequence_state(self)
        with state.lock:
            return bool(state.native.closed)

    @property
    def execution_policy(self) -> str:
        """Return the immutable request-scoped GPU execution policy."""
        return _sequence_state(self).execution_policy

    @property
    def execution_policy_statistics(self) -> dict[str, int | str]:
        """Return exact cumulative choices made by the GPU policy."""
        state = _sequence_state(self)
        with state.lock:
            statistics = state.native.workspace_statistics
        return {
            "schema_version": statistics["execution_policy_version"],
            "mode": state.execution_policy,
            "mode_code": statistics["execution_policy_mode"],
            "target_count": statistics["execution_policy_target_count"],
            "length_class_count": statistics[
                "execution_policy_length_class_count"
            ],
            "f1_run_count": statistics["execution_policy_f1_run_count"],
            "profile_packed_run_count": statistics[
                "f1_profile_packed_run_count"
            ],
            "profile_packed_profile_count": statistics[
                "f1_profile_packed_profile_count"
            ],
            "profile_scalar_profile_count": statistics[
                "f1_profile_scalar_profile_count"
            ],
            "length_class_run_count": statistics["f1_length_class_run_count"],
            "full_msv_compaction_run_count": statistics[
                "full_msv_compaction_run_count"
            ],
            "full_msv_legacy_run_count": statistics[
                "full_msv_legacy_run_count"
            ],
            "full_msv_packed_run_count": statistics[
                "full_msv_packed_run_count"
            ],
            "forward_candidates_per_warp": statistics[
                "execution_policy_forward_candidates_per_warp"
            ],
        }

    @property
    def memory_snapshot(self) -> dict[str, Any]:
        """Return serialized CUDA availability and device capacities."""
        state = _sequence_state(self)
        with state.lock:
            return cast(dict[str, Any], state.native.memory_snapshot)

    def close(self) -> None:
        state = _sequence_state(self)
        with state.lock:
            state.native.close()

    def __enter__(self) -> SequenceBatch:
        if self.closed:
            raise RuntimeError("sequence batch is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _filter_raw(self, optimized_profile: Any) -> list[tuple[Any, ...]]:
        state = _sequence_state(self)
        with state.lock:
            if optimized_profile.alphabet != state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            scores = memoryview(optimized_profile.sbv).cast("B")
            return cast(
                list[tuple[Any, ...]],
                state.native.filter_raw(
                    scores,
                    optimized_profile.sbv.shape[1],
                    optimized_profile.M,
                    optimized_profile.alphabet.Kp,
                    optimized_profile.tbm,
                    optimized_profile.tec,
                    optimized_profile.base_b,
                    optimized_profile.bias_b,
                    optimized_profile.scale_b,
                ),
            )

    def filter_ssv(self, optimized_profile: Any) -> list[dict[str, Any]]:
        return _format_results(self._filter_raw(optimized_profile))

    def cpu_candidates(self, optimized_profile: Any, F1: float = 0.02) -> list[int]:
        """Return target indexes that still require HMMER's CPU pipeline.

        Only finite direct-SSV scores with a P-value strictly greater than
        ``F1`` are omitted. Every fallback, overflow, empty, or invalid case
        is retained conservatively.
        """
        state = _sequence_state(self)
        with state.lock:
            if optimized_profile.alphabet != state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            if state.native.closed:
                raise RuntimeError("sequence batch is closed")

            try:
                threshold = float(F1)
            except (TypeError, ValueError, OverflowError):
                return list(range(len(self)))

            if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
                return list(range(len(self)))
            parameters = _f1_parameters(optimized_profile)
            if parameters is None:
                return list(range(len(self)))
            m_mu, m_lambda, scale = parameters

            scores = memoryview(optimized_profile.sbv).cast("B")
            return cast(
                list[int],
                state.native.cpu_candidates_raw(
                    scores,
                    optimized_profile.sbv.shape[1],
                    optimized_profile.M,
                    optimized_profile.alphabet.Kp,
                    optimized_profile.tbm,
                    optimized_profile.tec,
                    optimized_profile.base_b,
                    optimized_profile.bias_b,
                    scale,
                    m_mu,
                    m_lambda,
                    threshold,
                ),
            )

    def filter_ssv_many(
        self, optimized_profiles: Iterable[Any]
    ) -> list[list[dict[str, Any]]]:
        profiles = list(optimized_profiles)
        state = _sequence_state(self)
        with state.lock:
            if state.native.closed:
                raise RuntimeError("sequence batch is closed")
            for profile in profiles:
                if profile.alphabet != state.alphabet:
                    raise ValueError("profile and sequence alphabets differ")
            packed = _pack_profiles(profiles)
            raw = cast(
                list[list[tuple[Any, ...]]],
                state.native.filter_many_raw(*packed),
            )
        return [_format_results(profile_results) for profile_results in raw]

    def cpu_candidates_many(
        self, optimized_profiles: Iterable[Any], F1: float = 0.02
    ) -> list[list[int]]:
        profiles = list(optimized_profiles)
        state = _sequence_state(self)
        with state.lock:
            if state.native.closed:
                raise RuntimeError("sequence batch is closed")
            for profile in profiles:
                if profile.alphabet != state.alphabet:
                    raise ValueError("profile and sequence alphabets differ")
            try:
                threshold = float(F1)
            except (TypeError, ValueError, OverflowError):
                return [list(range(len(self))) for _ in profiles]
            if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
                return [list(range(len(self))) for _ in profiles]

            output: list[list[int] | None] = [None] * len(profiles)
            valid_profiles = []
            valid_indices = []
            m_mu = array("f")
            m_lambda = array("f")
            for index, profile in enumerate(profiles):
                parameters = _f1_parameters(profile)
                if parameters is None:
                    output[index] = list(range(len(self)))
                else:
                    profile_mu, profile_lambda, _ = parameters
                    valid_profiles.append(profile)
                    valid_indices.append(index)
                    m_mu.append(profile_mu)
                    m_lambda.append(profile_lambda)

            if valid_profiles:
                packed = _pack_profiles(valid_profiles)
                candidates = cast(
                    list[list[int]],
                    state.native.cpu_candidates_many_raw(
                        *packed, m_mu, m_lambda, threshold
                    ),
                )
                for index, profile_candidates in zip(
                    valid_indices, candidates, strict=True
                ):
                    output[index] = profile_candidates

            return cast(list[list[int]], output)

    def _candidate_csr_locked(
        self, profiles: list[Any], threshold: float
    ) -> tuple[array[int], array[int], bytes, array[int]]:
        """Build complete CSR rows, promoting invalid profiles fail-closed."""
        state = _sequence_state(self)
        if not profiles:
            return array("I"), array("Q", [0]), b"", array("I")
        if threshold == 1.0:
            return (
                array("I"),
                array("Q", [0]) * (len(profiles) + 1),
                bytes([1]) * len(profiles),
                array("I", range(len(state.native))),
            )

        parameters = [_f1_parameters(profile) for profile in profiles]
        parameters = [
            profile_parameters
            if (
                profile_parameters is not None
                and _native.f1_cutoff(
                    profile_parameters[0], profile_parameters[1], threshold
                )[0]
                != _native.F1_CUTOFF_INVALID
            )
            else None
            for profile_parameters in parameters
        ]
        valid_profiles = [
            profile
            for profile, profile_parameters in zip(profiles, parameters, strict=True)
            if profile_parameters is not None
        ]
        valid_indices = array("I")
        valid_offsets = array("Q", [0])
        if valid_profiles:
            m_mu = array(
                "f",
                [
                    cast(tuple[float, float, float], profile_parameters)[0]
                    for profile_parameters in parameters
                    if profile_parameters is not None
                ],
            )
            m_lambda = array(
                "f",
                [
                    cast(tuple[float, float, float], profile_parameters)[1]
                    for profile_parameters in parameters
                    if profile_parameters is not None
                ],
            )
            packed = _pack_profiles(valid_profiles)
            valid_indices, valid_offsets = cast(
                tuple[array[int], array[int]],
                state.native.cpu_candidates_many_csr_raw(
                    *packed, m_mu, m_lambda, threshold
                ),
            )

        if len(valid_profiles) == len(profiles):
            return (
                valid_indices,
                valid_offsets,
                bytes(len(profiles)),
                array("I"),
            )

        all_targets = array("I", range(len(state.native)))
        indices = array("I")
        offsets = array("Q", [0])
        all_rows = bytearray()
        valid_row = 0
        for profile_parameters in parameters:
            if profile_parameters is None:
                all_rows.append(1)
            else:
                start = valid_offsets[valid_row]
                stop = valid_offsets[valid_row + 1]
                indices.frombytes(memoryview(valid_indices)[start:stop].cast("B"))
                valid_row += 1
                all_rows.append(0)
            offsets.append(len(indices))
        return indices, offsets, bytes(all_rows), all_targets

    def candidate_batch(
        self,
        profile_pairs: Iterable[PressedProfilePair],
        F1: float = 0.02,
    ) -> CandidateBatch:
        """Bind exact CUDA candidates to immutable pressed-profile pairs."""
        sequence_state = _sequence_state(self)
        pairs = tuple(profile_pairs)
        try:
            threshold = float(F1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("F1 must be a finite number in [0, 1]") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("F1 must be a finite number in [0, 1]")

        profiles = []
        states = []
        for pair in pairs:
            if type(pair) is not PressedProfilePair:
                raise TypeError(
                    "candidate batches require pairs from load_pressed_profiles"
                )
            state = _pair_state(pair)
            if state.hmm.alphabet != sequence_state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            profiles.append(state.optimized_profile)
            states.append(state)

        unique_locks = {id(state.lock): state.lock for state in states}
        with ExitStack() as locks:
            for lock_id in sorted(unique_locks):
                locks.enter_context(unique_locks[lock_id])
            with sequence_state.lock:
                if sequence_state.native.closed:
                    raise RuntimeError("sequence batch is closed")
                indices, offsets, all_rows, all_targets = self._candidate_csr_locked(
                    profiles, threshold
                )

        return _new_candidate_batch(
            pairs,
            sequence_state.targets,
            sequence_state.residue_offsets,
            indices,
            offsets,
            all_rows,
            all_targets,
            threshold,
        )

    def postfilter_batch(
        self,
        profile_pairs: Iterable[PressedProfilePair],
        F1: float = 0.02,
    ) -> CandidateBatch:
        """Bind exact CUDA MSV, bias, and Viterbi records to pressed pairs."""
        return self._postfilter_batch(profile_pairs, F1, None)

    def postfilter_selection(
        self,
        selection: ProfileSelection,
        F1: float = 0.02,
    ) -> CandidateBatch:
        """Generate exact records from an immutable host profile selection."""
        return self._postfilter_selection(selection, F1, None)

    def _postfilter_forward_selection(
        self,
        selection: ProfileSelection,
        F1: float,
        F2: float,
        F3: float,
        bias_filter: bool,
        *,
        pipeline: Any | None = None,
        domain_guard: float = 2.0e-4,
        _rescore_compact_byte_budget: int = 0,
        _rescore_matrix_byte_budget: int = 0,
        _rescore_trace_byte_budget: int = 0,
        _rescore_test_fault: int = 0,
        telemetry: bool = False,
        sparse_journal_v3: bool = False,
        _ga_pruning: bool = False,
    ) -> CandidateBatch:
        """Build a sealed selection batch with bounded Forward results."""
        try:
            f2 = float(F2)
            f3 = float(F3)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Forward thresholds must be real numbers") from error
        if type(bias_filter) is not bool:
            raise TypeError("bias_filter must be bool")
        if type(telemetry) is not bool:
            raise TypeError("telemetry must be bool")
        if type(sparse_journal_v3) is not bool:
            raise TypeError("sparse_journal_v3 must be bool")
        if type(_ga_pruning) is not bool:
            raise TypeError("_ga_pruning must be bool")
        if pipeline is not None:
            return self._postfilter_forward_domain_selection(
                selection,
                F1,
                f2,
                f3,
                bias_filter,
                pipeline,
                domain_guard,
                _rescore_compact_byte_budget,
                _rescore_matrix_byte_budget,
                _rescore_trace_byte_budget,
                _rescore_test_fault,
                telemetry,
                sparse_journal_v3,
                _ga_pruning,
            )
        if any(
            value != 0
            for value in (
                _rescore_compact_byte_budget,
                _rescore_matrix_byte_budget,
                _rescore_trace_byte_budget,
                _rescore_test_fault,
            )
        ):
            raise ValueError("domain-rescore controls require a pipeline")
        if telemetry:
            raise ValueError(
                "generation telemetry requires a sealed fused pipeline batch"
            )
        if sparse_journal_v3:
            raise ValueError(
                "sparse journal v3 requires a sealed fused pipeline batch"
            )
        if _ga_pruning:
            raise ValueError("GA pruning requires a sealed fused pipeline batch")
        return self._postfilter_selection(
            selection, F1, (f2, f3, bias_filter)
        )

    def _postfilter_forward_domain_selection(
        self,
        selection: ProfileSelection,
        F1: float,
        f2: float,
        f3: float,
        bias_filter: bool,
        pipeline: Any,
        domain_guard: float,
        rescore_compact_byte_budget: int,
        rescore_matrix_byte_budget: int,
        rescore_trace_byte_budget: int,
        rescore_test_fault: int,
        telemetry: bool,
        sparse_journal_v3: bool,
        ga_pruning: bool,
    ) -> CandidateBatch:
        """Build one opaque fused Forward/domain continuation batch."""
        from . import _pipeline  # type: ignore[attr-defined]

        if type(pipeline) is not astra_pyhmmer.plan7.Pipeline:
            raise TypeError("pipeline must be exactly astra_pyhmmer.plan7.Pipeline")
        if type(selection) is not ProfileSelection:
            raise TypeError(
                "profile continuation requires ProfileSession.select output"
            )
        try:
            threshold = float(F1)
            guard = float(domain_guard)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("continuation thresholds must be real numbers") from error
        normalized_controls = []
        for name, value in (
            ("rescore compact byte budget", rescore_compact_byte_budget),
            ("rescore matrix byte budget", rescore_matrix_byte_budget),
            ("rescore trace byte budget", rescore_trace_byte_budget),
            ("rescore test fault", rescore_test_fault),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a nonnegative integer")
            try:
                normalized = operator.index(value)
            except TypeError as error:
                raise TypeError(
                    f"{name} must be a nonnegative integer"
                ) from error
            if normalized < 0:
                raise ValueError(f"{name} must be nonnegative")
            normalized_controls.append(normalized)
        (
            rescore_compact_byte_budget,
            rescore_matrix_byte_budget,
            rescore_trace_byte_budget,
            rescore_test_fault,
        ) = normalized_controls
        if (
            not math.isfinite(threshold)
            or not 0.0 <= threshold < 1.0
            or not math.isfinite(f2)
            or not 0.0 <= f2 <= 1.0
            or not math.isfinite(f3)
            or not 0.0 <= f3 <= 1.0
            or not math.isfinite(guard)
            or not 2.0e-4 <= guard <= 1.0
        ):
            raise ValueError("invalid fused continuation thresholds")
        if not bias_filter:
            raise ValueError("fused continuation requires bias filtering")
        if not _native.bias_host_environment_attested():
            raise RuntimeError(
                "fused continuation requires the attested floating-point environment"
            )
        if (
            not _pipeline._filter_scores_seam_available()
            or not _pipeline._filter_and_forward_scores_seam_available()
            or not _pipeline._simple_regions_seam_available()
        ):
            raise RuntimeError(
                "fused continuation requires the project-private HMMER seams"
            )

        sequence_state = _sequence_state(self)
        selection_state = _profile_selection_state(selection)
        if selection_state.alphabet != sequence_state.alphabet:
            raise ValueError("profile and sequence alphabets differ")
        states = [_pair_state(pair) for pair in selection_state.pairs]
        unique_locks = {id(state.lock): state.lock for state in states}
        capsule: Any | None = None
        native_stage_timings: Any | None = None
        generation_statistics: Any | None = None
        compact_tail_fingerprint = 0
        ga_target_cutoffs: array[float] | None = None

        with _lease_pipeline(pipeline):
            _pipeline._validate_simple_region_generation_bound(
                pipeline, threshold, f2, f3, bias_filter, guard
            )
            compact_seam_available = (
                _pipeline._compact_domains_seam_available()
            )
            if (
                any(normalized_controls)
                and not compact_seam_available
            ):
                raise RuntimeError(
                    "domain-rescore controls require the compact-domain seam"
                )
            if compact_seam_available:
                compact_tail_fingerprint = (
                    _pipeline._compact_tail_fingerprint_bound(pipeline)
                )
            if ga_pruning:
                if not sparse_journal_v3:
                    raise ValueError("GA pruning requires sparse journal v3")
                if pipeline.bit_cutoffs != "gathering":
                    raise ValueError("GA pruning requires gathering cutoffs")
                cutoff_values = []
                for pair in selection_state.pairs:
                    gathering = pair.cutoffs.gathering
                    if gathering is None or not math.isfinite(gathering[0]):
                        raise ValueError(
                            "GA pruning requires finite target gathering cutoffs"
                        )
                    cutoff_values.append(gathering[0])
                ga_target_cutoffs = array("f", cutoff_values)

            # Native generation is fully self-contained. Capture it under the
            # selection/sequence locks, then release them before pair locks to
            # preserve the package-wide pair -> sequence lock order.
            with selection_state.lock:
                if selection_state.native.closed:
                    raise RuntimeError("profile selection is closed")
                with sequence_state.lock:
                    if sequence_state.native.closed:
                        raise RuntimeError("sequence batch is closed")
                    native_generation, native_content = (
                        sequence_state.native._generation_and_content_for_seal()
                    )
                    if (
                        native_generation != sequence_state.native_generation
                        or native_content != sequence_state.content_fingerprint
                    ):
                        raise RuntimeError("native target identity changed")
                    selection_identity = selection_state.native.identity
                    identity_tokens = (
                        selection_state.native._identity_tokens_for_seal()
                    )
                    native_profile_fingerprints = (
                        selection_state.native._fingerprints_for_seal()
                    )
                    expected_profile_fingerprints = b"".join(
                        selection_state.profile_fingerprints
                    )
                    if native_profile_fingerprints != expected_profile_fingerprints:
                        raise RuntimeError(
                            "native profile selection fingerprint changed"
                        )
                    native_result = (
                        sequence_state.native._postfilter_forward_domain_selection_sealed(
                            selection_state.native,
                            threshold,
                            f2,
                            f3,
                            guard,
                            gathered_byte_budget=_FORWARD_SPECIAL_BYTE_BUDGET,
                            rescore_compact_byte_budget=(
                                rescore_compact_byte_budget
                            ),
                            rescore_matrix_byte_budget=(
                                rescore_matrix_byte_budget
                            ),
                            rescore_trace_byte_budget=(
                                rescore_trace_byte_budget
                            ),
                            _rescore_test_fault=rescore_test_fault,
                            generation_tail_fingerprint=(
                                compact_tail_fingerprint
                            ),
                            _return_stage_timings=True,
                            _return_generation_statistics=telemetry,
                            _direct_sparse_v3=sparse_journal_v3,
                            _ga_target_cutoffs=ga_target_cutoffs,
                        )
                    )
                    if telemetry:
                        (
                            capsule,
                            native_stage_timings,
                            generation_statistics,
                        ) = native_result
                    else:
                        capsule, native_stage_timings = native_result

            with ExitStack() as locks:
                for lock_id in sorted(unique_locks):
                    locks.enter_context(unique_locks[lock_id])
                if any(
                    selection_state.queries[index] is not state.hmm
                    or selection_state.profiles[index]
                    is not state.optimized_profile
                    for index, state in enumerate(states)
                ):
                    raise ValueError("selected pair objects differ")
                if tuple(
                    state.profile_fingerprint for state in states
                ) != selection_state.profile_fingerprints:
                    raise ValueError(
                        "selected pair optimized-profile snapshots differ"
                    )
                if any(
                    state.background_fingerprint
                    != selection_state.background_fingerprint
                    for state in states
                ):
                    raise ValueError("selected pair backgrounds differ")
                sealed_postfilter = (
                    _pipeline._seal_profile_selection_continuation_bound(
                        selection_state.queries,
                        selection_state.profiles,
                        sequence_state.targets,
                        memoryview(sequence_state.residue_offsets).cast("Q"),
                        threshold,
                        memoryview(selection_state.background_fingerprint),
                        capsule,
                        selection_identity,
                        memoryview(identity_tokens).cast("Q"),
                        memoryview(expected_profile_fingerprints),
                        native_generation,
                        memoryview(sequence_state.content_fingerprint),
                        pipeline,
                        guard,
                        native_stage_timings,
                        generation_statistics,
                        sparse_journal_v3,
                    )
                )
                capsule = None
                native_stage_timings = None
                generation_statistics = None

        offsets = array("Q", [0]) * (len(selection_state.pairs) + 1)
        return _new_candidate_batch(
            selection_state.pairs,
            sequence_state.targets,
            sequence_state.residue_offsets,
            array("I"),
            offsets,
            bytes(len(selection_state.pairs)),
            array("I"),
            threshold,
            sealed_postfilter=sealed_postfilter,
        )

    def _postfilter_selection(
        self,
        selection: ProfileSelection,
        F1: float,
        forward_options: tuple[float, float, bool] | None,
    ) -> CandidateBatch:
        from . import _pipeline  # type: ignore[attr-defined]

        if not _pipeline._filter_scores_seam_available():
            raise RuntimeError(
                "post-filter batches require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )
        if type(selection) is not ProfileSelection:
            raise TypeError(
                "postfilter_selection requires ProfileSession.select output"
            )
        try:
            threshold = float(F1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("F1 must be a finite number in [0, 1]") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("F1 must be a finite number in [0, 1]")

        sequence_state = _sequence_state(self)
        selection_state = _profile_selection_state(selection)
        if selection_state.alphabet != sequence_state.alphabet:
            raise ValueError("profile and sequence alphabets differ")
        forward: _ForwardAugmentation | None = None
        expected_forward_indices: Any = array("I")
        sealed_postfilter: Any | None = None
        seal_factory = getattr(_pipeline, "_seal_postfilter_batch_bound", None)
        with selection_state.lock:
            if selection_state.native.closed:
                raise RuntimeError("profile selection is closed")
            with sequence_state.lock:
                if sequence_state.native.closed:
                    raise RuntimeError("sequence batch is closed")
                if threshold == 1.0:
                    records = None
                    offsets = array("Q", [0]) * (len(selection_state.pairs) + 1)
                    all_rows = bytes([1]) * len(selection_state.pairs)
                    all_targets = (
                        array("I", range(len(sequence_state.native)))
                        if selection_state.pairs
                        else array("I")
                    )
                else:
                    raw_records, offsets = cast(
                        tuple[bytearray, array[int]],
                        sequence_state.native.postfilter_profile_selection_csr_raw(
                            selection_state.native,
                            threshold,
                        ),
                    )
                    records = bytes(raw_records)
                    all_rows = bytes(len(selection_state.pairs))
                    all_targets = array("I")

                    if (
                        forward_options is not None
                        and forward_options[2]
                        and math.isfinite(forward_options[1])
                        and forward_options[1] >= 0.0
                        and _pipeline._filter_and_forward_scores_seam_available()
                        and hasattr(
                            type(sequence_state.native),
                            "forward_profile_selection_raw",
                        )
                    ):
                        f2, f3, bias_filter = forward_options
                        (
                            forward_records,
                            forward_row_offsets,
                            expected_forward_indices,
                            special_offsets,
                            specials,
                            statistics,
                        ) = sequence_state.native.forward_profile_selection_raw(
                            selection_state.native,
                            memoryview(records),
                            memoryview(offsets),
                            memoryview(sequence_state.residue_offsets).cast("Q"),
                            f2,
                            f3,
                            gathered_byte_budget=(_FORWARD_SPECIAL_BYTE_BUDGET),
                        )
                        f2_bits = struct.unpack("=Q", struct.pack("=d", f2))[0]
                        f3_bits = struct.unpack("=Q", struct.pack("=d", f3))[0]
                        if statistics["generation_f3_bits"] != f3_bits:
                            raise RuntimeError(
                                "Forward generation F3 provenance changed"
                            )
                        if statistics["candidate_count"] != len(
                            expected_forward_indices
                        ):
                            raise RuntimeError(
                                "Forward result count differs from input"
                            )
                        if statistics["candidate_count"]:
                            forward = _ForwardAugmentation(
                                bytes(forward_records),
                                forward_row_offsets,
                                special_offsets,
                                specials,
                                f2_bits,
                                f3_bits,
                                bias_filter,
                            )

        if records is not None and seal_factory is not None:
            states = [_pair_state(pair) for pair in selection_state.pairs]
            unique_locks = {id(state.lock): state.lock for state in states}
            with ExitStack() as locks:
                for lock_id in sorted(unique_locks):
                    locks.enter_context(unique_locks[lock_id])
                background = astra_pyhmmer.plan7.Background(sequence_state.alphabet)
                canonical_background = _background_fingerprint(background)
                if any(
                    state.background_fingerprint != canonical_background
                    for state in states
                ):
                    raise ValueError(
                        "pressed profiles do not share the canonical "
                        "hmmpress background"
                    )
                seal_options: dict[str, Any] = {}
                if forward is not None:
                    seal_options = {
                        "forward_records": memoryview(forward.records),
                        "forward_offsets": forward.row_offsets,
                        "special_offsets": forward.special_offsets,
                        "specials": forward.specials,
                        "expected_forward_indices": memoryview(
                            expected_forward_indices
                        ),
                        "generation_f2_bits": forward.f2_bits,
                        "generation_f3_bits": forward.f3_bits,
                        "generation_bias_filter": forward.bias_filter,
                    }
                sealed_postfilter = seal_factory(
                    tuple(state.hmm for state in states),
                    tuple(state.optimized_profile for state in states),
                    sequence_state.targets,
                    memoryview(records),
                    memoryview(offsets),
                    memoryview(sequence_state.residue_offsets).cast("Q"),
                    threshold,
                    memoryview(canonical_background),
                    _residue_offsets_shared=True,
                    _background_fingerprint_shared=False,
                    **seal_options,
                )

        return _new_candidate_batch(
            selection_state.pairs,
            sequence_state.targets,
            sequence_state.residue_offsets,
            array("I"),
            offsets,
            all_rows,
            all_targets,
            threshold,
            postfilter_records=records,
            forward=forward if sealed_postfilter is None else None,
            sealed_postfilter=sealed_postfilter,
        )

    def _postfilter_forward_batch(
        self,
        profile_pairs: Iterable[PressedProfilePair],
        F1: float,
        F2: float,
        F3: float,
        bias_filter: bool,
    ) -> CandidateBatch:
        """Build a private post-filter batch with bounded Forward results."""
        try:
            f2 = float(F2)
            f3 = float(F3)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Forward thresholds must be real numbers") from error
        if type(bias_filter) is not bool:
            raise TypeError("bias_filter must be bool")
        return self._postfilter_batch(profile_pairs, F1, (f2, f3, bias_filter))

    def _postfilter_batch(
        self,
        profile_pairs: Iterable[PressedProfilePair],
        F1: float,
        forward_options: tuple[float, float, bool] | None,
    ) -> CandidateBatch:
        from . import _pipeline  # type: ignore[attr-defined]

        if not _pipeline._filter_scores_seam_available():
            raise RuntimeError(
                "post-filter batches require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

        sequence_state = _sequence_state(self)
        pairs = tuple(profile_pairs)
        try:
            threshold = float(F1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("F1 must be a finite number in [0, 1]") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("F1 must be a finite number in [0, 1]")

        profiles = []
        states = []
        for pair in pairs:
            if type(pair) is not PressedProfilePair:
                raise TypeError(
                    "post-filter batches require pairs from load_pressed_profiles"
                )
            state = _pair_state(pair)
            if state.hmm.alphabet != sequence_state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            profiles.append(state.optimized_profile)
            states.append(state)

        postfilter_records: bytes | None = None
        forward: _ForwardAugmentation | None = None
        sealed_postfilter: Any | None = None
        seal_factory = getattr(_pipeline, "_seal_postfilter_batch_bound", None)
        unique_locks = {id(state.lock): state.lock for state in states}
        with ExitStack() as locks:
            for lock_id in sorted(unique_locks):
                locks.enter_context(unique_locks[lock_id])
            with sequence_state.lock:
                if sequence_state.native.closed:
                    raise RuntimeError("sequence batch is closed")
                inputs = None
                if threshold < 1.0:
                    background = astra_pyhmmer.plan7.Background(sequence_state.alphabet)
                    inputs = _pack_postfilter_inputs(background, profiles, threshold)
                if inputs is None:
                    indices, offsets, all_rows, all_targets = (
                        self._candidate_csr_locked(profiles, threshold)
                    )
                else:
                    packed_bias, m_mu, m_lambda = inputs
                    packed = _pack_profiles(profiles)
                    with _native.ViterbiProfiles(profiles) as viterbi_profiles:
                        records, offsets = cast(
                            tuple[bytearray, array[int]],
                            sequence_state.native.postfilter_candidates_many_csr_raw(
                                *packed,
                                m_mu,
                                m_lambda,
                                threshold,
                                packed_bias,
                                profiles,
                                viterbi_profiles,
                            ),
                        )
                    postfilter_records = bytes(records)
                    if (
                        forward_options is not None
                        and forward_options[2]
                        and math.isfinite(forward_options[1])
                        and forward_options[1] >= 0.0
                        and _pipeline._filter_and_forward_scores_seam_available()
                        and hasattr(_native, "ForwardProfiles")
                        and hasattr(
                            type(sequence_state.native),
                            "forward_candidates_many_raw",
                        )
                    ):
                        f2, f3, bias_filter = forward_options
                        (
                            forward_row_offsets,
                            forward_indices,
                            forward_filters,
                        ) = _pipeline._select_forward_inputs_bound(
                            profiles,
                            memoryview(postfilter_records),
                            memoryview(offsets),
                            memoryview(sequence_state.residue_offsets).cast("Q"),
                            f2,
                        )
                        if len(forward_indices) != 0:
                            with _native.ForwardProfiles(profiles) as forward_profiles:
                                (
                                    forward_records,
                                    special_offsets,
                                    specials,
                                    statistics,
                                ) = sequence_state.native.forward_candidates_many_raw(
                                    forward_row_offsets,
                                    forward_indices,
                                    forward_filters,
                                    f3,
                                    profiles,
                                    forward_profiles,
                                    gathered_byte_budget=(_FORWARD_SPECIAL_BYTE_BUDGET),
                                )
                            f2_bits = struct.unpack("=Q", struct.pack("=d", f2))[0]
                            f3_bits = struct.unpack("=Q", struct.pack("=d", f3))[0]
                            if statistics["generation_f3_bits"] != f3_bits:
                                raise RuntimeError(
                                    "Forward generation F3 provenance changed"
                                )
                            if statistics["candidate_count"] != len(forward_indices):
                                raise RuntimeError(
                                    "Forward result count differs from input"
                                )
                            if seal_factory is None:
                                _pipeline._validate_forward_batch_bound(
                                    sequence_state.targets,
                                    memoryview(postfilter_records),
                                    memoryview(offsets),
                                    memoryview(forward_records),
                                    memoryview(forward_row_offsets),
                                    memoryview(forward_indices),
                                    memoryview(special_offsets),
                                    memoryview(specials),
                                )
                            forward = _ForwardAugmentation(
                                bytes(forward_records),
                                forward_row_offsets,
                                special_offsets,
                                specials,
                                f2_bits,
                                f3_bits,
                                bias_filter,
                            )
                    indices = array("I")
                    all_rows = bytes(len(profiles))
                    all_targets = array("I")
                    if seal_factory is not None:
                        canonical_background = _background_fingerprint(background)
                        if any(
                            state.background_fingerprint != canonical_background
                            for state in states
                        ):
                            raise ValueError(
                                "pressed profiles do not share the canonical "
                                "hmmpress background"
                            )
                        seal_options: dict[str, Any] = {}
                        if forward is not None:
                            seal_options = {
                                "forward_records": memoryview(forward.records),
                                "forward_offsets": forward.row_offsets,
                                "special_offsets": forward.special_offsets,
                                "specials": forward.specials,
                                "expected_forward_indices": memoryview(forward_indices),
                                "generation_f2_bits": forward.f2_bits,
                                "generation_f3_bits": forward.f3_bits,
                                "generation_bias_filter": forward.bias_filter,
                            }
                        sealed_postfilter = seal_factory(
                            tuple(state.hmm for state in states),
                            tuple(profiles),
                            sequence_state.targets,
                            memoryview(postfilter_records),
                            memoryview(offsets),
                            memoryview(sequence_state.residue_offsets).cast("Q"),
                            threshold,
                            memoryview(canonical_background),
                            _residue_offsets_shared=True,
                            _background_fingerprint_shared=False,
                            **seal_options,
                        )

        return _new_candidate_batch(
            pairs,
            sequence_state.targets,
            sequence_state.residue_offsets,
            indices,
            offsets,
            all_rows,
            all_targets,
            threshold,
            postfilter_records=postfilter_records,
            # The seal owns frozen Forward storage; retain the Python wrapper
            # only for the stock-extension fallback path.
            forward=forward if sealed_postfilter is None else None,
            sealed_postfilter=sealed_postfilter,
        )


def filter_ssv(
    optimized_profile: Any, sequences: Iterable[Any] | SequenceBatch
) -> list[dict[str, Any]]:
    """Run direct SSV without making the pipeline's F1 decision.

    ``eslENORESULT`` requires CPU full-MSV fallback, while ``eslERANGE`` is a
    guaranteed overflow that must be promoted. Empty targets are skipped by
    HMMER's comparison pipeline.
    """
    if isinstance(sequences, SequenceBatch):
        return sequences.filter_ssv(optimized_profile)
    with SequenceBatch(sequences, alphabet=optimized_profile.alphabet) as batch:
        return batch.filter_ssv(optimized_profile)


def cpu_candidates(
    optimized_profile: Any,
    sequences: Iterable[Any] | SequenceBatch,
    F1: float = 0.02,
) -> list[int]:
    """Return indexes that cannot be rejected safely by direct CUDA SSV."""
    if isinstance(sequences, SequenceBatch):
        return sequences.cpu_candidates(optimized_profile, F1)
    with SequenceBatch(sequences, alphabet=optimized_profile.alphabet) as batch:
        return batch.cpu_candidates(optimized_profile, F1)
