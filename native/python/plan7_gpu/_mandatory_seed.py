"""Offline proof machinery for the Phase 10 mandatory-seed experiment.

This module is deliberately outside production dispatch.  It turns the exact
signed-byte SSV recurrence into a conservative integer gain threshold and
enumerates a finite seed family whose absence certifies an F1 rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class SSVSeedParameters:
    """The byte constants and exact compiled F1 decision for one profile."""

    tbm: int
    tec: int
    base: int
    bias: int
    scale: float
    cutoff_mode: int
    cutoff_bit_score: float


@dataclass(frozen=True)
class SeedEntry:
    """One minimal threshold-crossing word and its model provenance."""

    word: bytes
    block_index: int
    model_start: int
    model_stop: int


@dataclass(frozen=True)
class MandatorySeedPlan:
    """A complete certificate for one profile and target-length cohort."""

    required_gain: int
    blocks: tuple[tuple[int, int], ...]
    quotas: tuple[int, ...]
    entries: tuple[SeedEntry, ...]
    unique_words: frozenset[bytes]
    enumerated_nodes: int
    truncated: bool


def _unsigned32(value: int) -> int:
    return value & 0xFFFFFFFF


def compile_required_gain(
    parameters: SSVSeedParameters,
    target_length: int,
    tjb: int,
    cutoff_decision: Callable[[int, int, int, float, int, float], int],
    *,
    status_ok: int,
    cpu_required: int,
) -> int | None:
    """Return the first local gain that exact SSV cannot definitely reject.

    ``None`` means that the pair must be sent to exact SSV unconditionally.
    The search covers the complete pre-wrap gain domain, replays HMMER's
    unsigned postprocessing, and refuses a nonmonotone decision sequence.
    """

    byte_values = (
        parameters.tbm,
        parameters.tec,
        parameters.base,
        parameters.bias,
        tjb,
    )
    if (
        target_length <= 0
        or target_length > 100_000
        or any(not 0 <= value <= 255 for value in byte_values)
        or tjb + parameters.tbm + parameters.tec + parameters.bias >= 127
    ):
        return None

    decisions: list[bool] = []
    for gain in range(128):
        raw_xe = 128 + gain
        if raw_xe >= 255 - parameters.bias:
            decisions.append(True)
            continue

        adjusted = _unsigned32(
            raw_xe + parameters.base - tjb - parameters.tbm - 128
        )
        if adjusted >= 255 - parameters.bias:
            decisions.append(True)
            continue

        xj = _unsigned32(adjusted - parameters.tec)
        if xj > parameters.base:
            decisions.append(True)
            continue

        numerator = xj - tjb - parameters.base
        action = cutoff_decision(
            status_ok,
            numerator,
            target_length,
            parameters.scale,
            parameters.cutoff_mode,
            parameters.cutoff_bit_score,
        )
        decisions.append(action == cpu_required)

    try:
        required = decisions.index(True)
    except ValueError:
        # Gain 127 always reaches the raw overflow guard for a valid bias.
        # Treat a violated invariant as unsupported rather than certifying.
        return None
    if any(not decision for decision in decisions[required:]):
        return None
    return required if required > 0 else None


def partition_model(model_length: int, block_count: int) -> tuple[tuple[int, int], ...]:
    """Partition model positions into nonempty, contiguous blocks."""

    if model_length <= 0:
        raise ValueError("model length must be positive")
    if block_count <= 0:
        raise ValueError("block count must be positive")
    count = min(model_length, block_count)
    return tuple(
        (index * model_length // count, (index + 1) * model_length // count)
        for index in range(count)
    )


def equal_integer_quotas(required_gain: int, block_count: int) -> tuple[int, ...]:
    """Allocate positive quotas with ``sum(quota - 1) == gain - 1``."""

    if required_gain <= 0:
        raise ValueError("required gain must be positive")
    if block_count <= 0:
        raise ValueError("block count must be positive")
    share, remainder = divmod(required_gain - 1, block_count)
    quotas = tuple(
        1 + share + (index < remainder) for index in range(block_count)
    )
    assert sum(value - 1 for value in quotas) == required_gain - 1
    return quotas


def window_seed_threshold(
    required_gain: int,
    model_length: int,
    target_length: int,
    maximum_word_length: int,
) -> int:
    """Return a sound threshold for any word of at most the given length.

    A diagonal segment contains at most ``ceil(min(M,L)/K)`` consecutive
    chunks of length at most ``K``.  If their integer total reaches
    ``required_gain``, one chunk must reach the returned threshold.
    """

    if required_gain <= 0:
        raise ValueError("required gain must be positive")
    if model_length <= 0 or target_length <= 0 or maximum_word_length <= 0:
        raise ValueError("model, target, and word lengths must be positive")
    maximum_alignment = min(model_length, target_length)
    chunks = (maximum_alignment + maximum_word_length - 1) // maximum_word_length
    return (required_gain + chunks - 1) // chunks


def _signed_byte(raw: int) -> int:
    return raw if raw < 128 else raw - 256


def score_gains(
    packed_scores: bytes | bytearray | memoryview,
    model_length: int,
    score_stride: int,
    alphabet_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Decode compact HMMER signed costs as additive SSV gains."""

    if model_length <= 0 or score_stride < alphabet_size or alphabet_size <= 0:
        raise ValueError("invalid compact score dimensions")
    view = memoryview(packed_scores).cast("B")
    required = model_length * score_stride
    if len(view) < required:
        raise ValueError("compact score table is truncated")
    return tuple(
        tuple(-_signed_byte(view[position * score_stride + residue])
              for residue in range(alphabet_size))
        for position in range(model_length)
    )


def enumerate_mandatory_seeds(
    gains: Sequence[Sequence[int]],
    required_gain: int,
    *,
    block_count: int = 4,
    alphabet_size: int = 20,
    association_limit: int = 1_000_000,
    node_limit: int = 10_000_000,
) -> MandatorySeedPlan:
    """Enumerate every minimal quota-crossing word, or fail open on a cap.

    A truncated plan is not a rejection certificate: callers must route that
    profile/length cohort through exact SSV.
    """

    model_length = len(gains)
    if model_length <= 0:
        raise ValueError("gain table is empty")
    if alphabet_size <= 0 or any(len(row) < alphabet_size for row in gains):
        raise ValueError("gain table has inconsistent alphabet rows")
    if association_limit <= 0 or node_limit <= 0:
        raise ValueError("enumeration limits must be positive")

    blocks = partition_model(model_length, block_count)
    quotas = equal_integer_quotas(required_gain, len(blocks))
    entries: list[SeedEntry] = []
    unique_words: set[bytes] = set()
    nodes = 0
    truncated = False

    for block_index, ((block_start, block_stop), quota) in enumerate(
        zip(blocks, quotas, strict=True)
    ):
        maximum = [max(row[:alphabet_size]) for row in gains]
        # Maximum additional score of any prefix beginning at ``position``.
        suffix_prefix_upper = [0] * (block_stop + 1)
        for position in range(block_stop - 1, block_start - 1, -1):
            suffix_prefix_upper[position] = max(
                0, maximum[position] + suffix_prefix_upper[position + 1]
            )

        for model_start in range(block_start, block_stop):
            if suffix_prefix_upper[model_start] < quota:
                continue
            stack: list[tuple[int, int, bytes]] = [(model_start, 0, b"")]
            while stack:
                position, score, word = stack.pop()
                if nodes >= node_limit or len(entries) >= association_limit:
                    truncated = True
                    break
                if position >= block_stop:
                    continue
                for residue in range(alphabet_size - 1, -1, -1):
                    nodes += 1
                    next_score = score + gains[position][residue]
                    next_word = word + bytes((residue,))
                    if next_score >= quota:
                        entry = SeedEntry(
                            next_word,
                            block_index,
                            model_start,
                            position + 1,
                        )
                        entries.append(entry)
                        unique_words.add(next_word)
                        if len(entries) >= association_limit:
                            truncated = True
                            break
                    elif (
                        position + 1 < block_stop
                        and next_score + suffix_prefix_upper[position + 1]
                        >= quota
                    ):
                        stack.append((position + 1, next_score, next_word))
                if truncated:
                    break
            if truncated:
                break
        if truncated:
            break

    return MandatorySeedPlan(
        required_gain=required_gain,
        blocks=blocks,
        quotas=quotas,
        entries=tuple(entries),
        unique_words=frozenset(unique_words),
        enumerated_nodes=nodes,
        truncated=truncated,
    )


def enumerate_window_seeds(
    gains: Sequence[Sequence[int]],
    threshold: int,
    maximum_word_length: int,
    *,
    alphabet_size: int = 20,
    association_limit: int = 1_000_000,
    node_limit: int = 10_000_000,
) -> MandatorySeedPlan:
    """Enumerate minimal threshold-crossing words of bounded length.

    The returned plan uses one logical block and records ``threshold`` as its
    gain.  It becomes a full F1 certificate only when ``threshold`` came from
    :func:`window_seed_threshold` for the corresponding cohort.
    """

    if maximum_word_length <= 0:
        raise ValueError("maximum word length must be positive")
    model_length = len(gains)
    if model_length <= 0:
        raise ValueError("gain table is empty")
    if alphabet_size <= 0 or any(len(row) < alphabet_size for row in gains):
        raise ValueError("gain table has inconsistent alphabet rows")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    entries: list[SeedEntry] = []
    unique_words: set[bytes] = set()
    nodes = 0
    truncated = False
    maximum = [max(row[:alphabet_size]) for row in gains]

    for model_start in range(model_length):
        model_stop = min(model_length, model_start + maximum_word_length)
        suffix_prefix_upper = [0] * (model_stop + 1)
        for position in range(model_stop - 1, model_start - 1, -1):
            suffix_prefix_upper[position] = max(
                0, maximum[position] + suffix_prefix_upper[position + 1]
            )
        if suffix_prefix_upper[model_start] < threshold:
            continue
        stack: list[tuple[int, int, bytes]] = [(model_start, 0, b"")]
        while stack:
            position, score, word = stack.pop()
            if nodes >= node_limit or len(entries) >= association_limit:
                truncated = True
                break
            if position >= model_stop:
                continue
            for residue in range(alphabet_size - 1, -1, -1):
                nodes += 1
                next_score = score + gains[position][residue]
                next_word = word + bytes((residue,))
                if next_score >= threshold:
                    entries.append(
                        SeedEntry(next_word, 0, model_start, position + 1)
                    )
                    unique_words.add(next_word)
                    if len(entries) >= association_limit:
                        truncated = True
                        break
                elif (
                    position + 1 < model_stop
                    and next_score + suffix_prefix_upper[position + 1]
                    >= threshold
                ):
                    stack.append((position + 1, next_score, next_word))
            if truncated:
                break
        if truncated:
            break

    return MandatorySeedPlan(
        required_gain=threshold,
        blocks=((0, model_length),),
        quotas=(threshold,),
        entries=tuple(entries),
        unique_words=frozenset(unique_words),
        enumerated_nodes=nodes,
        truncated=truncated,
    )


def maximum_diagonal_gain(
    gains: Sequence[Sequence[int]], sequence: Sequence[int]
) -> int:
    """Return the exact unclipped local gain underlying ordinary SSV."""

    model_length = len(gains)
    target_length = len(sequence)
    maximum = 0
    for diagonal in range(model_length + target_length - 1):
        delta = diagonal - (target_length - 1)
        target = -delta if delta < 0 else 0
        model = target + delta
        running = 0
        while target < target_length and model < model_length:
            residue = sequence[target]
            if residue < 0 or residue >= len(gains[model]):
                raise ValueError("sequence residue is outside the gain alphabet")
            running = max(0, running + gains[model][residue])
            maximum = max(maximum, running)
            target += 1
            model += 1
    return maximum


def maximum_bounded_diagonal_gain(
    gains: Sequence[Sequence[int]],
    sequence: Sequence[int],
    maximum_word_length: int,
) -> int:
    """Return the largest diagonal subarray sum with length at most ``K``."""

    if maximum_word_length <= 0:
        raise ValueError("maximum word length must be positive")
    model_length = len(gains)
    target_length = len(sequence)
    maximum = 0
    for diagonal in range(model_length + target_length - 1):
        delta = diagonal - (target_length - 1)
        target = -delta if delta < 0 else 0
        model = target + delta
        values: list[int] = []
        while target < target_length and model < model_length:
            residue = sequence[target]
            if residue < 0 or residue >= len(gains[model]):
                raise ValueError("sequence residue is outside the gain alphabet")
            values.append(gains[model][residue])
            target += 1
            model += 1
        for start in range(len(values)):
            score = 0
            for value in values[start : start + maximum_word_length]:
                score += value
                maximum = max(maximum, score)
    return maximum


def has_mandatory_seed(sequence: Sequence[int], plan: MandatorySeedPlan) -> bool:
    """Return whether a canonical target contains a word in ``plan``."""

    if plan.truncated:
        return True
    target = bytes(sequence)
    return any(target.find(word) >= 0 for word in plan.unique_words)


def seed_false_rejects(
    gains: Sequence[Sequence[int]],
    sequences: Iterable[Sequence[int]],
    plan: MandatorySeedPlan,
) -> tuple[int, int]:
    """Return ``(required_pairs, missed_pairs)`` for an independent oracle."""

    required = 0
    missed = 0
    for sequence in sequences:
        reaches = maximum_diagonal_gain(gains, sequence) >= plan.required_gain
        if reaches:
            required += 1
            missed += not has_mandatory_seed(sequence, plan)
    return required, missed
