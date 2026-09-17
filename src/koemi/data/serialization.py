from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from koemi.data.contracts import (
    INPUT_TAG,
    OUTPUT_TAG,
    SYSTEM_TAG,
    THINKING_TAG,
    DatasetRecord,
    reject_reserved_tags,
)
from koemi.data.tokenizer import TextTokenizer


SYSTEM_MARKER = f"{SYSTEM_TAG}\n"
INPUT_MARKER = f"{INPUT_TAG}\n"
THINKING_MARKER = f"\n{THINKING_TAG}\n"
OUTPUT_MARKER = f"\n{OUTPUT_TAG}\n"

ANSWER_TARGET = "answer"
THINKING_TARGET = "thinking"


@dataclass(frozen=True)
class RecordSegment:
    """One span of a serialized record with the supervision it carries.

    Segments are the unit a tokenizer is allowed to see. Encoding them one by one
    is what keeps a multi-byte token from straddling a span marker and shifting
    the supervised mask.
    """

    text: str
    supervised: bool
    thinking: bool


@dataclass(frozen=True)
class SerializedRecord:
    token_bytes: bytes
    supervised_positions: tuple[bool, ...]
    thinking_positions: tuple[bool, ...]


@dataclass(frozen=True)
class SerializedTokenRecord:
    token_ids: tuple[int, ...]
    supervised_positions: tuple[bool, ...]
    thinking_positions: tuple[bool, ...]


def reject_prompt_tags(system_text: str | None, user_text: str) -> None:
    reject_reserved_tags("system", system_text)
    reject_reserved_tags("prompt", user_text)


def system_prefix(system_text: str | None) -> str:
    if system_text is None:
        return ""
    return f"{SYSTEM_MARKER}{system_text}\n"


def prompt_segments(system_text: str | None, user_text: str, target: str) -> tuple[str, ...]:
    """Return the prompt spans in order, so a tokenizer can encode them separately."""
    if target not in {ANSWER_TARGET, THINKING_TARGET}:
        raise ValueError("prompt target must be answer or thinking")
    reject_prompt_tags(system_text, user_text)
    closing_marker = OUTPUT_MARKER if target == ANSWER_TARGET else THINKING_MARKER
    return (system_prefix(system_text), INPUT_MARKER, user_text, closing_marker)


def build_answer_prompt(system_text: str | None, user_text: str) -> str:
    return "".join(prompt_segments(system_text, user_text, ANSWER_TARGET))


def build_thinking_prompt(system_text: str | None, user_text: str) -> str:
    return "".join(prompt_segments(system_text, user_text, THINKING_TARGET))


def encode_segments(tokenizer: TextTokenizer, segments: Sequence[str]) -> list[int]:
    """Encode each span on its own so no token crosses a span boundary."""
    token_ids: list[int] = []
    for segment in segments:
        token_ids.extend(tokenizer.encode(segment))
    return token_ids


def encode_prompt(
    tokenizer: TextTokenizer,
    system_text: str | None,
    user_text: str,
    target: str = ANSWER_TARGET,
) -> list[int]:
    """Encode an inference prompt exactly the way training serialized the same spans."""
    return encode_segments(tokenizer, prompt_segments(system_text, user_text, target))


def strip_prompt(generated_text: str, prompt: str) -> str:
    if not generated_text.startswith(prompt):
        raise ValueError("the generated text does not start with the prompt it was conditioned on")
    return generated_text[len(prompt) :]


def record_segments(record: DatasetRecord) -> tuple[RecordSegment, ...]:
    """Return the ordered spans of a record with their supervision flags."""
    prefix = system_prefix(record.system_text)
    if record.output_text is None:
        return (
            RecordSegment(prefix, False, False),
            RecordSegment(record.input_text, True, False),
        )
    segments = [
        RecordSegment(prefix, False, False),
        RecordSegment(INPUT_MARKER, False, False),
        RecordSegment(record.input_text, False, False),
    ]
    if record.thinking_text is not None:
        segments.append(RecordSegment(THINKING_MARKER, False, False))
        segments.append(RecordSegment(record.thinking_text, True, True))
    segments.append(RecordSegment(OUTPUT_MARKER, False, False))
    segments.append(RecordSegment(record.output_text, True, False))
    return tuple(segments)


def serialize_record(record: DatasetRecord) -> SerializedRecord:
    segments = record_segments(record)
    encoded = tuple((segment, segment.text.encode("utf-8")) for segment in segments)
    return SerializedRecord(
        b"".join(payload for _, payload in encoded),
        tuple(segment.supervised for segment, payload in encoded for _ in payload),
        tuple(segment.thinking for segment, payload in encoded for _ in payload),
    )


def serialize_record_tokens(
    record: DatasetRecord,
    tokenizer: TextTokenizer,
) -> SerializedTokenRecord:
    """Serialize a record into tokenizer ids with masks aligned to those ids."""
    token_ids: list[int] = []
    supervised_positions: list[bool] = []
    thinking_positions: list[bool] = []
    for segment in record_segments(record):
        segment_ids = tokenizer.encode(segment.text)
        token_ids.extend(segment_ids)
        supervised_positions.extend(segment.supervised for _ in segment_ids)
        thinking_positions.extend(segment.thinking for _ in segment_ids)
    return SerializedTokenRecord(
        tuple(token_ids),
        tuple(supervised_positions),
        tuple(thinking_positions),
    )


def supervised_prefix_bytes(record: DatasetRecord) -> bytes:
    serialized = serialize_record(record)
    for position, is_supervised in enumerate(serialized.supervised_positions):
        if is_supervised:
            return serialized.token_bytes[:position]
    return serialized.token_bytes
