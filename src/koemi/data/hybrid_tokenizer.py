from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from koemi.configuration.settings import BYTE_VOCABULARY_SIZE, PAD_TOKEN_ID
from koemi.data.contracts import RESERVED_TAGS


FIRST_RESERVED_ID = PAD_TOKEN_ID + 1
FIRST_MERGE_ID = FIRST_RESERVED_ID + len(RESERVED_TAGS)
MINIMUM_VOCABULARY_SIZE = FIRST_MERGE_ID
VOCABULARY_FORMAT_VERSION = 1
VOCABULARY_KIND = "koemi-hybrid-bpe"

PRETOKEN_PATTERN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")

RESERVED_TOKEN_IDS = {tag: FIRST_RESERVED_ID + index for index, tag in enumerate(RESERVED_TAGS)}
_RESERVED_SPLIT_PATTERN = re.compile("|".join(re.escape(tag) for tag in RESERVED_TAGS))


@dataclass(frozen=True)
class HybridVocabulary:
    """Byte-level BPE vocabulary with reserved span markers and a byte fallback.

    Ids 0-255 are single bytes, id 256 is padding, the next four ids are the
    reserved span markers in the order of `RESERVED_TAGS`, and every id above
    them is one learned merge. Because the alphabet is the full byte range, any
    input encodes, and an unknown character degrades to its own bytes instead of
    to an unknown token.
    """

    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        token_bytes: list[bytes] = [bytes((value,)) for value in range(BYTE_VOCABULARY_SIZE)]
        token_bytes.append(b"")
        token_bytes.extend(tag.encode("utf-8") for tag in RESERVED_TAGS)
        for rank, pair in enumerate(self.merges):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("every merge must be a pair of token ids")
            merge_id = FIRST_MERGE_ID + rank
            for side in pair:
                if isinstance(side, bool) or not isinstance(side, int):
                    raise ValueError("merge token ids must be integers")
                if not (0 <= side < BYTE_VOCABULARY_SIZE or FIRST_MERGE_ID <= side < merge_id):
                    raise ValueError("a merge may only combine byte ids and earlier merges")
            token_bytes.append(token_bytes[pair[0]] + token_bytes[pair[1]])
        object.__setattr__(self, "_token_bytes", tuple(token_bytes))
        object.__setattr__(
            self,
            "_ranks",
            {pair: rank for rank, pair in enumerate(self.merges)},
        )

    @property
    def vocabulary_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    @property
    def merge_ranks(self) -> dict[tuple[int, int], int]:
        return dict(getattr(self, "_ranks"))

    def token_bytes(self, token_id: int) -> bytes:
        """Return the byte string one id expands to; padding expands to nothing."""
        table: tuple[bytes, ...] = getattr(self, "_token_bytes")
        if token_id < 0 or token_id >= len(table):
            raise ValueError("token id is outside this vocabulary")
        return table[token_id]

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": VOCABULARY_FORMAT_VERSION,
            "kind": VOCABULARY_KIND,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> HybridVocabulary:
        if not isinstance(payload, dict):
            raise ValueError("vocabulary payload must be a dictionary")
        if payload.get("format_version") != VOCABULARY_FORMAT_VERSION:
            raise ValueError("vocabulary format version is not supported")
        if payload.get("kind") != VOCABULARY_KIND:
            raise ValueError("vocabulary payload is not a Koemi hybrid vocabulary")
        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("vocabulary merges are invalid")
        merges: list[tuple[int, int]] = []
        for raw_merge in raw_merges:
            if not isinstance(raw_merge, (list, tuple)) or len(raw_merge) != 2:
                raise ValueError("every merge must be a pair of token ids")
            merges.append((int(raw_merge[0]), int(raw_merge[1])))
        return cls(tuple(merges))

    def save(self, path: str | Path) -> Path:
        target_path = Path(path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return target_path

    @classmethod
    def load(cls, path: str | Path) -> HybridVocabulary:
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"vocabulary file does not exist: {source_path}")
        return cls.from_payload(json.loads(source_path.read_text(encoding="utf-8")))


class HybridTokenizer:
    """Word and subword tokens with an exact byte fallback.

    Encoding splits the text on the reserved span markers, applies the
    pretokenizer to what is left and reduces every piece with the learned merges.
    `decode(encode(text)) == text` for any text, because a piece that no merge
    covers stays as its own bytes.
    """

    pad_token_id = PAD_TOKEN_ID

    def __init__(self, vocabulary: HybridVocabulary) -> None:
        self.vocabulary = vocabulary
        self.vocabulary_size = vocabulary.vocabulary_size
        self._ranks = vocabulary.merge_ranks
        self._piece_cache: dict[bytes, tuple[int, ...]] = {}

    def encode(self, text: str) -> list[int]:
        """Return the token ids for `text`, keeping reserved markers atomic."""
        token_ids: list[int] = []
        position = 0
        for match in _RESERVED_SPLIT_PATTERN.finditer(text):
            token_ids.extend(self._encode_plain(text[position : match.start()]))
            token_ids.append(RESERVED_TOKEN_IDS[match.group()])
            position = match.end()
        token_ids.extend(self._encode_plain(text[position:]))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Return the text for `token_ids`, replacing byte sequences that are not UTF-8."""
        return self.decode_bytes(token_ids).decode("utf-8", errors="replace")

    def decode_bytes(self, token_ids: Iterable[int]) -> bytes:
        """Return the raw bytes for `token_ids`, skipping ids outside the vocabulary."""
        pieces: list[bytes] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocabulary_size:
                continue
            pieces.append(self.vocabulary.token_bytes(token_id))
        return b"".join(pieces)

    def encode_bytes(self, data: bytes) -> list[int]:
        """Return the token ids for raw bytes, valid UTF-8 or not."""
        return self.encode(data.decode("utf-8", errors="surrogateescape"))

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        token_ids: list[int] = []
        position = 0
        for match in PRETOKEN_PATTERN.finditer(text):
            if match.start() > position:
                token_ids.extend(self._encode_piece(_piece_bytes(text[position : match.start()])))
            token_ids.extend(self._encode_piece(_piece_bytes(match.group())))
            position = match.end()
        if position < len(text):
            token_ids.extend(self._encode_piece(_piece_bytes(text[position:])))
        return token_ids

    def _encode_piece(self, piece: bytes) -> tuple[int, ...]:
        cached = self._piece_cache.get(piece)
        if cached is not None:
            return cached
        symbols = list(piece)
        encoded = tuple(apply_merges(symbols, self._ranks))
        self._piece_cache[piece] = encoded
        return encoded


def _piece_bytes(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")


def apply_merges(symbols: list[int], ranks: dict[tuple[int, int], int]) -> list[int]:
    """Reduce a byte id sequence by repeatedly applying the lowest-ranked merge."""
    while len(symbols) >= 2:
        best_rank: int | None = None
        best_index = -1
        for index in range(len(symbols) - 1):
            rank = ranks.get((symbols[index], symbols[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_index = index
        if best_rank is None:
            break
        symbols[best_index : best_index + 2] = [FIRST_MERGE_ID + best_rank]
    return symbols


def count_pretokens(texts: Iterable[str]) -> dict[bytes, int]:
    """Count pretokenizer pieces over a corpus, ignoring reserved span markers."""
    counts: dict[bytes, int] = {}
    for text in texts:
        for part in _RESERVED_SPLIT_PATTERN.split(text):
            position = 0
            for match in PRETOKEN_PATTERN.finditer(part):
                if match.start() > position:
                    gap = _piece_bytes(part[position : match.start()])
                    counts[gap] = counts.get(gap, 0) + 1
                piece = _piece_bytes(match.group())
                counts[piece] = counts.get(piece, 0) + 1
                position = match.end()
            if position < len(part):
                tail = _piece_bytes(part[position:])
                counts[tail] = counts.get(tail, 0) + 1
    return counts


def train_hybrid_vocabulary(
    texts: Iterable[str],
    vocabulary_size: int,
    *,
    minimum_frequency: int = 2,
) -> HybridVocabulary:
    """Learn byte-pair merges from a corpus and return the resulting vocabulary.

    `vocabulary_size` counts every id the model head emits, so the byte range,
    the padding id and the four reserved markers are already included. Learning
    stops early when no pair reaches `minimum_frequency`, which keeps a small
    corpus from producing merges that only one document justifies.
    """
    if vocabulary_size <= MINIMUM_VOCABULARY_SIZE:
        raise ValueError(f"vocabulary_size must exceed {MINIMUM_VOCABULARY_SIZE}")
    if minimum_frequency < 1:
        raise ValueError("minimum_frequency must be at least 1")
    merge_budget = vocabulary_size - FIRST_MERGE_ID
    piece_counts = count_pretokens(texts)
    if not piece_counts:
        raise ValueError("the corpus does not contain a single pretokenizer piece")
    words = [list(piece) for piece in piece_counts]
    counts = list(piece_counts.values())
    pair_counts: dict[tuple[int, int], int] = {}
    pair_words: dict[tuple[int, int], set[int]] = {}
    for word_index, symbols in enumerate(words):
        _add_word_pairs(symbols, counts[word_index], word_index, pair_counts, pair_words)

    merges: list[tuple[int, int]] = []
    while len(merges) < merge_budget and pair_counts:
        pair = max(pair_counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]
        if pair_counts[pair] < minimum_frequency:
            break
        merge_id = FIRST_MERGE_ID + len(merges)
        merges.append(pair)
        for word_index in sorted(pair_words.get(pair, ())):
            symbols = words[word_index]
            weight = counts[word_index]
            _remove_word_pairs(symbols, weight, word_index, pair_counts, pair_words)
            words[word_index] = _merge_symbols(symbols, pair, merge_id)
            _add_word_pairs(words[word_index], weight, word_index, pair_counts, pair_words)
        pair_counts.pop(pair, None)
        pair_words.pop(pair, None)
    return HybridVocabulary(tuple(merges))


def _merge_symbols(symbols: list[int], pair: tuple[int, int], merge_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(symbols):
        if (
            index + 1 < len(symbols)
            and symbols[index] == pair[0]
            and symbols[index + 1] == pair[1]
        ):
            merged.append(merge_id)
            index += 2
            continue
        merged.append(symbols[index])
        index += 1
    return merged


def _add_word_pairs(
    symbols: Sequence[int],
    weight: int,
    word_index: int,
    pair_counts: dict[tuple[int, int], int],
    pair_words: dict[tuple[int, int], set[int]],
) -> None:
    for left, right in zip(symbols, symbols[1:]):
        pair = (left, right)
        pair_counts[pair] = pair_counts.get(pair, 0) + weight
        pair_words.setdefault(pair, set()).add(word_index)


def _remove_word_pairs(
    symbols: Sequence[int],
    weight: int,
    word_index: int,
    pair_counts: dict[tuple[int, int], int],
    pair_words: dict[tuple[int, int], set[int]],
) -> None:
    for left, right in zip(symbols, symbols[1:]):
        pair = (left, right)
        remaining = pair_counts.get(pair, 0) - weight
        if remaining > 0:
            pair_counts[pair] = remaining
        else:
            pair_counts.pop(pair, None)
        words = pair_words.get(pair)
        if words is not None:
            words.discard(word_index)
            if not words:
                pair_words.pop(pair, None)


__all__ = [
    "FIRST_MERGE_ID",
    "FIRST_RESERVED_ID",
    "HybridTokenizer",
    "HybridVocabulary",
    "MINIMUM_VOCABULARY_SIZE",
    "PRETOKEN_PATTERN",
    "RESERVED_TOKEN_IDS",
    "VOCABULARY_FORMAT_VERSION",
    "apply_merges",
    "count_pretokens",
    "train_hybrid_vocabulary",
]
