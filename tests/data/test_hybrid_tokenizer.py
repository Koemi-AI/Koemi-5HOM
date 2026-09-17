from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.data.contracts import DatasetRecord, OUTPUT_TAG, RESERVED_TAGS
from koemi.data.hybrid_tokenizer import (
    FIRST_MERGE_ID,
    FIRST_RESERVED_ID,
    MINIMUM_VOCABULARY_SIZE,
    PRETOKEN_PATTERN,
    HybridTokenizer,
    HybridVocabulary,
    RESERVED_TOKEN_IDS,
    train_hybrid_vocabulary,
)
from koemi.data.serialization import serialize_record, serialize_record_tokens
from koemi.data.tokenizer import ByteTokenizer
from koemi.training.dataset import IGNORE_TARGET_ID, CausalByteDataset


CORPUS = (
    "def queue_push(queue, value):\n    queue.append(value)\n",
    "def queue_pop(queue):\n    return queue.pop(0)\n",
    "The queue preserves arrival order, so the queue is first in first out.\n",
    "A queue is first in first out and a stack is last in first out.\n",
) * 6


def build_vocabulary(vocabulary_size: int = MINIMUM_VOCABULARY_SIZE + 96) -> HybridVocabulary:
    return train_hybrid_vocabulary(CORPUS, vocabulary_size, minimum_frequency=2)


def ordered_pieces(text: str) -> list[str]:
    pieces: list[str] = []
    position = 0
    for match in PRETOKEN_PATTERN.finditer(text):
        if match.start() > position:
            pieces.append(text[position : match.start()])
        pieces.append(match.group())
        position = match.end()
    if position < len(text):
        pieces.append(text[position:])
    return pieces


class VocabularyLayoutTests(unittest.TestCase):
    def test_the_byte_range_and_padding_keep_their_ids(self) -> None:
        vocabulary = HybridVocabulary(())
        self.assertEqual(MINIMUM_VOCABULARY_SIZE, vocabulary.vocabulary_size)
        self.assertEqual(b"\x00", vocabulary.token_bytes(0))
        self.assertEqual(b"\xff", vocabulary.token_bytes(255))
        self.assertEqual(b"", vocabulary.token_bytes(PAD_TOKEN_ID))
        self.assertEqual(OUTPUT_TAG.encode("utf-8"), vocabulary.token_bytes(RESERVED_TOKEN_IDS[OUTPUT_TAG]))

    def test_a_merge_expands_to_the_bytes_it_joins(self) -> None:
        vocabulary = HybridVocabulary(((104, 105),))
        self.assertEqual(b"hi", vocabulary.token_bytes(FIRST_MERGE_ID))
        self.assertEqual(MINIMUM_VOCABULARY_SIZE + 1, vocabulary.vocabulary_size)

    def test_a_merge_over_padding_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            HybridVocabulary(((PAD_TOKEN_ID, 65),))

    def test_a_merge_over_a_later_id_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            HybridVocabulary(((65, FIRST_MERGE_ID + 5),))

    def test_a_vocabulary_round_trips_through_a_file(self) -> None:
        vocabulary = build_vocabulary()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.json"
            vocabulary.save(path)
            restored = HybridVocabulary.load(path)
        self.assertEqual(vocabulary.merges, restored.merges)

    def test_a_foreign_payload_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            HybridVocabulary.from_payload({"format_version": 1, "kind": "other", "merges": []})
        with self.assertRaises(ValueError):
            HybridVocabulary.from_payload({"format_version": 99, "kind": "koemi-hybrid-bpe", "merges": []})


class TrainingTests(unittest.TestCase):
    def test_a_frequent_word_collapses_into_fewer_tokens(self) -> None:
        tokenizer = HybridTokenizer(build_vocabulary())
        self.assertLess(len(tokenizer.encode("queue")), len("queue".encode("utf-8")))
        self.assertLess(len(tokenizer.encode(" queue")), len(" queue".encode("utf-8")))

    def test_the_learned_size_never_exceeds_the_budget(self) -> None:
        vocabulary = build_vocabulary(MINIMUM_VOCABULARY_SIZE + 40)
        self.assertLessEqual(vocabulary.vocabulary_size, MINIMUM_VOCABULARY_SIZE + 40)

    def test_training_is_deterministic(self) -> None:
        self.assertEqual(build_vocabulary().merges, build_vocabulary().merges)

    def test_a_budget_below_the_reserved_range_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            train_hybrid_vocabulary(CORPUS, MINIMUM_VOCABULARY_SIZE)

    def test_an_empty_corpus_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            train_hybrid_vocabulary((), MINIMUM_VOCABULARY_SIZE + 10)

    def test_the_pretokenizer_covers_every_character(self) -> None:
        text = "snake_case CamelCase 1234 -> {'key': 'value'} # comment\n\tTabbed\n"
        pieces = ordered_pieces(text)
        self.assertEqual(text, "".join(pieces))
        self.assertNotIn("", pieces)
        self.assertIn("snake_case", pieces)


class EncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HybridTokenizer(build_vocabulary())

    def assert_round_trip(self, text: str) -> None:
        self.assertEqual(text, self.tokenizer.decode(self.tokenizer.encode(text)))

    def test_ascii_round_trips(self) -> None:
        self.assert_round_trip("def queue_pop(queue):\n    return queue.pop(0)\n")

    def test_text_outside_the_corpus_round_trips(self) -> None:
        self.assert_round_trip("Um pequeno texto em portugues com acentuacao: ção, ãõ, é.")

    def test_emoji_and_wide_characters_round_trip(self) -> None:
        self.assert_round_trip("emoji 🙂🚀 kanji 日本語 math ∑∫≈")

    def test_control_and_whitespace_round_trip(self) -> None:
        self.assert_round_trip("a\r\n\tb\x00c  \n\n  d")

    def test_arbitrary_bytes_round_trip(self) -> None:
        payload = bytes(range(256))
        self.assertEqual(payload, self.tokenizer.decode_bytes(self.tokenizer.encode_bytes(payload)))

    def test_an_unknown_character_falls_back_to_its_bytes(self) -> None:
        token_ids = self.tokenizer.encode("𝒬")
        self.assertEqual("𝒬".encode("utf-8"), self.tokenizer.decode_bytes(token_ids))
        self.assertTrue(all(token_id < 256 for token_id in token_ids))

    def test_a_reserved_marker_becomes_one_token(self) -> None:
        for tag in RESERVED_TAGS:
            self.assertEqual([RESERVED_TOKEN_IDS[tag]], self.tokenizer.encode(tag))

    def test_a_marker_never_merges_with_its_neighbours(self) -> None:
        token_ids = self.tokenizer.encode(f"queue{OUTPUT_TAG}queue")
        marker_position = token_ids.index(RESERVED_TOKEN_IDS[OUTPUT_TAG])
        self.assertEqual("queue", self.tokenizer.decode(token_ids[:marker_position]))
        self.assertEqual("queue", self.tokenizer.decode(token_ids[marker_position + 1 :]))

    def test_every_id_stays_inside_the_head(self) -> None:
        token_ids = self.tokenizer.encode("def queue_push(queue, value):")
        self.assertTrue(all(0 <= token_id < self.tokenizer.vocabulary_size for token_id in token_ids))
        self.assertNotIn(PAD_TOKEN_ID, token_ids)

    def test_padding_decodes_to_nothing(self) -> None:
        self.assertEqual("ab", self.tokenizer.decode([97, PAD_TOKEN_ID, 98]))

    def test_an_id_above_the_vocabulary_is_skipped(self) -> None:
        self.assertEqual("a", self.tokenizer.decode([97, self.tokenizer.vocabulary_size + 5]))


class SerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = HybridTokenizer(build_vocabulary())
        self.record = DatasetRecord(
            identifier="queue-001",
            input_text="Explain FIFO in one sentence.",
            thinking_text="A queue preserves arrival order.",
            output_text="FIFO means first in, first out.",
            metadata={},
            system_text="Answer in one sentence.",
        )

    def test_the_byte_tokenizer_reproduces_the_byte_serialization(self) -> None:
        byte_record = serialize_record(self.record)
        token_record = serialize_record_tokens(self.record, ByteTokenizer())
        self.assertEqual(tuple(byte_record.token_bytes), token_record.token_ids)
        self.assertEqual(byte_record.supervised_positions, token_record.supervised_positions)
        self.assertEqual(byte_record.thinking_positions, token_record.thinking_positions)

    def test_the_masks_stay_aligned_with_the_hybrid_ids(self) -> None:
        serialized = serialize_record_tokens(self.record, self.tokenizer)
        self.assertEqual(len(serialized.token_ids), len(serialized.supervised_positions))
        self.assertEqual(len(serialized.token_ids), len(serialized.thinking_positions))
        thinking_ids = [
            token_id
            for token_id, is_thinking in zip(serialized.token_ids, serialized.thinking_positions)
            if is_thinking
        ]
        self.assertEqual(self.record.thinking_text, self.tokenizer.decode(thinking_ids))
        answer_ids = [
            token_id
            for token_id, is_supervised, is_thinking in zip(
                serialized.token_ids,
                serialized.supervised_positions,
                serialized.thinking_positions,
            )
            if is_supervised and not is_thinking
        ]
        self.assertEqual(self.record.output_text, self.tokenizer.decode(answer_ids))

    def test_the_whole_record_decodes_back_to_its_text(self) -> None:
        serialized = serialize_record_tokens(self.record, self.tokenizer)
        self.assertEqual(
            serialize_record(self.record).token_bytes,
            self.tokenizer.decode_bytes(serialized.token_ids),
        )

    def test_a_hybrid_dataset_produces_shorter_chunks(self) -> None:
        byte_dataset = CausalByteDataset((self.record,), 64)
        hybrid_dataset = CausalByteDataset((self.record,), 64, self.tokenizer)
        byte_tokens = sum(len(chunk.input_ids) for chunk in byte_dataset.chunks)
        hybrid_tokens = sum(len(chunk.input_ids) for chunk in hybrid_dataset.chunks)
        self.assertLess(hybrid_tokens, byte_tokens)
        self.assertTrue(
            any(
                target_id != IGNORE_TARGET_ID
                for chunk in hybrid_dataset.chunks
                for target_id in chunk.target_ids
            )
        )

    def test_a_plain_text_record_keeps_its_supervision(self) -> None:
        record = DatasetRecord(
            identifier="text-001",
            input_text="queue queue queue",
            thinking_text=None,
            output_text=None,
            metadata={},
        )
        serialized = serialize_record_tokens(record, self.tokenizer)
        self.assertTrue(all(serialized.supervised_positions))
        self.assertEqual(record.input_text, self.tokenizer.decode(serialized.token_ids))


if __name__ == "__main__":
    unittest.main()
