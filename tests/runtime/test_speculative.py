from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.runtime.fast_decode import SamplingPolicy, generate_batch
from koemi.runtime.speculative import (
    DraftProposal,
    ModelDrafter,
    NgramDrafter,
    accept_draft_tokens,
    speculative_generate,
)


VOCABULARY_SIZE = PAD_TOKEN_ID + 1


def build_model(**overrides) -> KoemiModel:
    settings = dict(
        embedding_size=16,
        memory_features=4,
        local_memory_size=4,
        salience_memory_size=3,
        expert_count=2,
        scan_chunk=8,
    )
    settings.update(overrides)
    model = KoemiModel(ModelSettings(**settings))
    model.eval()
    return model


def point_mass(token_ids: list[int]) -> DraftProposal:
    ids = torch.tensor(token_ids, dtype=torch.long)
    probabilities = torch.zeros((len(token_ids), VOCABULARY_SIZE))
    probabilities.scatter_(1, ids.unsqueeze(1), 1.0)
    return DraftProposal(ids, probabilities)


class ReplayDrafter:
    """Propose a fixed continuation, tracking how many tokens were committed."""

    def __init__(self, continuation: tuple[int, ...]) -> None:
        self.continuation = tuple(continuation)
        self.offset = 0

    def begin(self, context_ids) -> None:
        self.offset = 0

    def commit(self, token_ids) -> None:
        self.offset += len(tuple(token_ids))

    def propose(self, draft_length: int):
        candidate = self.continuation[self.offset : self.offset + draft_length]
        if not candidate:
            return None
        return point_mass(list(candidate))


def target_rows(rows: list[dict[int, float]]) -> torch.Tensor:
    probabilities = torch.zeros((len(rows), VOCABULARY_SIZE))
    for index, row in enumerate(rows):
        for token_id, mass in row.items():
            probabilities[index, token_id] = mass
    return probabilities


class AcceptanceTests(unittest.TestCase):
    def test_a_certain_target_accepts_every_proposed_token(self) -> None:
        proposal = point_mass([65, 66])
        probabilities = target_rows([{65: 1.0}, {66: 1.0}, {67: 1.0}])
        result = accept_draft_tokens(proposal, probabilities)
        self.assertEqual((65, 66, 67), result.token_ids)
        self.assertEqual(2, result.accepted_draft_tokens)
        self.assertTrue(result.used_bonus)

    def test_a_zero_mass_token_is_replaced_from_the_residual(self) -> None:
        proposal = point_mass([65, 66])
        probabilities = target_rows([{70: 1.0}, {66: 1.0}, {67: 1.0}])
        result = accept_draft_tokens(proposal, probabilities)
        self.assertEqual((70,), result.token_ids)
        self.assertEqual(0, result.accepted_draft_tokens)
        self.assertFalse(result.used_bonus)

    def test_a_rejection_in_the_middle_keeps_the_accepted_prefix(self) -> None:
        proposal = point_mass([65, 66, 67])
        probabilities = target_rows([{65: 1.0}, {90: 1.0}, {67: 1.0}, {68: 1.0}])
        result = accept_draft_tokens(proposal, probabilities)
        self.assertEqual((65, 90), result.token_ids)
        self.assertEqual(1, result.accepted_draft_tokens)
        self.assertFalse(result.used_bonus)

    def test_a_partial_target_mass_accepts_with_that_probability(self) -> None:
        proposal = point_mass([65])
        probabilities = target_rows([{65: 0.5, 66: 0.5}, {67: 1.0}])
        generator = torch.Generator().manual_seed(17)
        accepted = sum(
            1
            for _ in range(400)
            if accept_draft_tokens(proposal, probabilities, generator).accepted_draft_tokens == 1
        )
        self.assertGreater(accepted, 150)
        self.assertLess(accepted, 250)

    def test_a_residual_resample_stays_inside_the_target_support(self) -> None:
        proposal = point_mass([65])
        probabilities = target_rows([{65: 0.25, 66: 0.75}, {67: 1.0}])
        generator = torch.Generator().manual_seed(23)
        for _ in range(64):
            result = accept_draft_tokens(proposal, probabilities, generator)
            self.assertIn(result.token_ids[-1], {65, 66, 67})

    def test_a_mismatched_target_shape_is_refused(self) -> None:
        proposal = point_mass([65, 66])
        with self.assertRaises(ValueError):
            accept_draft_tokens(proposal, target_rows([{65: 1.0}, {66: 1.0}]))

    def test_an_empty_proposal_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            accept_draft_tokens(point_mass([]), target_rows([{65: 1.0}]))


class NgramDrafterTests(unittest.TestCase):
    def test_a_repeated_suffix_proposes_its_earlier_continuation(self) -> None:
        drafter = NgramDrafter(VOCABULARY_SIZE, minimum_order=2, maximum_order=4)
        drafter.begin([10, 11, 12, 13, 14, 10, 11])
        proposal = drafter.propose(3)
        self.assertIsNotNone(proposal)
        self.assertEqual([12, 13, 14], proposal.token_ids.tolist())
        self.assertEqual(1.0, float(proposal.probabilities[0, 12]))

    def test_an_unseen_suffix_proposes_nothing(self) -> None:
        drafter = NgramDrafter(VOCABULARY_SIZE, minimum_order=2, maximum_order=4)
        drafter.begin([10, 11, 12, 13])
        self.assertIsNone(drafter.propose(2))

    def test_committed_tokens_extend_the_searchable_context(self) -> None:
        drafter = NgramDrafter(VOCABULARY_SIZE, minimum_order=2, maximum_order=4)
        drafter.begin([10, 11, 12, 13])
        self.assertIsNone(drafter.propose(2))
        drafter.commit([14, 10, 11])
        proposal = drafter.propose(2)
        self.assertEqual([12, 13], proposal.token_ids.tolist())

    def test_a_short_context_proposes_nothing(self) -> None:
        drafter = NgramDrafter(VOCABULARY_SIZE, minimum_order=2, maximum_order=4)
        drafter.begin([10])
        self.assertIsNone(drafter.propose(2))


class SpeculativeGenerationTests(unittest.TestCase):
    def test_greedy_ngram_speculation_matches_the_greedy_baseline(self) -> None:
        torch.manual_seed(31)
        model = build_model()
        tokenizer = ByteTokenizer()
        prompt = "abab abab abab abab"
        baseline = generate_batch(
            model, tokenizer, (prompt,), 12, policy=SamplingPolicy(greedy=True)
        )
        speculative = speculative_generate(
            model,
            tokenizer,
            prompt,
            12,
            drafter=NgramDrafter(VOCABULARY_SIZE),
            policy=SamplingPolicy(greedy=True),
            draft_length=4,
        )
        self.assertEqual(baseline.token_ids[0], speculative.token_ids)

    def test_greedy_model_drafting_matches_the_greedy_baseline(self) -> None:
        torch.manual_seed(32)
        model = build_model()
        tokenizer = ByteTokenizer()
        prompt = "FIFO means"
        baseline = generate_batch(
            model, tokenizer, (prompt,), 10, policy=SamplingPolicy(greedy=True)
        )
        speculative = speculative_generate(
            model,
            tokenizer,
            prompt,
            10,
            drafter=ModelDrafter(model, policy=SamplingPolicy(greedy=True)),
            policy=SamplingPolicy(greedy=True),
            draft_length=3,
        )
        self.assertEqual(baseline.token_ids[0], speculative.token_ids)
        self.assertGreater(speculative.statistics.acceptance_rate, 0.5)

    def test_accepted_blocks_spend_fewer_forwards_than_committed_tokens(self) -> None:
        torch.manual_seed(33)
        model = build_model()
        tokenizer = ByteTokenizer()
        prompt = "FIFO means"
        baseline = generate_batch(
            model, tokenizer, (prompt,), 16, policy=SamplingPolicy(greedy=True)
        )
        speculative = speculative_generate(
            model,
            tokenizer,
            prompt,
            16,
            drafter=ReplayDrafter(baseline.token_ids[0]),
            policy=SamplingPolicy(greedy=True),
            draft_length=6,
        )
        self.assertEqual(baseline.token_ids[0], speculative.token_ids)
        self.assertEqual(16, speculative.statistics.committed_tokens)
        self.assertEqual(1.0, speculative.statistics.acceptance_rate)
        self.assertLess(speculative.statistics.target_forward_calls, 16)
        self.assertGreater(speculative.statistics.tokens_per_target_call, 2.0)

    def test_a_collapsed_acceptance_rate_is_reported(self) -> None:
        torch.manual_seed(36)
        model = build_model()
        tokenizer = ByteTokenizer()
        speculative = speculative_generate(
            model,
            tokenizer,
            "FIFO means",
            8,
            drafter=ReplayDrafter(tuple(range(200, 216))),
            policy=SamplingPolicy(greedy=True),
            draft_length=4,
        )
        self.assertEqual(8, speculative.statistics.committed_tokens)
        self.assertEqual(0.0, speculative.statistics.acceptance_rate)
        self.assertGreater(speculative.statistics.target_forward_calls, 8)

    def test_the_committed_sequence_agrees_with_a_fresh_forward(self) -> None:
        torch.manual_seed(34)
        model = build_model()
        tokenizer = ByteTokenizer()
        prompt = "queue queue queue"
        speculative = speculative_generate(
            model,
            tokenizer,
            prompt,
            8,
            drafter=NgramDrafter(VOCABULARY_SIZE),
            policy=SamplingPolicy(greedy=True),
            draft_length=4,
        )
        full_ids = list(tokenizer.encode(prompt)) + list(speculative.token_ids)
        with torch.no_grad():
            reference = model(torch.tensor([full_ids], dtype=torch.long))
        prompt_length = len(tokenizer.encode(prompt))
        for offset in range(len(speculative.token_ids) - 1):
            expected = int(reference.logits[0, prompt_length + offset - 1].argmax())
            self.assertEqual(expected, speculative.token_ids[offset])

    def test_a_model_drafter_rolls_its_state_back_after_a_proposal(self) -> None:
        torch.manual_seed(35)
        model = build_model()
        drafter = ModelDrafter(model, policy=SamplingPolicy(greedy=True))
        drafter.begin(ByteTokenizer().encode("stable"))
        first = drafter.propose(3)
        second = drafter.propose(3)
        self.assertEqual(first.token_ids.tolist(), second.token_ids.tolist())

    def test_an_empty_prompt_is_refused(self) -> None:
        model = build_model()
        with self.assertRaises(ValueError):
            speculative_generate(
                model,
                ByteTokenizer(),
                "",
                4,
                drafter=NgramDrafter(VOCABULARY_SIZE),
            )


if __name__ == "__main__":
    unittest.main()
