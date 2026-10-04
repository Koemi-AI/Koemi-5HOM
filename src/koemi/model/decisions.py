from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DecisionAnswer:
    kind: str
    value: str | float
    probabilities: tuple[float, ...]
    max_probability: float


def typed_decisions(logits: Tensor, labels: Sequence[Sequence[str]], kinds: Sequence[str],
                    temperature: float = 1.0) -> tuple[DecisionAnswer, ...]:
    if not math.isfinite(temperature) or not 0.1 <= temperature <= 10:
        raise ValueError("temperature must be finite and between 0.1 and 10")
    if logits.ndim != 2 or len(labels) != logits.shape[0] or len(kinds) != logits.shape[0]:
        raise ValueError("logits, labels and kinds must have matching decision rows")
    answers = []
    for row, options, kind in zip(logits, labels, kinds, strict=True):
        if (not options or len(options) > row.numel() or
                any(not isinstance(option, str) or not option for option in options) or
                len(set(options)) != len(options)):
            raise ValueError("each decision requires distinct nonempty option labels fitting the logits")
        if kind not in {"choice", "score", "noul"}:
            raise ValueError("kind must be choice, score or noul")
        if kind == "noul" and tuple(options) != ("false", "true"):
            raise ValueError("noul labels must be false, true in that order")
        values = row[:len(options)].detach().float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("valid option logits must be finite")
        probabilities = tuple((values / temperature).softmax(-1).cpu().tolist())
        selected = max(range(len(options)), key=probabilities.__getitem__)
        value = (options[selected] if kind == "choice" else
                 probabilities[1] if kind == "noul" else
                 sum(index * probability for index, probability in enumerate(probabilities)))
        answers.append(DecisionAnswer(kind, value, probabilities, max(probabilities)))
    return tuple(answers)
