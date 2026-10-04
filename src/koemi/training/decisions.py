from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


CHOICE = 0
SCORE = 1
NOUL = 2


def validate_distribution(logits: Tensor, target: Tensor, option_mask: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape != target.shape or logits.shape != option_mask.shape:
        raise ValueError("logits, target and option_mask require matching nonempty [batch, options] shapes")
    if option_mask.dtype != torch.bool or not bool(option_mask.any(dim=-1).all()):
        raise ValueError("every decision requires a boolean mask with at least one valid option")
    if not bool(torch.isfinite(logits.masked_select(option_mask)).all()):
        raise ValueError("valid option logits must be finite")
    if not bool(torch.isfinite(target).all()) or bool(target.lt(0).any()):
        raise ValueError("targets must be finite and non-negative")
    if bool(target.masked_select(~option_mask).ne(0).any()):
        raise ValueError("padded options cannot carry target probability")
    if not torch.allclose(target.float().sum(-1), torch.ones(logits.shape[0], device=target.device), atol=1e-6, rtol=1e-5):
        raise ValueError("every target distribution must sum to one")


def decision_loss(logits: Tensor, target: Tensor, option_mask: Tensor, qtype: Tensor,
                  ordinal_weight: float = 1.0) -> Tensor:
    validate_distribution(logits, target, option_mask)
    if qtype.shape != (logits.shape[0],) or qtype.dtype != torch.long or bool(((qtype < CHOICE) | (qtype > NOUL)).any()):
        raise ValueError("qtype must be a long vector of choice=0, score=1 or noul=2")
    if not math.isfinite(ordinal_weight) or ordinal_weight < 0:
        raise ValueError("ordinal_weight must be finite and non-negative")
    if bool((option_mask & (~option_mask).cummax(dim=-1).values).any()):
        raise ValueError("valid options must occupy a contiguous prefix")
    if bool(((qtype == NOUL) & option_mask.sum(-1).ne(2)).any()):
        raise ValueError("noul requires exactly two options in false, true order")
    logp = functional.log_softmax(logits.float().masked_fill(~option_mask, -torch.inf), dim=-1)
    safe_logp = logp.masked_fill(~option_mask, 0)
    loss = -(target.float() * safe_logp).sum(-1)
    probabilities = logp.exp()
    cdf_error = (probabilities.cumsum(-1) - target.float().cumsum(-1)).square()
    ordinal = (cdf_error * option_mask).sum(-1) / (option_mask.sum(-1) - 1).clamp_min(1)
    return (loss + ordinal_weight * ordinal * qtype.eq(SCORE)).mean()


@dataclass(frozen=True)
class DecisionTrainingSettings:
    epochs: int = 4
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 2.5e-5
    accumulation_steps: int = 1
    freeze_encoder: bool = False
    ordinal_weight: float = 1.0
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("epochs", "accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "encoder_learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.ordinal_weight) or self.ordinal_weight < 0:
            raise ValueError("ordinal_weight must be finite and non-negative")
        if not isinstance(self.freeze_encoder, bool):
            raise ValueError("freeze_encoder must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class DecisionTrainingResult:
    epoch_losses: tuple[float, ...]
    optimizer_steps: int
    decisions_seen: int


def validate_decision_batch(batch: dict[str, Tensor]) -> None:
    required = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target"}
    if not required.issubset(batch) or any(not isinstance(batch[name], Tensor) for name in required):
        raise ValueError("decision batch requires tensor inputs, marker positions/mask, qtype and target")
    ids, attention, positions, mask = (batch[name] for name in ("input_ids", "attention_mask", "marker_pos", "marker_mask"))
    if ids.ndim != 2 or ids.numel() == 0 or ids.dtype != torch.long or bool(ids.lt(0).any()):
        raise ValueError("input_ids must be nonempty non-negative long [batch, length]")
    if attention.shape != ids.shape or not bool(((attention == 0) | (attention == 1)).all()) or not bool(attention.bool().any(-1).all()):
        raise ValueError("attention_mask must match input_ids and keep at least one token per row")
    if positions.ndim != 2 or positions.shape[0] != ids.shape[0] or positions.shape != mask.shape or positions.dtype != torch.long:
        raise ValueError("marker_pos must be long [batch, options] matching marker_mask")
    validate_distribution(torch.zeros_like(batch["target"]), batch["target"], mask)
    if batch["target"].shape != positions.shape:
        raise ValueError("target must match marker_pos")
    valid_positions = positions.masked_select(mask)
    if bool(((valid_positions < 0) | (valid_positions >= ids.shape[1])).any()):
        raise ValueError("option markers must reference valid sequence positions")
    safe_positions = positions.clamp(0, ids.shape[1] - 1)
    if bool((mask & ~attention.bool().gather(1, safe_positions)).any()):
        raise ValueError("option markers cannot reference padded tokens")
    decision_loss(torch.zeros_like(batch["target"]), batch["target"], mask, batch["qtype"], 0)


def decision_forward(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    result = model(*(batch[name] for name in
                     ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    return result[0] if isinstance(result, tuple) else result


def train_decisions(model: nn.Module, batches: Sequence[dict[str, Tensor]],
                    settings: DecisionTrainingSettings,
                    logger: logging.Logger | None = None) -> DecisionTrainingResult:
    if not batches:
        raise ValueError("decision training requires at least one batch")
    for batch in batches:
        validate_decision_batch(batch)
    device = torch.device(settings.device)
    torch.manual_seed(settings.seed)
    model.to(device)
    encoder = getattr(model, "encoder", None)
    if settings.freeze_encoder and not isinstance(encoder, nn.Module):
        raise ValueError("freeze_encoder requires a model.encoder module")
    if settings.freeze_encoder:
        encoder.requires_grad_(False)
    encoder_ids = {id(parameter) for parameter in encoder.parameters()} if encoder is not None else set()
    encoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in encoder_ids]
    head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in encoder_ids]
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": settings.encoder_learning_rate})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": settings.learning_rate})
    if not groups:
        raise ValueError("decision model has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    logger = logger or logging.getLogger("koemi.decisions")
    history: list[float] = []
    updates = 0
    seen = 0
    for epoch in range(settings.epochs):
        model.train()
        if settings.freeze_encoder:
            encoder.eval()
        weighted_loss = 0.0
        epoch_count = 0
        for start in range(0, len(batches), settings.accumulation_steps):
            group = batches[start:start + settings.accumulation_steps]
            group_count = sum(batch["input_ids"].shape[0] for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = {name: batch[name].to(device) for name in
                         ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target")}
                logits = decision_forward(model, batch)
                loss = decision_loss(logits, batch["target"], batch["marker_mask"], batch["qtype"], settings.ordinal_weight)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("decision loss is not finite")
                count = batch["input_ids"].shape[0]
                (loss * count / group_count).backward()
                weighted_loss += float(loss.detach()) * count
                epoch_count += count
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        seen += epoch_count
        history.append(weighted_loss / epoch_count)
        logger.info("decision_epoch_completed epoch=%s loss=%.6f decisions=%s optimizer_steps=%s frozen_encoder=%s",
                    epoch + 1, history[-1], epoch_count, updates, settings.freeze_encoder)
    return DecisionTrainingResult(tuple(history), updates, seen)


def fit_decision_temperature(logits: Tensor, target: Tensor, option_mask: Tensor,
                             iterations: int = 50, *, minimum_temperature: float = 0.1,
                             maximum_temperature: float = 10.0) -> float:
    validate_distribution(logits, target, option_mask)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if (not math.isfinite(minimum_temperature) or not math.isfinite(maximum_temperature) or
            not 0 < minimum_temperature <= 1 <= maximum_temperature):
        raise ValueError("temperature bounds must be finite, positive and include one")
    values = logits.detach().float().cpu()
    gold = target.detach().float().cpu()
    mask = option_mask.detach().cpu()
    with torch.enable_grad():
        log_temperature = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], max_iter=iterations, line_search_fn="strong_wolfe")

        def objective(scale: Tensor) -> Tensor:
            logp = (values / scale).masked_fill(~mask, -torch.inf).log_softmax(-1).masked_fill(~mask, 0)
            return -(gold * logp).sum(-1).mean()

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = objective(log_temperature.clamp(math.log(minimum_temperature), math.log(maximum_temperature)).exp())
            loss.backward()
            return loss

        optimizer.step(closure)
        fitted = log_temperature.detach().clamp(math.log(minimum_temperature), math.log(maximum_temperature)).exp()
        fitted = fitted.clamp(minimum_temperature, maximum_temperature)
        baseline = objective(torch.ones(()))
        calibrated = objective(fitted)
    if not bool(torch.isfinite(calibrated)) or bool(calibrated > baseline):
        return 1.0
    return float(fitted)
