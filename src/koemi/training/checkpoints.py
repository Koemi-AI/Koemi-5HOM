from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from koemi.configuration.settings import ModelSettings
from koemi.data.hybrid_tokenizer import HybridVocabulary
from koemi.model.network import KoemiModel


CHECKPOINT_FORMAT_VERSION = 7
VOCABULARY_PARAMETER_NAMES = frozenset(
    {"embedding.weight", "token_predictor.weight", "token_predictor.bias"}
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: KoemiModel
    model_settings: ModelSettings
    vocabulary: HybridVocabulary | None = None


class CheckpointStore:
    def save(
        self,
        checkpoint_path: str | Path,
        model: KoemiModel,
        overwrite: bool = False,
        *,
        vocabulary: HybridVocabulary | None = None,
    ) -> Path:
        target_path = Path(checkpoint_path).expanduser().resolve()
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"checkpoint already exists: {target_path}")
        if vocabulary is not None and vocabulary.vocabulary_size != model.settings.vocabulary_size:
            raise ValueError("the vocabulary size does not match the model head")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_settings": model.settings.to_dict(),
            "model_state": model.state_dict(),
        }
        if vocabulary is not None:
            payload["vocabulary"] = vocabulary.to_payload()
        torch.save(payload, target_path)
        return target_path

    def load(self, checkpoint_path: str | Path, device: str = "cpu") -> LoadedCheckpoint:
        source_path = Path(checkpoint_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {source_path}")
        raw_checkpoint = torch.load(source_path, map_location=device, weights_only=True)
        checkpoint = self.validate_checkpoint(raw_checkpoint)
        model_settings = ModelSettings.from_dict(checkpoint["model_settings"])
        model = KoemiModel(model_settings).to(device)
        model.load_state_dict(checkpoint["model_state"])
        raw_vocabulary = checkpoint.get("vocabulary")
        vocabulary = None if raw_vocabulary is None else HybridVocabulary.from_payload(raw_vocabulary)
        if vocabulary is not None and vocabulary.vocabulary_size != model_settings.vocabulary_size:
            raise ValueError("the stored vocabulary does not match the checkpoint head")
        return LoadedCheckpoint(model, model_settings, vocabulary)

    def validate_checkpoint(self, raw_checkpoint: Any) -> dict[str, Any]:
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("checkpoint must contain a dictionary payload")
        if raw_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("checkpoint format version is not supported")
        if not isinstance(raw_checkpoint.get("model_settings"), dict):
            raise ValueError("checkpoint model settings are invalid")
        if not isinstance(raw_checkpoint.get("model_state"), dict):
            raise ValueError("checkpoint model state is invalid")
        raw_vocabulary = raw_checkpoint.get("vocabulary")
        if raw_vocabulary is not None and not isinstance(raw_vocabulary, dict):
            raise ValueError("checkpoint vocabulary payload is invalid")
        return raw_checkpoint


def expand_model_vocabulary(model: KoemiModel, vocabulary: HybridVocabulary) -> KoemiModel:
    """Grow a trained model to a hybrid vocabulary without discarding what it learned.

    Every id the model already knew keeps its exact embedding row and head row, so
    the byte-level behaviour is preserved. A new id starts at the mean of the rows
    of the bytes it expands to, which is a calibrated starting point rather than
    noise, and the new rows still need training before they carry real meaning.
    """
    target_size = vocabulary.vocabulary_size
    current_size = model.settings.vocabulary_size
    if target_size < current_size:
        raise ValueError("the target vocabulary must not be smaller than the current one")
    if target_size == current_size:
        return model
    device = model.embedding.weight.device
    expanded = KoemiModel(replace(model.settings, vocabulary_size=target_size)).to(device)
    carried_state = {
        name: value
        for name, value in model.state_dict().items()
        if name not in VOCABULARY_PARAMETER_NAMES
    }
    incompatible = expanded.load_state_dict(carried_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("the checkpoint carries parameters this model does not define")
    if set(incompatible.missing_keys) != set(VOCABULARY_PARAMETER_NAMES):
        raise ValueError("only the vocabulary parameters may be rebuilt during expansion")
    with torch.no_grad():
        expanded.embedding.weight[:current_size].copy_(model.embedding.weight)
        expanded.token_predictor.weight[:current_size].copy_(model.token_predictor.weight)
        expanded.token_predictor.bias[:current_size].copy_(model.token_predictor.bias)
        for token_id in range(current_size, target_size):
            byte_values = vocabulary.token_bytes(token_id)
            if not byte_values:
                continue
            rows = torch.tensor(list(byte_values), dtype=torch.long, device=device)
            expanded.embedding.weight[token_id].copy_(
                model.embedding.weight.index_select(0, rows).mean(dim=0)
            )
            expanded.token_predictor.weight[token_id].copy_(
                model.token_predictor.weight.index_select(0, rows).mean(dim=0)
            )
            expanded.token_predictor.bias[token_id].copy_(
                model.token_predictor.bias.index_select(0, rows).mean()
            )
    return expanded


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointStore",
    "LoadedCheckpoint",
    "VOCABULARY_PARAMETER_NAMES",
    "expand_model_vocabulary",
]
