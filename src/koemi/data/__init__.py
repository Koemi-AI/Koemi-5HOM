from koemi.data.contracts import DatasetRecord, DatasetValidationError
from koemi.data.hybrid_tokenizer import (
    HybridTokenizer,
    HybridVocabulary,
    train_hybrid_vocabulary,
)
from koemi.data.readers import DatasetLoadReport, load_dataset_records
from koemi.data.tokenizer import ByteTokenizer, TextTokenizer


def create_tokenizer(vocabulary: HybridVocabulary | None) -> TextTokenizer:
    """Return the tokenizer a checkpoint needs: hybrid when it carries a vocabulary."""
    if vocabulary is None:
        return ByteTokenizer()
    return HybridTokenizer(vocabulary)


__all__ = [
    "ByteTokenizer",
    "DatasetLoadReport",
    "DatasetRecord",
    "DatasetValidationError",
    "HybridTokenizer",
    "HybridVocabulary",
    "TextTokenizer",
    "create_tokenizer",
    "load_dataset_records",
    "train_hybrid_vocabulary",
]
