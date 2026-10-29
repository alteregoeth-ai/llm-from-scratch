from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class TokenizerProtocol(Protocol):
    """Minimal interface required from a tokenizer."""

    def encode(self, text: str) -> object: ...


def split_documents(
    documents: Sequence[str],
    validation_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split documents deterministically into train and validation sets."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if len(documents) < 2:
        raise ValueError("At least two documents are required.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(documents))

    validation_size = max(1, round(len(documents) * validation_ratio))
    validation_size = min(validation_size, len(documents) - 1)

    validation_indices = set(indices[:validation_size].tolist())

    train_documents = [
        document
        for index, document in enumerate(documents)
        if index not in validation_indices
    ]
    validation_documents = [
        document
        for index, document in enumerate(documents)
        if index in validation_indices
    ]

    return train_documents, validation_documents


def encode_documents(
    documents: Sequence[str],
    tokenizer: TokenizerProtocol,
    eos_token_id: int,
    vocab_size: int,
) -> np.ndarray:
    """Encode documents and append EOS after each document."""

    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id must be within the vocabulary.")

    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab_size exceeds the uint16 token range.")

    encoded_parts: list[np.ndarray] = []

    for document in documents:
        encoded = tokenizer.encode(document)
        token_ids = getattr(encoded, "ids", None)

        if token_ids is None:
            raise TypeError("tokenizer.encode(text) must return an object with ids.")

        tokens = np.asarray(token_ids, dtype=np.int64)

        if tokens.size and (tokens.min() < 0 or tokens.max() >= vocab_size):
            raise ValueError("Tokenizer produced an ID outside the vocabulary.")

        document_tokens = np.empty(tokens.size + 1, dtype=np.uint16)
        if tokens.size:
            document_tokens[:-1] = tokens.astype(np.uint16)
        document_tokens[-1] = eos_token_id

        encoded_parts.append(document_tokens)

    if not encoded_parts:
        return np.empty(0, dtype=np.uint16)

    return np.concatenate(encoded_parts)


def save_token_ids(token_ids: np.ndarray, output_path: str | Path) -> Path:
    """Save token IDs as a flat uint16 binary file."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be a one-dimensional array.")

    if token_ids.dtype != np.uint16:
        raise ValueError("token_ids must have dtype uint16.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token_ids.tofile(path)

    return path


def load_token_ids(input_path: str | Path) -> np.ndarray:
    """Load token IDs from a flat uint16 binary file."""

    return np.fromfile(input_path, dtype=np.uint16)
