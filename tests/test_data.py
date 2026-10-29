import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm.data import (
    encode_documents,
    load_token_ids,
    save_token_ids,
    split_documents,
)


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str) -> FakeEncoding:
        return FakeEncoding(self.mapping[text])


def test_split_documents_has_no_overlap() -> None:
    documents = [f"document-{i}" for i in range(100)]

    train, validation = split_documents(documents, validation_ratio=0.2, seed=42)

    assert len(train) == 80
    assert len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(documents)


def test_split_documents_is_deterministic() -> None:
    documents = [f"document-{i}" for i in range(20)]

    first = split_documents(documents, validation_ratio=0.2, seed=42)
    second = split_documents(documents, validation_ratio=0.2, seed=42)

    assert first == second


def test_split_documents_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_documents(["a", "b"], validation_ratio=1.0)

    with pytest.raises(ValueError, match="At least two"):
        split_documents(["only one document"])


def test_encode_documents_appends_eos() -> None:
    tokenizer = FakeTokenizer({"hello": [5, 6], "world": [7]})

    result = encode_documents(
        ["hello", "world"],
        tokenizer,
        eos_token_id=2,
        vocab_size=128,
    )

    assert result.tolist() == [5, 6, 2, 7, 2]
    assert result.dtype == np.uint16


def test_encode_documents_empty_input() -> None:
    tokenizer = FakeTokenizer({})

    result = encode_documents(
        [],
        tokenizer,
        eos_token_id=2,
        vocab_size=128,
    )

    assert result.size == 0
    assert result.dtype == np.uint16


def test_encode_documents_rejects_out_of_range_token() -> None:
    tokenizer = FakeTokenizer({"bad": [128]})

    with pytest.raises(ValueError, match="outside the vocabulary"):
        encode_documents(["bad"], tokenizer, eos_token_id=2, vocab_size=128)


def test_save_and_load_token_ids(tmp_path: Path) -> None:
    expected = np.array([1, 42, 255, 2], dtype=np.uint16)
    output_path = tmp_path / "nested" / "tokens.bin"

    saved_path = save_token_ids(expected, output_path)
    actual = load_token_ids(saved_path)

    assert saved_path == output_path
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)


def test_save_rejects_wrong_dtype(tmp_path: Path) -> None:
    tokens = np.array([1, 2, 3], dtype=np.int32)

    with pytest.raises(ValueError, match="uint16"):
        save_token_ids(tokens, tmp_path / "tokens.bin")
