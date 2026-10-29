import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm.dataset import TokenDataset


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokens.bin"
    np.arange(100, dtype=np.uint16).tofile(path)
    return path


def test_dataset_length(token_file: Path) -> None:
    dataset = TokenDataset(token_file, max_seq_len=8)

    assert len(dataset) == 92


def test_batch_shapes_and_shift(token_file: Path) -> None:
    dataset = TokenDataset(token_file, max_seq_len=8)

    x, y = dataset.get_batch(
        batch_size=4,
        rng=np.random.default_rng(42),
    )

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_values_are_contiguous(token_file: Path) -> None:
    dataset = TokenDataset(token_file, max_seq_len=8)

    x, y = dataset.get_batch(
        batch_size=3,
        rng=np.random.default_rng(123),
    )

    combined = torch.cat((x[:, :1], y), dim=1)

    assert torch.equal(combined[:, 1:], y)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_batch_sampling_is_reproducible(token_file: Path) -> None:
    dataset = TokenDataset(token_file, max_seq_len=8)

    x1, y1 = dataset.get_batch(
        batch_size=4,
        rng=np.random.default_rng(7),
    )
    x2, y2 = dataset.get_batch(
        batch_size=4,
        rng=np.random.default_rng(7),
    )

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        TokenDataset(tmp_path / "missing.bin", max_seq_len=8)


def test_rejects_too_short_file(tmp_path: Path) -> None:
    path = tmp_path / "short.bin"
    np.array([1, 2, 3], dtype=np.uint16).tofile(path)

    with pytest.raises(ValueError, match="at least"):
        TokenDataset(path, max_seq_len=3)


def test_rejects_invalid_sequence_length(token_file: Path) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        TokenDataset(token_file, max_seq_len=0)


def test_rejects_invalid_batch_size(token_file: Path) -> None:
    dataset = TokenDataset(token_file, max_seq_len=8)

    with pytest.raises(ValueError, match="batch_size"):
        dataset.get_batch(batch_size=0)


def test_rejects_malformed_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "malformed.bin"
    path.write_bytes(b"\x01")

    with pytest.raises(ValueError, match="divisible by 2"):
        TokenDataset(path, max_seq_len=8)
