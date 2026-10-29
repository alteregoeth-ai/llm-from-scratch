import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm.config import load_config
from llm.model import GPTLanguageModel


@pytest.fixture(scope="module")
def small_config() -> dict:
    return {
        "model": {
            "name": "test-gpt",
            "vocab_size": 128,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "max_seq_len": 16,
            "dropout": 0.0,
            "bias": False,
        }
    }


@pytest.fixture
def model(small_config: dict) -> GPTLanguageModel:
    torch.manual_seed(42)
    return GPTLanguageModel(small_config)


def test_forward_output_shape(model: GPTLanguageModel) -> None:
    input_ids = torch.randint(0, 128, (2, 8))

    logits, loss = model(input_ids)

    assert logits.shape == (2, 8, 128)
    assert loss is None


def test_forward_computes_loss(model: GPTLanguageModel) -> None:
    input_ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 8, 128)
    assert loss is not None
    assert torch.isfinite(loss)


def test_backward_produces_gradients(model: GPTLanguageModel) -> None:
    input_ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    _, loss = model(input_ids, targets)
    assert loss is not None

    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_generation_appends_tokens(model: GPTLanguageModel) -> None:
    input_ids = torch.randint(0, 128, (1, 4))

    generated = model.generate(input_ids, max_new_tokens=3)

    assert generated.shape == (1, 7)


def test_rejects_sequence_longer_than_context(
    model: GPTLanguageModel,
) -> None:
    input_ids = torch.randint(0, 128, (1, 17))

    with pytest.raises(ValueError, match="exceeds"):
        model(input_ids)


def test_rejects_mismatched_target_shape(
    model: GPTLanguageModel,
) -> None:
    input_ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 7))

    with pytest.raises(ValueError, match="same shape"):
        model(input_ids, targets)


def test_configured_model_can_be_instantiated() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "model.yaml")
    model = GPTLanguageModel(config)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count > 0
    assert model.vocab_size == 16000
    assert model.max_seq_len == 512
