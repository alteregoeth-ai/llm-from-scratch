import numpy as np
import pytest
import torch

from llm.dataset import TokenDataset
from llm.model import GPTLanguageModel
from llm.trainer import (
    evaluate_loss,
    get_learning_rate,
    load_checkpoint,
    save_checkpoint,
    train,
)


@pytest.fixture
def tiny_config(tmp_path):
    return {
        "model": {
            "vocab_size": 32,
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "max_seq_len": 8,
            "dropout": 0.0,
            "bias": False,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_steps": 1,
            "max_grad_norm": 1.0,
            "seed": 42,
        },
        "checkpoint": {
            "output_dir": str(tmp_path / "checkpoints"),
            "save_every": 1,
            "keep_last": 2,
        },
    }


@pytest.fixture
def tiny_datasets(tmp_path):
    rng = np.random.default_rng(42)
    train_tokens = rng.integers(0, 32, size=128, dtype=np.uint16)
    val_tokens = rng.integers(0, 32, size=64, dtype=np.uint16)

    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    return (
        TokenDataset(train_path, max_seq_len=8),
        TokenDataset(val_path, max_seq_len=8),
    )


def test_learning_rate_warmup_and_decay():
    assert get_learning_rate(0, 10, 2, 1.0) == pytest.approx(0.5)
    assert get_learning_rate(1, 10, 2, 1.0) == pytest.approx(1.0)
    assert get_learning_rate(10, 10, 2, 1.0) == pytest.approx(0.0)


def test_evaluate_loss_returns_finite_value(tiny_config, tiny_datasets):
    _, val_dataset = tiny_datasets
    model = GPTLanguageModel(tiny_config)

    loss = evaluate_loss(
        model,
        val_dataset,
        batch_size=2,
        device="cpu",
        num_batches=2,
    )

    assert np.isfinite(loss)
    assert loss > 0
    assert model.training


def test_save_and_load_checkpoint(tiny_config, tmp_path):
    model = GPTLanguageModel(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    path = tmp_path / "manual.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        step=7,
        best_val_loss=2.5,
        config=tiny_config,
    )

    restored_model = GPTLanguageModel(tiny_config)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=0.001,
    )

    metadata = load_checkpoint(
        path=path,
        model=restored_model,
        optimizer=restored_optimizer,
        device="cpu",
    )

    assert metadata["step"] == 7
    assert metadata["best_val_loss"] == pytest.approx(2.5)

    for original, restored in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.equal(original, restored)


def test_train_creates_checkpoints(tiny_config, tiny_datasets):
    train_dataset, val_dataset = tiny_datasets
    model = GPTLanguageModel(tiny_config)

    history = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=tiny_config,
        device="cpu",
        eval_batches=2,
        log_every=1,
    )

    checkpoint_dir = tmp_path_from_config(tiny_config)

    assert len(history) == 2
    assert all(np.isfinite(item["train_loss"]) for item in history)
    assert all(np.isfinite(item["val_loss"]) for item in history)
    assert (checkpoint_dir / "latest.pt").is_file()
    assert (checkpoint_dir / "best.pt").is_file()
    assert (checkpoint_dir / "step_000002.pt").is_file()


def tmp_path_from_config(config):
    from pathlib import Path

    return Path(config["checkpoint"]["output_dir"])
