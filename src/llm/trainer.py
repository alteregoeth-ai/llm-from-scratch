
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from llm.dataset import TokenDataset
from llm.model import GPTLanguageModel


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_learning_rate(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
) -> float:
    """Linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    if max_steps <= warmup_steps:
        return learning_rate

    decay_steps = max_steps - warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))

    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_loss(
    model: GPTLanguageModel,
    dataset: TokenDataset,
    batch_size: int,
    device: str | torch.device,
    num_batches: int = 20,
    seed: int = 12345,
    use_amp: bool = False,
) -> float:
    """Estimate average next-token loss over random validation batches."""
    if num_batches < 1:
        raise ValueError("num_batches must be greater than zero.")

    device = torch.device(device)

    was_training = model.training
    model.eval()

    rng = np.random.default_rng(seed)
    losses: list[float] = []

    try:
        for _ in range(num_batches):
            x, y = dataset.get_batch(
                batch_size=batch_size,
                device=device,
                rng=rng,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Model did not return a loss.")

            losses.append(float(loss.item()))

    finally:
        model.train(was_training)

    return sum(losses) / len(losses)


def save_checkpoint(
    path: str | Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: dict[str, Any],
    train_rng_state: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> Path:
    """Save model, optimizer, scaler, and training progress."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": (
                scaler.state_dict() if scaler is not None else None
            ),
            "best_val_loss": best_val_loss,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "train_rng_state": train_rng_state,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Restore model, optimizer, scaler, and random generator states."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")

    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    cuda_states = checkpoint.get("cuda_rng_states")

    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in cuda_states]
        )

    return {
        "step": checkpoint["step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "config": checkpoint["config"],
        "train_rng_state": checkpoint.get("train_rng_state"),
    }


def _prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    """Keep only the newest numbered checkpoints."""
    checkpoints = sorted(output_dir.glob("step_*.pt"))

    for old_checkpoint in checkpoints[:-keep_last]:
        old_checkpoint.unlink()


def train(
    model: GPTLanguageModel,
    train_dataset: TokenDataset,
    val_dataset: TokenDataset,
    config: dict[str, Any],
    device: str | torch.device,
    *,
    eval_batches: int = 20,
    log_every: int = 10,
    resume_from: str | Path | None = None,
) -> list[dict[str, float | int]]:
    """Train a GPT model for the configured number of optimizer steps."""
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]

    batch_size = training_config["batch_size"]
    accumulation_steps = training_config["gradient_accumulation_steps"]
    max_steps = training_config["max_steps"]
    learning_rate = training_config["learning_rate"]
    weight_decay = training_config["weight_decay"]
    warmup_steps = training_config["warmup_steps"]
    max_grad_norm = training_config["max_grad_norm"]
    seed = training_config["seed"]

    if eval_batches < 1:
        raise ValueError("eval_batches must be greater than zero.")

    if log_every < 1:
        raise ValueError("log_every must be greater than zero.")

    if max_steps < 1 or batch_size < 1 or accumulation_steps < 1:
        raise ValueError(
            "Training steps, batch size, and accumulation must be positive."
        )

    device = torch.device(device)
    model.to(device)

    set_seed(seed)

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    print(f"Device: {device}")
    print(
        f"Mixed precision (FP16): "
        f"{'enabled' if use_amp else 'disabled'}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_dir = Path(checkpoint_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_last = checkpoint_config["keep_last"]
    save_every = checkpoint_config["save_every"]

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")

    train_rng = np.random.default_rng(seed + 1)
    step = 0

    if resume_from is not None:
        metadata = load_checkpoint(
            path=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        train_rng_state = metadata["train_rng_state"]

        if train_rng_state is None:
            raise ValueError(
                "Checkpoint does not contain train_rng_state "
                "and cannot be used for exact training resume."
            )

        train_rng.bit_generator.state = train_rng_state

        step = metadata["step"]
        best_val_loss = metadata["best_val_loss"]

        if step >= max_steps:
            raise ValueError(
                f"Checkpoint step ({step}) must be lower than "
                f"configured max_steps ({max_steps})."
            )

        print(f"Resuming training from step {step}/{max_steps}")

    while step < max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0

        current_lr = get_learning_rate(
            step,
            max_steps,
            warmup_steps,
            learning_rate,
        )

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        for _ in range(accumulation_steps):
            x, y = train_dataset.get_batch(
                batch_size=batch_size,
                device=device,
                rng=train_rng,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Model did not return a loss.")

            scaled_loss = loss / accumulation_steps

            scaler.scale(scaled_loss).backward()

            accumulated_loss += (
                float(loss.detach().item()) / accumulation_steps
            )

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        scaler.step(optimizer)
        scaler.update()

        step += 1

        should_evaluate = (
            step % save_every == 0 or step == max_steps
        )

        if should_evaluate:
            val_loss = evaluate_loss(
                model,
                val_dataset,
                batch_size,
                device,
                num_batches=eval_batches,
                seed=seed + step,
                use_amp=use_amp,
            )

            record: dict[str, float | int] = {
                "step": step,
                "train_loss": accumulated_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }

            history.append(record)

            print(
                f"step {step}/{max_steps} | "
                f"train_loss {accumulated_loss:.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"lr {current_lr:.6g} | "
                f"scale {scaler.get_scale():.0f}"
            )

            save_checkpoint(
                output_dir / "latest.pt",
                model,
                optimizer,
                step,
                min(best_val_loss, val_loss),
                config,
                train_rng.bit_generator.state,
                scaler=scaler,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    config,
                    train_rng.bit_generator.state,
                    scaler=scaler,
                )

            save_checkpoint(
                output_dir / f"step_{step:06d}.pt",
                model,
                optimizer,
                step,
                best_val_loss,
                config,
                train_rng.bit_generator.state,
                scaler=scaler,
            )

            _prune_checkpoints(output_dir, keep_last)

        elif step % log_every == 0:
            print(
                f"step {step}/{max_steps} | "
                f"train_loss {accumulated_loss:.4f} | "
                f"lr {current_lr:.6g} | "
                f"scale {scaler.get_scale():.0f}"
            )

    return history
