from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Memory-mapped token stream for next-token prediction."""

    def __init__(
        self,
        path: str | Path,
        max_seq_len: int,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be greater than zero.")

        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"Token file not found: {self.path}")

        if self.path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
            raise ValueError("Token file size must be divisible by 2 bytes.")

        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

        if len(self.tokens) < max_seq_len + 1:
            raise ValueError(
                f"Token file must contain at least {max_seq_len + 1} tokens."
            )

        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Number of valid starting positions for a full sequence."""
        return len(self.tokens) - self.max_seq_len

    def get_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random contiguous sequences and shifted targets."""

        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero.")

        if rng is None:
            rng = np.random.default_rng()

        starts = rng.integers(0, len(self), size=batch_size)

        batch = np.stack(
            [
                self.tokens[start : start + self.max_seq_len + 1].astype(
                    np.int64
                )
                for start in starts
            ]
        )

        x = torch.from_numpy(batch[:, :-1].copy()).to(device)
        y = torch.from_numpy(batch[:, 1:].copy()).to(device)

        return x, y
