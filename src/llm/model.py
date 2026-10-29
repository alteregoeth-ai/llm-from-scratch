import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(d_model, dim=-1)

        q = q.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, d_model)

        return self.resid_dropout(self.proj(attended))


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, dropout: float, bias: bool) -> None:
        super().__init__()

        self.fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(
            d_model=d_model,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only Transformer for next-token prediction."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        model_config = config["model"]

        self.vocab_size = model_config["vocab_size"]
        self.max_seq_len = model_config["max_seq_len"]
        d_model = model_config["d_model"]
        n_heads = model_config["n_heads"]
        n_layers = model_config["n_layers"]
        dropout = model_config["dropout"]
        bias = model_config["bias"]

        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

        # Share input embedding and output projection weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        for name, parameter in self.named_parameters():
            if name.endswith(("attention.proj.weight", "mlp.proj.weight")):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * n_layers),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len].")

        _, seq_len = input_ids.shape

        if seq_len == 0:
            raise ValueError("input_ids must contain at least one token.")

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}."
            )

        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have the same shape as input_ids."
                )

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively from a prompt."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be zero or greater.")

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.max_seq_len :]

                logits, _ = self(context)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, k)
                    cutoff = values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < cutoff,
                        float("-inf"),
                    )

                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)

            return input_ids
        finally:
            self.train(was_training)
