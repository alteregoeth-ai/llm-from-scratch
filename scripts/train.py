import argparse

import torch

from llm.config import load_config
from llm.dataset import TokenDataset
from llm.model import GPTLanguageModel
from llm.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the language model from scratch.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['name']}")

    train_dataset = TokenDataset(
        path=config["data"]["train_path"],
        max_seq_len=config["model"]["max_seq_len"],
    )

    val_dataset = TokenDataset(
        path=config["data"]["val_path"],
        max_seq_len=config["model"]["max_seq_len"],
    )

    print(f"Training tokens: {len(train_dataset.tokens):,}")
    print(f"Validation tokens: {len(val_dataset.tokens):,}")

    model = GPTLanguageModel(config)

    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()
