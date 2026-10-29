
from pathlib import Path
from typing import Any

import yaml


def validate_config(config: dict[str, Any]) -> None:
    """Validate the model configuration before training."""

    required_sections = {
        "model",
        "tokenizer",
        "training",
        "data",
        "checkpoint",
    }

    missing_sections = required_sections - config.keys()
    if missing_sections:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    model = config["model"]
    tokenizer = config["tokenizer"]
    training = config["training"]
    data = config["data"]
    checkpoint = config["checkpoint"]

    # Model validation
    for key in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len"):
        if not isinstance(model.get(key), int) or isinstance(model.get(key), bool):
            raise ValueError(f"model.{key} must be an integer.")

        if model[key] <= 0:
            raise ValueError(f"model.{key} must be greater than zero.")

    if model["d_model"] % model["n_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads.")

    dropout = model.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
        raise ValueError("model.dropout must be a number.")

    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in the range [0, 1).")

    if not isinstance(model.get("bias"), bool):
        raise ValueError("model.bias must be true or false.")

    # Tokenizer validation
    if tokenizer.get("type") != "bpe":
        raise ValueError("Only the 'bpe' tokenizer type is currently supported.")

    if tokenizer.get("vocab_size") != model["vocab_size"]:
        raise ValueError(
            "tokenizer.vocab_size must match model.vocab_size."
        )

    if not tokenizer.get("path"):
        raise ValueError("tokenizer.path must not be empty.")

    if not tokenizer.get("eos_token"):
        raise ValueError("tokenizer.eos_token must not be empty.")

    # Training validation
    positive_integer_fields = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "warmup_steps",
    )

    for key in positive_integer_fields:
        value = training.get(key)

        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"training.{key} must be an integer.")

        if value < 0 or (key != "warmup_steps" and value == 0):
            raise ValueError(
                f"training.{key} must be "
                f"{'zero or greater' if key == 'warmup_steps' else 'greater than zero'}."
            )

    if training["warmup_steps"] > training["max_steps"]:
        raise ValueError(
            "training.warmup_steps cannot exceed training.max_steps."
        )

    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = training.get(key)

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"training.{key} must be a number.")

        if value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(
                f"training.{key} must be "
                f"{'zero or greater' if key == 'weight_decay' else 'greater than zero'}."
            )

    if not isinstance(training.get("seed"), int) or isinstance(
        training.get("seed"), bool
    ):
        raise ValueError("training.seed must be an integer.")

    # Data validation
    if data.get("dtype") != "uint16":
        raise ValueError("Only 'uint16' token data is currently supported.")

    if model["vocab_size"] > 65536:
        raise ValueError(
            "uint16 supports token IDs up to 65535; "
            "model.vocab_size cannot exceed 65536."
        )

    for key in ("train_path", "val_path"):
        if not data.get(key):
            raise ValueError(f"data.{key} must not be empty.")

    # Checkpoint validation
    if not checkpoint.get("output_dir"):
        raise ValueError("checkpoint.output_dir must not be empty.")

    for key in ("save_every", "keep_last"):
        value = checkpoint.get(key)

        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"checkpoint.{key} must be a positive integer.")


def load_config(path: str | Path = "configs/model.yaml") -> dict[str, Any]:
    """Load and validate the YAML configuration."""

    config_path = Path(path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    validate_config(config)

    # Resolve artifact paths relative to the project root, not the terminal.
    project_root = config_path.parent.parent

    for section, keys in {
        "tokenizer": ("path",),
        "data": ("train_path", "val_path"),
        "checkpoint": ("output_dir",),
    }.items():
        for key in keys:
            configured_path = Path(config[section][key])
            if not configured_path.is_absolute():
                config[section][key] = str(project_root / configured_path)

    return config
