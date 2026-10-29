
import copy
from pathlib import Path

import pytest
import yaml

from llm.config import load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


@pytest.fixture
def valid_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_load_config_successfully() -> None:
    config = load_config(CONFIG_PATH)

    assert config["model"]["name"] == "scratch-gpt"
    assert config["model"]["vocab_size"] == 16000
    assert config["model"]["d_model"] == 512
    assert config["model"]["n_heads"] == 8
    assert config["model"]["n_layers"] == 8


def test_load_config_resolves_paths() -> None:
    config = load_config(CONFIG_PATH)

    assert Path(config["tokenizer"]["path"]).is_absolute()
    assert Path(config["data"]["train_path"]).is_absolute()
    assert Path(config["data"]["val_path"]).is_absolute()
    assert Path(config["checkpoint"]["output_dir"]).is_absolute()


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(PROJECT_ROOT / "configs" / "missing.yaml")


def test_validate_config_rejects_missing_section(valid_config: dict) -> None:
    config = copy.deepcopy(valid_config)
    del config["model"]

    with pytest.raises(ValueError, match="Missing configuration sections"):
        validate_config(config)


def test_validate_config_rejects_invalid_attention_dimensions(
    valid_config: dict,
) -> None:
    config = copy.deepcopy(valid_config)
    config["model"]["d_model"] = 510

    with pytest.raises(ValueError, match="divisible"):
        validate_config(config)


def test_validate_config_rejects_invalid_dropout(valid_config: dict) -> None:
    config = copy.deepcopy(valid_config)
    config["model"]["dropout"] = 1.0

    with pytest.raises(ValueError, match="dropout"):
        validate_config(config)


def test_validate_config_rejects_vocab_mismatch(valid_config: dict) -> None:
    config = copy.deepcopy(valid_config)
    config["tokenizer"]["vocab_size"] = 12000

    with pytest.raises(ValueError, match="must match"):
        validate_config(config)


def test_validate_config_rejects_invalid_batch_size(valid_config: dict) -> None:
    config = copy.deepcopy(valid_config)
    config["training"]["batch_size"] = 0

    with pytest.raises(ValueError, match="batch_size"):
        validate_config(config)


def test_validate_config_rejects_invalid_data_type(valid_config: dict) -> None:
    config = copy.deepcopy(valid_config)
    config["data"]["dtype"] = "int32"

    with pytest.raises(ValueError, match="uint16"):
        validate_config(config)


def test_validate_config_rejects_invalid_checkpoint_interval(
    valid_config: dict,
) -> None:
    config = copy.deepcopy(valid_config)
    config["checkpoint"]["save_every"] = 0

    with pytest.raises(ValueError, match="save_every"):
        validate_config(config)
