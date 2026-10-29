from llm.config import load_config
from llm.model import GPTLanguageModel


def main() -> None:
    config = load_config()
    model = GPTLanguageModel(config)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    print(f"Model: {config['model']['name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter storage: {parameter_bytes / (1024 ** 2):.2f} MiB")
    print(f"Context length: {config['model']['max_seq_len']}")
    print(f"Vocabulary size: {config['model']['vocab_size']}")


if __name__ == "__main__":
    main()


