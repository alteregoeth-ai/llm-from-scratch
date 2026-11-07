# LLM from Scratch

A small GPT-style language model built from scratch with PyTorch: model architecture, data processing, and a full training pipeline with checkpoints. The goal of this project is to show, step by step, how to build and train your own LLM, even if you have never done it before.

> 📖 Full step-by-step article: *[link to the article coming soon]*

## What's inside

- **Model**: a decoder-only Transformer (the same family as GPT)
- **Data tools**: turning raw text into tokens the model can read
- **Training pipeline**: mixed precision, gradient accumulation, validation, checkpoints, resume after interruption
- **Configs**: all settings in one YAML file, with a separate one for Kaggle (free GPU)

## Project structure

```plaintext
llm-from-scratch/
├── configs/          settings (model.yaml, kaggle.yaml)
├── src/llm/          the main code
│   ├── config.py         reads and checks the settings
│   ├── data.py           turns text into tokens
│   ├── dataset.py        serves tokens to the model
│   ├── model.py          the model itself
│   └── trainer.py        the training process
├── scripts/          files you run directly
│   ├── train.py          starts training
│   └── inspect_model.py  shows the model's size
├── tests/            automatic checks
└── requirements.txt  list of libraries
```

## Quick start

### 1. Install the tools

You need:

- [Python 3.12](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [VS Code](https://code.visualstudio.com/) (or any code editor you like)

### 2. Get the code

Open a terminal (in VS Code: **Terminal → New Terminal**, or press `Ctrl + ``) and run:

```bash
git clone https://github.com/alteregoeth-ai/llm-from-scratch.git
cd llm-from-scratch
```

### 3. Create a virtual environment

A virtual environment keeps this project's libraries separate from everything else on your computer.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of the terminal line.

### 4. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Tell Python where the code is

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "$PWD\src"
```

**Mac / Linux:**

```bash
export PYTHONPATH=$PWD/src
```

> Steps 3 (activation) and 5 must be repeated every time you open a new terminal.

### 6. Check that everything works

```bash
python -c "import torch, tokenizers, yaml, numpy; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Dependencies installed successfully.')"
pytest
```

`CUDA available: False` is fine: it just means you have no NVIDIA GPU locally. We will train on Kaggle's free GPU.

### 7. Look at the model

```bash
python scripts/inspect_model.py
```

This builds the model without training it and prints its size (parameters, context length, vocabulary size).

## Next steps

1. **Prepare the data**: convert text into a file of tokens (covered in the article).
2. **Train the model**: run `python scripts/train.py --config configs/model.yaml`, or train for free on Kaggle with `configs/kaggle.yaml`.
3. **Experiment**: change the numbers in `configs/model.yaml` to make the model bigger or smaller.

## Notes

- Checkpoints are loaded with `weights_only=False`, so only load checkpoint files from sources you trust.
