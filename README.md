# Shrinking BERT

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#results">Results</a>
</p>

## About

This repository contains implementations of model compression techniques for BERT on the NER task (CoNLL-2003) — embedding and linear layer factorization, layer sharing and knowledge distillation — with the goal of reducing `bert-base-cased` to 20M parameters with minimal loss in F1.

## Installation

### Requirements
- Python 3.12.11
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/shrinking-bert.git
cd shrinking-bert

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

## How To Use

To run the training script, use the following command:

```bash
uv run train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `CONFIG_NAME`: A config file from `src/configs` (`train` by default, `distill` for knowledge distillation)
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides

## Examples

Fine-tune the `bert-base-cased` teacher with online logging:

```bash
uv run train.py model=baseline writer.run_name=baseline writer.mode=online
```

Fine-tune BERT with a factorized embedding matrix (rank 64):

```bash
uv run train.py model=factorized_embed writer.run_name=factorized_embed writer.mode=online
```

Distill the teacher into a 4-layer student with online logging:

```bash
uv run train.py -cn=distill writer.run_name=student_kd writer.mode=online
```

Distill the teacher into a 10-layer student with a factorized embedding matrix:

```bash
uv run train.py -cn=distill writer.run_name=student_kd_factorized_embed \
  model.embedding_rank=64 model.config_overrides.num_hidden_layers=10 writer.mode=online
```

## Results

See [REPORT.md](REPORT.md) for the results table and takeaways.
