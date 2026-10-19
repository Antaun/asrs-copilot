# asrs-copilot

NLP project for working with ASRS (Aviation Safety Reporting System) report narratives.

## Layout

```
configs/            experiment and model configs
data/raw/           original data (not tracked)
data/processed/     cleaned/tokenized data (not tracked)
models/             trained checkpoints (not tracked)
notebooks/          exploration
reports/figures/    plots and results
src/asrs_copilot/   package code
tests/              unit tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
