# First Start

## Requirements

- Python 3.14 or newer and `uv`.
- Ollama running locally with `qwen2.5:0.5b` available for prompt parsing.
- An OpenAlgo endpoint and API key for market data.

## Setup

From the repository root, sync the project dependencies into the project-managed virtual environment:

```sh
uv sync
```

Put local OpenAlgo settings in the ignored `.env` file. Do not commit it. Required: `OPENALGO_API_KEY`. Optional: `OPENALGO_HOST` (defaults to the endpoint configured in `src/trade_ai/openalgo.py`). The app reads process environment variables and does not load `.env` itself, so load the values in the shell before running it:

```sh
set -a
source .env
set +a
```

## Start the UI

```sh
mkdir -p .run_cache
export TMPDIR="$(pwd)/.run_cache"
export PIP_CACHE_DIR="$(pwd)/.run_cache"
uv run python src/trade_ai/app.py
```

The Gradio UI listens on port 7860. The current feature pipeline accepts one-minute source candles and derives its higher-timeframe features from them.

## Train or Retrain

```sh
set -a
source .env
set +a
uv run python src/trade_ai/train.py
```

Training downloads/caches historical one-minute candles under `tickData/` and saves checkpoints under `.run_cache/`. The current model expects 146 input features. Retrain after changing `FEATURE_COLUMNS`; an incompatible checkpoint is not used for inference, which remains disabled until a compatible checkpoint is trained.
