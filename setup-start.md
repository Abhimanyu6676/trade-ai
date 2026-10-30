# Setup and Start Guide

## Overview

This guide covers project requirements, initial setup, dependencies, pre-start checks, common commands, and common setup/start fixes.

Update it when setup or startup steps change, or when a new setup/start check is added. Keep credentials and private environment values out of tracked files and logs.

## Requirements

- Python 3.14 or newer, as specified by `.python-version` and `pyproject.toml`.
- `uv` for dependency and project environment management.
- Gymnasium and Stable-Baselines3 for the PPO meta-agent; `uv sync` installs them.
- The GNU OpenMP runtime (`libgomp1` on Debian/Ubuntu), required by LightGBM.
- An OpenAlgo endpoint and API key for market-data access.
- nodemon installed globally for development hot reloads ```npm i -g nodemon```

## Initial Setup

Run commands from the repository root.

1. Confirm Python and uv are available:

   ```sh
   python --version
   uv --version
   ```

2. Install and synchronize project dependencies:

   ```sh
   uv sync
   ```

   This creates or updates the project-managed `.venv` from `pyproject.toml` and `uv.lock`.

3. Configure OpenAlgo in a local, Git-ignored `.env` file:

   ```dotenv
   OPENALGO_API_KEY=your-api-key
   OPENALGO_HOST=https://your-openalgo-host
   TRADE_AI_CHAT_ORIGINS=http://localhost:3011,http://127.0.0.1:3011
   ```

   All three settings are required when starting the app. `OPENALGO_HOST` should point to the OpenAlgo endpoint, and `TRADE_AI_CHAT_ORIGINS` is a comma-separated list of allowed frontend origins. Do not commit `.env` or put credential values in tracked files.

4. The application loads the repository-root `.env` before startup. Shell variables take precedence. You can also export values explicitly when starting from a different environment:

   ```sh
   set -a
   source .env
   set +a
   ```

5. The startup validator checks that all settings declared in `APPLICATION_ENVIRONMENT_VARIABLES` are present and nonblank before loading models or starting the server.

## Dependencies Guide

- `uv sync` is the standard install/update command; dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
- Run project commands through `uv run` so they use the managed environment.
- Avoid installing dependencies into the system Python with `pip`.
- After changing dependency declarations, run `uv sync` to update the environment and lock file.

## Pre-Start Checks

Before launching the app or training, confirm:

- You are in the repository root and `uv sync` has completed.
- Python is 3.14 or newer.
- `OPENALGO_API_KEY` is set in the shell or declared in the repository-root `.env`. This check does not print its value:

  ```sh
  if test -n "$OPENALGO_API_KEY" || grep -qE '^[[:space:]]*OPENALGO_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]#]' .env 2>/dev/null; then
    echo "OPENALGO_API_KEY is configured"
  else
    echo "OPENALGO_API_KEY is missing"
  fi
  ```

- The configured OpenAlgo endpoint is reachable and its API key is valid.

## Common Commands

Start the brain runtime and FastAPI server on port 3010:

```sh
uv run python src/trade_ai/app.py
```

For development (Auto restart using nodemon), Run:

```sh
nodemon --watch . --ext py --exec "uv run python src/trade_ai/app.py"
```

OR run from project root, as we have setup `nodemon.json`

```sh
nodemon
```

OR, use the following for hot reloading

```sh
uv run jurigged --watch src/ -v src/trade_ai/app.py
```

The app loads the prediction models, then runs FastAPI/Uvicorn in the main
process. Structured `/api/brain/predict` requests fetch data and run inference
through that API process. Stop the app with `Ctrl+C`.

On hosts where the default temporary directory is mounted `noexec`, set repository-local temporary/cache paths before starting:

```sh
mkdir -p .run_cache
export TMPDIR="$(pwd)/.run_cache"
export PIP_CACHE_DIR="$(pwd)/.run_cache"
uv run python src/trade_ai/app.py
```

Train selected directional models in the foreground. The first selector is the
mindset (`long`, `short`, or `both`, default `both`); the second is the
component (`gru`, `lgb`, `rl`, or `all`, default `all`). The full run trains all
four base specialists and then the shared PPO meta-agent:

```sh
uv run python -m trade_ai.model.training.train_master both all
```

Examples:

```sh
uv run python -m trade_ai.model.training.train_master long lgb
uv run python -m trade_ai.model.training.train_master short gru
uv run python -m trade_ai.model.training.train_master both rl
```

RL is one shared policy over all four base models and requires compatible
out-of-sample predictions from each specialist. See
`src/trade_ai/model/training/training.md` for its validation and detached-run
logging details.

Stop a foreground app or training process with `Ctrl+C`.

## First Start

The app can start without trained artifacts, but it cannot produce a decision
until all four directional base checkpoints and a compatible RL policy exist.
Run the master training command above to create those artifacts. Checkpoint
schema or base-hash mismatches require retraining before prediction is enabled.

## Common Error Fixes

- **LightGBM fails with `libgomp.so.1: cannot open shared object file`:** install the OS OpenMP runtime on Debian/Ubuntu:

  ```sh
  sudo apt-get update
  sudo apt-get install -y libgomp1
  ```

- **`OPENALGO_API_KEY` missing:** confirm it is declared in the repository-root `.env` or exported in the shell. The application loads `.env` at startup; restart the app after changing the file.
- **OpenAlgo connection/authentication errors:** verify `OPENALGO_HOST`, confirm the endpoint is reachable from the machine, and check that the API key is valid. Keep the key private when collecting or sharing logs.
- **Temporary file execution errors on a `noexec` host:** use the repo-local `.run_cache` for `TMPDIR` and `PIP_CACHE_DIR` as shown in Common Commands.
- **Dependencies appear missing after a project update:** run `uv sync` from the repository root, then retry using `uv run`.
