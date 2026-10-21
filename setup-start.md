# Setup and Start Guide

## Overview

This guide covers project requirements, initial setup, dependencies, pre-start checks, common commands, and common setup/start fixes.

Update it when setup or startup steps change, or when a new setup/start check is added. Keep credentials and private environment values out of tracked files and logs.

## Requirements

- Python 3.14 or newer, as specified by `.python-version` and `pyproject.toml`.
- `uv` for dependency and project environment management.
- Ollama running locally with the `qwen2.5:0.5b` model available for prompt parsing.
- An OpenAlgo endpoint and API key for market-data access.

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
   ```

   `OPENALGO_API_KEY` is required. `OPENALGO_HOST` is optional and defaults to the endpoint configured in `src/trade_ai/openalgo_client.py`. Do not commit `.env` or put credential values in tracked files.

4. Make the `.env` values available to the process. The application does not load `.env` automatically:

   ```sh
   set -a
   source .env
   set +a
   ```

5. Confirm Ollama is running and the parser model is installed:

   ```sh
   ollama list
   ```

   If `qwen2.5:0.5b` is missing, install it with `ollama pull qwen2.5:0.5b`.

## Dependencies Guide

- `uv sync` is the standard install/update command; dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
- Run project commands through `uv run` so they use the managed environment.
- Avoid installing dependencies into the system Python with `pip`.
- After changing dependency declarations, run `uv sync` to update the environment and lock file.

## Pre-Start Checks

Before launching the app or training, confirm:

- You are in the repository root and `uv sync` has completed.
- Python is 3.14 or newer.
- `OPENALGO_API_KEY` is present in the current shell. This check does not print its value:

  ```sh
  test -n "$OPENALGO_API_KEY" && echo "OPENALGO_API_KEY is set" || echo "OPENALGO_API_KEY is missing"
  ```

- Ollama is running and `qwen2.5:0.5b` appears in `ollama list`.
- The configured OpenAlgo endpoint is reachable and its API key is valid.

## Common Commands

Start the Gradio UI on port 7860 and the FastAPI server on port 3010:

```sh
uv run python src/trade_ai/app.py
```

For development, Run:

```sh
nodemon --watch . --ext py --exec "uv run python src/trade_ai/app.py"
```

OR run from project root, as we have setup `nodemon.json`

```sh
nodemon
```

The FastAPI server runs Uvicorn in a background thread. Its `GET /` endpoint
returns `{"message": "Hello, World!"}`. Stop both services with `Ctrl+C`.

On hosts where the default temporary directory is mounted `noexec`, set repository-local temporary/cache paths before starting:

```sh
mkdir -p .run_cache
export TMPDIR="$(pwd)/.run_cache"
export PIP_CACHE_DIR="$(pwd)/.run_cache"
uv run python src/trade_ai/app.py
```

Run training in the foreground:

```sh
uv run python src/trade_ai/train.py
```

Stop a foreground app or training process with `Ctrl+C`.

## First Start

if .pth brain file is missing train you modal first

## Common Error Fixes

- **`OPENALGO_API_KEY` missing:** source `.env` into the same shell before launching the app or training. The app does not read `.env` on its own.
- **Ollama connection or model error:** start the Ollama service and run `ollama pull qwen2.5:0.5b` if the model is not listed by `ollama list`.
- **OpenAlgo connection/authentication errors:** verify `OPENALGO_HOST`, confirm the endpoint is reachable from the machine, and check that the API key is valid. Keep the key private when collecting or sharing logs.
- **Temporary file execution errors on a `noexec` host:** use the repo-local `.run_cache` for `TMPDIR` and `PIP_CACHE_DIR` as shown in Common Commands.
- **Dependencies appear missing after a project update:** run `uv sync` from the repository root, then retry using `uv run`.
