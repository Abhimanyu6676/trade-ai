# trade-ai

`trade-ai` is a Python application for generating market forecasts from historical one-minute OHLCV data. It retrieves market history through OpenAlgo, builds indicator-based inputs, and trains or runs a GRU model. A Gradio interface accepts prompts, while a local Ollama model parses supported requests.

The project produces forecasts and does not submit broker orders.

## Main Capabilities

- Fetch and cache historical market data through a shared OpenAlgo client.
- Prepare market data and indicators for model training and inference.
- Train the GRU model per ticker and save checkpoints locally.
- Run a Gradio interface for forecast requests and training controls.
- Parse supported natural-language requests with local Ollama.

## Getting Started

See [setup-start.md](setup-start.md) for requirements, dependency setup, environment configuration, startup, training, and troubleshooting.

Quick commands from the repository root:

```sh
uv sync
uv run python src/trade_ai/app.py
```

Set `OPENALGO_API_KEY` in the shell before starting the application or training. `OPENALGO_HOST` is optional. The application does not load `.env` automatically; setup-start.md explains how to export local settings.

## Project Structure
Always update this section whenever there is project structure change

```text
trade-ai/
├── agents.md                 # Agent handoff and maintenance instructions
├── context.md                # Current architecture and behavior notes
├── readme.md                 # Project summary and entry point
├── setup-start.md            # Setup, startup, and troubleshooting guide
├── pyproject.toml            # Python project metadata and dependencies
├── uv.lock                   # Locked dependency versions
├── .python-version           # Required Python version
├── tickData/                 # Cached historical market-data CSV files
└── src/trade_ai/
    ├── app.py                # Gradio application
    ├── ai_parser.py          # Prompt parsing through local Ollama
    ├── data_engine.py        # Market data retrieval and processing
    ├── features.py           # Model input preparation
    ├── model.py              # GRU model definition
    ├── openalgo_client.py    # Shared OpenAlgo client
    ├── prediction_engine.py  # Model loading and inference
    ├── stock_metadata.py     # Stock category and ticker metadata
    ├── train.py              # Model training
    ├── util.py               # Shared data utilities
    ├── server                # Python server directory
        ├── server.md         # holds the server endpoints notes and description
        ├── server.py         # Main server File
    └── indicator/            # Indicator calculations and notes
```

Runtime data and model checkpoints are stored under `.run_cache/` when created.
