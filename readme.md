# trade-ai

`trade-ai` forecasts market direction from one-minute OHLCV candles. It fetches history through OpenAlgo, calculates multi-timeframe features, and supports a three-class LightGBM and GRU ensemble. FastAPI exposes prediction, history, and analyser-status endpoints using a typed `UniversalServerResponse` envelope.

The project produces model forecasts only. It does not place or manage broker orders. A probability score is not a calibrated guarantee of prediction quality.

## Pipeline

1. **Data ingestion:** OpenAlgo returns one-minute candle history. Inference adds 30 calendar days before the requested display range for feature warm-up, orders candles by timestamp, resolves duplicate timestamps in favor of later rows, and limits processing to 25,000 rows.
2. **Feature engineering:** The pipeline calculates return features and RSI, ATR, EMA-distance, FVG, and liquidity-sweep features over 1m, 3m, 5m, 15m, 30m, and 1h intervals. GRU RSI is scaled to 0–1; LightGBM RSI uses its regular scale.
3. **Labeling:** Training labels are `-1` (stop-loss first), `0` (timeout), or `1` (take-profit first). Label horizons stay within one calendar date and end before the configured 15:15 market close.
4. **Training:** The master trainer trains LightGBM first, then the PyTorch GRU. Run it directly from the command line.
5. **Inference:** The engine combines model probabilities 50/50 when both models return results, and falls back to the available model if only one is ready.

GRU training and app inference share a 30-candle sequence-length setting. See [context.md](context.md) for implementation details and known limitations.

## Getting Started

See [setup-start.md](setup-start.md) for requirements, dependency setup, environment configuration, startup, training, and troubleshooting.

Quick commands from the repository root:

```sh
uv sync
uv run python src/trade_ai/app.py
```

Set `OPENALGO_API_KEY` in the shell before starting the application or training. `OPENALGO_HOST` is optional. The application does not load `.env` automatically; setup-start.md explains how to export local settings.

Run the full dual-model training pipeline in the foreground with:

```sh
uv run python -m trade_ai.model.training.train_master
```

## Project Structure

```text
trade-ai/
├── agents.md                         # Agent workflow and documentation maintenance rules
├── context.md                        # Current architecture, contracts, and known limitations
├── readme.md                         # Project overview and structure
├── setup-start.md                    # Installation, environment, startup, and troubleshooting
├── pyproject.toml                    # Project metadata and dependencies
├── uv.lock                           # Locked dependency versions
├── .python-version                   # Python version requirement
├── nodemon.json                      # Optional development restart configuration
└── src/trade_ai/
    ├── __init__.py                   # Package entry point
    ├── app.py                        # Loads prediction models and starts the API
    ├── openalgo/                     # OpenAlgo client, types, and conversion helpers
    ├── util.py                       # OHLCV ordering, merge, and resampling helpers
    ├── model/
    │   ├── stock_metadata.py         # Ticker metadata and encoded categories
    │   ├── data_engine.py             # History fetch, feature warm-up, display, inference call
    │   ├── features.py                # Multi-timeframe model features
    │   ├── training/
    │   │   ├── sequence_loader.py    # Same-calendar-date GRU sequence helpers
    │   │   ├── sequence_loader.md    # Session and sequence boundary notes
    │   │   ├── labels.py              # Triple Barrier labels for training
    │   │   ├── train_lgb.py           # LightGBM data preparation and trainer
    │   │   ├── train_gru.py           # GRU training and session-safe dataset
    │   │   └── train_master.py        # Selective model-training orchestration
    │   ├── models.py                  # Model architecture, loaded instances, and lifecycle
    │   ├── prediction_engine.py       # Model probability blending and inference
    │   ├── data_engine.md             # Data ingestion behavior notes
    │   ├── prediction_engine.md       # Ensemble behavior notes
    ├── indicator/
    │   ├── *.py                       # Indicator implementations
    │   └── *.md                       # Indicator documentation
    ├── server/
    │   ├── server.py                  # FastAPI server entry point
    │   ├── predict_route.py            # Structured prediction request route
    │   ├── openalgo_route.py          # OpenAlgo history and analyser routes
    │   ├── serverTypes.py              # Generic endpoint response envelope
    │   └── server.md                 # API endpoint notes
    └── strategy/
        └── strategy.md               # Strategy notes
```

Each FastAPI endpoint returns the generic `UniversalServerResponse` envelope:
`intent`, then `data.eventType`, with a successful payload in `data.data` or an
error in `data.error`. Endpoint payload and error contracts are kept in the
corresponding `*Types.py` module; global intents and events are in
`universalTypes.py`.

Training caches and model checkpoints are runtime files under `src/trade_ai/tickData/` and `src/trade_ai/.run_cache/` when generated; they are not part of the source tree. The prediction history request also caps processing at 25,000 rows; this is not a 10-million-row batch-processing loop.
