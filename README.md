# Autonomous AI Trading Pipeline (4GB VPS Optimized)

A lightweight, high-performance algorithmic trading and deep learning framework built for resource-constrained production environments. It trains custom GRU neural networks on multi-year historical minute data fetched through OpenAlgo, prioritizing financial expectancy over raw directional accuracy.

---

## Features
- **Memory-Aware Training:** Processes one ticker at a time, keeps its feature matrix in memory, and creates overlapping sequences per DataLoader sample (`batch_size=256`).
- **OpenAlgo Integration:** Seamlessly fetches, caches, and standardizes multi-year 1-minute historical datasets (`tickData/`).
- **Financial Expectancy Loss (`ReturnWeightedBCELoss`):** Weights classification accuracy based on actual forward return magnitudes.
- **Fault-Tolerant Execution:** Built-in retry loops and per-ticker exception handling ensure uninterrupted training batches.
- **Incremental Checkpointing:** Automatically saves model weights after each epoch to `.run_cache/gru_trading_model.pth`.
- **Causal Higher-Timeframe Features:** Resampled bars are timestamped at close, merged only after completion, and left missing until their indicators have warmed up.
- **Daily Macro Gate:** Uses the completed daily close, 10-session SMA, SMA slope, and close-vs-SMA alignment. The inference forecast is suppressed when the gate is neutral or opposing.
- **Environment-Based OpenAlgo Credentials:** Reads `OPENALGO_API_KEY` and optional `OPENALGO_HOST` from the environment.
- **One-Minute Source Data:** The feature pipeline requires one-minute candles, which it aggregates into completed 3-minute, 5-minute, 15-minute, 30-minute, and session-aligned 1-hour bars.

Set the OpenAlgo environment before starting the app or training:

```sh
export OPENALGO_API_KEY="your-api-key"
export OPENALGO_HOST="https://your-openalgo-host" # optional
```

The model currently uses 79 input features. EMA 9/21/50 distances, the 1-minute EMA 9/21 spread normalized by EMA21 and by Close, normalized 14-bar and 7-bar ATR, and 14-bar and 21-bar RSI are calculated for 1-minute, 3-minute, 5-minute, 15-minute, 30-minute, and 1-hour bars. ORB features capture the first 15, 30, and 60 minutes after the NSE open: range size, distances to the range high/low/midpoint, and close-based breakout direction. The opening range is only exposed after it completes. Liquidity, FVG, BAG, and MACD indicator modules remain available but are not used by the current feature pipeline or model inputs. Checkpoints include their feature names; old untagged or mismatched checkpoints are rejected. Retrain after feature or formula changes, even if the feature count is unchanged; prior 216-, 88-, 82-, and 64-input checkpoints are incompatible with this schema. Do not commit local `.env` files.

---

## Project Structure
```text
├── .run_cache/                             # Model state checkpoints (.pth)
├── tickData/                               # Cached historical 1-minute CSV datasets
├── src/
│   └── trade_ai/
│       ├── __init__.py
│       ├── ai_parser.py                    # NLP parser
│       ├── app.py                          # Main app
│       ├── data_engine.py                  # Candle retrieval and feature processing
│       ├── prediction_engine.py            # Model lifecycle and inference
│       ├── features.py                     # Feature engineering & normalization utilities
│       ├── stock_metadata.py                # Enum-backed stock metadata registry
│       ├── openalgo_client.py              # Shared OpenAlgo client and API operations
│       ├── util.py                         # Shared OHLCV normalization utilities
│       ├── model.py                        # Main model
│       └── train.py                        # Main training script & batched data loader
|
├── app.py                                  # Live inference application / UI launcher
├── requirements.json /.txt                 # Project dependencies
└── README.md

To start with an untrained model, enter “reset your brain” or “start as new” in the prompt. The app backs up the current checkpoint under `.run_cache/` with a timestamped `_backup_` filename, removes the original checkpoint, and disables inference until retraining and restarting the app to load the new checkpoint.
