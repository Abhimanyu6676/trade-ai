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

The model currently uses 78 input features. EMA 9/21/50 distances, normalized 14-bar and 7-bar ATR, volatility, 14-bar and 21-bar RSI, and relative MACD (12, 26, 9) line/signal/histogram features are calculated for 1-minute, 3-minute, 5-minute, 15-minute, 30-minute, and 1-hour bars. Feature schema changes require retraining the model. Do not commit local `.env` files.

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
│       ├── data_engine.py                  # Data fetching from openalgo & manipulation
│       ├── features.py                     # Feature engineering & normalization utilities
│       ├── model.py                        # Main model
│       └── train.py                        # Main training script & batched data loader
|
├── app.py                                  # Live inference application / UI launcher
├── requirements.json /.txt                 # Project dependencies
└── README.md
