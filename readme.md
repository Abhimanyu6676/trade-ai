# trade-ai

`trade-ai` forecasts long and short trade outcomes from one-minute OHLCV
candles. It fetches history through OpenAlgo, calculates multi-timeframe
features, and runs four three-class specialists (long/short LightGBM and GRU)
through one PPO meta-agent. FastAPI exposes prediction, history, and
analyser-status endpoints using a typed `UniversalServerResponse` envelope.

The service returns a decision signal and position-size suggestion. It does not
submit or manage broker orders. Model probabilities and confidence are not
calibrated guarantees of future performance.

## Pipeline

1. **Data ingestion:** OpenAlgo supplies one-minute candle history. Inference
   includes at least 30 prior trading sessions, sorts candles chronologically,
   resolves duplicates in favor of later rows, and accepts at most 25,000
   caller-supplied candles.
2. **Feature engineering:** Both directions use shared multi-timeframe returns,
   RSI, ATR, EMA-distance, FVG, sweep, and volatility-ratio features. Raw OHLCV
   values remain calculation inputs and are excluded from model inputs.
3. **Labeling:** Long and short use mirrored +1%/−0.5% and −1%/+0.5% barriers.
   Labels are trade-relative: `-1` stop loss, `0` timeout, `1` take profit.
4. **Training:** All configured tickers with usable history share chronological
   55/15/20/10 fitting, selection, walk-forward OOS, and final-holdout periods.
   The PPO meta-agent trains and selects on frozen base-model OOS forecasts;
   its final-holdout report does not decide whether the policy is saved.
5. **Inference:** The meta-agent receives all four probability vectors, their
   agreement, volatility, and optional portfolio state. It selects flat, long,
   or short exposure, a size multiplier, and a bracket multiplier of 0.75,
   1.0, or 1.25. The bracket multiplier scales both the baseline target and
   stop by the same amount, preserving the 2:1 ratio used by the labels. ATR
   is relative volatility context and does not define the bracket distances.

GRU training and inference share a 30-candle sequence length. See
[context.md](context.md) for architecture, data contracts, and known limits.

## Getting Started

See [setup-start.md](setup-start.md) for requirements, dependency setup,
environment configuration, startup, training, and troubleshooting.

Quick commands from the repository root:

```sh
uv sync
uv run python src/trade_ai/app.py
```

The full training command trains all four base specialists and the common
meta-agent:

```sh
uv run python -m trade_ai.model.training.train_master both all
```

## Project Structure

```text
trade-ai/
├── AGENTS.md                         # Agent workflow and documentation rules
├── context.md                        # Architecture, contracts, and limitations
├── readme.md                         # Project overview and structure
├── setup-start.md                    # Installation, startup, and training
├── pyproject.toml                    # Project metadata and dependencies
├── uv.lock                           # Locked dependency versions
└── src/trade_ai/
    ├── app.py                        # Loads artifacts and starts FastAPI
    ├── config/trading_config.py      # Trading rules, costs, and artifact paths
    ├── openalgo/                     # OpenAlgo client, types, conversions
    ├── model/
    │   ├── features.py               # Shared multi-timeframe model features
    │   ├── models.py                 # Four specialists, PPO, lifecycle
    │   ├── prediction_engine.py      # History orchestration and policy inference
    │   └── training/
    │       ├── long/                 # Long labels, trainers, env wrapper
    │       ├── short/                # Short labels, trainers, env wrapper
    │       ├── rl/                   # OOS predictions and PPO support
    │       ├── train_gru.py          # Shared directional GRU trainer
    │       ├── train_lgb.py          # Shared directional LightGBM trainer
    │       ├── train_rl.py           # Unified PPO training
    │       ├── trading_env.py        # Historical PPO environment
    │       ├── train_master.py       # Direction/component CLI orchestration
    │       └── training.md           # Training lifecycle and commands
    ├── indicator/                   # Technical indicator implementations
    ├── server/                      # FastAPI routes and API guidebook
    └── strategy/                    # Strategy design notes
```

The API response envelope is `intent` plus `data.eventType`, with successful
payloads in `data.data` and errors in `data.error`. Runtime candle caches and
model artifacts live under `src/trade_ai/model/training/tickData/` and
`src/trade_ai/model/.run_cache/`; they are not source files. The application can
start without trained artifacts, but prediction returns HTTP 503 until all four
base models and a compatible PPO policy are available.
