# trade-ai Agent Guide

Last reviewed: 2026-10-20. Treat this as the short, source-checked handoff for the repository; read only task-relevant files after this guide.

## Maintenance

- When behavior, model inputs, startup, environment requirements, or known operational gotchas change, update this guide and the relevant sections in `context.md` and `first-start.md` in the same change.
- Never write credential values, tokens, or private environment values into tracked files or logs.
- Do not add or run tests unless the user asks. No test command is currently configured in `pyproject.toml`.

## Project

- Python project managed with `uv`; requires Python 3.14 or newer (`.python-version`).
- Main UI: `src/trade_ai/app.py` (Gradio, port 7860).
- Prompt parsing: `src/trade_ai/ai_parser.py` (local Ollama model `qwen2.5:0.5b`).
- Candle retrieval and feature processing: `src/trade_ai/data_engine.py`.
- Model loading, lifecycle, and inference: `src/trade_ai/prediction_engine.py`.
- OpenAlgo client lifecycle and API operations: `src/trade_ai/openalgo.py`. Route every SDK call through its shared `openalgo` instance; `get_history()` applies `standardize_ohlcv_columns()` by default and accepts `standardize=False` for raw SDK responses.
- Shared data utilities: `src/trade_ai/util.py`. `load_cached_ohlcv(path, standardize=True)` loads an existing CSV or returns `None`; turn standardization off only when the caller needs the original CSV columns. `standardize_ohlcv_columns()` owns the common OHLCV column and timestamp-index normalization.
- Feature assembly: `src/trade_ai/features.py`.
- Enum-backed stock categories and ticker metadata: `src/trade_ai/stock_metadata.py`.
- Indicator calculations: `src/trade_ai/indicator/`, one module per indicator (`ema.py`, `rsi.py`, `volatility.py`, `atr.py`, `macd.py`, `macro_trend.py`, `fvg.py`, `bag.py`, and `liquidity.py`).
- GRU model: `src/trade_ai/model.py`.
- Training: `src/trade_ai/train.py`.
- This project produces forecasts; it does not currently submit broker orders.

## Feature and Model Contract

- The pipeline requires one-minute OHLCV source candles with candle-open timestamps. The UI and data engine reject other source intervals.
- Higher-timeframe bars are labeled at their close and backward-merged only after completion. Never backfill indicator warmup values from future rows.
- Resampled intervals: 3m, 5m, 15m, 30m, 1h, and daily. One-hour bars are aligned to the 09:15 market-session open.
- `add_multi_timeframe_features()` in `src/trade_ai/features.py` prepares timeframe bars, calculates and aligns all EMA features, then calculates other indicators by timeframe. Standard intraday timeframes are 1m, 3m, 5m, 15m, 30m, and 1h. EMA, ATR, volatility, RSI, FVG, BAG, and buy-side/sell-side liquidity are calculated for every one. `macd.py` remains available but is not wired into the current feature pipeline. To add another timeframe-based indicator, register its calculation in the non-EMA indicator map; the loop applies it across all standard timeframes. The daily macro gate remains a separate daily calculation. Indicator formulas live in separate modules under `src/trade_ai/indicator/`.
- FVG candidates require an opposing-side liquidity sweep and a three-candle imbalance. `FVG_Setup_Qualified` requires a 1.0 ATR displacement body, a close beyond pre-sweep 20-bar structure, and formation during a London/New York 07:00-10:00 local killzone. FVG inputs include CE touch, retest depth, preferred entry edge, candle-1 stop distance, and directional liquidity-target distance. BAG/BPR requires two qualified opposing FVGs within two bars, with the newer event closing beyond the previous zone while their zones overlap. BAG inputs include edge rejection, retest depth, stop/target distances, and reversal displacement. Naive timestamps use Asia/Kolkata; aware timestamps retain their timezone.
- Liquidity levels are confirmed swing highs/lows with two closed bars on the right. Sweep flags mean price wicked through a previously known level and closed back across it. Buy-side/sell-side level distances are signed ratios to close; no active level is encoded as zero.
- `FEATURE_COLUMNS` in `src/trade_ai/features.py` is authoritative. The current GRU input has 216 features: base returns; EMA 9/21/50 distances and both 1m EMA 9/21 spread variants (EMA21- and Close-normalized); multi-timeframe 14-bar and 7-bar ATR, volatility, 14-bar and 21-bar RSI, FVG/BAG state and rejection, and buy-side/sell-side liquidity and sweep features; daily macro gate; and enum-generated stock metadata flags. MACD is not part of the model inputs.
- Prompt requests such as “reset your brain” or “start as new” are detected by `ai_parser.py` and routed through the UI to `prediction_engine.reset_prediction_brain()`. Reset copies `.run_cache/gru_trading_model.pth` to a unique `.run_cache/gru_trading_model_backup_<timestamp>.pth`, removes the original, clears the loaded model object, and disables inference until retraining. If backup creation fails, the checkpoint is retained and the UI reports the error.
- Prompt requests such as “start training” or “train the model” are detected deterministically by `ai_parser.py` and routed to `train.start_model_training()`. Training orchestration and its background thread live in `train.py`; `prediction_engine.py` owns the inference model and exposes helpers to unload/reload it. Duplicate starts are rejected while training runs in the app process. Inference is disabled during training, and the completed checkpoint is loaded into the live app automatically. Training resumes from an existing checkpoint when present; use the reset command first for a fresh model. Progress and failures are printed in the app server console. Stopping/restarting the app also stops this in-process training run.
- `stock_metadata.py` owns the `MarketCap` and `Sector` string enums, typed `StockMetadata` registry, unknown-ticker defaults, enum-generated feature names, and `encode_categorical_metadata()`. Adding an enum member automatically adds its one-hot column to both feature output and `FEATURE_COLUMNS`.
- EMA values from higher timeframes are represented as scale-independent distances from the current close.
- `Macro_Trend_Gate` uses completed daily closes, a 10-session SMA, the SMA slope, and price alignment: `1` bullish, `-1` bearish, `0` neutral/warming up. Inference suppresses forecasts that do not match the gate.
- Forward-return labels are created only when `compute_normalized_features(..., include_targets=True)` is used for training. Inference must retain the newest rows with complete features.
- Retrain after feature formula changes even when `FEATURE_COLUMNS` keeps the same length. The former 146-input checkpoint is incompatible with the current 216-input FVG/BAG/EMA-spread feature set. Previous 25-, 34-, 35-, 39-, 48-, 54-, 60-, 78-, 80-, 98-, 116-, 140-, 146-, 152-, 215-, and 232-input checkpoints are also incompatible; inference stays disabled until a compatible model is trained. Checkpoints are saved to `.run_cache/gru_trading_model.pth` after each successful epoch.

## Training and Data

- Historical minute CSVs are cached under `tickData/`, one file per ticker.
- Training processes one ticker at a time. `SlidingWindowDataset` holds one feature matrix and returns sequence windows on demand; keep the DataLoader batching (`batch_size=256`) and avoid materializing all overlapping windows.
- Training currently uses five epochs, sequence length five, Adam, and `ReturnWeightedBCELoss`.
- Broker downloads use three attempts per chunk. OpenAlgo history is standardized by `openalgo.get_history()`; legacy cached OHLCV CSVs use `load_cached_ohlcv()`.

## Environment and Commands

- `OPENALGO_API_KEY` is required. `OPENALGO_HOST` is optional and defaults to the endpoint configured in `src/trade_ai/openalgo.py`.
- `.env` is ignored by Git, but the application does not load it automatically. Export its values into the shell before running commands, for example:

  ```sh
  set -a
  source .env
  set +a
  ```

- Install/sync dependencies with `uv sync`.
- Launch the UI with `uv run python src/trade_ai/app.py`.
- Train with `uv run python src/trade_ai/train.py`.
- From the running UI, enter “start training” to begin the same training job in the background; check the app server console for progress.
- If the VPS has `noexec` temporary storage issues, use the repo-local cache as `TMPDIR` and `PIP_CACHE_DIR` (see `first-start.md`).
