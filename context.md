# trade-ai Current Context

Last reviewed: 2026-10-21.

## Purpose and Architecture

A Python/`uv` project for market-data analysis and GRU-based directional forecasts. The current app does not place broker orders.

1. `src/trade_ai/app.py` accepts prompts and presents results through Gradio on port 7860.
2. `ai_parser.py` uses local Ollama `qwen2.5:0.5b` to extract actions, tickers, and lookback days. It detects supported reset and training commands before calling Ollama. Its date anchor comes from the system date at process startup.
3. `openalgo_client.py` owns the shared OpenAlgo client and performs history/status API calls.
4. `data_engine.py` requests candles through that client and builds features; `prediction_engine.py` owns model loading, lifecycle, and inference.
5. `util.py` provides shared OHLCV normalization for inference, downloaded data, and CSV cache reads.
6. `features.py` builds normalized base and multi-timeframe inputs using indicator formulas from `src/trade_ai/indicator/`.
7. `stock_metadata.py` defines enum-backed cap and sector categories, ticker mappings, defaults, and categorical encoding.
8. `model.py` defines the GRU classifier; `train.py` requests history, normalizes/caches it, and trains the model.

## Data and Feature Contract

- Input data must be one-minute OHLCV candles timestamped at candle open. The UI exposes only `1m`, and `fetch_and_clean_candles` rejects other source intervals.
- Resampled 3m, 5m, 15m, 30m, and 1h bars are timestamped at close. One-hour bars align to the 09:15 market-session open. Backward as-of joins use only completed bars; indicator warmup values must not be backfilled from future rows.
- `add_multi_timeframe_features()` in `features.py` prepares resampled bars and calculates/aligns EMA, ATR, volatility, and RSI across 1m, 3m, 5m, 15m, 30m, and 1h. Indicator values based on prices are represented as percentage-like ratios rather than raw price amounts where applicable.
- The active feature pipeline does not call the available Liquidity, FVG, BAG, or MACD modules. The daily macro gate is calculated separately. Indicator formulas are in separate modules under `src/trade_ai/indicator/`.
- Liquidity levels are confirmed swing highs/lows with two closed bars on the right. Sweep flags mean price wicked through a previously known level and closed back across it. Buy-side/sell-side level distances are signed ratios to close; no active level is encoded as zero.
- `FEATURE_COLUMNS` in `features.py` is authoritative for model inputs. The current schema has 79 columns: base returns; EMA 9/21/50 distances and both 1m EMA 9/21 spread variants (EMA21- and Close-normalized); multi-timeframe 14-bar and 7-bar ATR, volatility, and 14-bar and 21-bar RSI; 15m, 30m, and 1h opening-range context; daily macro gate; and enum-generated stock metadata flags. Liquidity, FVG, BAG, and MACD are not model inputs.
- `stock_metadata.py` owns the `MarketCap` and `Sector` string enums, typed `StockMetadata` registry, unknown-ticker defaults, enum-generated feature names, and `encode_categorical_metadata()`. Adding an enum member automatically adds its one-hot column to feature output and `FEATURE_COLUMNS`. Unknown tickers default to mid-cap and `OTHER` sector.
- Higher-timeframe EMA values are represented as scale-independent distances from the current close.
- The macro gate uses completed daily closes, a 10-session SMA, its slope, and close-vs-SMA alignment. It is `1` for aligned bullish conditions, `-1` for aligned bearish conditions, and `0` otherwise. Inference only allows a forecast when its direction agrees with this gate.
- ORB features use NSE's 09:15 Asia/Kolkata open and completed one-minute candles. Each opening window is fixed after 15, 30, or 60 minutes. Features include normalized range size, distances to range high/low/midpoint, and close-based breakout direction; values are exposed only after the range completes.
- Forward-return labels are created only when `compute_normalized_features(..., include_targets=True)` is used for training. Inference leaves targets disabled so the newest rows with complete features remain available.

## Model and Training State

- `StockPredictionBrain` builds its GRU input size from `len(FEATURE_COLUMNS)`.
- Checkpoints store a feature-schema version and exact feature names. Untagged or mismatched checkpoints are rejected, even when tensor dimensions match. The current schema is incompatible with older 216-, 88-, 82-, 64-, 25-, 34-, 35-, 39-, 48-, 54-, 60-, 78-, 80-, 98-, 116-, 140-, 146-, 152-, 215-, and 232-input checkpoints. Retrain after feature formula changes even if the number of columns stays the same. Inference stays disabled until a compatible checkpoint is available.
- Checkpoints are saved to `.run_cache/gru_trading_model.pth` after successful epochs.
- Prompt requests such as “reset your brain” or “start as new” are handled by `prediction_engine.reset_prediction_brain()`. Reset copies an existing checkpoint to a unique `.run_cache/gru_trading_model_backup_<timestamp>.pth`, removes the original, clears the loaded model object, and disables inference until retraining. If backup creation fails, the checkpoint is retained and the UI reports the error.
- Prompt requests such as “start training” or “train the model” are routed to `train.start_model_training()`. Training orchestration and its background thread live in `train.py`; `prediction_engine.py` owns inference state and checkpoint loading. Duplicate starts are rejected while training runs in the app process. Inference is disabled during training; the completed checkpoint is loaded into the live app automatically. Training resumes only from a compatible schema-tagged checkpoint; use the reset command first for a fresh model. Progress and failures go to the app server console. Stopping/restarting the app also stops this in-process training run.
- Training uses `SlidingWindowDataset`, which holds one ticker's feature matrix and returns sequence windows on demand. Keep the DataLoader batch size at 256 and avoid materializing all overlapping windows.
- Training currently runs five epochs with sequence length five, Adam, and `ReturnWeightedBCELoss`. Its per-batch weights scale absolute five-minute returns by 1000 plus a unit baseline, then multiply by a per-ticker positive-class weight capped at 10. Tickers with only one direction label are skipped. Per-ticker/epoch logs include positive/non-positive label counts and min/max/mean predicted probabilities.
- Direction targets are binary (`Future_Return_5m > 0`), not triple-barrier labels or a three-class neutral target. Probability diagnostics describe training behavior and are not intended to force high-confidence forecasts.
- Historical minute CSVs are cached under `tickData/`, one file per ticker. Broker downloads use three attempts per chunk.

## OpenAlgo and Data Utilities

- Route every OpenAlgo SDK call through the shared `openalgo` instance in `openalgo_client.py`. It lazily creates one SDK client from environment variables and exposes history and analyzer-status operations.
- `openalgo.get_history()` standardizes DataFrame OHLCV results through `standardize_ohlcv_columns()` by default; pass `standardize=False` to retain the SDK response. Training and inference use this shared client.
- `util.py` provides `load_cached_ohlcv(path, standardize=True)`, which loads an existing CSV or returns `None`, and `standardize_ohlcv_columns()` for lowercase/title-case OHLCV names and timestamps stored as either a dataframe index or column. Use `standardize=False` only when a caller needs the original CSV columns.

## Environment and Run Commands

- Requires Python 3.14 or newer (`.python-version`) and `uv`.
- `OPENALGO_API_KEY` is required. `OPENALGO_HOST` is optional and defaults to the endpoint configured in `openalgo_client.py`.
- `.env` is Git-ignored, but the application does not load it automatically. Export its values into the shell before running commands, for example:

  ```sh
  set -a
  source .env
  set +a
  ```

- Install/sync dependencies with `uv sync`.
- Launch the UI with `uv run python src/trade_ai/app.py`.
- Train with `uv run python src/trade_ai/train.py`.
- From the running UI, enter “start training” to launch the same training job in the background; check the app server console for progress.
- If the VPS has `noexec` temporary storage issues, use the repo-local cache as `TMPDIR` and `PIP_CACHE_DIR`; detailed steps are in `setup-start.md`.
