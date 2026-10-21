# trade-ai Current Context

Last reviewed: 2026-10-20.

## Purpose and Architecture

A Python/`uv` project for market-data analysis and GRU-based directional forecasts. The current app does not place broker orders.

1. `src/trade_ai/app.py` accepts a prompt and shows the result in Gradio on port 7860.
2. `ai_parser.py` uses local Ollama `qwen2.5:0.5b` to extract action, ticker, and lookback days. Its date anchor comes from the system date at process startup.
3. `openalgo_client.py` owns the shared OpenAlgo client and performs history/status API calls.
4. `data_engine.py` requests candles through that client and builds features; `prediction_engine.py` owns model loading, lifecycle, and inference.
5. `util.py` provides shared OHLCV normalization for inference, downloaded data, and CSV cache reads.
6. `features.py` builds normalized base and multi-timeframe inputs using indicator formulas from `src/trade_ai/indicator/`.
7. `stock_metadata.py` defines enum-backed cap and sector categories, ticker mappings, and defaults.
8. `model.py` defines the GRU classifier.
9. `train.py` requests history through `openalgo_client.py`, normalizes/caches it, and trains the model.

## Data and Feature Contract

- Input data must be one-minute OHLCV candles timestamped at candle open. The UI exposes only `1m`, and `fetch_and_clean_candles` rejects other intervals.
- Resampled 3m, 5m, 15m, 30m, and 1h bars are timestamped at close. One-hour bars align to the 09:15 session open. Backward as-of joins use only completed bars; warmup values are not backfilled.
- `add_multi_timeframe_features()` in `features.py` prepares resampled bars, calculates/aligns all EMA stacks, then applies the non-EMA indicator map to every standard intraday timeframe: 1m, 3m, 5m, 15m, 30m, and 1h. EMA spans are 9/21/50; normalized 14-bar and 7-bar ATR, volatility, 14-bar and 21-bar RSI, FVG, BAG, and liquidity features also cover every timeframe. The MACD indicator remains available but is not calculated in the current feature pipeline. Bullish FVG requires a sell-side sweep on its middle candle; bearish FVG requires a buy-side sweep. FVG midpoint distance represents consequent encroachment, and a close through the far edge flips the zone into an inverse FVG. FVG candidates require an opposing sweep and three-candle gap; playbook qualification requires a 1.0 ATR displacement body, a close beyond pre-sweep 20-bar structure, and formation during the London/New York 07:00-10:00 local killzone. FVG features include CE touch, retest depth, preferred edge, candle-1 stop distance, and directional liquidity-target distance. BAG requires overlapping opposing qualified FVGs within two bars and a close beyond the prior zone; it includes retest, rejection, stop/target, and reversal displacement features. Liquidity levels use swing highs/lows confirmed after two right-side bars; sweeps require a wick beyond an existing level and a close back across it. Distances are relative to close. Register new timeframe-based indicators in this map. The daily macro gate remains separate. Indicator formulas live in separate modules under `src/trade_ai/indicator/`.
- The 216-item `FEATURE_COLUMNS` list in `features.py` defines model inputs. It includes EMA 9/21/50 distances and the 1m EMA 9/21 spreads normalized by EMA21 and Close, normalized 14/7-bar ATR, volatility, 14/21-bar RSI, FVG, BAG/rejection, and liquidity/sweep features for all standard intraday timeframes; base returns; daily macro gate; and enum-generated stock metadata flags.
- `stock_metadata.py` owns `STOCK_METADATA`, the `MarketCap` and `Sector` string enums, the `StockMetadata` type, defaults, generated metadata feature names, and `encode_categorical_metadata()`. Adding a member to either enum automatically adds the corresponding one-hot feature to `FEATURE_COLUMNS`. Unknown tickers default to mid-cap and the `OTHER` sector.
- The macro gate uses completed daily closes, a 10-session SMA, its slope, and close-vs-SMA alignment. It is `1` for aligned bullish conditions, `-1` for aligned bearish conditions, and `0` otherwise. Inference only allows a forecast when its direction agrees with this gate.
- Training labels are optional and enabled with `include_targets=True`. Inference leaves them disabled so the latest complete feature rows remain available.

## Model and Training State

- `StockPredictionBrain` builds its GRU input size from `len(FEATURE_COLUMNS)`.
- `ai_parser.py` detects brain-reset and training-start phrases before calling Ollama. Reset copies an existing checkpoint to `.run_cache/gru_trading_model_backup_<timestamp>.pth`, deletes the original, clears the loaded model from memory, and disables inference until retraining. A failed backup leaves the checkpoint file intact and is reported to the user.
- Asking the running UI to “start training” or “train the model” calls `train.start_model_training()`. The training module owns the background thread and calls `train_model()`; `prediction_engine.py` owns inference state and checkpoint loading. Duplicate starts are rejected; inference is disabled while training runs. On completion, the app loads the saved checkpoint immediately. Training resumes from a checkpoint if one exists; reset first to train from scratch. Progress and failures go to the server console, and stopping the app stops in-process training.
- Training uses `SlidingWindowDataset`, which holds one ticker's feature matrix and produces windows on demand; DataLoader batch size is 256. Training currently runs five epochs, sequence length five, Adam, and `ReturnWeightedBCELoss`. Its per-batch weights scale absolute five-minute returns by 1000 plus a unit baseline and multiply that by a per-ticker positive-class weight capped at 10. Tickers with only one direction label are skipped. Per-ticker logs include direction label counts and min/max/mean predicted probabilities. Targets are binary positive vs. non-positive five-minute returns, not triple-barrier labels or a three-class neutral target.
- Per-ticker source data is cached in `tickData/`; model checkpoints are stored in `.run_cache/gru_trading_model.pth` after successful epochs.
- Previous 25-, 34-, 35-, 39-, 48-, 54-, 60-, 78-, 80-, 98-, 116-, 140-, 146-, 152-, 215-, and 232-input checkpoints are incompatible with the current 216-input schema. BAG changed from a Breakaway Gap to a Balanced Price Gap and gained a rejection feature; retrain before using forecasts. Inference is disabled when no compatible checkpoint loads.

## Environment and Run Commands

- Requires Python 3.14+ (`.python-version`) and `uv`.
- `OPENALGO_API_KEY` is required; `OPENALGO_HOST` is optional. `.env` is Git-ignored but is not loaded by the app. Source it into the shell before app or training commands.
- `src/trade_ai/openalgo_client.py` lazily creates one SDK client from those environment variables and exposes history and analyzer-status operations. `get_history()` standardizes DataFrame OHLCV results through `standardize_ohlcv_columns()` by default; pass `standardize=False` to retain the SDK response. Training and inference use its shared `openalgo` instance.
- `src/trade_ai/util.py` provides `load_cached_ohlcv(path, standardize=True)`, returning cached data or `None` when absent, and `standardize_ohlcv_columns()` for lowercase/title-case OHLCV names and timestamps stored as a dataframe index or column. Use `standardize=False` only when a caller needs untouched CSV column names.
- Sync dependencies: `uv sync`.
- Launch UI: `uv run python src/trade_ai/app.py`.
- Train/retrain: `uv run python src/trade_ai/train.py`.
- Alternatively, ask the running UI to “start training”; it runs in the background and updates the live model when complete.
- See `first-start.md` for `.env` setup and the repo-local temporary-cache workaround.
