# trade-ai Current Context

Last reviewed: 2026-10-20.

## Purpose and Architecture

A Python/`uv` project for market-data analysis and GRU-based directional forecasts. The current app does not place broker orders.

1. `src/trade_ai/app.py` accepts a prompt and shows the result in Gradio on port 7860.
2. `ai_parser.py` uses local Ollama `qwen2.5:0.5b` to extract action, ticker, and lookback days. Its date anchor comes from the system date at process startup.
3. `openalgo.py` owns the shared OpenAlgo client and performs history/status API calls.
4. `data_engine.py` requests candles through that client, builds features, and performs inference.
5. `util.py` provides shared OHLCV normalization for inference, downloaded data, and CSV cache reads.
6. `features.py` builds normalized base and multi-timeframe inputs using indicator formulas from `src/trade_ai/indicator/`.
7. `stock_metadata.py` defines enum-backed cap and sector categories, ticker mappings, and defaults.
8. `model.py` defines the GRU classifier.
9. `train.py` requests history through `openalgo.py`, normalizes/caches it, and trains the model.

## Data and Feature Contract

- Input data must be one-minute OHLCV candles timestamped at candle open. The UI exposes only `1m`, and `fetch_and_clean_candles` rejects other intervals.
- Resampled 3m, 5m, 15m, 30m, and 1h bars are timestamped at close. One-hour bars align to the 09:15 session open. Backward as-of joins use only completed bars; warmup values are not backfilled.
- `add_multi_timeframe_features()` in `features.py` prepares resampled bars, calculates/aligns all EMA stacks, then applies the non-EMA indicator map to every standard intraday timeframe: 1m, 3m, 5m, 15m, 30m, and 1h. EMA spans are 9/21/50; normalized 14-bar and 7-bar ATR, volatility, 14-bar and 21-bar RSI, and relative MACD (12, 26, 9) line/signal/histogram also cover every timeframe. MACD line, signal, and histogram are divided by their timeframe close and stored as decimal ratios. Register any new timeframe-based indicator in that map so it is calculated across the full set. The daily macro gate remains separate. Indicator formulas live in separate modules under `src/trade_ai/indicator/`.
- The 80-item `FEATURE_COLUMNS` list in `features.py` defines model inputs. It includes EMA 9/21/50 distances and normalized 14/7-bar ATR, volatility, 14/21-bar RSI, and relative MACD features for all standard intraday timeframes; base returns; daily macro gate; and enum-generated stock metadata flags.
- `stock_metadata.py` owns `STOCK_METADATA`, the `MarketCap` and `Sector` string enums, the `StockMetadata` type, defaults, generated metadata feature names, and `encode_categorical_metadata()`. Adding a member to either enum automatically adds the corresponding one-hot feature to `FEATURE_COLUMNS`. Unknown tickers default to mid-cap and the `OTHER` sector.
- The macro gate uses completed daily closes, a 10-session SMA, its slope, and close-vs-SMA alignment. It is `1` for aligned bullish conditions, `-1` for aligned bearish conditions, and `0` otherwise. Inference only allows a forecast when its direction agrees with this gate.
- Training labels are optional and enabled with `include_targets=True`. Inference leaves them disabled so the latest complete feature rows remain available.

## Model and Training State

- `StockPredictionBrain` builds its GRU input size from `len(FEATURE_COLUMNS)`.
- Training uses `SlidingWindowDataset`, which holds one ticker's feature matrix and produces windows on demand; DataLoader batch size is 256. Training currently runs five epochs, sequence length five, Adam, and `ReturnWeightedBCELoss`.
- Per-ticker source data is cached in `tickData/`; model checkpoints are stored in `.run_cache/gru_trading_model.pth` after successful epochs.
- Previous 25-, 34-, 35-, 39-, 48-, 54-, 60-, and 78-input checkpoints are incompatible with the current 80-input schema. Inference is disabled when no compatible checkpoint loads. Train a new model before using forecasts.

## Environment and Run Commands

- Requires Python 3.14+ (`.python-version`) and `uv`.
- `OPENALGO_API_KEY` is required; `OPENALGO_HOST` is optional. `.env` is Git-ignored but is not loaded by the app. Source it into the shell before app or training commands.
- `src/trade_ai/openalgo.py` lazily creates one SDK client from those environment variables and exposes history and analyzer-status operations. `get_history()` standardizes DataFrame OHLCV results through `standardize_ohlcv_columns()` by default; pass `standardize=False` to retain the SDK response. Training and inference use its shared `openalgo` instance.
- `src/trade_ai/util.py` provides `load_cached_ohlcv(path, standardize=True)`, returning cached data or `None` when absent, and `standardize_ohlcv_columns()` for lowercase/title-case OHLCV names and timestamps stored as a dataframe index or column. Use `standardize=False` only when a caller needs untouched CSV column names.
- Sync dependencies: `uv sync`.
- Launch UI: `uv run python src/trade_ai/app.py`.
- Train/retrain: `uv run python src/trade_ai/train.py`.
- See `first-start.md` for `.env` setup and the repo-local temporary-cache workaround.
