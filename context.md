# trade-ai Current Context

Last reviewed: 2026-10-24.

## Purpose and Runtime Flow

`trade-ai` is a Python application for market-data forecasts. It trains a LightGBM classifier and a PyTorch GRU classifier on three Triple Barrier outcomes. FastAPI exposes structured prediction requests, OpenAlgo history, and analyser status. It does not submit broker orders.

The market-analysis path is:

1. `app.py` initializes the model manager, which loads available checkpoints, then starts Uvicorn in the main process. FastAPI prediction requests call the async data engine directly.
2. `model/data_engine.py` fetches one-minute OHLCV candles through the shared OpenAlgo client. It requests a 30-calendar-day feature warm-up before the display-history range, sorts/deduplicates candles, and caps input at 25,000 rows. The shared OpenAlgo history and analyser-status client methods offload synchronous SDK calls to worker threads.
3. `model/features.py` builds the multi-timeframe features. Prediction routes offload feature engineering, inference, caller-candle conversion, and large history response encoding to worker threads so these synchronous operations do not block the API event loop. The data engine uses the feature result for the display grid and passes the raw OHLCV frame to inference.
4. `model/prediction_engine.py` engineers model-specific features, gets probabilities from the loaded model instances, and blends the probabilities when both models are ready. `model/models.py` owns the in-memory model instances and all loading, unloading, backup, and reload operations.

`POST /api/brain/predict` accepts structured ticker, exchange, lookback, and interval fields; supplied candle arrays are validated by `OhlcvCandle` in `openalgo/openalgoTypes.py`. The API process fetches candles and runs inference directly. The current engine supports `1m` source candles only. `POST /api/openalgo/history` fetches broker history through the shared OpenAlgo client and returns symbol, exchange, interval, and candles together in `data.data`; `GET /api/openalgo/analyser-status` queries analyser availability. All endpoints return `UniversalServerResponse` with an intent and output `eventType`; success data is stored in `data.data` and errors in `data.error`. `server/serverTypes.py` owns the generic response envelope, module-related domain types belong in their module's `*Types.py` file, and route-specific response contracts are defined with their routes. `model/modelTypes.py` and `openalgo/openalgoTypes.py` contain domain types and request/event literals, while `universalTypes.py` contains global intent/event types. Starting `app.py` loads the prediction models and runs FastAPI on port 3010 in the main process. The server allows browser requests from origins listed in `TRADE_AI_CHAT_ORIGINS`, defaulting to `http://localhost:3011,http://127.0.0.1:3011`.

## Data and Feature Contract

- Source data is one-minute candles timestamped at candle open. Expected columns are `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
- Candle columns use the lowercase schema `timestamp`, `open`, `high`, `low`, `close`, and `volume`. Training CSV caches and downloaded histories pass through `util.merge_ohlcv_history()` when loaded, which normalizes OHLCV column names, parses timestamps, stable-sorts chronologically, and keeps the later-appended row for duplicate timestamps. The feature builder also sorts direct inputs before calculating returns. The data engine applies the 25,000-row cap after sorting, preserving the newest tail.
- For a request of `days_back`, the display start is calculated as `today - (int(days_back * 1.5) + 5)` calendar days. The data request starts another 30 calendar days earlier to warm up indicators. The displayed frame is sliced back to the display start after feature calculation. This is a relative lookback from today, not an arbitrary target-date/time-window API.
- `model/features.py` builds the model-specific feature frames used by training and inference. Raw OHLCV remains in feature frames for calculations but is excluded from model inputs. See `src/trade_ai/model/features.md` for the feature counts, names, active timeframes, and scaling details. Source columns are selected in canonical order; unrelated extra input columns are excluded.
- Multi-timeframe bars are mapped back with backward `merge_asof` joins to use completed bars only.
- Feature calculation runs on the full chronological history, allowing prior sessions to warm up higher-timeframe indicators such as EMA 50 on 1h bars. GRU sequence windows are separately restricted to one calendar date.
- The 30-day buffer is not a mathematical guarantee that an EMA is fully converged. The user-facing market close is 15:15, with candle timestamps at candle open; the 15:14 candle is the last eligible candle and 15:15 is excluded.

## Labels and Training

- `model/training/labels.py` creates three mutually exclusive targets: `-1` stop-loss first, `0` timeout, `1` take-profit first. Its shared 60-bar maximum holding-period constant is used by both trainers for label look-ahead and purge size. It checks future candle closes, allows entries from 10:00 through 14:30 by default, and stops at the entry date boundary or 15:15 market close. Inputs must already be sorted chronologically.
- `model/training/train_lgb.py` loads each ticker's cached CSV or downloads history from 2020 in inclusive 30-calendar-day API chunks. It calculates LGB features and labels, then makes a chronological 80/20 split per ticker. It purges up to 60 feature rows before each validation boundary so training labels cannot inspect validation-period candles. It combines the per-ticker splits, maps classes to `0/1/2`, logs train/validation row and class counts plus their multi-logloss during training, and trains up to 500 LightGBM rounds with validation-based early stopping. Since each ticker's cutoff is based on its own labeled-row count, calendar cutoff dates can differ across tickers.
- `model/training/train_gru.py` trains a two-layer, 64-hidden-unit, three-class GRU. Its epoch limit must be 10–20; the orchestrated run allows up to 20 epochs, trains at least 10, then stops after 3 consecutive epochs without improved validation loss. It uses chronological per-ticker 80/20 splits and purges up to 60 feature rows before each validation boundary. Same-session sliding windows must end on labeled rows, though unlabeled feature rows can serve as context. It logs per-ticker and aggregate training/validation loss, accuracy, sequence counts, and class counts, and saves the checkpoint with the best validation loss. Validation features and labels are recomputed ticker by ticker each epoch to keep peak memory bounded.
- Both trainers use `src/trade_ai/tickData/<TICKER>_history.csv` when it exists. Cached rows are normalized, timestamp-parsed, sorted oldest-to-newest, and deduplicated by timestamp when loaded; caches are not automatically refreshed. If missing, the trainers await the asynchronous OpenAlgo client and download distinct inclusive 30-calendar-day chunks, allowing up to three attempts per chunk. A chunk that still fails aborts that ticker's download instead of training on partial history. Successful downloads are normalized, sorted, deduplicated, and cached.
- `model/training/train_master.py` accepts `gru`, `lgb`, or `both` (default). It backs up loaded checkpoint files to timestamped copies, unloads the selected models, and runs their training functions. It reloads the selected checkpoints only if every selected trainer returns successfully; after a training exception, those models stay unloaded in the current process until explicitly reloaded or the app restarts. Run it with `uv run python -m trade_ai.model.training.train_master [gru|lgb|both]`; see `model/training/training.md` for the step-by-step lifecycle.
- Model files are written under `src/trade_ai/model/.run_cache/` with the feature schema version in each filename: `lgb_trading_model_v{FEATURE_SCHEMA_VERSION}.txt` with a `.schema.json` sidecar, and `gru_trading_model_v{FEATURE_SCHEMA_VERSION}.pth` containing its feature schema metadata and state dictionary. Both loaders validate schema version and their model-specific ordered input columns; incompatible checkpoints must be retrained. Checkpoints in the old raw GRU state-dict format or without a LightGBM schema sidecar must also be retrained.

## Prediction Behavior and Known Limits

- `model/models.py` loads each existing checkpoint when the application starts. It infers GRU input width from the checkpoint, so GRU loading does not depend on a first prediction or cached candle data. Missing files produce a warning to train that model first.
- `model/prediction_engine.py` only requests loaded instances from `models.py`, performs inference, and blends available model probabilities. It does not load, unload, back up, or reload models.
- If both models return probabilities, the engine takes a 50/50 arithmetic mean. If only one returns probabilities, it uses that model alone. It returns the class with the largest probability (`-1`, `0`, or `1`) and that probability as confidence. Probabilities are not calibrated confidence guarantees.
- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `model/training/gru_sequence_builder.py`.
- `gru_sequence_builder.latest_session_sequence()` selects the last requested number of valid feature rows from the newest calendar date. It does not verify market hours or that timestamps are exactly one minute apart. Training windows likewise enforce calendar-date boundaries, not missing-candle continuity.
- `GRUTradingBrain` is defined in `model/models.py` and shared by the model manager and GRU training module.
- Training is orchestrated through `train_master.py`; it backs up current selected checkpoints and unloads selected models before training. It reloads selected checkpoints only after every selected trainer returns successfully; if training raises, selected models remain unloaded in the current process until explicitly reloaded or the app restarts.
- There is no liquidity/spread filter, probability calibration, automatic order submission, or documented 10-million-row processing loop in the active pipeline.

## Structured Brain API

- `POST /api/brain/predict` accepts explicit `ticker`, `exchange`, `days_back`, and `interval` fields. Exchange is `NSE` or `BSE`; the current inference pipeline accepts `1m` only.
- It can use either OpenAlgo history (`days_back`) or caller-supplied `candles` containing timestamp and lowercase OHLCV fields. The supplied array is limited to 25,000 rows and is validated before inference.
- The route returns a `MODEL/PREDICTION` envelope. Its response data contains the signal, confidence, and processed candle count; errors are strings in `data.error`.
- Invalid requests also return the same envelope with a string error and HTTP 422.

## Source Map

- `src/trade_ai/app.py`: initializes the model manager and runs the API in the main process.
- `src/trade_ai/model/data_engine.py`: OpenAlgo market-history fetch, warm-up buffer, feature display, inference call.
- `src/trade_ai/model/features.py` and `src/trade_ai/model/training/gru_sequence_builder.py`: feature schema and same-session GRU windows.
- `src/trade_ai/model/training/labels.py`: Triple Barrier target generation used by both trainers.
- `src/trade_ai/model/training/`: LightGBM and GRU trainers, master orchestration, and `training.md` lifecycle notes.
- `src/trade_ai/model/models.py`: GRU architecture, in-memory model manager, loading, unloading, timestamped checkpoint backup, and reload.
- `src/trade_ai/model/prediction_engine.py`: read-only ensemble inference using model instances from `models.py`.
- `src/trade_ai/openalgo/openalgo_client.py`, `src/trade_ai/openalgo/helper.py`, `src/trade_ai/util.py`, `src/trade_ai/model/stock_metadata.py`: shared client, OpenAlgo response conversion, candle utilities, metadata inputs.
- The OpenAlgo SDK's `analyzerstatus()` method is synchronous; call it without `await` and normalize its dictionary response in `OpenAlgo.get_analyzer_status()`.
- `src/trade_ai/indicator/`: indicator calculations and their explanatory Markdown files.
- `src/trade_ai/server/`: FastAPI application, `predict_route.py` prediction route, and `openalgo_route.py` history and analyser status routes.
- `src/trade_ai/server/serverTypes.py`: generic `UniversalServerResponse` envelope and response-data wrapper.
- `src/trade_ai/model/modelTypes.py`, `src/trade_ai/openalgo/openalgoTypes.py`, and `src/trade_ai/universalTypes.py`: internal domain/request types, event literals, and global intent/event literals. `modelTypes.py` owns the `ModelName` literal and `MODEL_NAMES` tuple; `openalgoTypes.py` owns OHLCV candle data and the validated prediction candle request model. Routes and model code import these module types when needed.

## Environment and Commands

- Python 3.14 or newer and `uv` are required.
- LightGBM's native library needs the GNU OpenMP runtime (`libgomp1` on Debian/Ubuntu).
- `OPENALGO_API_KEY` is required. `OPENALGO_HOST` is optional and defaults to the host in `openalgo_client.py`.
- The app does not load `.env` automatically; export its values into the shell before starting.
- Install dependencies with `uv sync`.
- Start the app with `uv run python src/trade_ai/app.py` (FastAPI 3010; models load in the main process).
- Run selected model training in the foreground with `uv run python -m trade_ai.model.training.train_master [gru|lgb|both]` (default: both).
- No test command is currently configured in `pyproject.toml`.
