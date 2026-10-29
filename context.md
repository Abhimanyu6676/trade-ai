# trade-ai Current Context

Last reviewed: 2026-10-24.

## Purpose and Runtime Flow

`trade-ai` is a Python application for market-data forecasts. It trains a LightGBM classifier and a PyTorch GRU classifier on three Triple Barrier outcomes. FastAPI exposes structured prediction requests, OpenAlgo history, and analyser status. It does not submit broker orders.

## Logging

`src/trade_ai/logger.py` configures the shared `trade_ai` logger at INFO level with timestamp, severity, and message fields. Application modules import this shared logger instead of creating module-specific loggers. Python's `logging.basicConfig()` leaves an existing host application's logging configuration in place.

The market-analysis path is:

1. `app.py` initializes the model manager, which loads available checkpoints, then starts Uvicorn in the main process. FastAPI prediction requests call the async `model/prediction_engine.py` orchestration directly.
2. `model/prediction_engine.py` resolves the prediction date, obtains prior-session history from `model/dataStore.py`, fetches target-day candles when they were not supplied, and runs inference on the merged frame. Candle conversion and dataframe/inference work are offloaded to worker threads.
3. `model/features.py` builds the multi-timeframe model features. `model/prediction_engine.py` gets probabilities from loaded model instances and blends probabilities when both models are ready. `model/models.py` owns in-memory model instances and all loading, unloading, backup, and reload operations.
4. The shared OpenAlgo history, holiday-calendar, and analyser-status client methods offload synchronous SDK calls to worker threads. Non-empty successful holiday calendars are cached in memory by year; a cache miss fetches that year, while failed, invalid, or empty responses are not cached so later calls can retry.

`POST /api/brain/predict` accepts structured ticker, exchange, lookback, and interval fields; supplied candle arrays are validated by `OhlcvCandle` in `openalgo/openalgoTypes.py`. The API process fetches candles and runs inference directly. The current engine supports `1m` source candles only. `POST /api/openalgo/history` fetches broker history through the shared OpenAlgo client and returns symbol, exchange, interval, and candles together in `data.data`; `GET /api/openalgo/analyser-status` queries analyser availability. All endpoints return `UniversalServerResponse` with an intent and output `eventType`; success data is stored in `data.data` and errors in `data.error`. `server/serverTypes.py` owns the generic response envelope, module-related domain types belong in their module's `*Types.py` file, and route-specific response contracts are defined with their routes. `model/modelTypes.py` and `openalgo/openalgoTypes.py` contain domain types and request/event literals, while `universalTypes.py` contains global intent/event types. Starting `app.py` loads the prediction models and runs FastAPI on port 3010 in the main process. The server allows browser requests from origins listed in `TRADE_AI_CHAT_ORIGINS`, defaulting to `http://localhost:3011,http://127.0.0.1:3011`.

## Data and Feature Contract

- Source data is one-minute candles timestamped at candle open. Expected columns are `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
- Candle columns use the lowercase schema `timestamp`, `open`, `high`, `low`, `close`, and `volume`. Training CSV caches, downloaded histories, and supplied candles pass through `util.merge_ohlcv_history()`, which normalizes OHLCV column names, promotes a timestamp-like index when the timestamp column is absent, converts offset-aware timestamps to Asia/Kolkata, treats naive timestamps as Asia/Kolkata local time, interprets Unix-second and Unix-millisecond timestamps as UTC instants, removes timezone metadata after conversion, stable-sorts chronologically, and keeps the later-appended row for duplicate timestamps. This permits history and caller candles with different timestamp timezone formats to merge consistently. The feature builder also sorts direct inputs before calculating returns.
- The `/predict` route anchors its inference frame to the latest supplied candle date, or today if no candles were supplied, and includes at least 30 prior trading sessions before inference. Prediction-day candles are merged into a transient frame and are not saved in `dataStore`.
- `model/features.py` builds the model-specific feature frames used by training and inference. Raw OHLCV remains in feature frames for calculations but is excluded from model inputs. See `src/trade_ai/model/features.md` for the feature counts, names, active timeframes, and scaling details. Source columns are selected in canonical order; unrelated extra input columns are excluded.
- `model/dataStore.py` provides the process-wide `dataStore` (`CandleStore`) singleton for pandas OHLCV history frames keyed by normalized `TICKER:EXCHANGE`. `get_history_before()` asynchronously returns the requested number of `1m` trading sessions ending before a target date. It checks cached date coverage and fetches requested history through OpenAlgo using the exchange holiday calendar when needed. For prediction, the cache is replaced with only the requested prior-session history; prediction-day candles are merged transiently for inference and are not stored. The event test and prediction routes use the store. It is cleared when the process exits.
- The prediction store verifies every expected exchange trading date in the requested history range. It logs intraday timestamp gaps greater than one minute with their date, surrounding times, and duration (up to 25 details per warning); such gaps may also reflect periods with no trades and are not automatically filled with synthetic candles.
- Multi-timeframe bars are mapped back with backward `merge_asof` joins to use completed bars only. Numeric minute timeframe labels such as `1m` and `5m` are translated to pandas' unambiguous `min` resampling alias; feature names retain their existing timeframe labels.
- Feature calculation runs on the full chronological history, allowing prior sessions to warm up higher-timeframe indicators such as EMA 50 on 1h bars. GRU sequence windows are separately restricted to one calendar date.
- The 30-day buffer is not a mathematical guarantee that an EMA is fully converged. The user-facing market close is 15:15, with candle timestamps at candle open; the 15:14 candle is the last eligible candle and 15:15 is excluded.

## Labels and Training

- `model/training/labels.py` creates three mutually exclusive targets: `-1` stop-loss first, `0` timeout, `1` take-profit first. Its shared 60-bar maximum holding-period constant is used by both trainers for label look-ahead and purge size. It checks future candle closes, uses absolute ATR in price units from the previous completed 1-minute candle for barrier distances, allows entries from 10:00 through 14:30 by default, and stops at the entry date boundary or 15:15 market close. Inputs must already be sorted chronologically. Label schema version 2 fixes prior labels that mistakenly used relative ATR as a price distance; checkpoints without the current label schema version are rejected.
- Label generation precomputes time-of-day and session-date arrays to reduce repeated pandas scalar access, while retaining the same barrier rules. It logs progress every 50,000 eligible entries.
- `model/training/train_lgb.py` loads each ticker's cached CSV or downloads history from 2021-11-01 through 2026-07-31 in inclusive 30-calendar-day API chunks. It calculates LGB features and labels, then makes a chronological 80/20 split per ticker. It purges up to 60 feature rows before each validation boundary so training labels cannot inspect validation-period candles. It combines the per-ticker splits, maps classes to `0/1/2`, logs train/validation row and class counts plus their multi-logloss during training, and trains up to 500 LightGBM rounds with validation-based early stopping. It keeps a new checkpoint only when its validation logloss beats the training class-prior baseline and records the selected iteration and metrics in the schema sidecar. Since each ticker's cutoff is based on its own labeled-row count, calendar cutoff dates can differ across tickers.
- `model/training/train_gru.py` trains a two-layer, 64-hidden-unit, three-class GRU. Its epoch limit must be 5–20; the orchestrated run allows up to 20 epochs, trains at least 5, then stops after 3 consecutive epochs without improved validation loss. It uses chronological per-ticker 80/20 splits and purges up to 60 feature rows before each validation boundary. Same-session sliding windows must end on labeled rows, though unlabeled feature rows can serve as context. It logs per-ticker and aggregate training/validation loss, accuracy, sequence counts, and class counts, and saves the checkpoint with the best validation loss. Features, labels, splits, and windows are prepared once per ticker and cached in memory across epochs; validation is still evaluated one ticker at a time.
- Both trainers use `src/trade_ai/model/training/tickData/<TICKER>_history.csv` when it exists. Cached rows are normalized, timestamp-parsed, sorted oldest-to-newest, and deduplicated by timestamp when loaded; caches are not automatically refreshed. If missing, the trainers await the asynchronous OpenAlgo client and download distinct inclusive 30-calendar-day chunks, allowing up to three attempts per chunk. A chunk that still fails aborts that ticker's download instead of training on partial history. Successful downloads are normalized, sorted, deduplicated, and cached.
- `model/training/train_master.py` accepts `gru`, `lgb`, or `both` (default). It backs up loaded checkpoint files to timestamped copies, unloads the selected models, and runs their training functions. It reloads the selected checkpoints only if every selected trainer returns successfully; after a training exception, those models stay unloaded in the current process until explicitly reloaded or the app restarts. Run it with `uv run python -m trade_ai.model.training.train_master [gru|lgb|both]`; see `model/training/training.md` for the step-by-step lifecycle.
- Model files are written under `src/trade_ai/model/.run_cache/` with the feature schema version in each filename: `lgb_trading_model_v{FEATURE_SCHEMA_VERSION}.txt` with a `.schema.json` sidecar, and `gru_trading_model_v{FEATURE_SCHEMA_VERSION}.pth` containing its feature schema metadata and state dictionary. Both loaders validate schema version and their model-specific ordered input columns; incompatible checkpoints must be retrained. Checkpoints in the old raw GRU state-dict format or without a LightGBM schema sidecar must also be retrained.

## Prediction Behavior and Known Limits

- `model/models.py` loads each existing checkpoint when the application starts. It infers GRU input width from the checkpoint, so GRU loading does not depend on a first prediction or cached candle data. Missing files produce a warning to train that model first.
- `model/prediction_engine.py` only requests loaded instances from `models.py`, performs inference, and blends available model probabilities. Before returning a successful prediction to the route, it logs the ticker, signal, confidence, processed candle count, latest candle timestamp, and model probability breakdown. `server/predict_route.py` logs the response result and breakdown or the error immediately before returning its envelope. The prediction engine does not load, unload, back up, or reload models.
- If both models return probabilities, the engine takes a 50/50 arithmetic mean. If only one returns probabilities, it uses that model alone. It returns the class with the largest probability (`-1`, `0`, or `1`) and that probability as confidence. Probabilities are not calibrated confidence guarantees.
- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `model/training/gru_sequence_builder.py`.
- `gru_sequence_builder.latest_session_sequence()` selects the last requested number of valid feature rows from the newest calendar date. It does not verify market hours or that timestamps are exactly one minute apart. Training windows likewise enforce calendar-date boundaries, not missing-candle continuity.
- `GRUTradingBrain` is defined in `model/models.py` and shared by the model manager and GRU training module.
- Training is orchestrated through `train_master.py`; it backs up current selected checkpoints and unloads selected models before training. It reloads selected checkpoints only after every selected trainer returns successfully; if training raises, selected models remain unloaded in the current process until explicitly reloaded or the app restarts.
- There is no liquidity/spread filter, probability calibration, automatic order submission, or documented 10-million-row processing loop in the active pipeline.

## Structured Brain API

- `GET /api/event/test` accepts `ticker` and optional `exchange` query parameters. `exchange` defaults to `NSE`; the route returns the trimmed, uppercased ticker and cached candle history as a JSON array of row objects in a `SERVER/SERVER_STATUS` response.
- `POST /api/brain/predict` accepts explicit `ticker`, `exchange`, `days_back`, and `interval` fields. Exchange is `NSE` or `BSE`; the current inference pipeline accepts `1m` only. `days_back` is limited to 90 sessions; values below 30 use 30 sessions.
- It can use either OpenAlgo history or caller-supplied `candles` containing timestamp and lowercase OHLCV fields. Supplied arrays must contain 1–25,000 candles and are validated before inference.
- The route returns a `MODEL/PREDICTION` envelope. Its response data contains the signal, confidence, processed candle count, `prediction_timestamp` for the latest candle included in inference, and a `breakdown` with LightGBM, GRU, and ensemble probability arrays in class order `[-1, 0, 1]`; unavailable model arrays are `null`. Request validation errors use HTTP 422; insufficient input uses 422, unavailable models use 503, upstream history failures use 502, and unexpected errors use 500. Each error remains a string in `data.error`.
- Invalid requests also return the same envelope with a string error and HTTP 422.

## Source Map

- `src/trade_ai/app.py`: initializes the model manager and runs the API in the main process.
- `src/trade_ai/environmentLoader.py`: loads root `.env` settings and provides configured application variables as JSON for internal callers.
- `src/trade_ai/logger.py`: configures and exports the shared project logger.
- `src/trade_ai/model/features.py` and `src/trade_ai/model/training/gru_sequence_builder.py`: feature schema and same-session GRU windows.
- `src/trade_ai/model/training/labels.py`: Triple Barrier target generation used by both trainers.
- `src/trade_ai/model/training/`: LightGBM and GRU trainers, master orchestration, and `training.md` lifecycle notes.
- `src/trade_ai/model/models.py`: GRU architecture, in-memory model manager, loading, unloading, timestamped checkpoint backup, and reload.
- `src/trade_ai/model/prediction_engine.py`: prior-history and target-candle orchestration plus read-only ensemble inference using model instances from `models.py`.
- `src/trade_ai/openalgo/openalgo_client.py`, `src/trade_ai/openalgo/helper.py`, `src/trade_ai/util.py`, `src/trade_ai/model/stock_metadata.py`: shared client, OpenAlgo response conversion, candle utilities, metadata inputs.
- The OpenAlgo SDK's `analyzerstatus()` method is synchronous; call it without `await` and normalize its dictionary response in `OpenAlgo.get_analyzer_status()`.
- `src/trade_ai/indicator/`: indicator calculations and their explanatory Markdown files.
- `src/trade_ai/server/`: FastAPI application, `predict_route.py` prediction route, and `openalgo_route.py` history and analyser status routes.
- `src/trade_ai/server/event_route.py`: event test route and its ticker/exchange query model.
- `src/trade_ai/server/serverTypes.py`: generic `UniversalServerResponse` envelope and response-data wrapper.
- `src/trade_ai/model/modelTypes.py`, `src/trade_ai/openalgo/openalgoTypes.py`, and `src/trade_ai/universalTypes.py`: internal domain/request types, event literals, and global intent/event literals. `modelTypes.py` owns the `ModelName` literal and `MODEL_NAMES` tuple; `openalgoTypes.py` owns the shared `OHLCVCandleDataType` Pydantic model and `OhlcvCandle`, which inherits its fields and adds prediction-input validation. Routes and model code import these module types when needed.

## Environment and Commands

- Python 3.14 or newer and `uv` are required.
- LightGBM's native library needs the GNU OpenMP runtime (`libgomp1` on Debian/Ubuntu).
- `APPLICATION_ENVIRONMENT_VARIABLES` in `environmentLoader.py` lists `OPENALGO_API_KEY`, `OPENALGO_HOST`, and `TRADE_AI_CHAT_ORIGINS`. The app loads the repository-root `.env` and validates that every listed value is present and nonblank before importing startup components, preserving existing process environment values. `load_environment()` returns configured application variables as a dictionary, and `get_environment_json()` serializes them for internal callers. The shared OpenAlgo client also invokes the loader before initialization for non-app entry points.
- Install dependencies with `uv sync`.
- Start the app with `uv run python src/trade_ai/app.py` (FastAPI 3010; models load in the main process).
- Run selected model training in the foreground with `uv run python -m trade_ai.model.training.train_master [gru|lgb|both]` (default: both).
- No test command is currently configured in `pyproject.toml`.
