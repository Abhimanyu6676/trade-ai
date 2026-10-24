import asyncio
import json
import logging
import os
from datetime import date, timedelta
from time import sleep

import lightgbm as lgb
import pandas as pd

from trade_ai.model.models import LGB_MODEL_PATH, LGB_MODEL_SCHEMA_PATH
from trade_ai.model.stock_metadata import STOCK_METADATA
from trade_ai.openalgo.openalgo_client import TYPE_OPENALGO_EXCHANGE, openalgo
from trade_ai.util import merge_ohlcv_history

from ..features import (
    FEATURE_SCHEMA_VERSION,
    LGB_MODEL_INPUT_COLUMNS,
    compute_normalized_features,
)
from .labels import MAX_HOLDING_PERIOD, apply_triple_barrier

# Configure paths and logging
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_stock_history_in_chunks(
    ticker: str,
    exchange: TYPE_OPENALGO_EXCHANGE = "NSE",
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2026, 7, 31),
    chunk_days: int = 30,
    retries: int = 3,
) -> pd.DataFrame | None:
    """Download historical 1-minute candles in inclusive date chunks."""
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    current_start = start_date
    all_chunks: list[pd.DataFrame] = []

    logger.info(
        "[%s] Downloading historical data from %s to %s via OpenAlgo...",
        ticker,
        start_date.isoformat(),
        end_date.isoformat(),
    )

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)

        success = False
        for attempt in range(retries):
            try:
                df_chunk = asyncio.run(
                    openalgo.get_history(
                        symbol=ticker,
                        exchange=exchange,
                        interval="1m",
                        start_date=current_start,
                        end_date=current_end,
                    )
                )
                if not isinstance(df_chunk, pd.DataFrame):
                    error_message = (
                        df_chunk.get("message", "non-DataFrame response")
                        if isinstance(df_chunk, dict)
                        else f"unexpected response type {type(df_chunk).__name__}"
                    )
                    raise RuntimeError(
                        f"OpenAlgo history request failed: {error_message}"
                    )
                if not df_chunk.empty:
                    all_chunks.append(df_chunk)
                success = True
                break
            except Exception as e:
                logger.warning(
                    "[%s] Retry %d failed for range %s to %s: %s",
                    ticker,
                    attempt + 1,
                    current_start,
                    current_end,
                    e,
                )
                if attempt + 1 < retries:
                    sleep(2)

        if not success:
            logger.error(
                "[%s] Failed chunk %s to %s after %d attempts; aborting download.",
                ticker,
                current_start,
                current_end,
                retries,
            )
            return None

        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        return None

    return merge_ohlcv_history(pd.concat(all_chunks))


def load_or_fetch_stock_data(ticker: str) -> pd.DataFrame | None:
    """Loads stock CSV from disk if available.

    If missing (e.g. new stock added to metadata), downloads via OpenAlgo and caches
    it.
    """
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")

    if os.path.exists(stock_csv):
        logger.info("[%s] Loading local dataset from disk: %s", ticker, stock_csv)
        return merge_ohlcv_history(pd.read_csv(stock_csv))

    logger.info("[%s] Local cache missing. Fetching new data...", ticker)
    df = download_stock_history_in_chunks(ticker)
    if df is not None and not df.empty:
        df.to_csv(stock_csv, index=False)
        logger.info("[%s] Saved newly downloaded history to %s", ticker, stock_csv)
        return df

    return None


def prepare_training_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Iterates through all stocks, builds features and triple barrier labels,

    splits each stock chronologically, purges the label look-ahead window, and
    remaps classes to {0, 1, 2} for LightGBM.
    """
    all_train_features = []
    all_val_features = []
    all_train_targets = []
    all_val_targets = []

    tickers = list(STOCK_METADATA.keys())

    for ticker in tickers:
        logger.info("Processing feature pipeline for %s...", ticker)
        raw_df = load_or_fetch_stock_data(ticker)
        if raw_df is None or raw_df.empty:
            logger.warning("Skipping %s: No data available.", ticker)
            continue

        # 1. Build multi-timeframe features
        df_feat = compute_normalized_features(raw_df, ticker, model_type="lgb")

        # Reset to a positional index so the split purge is measured in candle rows.
        df_feat = df_feat.reset_index(drop=True)

        # 3. Generate Triple Barrier target labels (-1, 0, 1) with time filters
        df_feat["target"] = apply_triple_barrier(
            df=df_feat,
            timestamp_col="timestamp",
            close_col="close",
            atr_col="ATR_14_1m",
            tp_multiplier=2.0,
            sl_multiplier=1.0,
            max_holding_period=MAX_HOLDING_PERIOD,
            start_hour=10,
            start_minute=0,
            end_hour=14,
            end_minute=30,
        )

        # 4. Drop rows with NaN targets
        df_clean = df_feat.dropna(subset=["target"]).copy()
        if df_clean.empty:
            logger.warning(
                "Skipping %s: Insufficient rows after cleaning NaNs.", ticker
            )
            continue

        feature_cols = list(LGB_MODEL_INPUT_COLUMNS)

        # Use each ticker's own chronological boundary so validation includes all
        # tickers instead of depending on their concatenation order.
        split_idx = int(len(df_clean) * 0.8)
        if split_idx == 0 or split_idx == len(df_clean):
            logger.warning(
                "Skipping %s: not enough labeled rows for both train and validation.",
                ticker,
            )
            continue

        validation_start_row = int(df_clean.index[split_idx])
        train_end_row = validation_start_row - MAX_HOLDING_PERIOD
        train_df = df_clean.loc[df_clean.index < train_end_row]
        val_df = df_clean.iloc[split_idx:]
        if train_df.empty or val_df.empty:
            logger.warning(
                "Skipping %s: not enough rows after purging the %d-candle label "
                "look-ahead window.",
                ticker,
                MAX_HOLDING_PERIOD,
            )
            continue

        all_train_features.append(train_df[feature_cols])
        all_val_features.append(val_df[feature_cols])
        all_train_targets.append(train_df["target"].astype(int) + 1)
        all_val_targets.append(val_df["target"].astype(int) + 1)

    if not all_train_features:
        raise RuntimeError(
            "Training aborted: No valid data processed across any ticker."
        )

    # Concatenate per-stock chronological horizons into global master sets
    X_train = pd.concat(all_train_features, ignore_index=True)
    X_val = pd.concat(all_val_features, ignore_index=True)
    y_train = pd.concat(all_train_targets, ignore_index=True)
    y_val = pd.concat(all_val_targets, ignore_index=True)

    if X_train.empty or X_val.empty:
        raise RuntimeError(
            "Training aborted: the combined train or validation set is empty."
        )

    logger.info(
        "Master training datasets assembled successfully via per-ticker timeline splits."
    )
    logger.info(
        "Train rows: %d, Val rows: %d, Features: %d",
        len(X_train),
        len(X_val),
        X_train.shape[1],
    )

    return X_train, X_val, y_train, y_val


def train_lightgbm_model():
    """Trains the LightGBM model on the concatenated master dataset and saves artifacts."""
    openalgo.initiate()
    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.info("🚀 Initializing LightGBM Training Pipeline...")

    X_train, X_val, y_train, y_val = prepare_training_data()

    def class_counts(targets: pd.Series) -> dict[int, int]:
        return {
            -1: int((targets == 0).sum()),
            0: int((targets == 1).sum()),
            1: int((targets == 2).sum()),
        }

    logger.info(
        "LightGBM dataset summary: train_rows=%d val_rows=%d features=%d "
        "train_classes=%s val_classes=%s",
        len(X_train),
        len(X_val),
        X_train.shape[1],
        class_counts(y_train),
        class_counts(y_val),
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    logger.info("Training LightGBM model across %d estimators...", 500)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "validation"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    train_scores = model.best_score.get("train", {})
    validation_scores = model.best_score.get("validation", {})
    logger.info(
        "LightGBM training summary: best_iteration=%d train_multi_logloss=%s "
        "validation_multi_logloss=%s",
        model.best_iteration,
        f"{train_scores['multi_logloss']:.5f}"
        if "multi_logloss" in train_scores
        else "unavailable",
        f"{validation_scores['multi_logloss']:.5f}"
        if "multi_logloss" in validation_scores
        else "unavailable",
    )

    model.save_model(LGB_MODEL_PATH)
    logger.info("Saved trained LightGBM model artifact to %s", LGB_MODEL_PATH)

    # Save schema mapping details
    schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(LGB_MODEL_INPUT_COLUMNS),
        "objective": "multiclass",
        "class_mapping": {0: -1, 1: 0, 2: 1},
    }
    with open(LGB_MODEL_SCHEMA_PATH, "w", encoding="utf-8") as schema_file:
        json.dump(schema, schema_file, indent=4)
    logger.info(
        "Saved LightGBM model schema configuration to %s", LGB_MODEL_SCHEMA_PATH
    )
    return model
