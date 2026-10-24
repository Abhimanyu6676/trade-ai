import json
import logging
import os
from datetime import date, timedelta

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from trade_ai.model.models import LGB_MODEL_PATH, LGB_MODEL_SCHEMA_PATH
from trade_ai.model.stock_metadata import STOCK_METADATA
from trade_ai.openalgo.openalgo_client import TYPE_OPENALGO_EXCHANGE, openalgo
from trade_ai.util import merge_ohlcv_history

from ..features import (
    FEATURE_SCHEMA_VERSION,
    LGB_MODEL_INPUT_COLUMNS,
    compute_normalized_features,
)
from .labels import apply_triple_barrier

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
    """Downloads historical 1-minute candles in date chunks to prevent API timeouts

    and saves/returns the combined dataset.
    """
    current_start = start_date
    all_chunks: list[pd.DataFrame] = []

    logger.info(
        "[%s] Downloading historical data from %s to %s via OpenAlgo...",
        ticker,
        start_date.isoformat(),
        end_date.isoformat(),
    )

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        success = False
        for attempt in range(retries):
            try:
                df_chunk = openalgo.get_history(
                    symbol=ticker,
                    exchange=exchange,
                    interval="1m",
                    start_date=start_date,
                    end_date=end_date,
                )
                if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                    all_chunks.append(df_chunk)
                success = True
                break
            except Exception as e:
                logger.warning(
                    "[%s] Retry %d failed for range %s to %s: %s",
                    ticker,
                    attempt + 1,
                    start_date,
                    end_date,
                    e,
                )
                import time

                time.sleep(2)

        if not success:
            logger.error(
                "[%s] Failed chunk %s to %s after %d attempts.",
                ticker,
                start_date,
                end_date,
                retries,
            )

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


def prepare_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Iterates through all stocks, builds features and triple barrier labels,

    drops warm-up/lookahead NaNs, and concatenates everything into a single
    monolithic dataset for LightGBM.
    """
    all_features = []
    all_targets = []

    tickers = list(STOCK_METADATA.keys())

    for ticker in tickers:
        logger.info("Processing feature pipeline for %s...", ticker)
        raw_df = load_or_fetch_stock_data(ticker)
        if raw_df is None or raw_df.empty:
            logger.warning("Skipping %s: No data available.", ticker)
            continue

        # 1. Build multi-timeframe features
        df_feat = compute_normalized_features(raw_df, ticker, model_type="lgb")

        # 3. Generate Triple Barrier target labels (-1, 0, 1) with time filters
        df_feat["target"] = apply_triple_barrier(
            df=df_feat,
            timestamp_col="timestamp",
            close_col="close",
            atr_col="ATR_14_1m",
            tp_multiplier=2.0,
            sl_multiplier=1.0,
            max_holding_period=60,
            start_hour=10,
            start_minute=0,
            end_hour=14,
            end_minute=30,
        )

        # 4. Drop rows with NaN targets (warm-up period + trading hour filters + end-of-file)
        df_clean = df_feat.dropna(subset=["target"]).copy()
        if df_clean.empty:
            logger.warning(
                "Skipping %s: Insufficient rows after cleaning NaNs.", ticker
            )
            continue

        # Select the canonical feature schema and target separately.
        feature_cols = list(LGB_MODEL_INPUT_COLUMNS)

        all_features.append(df_clean[feature_cols])
        # Remap Triple Barrier classes from {-1, 0, 1} to {0, 1, 2} for LightGBM multiclass objective
        mapped_targets = df_clean["target"].astype(int) + 1
        all_targets.append(mapped_targets)

    if not all_features:
        raise RuntimeError(
            "Training aborted: No valid data processed across any ticker."
        )

    # Concatenate all stocks into a single monolithic matrix for LightGBM
    X_master = pd.concat(all_features, ignore_index=True)
    y_master = pd.concat(all_targets, ignore_index=True)

    logger.info(
        "Master training dataset assembled successfully. Total rows across all"
        " stocks: %d, Features: %d",
        len(X_master),
        X_master.shape[1],
    )
    return X_master, y_master


def train_lightgbm_model():
    """Trains the LightGBM model on the concatenated master dataset and saves artifacts."""
    openalgo.initiate()
    os.makedirs(CACHE_DIR, exist_ok=True)

    logger.info("🚀 Initializing LightGBM Training Pipeline...")
    X, y = prepare_training_data()

    # Chronological or stratified split (using train_test_split with shuffle=False for time series integrity)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
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
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    # Save trained model to disk
    model.save_model(LGB_MODEL_PATH)
    with open(LGB_MODEL_SCHEMA_PATH, "w", encoding="utf-8") as schema_file:
        json.dump(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_columns": list(LGB_MODEL_INPUT_COLUMNS),
            },
            schema_file,
            indent=2,
        )
    logger.info(
        "✅ LightGBM training complete! Model weights saved to: %s", LGB_MODEL_PATH
    )
    return model
