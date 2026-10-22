import logging
import os
from datetime import date, timedelta

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from trade_ai.features import add_multi_timeframe_features
from trade_ai.openalgo_client import openalgo
from trade_ai.stock_metadata import STOCK_METADATA, encode_categorical_metadata
from trade_ai.training_engine.labels import apply_triple_barrier

# Configure paths and logging
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "lgb_trading_model.txt")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_stock_history_in_chunks(
    ticker: str,
    exchange: str = "NSE",
    start_date: date = date(2020, 1, 1),
    end_date: date = date.today(),
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
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        success = False
        for attempt in range(retries):
            try:
                df_chunk = openalgo.get_history(
                    symbol=ticker,
                    exchange=exchange,
                    interval="1m",
                    start_date=start_str,
                    end_date=end_str,
                    source="api",
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
                    start_str,
                    end_str,
                    e,
                )
                import time

                time.sleep(2)

        if not success:
            logger.error(
                "[%s] Failed chunk %s to %s after %d attempts.",
                ticker,
                start_str,
                end_str,
                retries,
            )

        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        return None

    full_df = pd.concat(all_chunks).drop_duplicates()
    # Standardize timestamp column naming if needed
    if "Timestamp" in full_df.columns and "timestamp" not in full_df.columns:
        full_df = full_df.rename(columns={"Timestamp": "timestamp"})

    return full_df.sort_values(by="timestamp").reset_index(drop=True)


def load_or_fetch_stock_data(ticker: str) -> pd.DataFrame | None:
    """Loads stock CSV from disk if available.

    If missing (e.g. new stock added to metadata), downloads via OpenAlgo and caches
    it.
    """
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")

    if os.path.exists(stock_csv):
        logger.info("[%s] Loading local dataset from disk: %s", ticker, stock_csv)
        df = pd.read_csv(stock_csv)
        return df

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

    for ticker in STOCK_METADATA.keys():
        logger.info("Processing feature pipeline for %s...", ticker)
        raw_df = load_or_fetch_stock_data(ticker)
        if raw_df is None or raw_df.empty:
            logger.warning("Skipping %s: No data available.", ticker)
            continue

        # 1. Build multi-timeframe features
        df_feat = add_multi_timeframe_features(raw_df)

        # 2. Encode sector and market cap metadata features
        df_feat = encode_categorical_metadata(df_feat, ticker)

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

        # Separate features (X) and target (y)
        # Exclude non-feature metadata columns like timestamp
        exclude_cols = ["timestamp", "target"]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

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
    model.save_model(MODEL_PATH)
    logger.info("✅ LightGBM training complete! Model weights saved to: %s", MODEL_PATH)
    return model


if __name__ == "__main__":
    train_lightgbm_model()
