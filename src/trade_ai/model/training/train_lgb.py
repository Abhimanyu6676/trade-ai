import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from trade_ai.logger import logger
from trade_ai.model.models import LGB_MODEL_PATH, LGB_MODEL_SCHEMA_PATH
from trade_ai.model.stock_metadata import METADATA_FEATURE_COLUMNS, STOCK_METADATA

from ..features import (
    FEATURE_SCHEMA_VERSION,
    LGB_MODEL_INPUT_COLUMNS,
    compute_normalized_features,
)
from .labels import (
    LABEL_SCHEMA_VERSION,
    MAX_HOLDING_PERIOD,
    apply_triple_barrier,
)
from .util import load_or_fetch_stock_data

# Configure training paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")


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


def train_lightgbm_model() -> bool:
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

    categorical_features: list[str] = list(METADATA_FEATURE_COLUMNS)
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
        categorical_feature=categorical_features,
    )

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

    validation_logloss = validation_scores.get("multi_logloss")
    train_class_priors = (
        y_train.value_counts(normalize=True)
        .reindex(range(3), fill_value=0.0)
        .to_numpy(dtype=float)
    )
    validation_labels = y_val.to_numpy(dtype=np.int64)
    baseline_validation_logloss = float(
        -np.log(
            np.clip(train_class_priors[validation_labels], 1e-15, 1.0)
        ).mean()
    )
    logger.info(
        "Class-prior baseline validation_multi_logloss=%.5f; "
        "model validation_multi_logloss=%s",
        baseline_validation_logloss,
        (
            f"{validation_logloss:.5f}"
            if validation_logloss is not None
            else "unavailable"
        ),
    )

    if validation_logloss is None or validation_logloss >= baseline_validation_logloss:
        logger.warning(
            "LightGBM did not beat the class-prior validation baseline; "
            "not replacing the existing checkpoint."
        )
        return False

    model.save_model(LGB_MODEL_PATH)
    logger.info("Saved trained LightGBM model artifact to %s", LGB_MODEL_PATH)

    # Save schema mapping details
    schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "feature_columns": list(LGB_MODEL_INPUT_COLUMNS),
        "objective": "multiclass",
        "class_mapping": {0: -1, 1: 0, 2: 1},
        "best_iteration": int(model.best_iteration),
        "train_multi_logloss": (
            float(train_scores["multi_logloss"])
            if "multi_logloss" in train_scores
            else None
        ),
        "validation_multi_logloss": (
            float(validation_scores["multi_logloss"])
            if "multi_logloss" in validation_scores
            else None
        ),
        "class_prior_validation_multi_logloss": baseline_validation_logloss,
    }
    with open(LGB_MODEL_SCHEMA_PATH, "w", encoding="utf-8") as schema_file:
        json.dump(schema, schema_file, indent=4)
    logger.info(
        "Saved LightGBM model schema configuration to %s", LGB_MODEL_SCHEMA_PATH
    )
    return True
