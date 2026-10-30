import os
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import lightgbm as lgb
import numpy as np
import pandas as pd

from trade_ai.config.trading_config import (
    SCHEMA_VERSION,
    Direction,
    model_artifact_paths,
)
from trade_ai.logger import logger
from trade_ai.model.stock_metadata import METADATA_FEATURE_COLUMNS, STOCK_METADATA
from trade_ai.openalgo.openalgo_client import openalgo

from ..features import (
    FEATURE_SCHEMA_VERSION,
    LGB_MODEL_INPUT_COLUMNS,
    compute_normalized_features,
)
from .labeling import (
    LABEL_SCHEMA_VERSION,
    MAX_HOLDING_PERIOD,
    apply_triple_barrier,
)
from .rl.oos_data import sha256_file, write_oos_predictions
from .util import (
    atomic_write_json,
    eligible_session_dates,
    global_walk_forward_plan,
    load_or_fetch_stock_data,
    WalkForwardPlan,
)


@dataclass
class LightGBMPartition:
    """One global, direction-specific partition assembled across tickers."""

    features: pd.DataFrame
    targets: pd.Series
    metadata: pd.DataFrame


@dataclass
class PreparedLightGBMData:
    """Selector data and shared date boundaries for one directional model."""

    plan: WalkForwardPlan
    selection_train: LightGBMPartition
    selection_validation: LightGBMPartition


def _walk_forward_plan() -> WalkForwardPlan:
    """Build one date plan from every ticker in the configured universe."""
    all_session_dates: set[pd.Timestamp] = set()
    for ticker in STOCK_METADATA:
        raw_df = load_or_fetch_stock_data(ticker)
        if raw_df is None or raw_df.empty:
            logger.warning("Skipping %s: No data available.", ticker)
            continue
        all_session_dates.update(eligible_session_dates(raw_df))
    if not all_session_dates:
        raise RuntimeError("Training aborted: no eligible candle dates were found.")
    return global_walk_forward_plan(all_session_dates)


def _collect_partitions(
    direction: Direction,
    plan: WalkForwardPlan,
    requests: dict[
        str,
        tuple[pd.Timestamp | None, pd.Timestamp | None, bool, bool],
    ],
) -> dict[str, LightGBMPartition]:
    """Collect named date ranges in one pass over the configured ticker universe.

    Each request is `(start_date, end_date, purge_before_end, keep_metadata)`.
    All date boundaries are shared across tickers; rows are never sampled at
    random. Missing or too-short ticker histories are skipped for that range.
    """
    feature_cols = list(LGB_MODEL_INPUT_COLUMNS)
    features_by_name: dict[str, list[pd.DataFrame]] = {name: [] for name in requests}
    targets_by_name: dict[str, list[pd.Series]] = {name: [] for name in requests}
    metadata_by_name: dict[str, list[pd.DataFrame]] = {name: [] for name in requests}
    tickers_by_name: dict[str, set[str]] = {name: set() for name in requests}

    for ticker in STOCK_METADATA:
        logger.info("Processing %s feature pipeline for %s...", direction, ticker)
        raw_df = load_or_fetch_stock_data(ticker)
        if raw_df is None or raw_df.empty:
            continue
        feature_frame = compute_normalized_features(raw_df, ticker, model_type="lgb")
        feature_frame = feature_frame.reset_index(drop=True)
        feature_frame["target"] = apply_triple_barrier(
            df=feature_frame,
            direction=direction,
            max_holding_period=MAX_HOLDING_PERIOD,
        )
        timestamps = pd.to_datetime(feature_frame["timestamp"])
        row_indices = np.arange(len(feature_frame))
        labeled_mask = feature_frame["target"].notna().to_numpy()
        for name, (start_date, end_date, purge_before_end, keep_metadata) in requests.items():
            start_row = (
                int(timestamps.searchsorted(start_date)) if start_date is not None else 0
            )
            end_row = (
                int(timestamps.searchsorted(end_date))
                if end_date is not None
                else len(feature_frame)
            )
            if purge_before_end:
                end_row = max(start_row, end_row - MAX_HOLDING_PERIOD)
            mask = labeled_mask & (row_indices >= start_row) & (row_indices < end_row)
            selected = feature_frame.loc[mask]
            if selected.empty:
                continue
            tickers_by_name[name].add(ticker)
            features_by_name[name].append(selected[feature_cols])
            targets_by_name[name].append(selected["target"].astype(int) + 1)
            if keep_metadata:
                metadata = selected[
                    ["timestamp", "close", "ATR_14_1m", "Vol_Ratio_10_30_1m"]
                ].copy()
                metadata.insert(0, "ticker", ticker)
                metadata_by_name[name].append(metadata)

    result: dict[str, LightGBMPartition] = {}
    for name in requests:
        if not features_by_name[name]:
            raise RuntimeError(
                f"No LightGBM rows were available for partition {name!r}"
            )
        features = pd.concat(features_by_name[name], ignore_index=True)
        targets = pd.concat(targets_by_name[name], ignore_index=True)
        metadata = (
            pd.concat(metadata_by_name[name], ignore_index=True)
            if metadata_by_name[name]
            else pd.DataFrame()
        )
        if len(features) != len(targets):
            raise RuntimeError(f"LightGBM {name} features and targets are misaligned")
        if requests[name][3] and len(features) != len(metadata):
            raise RuntimeError(f"LightGBM {name} metadata is misaligned")
        result[name] = LightGBMPartition(features, targets, metadata)
        logger.info(
            "%s %s partition: rows=%d tickers=%d dates=%d",
            direction,
            name,
            len(features),
            len(tickers_by_name[name]),
            (
                int(pd.to_datetime(metadata["timestamp"]).dt.normalize().nunique())
                if not metadata.empty
                else 0
            ),
        )
    return result


def prepare_training_data(direction: Direction = "long") -> PreparedLightGBMData:
    """Prepare chronological fit/selection slices for one directional model."""
    plan = _walk_forward_plan()
    partitions = _collect_partitions(
        direction,
        plan,
        {
            "selection_train": (None, plan.validation_start, True, False),
            "selection_validation": (
                plan.validation_start,
                plan.oos_start,
                True,
                False,
            ),
        },
    )
    return PreparedLightGBMData(
        plan,
        partitions["selection_train"],
        partitions["selection_validation"],
    )


def train_lightgbm_model(direction: Direction = "long") -> bool:
    """Train a direction-specific LightGBM model and save its schema artifacts."""
    model_path, schema_path = model_artifact_paths(direction, "lgb")
    openalgo.initiate()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing %s LightGBM training pipeline.", direction)

    prepared = prepare_training_data(direction)
    X_train = prepared.selection_train.features
    y_train = prepared.selection_train.targets
    X_val = prepared.selection_validation.features
    y_val = prepared.selection_validation.targets
    plan = prepared.plan

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

    selection_model = lgb.train(
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

    train_scores = selection_model.best_score.get("train", {})
    validation_scores = selection_model.best_score.get("validation", {})
    logger.info(
        "LightGBM training summary: best_iteration=%d train_multi_logloss=%s "
        "validation_multi_logloss=%s",
        selection_model.best_iteration,
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

    if (
        selection_model.best_iteration < 1
        or validation_logloss is None
        or validation_logloss >= baseline_validation_logloss
    ):
        logger.warning(
            "LightGBM did not beat the class-prior validation baseline; "
            "not replacing the existing checkpoint."
        )
        return False

    selected_iterations = int(selection_model.best_iteration)

    fold_metadata_parts: list[pd.DataFrame] = []
    fold_probability_parts: list[np.ndarray] = []
    for fold_index, (fold_start, fold_end) in enumerate(plan.oos_folds, start=1):
        fold_parts = _collect_partitions(
            direction,
            plan,
            {
                "fit": (None, fold_start, True, False),
                "oos": (fold_start, fold_end, False, True),
            },
        )
        fold_train = fold_parts["fit"]
        fold_oos = fold_parts["oos"]
        fold_model = lgb.train(
            params,
            lgb.Dataset(
                fold_train.features,
                label=fold_train.targets,
                categorical_feature=categorical_features,
            ),
            num_boost_round=selected_iterations,
        )
        fold_metadata = fold_oos.metadata.copy()
        fold_metadata["fold_id"] = f"fold_{fold_index}"
        fold_metadata["model_training_cutoff"] = fold_start.date().isoformat()
        fold_metadata_parts.append(fold_metadata)
        fold_probability_parts.append(
            np.asarray(fold_model.predict(fold_oos.features), dtype=np.float64)
        )
        logger.info(
            "%s LightGBM walk-forward fold %d/%d: fit rows=%d OOS rows=%d "
            "OOS dates=%s..%s",
            direction,
            fold_index,
            len(plan.oos_folds),
            len(fold_train.features),
            len(fold_oos.features),
            fold_start.date(),
            (fold_end - pd.Timedelta(days=1)).date(),
        )

    final_parts = _collect_partitions(
        direction,
        plan,
        {
            "final_fit": (None, plan.final_holdout_start, True, False),
            "final_holdout": (plan.final_holdout_start, None, False, True),
        },
    )
    final_train = final_parts["final_fit"]
    final_holdout = final_parts["final_holdout"]
    model = lgb.train(
        params,
        lgb.Dataset(
            final_train.features,
            label=final_train.targets,
            categorical_feature=categorical_features,
        ),
        num_boost_round=selected_iterations,
    )

    holdout_metadata = final_holdout.metadata.copy()
    holdout_metadata["fold_id"] = "final_holdout"
    holdout_metadata["model_training_cutoff"] = (
        plan.final_holdout_start.date().isoformat()
    )
    holdout_probabilities = np.asarray(
        model.predict(final_holdout.features), dtype=np.float64
    )
    temporary_model_path: str | None = None
    try:
        with NamedTemporaryFile(
            suffix=model_path.suffix,
            dir=model_path.parent,
            delete=False,
        ) as temporary_model:
            temporary_model_path = temporary_model.name
        model.save_model(temporary_model_path)
        os.replace(temporary_model_path, model_path)
    finally:
        if temporary_model_path and os.path.exists(temporary_model_path):
            os.unlink(temporary_model_path)
    logger.info(
        "Saved refit %s LightGBM artifact from %d rows through %s to %s",
        direction,
        len(final_train.features),
        plan.final_holdout_start.date(),
        model_path,
    )

    # Save schema mapping details
    schema = {
        "trading_schema_version": SCHEMA_VERSION,
        "direction": direction,
        "checkpoint_sha256": sha256_file(model_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "feature_columns": list(LGB_MODEL_INPUT_COLUMNS),
        "objective": "multiclass",
        "class_mapping": {0: -1, 1: 0, 2: 1},
        "selection_best_iteration": selected_iterations,
        "final_fit_rows": int(len(final_train.features)),
        "final_fit_cutoff": plan.final_holdout_start.date().isoformat(),
        "walk_forward_folds": len(plan.oos_folds),
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
    atomic_write_json(schema_path, schema)
    logger.info(
        "Saved %s LightGBM model schema configuration to %s", direction, schema_path
    )
    write_oos_predictions(
        direction,
        "lgb",
        pd.concat(fold_metadata_parts, ignore_index=True),
        np.concatenate(fold_probability_parts, axis=0),
        partition="oos",
    )
    write_oos_predictions(
        direction,
        "lgb",
        holdout_metadata,
        holdout_probabilities,
        partition="final",
    )
    return True
