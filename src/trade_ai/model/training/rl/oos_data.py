"""Persist and join leakage-aware base-model validation predictions."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import pandas as pd

from trade_ai.config.trading_config import (
    Direction,
    ModelFamily,
    RL_OOS_DIR,
    SCHEMA_VERSION,
    model_artifact_paths,
)
from trade_ai.model.features import FEATURE_SCHEMA_VERSION
from trade_ai.model.training.labeling import LABEL_SCHEMA_VERSION

PREDICTION_COLUMNS = ("p_stop", "p_timeout", "p_take_profit")
KEY_COLUMNS = ("ticker", "timestamp")
MARKET_COLUMNS = (
    "close",
    "ATR_14_1m",
    "Vol_Ratio_10_30_1m",
)
PROVENANCE_COLUMNS = ("fold_id", "model_training_cutoff")
PredictionPartition = Literal["oos", "final"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
    """Write JSON metadata with an atomic rename to avoid partial manifests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as temp_file:
            temp_path = temp_file.name
            json.dump(data, temp_file, indent=2)
            temp_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def oos_prediction_path(
    direction: Direction,
    family: ModelFamily,
    partition: PredictionPartition = "oos",
) -> Path:
    """Return the selected OOS-fold or final-holdout prediction path."""
    suffix = "oos" if partition == "oos" else "final"
    return RL_OOS_DIR / f"{direction}_{family}_{suffix}.csv"


def write_oos_predictions(
    direction: Direction,
    family: ModelFamily,
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    partition: PredictionPartition = "oos",
) -> Path:
    """Save chronological predictions and final-base-checkpoint provenance."""
    if partition not in ("oos", "final"):
        raise ValueError(f"Unsupported prediction partition: {partition!r}")
    checkpoint_path, _ = model_artifact_paths(direction, family)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot save {direction} {family} OOS predictions without "
            f"checkpoint {checkpoint_path}"
        )
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3 or len(probs) != len(metadata):
        raise ValueError("Prediction probabilities must have shape (rows, 3)")
    if (
        not np.isfinite(probs).all()
        or (probs < 0).any()
        or (probs > 1).any()
        or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError("Prediction probability rows must be finite and sum to 1")
    missing = set((*KEY_COLUMNS, *MARKET_COLUMNS, *PROVENANCE_COLUMNS)).difference(
        metadata.columns
    )
    if missing:
        raise ValueError(f"OOS metadata is missing columns: {sorted(missing)}")

    frame = metadata[
        [*KEY_COLUMNS, *MARKET_COLUMNS, *PROVENANCE_COLUMNS]
    ].copy().reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    for index, column in enumerate(PREDICTION_COLUMNS):
        frame[column] = probs[:, index]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.sort_values(["ticker", "timestamp"], kind="stable")
    if frame.empty:
        raise ValueError("Prediction data is empty after removing invalid rows")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Prediction rows contain duplicate ticker/timestamp keys")

    RL_OOS_DIR.mkdir(parents=True, exist_ok=True)
    path = oos_prediction_path(direction, family, partition)
    temp_path: str | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as temp_file:
            temp_path = temp_file.name
            frame.to_csv(temp_file, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    _atomic_write_json(
        Path(f"{path}.schema.json"),
        {
            "trading_schema_version": SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "direction": direction,
            "family": family,
            "partition": partition,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "rows": int(len(frame)),
            "class_order": [-1, 0, 1],
        },
    )
    return path


def _read_valid_predictions(
    direction: Direction,
    family: ModelFamily,
    partition: PredictionPartition,
) -> pd.DataFrame:
    """Read predictions only when their final base checkpoint still matches."""
    path = oos_prediction_path(direction, family, partition)
    manifest_path = Path(f"{path}.schema.json")
    checkpoint_path, _ = model_artifact_paths(direction, family)
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Missing {direction} {family} {partition} predictions; "
            "train that base model first."
        )
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Missing required checkpoint {checkpoint_path}")
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    expected = {
        "trading_schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "direction": direction,
        "family": family,
        "partition": partition,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Stale {direction} {family} {partition} predictions "
            f"({', '.join(mismatches)}); "
            "retrain the selected base model before training the RL agent."
        )
    if manifest.get("class_order") != [-1, 0, 1]:
        raise RuntimeError(f"{direction} {family} OOS class order is incompatible")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {
        *KEY_COLUMNS,
        *MARKET_COLUMNS,
        *PREDICTION_COLUMNS,
        *PROVENANCE_COLUMNS,
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{path} does not match the current OOS data schema")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError(f"{path} has duplicate ticker/timestamp rows")
    if frame[list(PROVENANCE_COLUMNS)].isna().any().any():
        raise RuntimeError(f"{path} is missing walk-forward provenance")
    if partition == "oos":
        valid_fold_ids = frame["fold_id"].astype(str).str.fullmatch(r"fold_[1-9]\d*")
    else:
        valid_fold_ids = frame["fold_id"].astype(str).eq("final_holdout")
    if not valid_fold_ids.all():
        raise RuntimeError(f"{path} contains invalid fold identifiers")
    training_cutoffs = pd.to_datetime(
        frame["model_training_cutoff"], errors="coerce"
    )
    prediction_dates = frame["timestamp"].dt.strftime("%Y-%m-%d")
    cutoff_dates = training_cutoffs.dt.strftime("%Y-%m-%d")
    if training_cutoffs.isna().any() or (prediction_dates < cutoff_dates).any():
        raise RuntimeError(f"{path} contains a prediction before its training cutoff")
    probabilities = frame[list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise RuntimeError(f"{path} contains invalid probability vectors")
    return frame


def _load_combined_predictions(partition: PredictionPartition) -> pd.DataFrame:
    """Join four schema- and checkpoint-matched prediction streams."""
    streams = [
        (direction, family, _read_valid_predictions(direction, family, partition))
        for direction in ("long", "short")
        for family in ("lgb", "gru")
    ]
    direction, family, combined = streams[0]
    combined = combined.rename(
        columns={column: f"{direction}_{family}_{column}" for column in PREDICTION_COLUMNS}
    )
    for direction, family, frame in streams[1:]:
        suffix = f"_{direction}_{family}"
        frame = frame.rename(
            columns={column: f"{direction}_{family}_{column}" for column in PREDICTION_COLUMNS}
        )
        combined = combined.merge(
            frame[
                [
                    *KEY_COLUMNS,
                    *MARKET_COLUMNS,
                    *PROVENANCE_COLUMNS,
                    *(
                        f"{direction}_{family}_{column}"
                        for column in PREDICTION_COLUMNS
                    ),
                ]
            ],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
            suffixes=("", suffix),
        )
        for shared_column in (*MARKET_COLUMNS, *PROVENANCE_COLUMNS):
            duplicate_column = f"{shared_column}{suffix}"
            if duplicate_column in combined:
                left = combined[shared_column]
                right = combined[duplicate_column]
                values_match = (
                    np.allclose(
                        left.to_numpy(dtype=float),
                        right.to_numpy(dtype=float),
                        rtol=1e-8,
                        atol=1e-10,
                    )
                    if shared_column in MARKET_COLUMNS
                    else left.astype(str).equals(right.astype(str))
                )
                if not values_match:
                    raise RuntimeError(
                        f"Base-model prediction streams disagree on {shared_column}"
                    )
                combined = combined.drop(columns=[duplicate_column])

    probability_columns = [
        f"{direction}_{family}_{column}"
        for direction in ("long", "short")
        for family in ("lgb", "gru")
        for column in PREDICTION_COLUMNS
    ]
    values = combined[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Joined OOS probabilities contain non-finite values")
    if len(combined) == 0:
        raise RuntimeError("The four OOS streams have no shared ticker/timestamp rows")
    combined["session_date"] = combined["timestamp"].dt.date.astype(str)
    return combined.sort_values(["ticker", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def load_combined_oos_predictions() -> pd.DataFrame:
    """Join expanding-fold OOS predictions for PPO fitting and selection."""
    return _load_combined_predictions("oos")


def load_combined_final_predictions() -> pd.DataFrame:
    """Join the newest untouched holdout predictions for final reporting."""
    return _load_combined_predictions("final")
