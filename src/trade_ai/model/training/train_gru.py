import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from trade_ai.config.trading_config import SCHEMA_VERSION, Direction, model_artifact_paths
from trade_ai.logger import logger
from trade_ai.model.models import GRUTradingBrain
from trade_ai.model.stock_metadata import STOCK_METADATA
from trade_ai.openalgo.openalgo_client import openalgo

from ..features import (
    FEATURE_SCHEMA_VERSION,
    GRU_MODEL_INPUT_COLUMNS,
    compute_normalized_features,
)
from .gru_sequence_builder import (
    DEFAULT_SEQUENCE_LENGTH,
    session_dates,
    valid_window_starts,
)
from .labeling import (
    LABEL_SCHEMA_VERSION,
    MAX_HOLDING_PERIOD,
    apply_triple_barrier,
)
from .rl.oos_data import write_oos_predictions
from .util import (
    eligible_session_dates,
    global_walk_forward_plan,
    load_or_fetch_stock_data,
)

MIN_EPOCHS = 5
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3


def _atomic_torch_save(checkpoint: dict[str, object], path: Path) -> None:
    """Save a GRU checkpoint in place only after serialization completes."""
    temporary_path: str | None = None
    try:
        with NamedTemporaryFile(
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


# --- 2. Memory-Efficient Sliding Window Dataset ---
class SlidingWindowDataset(Dataset):
    """Generates sliding time-series windows on-the-fly to prevent memory overflow."""

    def __init__(
        self,
        features_np: np.ndarray | torch.Tensor,
        targets_np: np.ndarray,
        session_dates_np: np.ndarray,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):
        self.features = (
            features_np
            if isinstance(features_np, torch.Tensor)
            else torch.tensor(features_np, dtype=torch.float32)
        )
        self.targets = torch.tensor(targets_np, dtype=torch.float32)
        self.seq_len = sequence_length
        if len(session_dates_np) != len(self.features):
            raise ValueError("session dates must align with feature rows")
        window_starts = valid_window_starts(session_dates_np, sequence_length)
        target_indices = window_starts + sequence_length - 1
        labeled_windows = np.isfinite(targets_np[target_indices])
        self.window_starts = window_starts[labeled_windows]

    def class_counts(self) -> dict[int, int]:
        """Count label classes represented by this dataset's sequences."""
        target_indices = self.window_starts + self.seq_len - 1
        labels = (
            self.targets[torch.as_tensor(target_indices, dtype=torch.long)]
            .to(torch.long)
            .numpy()
            + 1
        )
        counts = np.bincount(labels, minlength=3)
        return {-1: int(counts[0]), 0: int(counts[1]), 1: int(counts[2])}

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int):
        start_idx = int(self.window_starts[idx])
        target_idx = start_idx + self.seq_len - 1
        x_seq = self.features[start_idx : target_idx + 1]
        # Convert Triple Barrier classes {-1, 0, 1} to CrossEntropy classes {0, 1, 2}.
        y_val = self.targets[target_idx].to(torch.long) + 1
        return x_seq, y_val


GRUDatasetPartition = dict[str, tuple[SlidingWindowDataset, pd.DataFrame]]


def _collect_gru_partitions(
    direction: Direction,
    feature_cols: list[str],
    sequence_length: int,
    requests: dict[
        str,
        tuple[pd.Timestamp | None, pd.Timestamp | None, bool, bool],
    ],
) -> dict[str, GRUDatasetPartition]:
    """Build named chronological datasets across every configured ticker.

    Requests use (start_date, end_date, purge_before_end, include_metadata).
    A single feature tensor is shared by all requested ranges for each ticker.
    """
    partitions: dict[str, GRUDatasetPartition] = {name: {} for name in requests}
    for ticker in STOCK_METADATA:
        raw_frame = load_or_fetch_stock_data(ticker)
        if raw_frame is None or raw_frame.empty:
            logger.warning("Skipping %s: data unavailable.", ticker)
            continue
        feature_frame = compute_normalized_features(
            raw_frame, ticker, model_type="gru"
        ).reset_index(drop=True)
        feature_frame["target"] = apply_triple_barrier(
            feature_frame,
            direction=direction,
            max_holding_period=MAX_HOLDING_PERIOD,
        )
        del raw_frame

        df_clean = feature_frame.dropna(subset=["timestamp"])
        features_np = df_clean[feature_cols].to_numpy(dtype=np.float32)
        targets_np = pd.to_numeric(
            df_clean["target"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        dates_np = session_dates(df_clean["timestamp"])
        timestamps = pd.to_datetime(df_clean["timestamp"])
        row_indices = np.arange(len(targets_np))
        features_tensor = torch.tensor(features_np, dtype=torch.float32)

        for name, (start_date, end_date, purge_before_end, include_metadata) in requests.items():
            start_row = (
                int(timestamps.searchsorted(start_date)) if start_date is not None else 0
            )
            end_row = (
                int(timestamps.searchsorted(end_date))
                if end_date is not None
                else len(df_clean)
            )
            if purge_before_end:
                end_row = max(start_row, end_row - MAX_HOLDING_PERIOD)
            selected_targets = targets_np.copy()
            selected_targets[
                (row_indices < start_row)
                | (row_indices >= end_row)
                | ~np.isfinite(selected_targets)
            ] = np.nan
            dataset = SlidingWindowDataset(
                features_tensor,
                selected_targets,
                dates_np,
                sequence_length,
            )
            if not len(dataset):
                continue
            metadata = pd.DataFrame()
            if include_metadata:
                target_indices = dataset.window_starts + sequence_length - 1
                metadata = df_clean.iloc[target_indices][
                    ["timestamp", "close", "ATR_14_1m", "Vol_Ratio_10_30_1m"]
                ].copy()
                metadata.insert(0, "ticker", ticker)
            partitions[name][ticker] = (dataset, metadata)
        del feature_frame

    for name, partition in partitions.items():
        logger.info(
            "%s GRU partition %s: sequences=%d tickers=%d",
            direction,
            name,
            sum(len(dataset) for dataset, _ in partition.values()),
            len(partition),
        )
    return partitions


def _fit_gru_fixed_epochs(
    train_partition: GRUDatasetPartition,
    input_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> GRUTradingBrain:
    """Fit a fresh GRU for a fixed epoch count on all available train tickers."""
    torch.manual_seed(seed)
    model = GRUTradingBrain(
        input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for ticker in sorted(train_partition):
            dataset, _ = train_partition[ticker]
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
                total_samples += len(batch_y)
        if not total_samples:
            raise RuntimeError("GRU refit has no labeled training sequences")
        logger.info(
            "Fixed-epoch GRU refit epoch %d/%d: loss=%.4f sequences=%d",
            epoch + 1,
            epochs,
            total_loss / total_samples,
            total_samples,
        )
    model.eval()
    return model


def _predict_gru_partition(
    model: GRUTradingBrain,
    partition: GRUDatasetPartition,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return probabilities and endpoint metadata in matching ticker order."""
    metadata_parts: list[pd.DataFrame] = []
    probability_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for ticker in sorted(partition):
            dataset, metadata = partition[ticker]
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, drop_last=False
            )
            batches: list[np.ndarray] = []
            for batch_X, _ in loader:
                logits = model(batch_X.to(device))
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            if batches:
                metadata_parts.append(metadata)
                probability_parts.append(np.concatenate(batches, axis=0))
    if not metadata_parts:
        raise RuntimeError("GRU prediction partition produced no sequences")
    return (
        pd.concat(metadata_parts, ignore_index=True),
        np.concatenate(probability_parts, axis=0),
    )


def _evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, int, int]:
    """Return sample-weighted cross-entropy, correct predictions, and count."""
    total_loss = 0.0
    correct = 0
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            total_loss += criterion(logits, batch_y).item() * len(batch_y)
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            sample_count += len(batch_y)
    return (
        total_loss / sample_count if sample_count else float("nan"),
        correct,
        sample_count,
    )


# --- 3. Main GRU Training Pipeline ---
def train_gru_model(
    direction: Direction = "long",
    epochs: int = MAX_EPOCHS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_size: int = 256,
    learning_rate: float = 0.001,
):
    """Select GRU epochs, create expanding OOS forecasts, then refit for inference."""
    if not MIN_EPOCHS <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
    if sequence_length < 1 or batch_size < 1:
        raise ValueError("sequence_length and batch_size must be at least 1")

    model_path, _ = model_artifact_paths(direction, "gru")
    openalgo.initiate()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing %s GRU training pipeline on device: %s", direction, device)

    tickers = list(STOCK_METADATA.keys())
    feature_cols = list(GRU_MODEL_INPUT_COLUMNS)
    torch.manual_seed(42)
    model = GRUTradingBrain(
        input_dim=len(feature_cols), hidden_dim=64, num_layers=2, num_classes=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_epoch = 0
    criterion = nn.CrossEntropyLoss()

    # One shared date plan covers all tickers; ticker rows are never randomized
    # into different time periods.
    all_session_dates: set[pd.Timestamp] = set()
    for ticker in tickers:
        logger.info("Collecting eligible session dates for %s...", ticker)
        raw_frame = load_or_fetch_stock_data(ticker)
        if raw_frame is None or raw_frame.empty:
            logger.warning("Skipping %s: data unavailable.", ticker)
            continue
        all_session_dates.update(eligible_session_dates(raw_frame))
    if not all_session_dates:
        raise RuntimeError("No eligible GRU training session dates were prepared.")
    plan = global_walk_forward_plan(all_session_dates)
    selector_partitions = _collect_gru_partitions(
        direction,
        feature_cols,
        sequence_length,
        {
            "fit": (None, plan.validation_start, True, False),
            "validation": (
                plan.validation_start,
                plan.oos_start,
                True,
                False,
            ),
        },
    )
    selector_train = selector_partitions["fit"]
    selector_validation = selector_partitions["validation"]
    if not selector_train or not selector_validation:
        raise RuntimeError("No valid GRU fit and selection datasets were prepared.")

    for epoch in range(epochs):
        logger.info("--- Starting Epoch %d/%d ---", epoch + 1, epochs)
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for ticker in sorted(selector_train):
            train_dataset, _ = selector_train[ticker]
            dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )

            model.train()
            ticker_loss_sum = 0.0
            ticker_correct = 0
            ticker_samples = 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                actual_batch_size = len(batch_y)
                ticker_loss_sum += loss.item() * actual_batch_size
                ticker_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
                ticker_samples += actual_batch_size

            train_loss_sum += ticker_loss_sum
            train_correct += ticker_correct
            train_samples += ticker_samples
            logger.info(
                "[%s] Epoch %d/%d train: loss=%.4f accuracy=%.3f sequences=%d",
                ticker,
                epoch + 1,
                epochs,
                ticker_loss_sum / ticker_samples,
                ticker_correct / ticker_samples,
                ticker_samples,
            )

        if train_samples == 0:
            raise RuntimeError(f"No valid GRU training sequences in epoch {epoch + 1}.")

        # Validation rows select the epoch count; they are not used to score PPO.
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0
        for ticker in sorted(selector_validation):
            val_dataset, _ = selector_validation[ticker]
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
            )
            ticker_val_loss, ticker_val_correct, ticker_val_samples = _evaluate_dataset(
                model, val_loader, criterion, device
            )
            val_loss_sum += ticker_val_loss * ticker_val_samples
            val_correct += ticker_val_correct
            val_samples += ticker_val_samples
            logger.info(
                "[%s] Epoch %d/%d validation: loss=%.4f accuracy=%.3f sequences=%d",
                ticker,
                epoch + 1,
                epochs,
                ticker_val_loss,
                ticker_val_correct / ticker_val_samples,
                ticker_val_samples,
            )

        if val_samples == 0:
            raise RuntimeError("No valid GRU validation sequences were available.")

        train_loss = train_loss_sum / train_samples
        train_accuracy = train_correct / train_samples
        val_loss = val_loss_sum / val_samples
        val_accuracy = val_correct / val_samples
        logger.info(
            "Epoch %d/%d summary: train_loss=%.4f train_accuracy=%.3f "
            "train_sequences=%d val_loss=%.4f val_accuracy=%.3f val_sequences=%d",
            epoch + 1,
            epochs,
            train_loss,
            train_accuracy,
            train_samples,
            val_loss,
            val_accuracy,
            val_samples,
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch + 1
            logger.info(
                "Selected GRU epoch %d (validation loss %.4f).",
                best_epoch,
                best_validation_loss,
            )

        if (
            epoch + 1 >= MIN_EPOCHS
            and epoch + 1 - best_epoch >= EARLY_STOPPING_PATIENCE
        ):
            logger.info(
                "Early stopping at epoch %d: validation loss did not improve for "
                "%d epochs; best epoch was %d (loss %.4f).",
                epoch + 1,
                EARLY_STOPPING_PATIENCE,
                best_epoch,
                best_validation_loss,
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            "GRU training produced no checkpoint; no valid validation result was available."
        )

    logger.info(
        "GRU epoch selection completed: best_epoch=%d validation_loss=%.4f",
        best_epoch,
        best_validation_loss,
    )

    oos_metadata_parts: list[pd.DataFrame] = []
    oos_probability_parts: list[np.ndarray] = []
    for fold_index, (fold_start, fold_end) in enumerate(plan.oos_folds, start=1):
        fold_partitions = _collect_gru_partitions(
            direction,
            feature_cols,
            sequence_length,
            {
                "fit": (None, fold_start, True, False),
                "oos": (fold_start, fold_end, False, True),
            },
        )
        fold_model = _fit_gru_fixed_epochs(
            fold_partitions["fit"],
            len(feature_cols),
            best_epoch,
            batch_size,
            learning_rate,
            device,
            seed=42 + fold_index,
        )
        fold_metadata, fold_probabilities = _predict_gru_partition(
            fold_model,
            fold_partitions["oos"],
            batch_size,
            device,
        )
        fold_metadata["fold_id"] = f"fold_{fold_index}"
        fold_metadata["model_training_cutoff"] = fold_start.date().isoformat()
        oos_metadata_parts.append(fold_metadata)
        oos_probability_parts.append(fold_probabilities)
        logger.info(
            "%s GRU walk-forward fold %d/%d: fit tickers=%d OOS rows=%d "
            "OOS dates=%s..%s",
            direction,
            fold_index,
            len(plan.oos_folds),
            len(fold_partitions["fit"]),
            len(fold_metadata),
            fold_start.date(),
            (fold_end - pd.Timedelta(days=1)).date(),
        )

    final_partitions = _collect_gru_partitions(
        direction,
        feature_cols,
        sequence_length,
        {
            "fit": (None, plan.final_holdout_start, True, False),
            "final": (plan.final_holdout_start, None, False, True),
        },
    )
    final_model = _fit_gru_fixed_epochs(
        final_partitions["fit"],
        len(feature_cols),
        best_epoch,
        batch_size,
        learning_rate,
        device,
        seed=42 + len(plan.oos_folds) + 1,
    )
    _atomic_torch_save(
        {
            "trading_schema_version": SCHEMA_VERSION,
            "direction": direction,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "feature_columns": feature_cols,
            "selection_best_epoch": best_epoch,
            "selection_validation_loss": best_validation_loss,
            "final_fit_sequences": int(
                sum(len(dataset) for dataset, _ in final_partitions["fit"].values())
            ),
            "final_fit_cutoff": plan.final_holdout_start.date().isoformat(),
            "walk_forward_folds": len(plan.oos_folds),
            "state_dict": final_model.state_dict(),
        },
        model_path,
    )
    final_metadata, final_probabilities = _predict_gru_partition(
        final_model,
        final_partitions["final"],
        batch_size,
        device,
    )
    final_metadata["fold_id"] = "final_holdout"
    final_metadata["model_training_cutoff"] = (
        plan.final_holdout_start.date().isoformat()
    )

    write_oos_predictions(
        direction,
        "gru",
        pd.concat(oos_metadata_parts, ignore_index=True),
        np.concatenate(oos_probability_parts, axis=0),
        partition="oos",
    )
    write_oos_predictions(
        direction,
        "gru",
        final_metadata,
        final_probabilities,
        partition="final",
    )
    logger.info(
        "Saved refit %s GRU checkpoint through %s after selecting %d epochs: %s",
        direction,
        plan.final_holdout_start.date(),
        best_epoch,
        model_path,
    )
