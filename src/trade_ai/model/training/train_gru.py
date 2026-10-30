import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from trade_ai.logger import logger
from trade_ai.model.models import GRU_MODEL_PATH, GRUTradingBrain
from trade_ai.model.stock_metadata import STOCK_METADATA

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
from .labels import (
    LABEL_SCHEMA_VERSION,
    MAX_HOLDING_PERIOD,
    apply_triple_barrier,
)
from .util import load_or_fetch_stock_data

# Directory and path configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")

MIN_EPOCHS = 5
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3


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


def _prepare_ticker_datasets(
    ticker: str,
    feature_cols: list[str],
    sequence_length: int,
    log_split: bool = False,
) -> tuple[SlidingWindowDataset, SlidingWindowDataset] | None:
    """Build chronological train and validation windows for one ticker."""
    raw_df = load_or_fetch_stock_data(ticker)
    if raw_df is None or raw_df.empty:
        logger.warning("Skipping %s: Data unavailable.", ticker)
        return None

    df_feat = compute_normalized_features(raw_df, ticker, model_type="gru")
    df_feat = df_feat.reset_index(drop=True)
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

    # Keep unlabeled rows as sequence context, while splitting labeled endpoints.
    df_clean = df_feat.dropna(subset=["timestamp"]).copy()
    features_np = df_clean[feature_cols].to_numpy(dtype=np.float32)
    targets_series = pd.to_numeric(df_clean["target"], errors="coerce")
    targets_np = np.asarray(targets_series, dtype=np.float32)
    dates_np = session_dates(df_clean["timestamp"])
    labeled_indices = np.where(np.isfinite(targets_np))[0]
    split_idx = int(len(labeled_indices) * 0.8)
    if split_idx == 0 or split_idx == len(labeled_indices):
        logger.warning(
            "Skipping %s: not enough labeled rows for train and validation.", ticker
        )
        return None

    validation_start_row = int(labeled_indices[split_idx])
    train_end_row = validation_start_row - MAX_HOLDING_PERIOD
    row_indices = np.arange(len(targets_np))
    train_targets = targets_np.copy()
    train_targets[(row_indices >= train_end_row) | ~np.isfinite(train_targets)] = np.nan
    val_targets = targets_np.copy()
    val_targets[(row_indices < validation_start_row) | ~np.isfinite(val_targets)] = (
        np.nan
    )

    # Share features between train and validation datasets to limit memory use.
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    train_dataset = SlidingWindowDataset(
        features_tensor, train_targets, dates_np, sequence_length
    )
    val_dataset = SlidingWindowDataset(
        features_tensor, val_targets, dates_np, sequence_length
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.warning(
            "Skipping %s: no complete train or validation sequences after the "
            "%d-row purge.",
            ticker,
            MAX_HOLDING_PERIOD,
        )
        return None

    if log_split:
        logger.info(
            "[%s] Split: labeled_rows=%d purged_rows=%d train_sequences=%d "
            "val_sequences=%d train_classes=%s val_classes=%s",
            ticker,
            len(labeled_indices),
            MAX_HOLDING_PERIOD,
            len(train_dataset),
            len(val_dataset),
            train_dataset.class_counts(),
            val_dataset.class_counts(),
        )
    return train_dataset, val_dataset


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
    epochs: int = MAX_EPOCHS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_size: int = 256,
    learning_rate: float = 0.001,
):
    """Train with chronological validation and early stopping after the minimum."""
    if not MIN_EPOCHS <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
    if sequence_length < 1 or batch_size < 1:
        raise ValueError("sequence_length and batch_size must be at least 1")

    openalgo.initiate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("🚀 Initializing GRU Training Pipeline on device: %s", device)

    tickers = list(STOCK_METADATA.keys())
    feature_cols = list(GRU_MODEL_INPUT_COLUMNS)
    model = GRUTradingBrain(
        input_dim=len(feature_cols), hidden_dim=64, num_layers=2, num_classes=3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_epoch = 0
    criterion = nn.CrossEntropyLoss()

    # Feature and label preparation is deterministic for a given history. Keep
    # each ticker's datasets in memory so every epoch only performs GRU work.
    prepared_datasets: dict[str, tuple[SlidingWindowDataset, SlidingWindowDataset]] = {}
    for ticker in tickers:
        logger.info("Preparing GRU datasets for %s...", ticker)
        datasets = _prepare_ticker_datasets(
            ticker, feature_cols, sequence_length, log_split=True
        )
        if datasets is not None:
            prepared_datasets[ticker] = datasets

    if not prepared_datasets:
        raise RuntimeError(
            "No valid GRU training and validation datasets were prepared."
        )
    logger.info(
        "Prepared GRU datasets for %d/%d tickers; reusing them across epochs.",
        len(prepared_datasets),
        len(tickers),
    )

    for epoch in range(epochs):
        logger.info("--- Starting Epoch %d/%d ---", epoch + 1, epochs)
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for ticker in tickers:
            datasets = prepared_datasets.get(ticker)
            if datasets is None:
                continue
            train_dataset, _ = datasets
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

        # Evaluate one ticker at a time using its cached validation dataset.
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0
        for ticker in tickers:
            datasets = prepared_datasets.get(ticker)
            if datasets is None:
                continue
            _, val_dataset = datasets
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
            torch.save(
                {
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "label_schema_version": LABEL_SCHEMA_VERSION,
                    "feature_columns": feature_cols,
                    "state_dict": model.state_dict(),
                },
                GRU_MODEL_PATH,
            )
            logger.info(
                "Saved best GRU checkpoint at epoch %d (validation loss %.4f).",
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
        "GRU training completed: best_epoch=%d best_validation_loss=%.4f checkpoint=%s",
        best_epoch,
        best_validation_loss,
        GRU_MODEL_PATH,
    )
