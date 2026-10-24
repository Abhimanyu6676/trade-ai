import asyncio
import logging
import os
from datetime import date, timedelta
from time import sleep

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from trade_ai.model.models import GRU_MODEL_PATH, GRUTradingBrain
from trade_ai.model.stock_metadata import STOCK_METADATA
from trade_ai.openalgo.openalgo_client import TYPE_OPENALGO_EXCHANGE, openalgo
from trade_ai.util import merge_ohlcv_history

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
from .labels import MAX_HOLDING_PERIOD, apply_triple_barrier

# Directory and path configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MIN_EPOCHS = 10
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


# --- 3. Data Downloader and Ingestion Utilities ---
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
                    "[%s] Chunk retry %d failed for %s to %s: %s",
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
    """Loads local CSV if present; otherwise fetches from OpenAlgo and caches to disk."""
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")

    if os.path.exists(stock_csv):
        logger.info("[%s] Loading local dataset: %s", ticker, stock_csv)
        return merge_ohlcv_history(pd.read_csv(stock_csv))

    logger.info("[%s] Local cache missing. Downloading historical data...", ticker)
    df = download_stock_history_in_chunks(ticker)
    if df is not None and not df.empty:
        df.to_csv(stock_csv, index=False)
        logger.info("[%s] Cached newly fetched history to %s", ticker, stock_csv)
        return df

    return None


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
        atr_col="ATR_14_1m",
        tp_multiplier=2.0,
        sl_multiplier=1.0,
        max_holding_period=MAX_HOLDING_PERIOD,
        start_hour=10,
        start_minute=0,
        end_hour=14,
        end_minute=30,
    )

    # Keep unlabeled rows as sequence context, while splitting labeled endpoints.
    df_clean = df_feat.dropna(subset=["timestamp"]).copy()
    features_np = df_clean[feature_cols].to_numpy(dtype=np.float32)
    targets_np = pd.to_numeric(df_clean["target"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    dates_np = session_dates(df_clean["timestamp"])
    labeled_indices = np.flatnonzero(np.isfinite(targets_np))
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


# --- 4. Main GRU Training Pipeline ---
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

    for epoch in range(epochs):
        logger.info("--- Starting Epoch %d/%d ---", epoch + 1, epochs)
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for ticker in tickers:
            datasets = _prepare_ticker_datasets(
                ticker, feature_cols, sequence_length, log_split=(epoch == 0)
            )
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
            del train_dataset, datasets

        if train_samples == 0:
            raise RuntimeError(f"No valid GRU training sequences in epoch {epoch + 1}.")

        # Rebuild validation one ticker at a time to keep peak memory bounded.
        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0
        for ticker in tickers:
            datasets = _prepare_ticker_datasets(ticker, feature_cols, sequence_length)
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
            del val_dataset, datasets

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
