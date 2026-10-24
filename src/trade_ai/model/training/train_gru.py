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
from .labels import apply_triple_barrier
from .sequence_loader import (
    DEFAULT_SEQUENCE_LENGTH,
    session_dates,
    valid_window_starts,
)

# Directory and path configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- 2. Memory-Efficient Sliding Window Dataset ---
class SlidingWindowDataset(Dataset):
    """Generates sliding time-series windows on-the-fly to prevent memory overflow."""

    def __init__(
        self,
        features_np: np.ndarray,
        targets_np: np.ndarray,
        session_dates_np: np.ndarray,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ):
        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.float32)
        self.seq_len = sequence_length
        if len(session_dates_np) != len(self.features):
            raise ValueError("session dates must align with feature rows")
        window_starts = valid_window_starts(session_dates_np, sequence_length)
        target_indices = window_starts + sequence_length - 1
        labeled_windows = np.isfinite(targets_np[target_indices])
        self.window_starts = window_starts[labeled_windows]

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
        current_end = min(
            current_start + timedelta(days=chunk_days - 1), end_date
        )

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


# --- 4. Main GRU Training Pipeline ---
def train_gru_model(
    epochs: int = 10,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_size: int = 256,
    learning_rate: float = 0.001,
):
    """Trains the GRU network sequentially across stocks to maintain strict memory efficiency."""
    openalgo.initiate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("🚀 Initializing GRU Training Pipeline on device: %s", device)

    tickers = list(STOCK_METADATA.keys())
    feature_cols = list(GRU_MODEL_INPUT_COLUMNS)
    model = None
    optimizer = None
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        logger.info("--- Starting Epoch %d/%d ---", epoch + 1, epochs)
        total_epoch_loss = 0.0
        processed_stocks = 0

        for ticker in tickers:
            raw_df = load_or_fetch_stock_data(ticker)
            if raw_df is None or raw_df.empty:
                logger.warning("Skipping %s: Data unavailable.", ticker)
                continue

            # Build technical features; GRU feature calculation excludes metadata.
            df_feat = compute_normalized_features(raw_df, ticker, model_type="gru")

            # Triple barrier target generation
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

            # Keep feature rows with NaN targets as sequence context (for example,
            # 09:30-09:59 candles can inform the first labeled 10:00 sample).
            df_clean = df_feat.dropna(subset=["timestamp"]).copy()
            X_np = df_clean[feature_cols].to_numpy(dtype=np.float32)
            y_np = pd.to_numeric(df_clean["target"], errors="coerce").to_numpy(
                dtype=np.float32
            )
            session_dates_np = session_dates(df_clean["timestamp"])

            # Lazy-initialize model architecture once input dimension is known
            if model is None:
                input_dim = len(feature_cols)
                model = GRUTradingBrain(
                    input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                logger.info(
                    "Initialized GRU model with Input Dim: %d, Hidden Dim: 64",
                    input_dim,
                )

            # Type assertion to satisfy linters that optimizer is now initialized
            assert optimizer is not None

            # Build batched dataset and dataloader
            dataset = SlidingWindowDataset(
                X_np,
                y_np,
                session_dates_np=session_dates_np,
                sequence_length=sequence_length,
            )
            if len(dataset) == 0:
                logger.warning(
                    "Skipping %s: No complete same-session sequences available.", ticker
                )
                continue
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )

            model.train()
            ticker_loss = 0.0
            batch_count = 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                ticker_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_ticker_loss = ticker_loss / batch_count
                total_epoch_loss += avg_ticker_loss
                processed_stocks += 1
                logger.info(
                    "[%s] Epoch %d complete | Avg Loss: %.4f",
                    ticker,
                    epoch + 1,
                    avg_ticker_loss,
                )

        if processed_stocks > 0:
            epoch_avg_loss = total_epoch_loss / processed_stocks
            logger.info(
                "Epoch %d Summary | Overall Loss: %.4f", epoch + 1, epoch_avg_loss
            )

            # Save state checkpoint
            if model is not None:
                torch.save(
                    {
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_columns": feature_cols,
                        "state_dict": model.state_dict(),
                    },
                    GRU_MODEL_PATH,
                )
                logger.info("💾 Checkpoint saved to: %s", GRU_MODEL_PATH)
        else:
            logger.error("No valid stocks processed during Epoch %d.", epoch + 1)
            break

    logger.info(
        "✅ GRU Training pipeline completed successfully. Weights saved to %s",
        GRU_MODEL_PATH,
    )
