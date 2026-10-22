import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from trade_ai.features import compute_normalized_features
from trade_ai.openalgo_client import openalgo
from trade_ai.stock_metadata import STOCK_METADATA

from .labels import apply_triple_barrier

# Directory and path configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- 1. PyTorch GRU Neural Network Architecture ---
class GRUTradingBrain(nn.Module):
    """Two-layer GRU architecture with dropout and a linear classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Extract output from the last time-step of the sequence
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits


# --- 2. Memory-Efficient Sliding Window Dataset ---
class SlidingWindowDataset(Dataset):
    """Generates sliding time-series windows on-the-fly to prevent memory overflow."""

    def __init__(
        self,
        features_np: np.ndarray,
        targets_np: np.ndarray,
        sequence_length: int = 30,
    ):
        self.features = torch.tensor(features_np, dtype=torch.float32)
        self.targets = torch.tensor(targets_np, dtype=torch.long)
        self.seq_len = sequence_length

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx: int):
        target_idx = idx + self.seq_len - 1
        x_seq = self.features[idx : target_idx + 1]
        y_val = self.targets[target_idx]
        return x_seq, y_val


# --- 3. Data Downloader and Ingestion Utilities ---
def download_stock_history_in_chunks(
    ticker: str,
    exchange: str = "NSE",
    start_date: date = date(2020, 1, 1),
    end_date: date = date.today(),
    chunk_days: int = 30,
    retries: int = 3,
) -> pd.DataFrame | None:
    """Downloads historical 1-minute candles in chunks via OpenAlgo API."""
    current_start = start_date
    all_chunks: list[pd.DataFrame] = []

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
                    "[%s] Chunk retry %d failed for %s to %s: %s",
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
                "[%s] Failed chunk %s to %s after max retries.",
                ticker,
                start_str,
                end_str,
            )

        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        return None

    full_df = pd.concat(all_chunks).drop_duplicates()
    if "Timestamp" in full_df.columns and "timestamp" not in full_df.columns:
        full_df = full_df.rename(columns={"Timestamp": "timestamp"})

    return full_df.sort_values(by="timestamp").reset_index(drop=True)


def load_or_fetch_stock_data(ticker: str) -> pd.DataFrame | None:
    """Loads local CSV if present; otherwise fetches from OpenAlgo and caches to disk."""
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")

    if os.path.exists(stock_csv):
        logger.info("[%s] Loading local dataset: %s", ticker, stock_csv)
        return pd.read_csv(stock_csv)

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
    sequence_length: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
):
    """Trains the GRU network sequentially across stocks to maintain strict memory efficiency."""
    openalgo.initiate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("🚀 Initializing GRU Training Pipeline on device: %s", device)

    tickers = list(STOCK_METADATA.keys())
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

            # Feature engineering and metadata encoding
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

            # Drop warm-up and time-filter NaNs
            df_clean = df_feat.dropna(subset=["target"]).copy()
            if len(df_clean) <= sequence_length:
                logger.warning(
                    "Skipping %s: Insufficient valid rows after filtering.", ticker
                )
                continue

            feature_cols = [
                col for col in df_clean.columns if col not in ["timestamp", "target"]
            ]
            X_np = df_clean[feature_cols].to_numpy(dtype=np.float32)

            # Remap target classes: {-1, 0, 1} -> {0, 1, 2} for CrossEntropyLoss
            y_np = (df_clean["target"].astype(int) + 1).to_numpy(dtype=np.int64)

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
            dataset = SlidingWindowDataset(X_np, y_np, sequence_length=sequence_length)
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
                torch.save(model.state_dict(), MODEL_PATH)
                logger.info("💾 Checkpoint saved to: %s", MODEL_PATH)
        else:
            logger.error("No valid stocks processed during Epoch %d.", epoch + 1)
            break

    logger.info(
        "✅ GRU Training pipeline completed successfully. Weights saved to %s",
        MODEL_PATH,
    )


if __name__ == "__main__":
    train_gru_model()
