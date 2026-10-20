import datetime
import os
import time  # Make sure time is imported at the top of your train.py

import numpy as np
import pandas as pd
import torch
from openalgo import api
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from trade_ai.data_engine import OPENALGO_API_KEY, OPENALGO_HOST
from trade_ai.features import (
    FEATURE_COLUMNS,
    STOCK_METADATA,
    compute_normalized_features,
)
from trade_ai.model import StockPredictionBrain

# Directory configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")

# Initialize OpenAlgo client
client = api(api_key=OPENALGO_API_KEY, host=OPENALGO_HOST)


class ReturnWeightedBCELoss(nn.Module):
    """Custom Loss Function: Aligns model training with financial expectancy

    rather than raw classification accuracy by weighting errors proportional
    to the magnitude of the actual future return.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, predictions, targets, future_returns):
        bce_loss = self.bce(predictions, targets)
        weights = torch.abs(future_returns) + 1e-4
        return torch.mean(bce_loss * weights)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures standard capitalization for OHLCV columns."""
    df = df.copy()
    if df.index.name == "timestamp" or (
        "timestamp" in df.columns and "Timestamp" not in df.columns
    ):
        df = df.reset_index()

    clean_cols = {}
    for col in df.columns:
        c_low = str(col).lower()
        if c_low in ["timestamp", "open", "high", "low", "close", "volume"]:
            clean_cols[col] = c_low.capitalize()
        else:
            clean_cols[col] = str(col)
    return (
        df.rename(columns=clean_cols).sort_values(by="Timestamp").reset_index(drop=True)
    )


def download_and_cache_stock_history(
    ticker: str, exchange: str = "NSE"
) -> pd.DataFrame | None:
    """Downloads historical 1-minute data month-by-month with retry logic and delays

    to prevent OpenAlgo server timeouts.
    """
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")

    if os.path.exists(stock_csv):
        print(f"[{ticker}] Found local dataset in 'tickData'. Loading from disk...")
        df_cached = pd.read_csv(stock_csv)
        df_cached = standardize_columns(df_cached)
        print(
            f"[{ticker}] Loaded and cleaned {len(df_cached):,} total rows from cache."
        )
        return df_cached

    print(
        f"[{ticker}] Local cache missing. Downloading historical data from"
        " 2020-01-01 to 2026-07-31..."
    )

    start_date = datetime.date(2020, 1, 1)
    target_end_date = datetime.date(2026, 7, 31)
    today = datetime.date.today()
    end_date = min(target_end_date, today)

    total_days = (end_date - start_date).days
    current_start = start_date
    all_chunks = []
    downloaded_rows = 0

    while current_start < end_date:
        current_end = min(current_start + datetime.timedelta(days=30), end_date)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        days_elapsed = (current_start - start_date).days
        progress_pct = min(100.0, (days_elapsed / total_days) * 100)

        # Retry loop for stability against network/API hiccups
        success = False
        retries = 3
        for attempt in range(retries):
            try:
                print(
                    f"  [{progress_pct:.1f}%] Fetching {ticker} | {start_str} to"
                    f" {end_str} (Attempt {attempt + 1})..."
                )
                df_chunk = client.history(
                    symbol=ticker,
                    exchange=exchange,
                    interval="1m",
                    start_date=start_str,
                    end_date=end_str,
                    source="api",
                )

                if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                    chunk_rows = len(df_chunk)
                    downloaded_rows += chunk_rows
                    all_chunks.append(df_chunk)
                    print(
                        f"    -> Downloaded {chunk_rows:,} candles (Total:"
                        f" {downloaded_rows:,})"
                    )

                success = True
                break
            except Exception as e:
                print(
                    f"  ⚠️ Warning for {ticker} ({start_str} to {end_str}): {e}."
                    f" Retrying..."
                )
                time.sleep(2)  # Wait 2 seconds before retrying

        if not success:
            print(
                f"  ❌ Failed chunk for {ticker} from {start_str} to {end_str} after"
                f" {retries} attempts."
            )

        # Brief pause between chunks to prevent overwhelming the OpenAlgo server
        time.sleep(0.5)
        current_start = current_end + datetime.timedelta(days=1)

    if not all_chunks:
        print(f"❌ Failed to fetch any historical data for {ticker}.")
        return None

    full_df = pd.concat(all_chunks).drop_duplicates()
    full_df = standardize_columns(full_df)

    full_df.to_csv(stock_csv, index=False)
    print(
        f"✅ [{ticker}] Complete! Saved {len(full_df):,} total candles to '{stock_csv}'"
    )

    return full_df


def create_batched_loader(
    df: pd.DataFrame, sequence_length: int = 5, batch_size: int = 256
):
    """Converts dataframe into sliding sequences and wraps them in a memory-efficient DataLoader."""
    data = df[FEATURE_COLUMNS].values
    targets = df["Target_Direction"].values
    returns = df["Future_Return_5m"].values

    num_samples = len(data) - sequence_length
    if num_samples <= 0:
        return None

    xs, ys, rs = [], [], []
    for i in range(num_samples):
        xs.append(data[i : i + sequence_length])
        ys.append(targets[i + sequence_length - 1])
        rs.append(returns[i + sequence_length - 1])

    dataset = TensorDataset(
        torch.tensor(np.array(xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1),
        torch.tensor(np.array(rs), dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model():
    """Sequential stock-by-stock training loop designed to fit within 4GB RAM limits using mini-batches."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(TICK_DATA_DIR, exist_ok=True)

    print("🚀 Initializing Training Engine (Memory-Optimized)...")
    model = StockPredictionBrain(
        input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=2
    )

    # Load existing weights if resuming from a checkpoint
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            )
            print(f"📂 Loaded existing checkpoint weights from {MODEL_PATH}")
        except Exception:
            pass

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ReturnWeightedBCELoss()

    model.train()
    epochs = 5
    sequence_length = 5
    batch_size = 256  # Keeps RAM usage minimal

    tickers = list(STOCK_METADATA.keys())

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        total_epoch_loss = 0.0
        processed_stocks = 0

        for ticker in tickers:
            try:
                print(f"\nProcessing ticker: {ticker}")
                raw_df = download_and_cache_stock_history(ticker)
                if raw_df is None or raw_df.empty:
                    print(f"⚠️ Skipping {ticker}: Empty data.")
                    continue

                processed_df = compute_normalized_features(raw_df, ticker)
                if len(processed_df) <= sequence_length:
                    print(
                        f"[{ticker}] Insufficient rows after feature warmup. Skipping."
                    )
                    continue

                # Create memory-efficient DataLoader
                loader = create_batched_loader(
                    processed_df, sequence_length, batch_size
                )
                if loader is None:
                    continue

                ticker_loss = 0.0
                batches = 0

                # Train batch by batch
                for batch_X, batch_y, batch_r in loader:
                    optimizer.zero_grad()
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y, batch_r)

                    loss.backward()
                    optimizer.step()

                    ticker_loss += loss.item()
                    batches += 1

                avg_ticker_loss = ticker_loss / batches if batches > 0 else 0.0
                total_epoch_loss += avg_ticker_loss
                processed_stocks += 1
                print(
                    f"✅ Successfully trained on {ticker} | Avg Loss: {avg_ticker_loss:.4f}"
                )

            except Exception as e:
                print(f"❌ Error training on {ticker}: {e}")
                continue

        if processed_stocks > 0:
            avg_loss = total_epoch_loss / processed_stocks
            print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

            # Checkpoint save after every successful epoch
            torch.save(model.state_dict(), MODEL_PATH)
            print(
                f"💾 Checkpoint saved: Model weights updated for Epoch {epoch + 1} -> {MODEL_PATH}"
            )
        else:
            print("⚠️ No stocks successfully processed during this epoch.")
            break

        # Optional: clear cache/garbage collector between epochs to keep RAM clean
        import gc

        gc.collect()

    print(f"\n✅ Training fully complete! Final weights saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
