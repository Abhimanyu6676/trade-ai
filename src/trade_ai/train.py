import datetime
import os
import threading
import time  # Make sure time is imported at the top of your train.py
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from trade_ai.features import (
    FEATURE_COLUMNS,
    compute_normalized_features,
)
from trade_ai.model import (
    StockPredictionBrain,
    load_model_checkpoint,
    save_model_checkpoint,
)
from trade_ai.openalgo_client import openalgo
from trade_ai.stock_metadata import STOCK_METADATA
from trade_ai.util import load_cached_ohlcv

# Directory configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TICK_DATA_DIR = os.path.join(BASE_DIR, "tickData")
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")
TRAINING_LOCK = threading.Lock()
TRAINING_THREAD = None
TRAINING_STATE = "idle"
TRAINING_ERROR = None

class ReturnWeightedBCELoss(nn.Module):
    """Balance direction classes and emphasize meaningful forward returns."""

    def __init__(self, positive_weight: float):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.positive_weight = positive_weight

    def forward(self, predictions, targets, future_returns):
        bce_loss = self.bce(predictions, targets)
        # Five-minute returns are small decimals. Scale them before adding a
        # unit baseline so return weighting cannot shrink gradients toward zero.
        return_weights = torch.abs(future_returns) * 1000.0 + 1.0
        class_weights = torch.where(
            targets > 0.5,
            torch.as_tensor(
                self.positive_weight, dtype=targets.dtype, device=targets.device
            ),
            1.0,
        )
        weights = return_weights * class_weights
        return torch.mean(bce_loss * weights)


def download_and_cache_stock_history(
    ticker: str, exchange: str = "NSE"
) -> pd.DataFrame | None:
    """Load cached history or download it and write the resulting CSV cache."""
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")
    df_cached = load_cached_ohlcv(stock_csv)
    if df_cached is not None:
        print(f"[{ticker}] Found local dataset in 'tickData'. Loading from disk...")
        print(
            f"[{ticker}] Loaded and cleaned {len(df_cached):,} total rows from cache."
        )
        return df_cached

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date.today()
    print(
        f"[{ticker}] Local cache missing. Downloading historical data from"
        f" {start_date.isoformat()} to {end_date.isoformat()}..."
    )

    full_df = download_stock_history_in_chunks(
        ticker=ticker,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
    )
    if full_df is None:
        print(f"❌ Failed to fetch any historical data for {ticker}.")
        return None

    full_df.to_csv(stock_csv, index=False)
    print(
        f"✅ [{ticker}] Complete! Saved {len(full_df):,} total candles to '{stock_csv}'"
    )
    return full_df


def download_stock_history_in_chunks(
    ticker: str,
    exchange: str,
    start_date: datetime.date,
    end_date: datetime.date,
    chunk_days: int = 30,
    retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch and combine historical 1-minute candles in date chunks.

    Each chunk retries on API exceptions and pauses between requests to reduce
    the chance of OpenAlgo server timeouts. Failed chunks are skipped after all
    retries, allowing any successful chunks to be returned.
    """
    total_days = (end_date - start_date).days
    current_start = start_date
    all_chunks: list[pd.DataFrame] = []
    downloaded_rows = 0

    while current_start < end_date:
        chunk_end = current_start + datetime.timedelta(days=chunk_days)
        current_end = min(chunk_end, end_date)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        days_elapsed = (current_start - start_date).days
        progress_pct = min(100.0, (days_elapsed / total_days) * 100)

        # Retry loop for stability against network/API hiccups
        success = False
        for attempt in range(retries):
            try:
                print(
                    f"  [{progress_pct:.1f}%] Fetching {ticker} | {start_str} to"
                    f" {end_str} (Attempt {attempt + 1})..."
                )
                df_chunk = openalgo.get_history(
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
        return None

    full_df = pd.concat(all_chunks).drop_duplicates()
    return full_df.sort_values(by="Timestamp").reset_index(drop=True)


class SlidingWindowDataset(Dataset):
    """Keep one feature matrix in memory and build sequences per requested sample."""

    def __init__(self, df: pd.DataFrame, sequence_length: int = 5):
        self.features = torch.from_numpy(
            df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        )
        self.targets = torch.from_numpy(
            df["Target_Direction"].to_numpy(dtype=np.float32, copy=True)
        )
        self.returns = torch.from_numpy(
            df["Future_Return_5m"].to_numpy(dtype=np.float32, copy=True)
        )
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length + 1)

    def __getitem__(self, index):
        target_index = index + self.sequence_length - 1
        return (
            self.features[index : target_index + 1],
            self.targets[target_index].unsqueeze(0),
            self.returns[target_index].unsqueeze(0),
        )


def create_batched_loader(
    df: pd.DataFrame, sequence_length: int = 5, batch_size: int = 256
):
    """Build batches without materializing every overlapping sequence."""
    dataset = SlidingWindowDataset(df, sequence_length)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model() -> int:
    """Sequential stock-by-stock training loop designed to fit within 4GB RAM limits using mini-batches."""
    # Fail immediately with a clear message if credentials are missing.
    openalgo.initiate()
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(TICK_DATA_DIR, exist_ok=True)

    print("🚀 Initializing Training Engine (Memory-Optimized)...")
    model = StockPredictionBrain(
        input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=2
    )

    # Load existing weights if resuming from a checkpoint
    if os.path.exists(MODEL_PATH):
        try:
            load_model_checkpoint(model, MODEL_PATH)
            print(f"📂 Loaded existing checkpoint weights from {MODEL_PATH}")
        except Exception as error:
            print(f"⚠️ Ignoring incompatible checkpoint: {error}. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epochs = 5
    sequence_length = 5
    batch_size = 256  # Keeps RAM usage minimal

    tickers = list(STOCK_METADATA.keys())
    successful_epochs = 0

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

                processed_df = compute_normalized_features(
                    raw_df, ticker, include_targets=True
                )
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

                # Balance positive and non-positive direction labels using
                # the full ticker dataset, rather than noisy per-batch ratios.
                target_indices = slice(sequence_length - 1, None)
                training_dataset = cast(SlidingWindowDataset, loader.dataset)
                training_targets = training_dataset.targets[target_indices]
                positive_count = int((training_targets > 0.5).sum().item())
                negative_count = len(training_targets) - positive_count
                if positive_count == 0 or negative_count == 0:
                    print(
                        f"⚠️ Skipping {ticker}: training labels contain only one"
                        " direction class."
                    )
                    continue

                positive_weight = min(negative_count / positive_count, 10.0)
                criterion = ReturnWeightedBCELoss(positive_weight)
                print(
                    f"[{ticker}] Direction labels: {positive_count:,} positive / "
                    f"{negative_count:,} non-positive; positive class weight: "
                    f"{positive_weight:.2f}"
                )

                ticker_loss = 0.0
                batches = 0
                prediction_min = float("inf")
                prediction_max = float("-inf")
                prediction_sum = 0.0
                prediction_count = 0

                # Train batch by batch
                for batch_X, batch_y, batch_r in loader:
                    optimizer.zero_grad()
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y, batch_r)

                    prediction_min = min(prediction_min, preds.min().item())
                    prediction_max = max(prediction_max, preds.max().item())
                    prediction_sum += preds.sum().item()
                    prediction_count += preds.numel()

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
                if prediction_count:
                    print(
                        f"[{ticker}] Prediction probabilities: min="
                        f"{prediction_min:.4f}, max={prediction_max:.4f}, "
                        f"mean={prediction_sum / prediction_count:.4f}"
                    )

            except Exception as e:
                print(f"❌ Error training on {ticker}: {e}")
                continue

        if processed_stocks > 0:
            avg_loss = total_epoch_loss / processed_stocks
            print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

            # Checkpoint save after every successful epoch
            save_model_checkpoint(model, MODEL_PATH)
            successful_epochs += 1
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
    return successful_epochs


def _train_model_in_background():
    """Run training asynchronously and refresh the app's inference model."""
    global TRAINING_STATE, TRAINING_ERROR

    try:
        completed_epochs = train_model()
        if completed_epochs == 0:
            raise RuntimeError(
                "Training finished without a successful epoch. Check data and "
                "OpenAlgo availability."
            )

        from trade_ai.prediction_engine import reload_prediction_brain

        if not reload_prediction_brain():
            raise RuntimeError("Training finished, but its checkpoint could not be loaded.")

        with TRAINING_LOCK:
            TRAINING_STATE = "completed"
            TRAINING_ERROR = None
        print("✅ [Training] Completed; the app loaded the latest checkpoint.")
    except Exception as error:
        from trade_ai.prediction_engine import reload_prediction_brain

        restored_model = reload_prediction_brain()
        with TRAINING_LOCK:
            TRAINING_STATE = "failed"
            TRAINING_ERROR = str(error)
        if restored_model:
            print(
                "⚠️ [Training] Background training failed; loaded the latest "
                f"available checkpoint instead: {error}"
            )
        else:
            print(f"❌ [Training] Background training failed: {error}")


def start_model_training() -> str:
    """Start one training run in the background without blocking the UI."""
    global TRAINING_THREAD, TRAINING_STATE, TRAINING_ERROR

    with TRAINING_LOCK:
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            return "Training is already running. Follow progress in the app server console."

        from trade_ai.prediction_engine import unload_prediction_brain

        unload_prediction_brain()
        TRAINING_STATE = "running"
        TRAINING_ERROR = None
        TRAINING_THREAD = threading.Thread(
            target=_train_model_in_background,
            name="trade-ai-training",
            daemon=True,
        )
        TRAINING_THREAD.start()

    return (
        "Training started in the background. Progress is printed in the app "
        "server console; inference will become available when training completes."
    )


if __name__ == "__main__":
    train_model()
