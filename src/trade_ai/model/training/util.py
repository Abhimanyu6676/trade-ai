"""Shared history loading and download helpers for model training."""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Iterable

import numpy as np
import pandas as pd

from trade_ai.logger import logger
from trade_ai.config.trading_config import (
    ENTRY_END_HOUR,
    ENTRY_END_MINUTE,
    ENTRY_START_HOUR,
    ENTRY_START_MINUTE,
)
from trade_ai.openalgo.openalgo_client import openalgo
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE
from trade_ai.util import merge_ohlcv_history

TICK_DATA_DIR = os.path.join(os.path.dirname(__file__), "tickData")


@dataclass(frozen=True)
class WalkForwardPlan:
    """Shared whole-session boundaries for selection, PPO folds, and holdout."""

    validation_start: pd.Timestamp
    oos_start: pd.Timestamp
    final_holdout_start: pd.Timestamp
    oos_folds: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]


SELECTION_FIT_FRACTION = 0.55
SELECTION_END_FRACTION = 0.70
WALK_FORWARD_OOS_END_FRACTION = 0.90
WALK_FORWARD_FOLD_COUNT = 4


def atomic_write_json(path: str | Path, data: dict[str, object]) -> None:
    """Write a JSON sidecar to a temporary file and atomically replace it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as output_file:
            temporary_path = output_file.name
            json.dump(data, output_file, indent=2)
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def global_chronological_boundaries(
    feature_frames: Iterable[pd.DataFrame],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Choose global timestamp boundaries for fit, selection, and OOS periods.

    The function uses pooled session dates so every ticker shares the same
    whole-session boundaries. Later history from one ticker cannot enter the
    fit while an earlier session from another ticker is treated as OOS.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")

    labeled_dates = [
        pd.to_datetime(frame.loc[frame["target"].notna(), "timestamp"])
        for frame in feature_frames
        if not frame.empty and "target" in frame
    ]
    labeled_dates = [values for values in labeled_dates if not values.empty]
    if not labeled_dates:
        raise RuntimeError("No labeled timestamps are available for chronological splits")
    pooled_timestamps = pd.to_datetime(pd.concat(labeled_dates, ignore_index=True))
    return global_session_date_boundaries(
        pooled_timestamps.dt.normalize().tolist(),
        train_fraction,
        validation_fraction,
    )


def eligible_session_dates(frame: pd.DataFrame) -> set[pd.Timestamp]:
    """Return source-candle dates with at least one eligible label entry time."""
    if "timestamp" not in frame.columns:
        raise ValueError("candle frame is missing its timestamp column")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    seconds = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
    start_seconds = ENTRY_START_HOUR * 3600 + ENTRY_START_MINUTE * 60
    end_seconds = ENTRY_END_HOUR * 3600 + ENTRY_END_MINUTE * 60
    return set(timestamps.loc[(seconds >= start_seconds) & (seconds <= end_seconds)].dt.normalize())


def global_session_date_boundaries(
    session_dates: Iterable[date | pd.Timestamp],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Choose shared whole-session boundaries from unique pooled trade dates."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    unique_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(value).normalize() for value in session_dates})
    )
    if len(unique_dates) < 3:
        raise RuntimeError("At least three unique session dates are required for three-way splits")

    validation_index = max(
        1,
        min(len(unique_dates) - 2, int(len(unique_dates) * train_fraction)),
    )
    oos_index = max(
        validation_index + 1,
        min(
            len(unique_dates) - 1,
            int(len(unique_dates) * (train_fraction + validation_fraction)),
        ),
    )
    return unique_dates[validation_index], unique_dates[oos_index]


def global_walk_forward_plan(
    session_dates: Iterable[date | pd.Timestamp],
    fold_count: int = WALK_FORWARD_FOLD_COUNT,
) -> WalkForwardPlan:
    """Plan shared chronological periods over pooled ticker session dates.

    The first 55% fits an initial base model, the next 15% selects its epoch or
    boosting rounds, the next 20% is divided into expanding OOS folds, and the
    newest 10% is reserved for final evaluation.
    """
    if fold_count < 2:
        raise ValueError("at least two walk-forward OOS folds are required")
    unique_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(value).normalize() for value in session_dates})
    )
    if not len(unique_dates):
        raise RuntimeError(
            "No unique session dates are available for the walk-forward plan"
        )

    validation_index = int(len(unique_dates) * SELECTION_FIT_FRACTION)
    oos_index = int(len(unique_dates) * SELECTION_END_FRACTION)
    final_index = int(len(unique_dates) * WALK_FORWARD_OOS_END_FRACTION)
    if (
        validation_index < 1
        or oos_index <= validation_index
        or final_index - oos_index < fold_count
        or final_index >= len(unique_dates)
    ):
        raise RuntimeError(
            "Not enough unique session dates for initial fit, validation, "
            f"{fold_count} OOS folds, and a final holdout"
        )
    fold_indices = np.linspace(oos_index, final_index, fold_count + 1).astype(int)
    if len(set(fold_indices.tolist())) != fold_count + 1:
        raise RuntimeError(
            "Not enough unique session dates to form non-empty OOS folds"
        )
    fold_boundaries = tuple(unique_dates[index] for index in fold_indices)
    folds = tuple(zip(fold_boundaries[:-1], fold_boundaries[1:], strict=True))
    return WalkForwardPlan(
        validation_start=unique_dates[validation_index],
        oos_start=unique_dates[oos_index],
        final_holdout_start=unique_dates[final_index],
        oos_folds=folds,
    )


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

    logger.info(
        "[%s] Downloading historical data from %s to %s via OpenAlgo...",
        ticker,
        start_date.isoformat(),
        end_date.isoformat(),
    )

    while current_start <= end_date:
        # Treat both endpoints as inclusive, then resume on the following day.
        current_end = min(
            current_start + timedelta(days=chunk_days) - timedelta(days=1),
            end_date,
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
                if df_chunk.empty:
                    if all_chunks:
                        raise RuntimeError(
                            "OpenAlgo returned an empty chunk after history had "
                            "already started; refusing a gapped dataset"
                        )
                    logger.info(
                        "[%s] No history in pre-listing range %s to %s; "
                        "continuing to next chunk.",
                        ticker,
                        current_start,
                        current_end,
                    )
                    success = True
                    break

                all_chunks.append(df_chunk)
                logger.info(
                    "[%s] Received %d candles for range %s to %s.",
                    ticker,
                    len(df_chunk),
                    current_start,
                    current_end,
                )
                success = True
                break
            except Exception as exc:
                logger.warning(
                    "[%s] Retry %d failed for range %s to %s: %s",
                    ticker,
                    attempt + 1,
                    current_start,
                    current_end,
                    exc,
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
        logger.warning(
            "[%s] No history was available in any requested chunk; skipping ticker.",
            ticker,
        )
        return None

    return merge_ohlcv_history(pd.concat(all_chunks))


def load_or_fetch_stock_data(ticker: str) -> pd.DataFrame | None:
    """Load normalized cached history or download it and cache it locally."""
    os.makedirs(TICK_DATA_DIR, exist_ok=True)
    stock_csv = os.path.join(TICK_DATA_DIR, f"{ticker}_history.csv")

    if os.path.exists(stock_csv):
        logger.info("[%s] Loading local dataset from disk: %s", ticker, stock_csv)
        return merge_ohlcv_history(pd.read_csv(stock_csv))

    logger.info("[%s] Local cache missing. Fetching new data...", ticker)
    df = download_stock_history_in_chunks(ticker)
    if df is not None and not df.empty:
        df.to_csv(stock_csv, index=False)
        logger.info("[%s] Saved newly downloaded history to %s", ticker, stock_csv)
        return df

    logger.warning("[%s] No data available to cache.", ticker)
    return None
