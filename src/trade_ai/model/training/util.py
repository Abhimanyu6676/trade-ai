"""Shared history loading and download helpers for model training."""

import asyncio
import os
from datetime import date, timedelta
from time import sleep

import pandas as pd

from trade_ai.logger import logger
from trade_ai.openalgo.openalgo_client import openalgo
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE
from trade_ai.util import merge_ohlcv_history

TICK_DATA_DIR = os.path.join(os.path.dirname(__file__), "tickData")


def download_stock_history_in_chunks(
    ticker: str,
    exchange: TYPE_OPENALGO_EXCHANGE = "NSE",
    start_date: date = date(2021, 11, 1),
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
