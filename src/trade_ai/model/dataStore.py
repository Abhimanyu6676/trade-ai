"""Process-wide in-memory store for OHLCV history DataFrames."""

from __future__ import annotations

import asyncio
import datetime
from threading import Lock
from typing import Any, Self, cast

import pandas as pd
from pandas.api.typing import NaTType

from trade_ai.logger import logger
from trade_ai.openalgo.openalgo_client import openalgo
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE, generateSymbolKey
from trade_ai.util import merge_ohlcv_history


class CandleStore:
    """A singleton mapping OHLCV history frames by ``TICKER:EXCHANGE``.

    The store is shared by imports in this Python process. Data is in memory
    only and is cleared when the process exits.
    """

    _instance: CandleStore | None = None
    _instance_lock = Lock()
    _candles_by_key: dict[str, pd.DataFrame]
    _data_lock: Lock

    def __new__(cls) -> Self:
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._candles_by_key = {}
                instance._data_lock = Lock()
                cls._instance = instance
            return cast(Self, cls._instance)

    async def get_candles(
        self, ticker: str, exchange: TYPE_OPENALGO_EXCHANGE
    ) -> pd.DataFrame | None:
        """Return 30 trading sessions before today, fetching missing history."""
        return await self.get_history_before(ticker, exchange, datetime.date.today())

    async def get_history_before(
        self,
        ticker: str,
        exchange: TYPE_OPENALGO_EXCHANGE,
        prediction_date: datetime.date,
        sessions: int = 30,
    ) -> pd.DataFrame | None:
        """Return ``sessions`` of 1m history ending before prediction_date."""
        symbol_key = generateSymbolKey(ticker, exchange)
        expected_dates = await openalgo.getPreviousTradeDays(
            date=prediction_date, days=sessions, exchange=exchange
        )
        if len(expected_dates) != sessions:
            logger.error("Could not resolve trading dates for %s", symbol_key)
            return None
        start_date = expected_dates[-1]
        end_date = expected_dates[0]

        logger.info(
            "Required history for %s: %s trading sessions, %s through %s "
            "(prediction date %s)",
            symbol_key,
            sessions,
            start_date,
            end_date,
            prediction_date,
        )

        with self._data_lock:
            cached = self._candles_by_key.get(symbol_key)
            cached_copy = cached.copy(deep=True) if cached is not None else None
        if cached_copy is not None:
            valid, first, last = await asyncio.to_thread(_chronology_info, cached_copy)
            if not valid:
                logger.warning(
                    "Cached history for %s is not chronologically valid "
                    "(first=%s, last=%s, rows=%s); it will be normalized before use",
                    symbol_key,
                    first,
                    last,
                    len(cached_copy),
                )
            if await asyncio.to_thread(_covers_dates, cached_copy, expected_dates):
                await asyncio.to_thread(_log_intraday_gaps, cached_copy, symbol_key)
                history = await asyncio.to_thread(
                    _slice_dates, cached_copy, start_date, end_date
                )
                cache_already_matches = (
                    valid
                    and _coerce_timestamp(first).date() == start_date
                    and _coerce_timestamp(last).date() == end_date
                    and len(history) == len(cached_copy)
                )
                if cache_already_matches:
                    logger.debug(
                        "Reusing existing prior-session history for %s",
                        symbol_key,
                    )
                    return history
                return await asyncio.to_thread(
                    self.replace_candles, ticker, exchange, history
                )

        logger.info(
            "Fetching %s 1-minute history for %s from %s through %s",
            sessions,
            symbol_key,
            start_date,
            end_date,
        )
        try:
            fetched = await openalgo.get_history(
                symbol=ticker.strip().upper(),
                exchange=exchange,
                interval="1m",
                start_date=start_date,
                end_date=end_date,
            )
            if not isinstance(fetched, pd.DataFrame) or fetched.empty:
                logger.error("OpenAlgo returned no candle history for %s", symbol_key)
                return None
            valid, first, last = await asyncio.to_thread(_chronology_info, fetched)
            if not valid:
                logger.warning(
                    "Fetched history for %s is not chronologically valid "
                    "(first=%s, last=%s, rows=%s); sorting and deduplication "
                    "will be applied before storage",
                    symbol_key,
                    first,
                    last,
                    len(fetched),
                )
            history = await asyncio.to_thread(merge_ohlcv_history, fetched)
            if not await asyncio.to_thread(_covers_dates, history, expected_dates):
                logger.error(
                    "History for %s is missing one or more required trading dates "
                    "between %s and %s before prediction date %s",
                    symbol_key,
                    start_date,
                    end_date,
                    prediction_date,
                )
                return None
            await asyncio.to_thread(_log_intraday_gaps, history, symbol_key)
            history = await asyncio.to_thread(
                _slice_dates, history, start_date, end_date
            )
            return await asyncio.to_thread(
                self.replace_candles, ticker, exchange, history
            )
        except Exception:
            logger.exception("Failed to fetch candles for %s", symbol_key)
            return None

    def replace_candles(
        self, ticker: str, exchange: TYPE_OPENALGO_EXCHANGE, candles: pd.DataFrame
    ) -> pd.DataFrame:
        """Replace the ticker cache with the latest request's candle frame."""
        symbol_key = generateSymbolKey(ticker, exchange)
        replacement = merge_ohlcv_history(candles)
        with self._data_lock:
            self._candles_by_key[symbol_key] = replacement.copy(deep=True)
        first, last = _frame_range(replacement)
        logger.info(
            "Replaced in-memory candle store for %s: %s rows, date range %s through %s",
            symbol_key,
            len(replacement),
            first,
            last,
        )
        return replacement


def _timestamp_dates(frame: pd.DataFrame) -> pd.Series:
    frame = merge_ohlcv_history(frame)
    return pd.to_datetime(frame["timestamp"]).dt.date


def _chronology_info(frame: pd.DataFrame) -> tuple[bool, object, object]:
    if frame.empty:
        return False, "empty history", "empty history"
    normalized = frame.rename(
        columns={column: str(column).casefold() for column in frame.columns}
    )
    if "timestamp" in normalized.columns:
        timestamps = pd.to_datetime(normalized["timestamp"], errors="coerce")
    elif isinstance(normalized.index, pd.DatetimeIndex):
        timestamps = pd.Series(normalized.index)
    else:
        return False, "missing timestamp", "missing timestamp"
    valid = (
        not timestamps.isna().any()
        and timestamps.is_monotonic_increasing
        and not timestamps.duplicated().any()
    )
    return valid, timestamps.iloc[0], timestamps.iloc[-1]


def _coerce_timestamp(value: object) -> datetime.datetime | NaTType:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime.datetime):
        return pd.Timestamp(value)
    if isinstance(value, datetime.date):
        return pd.Timestamp(value)
    try:
        return pd.to_datetime(cast(Any, value))
    except (TypeError, ValueError):
        return pd.NaT


def _frame_range(frame: pd.DataFrame) -> tuple[object, object]:
    normalized = merge_ohlcv_history(frame)
    timestamps = pd.to_datetime(normalized["timestamp"])
    return timestamps.min(), timestamps.max()


def _covers_dates(frame: pd.DataFrame, expected_dates: list[datetime.date]) -> bool:
    if frame.empty:
        return False
    try:
        available_dates = set(_timestamp_dates(frame))
        return set(expected_dates).issubset(available_dates)
    except (TypeError, ValueError, KeyError) as error:
        logger.warning("Unable to validate candle-history dates: %s", error)
        return False


def _log_intraday_gaps(frame: pd.DataFrame, symbol_key: str) -> None:
    normalized = merge_ohlcv_history(frame)
    timestamps = pd.to_datetime(normalized["timestamp"])
    session_dates = timestamps.dt.date
    gaps: list[tuple[datetime.date, pd.Timestamp, pd.Timestamp, pd.Timedelta]] = []
    for session_date, session_timestamps in timestamps.groupby(session_dates):
        values = session_timestamps.tolist()
        for previous, current in zip(values, values[1:]):
            duration = current - previous
            if duration > pd.Timedelta(minutes=1):
                gaps.append((session_date, previous, current, duration))
    if gaps:
        max_logged_gaps = 25
        gap_details = "; ".join(
            f"{session_date} {previous:%H:%M:%S}->{current:%H:%M:%S} ({duration})"
            for session_date, previous, current, duration in gaps[:max_logged_gaps]
        )
        if len(gaps) > max_logged_gaps:
            gap_details += f"; {len(gaps) - max_logged_gaps} additional gaps omitted"
        logger.warning(
            "History for %s has %s intraday gaps greater than one minute; "
            "largest gap is %s. Gap date/time ranges: %s. These may reflect "
            "missing source bars or minutes with no trades.",
            symbol_key,
            len(gaps),
            max(gap[3] for gap in gaps),
            gap_details,
        )


def _slice_dates(
    frame: pd.DataFrame, start: datetime.date, end: datetime.date
) -> pd.DataFrame:
    normalized = merge_ohlcv_history(frame)
    dates = pd.to_datetime(normalized["timestamp"]).dt.date
    return normalized.loc[(dates >= start) & (dates <= end)].reset_index(drop=True)


dataStore = CandleStore()
