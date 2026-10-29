"""Process-wide in-memory store for OHLCV history DataFrames."""

from __future__ import annotations

import datetime
from threading import Lock
from typing import Self, cast

import pandas as pd

from trade_ai.logger import logger
from trade_ai.openalgo.openalgo_client import openalgo
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE, generateSymbolKey


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
        """Return cached history, fetching 30 days on a cache miss."""
        symbolKey = generateSymbolKey(ticker, exchange)
        with self._data_lock:
            history = self._candles_by_key.get(symbolKey)
            if history is not None:
                return history.copy(deep=True)

        logger.info("No cached candles for %s; fetching history", symbolKey)
        return await self.fetchCandles(ticker, exchange)

    async def fetchCandles(
        self, ticker: str, exchange: TYPE_OPENALGO_EXCHANGE
    ) -> pd.DataFrame | None:
        """Fetch and cache the latest 30 trading sessions of one-minute history."""
        symbolKey = generateSymbolKey(ticker, exchange)
        lastTradingDate = await openalgo.getPreviousTradeDay(
            date=datetime.date.today(),
            exchange=exchange,
        ) or datetime.date.today()
        startDate = await openalgo.getPreviousTradeDay(
            date=lastTradingDate,
            days=30,
            exchange=exchange,
        ) or lastTradingDate - datetime.timedelta(days=30)

        logger.info(
            "Fetching 1-minute candles for %s from %s through %s",
            symbolKey,
            startDate,
            lastTradingDate,
        )
        try:
            history = await openalgo.get_history(
                symbol=ticker.strip().upper(),
                exchange=exchange,
                interval="1m",
                start_date=startDate,
                end_date=lastTradingDate,
            )
            if not isinstance(history, pd.DataFrame) or history.empty:
                logger.error("OpenAlgo returned no candle history for %s", symbolKey)
                return None

            timestamp_column = next(
                (
                    column
                    for column in history.columns
                    if str(column).casefold() == "timestamp"
                ),
                None,
            )
            if timestamp_column is not None:
                timestamp_values = history[timestamp_column]
                first_timestamp = timestamp_values.iloc[0]
                last_timestamp = timestamp_values.iloc[-1]
            else:
                first_timestamp = history.index[0]
                last_timestamp = history.index[-1]

            logger.info(
                "Fetched %s history for %s; first timestamp: %s, last timestamp: %s",
                len(history),
                symbolKey,
                first_timestamp,
                last_timestamp,
            )

            logger.info("Last active market day for %s: %s", symbolKey, lastTradingDate)

            history = history.copy(deep=True)
            with self._data_lock:
                self._candles_by_key[symbolKey] = history
            return history.copy(deep=True)
        except Exception:
            logger.exception("Failed to fetch candles for %s", symbolKey)
            return None


dataStore = CandleStore()
