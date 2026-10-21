"""OpenAlgo client lifecycle and API operations."""

import os
from typing import Any

import pandas as pd
from openalgo import api

from trade_ai.util import standardize_ohlcv_columns

DEFAULT_OPENALGO_HOST = "https://openalgo-1.sternetsmart.com"


class OpenAlgo:
    """Lazily initialize and reuse the OpenAlgo SDK client."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def initiate(self) -> Any:
        """Create the SDK client once from the current environment settings."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get("OPENALGO_API_KEY")
        if not api_key:
            raise RuntimeError("OPENALGO_API_KEY is required to connect to OpenAlgo")

        host = os.environ.get("OPENALGO_HOST", DEFAULT_OPENALGO_HOST)
        self._client = api(api_key=api_key, host=host)
        return self._client

    def get_history(
        self,
        *,
        symbol: str,
        exchange: str,
        interval: str,
        start_date: str,
        end_date: str,
        source: str = "api",
        standardize: bool = True,
    ) -> Any:
        """Fetch history, standardizing DataFrame OHLCV columns by default."""
        history = self.initiate().history(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
        if standardize and isinstance(history, pd.DataFrame):
            return standardize_ohlcv_columns(history)
        return history

    def get_analyzer_status(self) -> Any:
        """Return the OpenAlgo analyzer status response."""
        return self.initiate().analyzerstatus()


openalgo = OpenAlgo()
