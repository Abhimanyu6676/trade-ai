"""Helpers for building GRU sequences without crossing trading sessions."""

import numpy as np
import pandas as pd

DEFAULT_SEQUENCE_LENGTH = 30


def session_dates(timestamps: pd.Series) -> np.ndarray:
    """Return local calendar dates for candle timestamps in their current order."""
    parsed = pd.to_datetime(timestamps, errors="coerce")
    if parsed.isna().any():
        raise ValueError("timestamps contain values that cannot be parsed")
    return parsed.dt.date.to_numpy()


def latest_session_sequence(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    timestamp_column: str = "timestamp",
) -> np.ndarray | None:
    """Return the latest full sequence from the newest session, if available."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if frame.empty:
        return None

    dates = session_dates(frame[timestamp_column])
    latest_date = dates[-1]
    session = frame.loc[dates == latest_date]
    if len(session) < sequence_length:
        return None

    return session[feature_columns].iloc[-sequence_length:].to_numpy(dtype=np.float32)


def valid_window_starts(dates: np.ndarray, sequence_length: int) -> np.ndarray:
    """Indices whose complete sliding window has one session date."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if len(dates) < sequence_length:
        return np.empty(0, dtype=np.int64)

    # A valid start has no date transition in its sequence.
    changes = np.r_[True, dates[1:] != dates[:-1]]
    session_starts = np.flatnonzero(changes)
    session_ends = np.r_[session_starts[1:], len(dates)]
    starts = [
        start
        for session_start, session_end in zip(session_starts, session_ends)
        for start in range(session_start, session_end - sequence_length + 1)
    ]
    return np.asarray(starts, dtype=np.int64)
