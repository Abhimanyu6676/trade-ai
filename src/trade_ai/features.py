import pandas as pd

from trade_ai.indicator import (
    calculate_atr_pct,
    calculate_ema,
    calculate_macro_trend_gate,
    calculate_orb,
    calculate_rsi,
    calculate_volatility,
    calculate_vwap,
)
from trade_ai.stock_metadata import (
    METADATA_FEATURE_COLUMNS,
    encode_categorical_metadata,
)


def _resample_close(series: pd.Series, rule: str, offset: str | None = None):
    """Return completed-bar closes, optionally aligned to a session offset."""
    if offset is None:
        return series.resample(rule, label="right", closed="left").last().dropna()
    return (
        series.resample(rule, label="right", closed="left", offset=offset)
        .last()
        .dropna()
    )


def _resample_ohlcv(
    df: pd.DataFrame, rule: str, offset: str | None = None
) -> pd.DataFrame:
    """Aggregate candle-open OHLCV rows into completed bars labeled at close."""
    if offset is not None:
        return pd.DataFrame(
            {
                "Open": df["Open"]
                .resample(rule, label="right", closed="left", offset=offset)
                .first(),
                "High": df["High"]
                .resample(rule, label="right", closed="left", offset=offset)
                .max(),
                "Low": df["Low"]
                .resample(rule, label="right", closed="left", offset=offset)
                .min(),
                "Close": df["Close"]
                .resample(rule, label="right", closed="left", offset=offset)
                .last(),
                "Volume": df["Volume"]
                .resample(rule, label="right", closed="left", offset=offset)
                .sum(),
            }
        ).dropna()

    return pd.DataFrame(
        {
            "Open": df["Open"].resample(rule, label="right", closed="left").first(),
            "High": df["High"].resample(rule, label="right", closed="left").max(),
            "Low": df["Low"].resample(rule, label="right", closed="left").min(),
            "Close": df["Close"].resample(rule, label="right", closed="left").last(),
            "Volume": df["Volume"].resample(rule, label="right", closed="left").sum(),
        }
    ).dropna()


def _merge_timeframe_features(
    df: pd.DataFrame, values: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    """Backward-merge a batch of timeframe features after each bar completes."""
    if timeframe != "1m":
        values = values.rename(columns=lambda name: f"{name}_{timeframe}")
    feature_frame = values.rename_axis("Timestamp").reset_index()
    return pd.merge_asof(
        df.sort_values("Timestamp"),
        feature_frame.sort_values("Timestamp"),
        on="Timestamp",
        direction="backward",
    )


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, session VWAP, and other indicator features.

    EMA distances: 1m, 3m, 5m, 15m, 30m, and 1h (spans 9, 21, 50).
    ATR, volatility, and RSI features are calculated on 1m, 3m, 5m, 15m, 30m,
    and 1h. VWAP distance and five-bar rate of change use each timeframe's
    candles and reset at the 09:15 Asia/Kolkata session open each day.
    ORB features use the first 15m, 30m, and 1h from the NSE session open.
    The completed-daily 10-session macro gate is calculated separately.
    """
    # Section 1: Prepare sorted one-minute candles and their timeframe closes.
    df = df.copy()
    if "Timestamp" in df.columns:
        timestamp = pd.to_datetime(df["Timestamp"])
        df = (
            df.assign(Timestamp=timestamp)
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    temp_df = df.set_index("Timestamp")
    df_3m = _resample_ohlcv(temp_df, "3min")
    df_5m = _resample_ohlcv(temp_df, "5min")
    df_15m = _resample_ohlcv(temp_df, "15min")
    df_30m = _resample_ohlcv(temp_df, "30min")
    df_1h = _resample_ohlcv(temp_df, "1h", offset="15min")
    timeframe_bars = {
        "1m": temp_df,
        "3m": df_3m,
        "5m": df_5m,
        "15m": df_15m,
        "30m": df_30m,
        # NSE session starts at 09:15, so hourly buckets use a 15-minute offset.
        "1h": df_1h,
    }

    # Section 2: Calculate every EMA first, then align completed higher-timeframe
    # values to the one-minute rows. EMA distances and the 1m pair spread are
    # normalized by price so they are scale-independent.
    for timeframe, bars in timeframe_bars.items():
        close = bars["Close"]
        if timeframe == "1m":
            for span in (9, 21, 50):
                ema = calculate_ema(close, span)
                ema_values = ema.to_numpy()
                df[f"EMA_{span}_1m"] = ema_values
                df[f"EMA_{span}_Dist"] = (
                    df["Close"].to_numpy() - ema_values
                ) / ema_values
            df["EMA_9_21_Spread"] = (
                df["EMA_9_1m"] - df["EMA_21_1m"]
            ) / df["EMA_21_1m"]
            df["EMA_9_21_Close_Spread"] = (
                df["EMA_9_1m"] - df["EMA_21_1m"]
            ) / df["Close"]
            continue

        ema_frame = pd.DataFrame(index=close.index)
        for span in (9, 21, 50):
            ema_frame[f"EMA_{span}_{timeframe}"] = calculate_ema(close, span)
        ema_frame = ema_frame.reset_index()
        ema_columns = [f"EMA_{span}_{timeframe}" for span in (9, 21, 50)]
        df = pd.merge_asof(
            df.sort_values("Timestamp"),
            ema_frame[["Timestamp", *ema_columns]].sort_values("Timestamp"),
            on="Timestamp",
            direction="backward",
        )
        for span in (9, 21, 50):
            ema_column = f"EMA_{span}_{timeframe}"
            df[f"{ema_column}_Dist"] = (df["Close"] - df[ema_column]) / df[ema_column]

    # Section 3: Register each timeframe-based indicator once. The loop applies
    # every registered calculation to every timeframe in timeframe_bars, then
    # aligns only completed higher-timeframe values to the one-minute rows.
    for timeframe, bars in timeframe_bars.items():
        indicator_values = {
            "ATR_Pct_14": calculate_atr_pct(bars, period=14),
            "ATR_Pct_7": calculate_atr_pct(bars, period=7),
            "Volatility_20": calculate_volatility(bars["Close"], period=20),
            "RSI_14_Norm": calculate_rsi(bars["Close"], period=14),
            "RSI_21_Norm": calculate_rsi(bars["Close"], period=21),
        }
        indicator_values.update(calculate_vwap(bars).to_dict(orient="series"))
        indicator_frame = pd.DataFrame(indicator_values)
        df = _merge_timeframe_features(df, indicator_frame, timeframe)

    # Opening ranges are measured from one-minute candles starting at the NSE
    # open. Each range is only exposed after its full opening window completes.
    for window_minutes, window_label in (
        (15, "15m"),
        (30, "30m"),
        (60, "1h"),
    ):
        orb_values = calculate_orb(
            temp_df, window_minutes, window_label=window_label
        )
        df = _merge_timeframe_features(df, orb_values, "1m")

    # Daily: gate on completed daily close vs 10-session SMA and its slope.
    daily_close = _resample_close(temp_df["Close"], "1D")
    daily_gate = calculate_macro_trend_gate(daily_close, period=10)
    daily_gate.index.name = "Timestamp"
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        daily_gate.rename("Macro_Trend_Gate").reset_index(),
        on="Timestamp",
        direction="backward",
    )

    return df.reset_index(drop=True)


def compute_normalized_features(
    df: pd.DataFrame, ticker: str, include_targets: bool = False
) -> pd.DataFrame:
    """Build return features, technical indicators, metadata, and optional labels."""
    df = df.copy()

    # Section 1: One-minute return features.
    df["Return_1m"] = df["Close"].pct_change(1)
    df["Return_5m"] = df["Close"].pct_change(5)
    df["Return_15m"] = df["Close"].pct_change(15)

    # Section 2: EMA stacks, timeframe-specific indicators, and daily gate.
    df = add_multi_timeframe_features(df)

    # Section 3: Ticker-level market-cap and sector flags.
    df = encode_categorical_metadata(df, ticker)

    # Section 4: Add forward labels only for training; keep the latest row for inference.
    required_columns = FEATURE_COLUMNS
    if include_targets:
        df["Future_Return_5m"] = df["Close"].shift(-5).pct_change(5)
        df["Target_Direction"] = (df["Future_Return_5m"] > 0).astype(float)
        required_columns = [*FEATURE_COLUMNS, "Future_Return_5m", "Target_Direction"]

    return df.dropna(subset=required_columns)


# Complete Feature Schema Definition
FEATURE_COLUMNS = [
    # Percentage Returns (3)
    "Return_1m",
    "Return_5m",
    "Return_15m",
    # Multi-timeframe session VWAP distance and five-bar slope
    "VWAP_Dist",
    "VWAP_ROC_5",
    "VWAP_Dist_3m",
    "VWAP_ROC_5_3m",
    "VWAP_Dist_5m",
    "VWAP_ROC_5_5m",
    "VWAP_Dist_15m",
    "VWAP_ROC_5_15m",
    "VWAP_Dist_30m",
    "VWAP_ROC_5_30m",
    "VWAP_Dist_1h",
    "VWAP_ROC_5_1h",
    # Multi-Timeframe EMA Distances
    ## 1m EMA Distances and 9/21 Spreads (5)
    "EMA_9_Dist",
    "EMA_21_Dist",
    "EMA_50_Dist",
    "EMA_9_21_Spread",
    "EMA_9_21_Close_Spread",
    ## 5m EMA Distances (3)
    "EMA_9_3m_Dist",
    "EMA_21_3m_Dist",
    "EMA_50_3m_Dist",
    ## 3m EMA Distances (3)
    "EMA_9_5m_Dist",
    "EMA_21_5m_Dist",
    "EMA_50_5m_Dist",
    ## 15m EMA Distances (3)
    "EMA_9_15m_Dist",
    "EMA_21_15m_Dist",
    "EMA_50_15m_Dist",
    ## 30m EMA Distances (3)
    "EMA_9_30m_Dist",
    "EMA_21_30m_Dist",
    "EMA_50_30m_Dist",
    ## 1h EMA Distances (3)
    "EMA_9_1h_Dist",
    "EMA_21_1h_Dist",
    "EMA_50_1h_Dist",
    # Multi-Timeframe ATR Context (6)
    "ATR_Pct_14",
    "ATR_Pct_14_3m",
    "ATR_Pct_14_5m",
    "ATR_Pct_14_15m",
    "ATR_Pct_14_30m",
    "ATR_Pct_14_1h",
    # Multi-Timeframe 7-bar ATR Context (6)
    "ATR_Pct_7",
    "ATR_Pct_7_3m",
    "ATR_Pct_7_5m",
    "ATR_Pct_7_15m",
    "ATR_Pct_7_30m",
    "ATR_Pct_7_1h",
    # Multi-Timeframe Volatility (6)
    "Volatility_20",
    "Volatility_20_3m",
    "Volatility_20_5m",
    "Volatility_20_15m",
    "Volatility_20_30m",
    "Volatility_20_1h",
    # Multi-Timeframe 14-bar RSI (6)
    "RSI_14_Norm",
    "RSI_14_Norm_3m",
    "RSI_14_Norm_5m",
    "RSI_14_Norm_15m",
    "RSI_14_Norm_30m",
    "RSI_14_Norm_1h",
    # Multi-Timeframe 21-bar RSI (6)
    "RSI_21_Norm",
    "RSI_21_Norm_3m",
    "RSI_21_Norm_5m",
    "RSI_21_Norm_15m",
    "RSI_21_Norm_30m",
    "RSI_21_Norm_1h",
    # Opening Range Breakout context (5 features per range)
    "ORB_15m_Range_Pct",
    "ORB_15m_High_Distance_Pct",
    "ORB_15m_Low_Distance_Pct",
    "ORB_15m_Midpoint_Distance_Pct",
    "ORB_15m_Breakout_Direction",
    "ORB_30m_Range_Pct",
    "ORB_30m_High_Distance_Pct",
    "ORB_30m_Low_Distance_Pct",
    "ORB_30m_Midpoint_Distance_Pct",
    "ORB_30m_Breakout_Direction",
    "ORB_1h_Range_Pct",
    "ORB_1h_High_Distance_Pct",
    "ORB_1h_Low_Distance_Pct",
    "ORB_1h_Midpoint_Distance_Pct",
    "ORB_1h_Breakout_Direction",
    # Macro Trend Gate (1)
    "Macro_Trend_Gate",
    # One-hot metadata feature columns generated from the metadata enums.
    *METADATA_FEATURE_COLUMNS,
]
