import pandas as pd

from trade_ai.indicator import (
    calculate_atr_pct,
    calculate_ema,
    calculate_macd,
    calculate_macro_trend_gate,
    calculate_rsi,
    calculate_volatility,
)

# Metadata Registry for your stock basket (Market Cap & Sector mapping)
STOCK_METADATA = {
    # Large Caps
    "RELIANCE": {"cap": "large", "sector": "energy"},
    "TCS": {"cap": "large", "sector": "it"},
    "INFY": {"cap": "large", "sector": "it"},
    "SBIN": {"cap": "large", "sector": "banking"},
    "HDFCBANK": {"cap": "large", "sector": "banking"},
    "TMPV": {"cap": "large", "sector": "auto"},
    # "HINDZINC": {"cap": "large", "sector": "metals"},
    # Mid Caps
    "FEDERALBNK": {"cap": "mid", "sector": "banking"},
    "COFORGE": {"cap": "mid", "sector": "it"},
    "VOLTAS": {"cap": "mid", "sector": "consumer"},
    # Small Caps
    "SUZLON": {"cap": "small", "sector": "energy"},
    "KPITTECH": {"cap": "small", "sector": "it"},
}


def encode_categorical_metadata(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Adds scale-agnostic One-Hot Encoded market-cap and sector flags."""
    meta = STOCK_METADATA.get(ticker.upper(), {"cap": "mid", "sector": "other"})

    df["Cap_Large"] = 1.0 if meta["cap"] == "large" else 0.0
    df["Cap_Mid"] = 1.0 if meta["cap"] == "mid" else 0.0
    df["Cap_Small"] = 1.0 if meta["cap"] == "small" else 0.0

    df["Sector_Banking"] = 1.0 if meta["sector"] == "banking" else 0.0
    df["Sector_IT"] = 1.0 if meta["sector"] == "it" else 0.0
    df["Sector_Auto"] = 1.0 if meta["sector"] == "auto" else 0.0
    df["Sector_Energy"] = 1.0 if meta["sector"] == "energy" else 0.0
    df["Sector_Other"] = (
        1.0 if meta["sector"] not in ["banking", "it", "auto", "energy"] else 0.0
    )

    return df


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


def _merge_timeframe_series(
    df: pd.DataFrame, values: pd.Series, feature_name: str
) -> pd.DataFrame:
    """Backward-merge a timeframe series after its bar has completed."""
    feature_frame = values.rename(feature_name).rename_axis("Timestamp").reset_index()
    return pd.merge_asof(
        df.sort_values("Timestamp"),
        feature_frame.sort_values("Timestamp"),
        on="Timestamp",
        direction="backward",
    )


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA and other indicator features, organized by calculation stage.

    EMA distances: 1m, 3m, 5m, 15m, 30m, and 1h (spans 9, 21, 50).
    ATR, volatility, and RSI are calculated on 1m, 3m, 5m, 15m, 30m, and 1h.
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
    # values to the one-minute rows. All EMA model inputs are close-to-EMA ratios.
    for timeframe, bars in timeframe_bars.items():
        close = bars["Close"]
        if timeframe == "1m":
            for span in (9, 21, 50):
                ema = calculate_ema(close, span)
                df[f"EMA_{span}_Dist"] = (df["Close"] - ema) / ema
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
        macd_values = calculate_macd(
            bars["Close"], fast_period=12, slow_period=26, signal_period=9
        )
        indicator_values.update(
            {
                "MACD_Line_Pct": macd_values["MACD_Line_Pct"],
                "MACD_Signal_Pct": macd_values["MACD_Signal_Pct"],
                "MACD_Hist_Pct": macd_values["MACD_Hist_Pct"],
            }
        )
        for base_name, values in indicator_values.items():
            feature_name = (
                base_name if timeframe == "1m" else f"{base_name}_{timeframe}"
            )
            df = _merge_timeframe_series(df, values, feature_name)

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


# Complete Feature Schema Definition (78 Features)
FEATURE_COLUMNS = [
    # Percentage Returns (3)
    "Return_1m",
    "Return_5m",
    "Return_15m",
    # Multi-Timeframe EMA Distances
    ## Base EMA Distances (3)
    "EMA_9_Dist",
    "EMA_21_Dist",
    "EMA_50_Dist",
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
    # Relative MACD (12, 26, 9), all timeframes (18)
    "MACD_Line_Pct",
    "MACD_Signal_Pct",
    "MACD_Hist_Pct",
    "MACD_Line_Pct_3m",
    "MACD_Signal_Pct_3m",
    "MACD_Hist_Pct_3m",
    "MACD_Line_Pct_5m",
    "MACD_Signal_Pct_5m",
    "MACD_Hist_Pct_5m",
    "MACD_Line_Pct_15m",
    "MACD_Signal_Pct_15m",
    "MACD_Hist_Pct_15m",
    "MACD_Line_Pct_30m",
    "MACD_Signal_Pct_30m",
    "MACD_Hist_Pct_30m",
    "MACD_Line_Pct_1h",
    "MACD_Signal_Pct_1h",
    "MACD_Hist_Pct_1h",
    # Macro Trend Gate (1)
    "Macro_Trend_Gate",
    # Market Cap Categories (3)
    "Cap_Large",
    "Cap_Mid",
    "Cap_Small",
    # Sector Categories (5)
    "Sector_Banking",
    "Sector_IT",
    "Sector_Auto",
    "Sector_Energy",
    "Sector_Other",
]
