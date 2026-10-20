import pandas as pd

# Metadata Registry for your stock basket (Market Cap & Sector mapping)
STOCK_METADATA = {
    # Large Caps
    "RELIANCE": {"cap": "large", "sector": "energy"},
    "TCS": {"cap": "large", "sector": "it"},
    "INFY": {"cap": "large", "sector": "it"},
    "SBIN": {"cap": "large", "sector": "banking"},
    "HDFCBANK": {"cap": "large", "sector": "banking"},
    "TMPV": {"cap": "large", "sector": "auto"},
    # "HINDZINC": {"cap": "large", "sector": ""},
    # Mid Caps
    "FEDERALBNK": {"cap": "mid", "sector": "banking"},
    "COFORGE": {"cap": "mid", "sector": "it"},
    "VOLTAS": {"cap": "mid", "sector": "consumer"},
    # Small Caps
    "SUZLON": {"cap": "small", "sector": "energy"},
    "KPITTECH": {"cap": "small", "sector": "it"},
}


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates the Exponential Moving Average (EMA) for a price series."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI) normalized between 0.0 and 1.0."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))) / 100.0


def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Modular function: Calculates rolling standard deviation of percentage returns (Volatility)."""
    returns = series.pct_change(1)
    return returns.rolling(window=period).std()


def calculate_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Modular function: Calculates Normalized Average True Range (ATR Percentage) for OHLC data."""
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / df["Close"]


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


def add_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples base candles into 5-minute and 30-minute views,

    calculating full EMA stacks (9, 21, 50), ATR, and Volatility.
    """
    df = df.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp")

    temp_df = df.set_index("Timestamp")

    # 1. 5-Minute Resampling (EMA 9, 21 Stack + ATR Pct)
    df_5m = pd.DataFrame(
        {
            "Open": temp_df["Open"].resample("5Min").first(),
            "High": temp_df["High"].resample("5Min").max(),
            "Low": temp_df["Low"].resample("5Min").min(),
            "Close": temp_df["Close"].resample("5Min").last(),
            "Volume": temp_df["Volume"].resample("5Min").sum(),
        }
    ).dropna()

    df_5m["EMA_9_5m"] = calculate_ema(df_5m["Close"], 9)
    df_5m["EMA_21_5m"] = calculate_ema(df_5m["Close"], 21)
    df_5m["ATR_Pct_5m"] = calculate_atr_pct(df_5m, 14)
    df_5m = df_5m.reset_index()

    # 2. 30-Minute Resampling (Full EMA 9, 21, 50 Stack + Volatility)
    df_30m = pd.DataFrame(
        {
            "Open": temp_df["Open"].resample("30Min").first(),
            "High": temp_df["High"].resample("30Min").max(),
            "Low": temp_df["Low"].resample("30Min").min(),
            "Close": temp_df["Close"].resample("30Min").last(),
            "Volume": temp_df["Volume"].resample("30Min").sum(),
        }
    ).dropna()

    df_30m["EMA_9_30m"] = calculate_ema(df_30m["Close"], 9)
    df_30m["EMA_21_30m"] = calculate_ema(df_30m["Close"], 21)
    df_30m["EMA_50_30m"] = calculate_ema(df_30m["Close"], 50)
    df_30m["Volatility_30m"] = calculate_volatility(df_30m["Close"], 20)
    df_30m = df_30m.reset_index()

    df = df.reset_index(drop=True)

    # Merge using merge_asof (backward direction to prevent lookahead bias)
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        df_5m[["Timestamp", "EMA_9_5m", "EMA_21_5m", "ATR_Pct_5m"]].sort_values(
            "Timestamp"
        ),
        on="Timestamp",
        direction="backward",
    )
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        df_30m[
            [
                "Timestamp",
                "EMA_9_30m",
                "EMA_21_30m",
                "EMA_50_30m",
                "Volatility_30m",
            ]
        ].sort_values("Timestamp"),
        on="Timestamp",
        direction="backward",
    )

    # Backfill missing initial warmup periods
    df[
        [
            "EMA_9_5m",
            "EMA_21_5m",
            "ATR_Pct_5m",
            "EMA_9_30m",
            "EMA_21_30m",
            "EMA_50_30m",
            "Volatility_30m",
        ]
    ] = df[
        [
            "EMA_9_5m",
            "EMA_21_5m",
            "ATR_Pct_5m",
            "EMA_9_30m",
            "EMA_21_30m",
            "EMA_50_30m",
            "Volatility_30m",
        ]
    ].bfill()

    # Scale-agnostic relative distance metrics for higher TF EMAs
    df["EMA_9_5m_Dist"] = (df["Close"] - df["EMA_9_5m"]) / df["EMA_9_5m"]
    df["EMA_21_5m_Dist"] = (df["Close"] - df["EMA_21_5m"]) / df["EMA_21_5m"]
    df["EMA_9_30m_Dist"] = (df["Close"] - df["EMA_9_30m"]) / df["EMA_9_30m"]
    df["EMA_21_30m_Dist"] = (df["Close"] - df["EMA_21_30m"]) / df["EMA_21_30m"]
    df["EMA_50_30m_Dist"] = (df["Close"] - df["EMA_50_30m"]) / df["EMA_50_30m"]

    # 10-Day Macro Trend Gate
    macro_span = min(len(df), 3750)
    macro_ema = calculate_ema(df["Close"], span=macro_span)
    df["Macro_Trend_Gate"] = (df["Close"] > macro_ema).astype(float)

    return df


def compute_normalized_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Computes all normalized percentage features, multi-timeframe indicators,

    volatility, ATR, and category metadata tags.
    """
    df = df.copy()

    # 1. Percentage Returns
    df["Return_1m"] = df["Close"].pct_change(1)
    df["Return_5m"] = df["Close"].pct_change(5)
    df["Return_15m"] = df["Close"].pct_change(15)

    # 2. Base EMA Distances
    ema_9 = calculate_ema(df["Close"], span=9)
    ema_21 = calculate_ema(df["Close"], span=21)
    ema_50 = calculate_ema(df["Close"], span=50)

    df["EMA_9_Dist"] = (df["Close"] - ema_9) / ema_9
    df["EMA_21_Dist"] = (df["Close"] - ema_21) / ema_21
    df["EMA_50_Dist"] = (df["Close"] - ema_50) / ema_50

    # 3. Base Volatility & ATR Percentage using modular functions
    df["Volatility_20"] = calculate_volatility(df["Close"], period=20)
    df["ATR_Pct"] = calculate_atr_pct(df, period=14)

    # 4. Bounded Oscillator
    df["RSI_Norm"] = calculate_rsi(df["Close"], period=14)

    # 5. Multi-Timeframe Stacks & Macro Gate
    df = add_multi_timeframe_features(df)

    # 6. Categorical Metadata Encoding
    df = encode_categorical_metadata(df, ticker)

    # 7. Target Labels for Training
    df["Future_Return_5m"] = df["Close"].shift(-5).pct_change(5)
    df["Target_Direction"] = (df["Future_Return_5m"] > 0).astype(float)

    return df.dropna()


# Complete Expanded Feature Schema Definition (27 Features)
FEATURE_COLUMNS = [
    # Percentage Returns (3)
    "Return_1m",
    "Return_5m",
    "Return_15m",
    # Base EMA Distances (3)
    "EMA_9_Dist",
    "EMA_21_Dist",
    "EMA_50_Dist",
    # Volatility & ATR Context (4)
    "Volatility_20",
    "ATR_Pct",
    "ATR_Pct_5m",
    "Volatility_30m",
    # Bounded Oscillators (1)
    "RSI_Norm",
    # Multi-Timeframe 5m EMA Distances (2)
    "EMA_9_5m_Dist",
    "EMA_21_5m_Dist",
    # Multi-Timeframe 30m EMA Distances (3)
    "EMA_9_30m_Dist",
    "EMA_21_30m_Dist",
    "EMA_50_30m_Dist",
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
