import pandas as pd


def calculate_ema(df: pd.DataFrame, period: int = 9) -> pd.Series:
    """Calculates Exponential Moving Average completely from scratch using standard pandas math."""
    if len(df) < period:
        return pd.Series(index=df.index, dtype="float64")

    close_prices = df["Close"]
    # Establish a baseline using a Simple Moving Average (SMA) for the first element
    sma = close_prices.iloc[:period].mean()

    # Create the EMA output container series
    ema_series = pd.Series(index=df.index, dtype="float64")
    ema_series.iloc[period - 1] = sma

    # Calculate the smoothing multiplier weight variable
    multiplier = 2 / (period + 1)

    # Compute the remaining elements sequentially based on exponential scaling formulas
    for i in range(period, len(df)):
        prev_ema = ema_series.iloc[i - 1]
        current_close = close_prices.iloc[i]
        ema_series.iloc[i] = (current_close - prev_ema) * multiplier + prev_ema

    return ema_series


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Wilder's Relative Strength Index from scratch using native lists to bypass Pandas locks."""
    if len(df) <= period:
        return pd.Series(index=df.index, dtype="float64")

    if "Close" not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    # 1. Day-to-day price differences
    delta = df["Close"].diff()

    # 2. Separate gains and absolute negative losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 3. Establish initial SMA rolling windows baselines
    avg_gain_series = gain.rolling(window=period, min_periods=period).mean()
    avg_loss_series = loss.rolling(window=period, min_periods=period).mean()

    # 4. Convert to native Python lists to ensure 100% write capability
    avg_gain_list = avg_gain_series.tolist()
    avg_loss_list = avg_loss_series.tolist()
    gain_list = gain.tolist()
    loss_list = loss.tolist()

    # 5. Dynamically isolate the first index containing a valid calculated window baseline
    first_valid_idx = None
    for idx, val in enumerate(avg_gain_list):
        # Scan through the initial rows until we hit a real numerical baseline
        if val is not None and str(val) != "nan":
            first_valid_idx = idx
            break

    # If no valid baseline window could be computed, abort calculation safely
    if first_valid_idx == None:
        return pd.Series(index=df.index, dtype="float64")

    # 6. Run Wilder's Exponential Smoothing Loop using raw Python variables
    for i in range(first_valid_idx + 1, len(df)):
        avg_gain_list[i] = (avg_gain_list[i - 1] * (period - 1) + gain_list[i]) / period
        avg_loss_list[i] = (avg_loss_list[i - 1] * (period - 1) + loss_list[i]) / period

    # 7. Convert finished arrays back into Pandas series instances
    final_avg_gain = pd.Series(avg_gain_list, index=df.index)
    final_avg_loss = pd.Series(avg_loss_list, index=df.index)

    # 8. Compute the final Relative Strength (RS) ratio and translate it into a 0-100 bound scale
    rs = final_avg_gain / final_avg_loss.replace(0, 0.00001)
    rsi = 100 - (100 / (1 + rs))

    return rsi
