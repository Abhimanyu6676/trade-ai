Earlier, we established a strict rule: LightGBM gets sector/metadata columns, but the GRU should stay pure and never see static categorical metadata.


Right now, your code injects ⁠METADATA_FEATURE_COLUMNS⁠ into the dataframe regardless of whether ⁠model_type⁠ is ⁠"lgb"⁠ or ⁠"gru"⁠. If you pass this dataframe into your GRU, it will receive static columns (like sector flags) at every time step, which bloats its input space.

## Suggested Fix:

Conditionally include ⁠METADATA_FEATURE_COLUMNS⁠ based on the ⁠model_type⁠:

python
```
def compute_normalized_features(
    df: pd.DataFrame,
    ticker: str,
    model_type: Literal["gru", "lgb"],
    timeframes=None,
    ema_spans=None,
    relative: bool = True,
    include_targets: bool = False,
) -> pd.DataFrame:
    df = lowercase_ohlcv_columns(df)

    # Section 1: 1m, 5m, 15m Rate of Change features.
    df["ROC_1m"] = df["close"].pct_change(1)
    df["ROC_5m"] = df["close"].pct_change(5)
    df["ROC_15m"] = df["close"].pct_change(15)

    # Section 2: Multi-timeframe technical indicators.
    df = _add_multi_timeframe_features(df, model_type, timeframes, ema_spans, relative)

    # Section 3: Metadata (ONLY for LightGBM, strictly excluded for GRU)
    active_feature_columns = BASE_FEATURE_COLUMNS.copy()
    if model_type == "lgb":
        df = encode_categorical_metadata(df, ticker)
        active_feature_columns.extend(METADATA_FEATURE_COLUMNS)
    else:
        # Explicitly drop or ensure metadata columns aren't expected by GRU
        for col in METADATA_FEATURE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Section 4: Add forward labels for training
    required_columns = active_feature_columns
    if include_targets:
        df["Future_ROC_5m"] = df["close"].shift(-5).pct_change(5)
        df["Target_Direction"] = (df["Future_ROC_5m"] > 0).astype(float)
        required_columns = [*active_feature_columns, "Future_ROC_5m", "Target_Direction"]

    return df.dropna(subset=required_columns)
```


To properly enforce our rule—keeping static categorical metadata in LightGBM while stripping them out for the GRU—we need to split ⁠FEATURE_COLUMNS⁠ into base technical features and metadata features, and conditionally apply them based on the ⁠model_type⁠.

