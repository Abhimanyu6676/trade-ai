## Feature rules

- Do not include raw OHLCV values as model inputs. Keep candles in the source
  frame for feature calculations only.
- Every feature must use an appropriate normalized or relative scale. If a
  feature is unnormalized or non-relative, flag it and explain whether it is an
  intentional bounded, binary, or categorical input.
- Keep static ticker metadata out of GRU inputs. LightGBM receives the metadata
  one-hot flags.
- On every feature/schema change, update both model counts and their complete
  feature/style lists below, even when only one model's input changes.

`features.py` defines `BASE_FEATURE_COLUMNS`, `GRU_MODEL_INPUT_COLUMNS`, and
`LGB_MODEL_INPUT_COLUMNS`. `compute_normalized_features(..., model_type=...)`
encodes metadata for LightGBM and removes metadata columns for GRU. Training,
inference, and checkpoint validation use the corresponding model-specific input
tuple. Checkpoint filenames include the schema version; checkpoints also store
the version and ordered input names. The current feature schema version is 5.

## Model input inventory

The tables show every input name and its scaling style. The technical features
are shared by both models; RSI scaling differs. GRU has 35 inputs. LightGBM has
those 35 plus 10 metadata flags, for 45 total. Active timeframes are 1m, 5m,
15m, and 1h; 3m and 30m are excluded.

### Returns — 3 features per model

`pct_change()` returns a fractional ratio: `0.01` represents a 1% change.

| Feature | GRU style | LightGBM style |
| --- | --- | --- |
| `ROC_1m` | Relative return ratio | Relative return ratio |
| `ROC_5m` | Relative return ratio | Relative return ratio |
| `ROC_15m` | Relative return ratio | Relative return ratio |

### Multi-timeframe indicators — 32 features per model

The volatility ratio is rolling standard deviation of one-bar percentage
returns over 10 bars divided by the same measure over 30 bars. Periods are bars
of the given timeframe (so 10 bars on 5m data spans 50 minutes).

| Timeframe | Feature | Input name | GRU style | LightGBM style |
| --- | --- | --- | --- | --- |
| 1m | RSI | `RSI_14_1m` | Bounded 0–1 | Bounded 0–100 |
| 1m | ATR | `ATR_14_1m` | Relative to close | Relative to close |
| 1m | EMA 9 distance | `EMA9_Dist_1m` | Relative to EMA | Relative to EMA |
| 1m | EMA 21 distance | `EMA21_Dist_1m` | Relative to EMA | Relative to EMA |
| 1m | EMA 50 distance | `EMA50_Dist_1m` | Relative to EMA | Relative to EMA |
| 1m | FVG | `FVG_1m` | Binary 0/1 | Binary 0/1 |
| 1m | Sweep | `Sweep_1m` | Binary 0/1 | Binary 0/1 |
| 1m | Volatility ratio | `Vol_Ratio_10_30_1m` | Relative volatility ratio | Relative volatility ratio |
| 5m | RSI | `RSI_14_5m` | Bounded 0–1 | Bounded 0–100 |
| 5m | ATR | `ATR_14_5m` | Relative to close | Relative to close |
| 5m | EMA 9 distance | `EMA9_Dist_5m` | Relative to EMA | Relative to EMA |
| 5m | EMA 21 distance | `EMA21_Dist_5m` | Relative to EMA | Relative to EMA |
| 5m | EMA 50 distance | `EMA50_Dist_5m` | Relative to EMA | Relative to EMA |
| 5m | FVG | `FVG_5m` | Binary 0/1 | Binary 0/1 |
| 5m | Sweep | `Sweep_5m` | Binary 0/1 | Binary 0/1 |
| 5m | Volatility ratio | `Vol_Ratio_10_30_5m` | Relative volatility ratio | Relative volatility ratio |
| 15m | RSI | `RSI_14_15m` | Bounded 0–1 | Bounded 0–100 |
| 15m | ATR | `ATR_14_15m` | Relative to close | Relative to close |
| 15m | EMA 9 distance | `EMA9_Dist_15m` | Relative to EMA | Relative to EMA |
| 15m | EMA 21 distance | `EMA21_Dist_15m` | Relative to EMA | Relative to EMA |
| 15m | EMA 50 distance | `EMA50_Dist_15m` | Relative to EMA | Relative to EMA |
| 15m | FVG | `FVG_15m` | Binary 0/1 | Binary 0/1 |
| 15m | Sweep | `Sweep_15m` | Binary 0/1 | Binary 0/1 |
| 15m | Volatility ratio | `Vol_Ratio_10_30_15m` | Relative volatility ratio | Relative volatility ratio |
| 1h | RSI | `RSI_14_1h` | Bounded 0–1 | Bounded 0–100 |
| 1h | ATR | `ATR_14_1h` | Relative to close | Relative to close |
| 1h | EMA 9 distance | `EMA9_Dist_1h` | Relative to EMA | Relative to EMA |
| 1h | EMA 21 distance | `EMA21_Dist_1h` | Relative to EMA | Relative to EMA |
| 1h | EMA 50 distance | `EMA50_Dist_1h` | Relative to EMA | Relative to EMA |
| 1h | FVG | `FVG_1h` | Binary 0/1 | Binary 0/1 |
| 1h | Sweep | `Sweep_1h` | Binary 0/1 | Binary 0/1 |
| 1h | Volatility ratio | `Vol_Ratio_10_30_1h` | Relative volatility ratio | Relative volatility ratio |

### LightGBM-only metadata — 10 features

| Feature | LightGBM style | GRU |
| --- | --- | --- |
| `Cap_Large` | One-hot flag 0/1 | Excluded |
| `Cap_Mid` | One-hot flag 0/1 | Excluded |
| `Cap_Small` | One-hot flag 0/1 | Excluded |
| `Sector_Banking` | One-hot flag 0/1 | Excluded |
| `Sector_IT` | One-hot flag 0/1 | Excluded |
| `Sector_Auto` | One-hot flag 0/1 | Excluded |
| `Sector_Energy` | One-hot flag 0/1 | Excluded |
| `Sector_Consumer` | One-hot flag 0/1 | Excluded |
| `Sector_Metals` | One-hot flag 0/1 | Excluded |
| `Sector_Other` | One-hot flag 0/1 | Excluded |

## Scaling audit

No model input is a raw price or volume value. ROC, ATR, EMA, and volatility
ratio inputs are relative/dimensionless ratios. RSI is bounded, with model-
specific ranges shown above. FVG,
Sweep, and LightGBM metadata flags are normalized to 0/1 but are not relative
to price; flag them in audits as intentional dimensionless/categorical inputs.
RSI is also dimensionless rather than price-relative; LightGBM's 0–100 range is
not the same as GRU's 0–1 scaling. Current non-relative inputs to call out are
the 8 FVG/Sweep flags in each model, plus the ten LightGBM metadata flags and
the model-specific RSI inputs.

