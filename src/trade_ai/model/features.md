## Feature rules

- Do not include raw OHLCV values as model inputs. Keep candles in the source
  frame for feature calculations only.
- Every feature must use an appropriate normalized or relative scale. If a
  feature is unnormalized or non-relative, flag it and explain whether it is an
  intentional bounded, binary, or categorical input.
- Keep static ticker metadata out of GRU inputs. LightGBM receives fixed-width
  integer codes for `Cap` and `Sector`, declared as native categorical inputs.
- On every feature/schema change, update both model counts and their complete
  feature/style lists below, even when only one model's input changes.

`features.py` defines `BASE_FEATURE_COLUMNS`, `GRU_MODEL_INPUT_COLUMNS`, and
`LGB_MODEL_INPUT_COLUMNS`. `compute_normalized_features(..., model_type=...)`
encodes metadata for LightGBM and removes metadata columns for GRU. Training,
inference, and checkpoint validation use the corresponding model-specific input
tuple. Checkpoint filenames include the schema version; checkpoints also store
the version and ordered input names. The current feature schema version is 6.

## Model input inventory

The tables show every input name and its scaling style. The technical features
are shared by both models; RSI scaling differs. GRU has 35 inputs. LightGBM has
those 35 plus two categorical metadata inputs, for 37 total. Active timeframes are 1m, 5m,
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

### LightGBM-only metadata — 2 features

| Feature | LightGBM style | GRU |
| --- | --- | --- |
| `Cap` | Fixed integer category code; native categorical | Excluded |
| `Sector` | Fixed integer category code; native categorical | Excluded |

`Cap` codes are large=0, mid=1, small=2. `Sector` codes are banking=0,
IT=1, auto=2, energy=3, consumer=4, metals=5, other=6. Keep existing codes
fixed and append codes when adding categories. Unknown tickers use `Other`.
When a new sector is added, update the enum, metadata mapping, and category code,
then retrain LightGBM so it can learn that category. Feature width remains fixed.

## Scaling audit

No model input is a raw price or volume value. ROC, ATR, EMA, and volatility
ratio inputs are relative/dimensionless ratios. RSI is bounded, with model-
specific ranges shown above. FVG and Sweep inputs are binary 0/1 flags. `Cap`
and `Sector` are non-relative integer identifiers explicitly declared as
categorical in LightGBM; they are not continuous measurements. RSI is also
dimensionless rather than price-relative; LightGBM's 0–100 range is not the
same as GRU's 0–1 scaling. Current non-relative inputs include the 8 FVG/Sweep
flags in each model, the two LightGBM category codes, and model-specific RSI.

