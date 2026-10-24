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
the version and ordered input names.

## Model input inventory

The tables show every input name and its scaling style. The technical features
are shared by both models; RSI scaling differs. GRU has 31 inputs. LightGBM has
those 31 plus 10 metadata flags, for 41 total. Active timeframes are 1m, 5m,
15m, and 1h; 3m and 30m are excluded.

### Returns — 3 features per model

`pct_change()` returns a fractional ratio: `0.01` represents a 1% change.

| Feature | GRU style | LightGBM style |
| --- | --- | --- |
| `ROC_1m` | Relative return ratio | Relative return ratio |
| `ROC_5m` | Relative return ratio | Relative return ratio |
| `ROC_15m` | Relative return ratio | Relative return ratio |

### Multi-timeframe indicators — 28 features per model

| Timeframe | RSI | ATR | EMA 9 distance | EMA 21 distance | EMA 50 distance | FVG | Sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1m | `RSI_14_1m` (GRU: 0–1; LGB: 0–100) | `ATR_14_1m` (relative to close) | `EMA9_Dist_1m` (relative to EMA) | `EMA21_Dist_1m` (relative to EMA) | `EMA50_Dist_1m` (relative to EMA) | `FVG_1m` (binary 0/1) | `Sweep_1m` (binary 0/1) |
| 5m | `RSI_14_5m` (GRU: 0–1; LGB: 0–100) | `ATR_14_5m` (relative to close) | `EMA9_Dist_5m` (relative to EMA) | `EMA21_Dist_5m` (relative to EMA) | `EMA50_Dist_5m` (relative to EMA) | `FVG_5m` (binary 0/1) | `Sweep_5m` (binary 0/1) |
| 15m | `RSI_14_15m` (GRU: 0–1; LGB: 0–100) | `ATR_14_15m` (relative to close) | `EMA9_Dist_15m` (relative to EMA) | `EMA21_Dist_15m` (relative to EMA) | `EMA50_Dist_15m` (relative to EMA) | `FVG_15m` (binary 0/1) | `Sweep_15m` (binary 0/1) |
| 1h | `RSI_14_1h` (GRU: 0–1; LGB: 0–100) | `ATR_14_1h` (relative to close) | `EMA9_Dist_1h` (relative to EMA) | `EMA21_Dist_1h` (relative to EMA) | `EMA50_Dist_1h` (relative to EMA) | `FVG_1h` (binary 0/1) | `Sweep_1h` (binary 0/1) |

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

No model input is a raw price or volume value. ROC, ATR, and EMA inputs are
relative ratios. RSI is bounded, with model-specific ranges shown above. FVG,
Sweep, and LightGBM metadata flags are normalized to 0/1 but are not relative
to price; flag them in audits as intentional dimensionless/categorical inputs.
RSI is also dimensionless rather than price-relative; LightGBM's 0–100 range is
not the same as GRU's 0–1 scaling. Current non-relative inputs to call out are
the 8 FVG/Sweep flags in each model, plus the ten LightGBM metadata flags and
the model-specific RSI inputs.

