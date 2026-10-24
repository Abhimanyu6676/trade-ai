# Model Training Lifecycle

`train_master.py` accepts `gru`, `lgb`, or `both` (default). With `both`, it
trains LightGBM first and GRU second.

## Steps

1. **Select models.** Resolve the requested selection to the model names to train.
2. **Back up checkpoints.** Copy existing selected model files to timestamped
   backups, including LightGBM schema manifests. Missing files are skipped.
3. **Unload selected models.** Clear their in-memory manager references before
   training.
4. **Prepare and train LightGBM** when selected. For each ticker, load cached
   OHLCV history or fetch and cache it. Normalize candle columns, parse and
   sort timestamps oldest-to-newest, and keep only the last row per timestamp
   before computing features and Triple Barrier labels. Drop rows with missing
   targets. Combine ticker data, make an unshuffled 80/20 train/validation
   split, and train up to 500 rounds with early stopping. Save the checkpoint.
5. **Prepare and train GRU** when selected. For each ticker and epoch, load
   history with the same normalization, chronological sorting, and timestamp
   deduplication, then compute features and labels and build same-session
   sequences. Rows with missing targets may be sequence context, but each
   sequence must end on a labeled row. Train the two-layer GRU in batches; save
   its state after each successful epoch. The checkpoint includes the feature
   schema version and ordered input-column names. Each trainer selects its
   model-specific schema; GRU excludes static metadata, LightGBM includes it,
   and both exclude raw OHLCV from model inputs. See `../features.md` for counts
   and full input lists.
6. **Reload selected models.** A `finally` block validates checkpoint schema
   metadata and reloads selected models, including when training raises an
   error. A schema mismatch leaves that model unloaded and requires retraining.

The default GRU run uses 10 epochs, 30 candles per sequence, and batches of 256.
Checkpoint files are under `src/trade_ai/model/.run_cache/`:
`lgb_trading_model_v{FEATURE_SCHEMA_VERSION}.txt` with its `.schema.json`
manifest, and `gru_trading_model_v{FEATURE_SCHEMA_VERSION}.pth` with schema
metadata in the checkpoint. See `../features.md` for the active feature schema
version and model input inventories.

Run training in the foreground from the repository root with:

```sh
uv run python -m trade_ai.model.training.train_master [gru|lgb|both]
```
