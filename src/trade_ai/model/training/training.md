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
   targets. Split each ticker chronologically into 80% training and 20%
   validation rows, then purge up to 60 feature rows before each validation
   boundary so training labels cannot inspect validation-period candles. Combine
   the per-ticker splits and train up to 500 rounds. Training and validation
   row counts and class distributions are logged; LightGBM reports both losses
   every 50 rounds and uses validation loss for early stopping after 50 rounds
   without improvement. For cache misses in
   either trainer, downloaders await OpenAlgo and request distinct inclusive
   30-calendar-day chunks, with up to three attempts per chunk. A chunk that
   fails all attempts aborts that ticker's download rather than using partial
   history. Save the checkpoint.
5. **Prepare and train GRU** when selected. For each ticker, split labeled rows
   chronologically into 80% training and 20% validation, then purge up to 60
   candle rows before the validation boundary so training labels cannot inspect
   validation-period candles. Rows with missing targets may provide sequence
   context, but each sequence must end on a labeled row and stay within one
   session. Each epoch logs per-ticker training and validation loss/accuracy,
   sequence counts, and the initial split's class counts, plus aggregate epoch
   metrics. Validation features and labels are recomputed one ticker at a time
   each epoch to keep peak memory bounded. GRU accepts an epoch limit from 10 through 20; it always trains at
   least 10 epochs, then stops if validation loss has not improved for 3
   consecutive epochs, or at the 20-epoch limit. It saves the checkpoint with
   the best validation loss. The checkpoint includes the feature schema version
   and ordered input-column names. Each trainer selects its model-specific
   schema; GRU excludes static metadata, LightGBM includes it, and both exclude
   raw OHLCV from model inputs. See `../features.md` for counts and full input
   lists. GRU training raises an error if no valid train and validation sequences
   are available.
6. **Reload selected models after success.** The `finally` block reloads the
   selected models only if every selected trainer returned successfully. If a
   trainer raises an error, reload is skipped and the selected models remain
   unloaded in the current process until they are explicitly reloaded or the app
   restarts. A schema mismatch during a successful run still leaves that model
   unloaded and requires retraining.

The default GRU run allows up to 20 epochs, with a minimum of 10, uses 30
candles per sequence, and batches of 256. Early stopping uses a patience of 3
epochs after the minimum training period.
Checkpoint files are under `src/trade_ai/model/.run_cache/`:
`lgb_trading_model_v{FEATURE_SCHEMA_VERSION}.txt` with its `.schema.json`
manifest, and `gru_trading_model_v{FEATURE_SCHEMA_VERSION}.pth` with schema
metadata in the checkpoint. See `../features.md` for the active feature schema
version and model input inventories.

Run training in the foreground from the repository root with:

```sh
uv run python -m trade_ai.model.training.train_master [gru|lgb|both]
```
