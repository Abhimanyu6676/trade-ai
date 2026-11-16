# Model Training Lifecycle

The training package has two directional specialist pipelines and one shared
PPO meta-agent. Long and short use the same multi-timeframe feature definitions
but separate label semantics and checkpoint directories. The meta-agent sees
frozen, chronological out-of-sample predictions from all four specialists.
Training only creates forecast and policy artifacts; it does not send broker
orders.

## Configuration and labels

`trade_ai/config/trading_config.py` is the source of truth for the strategy
schema, baseline barriers, supported bracket multipliers, holding period,
market times, model directories, and RL simulation costs. The current rules
are:

- Long take profit is +1.0% and stop loss is -0.5% from entry.
- Short take profit is -1.0% and stop loss is +0.5% from entry.
- Labels use `1` for take profit first, `-1` for stop loss first, and `0` for
  timeout for both directions. The classes describe the outcome of the
  direction-specific trade.
- Entries are allowed from 10:00 through 14:30. Future close checks stop at the
  earlier of 60 bars, the calendar-date boundary, or the 15:15 market close.
  The candle timestamp is at candle open, so the 15:14 candle is eligible and
  15:15 is excluded.

The base labelers always use the baseline barriers above. PPO selects a bracket
multiplier of 0.75, 1.0, or 1.25 at entry and scales both distances by that same
value, preserving the 2:1 target-to-stop ratio. Raw OHLCV stays in the
calculation frame and is excluded from all four base-model inputs.

Increment `SCHEMA_VERSION` whenever features, labels, direction semantics, RL
observation fields, or action meanings change. Loaders reject artifacts with an
older version and request retraining. Feature and label schema versions are
also kept in each manifest for specific compatibility errors. The PPO action
and observation contract is schema version 3. Because the shared trading schema
is recorded in the base-model manifests too, existing version-2 base models and
PPO artifacts must all be retrained before prediction resumes.

## Training selectors

`train_master_helper.py` supports these model and download flags:
`--long_lgb`, `--long_gru`, `--short_lgb`, `--short_gru`, `--meta_ensemble`,
`--download`, and `--all`. In an interactive terminal, the CLI always opens a
multi-option selector; any flags supplied on the command line are preselected
for confirmation and can be changed before training starts. In a non-interactive
terminal, explicit flags run directly because a selector cannot be shown. `--all`
downloads missing ticker history, trains all four specialists, then trains PPO.
Use `--download` with model flags to request the download step alongside selected
training stages. The command stores long and short specialist selections in
separate typed tuples. Canceling the selector or confirming an empty selection
exits with a clear message without starting a stage.

Examples:

```sh
# Request all supported model options
uv run python -m trade_ai.model.training.train_master --all

# Request only short-side GRU
uv run python -m trade_ai.model.training.train_master --short_gru

# Request only the shared policy
uv run python -m trade_ai.model.training.train_master --meta_ensemble
```

## Training execution

`train_master.py` runs every selected stage synchronously in the current
process. It does not initialize the inference model manager, back up or unload
loaded models, or reload artifacts afterward. Trainer modules manage their own
artifact writes and validation. This command is intended to run independently
on the training VPS; deploy its resulting artifacts to the inference VPS and
restart or reload that service separately.

The runner executes selected long specialists before selected short specialists,
and runs LightGBM before GRU within each direction. If a selected specialist
fails, the runner still attempts the other selected specialists. PPO runs after
selected base models and is skipped if any of those base stages fail. The CLI
does not create detached processes; use tmux to keep the terminal session running
after you detach. It does not set CPU affinity or thread limits, so LightGBM,
PyTorch, and other libraries use their own defaults.
Each base model must have exactly one registered trainer; duplicate registrations
and missing model handlers fail during CLI initialization. Stage failures are
logged and returned as unsuccessful results; stages that return normally log
their elapsed time and result.

Both trainers use the ticker universe in `model/stock_metadata.py`, load local
OHLCV CSVs through `training/util.py`, build direction-specific labels, and
attempt every ticker configured in `STOCK_METADATA`. Tickers with missing or
insufficient data are skipped and logged. One pooled calendar gives
every ticker the same whole-session boundaries: 55% initial fitting, the next
15% model selection, the next 20% in four expanding OOS forecast folds, and the
newest 10% as a final holdout. Up to 60 candle rows are purged before fit
boundaries so label look-ahead does not cross into the next period.

Both base-model collectors use shared helpers in `training/util.py`.
`prepare_directional_training_frame()` loads one ticker, builds the requested
model's features, and attaches direction-specific Triple Barrier targets.
`date_partition_row_bounds()` applies the shared date splits and fit purge;
`build_market_metadata()` selects the price and 1-minute context used by PPO.
The collectors keep model-specific sample construction: LightGBM uses
individual rows, while GRU builds same-session sequences and attaches each
target to its sequence endpoint.

LightGBM uses its native categorical `Cap` and `Sector` fields, selects the
boosting-round count on the middle validation period, and proceeds only if its
validation logloss beats the training class-prior baseline. GRU uses
same-calendar-date 30-candle windows, keeps unlabeled rows as context, and
selects its epoch by validation loss. With those settings fixed, each fold gets
a fresh model fitted on all eligible prior dates (with the boundary purge) and
predicts only its next chronological block. After fold generation, each family
refits on all eligible history before the newest 10%, using the selected round
or epoch count, and saves that model for inference. A LightGBM run that fails
its validation baseline gate does not replace its existing checkpoint or write
new prediction streams.

The fold predictions include ticker, timestamp, close, ATR, volatility ratio,
fold ID, and the exclusive model-training cutoff. The four OOS manifests record
schemas and final-checkpoint SHA-256 digests; PPO rejects missing, stale, or
misaligned streams. Final-holdout predictions are saved separately. Splits are
not randomized across rows or tickers: missing tickers are skipped, while all
eligible tickers use the same global session dates. GRU shuffles minibatches
within its fit partition, LightGBM uses its configured bagging/feature sampling,
and PPO samples episodes within its fit partition; these do not alter which
dates belong to fit, selection, OOS, or holdout.

## Unified PPO meta-agent

The Gymnasium environment observes all four three-class probability vectors,
long and short model agreement, scaled relative 1m ATR and volatility ratio,
portfolio position, unrealized return, bars held, and the active bracket
multiplier. ATR is volatility context for the policy; it does not set the
profit or stop distance. Its 19 discrete actions are flat or long/short at
0.5x/1x/1.5x size, crossed with bracket widths 0.75x/1.0x/1.25x. The flat
action is shared. At entry, the selected width scales both baseline barriers:
long TP +1.0% / SL -0.5%, short TP -1.0% / SL +0.5%. The selected multiplier
stays fixed while the position remains open. The 20-value observation and
19-action contract are stored in the policy sidecar and validated when loading.

The PPO inputs have bounded or relative scales: probabilities and agreement are
0–1, ATR is relative to close then clipped/scaled, the volatility ratio is
clipped/scaled, position is divided by the maximum size, unrealized portfolio
return is clipped/scaled, holding bars are divided by the 60-bar limit, and the
active bracket multiplier is mapped to [-1, 1] while in a position. There are
no raw price, volume, timestamp, or ticker-id observations.

The historical simulator uses close-to-close returns and the baseline
direction-specific barriers scaled by the selected entry multiplier, a 60-bar
cap, an estimated 5 bp one-way transaction cost, an entry penalty when the
side's models disagree, and an incremental drawdown penalty. It ends episodes
at the ticker's calendar-session boundary.
It does not model intrabar ordering, spreads, market impact, partial fills, or
broker execution. The 5 bp cost and reward weights are modeling assumptions in
the centralized config, not measured OpenAlgo costs.

PPO training and policy selection use only aligned predictions from the four
walk-forward OOS folds. The earlier 80% of their session dates is PPO fit data;
the later 20% selects the deterministic candidate. PPO retains a candidate only
when its mean reward on this selection period beats the flat zero-reward
baseline. After that decision, the trainer evaluates every available
ticker/session episode from the newest 10% final holdout and records the result
in the policy schema. The holdout score does not decide whether to save the
candidate. It is a post-selection report for that run, not evidence of live
profitability; repeated runs and manual tuning after viewing it can contaminate
the holdout.

The training modules run these chronological splits, purges, validation gates,
OOS generation, provenance checks, and final reporting automatically. This
reduces leakage risk but cannot identify look-ahead already present in upstream
data or feature calculations, prevent survivorship bias, or make repeated human
selection on the final holdout safe.

## Artifacts and loading

`training/util.py:download_missing_stock_data()` downloads missing ticker CSVs
when invoked. It uses the chunked downloader in the same module, which requests
adjacent inclusive 30-calendar-day OpenAlgo chunks with up to three attempts.
The current default download window ends 2026-07-31; a ticker listed after that
date, such as JPL, is skipped unless a newer local cache exists. Empty initial
chunks—including OpenAlgo's “no data available” response—are skipped until the
first data chunk. Any empty/no-data chunk after history begins rejects the full
download so a gapped CSV is not saved. If all chunks are empty, the ticker is
not cached. Failed downloads are not cached, and existing history caches are
not automatically refreshed. Model trainers load local CSVs only.

Artifacts are stored under `src/trade_ai/model/.run_cache/`:

- `long/lgb_model.txt` plus its schema sidecar.
- `long/gru_model.pth`.
- `short/lgb_model.txt` plus its schema sidecar.
- `short/gru_model.pth`.
- `rl/meta_agent.zip` plus `meta_agent.schema.json`.
- `rl/oos/` contains per-specialist `*_oos.csv` walk-forward predictions and
  `*_final.csv` final-holdout predictions, each with a schema sidecar.

The model manager validates trading, feature, label, direction, and ordered
input schemas for every base model. It loads PPO only when the observation and
action contracts match and the four checkpoint hashes equal the hashes recorded
in the policy manifest. After a base model changes, the existing PPO policy
will not load until PPO is retrained against those exact base files. The
training CLI does not back up the previous policy.

When `--meta_ensemble` is selected, PPO trains in the same process after
selected base stages succeed. A PPO-only request uses current OOS predictions
from all four specialists.

## Launching training

From the repository root, start a tmux session and run the selected command:

```sh
tmux new -s trade-ai-training
uv run python -m trade_ai.model.training.train_master --all
```

In an interactive terminal, the selector opens even when flags are supplied;
those choices start preselected for confirmation. In a non-interactive terminal,
explicit flags run directly. Training stays attached to the tmux pane. Detach
with `Ctrl+B`, then `D`; reconnect with `tmux attach -t trade-ai-training`.
Stop training with `Ctrl+C` in the tmux pane.
