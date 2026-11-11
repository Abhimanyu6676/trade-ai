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
`--download`, and `--all`. In a foreground interactive invocation,
`parse_training_args()` shows the multi-option selector and returns the selected
flags. A detached invocation includes `--worker`; the parser then returns its
flags directly without displaying the selector.

`startInBackground()` launches the same `train_master` module with every
selected flag plus `--worker`, using the current Python interpreter and a
detached session. The child converts the parsed flags to `TRAINING_COMMAND` and
calls `backgroundTrainingWorker()`. That handler is currently a placeholder;
dispatching download and model-training stages remains pending.

`rl` always trains the one unified policy against all four current specialists,
even when the first selector is only one side. A long-only or short-only `all`
run therefore requires compatible OOS streams for the opposite side before the
common policy can train.

Examples:

```sh
# Request all supported model options
uv run python -m trade_ai.model.training.train_master --all

# Request only short-side GRU
uv run python -m trade_ai.model.training.train_master --short_gru

# Request only the shared policy
uv run python -m trade_ai.model.training.train_master --meta_ensemble
```

## Base specialist lifecycle

The planned `train_master.py` dispatcher will back up selected checkpoints and
sidecars, unload selected models, and launch one isolated worker per selected
direction. The current `backgroundTrainingWorker()` is still a placeholder, so
this orchestration is not active yet. Once connected, `both` is intended to run
the long and short workers concurrently on disjoint CPU affinity masks when the
platform supports affinity and at least two CPUs are assigned. Each direction
worker will run its selected LightGBM stage followed by its selected GRU stage.

The planned `TRADE_AI_BASE_TRAINING_CPUS` setting controls the total logical CPU
budget across concurrent direction workers. It defaults to all logical CPUs available in the worker's
current affinity mask. CPU IDs are split into disjoint sets as evenly as
possible. If the budget is one CPU, or CPU affinity is unavailable, direction
workers run sequentially. If a configured budget exceeds the available CPU
count, the worker logs a warning and caps it to the available count. Invalid or
non-positive values are logged and fall back to the default. Each child applies
its CPU affinity before importing training libraries, sets OpenMP/MKL/OpenBLAS/
NumExpr thread limits to its assigned CPU count, configures PyTorch intra-op
threads to that count, and sets LightGBM's `num_threads` to that count.
Set these variables in the repository-root `.env` or in the shell/service
environment that launches the CLI. Shell variables take precedence, and the
training launcher passes the resolved values to its workers. For a four-CPU
machine, one explicit configuration is:

```sh
export TRADE_AI_BASE_TRAINING_CPUS=4
export TRADE_AI_PPO_CPUS=3
```

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
in the policy manifest. After a base model changes, the previous policy is
backed up and will not be used until retrained against those exact base files.

Planned orchestration runs PPO in its own child process only after all selected
base workers succeed. This stage is not connected to `backgroundTrainingWorker()` yet.
`TRADE_AI_PPO_CPUS` sets its logical CPU budget and defaults to three CPUs,
capped to the available affinity mask. A configured value above the available
count is logged and capped. The PPO child pins its process where supported and
applies the same native-library and PyTorch thread limits before training.
RL-only requests use this PPO worker directly with existing OOS data.

## Launching training

From the repository root, pass model flags to the launcher. For example:

```sh
uv run python -m trade_ai.model.training.train_master --all
```

In an interactive foreground shell, the launcher opens the multi-option
selector. It then prints the detached worker PID and log path. Standard output
and errors go to `logs/FS_<feature-version>LS_<label-version>_<timestamp>/worker.log`.
The same directory contains `worker.pid` with the detached worker's process ID.
The launcher sets common native thread-pool and training CPU budget variables to
the number of CPUs available to it, overriding inherited one-core limits. The
worker handler is still a placeholder, so model training and downloading do not
start until stage dispatch is connected.
