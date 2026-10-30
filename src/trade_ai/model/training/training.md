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

Run `train_master.py` with two optional positional arguments:

1. Mindset: `long`, `short`, or `both` (default `both`).
2. Component: `gru`, `lgb`, `rl`, or `all` (default `all`).

The launcher validates both selectors before starting. In an interactive
foreground shell, no arguments opens selectors for both values; supplied
arguments are shown for yes/no confirmation. In a non-interactive shell or
background launch, omitted arguments use the defaults and supplied arguments
are validated without prompting. Every accepted run starts the actual training
worker in a detached process, so the launcher returns immediately and the
terminal can be closed. It records the supplied and resolved arguments in a
timestamped JSON file and in that run's log under the repository-root `logs/`
directory. The worker PID is saved to `logs/train.pid`.

`rl` always trains the one unified policy against all four current specialists,
even when the first selector is only one side. A long-only or short-only `all`
run therefore requires compatible OOS streams for the opposite side before the
common policy can train.

Examples:

```sh
# Launch every specialist, then the shared policy, in the background
uv run python -m trade_ai.model.training.train_master both all

# Launch only short-side GRU in the background
uv run python -m trade_ai.model.training.train_master short gru

# Train only the common RL policy from existing compatible OOS streams
uv run python -m trade_ai.model.training.train_master both rl
```

## Base specialist lifecycle

For each selected direction, `train_master.py` backs up the prior checkpoint
and sidecar, unloads selected models, and then runs LightGBM followed by GRU.
Both trainers use the ticker universe in `model/stock_metadata.py`, normalize
cached or downloaded OHLCV through `training/util.py`, build direction-specific
labels, and attempt every ticker configured in `STOCK_METADATA`. Tickers with
missing or insufficient data are skipped and logged. One pooled calendar gives
every ticker the same whole-session boundaries: 55% initial fitting, the next
15% model selection, the next 20% in four expanding OOS forecast folds, and the
newest 10% as a final holdout. Up to 60 candle rows are purged before fit
boundaries so label look-ahead does not cross into the next period.

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

## Launching training

From the repository root, start a run with:

```sh
uv run python -m trade_ai.model.training.train_master both all
```

Replace `both all` with any valid mindset and component. The launcher prints the
worker PID, log path, and metadata path. Training continues after the launcher
returns or its terminal closes.

Follow the latest full run's log:

```sh
tail -f "$(ls -t logs/train_both_all_*.log | head -n 1)"
```

Check whether it is still running:

```sh
ps -p "$(cat logs/train.pid)" -o pid,stat,etime,cmd
```

If `ps` prints no process, inspect the log's final lines. A LightGBM baseline
rejection or an RL selection rejection is reported as a completed run with
skips; exceptions are reported as a failed run.
