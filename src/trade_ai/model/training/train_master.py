"""Run selected model-training stages in the current process."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from trade_ai.environmentLoader import load_environment
from trade_ai.logger import logger
from trade_ai.model.modelTypes import BASE_MODELS
from trade_ai.model.training.train_master_helper import (
    TrainingCommand,
    _literal_string_values,
    log_duration,
    parse_training_args,
    parse_training_command,
    safe_stage,
)

MODEL_TRAINERS: dict[BASE_MODELS, Callable[[], bool]] = {}

# Enforce LightGBM before GRU within each directional group.
FAMILY_PRIORITY: dict[str, int] = {
    "lgb": 0,
    "gru": 1,
}


def register_trainer(
    model_name: BASE_MODELS,
) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
    """Register a specialist training function by its model name."""

    def decorator(func: Callable[[], bool]) -> Callable[[], bool]:
        if model_name in MODEL_TRAINERS:
            raise ValueError(f"Trainer already registered for {model_name}")
        MODEL_TRAINERS[model_name] = func
        return func

    return decorator


@register_trainer("long_lgb")
def _run_long_lgb() -> bool:
    from trade_ai.model.training.long.train_long_lgb import train_long_lgb

    if not train_long_lgb():
        logger.warning("Long LightGBM did not pass its validation gate.")
        return False
    return True


@register_trainer("long_gru")
def _run_long_gru() -> bool:
    from trade_ai.model.training.long.train_long_gru import train_long_gru

    train_long_gru(epochs=20)
    return True


@register_trainer("short_lgb")
def _run_short_lgb() -> bool:
    from trade_ai.model.training.short.train_short_lgb import train_short_lgb

    if not train_short_lgb():
        logger.warning("Short LightGBM did not pass its validation gate.")
        return False
    return True


@register_trainer("short_gru")
def _run_short_gru() -> bool:
    from trade_ai.model.training.short.train_short_gru import train_short_gru

    train_short_gru(epochs=20)
    return True


def _validate_registered_trainers() -> None:
    """Fail fast at import time if any BASE_MODELS member lacks a trainer."""
    expected_models = set(_literal_string_values(BASE_MODELS))
    missing_models = sorted(expected_models - MODEL_TRAINERS.keys())
    if missing_models:
        raise RuntimeError(
            f"Missing registered trainers for BASE_MODELS: {', '.join(missing_models)}"
        )


_validate_registered_trainers()


def _order_group_models(models: Sequence[BASE_MODELS]) -> tuple[BASE_MODELS, ...]:
    """Order a directional group so LightGBM runs before GRU."""
    return tuple(
        sorted(
            models,
            key=lambda name: FAMILY_PRIORITY.get(name.rsplit("_", maxsplit=1)[-1], 99),
        )
    )


class TrainingRunner:
    """Run selected training stages in the current process."""

    @safe_stage("Training run failed.")
    def run(self, command: TrainingCommand) -> bool:
        """Download requested data and execute the selected training stages."""
        has_base_models = bool(command.long_models or command.short_models)
        if not has_base_models and not command.train_meta_agent:
            # A download-only command reports failure if any ticker could not be fetched.
            return self._download_data(command.download_data)

        # Download before training imports; partial ticker failures do not stop the run.
        self._download_data(command.download_data)

        # Run long group first (lgb -> gru), then short group (lgb -> gru).
        long_succeeded = self._train_direction_group("Long", command.long_models)
        short_succeeded = self._train_direction_group("Short", command.short_models)
        base_training_succeeded = long_succeeded and short_succeeded

        meta_training_succeeded = True
        if command.train_meta_agent:
            if has_base_models and not base_training_succeeded:
                logger.warning(
                    "Skipping PPO because at least one selected base model failed."
                )
                meta_training_succeeded = False
            else:
                meta_training_succeeded = self._train_meta_agent()

        return base_training_succeeded and meta_training_succeeded

    @staticmethod
    @log_duration("Data download")
    def _download_data(requested: bool) -> bool:
        """Download missing ticker history when the command requests it."""
        if not requested:
            return True

        # Keep the downloader import local so a download-only command avoids ML imports.
        from trade_ai.model.training.util import download_missing_stock_data

        failed_tickers = download_missing_stock_data()
        if not failed_tickers:
            return True

        # Training can still use all ticker histories that were downloaded successfully.
        logger.warning(
            "Continuing with available ticker data; downloads failed for: %s",
            ", ".join(failed_tickers),
        )
        return False

    def _train_direction_group(
        self, direction_label: str, models: Sequence[BASE_MODELS]
    ) -> bool:
        """Run a directional group (Long or Short) ordered as LightGBM then GRU."""
        ordered_models = _order_group_models(models)
        if not ordered_models:
            return True

        logger.info(
            "Starting %s group batch: %s",
            direction_label,
            ", ".join(ordered_models),
        )

        return self._train_base_models(ordered_models)

    def _train_base_models(self, model_names: tuple[BASE_MODELS, ...]) -> bool:
        """Run each selected specialist and retain failures without skipping peers."""
        succeeded = True

        # Keep every requested model attempt independent so one failure does not skip others.
        for model_name in model_names:
            model_succeeded = self._train_base_model(model_name)
            succeeded = model_succeeded and succeeded

        return succeeded

    @staticmethod
    @safe_stage("Training failed for %s.", include_first_arg=True)
    @log_duration("Specialist training")
    def _train_base_model(model_name: BASE_MODELS) -> bool:
        """Train one direction/family specialist and report whether it saved."""
        trainer = MODEL_TRAINERS.get(model_name)
        if trainer is None:
            logger.error("No trainer registered for model: %s", model_name)
            return False

        succeeded = trainer()
        if succeeded:
            logger.info("Completed training for %s.", model_name)
        return succeeded

    @safe_stage("PPO training failed.")
    @log_duration("PPO meta-agent training")
    def _train_meta_agent(self) -> bool:
        """Train the shared PPO policy using current specialist predictions."""
        # Import the PPO stack only when its training stage is selected.
        from trade_ai.model.training.train_rl import train_meta_agent

        trained = train_meta_agent()
        if not trained:
            logger.warning("PPO did not pass its policy-selection baseline.")
        return trained


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the training selection and run it in the current process."""
    # Load the repository environment for OpenAlgo and other configured services.
    load_environment()

    # Use the provided arguments when called programmatically, or read the CLI arguments.
    args = list(sys.argv[1:] if argv is None else argv)
    selected_options = parse_training_args(args)
    command = parse_training_command(selected_options or [])
    if command is None:
        print("Select at least one training option.", file=sys.stderr)
        return 1

    # Return a process exit status so tmux records whether the training run succeeded.
    return 0 if TrainingRunner().run(command) else 1


if __name__ == "__main__":
    raise SystemExit(main())
