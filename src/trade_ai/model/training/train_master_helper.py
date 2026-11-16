"""Training command types, argument parsing, and interactive selection."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, ParamSpec, cast, get_args

from InquirerPy.base.control import Choice
from InquirerPy.prompts.checkbox import CheckboxPrompt

from trade_ai.logger import logger
from trade_ai.model.modelTypes import (
    LONG_BASE_MODELS,
    MODEL_NAMES,
    PREDICTION_MODELS,
    SHORT_BASE_MODELS,
)

TrainingOption = PREDICTION_MODELS | Literal["all", "download"]
P = ParamSpec("P")


@dataclass(frozen=True)
class TrainingCommand:
    """Describe which model stages and optional data download to run."""

    long_models: tuple[LONG_BASE_MODELS, ...] = ()
    short_models: tuple[SHORT_BASE_MODELS, ...] = ()
    train_meta_agent: bool = False
    download_data: bool = False


def safe_stage(
    error_message: str, *, include_first_arg: bool = False
) -> Callable[[Callable[P, bool]], Callable[P, bool]]:
    """Catch exceptions in a training stage, log them, and return False."""

    def decorator(func: Callable[P, bool]) -> Callable[P, bool]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> bool:
            # Preserve the stage result unless an exception needs conversion to failure.
            try:
                return func(*args, **kwargs)
            except Exception:
                # Add the first argument when the stage message identifies a model.
                if include_first_arg and args:
                    logger.exception(error_message, args[0])
                else:
                    logger.exception(error_message)
                return False

        return wrapper

    return decorator


def log_duration(stage_label: str) -> Callable[[Callable[P, bool]], Callable[P, bool]]:
    """Log how long a training or download stage takes to complete."""

    def decorator(func: Callable[P, bool]) -> Callable[P, bool]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> bool:
            # Measure only the wrapped stage so the log reports its own runtime.
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(
                "%s finished in %.2fs (success=%s).", stage_label, elapsed, result
            )
            return result

        return wrapper

    return decorator


def _literal_string_values(type_alias: object) -> tuple[str, ...]:
    """Flatten nested Literal and union aliases into their string values."""
    values: list[str] = []

    # Union aliases can contain nested Literal aliases, so expand them recursively.
    for value in get_args(type_alias):
        if isinstance(value, str):
            values.append(value)
        else:
            values.extend(_literal_string_values(value))

    # Keep the declared option order while removing duplicate values.
    return tuple(dict.fromkeys(values))


def _split_base_models(
    selected_options: Sequence[TrainingOption],
) -> tuple[
    tuple[LONG_BASE_MODELS, ...],
    tuple[SHORT_BASE_MODELS, ...],
]:
    """Separate selected specialist names into direction-specific tuples."""
    # Filter against each direction's Literal so command fields retain their types.
    long_models = cast(
        tuple[LONG_BASE_MODELS, ...],
        tuple(
            option
            for option in selected_options
            if option in get_args(LONG_BASE_MODELS)
        ),
    )
    short_models = cast(
        tuple[SHORT_BASE_MODELS, ...],
        tuple(
            option
            for option in selected_options
            if option in get_args(SHORT_BASE_MODELS)
        ),
    )
    return long_models, short_models


def _select_options(
    selected_options: list[TrainingOption],
) -> list[TrainingOption]:
    """Show the terminal selector and return the user's checked options."""
    options = cast(tuple[TrainingOption, ...], _literal_string_values(TrainingOption))

    # Preselect options supplied on the command line for easy review and adjustment.
    selection = CheckboxPrompt(
        message="Choose model-training options",
        choices=[
            Choice(option, enabled=True)
            if option in selected_options
            else Choice(option)
            for option in options
        ],
        instruction="(Use space to select, arrows to move, enter to confirm)",
    ).execute()
    return cast(list[TrainingOption], selection)


def parse_training_args(argv: list[str]) -> list[TrainingOption] | None:
    """Parse training flags and confirm the selection interactively when possible."""
    parser = argparse.ArgumentParser(description="Select model-training stages.")

    # Derive flags from the model type aliases to keep the CLI and contracts aligned.
    options = cast(tuple[TrainingOption, ...], _literal_string_values(TrainingOption))
    for option in options:
        parser.add_argument(f"--{option}", action="store_true", dest=option)

    # Resolve the command line before deciding whether an interactive prompt is useful.
    parsed = parser.parse_args(argv)
    selected_options: list[TrainingOption] = [
        option for option in options if getattr(parsed, option)
    ]

    # Non-interactive runs cannot show a selector, so use any explicit flags as supplied.
    if not sys.stdin.isatty():
        return selected_options

    # Always show the selector in a TTY; explicit flags are preselected for confirmation.
    return _select_options(selected_options) or None


def parse_training_command(
    selected_options: list[TrainingOption],
) -> TrainingCommand | None:
    """Convert selected CLI options into the stages executed by the runner."""
    if not selected_options:
        return None

    # The all option intentionally includes the one-time history download.
    if "all" in selected_options:
        long_models, short_models = _split_base_models(MODEL_NAMES)
        return TrainingCommand(
            long_models=long_models,
            short_models=short_models,
            train_meta_agent=True,
            download_data=True,
        )

    # Keep long and short specialists in their matching directional command fields.
    long_models, short_models = _split_base_models(selected_options)
    return TrainingCommand(
        long_models=long_models,
        short_models=short_models,
        train_meta_agent="meta_ensemble" in selected_options,
        download_data="download" in selected_options,
    )
