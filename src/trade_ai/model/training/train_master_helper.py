"""Training command types, argument parsing, and interactive selection."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Literal, cast, get_args

from InquirerPy.base.control import Choice
from InquirerPy.prompts.checkbox import CheckboxPrompt

from trade_ai.model.modelTypes import BASE_MODELS, MODEL_NAMES, PREDICTION_MODELS

TrainingOption = PREDICTION_MODELS | Literal["all", "download"]


@dataclass(frozen=True)
class TrainingCommand:
    """Describe which model stages and optional data download to run."""

    base_models: tuple[BASE_MODELS, ...] = ()
    train_meta_agent: bool = False
    download_data: bool = False


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
        return TrainingCommand(
            base_models=MODEL_NAMES,
            train_meta_agent=True,
            download_data=True,
        )

    # Build a typed command from the selected specialist, meta-agent, and download flags.
    base_models = cast(
        tuple[BASE_MODELS, ...],
        tuple(option for option in selected_options if option in MODEL_NAMES),
    )
    return TrainingCommand(
        base_models=base_models,
        train_meta_agent="meta_ensemble" in selected_options,
        download_data="download" in selected_options,
    )
