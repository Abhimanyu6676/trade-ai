import argparse
import os  # Used to inherit the environment and determine the available CPU set.
import sys  # Used to avoid opening an interactive selector without a foreground terminal.
from dataclasses import dataclass
from typing import Literal, cast, get_args

from InquirerPy.base.control import Choice
from InquirerPy.prompts.checkbox import CheckboxPrompt

from trade_ai.model.modelTypes import (
    LONG_BASE_MODELS,
    META_MODELS,
    PREDICTION_MODELS,
    SHORT_BASE_MODELS,
)

TrainingModels = PREDICTION_MODELS | Literal["all"]

DownloadOnlyInput = "download"
DownloadOnlyCmd = Literal["download"]

TrainingCmds = TrainingModels | DownloadOnlyCmd


def _available_cpu_ids() -> list[int]:
    """Return the logical CPU IDs currently available to this process."""

    # Prefer the active affinity mask so containers and schedulers are respected.
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_ids = sorted(os.sched_getaffinity(0))
            if cpu_ids:
                return cpu_ids
        except OSError:
            pass  # Use the portable CPU-count fallback if affinity cannot be read.

    # Use at least one CPU when the platform does not expose process affinity.
    return list(range(max(1, os.cpu_count() or 1)))


def _literal_string_values(type_alias: object) -> tuple[str, ...]:
    """Flatten nested Literal and union aliases into their string values."""
    values: list[str] = []

    # Union aliases may contain other Literal aliases, so expand them recursively.
    for value in get_args(type_alias):
        if isinstance(value, str):
            values.append(value)
        else:
            values.extend(_literal_string_values(value))

    # Preserve alias order while avoiding duplicate choices from nested unions.
    return tuple(dict.fromkeys(values))


@dataclass
class MODELS_TO_TRAIN:
    all: bool | None = None
    long: list[LONG_BASE_MODELS] | None = None
    short: list[SHORT_BASE_MODELS] | None = None
    meta: list[META_MODELS] | None = None


ALL_MODELS_TO_TRAIN: MODELS_TO_TRAIN = MODELS_TO_TRAIN(
    long=["long_lgb", "long_gru"],
    short=["short_lgb", "short_gru"],
    meta=["meta_ensemble"],
)


@dataclass
class TRAINING_COMMAND:
    downloadData: bool | None = None
    trainingSet: MODELS_TO_TRAIN | None = None


DOWNLOAD_N_TRAIN: TRAINING_COMMAND = TRAINING_COMMAND(
    downloadData=True, trainingSet=ALL_MODELS_TO_TRAIN
)


def _multi_options_selector(
    label: str,
    choices: list[TrainingCmds],
    selected: list[TrainingCmds],
) -> list[TrainingCmds]:
    """Prompt for one supported selector, accepting either its name or number."""
    selection = CheckboxPrompt(
        message=label,
        choices=[
            Choice(choice, enabled=True) if choice in selected else Choice(choice)
            for choice in choices
        ],
        instruction="(Use space to select, arrows to move, enter to confirm)",
    ).execute()

    print("\nYou selected:", selection)
    return selection


def parseTrainingCommand(
    selectedCommands: list[TrainingCmds],
) -> TRAINING_COMMAND | None:

    if selectedCommands:
        if "all" in selectedCommands:
            return DOWNLOAD_N_TRAIN
        else:
            # Group selected models according to the Literal aliases owned by modelTypes.
            longModels = cast(
                list[LONG_BASE_MODELS],
                [
                    model
                    for model in selectedCommands
                    if model in get_args(LONG_BASE_MODELS)
                ],
            )
            shortModels = cast(
                list[SHORT_BASE_MODELS],
                [
                    model
                    for model in selectedCommands
                    if model in get_args(SHORT_BASE_MODELS)
                ],
            )
            metaModels = cast(
                list[META_MODELS],
                [model for model in selectedCommands if model in get_args(META_MODELS)],
            )

            modelsToTrain = TRAINING_COMMAND(
                downloadData=DownloadOnlyInput in selectedCommands,
                trainingSet=MODELS_TO_TRAIN(
                    long=longModels or None,
                    short=shortModels or None,
                    meta=metaModels or None,
                ),
            )
            return modelsToTrain

    return None


def parse_training_args(argv: list[str]) -> list[TrainingCmds] | None:
    """Parse one supported model-selection flag from command-line arguments."""
    parser = argparse.ArgumentParser(description="Select one training model.")

    # Use the Literal as the single source of truth for available flags.
    cmd_options = cast(tuple[TrainingCmds, ...], _literal_string_values(TrainingCmds))

    # Parse each selection independently so we can give a clear error if several are set.
    for model in cmd_options:
        parser.add_argument(f"--{model}", action="store_true", dest=model)

    # Consume the internal worker flag so it is not rejected as an unknown argument.
    parser.add_argument(  # Register the flag used by detached child processes.
        "--worker",  # Identify the child process as a background worker.
        action="store_true",  # Store whether the child flag was supplied.
        help=argparse.SUPPRESS,  # Hide this internal flag from normal CLI help.
    )  # Finish registering the internal worker flag.

    parsed = parser.parse_args(argv)
    selectedOptions: list[TrainingCmds] = [
        model for model in cmd_options if getattr(parsed, model)
    ]

    # Return parsed flags directly for detached workers and non-interactive invocations.
    if parsed.worker or not sys.stdin.isatty():
        return selectedOptions

    if not selectedOptions:
        print("You haven't provided model to train Kindly select from the following:")

    choiceConfirmed: list[TrainingCmds] = _multi_options_selector(
        "Choose and Confirm Your Models selection for training",
        list(cmd_options),
        selectedOptions,
    )

    if "all" in choiceConfirmed:
        choiceConfirmed = list(set(list(cmd_options) + choiceConfirmed) - {"all"})

    if choiceConfirmed:
        # print(f"selected model options are {choiceConfirmed}")
        # return parseTrainingCommand(choiceConfirmed)
        return choiceConfirmed

    return None
