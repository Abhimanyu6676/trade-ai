"""Launch detached training runs and coordinate base models with PPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from trade_ai.logger import logger
from trade_ai.model.modelTypes import ModelName

TrainingMindset = Literal["long", "short", "both"]
TrainingComponent = Literal["gru", "lgb", "rl", "all"]
MINDSET_DIRECTIONS: dict[TrainingMindset, tuple[Literal["long", "short"], ...]] = {
    "long": ("long",),
    "short": ("short",),
    "both": ("long", "short"),
}

TRAINING_LOCK = threading.Lock()
TRAINING_THREAD: threading.Thread | None = None
TRAINING_STATE = "idle"
TRAINING_ERROR: str | None = None
TRAINING_MODULE = "trade_ai.model.training.train_master"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
LOGS_DIRECTORY = REPOSITORY_ROOT / "logs"
PID_FILE = LOGS_DIRECTORY / "train.pid"


def _selected_base_models(
    mindset: TrainingMindset, component: TrainingComponent
) -> tuple[ModelName, ...]:
    """Resolve selected directional/family pairs into model-manager keys."""
    families = ("lgb", "gru") if component == "all" else (component,)
    return tuple(
        f"{direction}_{family}"
        for direction in MINDSET_DIRECTIONS[mindset]
        for family in families
        if family in ("lgb", "gru")
    )  # type: ignore[return-value]


def run_training(
    mindset: TrainingMindset = "both",
    component: TrainingComponent = "all",
) -> bool:
    """Train requested specialists and optionally the unified meta-agent.

    RL is a common component over all four frozen base models. Selecting it with
    a single mindset still validates and trains from the combined OOS streams.
    """
    from trade_ai.model.models import get_model_manager

    from .long.train_long_gru import train_long_gru
    from .long.train_long_lgb import train_long_lgb
    from .short.train_short_gru import train_short_gru
    from .short.train_short_lgb import train_short_lgb
    from .train_rl import train_meta_agent

    directions = MINDSET_DIRECTIONS[mindset]
    selected_models = _selected_base_models(mindset, component)
    train_rl = component in ("rl", "all")
    model_manager = get_model_manager()
    model_manager.backup_current_models(selected_models)
    if selected_models or train_rl:
        model_manager.backup_meta_agent()
        model_manager.unload_meta_agent()
    if selected_models:
        model_manager.unload_models(selected_models)

    training_succeeded = False
    try:
        all_checkpoints_saved = True
        if component in ("lgb", "all"):
            for direction in directions:
                logger.info("Starting %s LightGBM training.", direction)
                saved = train_long_lgb() if direction == "long" else train_short_lgb()
                if not saved:
                    all_checkpoints_saved = False
                    logger.warning(
                        "%s LightGBM did not beat its validation baseline; "
                        "its existing checkpoint remains unchanged.",
                        direction,
                    )

        if component in ("gru", "all"):
            for direction in directions:
                logger.info("Starting %s GRU training.", direction)
                if direction == "long":
                    train_long_gru(epochs=20)
                else:
                    train_short_gru(epochs=20)
                logger.info("%s GRU training completed.", direction)

        rl_trained = True
        if train_rl:
            logger.info("Starting unified PPO meta-agent training.")
            rl_trained = train_meta_agent()
            if not rl_trained:
                logger.warning(
                    "The RL policy did not beat the flat selection baseline; "
                    "its previous artifact was not replaced."
                )

        training_succeeded = True
        return all_checkpoints_saved and rl_trained
    except Exception:
        logger.exception("Training failed for %s/%s.", mindset, component)
        raise
    finally:
        if training_succeeded:
            if selected_models:
                model_manager.reload_models(selected_models)
            if selected_models or train_rl:
                model_manager.reload_meta_agent()
        else:
            logger.warning(
                "Skipping model reload after failed %s/%s training; affected models "
                "will remain unloaded until reloaded explicitly or the app restarts.",
                mindset,
                component,
            )


def run_full_training_pipeline() -> bool:
    """Backward-compatible entry point for the full long/short/RL pipeline."""
    return run_training("both", "all")


def _background_training_worker(
    mindset: TrainingMindset, component: TrainingComponent
) -> None:
    global TRAINING_STATE, TRAINING_ERROR
    try:
        completed = run_training(mindset, component)
        with TRAINING_LOCK:
            TRAINING_STATE = "completed" if completed else "completed_with_skips"
            TRAINING_ERROR = None
        logger.info(
            "Training completed for %s/%s (all checkpoints accepted=%s).",
            mindset,
            component,
            completed,
        )
    except Exception as error:
        with TRAINING_LOCK:
            TRAINING_STATE = "failed"
            TRAINING_ERROR = str(error)
        logger.error("Background training failed: %s", error)


def start_master_training(
    mindset: TrainingMindset = "both",
    component: TrainingComponent = "all",
) -> str:
    """Start selected training jobs asynchronously in a daemon thread."""
    global TRAINING_THREAD, TRAINING_STATE, TRAINING_ERROR
    with TRAINING_LOCK:
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            return "Training is already running in the background."
        TRAINING_STATE = "running"
        TRAINING_ERROR = None
        TRAINING_THREAD = threading.Thread(
            target=_background_training_worker,
            args=(mindset, component),
            name="trade-ai-training",
            daemon=True,
        )
        TRAINING_THREAD.start()
    return f"Training started for: {mindset}/{component}."


def get_training_status() -> dict[str, str | None]:
    """Return the current training state and optional error message."""
    with TRAINING_LOCK:
        return {"state": TRAINING_STATE, "error": TRAINING_ERROR}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch long/short specialist and PPO training as a detached process."
        )
    )
    parser.add_argument(
        "mindset",
        nargs="?",
        choices=("long", "short", "both"),
        default=None,
        help="Directional specialists to train (default: both).",
    )
    parser.add_argument(
        "component",
        nargs="?",
        choices=("gru", "lgb", "rl", "all"),
        default=None,
        help="Model component to train (default: all).",
    )
    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _prompt_selector(label: str, choices: Sequence[str], default: str) -> str:
    """Prompt for one supported selector, accepting either its name or number."""
    print(f"Select {label}:")
    for index, choice in enumerate(choices, start=1):
        marker = " (default)" if choice == default else ""
        print(f"  {index}. {choice}{marker}")

    while True:
        try:
            value = input(f"{label} [{default}]: ").strip().casefold()
        except EOFError:
            return default
        if not value:
            return default
        if value.isdecimal():
            index = int(value) - 1
            if 0 <= index < len(choices):
                return choices[index]
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}.")


def _confirm_start(mindset: str, component: str) -> bool:
    """Require an explicit yes before launching provided foreground selectors."""
    print(f"Training selection: mindset={mindset}, component={component}")
    while True:
        try:
            answer = input("Launch this training run in the background? [y/N]: ")
        except EOFError:
            return False
        normalized = answer.strip().casefold()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"", "n", "no"}:
            return False
        print("Please answer yes or no.")


def _write_json_atomically(path: Path, data: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _active_training_pid() -> int | None:
    """Return the PID from the pid file when it still runs this module."""
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except FileNotFoundError, ValueError, OSError:
        return None

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid

    command_line_path = Path("/proc") / str(pid) / "cmdline"
    try:
        command_line = command_line_path.read_bytes().replace(b"\0", b" ")
    except OSError:
        return pid
    if TRAINING_MODULE.encode() in command_line:
        return pid
    return None


def _launch_training(
    mindset: TrainingMindset,
    component: TrainingComponent,
    provided_args: list[str],
    interactive: bool,
) -> int:
    """Persist launch arguments and spawn the training worker detached."""
    active_pid = _active_training_pid()
    if active_pid is not None:
        print(
            f"Training is already running (PID {active_pid}, recorded in {PID_FILE}).",
            file=sys.stderr,
        )
        return 1

    LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"train_{mindset}_{component}_{run_id}"
    log_path = LOGS_DIRECTORY / f"{run_name}.log"
    metadata_path = LOGS_DIRECTORY / f"{run_name}.json"
    metadata: dict[str, object] = {
        "status": "launching",
        "requested_at": started_at.isoformat(timespec="seconds"),
        "provided_args": provided_args,
        "resolved_args": {"mindset": mindset, "component": component},
        "interactive_foreground": interactive,
        "log_file": str(log_path.resolve()),
    }
    launch_record = {
        "event": "training_launch_requested",
        "provided_args": provided_args,
        "resolved_args": metadata["resolved_args"],
        "metadata_file": str(metadata_path.resolve()),
    }
    _write_json_atomically(metadata_path, metadata)

    command = [
        sys.executable,
        "-u",
        "-m",
        TRAINING_MODULE,
        "--_worker",
        mindset,
        component,
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(launch_record, sort_keys=True) + "\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as error:
        metadata["status"] = "launch_failed"
        metadata["error"] = str(error)
        _write_json_atomically(metadata_path, metadata)
        print(f"Could not start training: {error}", file=sys.stderr)
        return 1

    metadata["status"] = "launched"
    metadata["pid"] = process.pid
    _write_json_atomically(metadata_path, metadata)
    temporary_pid_file = PID_FILE.with_suffix(".pid.tmp")
    temporary_pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    temporary_pid_file.replace(PID_FILE)
    logger.info(
        "Training launcher accepted args=%s; resolved mindset=%s component=%s; "
        "pid=%s; log=%s",
        provided_args,
        mindset,
        component,
        process.pid,
        log_path.resolve(),
    )
    print(f"Training PID: {process.pid}")
    print(f"Training log: {log_path.resolve()}")
    print(f"Run metadata: {metadata_path.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve selectors, confirm interactive requests, then always detach work."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args._worker:
        if args.mindset is None or args.component is None:
            parser.error("the internal worker requires mindset and component")
        return 0 if run_training(args.mindset, args.component) else 1

    provided_args = [
        value for value in (args.mindset, args.component) if value is not None
    ]
    interactive = sys.stdin.isatty()
    if interactive and not provided_args:
        mindset = _prompt_selector("mindset", ("long", "short", "both"), "both")
        component = _prompt_selector("component", ("gru", "lgb", "rl", "all"), "all")
    else:
        mindset = args.mindset or "both"
        component = args.component or "all"
        if interactive and not _confirm_start(mindset, component):
            print("Training launch cancelled.")
            return 0

    return _launch_training(
        mindset,
        component,
        provided_args=provided_args,
        interactive=interactive,
    )


if __name__ == "__main__":
    raise SystemExit(main())
