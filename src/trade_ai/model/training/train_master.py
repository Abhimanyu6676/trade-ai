"""Launch detached training runs and coordinate base models with PPO."""

from __future__ import annotations

import os  # Used to inherit the environment and determine the available CPU set.
import subprocess  # Used to launch the selected command as a detached child process.
import sys
import time
from collections.abc import Sequence
from datetime import datetime  # Used to make a unique timestamped worker log folder.
from pathlib import Path

from trade_ai.model.features import (
    FEATURE_SCHEMA_VERSION,  # Version the feature schema in each run folder.
)
from trade_ai.model.training.labeling import (
    LABEL_SCHEMA_VERSION,  # Version the label schema in each run folder.
)
from trade_ai.model.training.train_master_helper import (
    TRAINING_COMMAND,
    TrainingCmds,
    _available_cpu_ids,
    parse_training_args,
    parseTrainingCommand,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
LOGS_DIRECTORY = (
    REPOSITORY_ROOT / "logs"
)  # Store detached worker logs under the repository logs folder.


def backgroundTrainingWorker(trainingCmd: TRAINING_COMMAND) -> int:
    """Placeholder entry point for a selected command in the detached process."""

    print(f"background worker args are {trainingCmd}")

    # Keep this temporary body explicit until the model-stage dispatcher is added.
    counter = 0  # Start the placeholder work counter at zero.

    while counter < 15:  # Run the placeholder body a fixed number of times.
        print(f"Counter is : {counter}")  # Emit the current placeholder progress value.
        time.sleep(5)
        counter += 1  # Advance the placeholder counter before its next iteration.

    return 0  # Report successful completion of the current placeholder handler.


def startInBackground(trainingArgs: list[TrainingCmds]) -> None:
    """Start this selection again as a detached worker invocation."""

    # Reuse the current interpreter and module command so the child runs in the same environment.
    workerCommand = [
        sys.executable,  # Reuse the current Python environment.
        "-u",  # Write worker output immediately so the log stays current.
        "-m",  # Run the module through Python's module launcher.
        "trade_ai.model.training.train_master",
    ]

    # Add each selected option back as a CLI flag so the worker parses the same selection.
    workerCommand.extend(f"--{trainingArg}" for trainingArg in trainingArgs)
    workerCommand.append("--worker")  # add background-worker Arg in main().

    # Give numerical libraries the full CPU count available to this process.
    cpu_count = len(_available_cpu_ids())
    # Preserve the launcher's configured environment.
    worker_environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):  # Cover the common native thread pools used by model training.
        worker_environment[variable] = str(
            cpu_count
        )  # Override any inherited thread limit.
    worker_environment["TRADE_AI_WORKER_CPU_COUNT"] = str(
        cpu_count
    )  # Let LightGBM use the same full count.
    worker_environment["TRADE_AI_BASE_TRAINING_CPUS"] = str(
        cpu_count
    )  # Give base training every available CPU.
    worker_environment["TRADE_AI_PPO_CPUS"] = str(
        cpu_count
    )  # Give PPO training every available CPU.

    # Create a versioned run directory so each worker's complete output is retained.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = LOGS_DIRECTORY / (
        f"FS_{FEATURE_SCHEMA_VERSION}_LS_{LABEL_SCHEMA_VERSION}_{timestamp}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    worker_log_path = run_directory / "worker.log"

    # Redirect both output streams to the log while detaching the worker from the terminal.
    with worker_log_path.open("w", encoding="utf-8", buffering=1) as worker_log:
        process = subprocess.Popen(  # Create the new worker process.
            workerCommand,  # Run the same module with the selected options and worker flag.
            cwd=REPOSITORY_ROOT,  # Keep imports and relative paths rooted in the repository.
            stdin=subprocess.DEVNULL,  # Prevent the worker from waiting for terminal input.
            stdout=worker_log,  # Save all worker standard output in its run directory.
            stderr=subprocess.STDOUT,  # Save worker errors in the same log file.
            env=worker_environment,  # Pass the full-core thread settings into the worker.
            start_new_session=True,  # Detach the child from the launcher's terminal session.
        )  # Finish starting the detached worker.

    worker_pid_path = run_directory / "worker.pid"
    worker_pid_path.write_text(f"{process.pid}\n", encoding="utf-8")

    print(f"Training worker started with PID: {process.pid}")
    print(f"Training worker log: {worker_log_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the selected command and route it to the parent or worker flow."""

    # 1. Fallback to sys.argv[1:] if no explicit argv sequence is provided
    if argv is None:
        argv = sys.argv[1:]

    args_list = list(argv)
    print(f"Received arguments in main: {args_list}")

    # Route only the explicit worker flag to the background command handler.
    if "--worker" in args_list:
        print(f"This worker thread command, and args are {args_list}")

        trainingArgs = parse_training_args(args_list)
        if not trainingArgs:
            print("Training Args is invalid while starting worker, skipping worker")
            return 1
        else:
            trainingCmd = parseTrainingCommand(trainingArgs)
            if trainingCmd is not None:
                # Return the worker handler's status to the child process exit code.
                return backgroundTrainingWorker(trainingCmd)
            else:
                print("Training Cmd is invalid while starting worker, skipping worker")
                return 1

    else:
        print("Now parsing args and sending with working after confirmation")

        trainingArgs = parse_training_args(args_list)
        print(f"trainingArgs: {trainingArgs}")

        if not trainingArgs:
            print("Training command is invalid, Please try again")
            return 1
        else:
            startInBackground(trainingArgs)

    # Report that the foreground process has handed work to its detached child.
    print("Training Foreground Script ended, after starting background worker thread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
