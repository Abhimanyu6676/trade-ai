"""Launch detached training runs and coordinate base models with PPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from trade_ai.logger import logger
from trade_ai.model.modelTypes import BASE_MODELS
from trade_ai.model.training.util import atomic_write_json

TrainingMindset = Literal["long", "short", "both"]
TrainingComponent = Literal["gru", "lgb", "rl", "all"]
MINDSET_DIRECTIONS: dict[TrainingMindset, tuple[Literal["long", "short"], ...]] = {
    "long": ("long",),
    "short": ("short",),
    "both": ("long", "short"),
}

TRAINING_MODULE = "trade_ai.model.training.train_master"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
LOGS_DIRECTORY = REPOSITORY_ROOT / "logs"
PID_FILE = LOGS_DIRECTORY / "train.pid"
BASE_CPU_BUDGET_ENV = "TRADE_AI_BASE_TRAINING_CPUS"
PPO_CPU_BUDGET_ENV = "TRADE_AI_PPO_CPUS"
WORKER_CPU_IDS_ARG = "--_cpu-ids"


def _selected_base_models(
    mindset: TrainingMindset, component: TrainingComponent
) -> tuple[BASE_MODELS, ...]:
    """Resolve selected directional/family pairs into model-manager keys."""
    families = ("lgb", "gru") if component == "all" else (component,)
    return tuple(
        f"{direction}_{family}"
        for direction in MINDSET_DIRECTIONS[mindset]
        for family in families
        if family in ("lgb", "gru")
    )  # type: ignore[return-value]


def _available_cpu_ids() -> list[int]:
    """Return logical CPU IDs available to this process, respecting its cpuset."""
    # Prefer the process affinity mask because containers and schedulers may
    # expose fewer CPUs than the host reports through os.cpu_count().
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_ids = sorted(os.sched_getaffinity(0))
            if cpu_ids:
                return cpu_ids
        except OSError:
            logger.exception("Could not read the current CPU affinity mask.")

    # Fall back to the portable logical CPU count when affinity is unavailable.
    return list(range(max(1, os.cpu_count() or 1)))


def _resolve_cpu_budget(env_name: str, default: int, available_count: int) -> int:
    """Resolve a positive environment CPU budget and clamp it to available CPUs."""
    raw_value = os.environ.get(env_name)
    requested = default

    # Parse optional user configuration while preserving a safe machine-based default.
    if raw_value is not None:
        try:
            requested = int(raw_value)
        except ValueError:
            logger.warning(
                "%s=%r is not an integer; using the default of %d CPUs.",
                env_name,
                raw_value,
                default,
            )
            requested = default

    # Reject zero and negative values rather than creating an empty affinity set.
    if requested < 1:
        logger.warning(
            "%s=%d must be positive; using the default of %d CPUs.",
            env_name,
            requested,
            default,
        )
        requested = default

    # A configured budget may exceed this machine or container's available CPUs.
    if requested > available_count:
        logger.warning(
            "%s requests %d CPUs but only %d are available; using %d.",
            env_name,
            requested,
            available_count,
            available_count,
        )
        return available_count
    return requested


def _configure_worker_cpu(cpu_ids: list[int]) -> int:
    """Pin a model worker and limit native thread pools to its CPU allocation."""
    if not cpu_ids:
        raise ValueError("CPU worker allocation cannot be empty")

    # Apply Linux affinity before importing numerical and ML libraries.
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))
        logger.info("Worker PID %d pinned to logical CPUs %s.", os.getpid(), cpu_ids)
    else:
        logger.warning(
            "CPU affinity is unavailable on this platform; limiting native thread "
            "pools to %d CPUs without pinning.",
            len(cpu_ids),
        )

    # Configure OpenMP and common numerical backends before their imports.
    cpu_count = len(cpu_ids)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(cpu_count)
    os.environ["TRADE_AI_WORKER_CPU_COUNT"] = str(cpu_count)
    return cpu_count


def _configure_torch_threads(cpu_count: int) -> None:
    """Set PyTorch CPU intra-op parallelism to the worker's CPU allocation."""
    # Import Torch only after worker affinity and native thread environment are set.
    import torch

    torch.set_num_threads(cpu_count)
    logger.info("Configured PyTorch intra-op threads to %d.", cpu_count)


def _run_direction_worker(
    direction: Literal["long", "short"], component: TrainingComponent
) -> bool:
    """Train the selected models for one direction and log each stage failure."""
    succeeded = True

    # Train LightGBM first so its validation gate does not prevent the GRU attempt.
    if component in ("lgb", "all"):
        try:
            if direction == "long":
                from .long.train_long_lgb import train_long_lgb

                saved = train_long_lgb()
            else:
                from .short.train_short_lgb import train_short_lgb

                saved = train_short_lgb()
            if not saved:
                succeeded = False
                logger.error(
                    "%s LightGBM training did not save a checkpoint; "
                    "its validation baseline gate was not met.",
                    direction,
                )
        except Exception:
            succeeded = False
            logger.exception("%s LightGBM training failed.", direction)

    # Train GRU even if this direction's LightGBM stage failed.
    if component in ("gru", "all"):
        try:
            cpu_count = len(_available_cpu_ids())
            _configure_torch_threads(cpu_count)
            if direction == "long":
                from .long.train_long_gru import train_long_gru

                train_long_gru(epochs=20)
            else:
                from .short.train_short_gru import train_short_gru

                train_short_gru(epochs=20)
            logger.info("%s GRU training completed.", direction)
        except Exception:
            succeeded = False
            logger.exception("%s GRU training failed.", direction)

    return succeeded


def _run_ppo_worker() -> bool:
    """Train PPO in its own process after successful base-model workers."""
    try:
        # Apply the assigned CPU count before Stable-Baselines3 imports Torch.
        cpu_count = len(_available_cpu_ids())
        _configure_torch_threads(cpu_count)
        from .train_rl import train_meta_agent

        trained = train_meta_agent()
        if not trained:
            logger.error(
                "PPO training did not save a checkpoint because its selection "
                "score did not beat the flat baseline."
            )
        return trained
    except Exception:
        logger.exception("PPO training failed.")
        return False


def _spawn_direction_workers(
    directions: tuple[Literal["long", "short"], ...],
    component: TrainingComponent,
    cpu_ids: list[int],
    cpu_budget: int,
) -> bool:
    """Run direction workers concurrently on disjoint CPU sets when possible."""
    worker_cpu_ids = cpu_ids[:cpu_budget]
    affinity_supported = hasattr(os, "sched_setaffinity") and hasattr(
        os, "sched_getaffinity"
    )
    run_sequentially = len(directions) > 1 and (
        cpu_budget < 2 or not affinity_supported
    )

    # Split the effective base budget into non-overlapping, near-equal masks.
    allocations: dict[Literal["long", "short"], list[int]] = {}
    if len(directions) == 1:
        allocations[directions[0]] = worker_cpu_ids
    elif not run_sequentially:
        split_at = cpu_budget // 2
        allocations[directions[0]] = worker_cpu_ids[:split_at]
        allocations[directions[1]] = worker_cpu_ids[split_at:]
        logger.info(
            "Base CPU budget=%d split between directions: %s=%s, %s=%s.",
            cpu_budget,
            directions[0],
            allocations[directions[0]],
            directions[1],
            allocations[directions[1]],
        )
    else:
        logger.warning(
            "Cannot run both direction workers on disjoint CPU sets with a "
            "%d-CPU budget or without affinity support; running them sequentially.",
            cpu_budget,
        )
        allocations = {direction: worker_cpu_ids for direction in directions}

    succeeded = True
    processes: list[tuple[Literal["long", "short"], subprocess.Popen]] = []

    # Launch each direction as an isolated Python process with its CPU mask.
    for direction in directions:
        assigned_ids = allocations[direction]
        command = [
            sys.executable,
            "-u",
            "-m",
            TRAINING_MODULE,
            "--_direction-worker",
            direction,
            component,
            WORKER_CPU_IDS_ARG,
            ",".join(str(cpu_id) for cpu_id in assigned_ids),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError:
            succeeded = False
            logger.exception(
                "Could not start the %s/%s base-model worker.", direction, component
            )
            continue

        logger.info(
            "Started %s base-model worker PID %d on CPUs %s.",
            direction,
            process.pid,
            assigned_ids,
        )
        if run_sequentially:
            # Wait before launching the next direction when masks cannot be disjoint.
            return_code = process.wait()
            if return_code != 0:
                succeeded = False
                logger.error(
                    "%s base-model worker PID %d failed with exit code %d.",
                    direction,
                    process.pid,
                    return_code,
                )
        else:
            processes.append((direction, process))

    # Join all launched direction workers and retain failures without cancelling peers.
    for direction, process in processes:
        return_code = process.wait()
        if return_code != 0:
            succeeded = False
            logger.error(
                "%s base-model worker PID %d failed with exit code %d.",
                direction,
                process.pid,
                return_code,
            )
        else:
            logger.info(
                "%s base-model worker PID %d completed successfully.",
                direction,
                process.pid,
            )
    return succeeded


def _spawn_ppo_worker(cpu_ids: list[int], cpu_budget: int) -> bool:
    """Run PPO as a child process pinned to its configured CPU allocation."""
    assigned_ids = cpu_ids[:cpu_budget]
    command = [
        sys.executable,
        "-u",
        "-m",
        TRAINING_MODULE,
        "--_ppo-worker",
        WORKER_CPU_IDS_ARG,
        ",".join(str(cpu_id) for cpu_id in assigned_ids),
    ]
    try:
        # The child inherits this run's log descriptors and writes its own stage details.
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError:
        logger.exception("Could not start the PPO worker.")
        return False

    logger.info("Started PPO worker PID %d on CPUs %s.", process.pid, assigned_ids)
    return_code = process.wait()
    if return_code != 0:
        logger.error(
            "PPO worker PID %d failed with exit code %d.", process.pid, return_code
        )
        return False
    logger.info("PPO worker PID %d completed successfully.", process.pid)
    return True


def run_training(
    mindset: TrainingMindset = "both",
    component: TrainingComponent = "all",
) -> bool:
    """Train requested specialists and optionally the unified meta-agent.

    RL is a common component over all four frozen base models. Selecting it with
    a single mindset still validates and trains from the combined OOS streams.
    """
    from trade_ai.model.models import get_model_manager

    from .util import download_missing_stock_data

    directions = MINDSET_DIRECTIONS[mindset]
    selected_models = _selected_base_models(mindset, component)
    train_rl = component in ("rl", "all")
    failed_downloads = download_missing_stock_data()
    if failed_downloads:
        logger.warning(
            "Continuing training without unavailable ticker data: %s",
            ", ".join(failed_downloads),
        )
    model_manager = get_model_manager()
    model_manager.backup_current_models(selected_models)
    if selected_models or train_rl:
        model_manager.backup_meta_agent()
        model_manager.unload_meta_agent()
    if selected_models:
        model_manager.unload_models(selected_models)

    training_succeeded = False
    try:
        # Resolve CPU masks once for this run, respecting container affinity.
        cpu_ids = _available_cpu_ids()
        base_cpu_budget = _resolve_cpu_budget(
            BASE_CPU_BUDGET_ENV, len(cpu_ids), len(cpu_ids)
        )
        logger.info(
            "Available logical CPUs=%s; base training CPU budget=%d.",
            cpu_ids,
            base_cpu_budget,
        )

        # Direction workers own their separate checkpoints and continue independently.
        base_training_succeeded = True
        if selected_models:
            base_training_succeeded = _spawn_direction_workers(
                directions, component, cpu_ids, base_cpu_budget
            )

        rl_trained = True
        if train_rl:
            if not base_training_succeeded:
                rl_trained = False
                logger.warning(
                    "Skipping PPO because at least one selected base-model "
                    "training process failed."
                )
            else:
                ppo_default = min(3, len(cpu_ids))
                ppo_cpu_budget = _resolve_cpu_budget(
                    PPO_CPU_BUDGET_ENV, ppo_default, len(cpu_ids)
                )
                logger.info(
                    "Starting unified PPO meta-agent training with %d CPUs.",
                    ppo_cpu_budget,
                )
                rl_trained = _spawn_ppo_worker(cpu_ids, ppo_cpu_budget)

        # Even on a stage failure, the parent has joined all workers and can reload
        # checkpoints that completed successfully before returning a failed status.
        training_succeeded = True
        return base_training_succeeded and rl_trained
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Launch training or download missing historical ticker data.")
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
        "--download-only",
        action="store_true",
        help="Download and cache missing ticker history, then exit without training.",
    )
    parser.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_direction-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--_ppo-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(WORKER_CPU_IDS_ARG, help=argparse.SUPPRESS)
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
    atomic_write_json(metadata_path, metadata)

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
        atomic_write_json(metadata_path, metadata)
        print(f"Could not start training: {error}", file=sys.stderr)
        return 1

    metadata["status"] = "launched"
    metadata["pid"] = process.pid
    atomic_write_json(metadata_path, metadata)
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
    # Load optional CPU budgets from .env before resolving or spawning workers.
    from trade_ai.environmentLoader import load_environment

    load_environment()

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Internal child modes configure affinity before importing model libraries.
    if args._direction_worker or args._ppo_worker:
        if (
            args._worker
            or args.download_only
            or args._direction_worker == args._ppo_worker
        ):
            parser.error("internal training worker options cannot be combined")
        if not args._cpu_ids:
            parser.error("internal training workers require --_cpu-ids")
        try:
            cpu_ids = sorted({int(cpu_id) for cpu_id in args._cpu_ids.split(",")})
            if not cpu_ids or any(cpu_id < 0 for cpu_id in cpu_ids):
                raise ValueError("CPU IDs must be a non-empty list of non-negative IDs")
            _configure_worker_cpu(cpu_ids)
        except OSError, ValueError:
            logger.exception("Could not configure the training worker CPU allocation.")
            return 1

        if args._direction_worker:
            if args.mindset not in ("long", "short") or args.component not in (
                "lgb",
                "gru",
                "all",
            ):
                parser.error(
                    "direction workers require a long/short direction and "
                    "lgb/gru/all component"
                )
            succeeded = _run_direction_worker(args.mindset, args.component)
            if succeeded:
                logger.info(
                    "%s base-model worker completed successfully.", args.mindset
                )
                return 0
            logger.error("%s base-model worker completed with failures.", args.mindset)
            return 1

        if args.mindset is not None or args.component is not None:
            parser.error("PPO worker does not accept training selectors")
        return 0 if _run_ppo_worker() else 1

    if args.download_only:
        if args.mindset is not None or args.component is not None:
            parser.error("--download-only cannot be combined with training selectors")
        from .util import download_missing_stock_data

        failed_downloads = download_missing_stock_data()
        return 1 if failed_downloads else 0

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
