import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _build_command(entry: str, base_args: List[str], extra_args: str) -> List[str]:
    """Compose a subprocess command using the current Python executable."""
    cmd: List[str] = [sys.executable, entry, *base_args]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def _run_stage(name: str, cmd: List[str]) -> float:
    """Run a stage and exit immediately if it fails."""
    print(f"\n[run_full_eval] Starting {name}: {' '.join(cmd)}")
    start = time.perf_counter()
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    print(f"[run_full_eval] Finished {name}")
    return time.perf_counter() - start


def _extract_flag_value(cmd: List[str], flag: str, cast=float) -> Optional[float]:
    """Grab the value following `flag` in cmd, if it exists."""
    for idx, token in enumerate(cmd):
        if token == flag and idx + 1 < len(cmd):
            try:
                return cast(cmd[idx + 1])
            except (TypeError, ValueError):
                return None
    return None


def _init_wandb(args: argparse.Namespace, train_cmd: List[str]):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required but not installed. Please `pip install wandb`.") from exc

    config = {
        "source_path": args.source_path,
        "model_path": args.model_path,
        "train_args": args.train_args,
        "render_args": args.render_args,
        "metrics_args": args.metrics_args,
        "lambda_converge": _extract_flag_value(train_cmd, "--lambda_converge"),
        "converge_knn": _extract_flag_value(train_cmd, "--converge_knn", int),
        "converge_interval": _extract_flag_value(train_cmd, "--converge_interval", int),
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
    )
    return run


def _log_metrics_from_file(model_path: str, method_name: str) -> Optional[dict]:
    results_path = Path(model_path) / "results.json"
    if not results_path.exists():
        print(f"[run_full_eval] Warning: {results_path} not found, skipping metric logging.")
        return None
    with open(results_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    metrics = data.get(method_name)
    if metrics is None:
        print(f"[run_full_eval] Warning: method '{method_name}' not found in results.json.")
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train.py -> render.py -> metrics.py sequentially."
    )
    parser.add_argument(
        "-s",
        "--source_path",
        required=True,
        help="Path passed to train.py via -s/--source_path.",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Model output directory used across all steps.",
    )
    parser.add_argument(
        "--train-args",
        default="",
        help="Extra CLI arguments appended after the default train.py invocation.",
    )
    parser.add_argument(
        "--render-args",
        default="--eval --skip_train",
        help="Extra CLI arguments appended after the default render.py invocation.",
    )
    parser.add_argument(
        "--metrics-args",
        default="",
        help="Extra CLI arguments appended after the default metrics.py invocation.",
    )
    parser.add_argument(
        "--metrics-method",
        default="ours",
        help="Method directory name under test/ used for logging metrics.",
    )
    parser.add_argument(
        "--wandb-project",
        default="",
        help="If provided, enables Weights & Biases logging to this project.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity / team.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name.",
    )
    args = parser.parse_args()

    train_cmd = _build_command(
        "train.py",
        ["-s", args.source_path, "-m", args.model_path],
        args.train_args,
    )
    render_cmd = _build_command(
        "render.py",
        ["-m", args.model_path],
        args.render_args,
    )
    metrics_cmd = _build_command(
        "metrics.py",
        ["-m", args.model_path],
        args.metrics_args,
    )

    wandb_run = _init_wandb(args, train_cmd)

    train_time = _run_stage("training", train_cmd)
    if wandb_run:
        wandb_run.log({"timing/train_seconds": train_time})
    render_time = _run_stage("rendering", render_cmd)
    if wandb_run:
        wandb_run.log({"timing/render_seconds": render_time})
    metrics_time = _run_stage("metrics", metrics_cmd)

    metrics = _log_metrics_from_file(args.model_path, args.metrics_method)
    if wandb_run:
        if metrics:
            wandb_run.log(
                {
                    "metrics/ssim": metrics.get("SSIM"),
                    "metrics/psnr": metrics.get("PSNR"),
                    "metrics/lpips": metrics.get("LPIPS"),
                }
            )
        wandb_run.log(
            {
                "timing/metrics_seconds": metrics_time,
                "timing/total_seconds": train_time + render_time + metrics_time,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
