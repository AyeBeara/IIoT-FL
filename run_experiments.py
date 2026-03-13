#!/usr/bin/env python3
"""Run a fixed FL hyperparameter sweep and collect results into CSV.

Sweep factors:
- model.d-token: 64, 128, 256
- train.lr: 1e-4, 5e-4, 1e-3, 5e-3
- num-server-rounds: 10, 25, 40, 50
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import time
from pathlib import Path
from typing import Iterable

TOKEN_DIMS = [64, 128, 256]
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3]
TRAINING_ROUNDS = [10, 25, 40, 50]

SUMMARY_COLUMNS = [
    "experiment_id",
    "token_dim",
    "learning_rate",
    "training_rounds",
    "status",
    "duration_sec",
    "run_metrics_csv",
    "final_eval_round",
    "final_eval_loss",
    "final_eval_rul_mae_log",
    "final_eval_fail_f1",
    "final_eval_fail_accuracy",
    "final_eval_fail_precision",
    "final_eval_fail_recall",
]


def build_experiments() -> list[tuple[int, float, int]]:
    return list(itertools.product(TOKEN_DIMS, LEARNING_RATES, TRAINING_ROUNDS))


def parse_final_eval(metrics_csv: Path) -> dict[str, str]:
    if not metrics_csv.exists():
        return {}

    final_row: dict[str, str] | None = None
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "evaluate":
                continue
            if final_row is None:
                final_row = row
                continue
            try:
                cur_round = int(row.get("round", "-1"))
                best_round = int(final_row.get("round", "-1"))
            except ValueError:
                continue
            if cur_round >= best_round:
                final_row = row

    return final_row or {}


def append_summary_row(summary_csv: Path, row: dict[str, object]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def existing_completed_keys(summary_csv: Path) -> set[tuple[int, float, int]]:
    if not summary_csv.exists():
        return set()

    completed: set[tuple[int, float, int]] = set()
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                key = (
                    int(row["token_dim"]),
                    float(row["learning_rate"]),
                    int(row["training_rounds"]),
                )
            except (KeyError, ValueError):
                continue
            completed.add(key)
    return completed


def iter_runs(
    experiments: Iterable[tuple[int, float, int]],
    max_runs: int | None,
) -> Iterable[tuple[int, tuple[int, float, int]]]:
    for idx, combo in enumerate(experiments, start=1):
        if max_runs is not None and idx > max_runs:
            break
        yield idx, combo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FL sweep for IIoT-FL")
    parser.add_argument(
        "--summary-csv",
        default="metrics/experiments/summary.csv",
        help="Path to aggregate experiment summary CSV",
    )
    parser.add_argument(
        "--run-metrics-dir",
        default="metrics/experiments/runs",
        help="Directory to store per-run Flower metrics CSV files",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of runs (useful for smoke tests)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments already marked as successful in summary CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing Flower",
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    run_metrics_dir = Path(args.run_metrics_dir)
    run_metrics_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments()
    completed_keys = existing_completed_keys(summary_csv) if args.resume else set()

    print(f"Planned combinations: {len(experiments)}")

    for exp_id, (token_dim, lr, rounds) in iter_runs(experiments, args.max_runs):
        key = (token_dim, lr, rounds)
        if key in completed_keys:
            print(
                f"[{exp_id:03d}/{len(experiments)}] SKIP token={token_dim} lr={lr} rounds={rounds}"
            )
            continue

        lr_slug = f"{lr:.0e}".replace("-", "m")
        run_csv = run_metrics_dir / (
            f"exp_{exp_id:03d}_token{token_dim}_lr{lr_slug}_rounds{rounds}.csv"
        )

        run_config = (
            f"model.d-token={token_dim} "
            f"train.lr={lr} "
            f"num-server-rounds={rounds} "
            f"metrics.csv-path=\"{run_csv}\""
        )
        cmd = ["flwr", "run", ".", "--run-config", run_config]

        print(
            f"[{exp_id:03d}/{len(experiments)}] RUN token={token_dim} lr={lr} rounds={rounds}"
        )

        if args.dry_run:
            print("  ", " ".join(cmd))
            continue

        start = time.time()
        proc = subprocess.run(cmd, check=False)
        duration_sec = round(time.time() - start, 2)

        final_eval = parse_final_eval(run_csv)
        status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"

        row = {
            "experiment_id": exp_id,
            "token_dim": token_dim,
            "learning_rate": lr,
            "training_rounds": rounds,
            "status": status,
            "duration_sec": duration_sec,
            "run_metrics_csv": str(run_csv),
            "final_eval_round": final_eval.get("round", ""),
            "final_eval_loss": final_eval.get("loss", ""),
            "final_eval_rul_mae_log": final_eval.get("rul_mae_log", ""),
            "final_eval_fail_f1": final_eval.get("fail_f1", ""),
            "final_eval_fail_accuracy": final_eval.get("fail_accuracy", ""),
            "final_eval_fail_precision": final_eval.get("fail_precision", ""),
            "final_eval_fail_recall": final_eval.get("fail_recall", ""),
        }
        append_summary_row(summary_csv, row)

        print(
            f"  -> {status}, duration={duration_sec}s, final_eval_f1={row['final_eval_fail_f1']}"
        )

    print(f"Summary written to: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
