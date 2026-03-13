import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from iiot_fl.model import IIoTFLNet, DualTaskLoss

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "timestamp",
    "round",
    "phase",
    "machine_type",
    "num_samples",
    "loss",
    "avg_loss",
    "avg_rul_loss",
    "avg_fail_loss",
    "rul_mae_log",
    "fail_accuracy",
    "fail_f1",
    "fail_precision",
    "fail_recall",
]


def append_metrics_to_csv(
    metrics_dir: str | None,
    machine_type: str,
    phase: str,
    round_num: int,
    num_samples: int,
    loss: float,
    metrics: Dict[str, float],
) -> None:
    if not metrics_dir:
        return

    try:
        out_dir = Path(metrics_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"{machine_type}.csv"

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": round_num,
            "phase": phase,
            "machine_type": machine_type,
            "num_samples": num_samples,
            "loss": loss,
            "avg_loss": metrics.get("avg_loss", ""),
            "avg_rul_loss": metrics.get("avg_rul_loss", ""),
            "avg_fail_loss": metrics.get("avg_fail_loss", ""),
            "rul_mae_log": metrics.get("rul_mae_log", ""),
            "fail_accuracy": metrics.get("fail_accuracy", ""),
            "fail_f1": metrics.get("fail_f1", ""),
            "fail_precision": metrics.get("fail_precision", ""),
            "fail_recall": metrics.get("fail_recall", ""),
        }

        write_header = not file_path.exists()
        with file_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        logger.exception("Failed to append metrics CSV for machine '%s'", machine_type)


def get_parameters(model: nn.Module) -> list:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list):
    from collections import OrderedDict

    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def train(
    model: IIoTFLNet,
    train_loader: DataLoader,
    criterion: DualTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    local_epochs: int,
) -> Dict[str, float]:
    model.train()
    final_metrics = {}

    for epoch in range(local_epochs):
        total_loss = 0.0
        total_rul = 0.0
        total_fail = 0.0
        n = 0

        for x, rul_true, fail_true in train_loader:
            x = x.to(device)
            rul_true = rul_true.to(device)
            fail_true = fail_true.to(device)

            optimizer.zero_grad()
            rul_pred, fail_logit = model(x)

            loss, rul_loss, fail_loss = criterion(
                rul_pred, rul_true, fail_logit, fail_true
            )
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = len(rul_true)
            total_loss += loss.item() * batch_n
            total_rul += rul_loss.item() * batch_n
            total_fail += fail_loss.item() * batch_n
            n += batch_n

        if scheduler is not None:
            scheduler.step()

        final_metrics = {
            "avg_loss": total_loss / n,
            "avg_rul_loss": total_rul / n,
            "avg_fail_loss": total_fail / n,
        }
        logger.info(
            "  Epoch %d/%d | loss=%.4f | rul=%.4f | fail=%.4f",
            epoch + 1,
            local_epochs,
            final_metrics["avg_loss"],
            final_metrics["avg_rul_loss"],
            final_metrics["avg_fail_loss"],
        )

    return final_metrics


def evaluate(
    model: IIoTFLNet,
    val_loader: DataLoader,
    criterion: DualTaskLoss,
    device: torch.device,
) -> Tuple[float, int, Dict[str, float]]:
    model.eval()

    total_loss = 0.0
    rul_mae = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0
    n = 0

    with torch.no_grad():
        for x, rul_true, fail_true in val_loader:
            x = x.to(device)
            rul_true = rul_true.to(device)
            fail_true = fail_true.to(device)

            rul_pred, fail_logit = model(x)
            loss, _, _ = criterion(rul_pred, rul_true, fail_logit, fail_true)

            batch_n = len(rul_true)
            total_loss += loss.item() * batch_n

            rul_mae += (
                torch.abs(torch.log1p(rul_pred.squeeze()) - torch.log1p(rul_true))
                .sum()
                .item()
            )

            preds = (torch.sigmoid(fail_logit.squeeze()) > 0.5).long()
            gt = fail_true.long()
            tp += ((preds == 1) & (gt == 1)).sum().item()
            fp += ((preds == 1) & (gt == 0)).sum().item()
            tn += ((preds == 0) & (gt == 0)).sum().item()
            fn += ((preds == 0) & (gt == 1)).sum().item()
            n += batch_n

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {
        "rul_mae_log": rul_mae / n,
        "fail_accuracy": (tp + tn) / n,
        "fail_f1": f1,
        "fail_precision": precision,
        "fail_recall": recall,
    }

    logger.info(
        "  Eval | loss=%.4f | rul_mae_log=%.4f | fail_acc=%.4f | fail_f1=%.4f",
        total_loss / n,
        metrics["rul_mae_log"],
        metrics["fail_accuracy"],
        metrics["fail_f1"],
    )

    return float(total_loss / n), n, metrics
