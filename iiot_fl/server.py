import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdam, FedAvg

from iiot_fl.config import build_model_config, build_strategy_config
from iiot_fl.dataset import INPUT_DIM
from iiot_fl.model import IIoTFLNet
from iiot_fl.task import get_parameters

logger = logging.getLogger(__name__)

SERVER_METRICS_CSV: Path | None = None
FIT_ROUND = 0
EVAL_ROUND = 0

SERVER_CSV_COLUMNS = [
    "timestamp",
    "round",
    "phase",
    "num_clients",
    "total_samples",
    "loss",
    "rul_mae_log",
    "fail_f1",
    "fail_accuracy",
    "fail_precision",
    "fail_recall",
    "avg_loss",
    "avg_rul_loss",
    "avg_fail_loss",
]


def _append_server_metrics_csv(
    phase: str,
    round_num: int,
    num_clients: int,
    total_samples: int,
    metrics: Metrics,
) -> None:
    if SERVER_METRICS_CSV is None:
        return

    try:
        SERVER_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not SERVER_METRICS_CSV.exists()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round": round_num,
            "phase": phase,
            "num_clients": num_clients,
            "total_samples": total_samples,
            "loss": metrics.get("loss", ""),
            "rul_mae_log": metrics.get("rul_mae_log", ""),
            "fail_f1": metrics.get("fail_f1", ""),
            "fail_accuracy": metrics.get("fail_accuracy", ""),
            "fail_precision": metrics.get("fail_precision", ""),
            "fail_recall": metrics.get("fail_recall", ""),
            "avg_loss": metrics.get("avg_loss", ""),
            "avg_rul_loss": metrics.get("avg_rul_loss", ""),
            "avg_fail_loss": metrics.get("avg_fail_loss", ""),
        }

        with SERVER_METRICS_CSV.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SERVER_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        logger.exception("Failed writing server metrics CSV")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global EVAL_ROUND

    if not metrics:
        return {}

    total_samples = sum(n for n, _ in metrics)
    aggregated = {}
    keys = list(metrics[0][1].keys())

    for key in keys:
        aggregated[key] = (
            sum(n * m[key] for n, m in metrics if key in m) / total_samples
        )

    logger.info(
        "Round aggregated metrics | loss=%.4f | rul_mae=%.4f | f1=%.4f",
        aggregated.get("loss", float("nan")),
        aggregated.get("rul_mae_log", float("nan")),
        aggregated.get("fail_f1", float("nan")),
    )

    EVAL_ROUND += 1
    _append_server_metrics_csv(
        phase="evaluate",
        round_num=EVAL_ROUND,
        num_clients=len(metrics),
        total_samples=total_samples,
        metrics=aggregated,
    )

    return aggregated


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global FIT_ROUND

    if not metrics:
        return {}

    total_samples = sum(n for n, _ in metrics)
    aggregated = {}
    keys = list(metrics[0][1].keys())

    for key in keys:
        aggregated[key] = (
            sum(n * m[key] for n, m in metrics if key in m) / total_samples
        )

    FIT_ROUND += 1
    _append_server_metrics_csv(
        phase="fit",
        round_num=FIT_ROUND,
        num_clients=len(metrics),
        total_samples=total_samples,
        metrics=aggregated,
    )

    return aggregated


def server_fn(context: Context) -> ServerAppComponents:
    global SERVER_METRICS_CSV, FIT_ROUND, EVAL_ROUND

    config = context.run_config
    model_config = build_model_config(config)
    strategy_config = build_strategy_config(config)

    model_config["input_dim"] = INPUT_DIM

    metrics_csv_path = str(config.get("metrics.csv-path", "metrics/flwr_metrics.csv"))
    SERVER_METRICS_CSV = Path(metrics_csv_path) if metrics_csv_path else None
    if SERVER_METRICS_CSV is not None and SERVER_METRICS_CSV.exists():
        SERVER_METRICS_CSV.unlink()
    FIT_ROUND = 0
    EVAL_ROUND = 0

    init_model = IIoTFLNet(model_config)
    init_ndarrays = get_parameters(init_model)
    init_params = ndarrays_to_parameters(init_ndarrays)

    logger.info(
        "Global model initialized | input_dim=%d | d_token=%d | b_blocks=%d",
        model_config["input_dim"],
        model_config["d_token"],
        model_config["n_blocks"],
    )

    common_kwargs = dict(
        fraction_fit=float(config["fraction-fit"]),
        fraction_evaluate=float(config["fraction-evaluate"]),
        min_fit_clients=int(config["min-fit-clients"]),
        min_evaluate_clients=int(config["min-evaluate-clients"]),
        min_available_clients=int(config["min-available-clients"]),
        initial_parameters=init_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )

    if strategy_config["name"] == "FedAdam":
        strategy = FedAdam(
            **common_kwargs,
            eta=strategy_config["server_lr"],
            eta_l=float(config["train.lr"]),
            beta_1=strategy_config["server_momentum"],
            beta_2=0.999,
            tau=float(strategy_config["tau"])
        )

        logger.info(
            "Strategy: FedAdam | server_lr=%.4f | tau=%.1e",
            strategy_config["server_lr"],
            strategy_config["tau"],
        )

    else:
        strategy = FedAvg(**common_kwargs)
        logger.info("Strategy: FedAvg")

    server_config = ServerConfig(num_rounds=int(config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
