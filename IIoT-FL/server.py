import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from task import IIoTFLNet

app = ServerApp()


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = IIoTFLNet()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    

@app.main()
def main(grid: Grid, context: Context) -> None:

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num_server-rounds"]
    lr: float = context.run_config["learning-rate"]

    global_model = IIoTFLNet(
        input_dim=context.run_config["input-dim"],
        hidden_dim=context.run_config["hidden-dim"],
        bottleneck=context.run_config["bottleneck"],
        head_dim=context.run_config["head-dim"],
        dropout=context.run_config["dropout"],
    )
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )