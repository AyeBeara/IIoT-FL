import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_dataset
from flwr_datasets.partitioner import PathologicalPartitioner

import kagglehub
import numpy as np

class IIoTFLNet(nn.Module):
    class DSB(nn.Module):
        def __init__(
                self,
                in_channels: int,
                bottleneck: int,
                out_channels: int,
                dropout: float=0.0,
        ):
            super(IIoTFLNet.DSB, self).__init__()
            self.dsb = nn.Sequential(
                nn.Linear(in_channels, bottleneck, bias=False),
                nn.SiLU(),
                nn.Linear(bottleneck, out_channels),
                nn.SiLU(),
                nn.LayerNorm(out_channels),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.dsb(x)


    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            bottleneck: int,
            head_dim: int,
            dropout: float=0.0,
    ):
        super(IIoTFLNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.trunk = nn.Sequential(
            self.DSB(hidden_dim, bottleneck, head_dim, dropout),
            self.DSB(head_dim, bottleneck, head_dim, dropout)
        )

        self.rul_head = nn.Sequential(
            self.Linear(hidden_dim, head_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            self.Linear(head_dim, 1)
        )

        self.failure_head = nn.Sequential(
            self.Linear(hidden_dim, head_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            self.Linear(head_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc(x)
        x = self.trunk(x)
        rul = self.rul_head(x)
        failure = self.failure_head(x)
        return rul, failure

class DualTaskLoss(nn.Module):
    def __init__(
            self,
            pos_weight: torch.Tensor | None = None, 
            huber_delta: float = 30.0
        ):
        super(DualTaskLoss, self).__init__()
        self.log_sigma_rul = nn.Parameter(torch.zeros(1))
        self.log_sigma_failure = nn.Parameter(torch.zeros(1))

        self.huber = nn.HuberLoss(delta=huber_delta, reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
            self,
            rul_pred: torch.Tensor,
            failure_pred: torch.Tensor,
            rul_target: torch.Tensor,
            failure_target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        rul_pred = F.softplus(rul_pred)

        loss_rul = self.huber(rul_pred, rul_target)
        loss_failure = self.bce(failure_pred, failure_target)

        # Uncertainty-weighted sum: L = L1/(2σ1²) + L2/(2σ2²) + log(σ1·σ2)
        precision_rul = torch.exp(-2 * self.log_sigma_rul)
        precision_failure = torch.exp(-2 * self.log_sigma_failure)
        
        total = (precision_rul * loss_rul + self.log_sigma_rul + precision_failure * loss_failure + self.log_sigma_failure)

        return total, {
            'loss_rul': loss_rul.item(),
            'loss_failure': loss_failure.item(),
            'sigma_rul': torch.exp(self.log_sigma_rul).item(),
            'sigma_failure': torch.exp(self.log_sigma_failure).item()
        }

@torch.inference_mode()
def predict(model: IIoTFLNet, x: torch.Tensor) -> dict:
    model.eval()
    rul_pred, failure_pred = model(x)
    rul_days = F.softplus(rul_pred).squeeze(1)
    failure_prob = torch.sigmoid(failure_pred).squeeze(1)
    return {
        'rul_days': rul_days,
        'failure_prob': failure_prob,
        'failure_flag': failure_prob > 0.5
    }

pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

fds = None

def apply_transforms(batch):
    batch["sensor_values"] = [pytorch_transforms(values) for values in batch["sensor_values"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    global fds
    if fds is None:
        path = kagglehub.dataset_download("canozensoy/industrial-iot-dataset-synthetic")
        data = path + "/factory_sensor_simulator_2040.csv"
        dataset = load_dataset("csv", datafiles=data)

        partitioner = PathologicalPartitioner(np.unique(dataset["Machine_Type"]), "Machine_Type",num_classes_per_partition=1)
        partitioner.dataset = dataset
        fds = partitioner
    
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def load_centralized_dataset():
    path = kagglehub.dataset_download("canozensoy/industrial-iot-dataset-synthetic")
    data = path + "/factory_sensor_simulator_2040.csv"
    test_dataset = load_dataset("csv", datafiles=data, split="train")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def train(model, trainloader, epochs, lr, device):
    model.to(device)
    criterion = DualTaskLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            pass