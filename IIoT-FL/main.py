

import kagglehub
import os
from pathlib import Path
from socketserver import DatagramRequestHandler
import token

from scipy import special
import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.utils.data import DataLoader, random_split


from utils import (
    set_seed,
    init_wandb,
    finish_wandb,
    plot_learning_curves_classification,
    visualize_results,
    get_predictions
)

################ CONFIGURATION ################
FIG_DIR = Path("./figures")
MODEL_DIR = Path("./models")

USE_WANDB = True

path = kagglehub.dataset_download("canozensoy/industrial-iot-dataset-synthetic")


################ MODEL ################
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__(
            self,

        )
        # TODO: Define your CNN-LSTM architecture here

        raise NotImplementedError("Define the CNN-LSTM architecture in the __init__ method.")

    def forward(self, x):
        # TODO: Define the forward pass
        raise NotImplementedError("Define the forward pass in the forward method.")
    

################ TRAINING AND EVALUATION ################
def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> tuple[float, float]:

    raise NotImplementedError("Implement the training loop for one epoch in the train_one_epoch function.")

def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> tuple[float, float]:

    raise NotImplementedError("Implement the evaluation loop in the evaluate function.")

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = 20,
        use_wandb: bool = USE_WANDB,
        verbose: bool = True
) -> dict[str, list[float]]:

    raise NotImplementedError("Implement the full training loop in the train_model function.")

################ MAIN ################
def main():
    ################ SETUP ################
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.pt"

    ################ HYPERPARAMETERS ################

    

    ################ DATA ################


if __name__ == "__main__":
    main()