from typing import Any

def extract_section(run_config: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Separate run_config into sections based on key prefixes."""

    split = {}
    search = f"{prefix}."
    for k, v in run_config.items():
        if k.startswith(search):
            split[k[len(search):]] = v
    return split

def build_model_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Return model configurations from the run_config."""

    raw_model_config = extract_section(run_config, "model")
    return {
        "input_dim":            int(raw_model_config["input-dim"]),
        "d_token":              int(raw_model_config["d-token"]),
        "n_blocks":             int(raw_model_config["n-blocks"]),
        "attention_heads":      int(raw_model_config["attention-heads"]),
        "dropout":              float(raw_model_config["dropout"]),
        "ffn_dim_multiplier":   float(raw_model_config["ffn-dim-multiplier"]),
    }

def build_train_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Return training configurations from the run_config."""

    raw_train_config = extract_section(run_config, "train")
    return {
        "local_epochs":         int(raw_train_config["local-epochs"]),
        "batch_size":           int(raw_train_config["batch-size"]),
        "lr":                   float(raw_train_config["lr"]),
        "weight_decay":         float(raw_train_config["weight-decay"]),
        "rul_loss_weight":      float(raw_train_config["rul-loss-weight"]),
        "failure_loss_weight":  float(raw_train_config["failure-loss-weight"]),
        "scheduler":            str(raw_train_config["scheduler"])
    }

def build_strategy_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Return federated strategy hyperparameters."""
    
    raw_strategy_config = extract_section(run_config, "strategy")
    return {
        "name":             str(raw_strategy_config["name"]),
        "server_lr":        float(raw_strategy_config["server-lr"]),
        "server_momentum":  float(raw_strategy_config["server-momentum"]),
        "tau":              float(raw_strategy_config["tau"])
    }