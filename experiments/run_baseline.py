import argparse
import logging
import os
import random
import sys
from typing import Any, Dict
import numpy as np
import torch
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clients.client import FLClient
from src.evaluation.metrics import MetricsLogger
from src.federated.fedavg import FedAvgCoordinator
from src.server.server import FLServer
from src.utils.data import get_client_class_distribution, load_federated_dataset


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for full experimental reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="FedAvg Baseline Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--clients", type=int, default=None, help="Override number of clients")
    parser.add_argument("--alpha", type=float, default=None, help="Override Dirichlet alpha")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--local-epochs", type=int, default=None, help="Override local epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config) if os.path.exists(args.config) else {}

    exp_cfg = config.get("experiment", {})
    data_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    fed_cfg = config.get("federated", {})
    log_cfg = config.get("logging", {})

    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    set_seed(seed)

    device_str = args.device if args.device is not None else exp_cfg.get("device", "auto")
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    # Logging setup
    log_level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("FedAvgBaseline")
    logger.info(f"Initialized Experiment with Seed={seed}, Device={device}")

    # Parameters
    num_clients = args.clients if args.clients is not None else data_cfg.get("num_clients", 5)
    partition_strategy = data_cfg.get("partition_strategy", "dirichlet")
    alpha = args.alpha if args.alpha is not None else data_cfg.get("dirichlet_alpha", 0.5)
    batch_size = args.batch_size if args.batch_size is not None else fed_cfg.get("batch_size", 32)
    dataset_name = data_cfg.get("name", "mnist")
    data_dir = data_cfg.get("data_dir", "./data")

    num_rounds = args.rounds if args.rounds is not None else fed_cfg.get("num_rounds", 10)
    local_epochs = args.local_epochs if args.local_epochs is not None else fed_cfg.get("local_epochs", 2)
    client_fraction = fed_cfg.get("client_fraction", 1.0)
    learning_rate = args.lr if args.lr is not None else fed_cfg.get("learning_rate", 0.01)
    momentum = fed_cfg.get("momentum", 0.9)
    weight_decay = fed_cfg.get("weight_decay", 0.0001)
    optimizer_type = fed_cfg.get("optimizer", "sgd")

    architecture = model_cfg.get("architecture", "cnn")
    num_classes = model_cfg.get("num_classes", 10)
    in_channels = model_cfg.get("in_channels", 1)

    output_dir = exp_cfg.get("output_dir", "./results")
    exp_name = exp_cfg.get("name", "fedavg_baseline")

    # 1. Load and partition data
    logger.info(
        f"Loading dataset '{dataset_name}' with {num_clients} clients, strategy='{partition_strategy}' (alpha={alpha})..."
    )
    client_splits, global_test_loader = load_federated_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        dirichlet_alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )

    # Print class distribution per client
    dist_matrix = get_client_class_distribution(client_splits, num_classes=num_classes)
    logger.info("Non-IID Client Class Distribution Matrix (rows: clients, cols: classes 0-9):")
    for k in range(num_clients):
        logger.info(f"Client {k:02d} (n={client_splits[k].num_train_samples:4d}): {dist_matrix[k].tolist()}")

    # 2. Instantiate Server
    server = FLServer(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        device=device,
    )
    logger.info(f"Initialized FLServer with architecture='{architecture}', model_size={server.model_size_bytes} bytes")

    # 3. Instantiate Clients
    clients = []
    for split in client_splits:
        client = FLClient(
            client_id=split.client_id,
            train_loader=split.train_loader,
            test_loader=split.test_loader,
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            device=device,
        )
        clients.append(client)
    logger.info(f"Initialized {len(clients)} FLClients successfully.")

    # 4. Coordinator & Metrics
    metrics_logger = MetricsLogger(output_dir=output_dir, experiment_name=exp_name)
    coordinator = FedAvgCoordinator(
        server=server,
        clients=clients,
        global_test_loader=global_test_loader,
        num_rounds=num_rounds,
        client_fraction=client_fraction,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        metrics_logger=metrics_logger,
        seed=seed,
    )

    # 5. Run FedAvg baseline
    df_metrics = coordinator.run()

    # 6. Output Final Results Summary
    logger.info("=" * 70)
    logger.info("FEDAVG BASELINE EXPERIMENT COMPLETED")
    logger.info("=" * 70)
    print("\n" + coordinator.metrics_logger.get_summary_table() + "\n")


if __name__ == "__main__":
    main()
