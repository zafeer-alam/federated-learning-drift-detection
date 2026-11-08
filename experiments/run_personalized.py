import argparse
import logging
import os
import random
import sys
from typing import Any, Dict
import numpy as np
import torch
import yaml

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clients.client import FLClient
from src.evaluation.metrics import MetricsLogger
from src.federated.personalized import PersonalizedFLCoordinator
from src.server.server import FLServer
from src.utils.data import get_client_class_distribution, load_federated_dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Personalized FL Baseline Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/personalized.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default=None, choices=["personalized_finetune", "local_only"], help="FL mode")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--clients", type=int, default=None, help="Override number of clients")
    parser.add_argument("--alpha", type=float, default=None, help="Override Dirichlet alpha")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--local-epochs", type=int, default=None, help="Override local epochs")
    parser.add_argument("--finetune-epochs", type=int, default=None, help="Override finetune epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = load_yaml_config(args.config) if os.path.exists(args.config) else {}

    exp_cfg = config.get("experiment", {})
    data_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    fed_cfg = config.get("federated", {})
    log_cfg = config.get("logging", {})

    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    set_seed(seed)

    device_str = args.device if args.device is not None else exp_cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str

    mode = args.mode or exp_cfg.get("mode", "personalized_finetune")
    exp_name = f"{mode}_mnist"

    log_level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("PersonalizedFL")
    logger.info(f"Initialized Experiment '{exp_name}' [mode={mode}], Seed={seed}, Device={device}")

    num_clients = args.clients if args.clients is not None else data_cfg.get("num_clients", 5)
    partition_strategy = data_cfg.get("partition_strategy", "dirichlet")
    alpha = args.alpha if args.alpha is not None else data_cfg.get("dirichlet_alpha", 0.5)
    dataset_name = data_cfg.get("name", "mnist")
    data_dir = data_cfg.get("data_dir", "./data")

    num_rounds = args.rounds if args.rounds is not None else fed_cfg.get("num_rounds", 5)
    local_epochs = args.local_epochs if args.local_epochs is not None else fed_cfg.get("local_epochs", 1)
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else fed_cfg.get("finetune_epochs", 1)
    client_fraction = fed_cfg.get("client_fraction", 1.0)
    learning_rate = args.lr if args.lr is not None else fed_cfg.get("learning_rate", 0.01)
    finetune_lr = fed_cfg.get("finetune_lr", 0.005)
    momentum = fed_cfg.get("momentum", 0.9)
    weight_decay = fed_cfg.get("weight_decay", 0.0001)
    optimizer_type = fed_cfg.get("optimizer", "sgd")

    architecture = model_cfg.get("architecture", "cnn")
    num_classes = model_cfg.get("num_classes", 10)
    in_channels = model_cfg.get("in_channels", 1)
    output_dir = exp_cfg.get("output_dir", "./results")

    # 1. Load partitioned data
    client_splits, global_test_loader = load_federated_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        dirichlet_alpha=alpha,
        batch_size=32,
        seed=seed,
    )

    # 2. Server (only needed for personalized_finetune mode)
    server = None
    if mode == "personalized_finetune":
        server = FLServer(
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            device=device,
        )

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

    # 4. Coordinator
    metrics_logger = MetricsLogger(output_dir=output_dir, experiment_name=exp_name)
    coordinator = PersonalizedFLCoordinator(
        server=server,
        clients=clients,
        global_test_loader=global_test_loader,
        mode=mode,
        num_rounds=num_rounds,
        client_fraction=client_fraction,
        local_epochs=local_epochs,
        finetune_epochs=finetune_epochs,
        learning_rate=learning_rate,
        finetune_lr=finetune_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        metrics_logger=metrics_logger,
        seed=seed,
    )

    # 5. Execute Run
    df = coordinator.run()

    # 6. Print summary
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT COMPLETED: {exp_name.upper()}")
    logger.info("=" * 70)
    print("\n" + coordinator.metrics_logger.get_summary_table() + "\n")


if __name__ == "__main__":
    main()
