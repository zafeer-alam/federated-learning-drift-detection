import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.clients.client import FLClient
from src.evaluation.metrics import MetricsLogger, RoundMetrics
from src.server.server import FLServer

logger = logging.getLogger(__name__)


class FedAvgCoordinator:
    """
    Coordinates execution of standard Federated Averaging (FedAvg).
    Orchestrates round sampling, parameter broadcasting, local client training,
    server aggregation, and comprehensive metric evaluation.
    """
    def __init__(
        self,
        server: FLServer,
        clients: List[FLClient],
        global_test_loader: DataLoader,
        num_rounds: int = 10,
        client_fraction: float = 1.0,
        local_epochs: int = 2,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        optimizer_type: str = "sgd",
        metrics_logger: Optional[MetricsLogger] = None,
        seed: int = 42,
    ):
        self.server = server
        self.clients = clients
        self.global_test_loader = global_test_loader
        self.num_rounds = num_rounds
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.metrics_logger = metrics_logger or MetricsLogger()
        self.rng = np.random.RandomState(seed)
        self.cumulative_comm_bytes = 0

    def select_clients(self) -> List[FLClient]:
        """
        Samples a subset of clients for participation in the current round.
        """
        num_to_sample = max(1, int(round(self.client_fraction * len(self.clients))))
        selected_indices = self.rng.choice(len(self.clients), size=num_to_sample, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round(self, round_idx: int) -> RoundMetrics:
        """
        Executes one complete FedAvg communication round.
        """
        round_start_time = time.time()
        selected_clients = self.select_clients()
        
        # 1. Server broadcasts global model parameters
        global_params = self.server.broadcast_parameters(num_clients=len(selected_clients))
        round_broadcast_bytes = self.server.model_size_bytes * len(selected_clients)
        
        client_updates = []
        client_train_losses = []
        round_upload_bytes = 0
        
        # 2. Clients receive parameters and train locally
        for client in selected_clients:
            client.set_parameters(global_params)
            updated_params, num_samples, train_metrics = client.train(
                epochs=self.local_epochs,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                optimizer_type=self.optimizer_type,
            )
            client_updates.append((updated_params, num_samples))
            client_train_losses.append(train_metrics["train_loss"])
            round_upload_bytes += self.server.model_size_bytes
            
        # 3. Server aggregates client updates
        self.server.aggregate(client_updates)
        
        # Total communication for this round (downlink + uplink)
        round_comm_bytes = round_broadcast_bytes + round_upload_bytes
        self.cumulative_comm_bytes += round_comm_bytes
        
        # 4. Global model evaluation on held-out test data
        global_eval = self.server.evaluate(self.global_test_loader)
        
        # 5. Local per-client evaluations on client test data using aggregated model
        latest_global_params = self.server.get_parameters()
        per_client_test_acc: Dict[int, float] = {}
        client_test_losses: List[float] = []
        client_test_accs: List[float] = []
        
        for client in self.clients:
            eval_metrics = client.evaluate(latest_global_params)
            per_client_test_acc[client.client_id] = eval_metrics["accuracy"]
            client_test_losses.append(eval_metrics["loss"])
            client_test_accs.append(eval_metrics["accuracy"])
            
            client.state.record_round_metrics(
                round_idx=round_idx,
                metrics={
                    "test_accuracy": eval_metrics["accuracy"],
                    "test_loss": eval_metrics["loss"],
                },
            )
            
        round_duration = time.time() - round_start_time
        
        round_metrics = RoundMetrics(
            round=round_idx,
            global_loss=global_eval["loss"],
            global_accuracy=global_eval["accuracy"],
            mean_client_train_loss=float(np.mean(client_train_losses)),
            mean_client_test_loss=float(np.mean(client_test_losses)),
            mean_client_test_accuracy=float(np.mean(client_test_accs)),
            std_client_test_accuracy=float(np.std(client_test_accs)),
            per_client_test_accuracy=per_client_test_acc,
            round_comm_bytes=round_comm_bytes,
            cumulative_comm_bytes=self.cumulative_comm_bytes,
            round_duration_sec=round_duration,
        )
        
        self.metrics_logger.log_round(round_metrics)
        logger.info(
            f"Round {round_idx:02d}/{self.num_rounds:02d} | "
            f"Global Acc: {round_metrics.global_accuracy * 100:.2f}% | "
            f"Global Loss: {round_metrics.global_loss:.4f} | "
            f"Mean Client Acc: {round_metrics.mean_client_test_accuracy * 100:.2f}% | "
            f"Comm: {round_comm_bytes / 1024:.1f} KB | "
            f"Duration: {round_duration:.2f}s"
        )
        
        return round_metrics

    def run(self) -> pd.DataFrame:
        """
        Executes all rounds of FedAvg training.
        """
        logger.info(
            f"Starting FedAvg run: {self.num_rounds} rounds, {len(self.clients)} clients, "
            f"fraction={self.client_fraction}, local_epochs={self.local_epochs}"
        )
        for r in range(1, self.num_rounds + 1):
            self.run_round(r)
            
        csv_path, json_path = self.metrics_logger.save_results()
        logger.info(f"Training completed. Metrics saved to {csv_path} and {json_path}")
        return self.metrics_logger.to_dataframe()
