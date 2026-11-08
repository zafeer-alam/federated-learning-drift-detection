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


class PersonalizedFLCoordinator:
    """
    Coordinates Personalized Federated Learning baselines:
    1. mode='personalized_finetune': FedAvg global aggregation combined with
       per-client local fine-tuning on client-specific non-IID distributions.
    2. mode='local_only': Pure independent local training with zero server aggregation
       and zero communication overhead.
    """
    def __init__(
        self,
        server: Optional[FLServer],
        clients: List[FLClient],
        global_test_loader: DataLoader,
        mode: str = "personalized_finetune",
        num_rounds: int = 10,
        client_fraction: float = 1.0,
        local_epochs: int = 2,
        finetune_epochs: int = 1,
        learning_rate: float = 0.01,
        finetune_lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        optimizer_type: str = "sgd",
        metrics_logger: Optional[MetricsLogger] = None,
        seed: int = 42,
    ):
        self.server = server
        self.clients = clients
        self.global_test_loader = global_test_loader
        self.mode = mode.lower().strip()
        self.num_rounds = num_rounds
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.finetune_epochs = finetune_epochs
        self.learning_rate = learning_rate
        self.finetune_lr = finetune_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.metrics_logger = metrics_logger or MetricsLogger()
        self.rng = np.random.RandomState(seed)
        self.cumulative_comm_bytes = 0

    def select_clients(self) -> List[FLClient]:
        """Samples client cohort for current round."""
        num_to_sample = max(1, int(round(self.client_fraction * len(self.clients))))
        selected_indices = self.rng.choice(len(self.clients), size=num_to_sample, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round_local_only(self, round_idx: int) -> RoundMetrics:
        """
        Executes one round where clients train completely independently
        with zero communication with the server.
        """
        start_time = time.time()
        client_train_losses = []
        client_test_accs = []
        client_test_losses = []
        per_client_accs: Dict[int, float] = {}
        
        # Each client trains independently on its own local data
        for client in self.clients:
            train_metrics = client.train_local_only(
                epochs=self.local_epochs,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                optimizer_type=self.optimizer_type,
            )
            client_train_losses.append(train_metrics["train_loss"])
            
            # Evaluate on client's local test set
            eval_metrics = client.evaluate()
            client_test_accs.append(eval_metrics["accuracy"])
            client_test_losses.append(eval_metrics["loss"])
            per_client_accs[client.client_id] = eval_metrics["accuracy"]
            
        # Global evaluation: evaluate Client 0's model on global test set to measure isolated generalization
        global_eval = self.clients[0].evaluate()
        round_duration = time.time() - start_time
        
        round_metrics = RoundMetrics(
            round=round_idx,
            global_loss=global_eval["loss"],
            global_accuracy=float(np.mean(client_test_accs)), # mean local accuracy
            mean_client_train_loss=float(np.mean(client_train_losses)),
            mean_client_test_loss=float(np.mean(client_test_losses)),
            mean_client_test_accuracy=float(np.mean(client_test_accs)),
            std_client_test_accuracy=float(np.std(client_test_accs)),
            per_client_test_accuracy=per_client_accs,
            round_comm_bytes=0, # ZERO communication for local training
            cumulative_comm_bytes=0,
            round_duration_sec=round_duration,
        )
        
        self.metrics_logger.log_round(round_metrics)
        logger.info(
            f"Round {round_idx:02d}/{self.num_rounds:02d} [Local Only] | "
            f"Mean Local Acc: {round_metrics.mean_client_test_accuracy * 100:.2f}% | "
            f"Loss: {round_metrics.mean_client_test_loss:.4f} | "
            f"Comm: 0.0 KB | Duration: {round_duration:.2f}s"
        )
        return round_metrics

    def run_round_personalized(self, round_idx: int) -> RoundMetrics:
        """
        Executes one round of Personalized FL:
        1. FedAvg global aggregation (shared global backbone)
        2. Per-client fine-tuning to specialize on local non-IID distribution
        """
        if self.server is None:
            raise ValueError("Server must be provided for personalized_finetune mode.")
            
        start_time = time.time()
        selected_clients = self.select_clients()
        
        # 1. Server broadcasts global model
        global_params = self.server.broadcast_parameters(num_clients=len(selected_clients))
        round_broadcast_bytes = self.server.model_size_bytes * len(selected_clients)
        
        client_updates = []
        client_train_losses = []
        round_upload_bytes = 0
        
        # 2. Local FedAvg training for global aggregation
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
            
        # 3. Server aggregates updates into new global model
        self.server.aggregate(client_updates)
        new_global_params = self.server.get_parameters()
        
        # Total round communication
        round_comm_bytes = round_broadcast_bytes + round_upload_bytes
        self.cumulative_comm_bytes += round_comm_bytes
        
        # 4. Evaluate global model on held-out global test set
        global_eval = self.server.evaluate(self.global_test_loader)
        
        # 5. Personalization step: each client fine-tunes from latest global model on local data
        personalized_accs = []
        personalized_losses = []
        per_client_accs: Dict[int, float] = {}
        
        for client in self.clients:
            client.fine_tune(
                base_parameters=new_global_params,
                epochs=self.finetune_epochs,
                lr=self.finetune_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                optimizer_type=self.optimizer_type,
            )
            eval_metrics = client.evaluate_personalized()
            personalized_accs.append(eval_metrics["accuracy"])
            personalized_losses.append(eval_metrics["loss"])
            per_client_accs[client.client_id] = eval_metrics["accuracy"]
            
        round_duration = time.time() - start_time
        
        round_metrics = RoundMetrics(
            round=round_idx,
            global_loss=global_eval["loss"],
            global_accuracy=global_eval["accuracy"],
            mean_client_train_loss=float(np.mean(client_train_losses)),
            mean_client_test_loss=float(np.mean(personalized_losses)),
            mean_client_test_accuracy=float(np.mean(personalized_accs)),
            std_client_test_accuracy=float(np.std(personalized_accs)),
            per_client_test_accuracy=per_client_accs,
            round_comm_bytes=round_comm_bytes,
            cumulative_comm_bytes=self.cumulative_comm_bytes,
            round_duration_sec=round_duration,
        )
        
        self.metrics_logger.log_round(round_metrics)
        logger.info(
            f"Round {round_idx:02d}/{self.num_rounds:02d} [Personalized] | "
            f"Global Acc: {round_metrics.global_accuracy * 100:.2f}% | "
            f"Personalized Acc: {round_metrics.mean_client_test_accuracy * 100:.2f}% | "
            f"Comm: {round_comm_bytes / 1024:.1f} KB | Duration: {round_duration:.2f}s"
        )
        return round_metrics

    def run_round(self, round_idx: int) -> RoundMetrics:
        """Executes one round according to coordinator mode."""
        if self.mode == "local_only":
            return self.run_round_local_only(round_idx)
        else:
            return self.run_round_personalized(round_idx)

    def run(self) -> pd.DataFrame:
        """Executes all communication rounds."""
        logger.info(
            f"Starting Personalized FL run [mode={self.mode}]: {self.num_rounds} rounds, "
            f"{len(self.clients)} clients, finetune_epochs={self.finetune_epochs}"
        )
        for r in range(1, self.num_rounds + 1):
            self.run_round(r)
            
        csv_path, json_path = self.metrics_logger.save_results()
        logger.info(f"Run completed. Metrics saved to {csv_path} and {json_path}")
        return self.metrics_logger.to_dataframe()
