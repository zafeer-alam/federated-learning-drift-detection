import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model import (
    compute_model_size_bytes,
    get_model,
    get_model_parameters,
    set_model_parameters,
)


class FLServer:
    """
    Central Federated Learning Server responsible for:
    - Maintaining the canonical global model
    - Broadcasting global parameters
    - Performing FedAvg aggregation weighted by client sample sizes
    - Evaluating global model generalization on held-out global test data
    - Tracking server-side communication overhead
    """
    def __init__(
        self,
        architecture: str = "cnn",
        num_classes: int = 10,
        in_channels: int = 1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = get_model(architecture, in_channels=in_channels, num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.current_round: int = 0
        self.cumulative_bytes_broadcast: int = 0
        self.cumulative_bytes_received: int = 0
        self.model_size_bytes = compute_model_size_bytes(self.model)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns a detached copy of global model parameters on CPU.
        """
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Loads parameters into global model.
        """
        set_model_parameters(self.model, parameters)

    def broadcast_parameters(self, num_clients: int) -> Dict[str, torch.Tensor]:
        """
        Returns parameters for distribution and tracks cumulative server broadcast bytes.
        """
        params = self.get_parameters()
        self.cumulative_bytes_broadcast += self.model_size_bytes * num_clients
        return params

    def aggregate(
        self,
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Performs Federated Averaging (FedAvg):
        theta_{t+1} = sum_{k in S_t} (n_k / n) * theta_{t+1}^k

        Args:
            client_updates: List of tuples (client_parameters, num_samples)
        Returns:
            Aggregated global parameters dictionary
        """
        if not client_updates:
            raise ValueError("Cannot aggregate empty client updates.")

        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples <= 0:
            raise ValueError("Total sample count across clients must be positive.")

        # Account for payload bytes received by server
        for params, _ in client_updates:
            param_bytes = sum(p.nelement() * p.element_size() for p in params.values())
            self.cumulative_bytes_received += param_bytes

        # Initialize accumulator with zeros
        base_params = client_updates[0][0]
        aggregated_params: Dict[str, torch.Tensor] = {}
        for key in base_params.keys():
            aggregated_params[key] = torch.zeros_like(base_params[key], dtype=torch.float32)

        # Weighted accumulation
        for client_params, num_samples in client_updates:
            weight = float(num_samples) / float(total_samples)
            for key in aggregated_params.keys():
                # Cast to float32 for aggregation, then match target tensor dtype
                client_tensor = client_params[key].to(torch.float32)
                aggregated_params[key] += weight * client_tensor

        # Cast back to original dtypes if necessary
        for key in aggregated_params.keys():
            target_dtype = base_params[key].dtype
            if aggregated_params[key].dtype != target_dtype:
                aggregated_params[key] = aggregated_params[key].to(target_dtype)

        self.set_parameters(aggregated_params)
        self.current_round += 1
        return aggregated_params

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluates current global model on a central test loader.
        """
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * len(target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total_samples,
        }
