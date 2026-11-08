import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.clients.client_state import ClientState
from src.models.model import (
    compute_model_size_bytes,
    get_model,
    get_model_parameters,
    set_model_parameters,
)


class FLClient:
    """
    Standard Federated Learning Client capable of local training, evaluation,
    and rigorous tracking of computation time and communication bandwidth.
    """
    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        architecture: str = "cnn",
        num_classes: int = 10,
        in_channels: int = 1,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.model = get_model(architecture, in_channels=in_channels, num_classes=num_classes).to(self.device)
        self.personalized_model = get_model(architecture, in_channels=in_channels, num_classes=num_classes).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        num_train = len(train_loader.dataset) if train_loader.dataset is not None else 0
        num_test = len(test_loader.dataset) if test_loader.dataset is not None else 0
        
        self.state = ClientState(
            client_id=client_id,
            num_train_samples=num_train,
            num_test_samples=num_test,
        )

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Loads parameters from server and accounts for inbound communication cost.
        """
        set_model_parameters(self.model, parameters)
        param_bytes = sum(p.nelement() * p.element_size() for p in parameters.values())
        self.state.record_communication(bytes_received=param_bytes)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns detached CPU parameters of current local model.
        """
        return get_model_parameters(self.model)

    def train(
        self,
        epochs: int = 1,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        optimizer_type: str = "sgd",
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
        """
        Executes local training on client dataset for specified number of epochs.
        Returns:
            - updated parameters
            - number of train samples
            - training metrics dictionary
        """
        self.model.to(self.device)
        self.model.train()
        
        opt_type = optimizer_type.lower().strip()
        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        start_time = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(target)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
                
        elapsed = time.time() - start_time
        self.state.training_time_seconds += elapsed
        self.state.total_local_epochs += epochs
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        
        updated_params = self.get_parameters()
        
        # Account for outbound model upload communication cost
        param_bytes = sum(p.nelement() * p.element_size() for p in updated_params.values())
        self.state.record_communication(bytes_sent=param_bytes)
        
        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "training_time": elapsed,
        }
        return updated_params, self.state.num_train_samples, metrics

    def evaluate(self, parameters: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Evaluates current model (or optional provided parameters) on local test data.
        """
        if parameters is not None:
            self.set_parameters(parameters)
            
        self.model.to(self.device)
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item() * len(target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
                
        avg_loss = test_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total_samples,
        }

    def train_local_only(
        self,
        epochs: int = 1,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        optimizer_type: str = "sgd",
    ) -> Dict[str, float]:
        """
        Trains model purely locally without server communication.
        Used for the independent Local Training baseline (zero communication cost).
        """
        self.model.to(self.device)
        self.model.train()
        
        opt_type = optimizer_type.lower().strip()
        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        start_time = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(target)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
                
        elapsed = time.time() - start_time
        self.state.training_time_seconds += elapsed
        self.state.total_local_epochs += epochs
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "training_time": elapsed,
        }

    def fine_tune(
        self,
        base_parameters: Optional[Dict[str, torch.Tensor]] = None,
        epochs: int = 1,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        optimizer_type: str = "sgd",
    ) -> Dict[str, float]:
        """
        Fine-tunes the personalized model on local client data.
        If base_parameters is provided (e.g. latest global model), initializes
        personalized model from base_parameters before fine-tuning.
        """
        if base_parameters is not None:
            set_model_parameters(self.personalized_model, base_parameters)
            
        self.personalized_model.to(self.device)
        self.personalized_model.train()
        
        opt_type = optimizer_type.lower().strip()
        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.personalized_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.personalized_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        start_time = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.personalized_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(target)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
                
        elapsed = time.time() - start_time
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        
        self.state.personalized_parameters = get_model_parameters(self.personalized_model)
        
        return {
            "finetune_loss": avg_loss,
            "finetune_accuracy": accuracy,
            "finetune_time": elapsed,
        }

    def evaluate_personalized(self) -> Dict[str, float]:
        """
        Evaluates the client's current personalized model on local test data.
        """
        self.personalized_model.to(self.device)
        self.personalized_model.eval()
        
        test_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.personalized_model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item() * len(target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += len(target)
                
        avg_loss = test_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "num_samples": total_samples,
        }

    def get_personalized_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns detached CPU parameters of personalized model."""
        return get_model_parameters(self.personalized_model)

    def set_personalized_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Loads parameters into personalized model."""
        set_model_parameters(self.personalized_model, parameters)
