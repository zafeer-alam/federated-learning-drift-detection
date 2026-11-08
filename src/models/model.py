import copy
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    Standard 2-layer Convolutional Neural Network for MNIST/Fashion-MNIST,
    as specified in McMahan et al. (2017) 'Communication-Efficient Learning
    of Deep Networks from Decentralized Data'.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNIST_MLP(nn.Module):
    """
    Standard 2-hidden layer Multi-Layer Perceptron (2NN) for MNIST,
    following McMahan et al. (2017).
    """
    def __init__(self, in_features: int = 28 * 28, hidden_dim: int = 200, num_classes: int = 10):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(architecture: str = "cnn", in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """
    Factory function returning a model instance based on architecture string.
    """
    arch = architecture.lower().strip()
    if arch == "cnn":
        return MNIST_CNN(in_channels=in_channels, num_classes=num_classes)
    elif arch == "mlp" or arch == "2nn":
        return MNIST_MLP(in_features=28 * 28 * in_channels, hidden_dim=200, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: 'cnn', 'mlp'")


def get_model_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Returns a deep copy of model's state_dict with tensors detached on CPU.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_model_parameters(model: nn.Module, parameters: Dict[str, torch.Tensor]) -> None:
    """
    Loads parameter dictionary into model with strict key checking.
    """
    model.load_state_dict(parameters, strict=True)


def count_parameters(model: nn.Module) -> int:
    """
    Returns total number of trainable parameters in model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_size_bytes(model: nn.Module) -> int:
    """
    Computes total memory size in bytes of model parameters.
    Used for rigorous communication cost accounting in FL.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes
