"""
Neural Network Architectures and Parameter Utilities.
"""
from src.models.model import (
    MNIST_CNN,
    MNIST_MLP,
    get_model,
    get_model_parameters,
    set_model_parameters,
    count_parameters,
    compute_model_size_bytes,
)

__all__ = [
    "MNIST_CNN",
    "MNIST_MLP",
    "get_model",
    "get_model_parameters",
    "set_model_parameters",
    "count_parameters",
    "compute_model_size_bytes",
]
