import pytest
import torch
import torch.nn as nn
from src.models.model import (
    MNIST_CNN,
    MNIST_MLP,
    compute_model_size_bytes,
    count_parameters,
    get_model,
    get_model_parameters,
    set_model_parameters,
)


def test_mnist_cnn_forward():
    model = MNIST_CNN(in_channels=1, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10), f"Expected shape (4, 10), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_mnist_mlp_forward():
    model = MNIST_MLP(in_features=784, hidden_dim=200, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10), f"Expected shape (4, 10), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_get_model_factory():
    cnn = get_model("cnn")
    assert isinstance(cnn, MNIST_CNN)

    mlp = get_model("mlp")
    assert isinstance(mlp, MNIST_MLP)

    with pytest.raises(ValueError):
        get_model("unknown_model_arch")


def test_parameter_extraction_and_setting():
    model1 = MNIST_CNN()
    model2 = MNIST_CNN()

    # Verify initial parameters differ (due to random initialization)
    p1 = get_model_parameters(model1)
    p2 = get_model_parameters(model2)

    differs = False
    for k in p1.keys():
        if not torch.allclose(p1[k], p2[k]):
            differs = True
            break
    assert differs, "Randomly initialized models should not have identical weights"

    # Set model2 weights to model1 weights
    set_model_parameters(model2, p1)
    p2_updated = get_model_parameters(model2)

    for k in p1.keys():
        assert torch.allclose(p1[k], p2_updated[k]), f"Mismatch in layer {k} after set_model_parameters"


def test_parameter_counting_and_sizing():
    model = MNIST_CNN()
    num_params = count_parameters(model)
    assert num_params > 0, "Parameter count must be positive"

    size_bytes = compute_model_size_bytes(model)
    # 4 bytes per float32 parameter
    assert size_bytes >= num_params * 4, "Byte size should account for parameters"


if __name__ == "__main__":
    pytest.main([__file__])
