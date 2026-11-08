import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.clients.client import FLClient
from src.evaluation.metrics import MetricsLogger
from src.federated.fedavg import FedAvgCoordinator
from src.server.server import FLServer


def test_fedavg_weighted_aggregation_math():
    """
    Verifies that FLServer.aggregate computes the exact mathematically expected
    weighted average of client parameters:
    theta = sum_k (n_k / n_total) * theta_k
    """
    server = FLServer(architecture="mlp", num_classes=2, in_channels=1)
    
    # Create two synthetic sets of parameters
    params1 = {
        "fc1.weight": torch.full((200, 784), 1.0),
        "fc1.bias": torch.full((200,), 1.0),
        "fc2.weight": torch.full((200, 200), 1.0),
        "fc2.bias": torch.full((200,), 1.0),
        "fc3.weight": torch.full((2, 200), 1.0),
        "fc3.bias": torch.full((2,), 1.0),
    }
    params2 = {
        "fc1.weight": torch.full((200, 784), 4.0),
        "fc1.bias": torch.full((200,), 4.0),
        "fc2.weight": torch.full((200, 200), 4.0),
        "fc2.bias": torch.full((200,), 4.0),
        "fc3.weight": torch.full((2, 200), 4.0),
        "fc3.bias": torch.full((2,), 4.0),
    }
    
    # n1 = 10, n2 = 20 -> expected = (10*1.0 + 20*4.0) / 30 = 90 / 30 = 3.0
    client_updates = [(params1, 10), (params2, 20)]
    aggregated = server.aggregate(client_updates)
    
    for k in aggregated.keys():
        assert torch.allclose(aggregated[k], torch.full_like(aggregated[k], 3.0)), (
            f"Layer {k} failed weighted aggregation expectation"
        )


def test_fedavg_end_to_end_synthetic():
    """
    Fast end-to-end integration test of 2 communication rounds
    using small synthetic datasets.
    """
    device = "cpu"
    # Create synthetic datasets
    x1 = torch.randn(32, 1, 28, 28)
    y1 = torch.randint(0, 10, (32,))
    x2 = torch.randn(32, 1, 28, 28)
    y2 = torch.randint(0, 10, (32,))
    
    train_loader1 = DataLoader(TensorDataset(x1, y1), batch_size=8, shuffle=True)
    test_loader1 = DataLoader(TensorDataset(x1, y1), batch_size=8, shuffle=False)
    
    train_loader2 = DataLoader(TensorDataset(x2, y2), batch_size=8, shuffle=True)
    test_loader2 = DataLoader(TensorDataset(x2, y2), batch_size=8, shuffle=False)
    
    global_test_loader = DataLoader(
        TensorDataset(torch.cat([x1, x2]), torch.cat([y1, y2])),
        batch_size=16,
        shuffle=False,
    )
    
    server = FLServer(architecture="cnn", num_classes=10, in_channels=1, device=device)
    c1 = FLClient(0, train_loader1, test_loader1, architecture="cnn", num_classes=10, device=device)
    c2 = FLClient(1, train_loader2, test_loader2, architecture="cnn", num_classes=10, device=device)
    
    logger = MetricsLogger(output_dir="./test_results", experiment_name="test_fedavg")
    
    coordinator = FedAvgCoordinator(
        server=server,
        clients=[c1, c2],
        global_test_loader=global_test_loader,
        num_rounds=2,
        client_fraction=1.0,
        local_epochs=1,
        learning_rate=0.01,
        metrics_logger=logger,
        seed=42,
    )
    
    df = coordinator.run()
    assert len(df) == 2, f"Expected 2 rounds in results DataFrame, got {len(df)}"
    assert "global_accuracy" in df.columns
    assert "round_comm_bytes" in df.columns
    assert df["cumulative_comm_bytes"].iloc[-1] > df["cumulative_comm_bytes"].iloc[0]


if __name__ == "__main__":
    pytest.main([__file__])
