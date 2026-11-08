import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.clients.client import FLClient
from src.evaluation.metrics import MetricsLogger
from src.federated.personalized import PersonalizedFLCoordinator
from src.server.server import FLServer


def test_client_fine_tuning():
    """Verifies that fine_tune adapts personalized model parameters."""
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    
    client = FLClient(client_id=0, train_loader=loader, test_loader=loader, architecture="mlp", num_classes=10)
    
    init_params = client.get_personalized_parameters()
    metrics = client.fine_tune(epochs=1, lr=0.05)
    updated_params = client.get_personalized_parameters()
    
    assert "finetune_loss" in metrics
    assert "finetune_accuracy" in metrics
    
    # Check that weights were updated
    changed = False
    for k in init_params.keys():
        if not torch.allclose(init_params[k], updated_params[k]):
            changed = True
            break
    assert changed, "Personalized parameters should change after fine_tune"
    
    eval_res = client.evaluate_personalized()
    assert "accuracy" in eval_res
    assert "loss" in eval_res


def test_local_only_training_zero_communication():
    """Verifies local-only training logs zero communication bytes."""
    x1 = torch.randn(16, 1, 28, 28)
    y1 = torch.randint(0, 5, (16,))
    loader1 = DataLoader(TensorDataset(x1, y1), batch_size=8)
    
    c1 = FLClient(0, loader1, loader1, architecture="mlp", num_classes=5)
    c2 = FLClient(1, loader1, loader1, architecture="mlp", num_classes=5)
    
    logger = MetricsLogger(output_dir="./test_results", experiment_name="test_local_only")
    coordinator = PersonalizedFLCoordinator(
        server=None,
        clients=[c1, c2],
        global_test_loader=loader1,
        mode="local_only",
        num_rounds=2,
        local_epochs=1,
        metrics_logger=logger,
    )
    
    df = coordinator.run()
    assert len(df) == 2
    assert (df["round_comm_bytes"] == 0).all(), "Local only training must have zero communication"
    assert (df["cumulative_comm_bytes"] == 0).all()


def test_personalized_finetune_end_to_end():
    """Verifies end-to-end execution of personalized fine-tuning coordinator."""
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 5, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    
    server = FLServer(architecture="mlp", num_classes=5)
    c1 = FLClient(0, loader, loader, architecture="mlp", num_classes=5)
    c2 = FLClient(1, loader, loader, architecture="mlp", num_classes=5)
    
    logger = MetricsLogger(output_dir="./test_results", experiment_name="test_personalized_fl")
    coordinator = PersonalizedFLCoordinator(
        server=server,
        clients=[c1, c2],
        global_test_loader=loader,
        mode="personalized_finetune",
        num_rounds=2,
        local_epochs=1,
        finetune_epochs=1,
        metrics_logger=logger,
    )
    
    df = coordinator.run()
    assert len(df) == 2
    assert "mean_client_test_accuracy" in df.columns
    assert df["cumulative_comm_bytes"].iloc[-1] > 0


if __name__ == "__main__":
    pytest.main([__file__])
