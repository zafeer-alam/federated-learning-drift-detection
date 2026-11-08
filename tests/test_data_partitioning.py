import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from src.utils.data import (
    partition_dataset_dirichlet,
    partition_dataset_shards,
    partition_dataset_iid,
)


class DummyClassificationDataset(Dataset):
    """Synthetic dataset for fast, offline unit testing of partition logic."""
    def __init__(self, num_samples: int = 1000, num_classes: int = 10):
        self.data = torch.randn(num_samples, 1, 28, 28)
        self.targets = torch.randint(0, num_classes, (num_samples,)).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_dirichlet_partitioning_disjoint_and_complete():
    dataset = DummyClassificationDataset(num_samples=500, num_classes=5)
    num_clients = 4
    alpha = 0.5
    seed = 123

    client_indices = partition_dataset_dirichlet(dataset, num_clients=num_clients, alpha=alpha, seed=seed)

    assert len(client_indices) == num_clients

    # Check completeness
    all_assigned = []
    for indices in client_indices:
        all_assigned.extend(indices)

    assert len(all_assigned) == len(dataset), "Not all samples were allocated"
    assert len(set(all_assigned)) == len(dataset), "Sample allocations overlap across clients"


def test_dirichlet_partitioning_determinism():
    dataset = DummyClassificationDataset(num_samples=300, num_classes=5)
    splits1 = partition_dataset_dirichlet(dataset, num_clients=3, alpha=0.5, seed=42)
    splits2 = partition_dataset_dirichlet(dataset, num_clients=3, alpha=0.5, seed=42)
    splits3 = partition_dataset_dirichlet(dataset, num_clients=3, alpha=0.5, seed=99)

    assert splits1 == splits2, "Same seed must yield identical partitions"
    assert splits1 != splits3, "Different seeds should yield different partitions"


def test_shard_partitioning():
    dataset = DummyClassificationDataset(num_samples=400, num_classes=4)
    num_clients = 4
    shards_per_client = 2

    client_indices = partition_dataset_shards(
        dataset, num_clients=num_clients, shards_per_client=shards_per_client, seed=42
    )

    assert len(client_indices) == num_clients
    total_assigned = sum(len(c) for c in client_indices)
    assert total_assigned == len(dataset)


def test_iid_partitioning():
    dataset = DummyClassificationDataset(num_samples=300, num_classes=3)
    num_clients = 3
    client_indices = partition_dataset_iid(dataset, num_clients=num_clients, seed=42)

    assert len(client_indices) == num_clients
    total_assigned = sum(len(c) for c in client_indices)
    assert total_assigned == len(dataset)
    for c in client_indices:
        assert len(c) == 100


if __name__ == "__main__":
    pytest.main([__file__])
