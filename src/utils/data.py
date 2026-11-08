from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class ClientDataSplit:
    """
    Holds client local datasets and loaders.
    """
    client_id: int
    train_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    test_loader: DataLoader
    class_counts: Dict[int, int]
    num_train_samples: int
    num_test_samples: int


def get_dataset(name: str = "mnist", data_dir: str = "./data", download: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Loads raw train and test datasets from torchvision.
    """
    name_clean = name.lower().strip()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    if name_clean == "mnist":
        train_data = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        test_data = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
    elif name_clean == "fashion_mnist":
        transform_fashion = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_data = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform_fashion)
        test_data = datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform_fashion)
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported: 'mnist', 'fashion_mnist'")
        
    return train_data, test_data


def partition_dataset_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> List[List[int]]:
    """
    Partitions dataset indices across num_clients using Dirichlet distribution Dir(alpha).
    Lower alpha generates more severe non-IID label skew.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    
    # Iterate through each class and allocate among clients according to Dirichlet weights
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Ensure numerical safety and compute split points
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())
            
    # Shuffle each client's indices
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
        
    return client_indices


def partition_dataset_shards(
    dataset: Dataset,
    num_clients: int,
    shards_per_client: int = 2,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partitions dataset by sorting by class and dividing into shards.
    Each client receives shards_per_client shards (pathological non-IID).
    Follows McMahan et al. (2017).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    
    # Sort indices by label
    sorted_indices = np.argsort(targets)
    total_shards = num_clients * shards_per_client
    shard_size = len(sorted_indices) // total_shards
    
    # Break into shards
    shards = [sorted_indices[i * shard_size : (i + 1) * shard_size].tolist() for i in range(total_shards)]
    rng.shuffle(shards)
    
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for k in range(num_clients):
        for s in range(shards_per_client):
            client_indices[k].extend(shards[k * shards_per_client + s])
        rng.shuffle(client_indices[k])
        
    return client_indices


def partition_dataset_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partitions dataset indices uniformly at random across clients (IID).
    """
    rng = np.random.RandomState(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]


def get_client_class_distribution(
    client_splits: List[ClientDataSplit],
    num_classes: int = 10,
) -> np.ndarray:
    """
    Returns matrix of shape (num_clients, num_classes) containing sample counts per class.
    """
    num_clients = len(client_splits)
    matrix = np.zeros((num_clients, num_classes), dtype=int)
    for k, split in enumerate(client_splits):
        for c, count in split.class_counts.items():
            if c < num_classes:
                matrix[k, c] = count
    return matrix


def load_federated_dataset(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    num_clients: int = 5,
    partition_strategy: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    test_split_ratio: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[List[ClientDataSplit], DataLoader]:
    """
    Main function to download, partition, and create client data loaders
    along with a centralized global test loader.
    """
    train_data, test_data = get_dataset(name=dataset_name, data_dir=data_dir, download=True)
    
    strategy = partition_strategy.lower().strip()
    if strategy == "dirichlet":
        client_partition_indices = partition_dataset_dirichlet(
            train_data, num_clients=num_clients, alpha=dirichlet_alpha, seed=seed
        )
    elif strategy == "shard":
        client_partition_indices = partition_dataset_shards(
            train_data, num_clients=num_clients, shards_per_client=2, seed=seed
        )
    elif strategy == "iid":
        client_partition_indices = partition_dataset_iid(
            train_data, num_clients=num_clients, seed=seed
        )
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}. Options: 'dirichlet', 'shard', 'iid'")
        
    rng = np.random.RandomState(seed)
    client_splits: List[ClientDataSplit] = []
    
    train_targets = np.array(train_data.targets)
    
    for client_id, indices in enumerate(client_partition_indices):
        indices_arr = np.array(indices)
        rng.shuffle(indices_arr)
        
        # Split local client data into train and test
        num_local = len(indices_arr)
        num_test = int(num_local * test_split_ratio)
        num_train = num_local - num_test
        
        local_train_idx = indices_arr[:num_train].tolist()
        local_test_idx = indices_arr[num_train:].tolist()
        
        local_train_subset = Subset(train_data, local_train_idx)
        local_test_subset = Subset(train_data, local_test_idx)
        
        train_loader = DataLoader(local_train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(local_test_subset, batch_size=batch_size, shuffle=False)
        
        # Compute class count distribution for this client
        client_labels = train_targets[local_train_idx]
        unique_classes, counts = np.unique(client_labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(unique_classes, counts)}
        
        split = ClientDataSplit(
            client_id=client_id,
            train_dataset=local_train_subset,
            test_dataset=local_test_subset,
            train_loader=train_loader,
            test_loader=test_loader,
            class_counts=class_counts,
            num_train_samples=len(local_train_idx),
            num_test_samples=len(local_test_idx),
        )
        client_splits.append(split)
        
    # Global test loader for overall evaluation
    global_test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False)
    
    return client_splits, global_test_loader
