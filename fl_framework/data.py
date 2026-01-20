"""Dataset loading and non-IID partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    data_dir: str = "~/autodl-tmp"
    batch_size: int = 64
    num_workers: int = 2
    num_clients: int = 10
    non_iid_alpha: float = 0.5
    celeba_attr: str = "Smiling"


def _get_label_targets(dataset, celeba_attr: str) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        labels = torch.tensor(dataset.targets)
        return labels
    if hasattr(dataset, "labels"):
        labels = torch.tensor(dataset.labels)
        return labels
    if hasattr(dataset, "attr"):
        attr_names = dataset.attr_names
        if dataset.attr.ndim == 1:
            labels = dataset.attr
            if labels.min() < 0:
                labels = (labels + 1) // 2
            return labels
        if dataset.attr.ndim == 2:
            if celeba_attr not in attr_names:
                raise ValueError(f"CelebA attribute {celeba_attr} not found in dataset")
            labels = dataset.attr[:, attr_names.index(celeba_attr)]
            if labels.min() < 0:
                labels = (labels + 1) // 2
            return labels
    raise ValueError("Dataset does not expose labels for non-IID partitioning")


def _dirichlet_partition(labels: torch.Tensor, num_clients: int, alpha: float) -> List[List[int]]:
    num_classes = int(labels.max().item() + 1)
    class_indices = [torch.where(labels == cls)[0].tolist() for cls in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.default_rng()
    for cls, indices in enumerate(class_indices):
        rng.shuffle(indices)
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        splits = (np.cumsum(proportions) * len(indices)).astype(int)
        split_indices = np.split(indices, splits[:-1])
        for client_id, split in enumerate(split_indices):
            client_indices[client_id].extend(split.tolist())
    return client_indices


def build_dataset(config: DatasetConfig) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if config.name == "cifar10":
        train = datasets.CIFAR10(config.data_dir, train=True, download=False, transform=transform)
        test = datasets.CIFAR10(config.data_dir, train=False, download=False, transform=transform)
    elif config.name == "cifar100":
        train = datasets.CIFAR100(config.data_dir, train=True, download=False, transform=transform)
        test = datasets.CIFAR100(config.data_dir, train=False, download=False, transform=transform)
    elif config.name == "mnist":
        train = datasets.MNIST(config.data_dir, train=True, download=False, transform=transform)
        test = datasets.MNIST(config.data_dir, train=False, download=False, transform=transform)
    elif config.name == "celeba":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train = datasets.CelebA(config.data_dir, split="train", download=True, transform=transform, target_type="attr")
        test = datasets.CelebA(config.data_dir, split="test", download=True, transform=transform, target_type="attr")
    else:
        raise ValueError(f"Unsupported dataset: {config.name}")
    return train, test


def build_dataloaders(config: DatasetConfig) -> Tuple[Dict[int, DataLoader], DataLoader]:
    train_dataset, test_dataset = build_dataset(config)
    labels = _get_label_targets(train_dataset, config.celeba_attr)
    client_indices = _dirichlet_partition(labels, config.num_clients, config.non_iid_alpha)
    client_loaders: Dict[int, DataLoader] = {}
    for client_id, indices in enumerate(client_indices):
        subset = Subset(train_dataset, indices)
        # Use drop_last=True to avoid batch size 1 issues with BatchNorm
        client_loaders[client_id] = DataLoader(
            subset,
            batch_size=min(config.batch_size, len(indices)),  # Adjust batch size if client has few samples
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=len(indices) > 1,  # Only drop last if we have more than 1 sample
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return client_loaders, test_loader
