"""MNIST and CIFAR-10 data loading for unconditional DDPM training."""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transform(
    image_size: int,
    dataset_name: str = "cifar10",
    train: bool = True,
    augment: Optional[bool] = None,
) -> Callable:
    """Build a transform that maps images from [0, 1] to [-1, 1].

    ``train`` selects the official dataset split. ``augment`` controls data
    augmentation separately so FID can use the 5,000 training images without
    applying a new random flip every time an image is read.
    """

    name = dataset_name.lower()
    if augment is None:
        augment = train
    transform_list = [transforms.ToTensor()]
    native_size = 28 if name == "mnist" else 32
    if image_size != native_size:
        transform_list.append(transforms.Resize(image_size))
    if name == "cifar10" and augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    channels = 1 if name == "mnist" else 3
    transform_list.append(
        transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
    )
    return transforms.Compose(transform_list)


class DiffusionDataset(Dataset):
    """Drop labels because Project 1 trains an unconditional model."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.base[index]
        return image


def get_dataset(
    name: str,
    root: str = "./data",
    image_size: Optional[int] = None,
    train: bool = True,
    augment: Optional[bool] = None,
) -> Dataset:
    name = name.lower()
    if name not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset: {name}")
    if image_size is None:
        image_size = 28 if name == "mnist" else 32

    transform = get_transform(image_size, name, train=train, augment=augment)
    if name == "mnist":
        base = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    else:
        base = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    return DiffusionDataset(base)


def get_dataloader(
    name: str,
    batch_size: int,
    root: str = "./data",
    image_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    generator: Optional[torch.Generator] = None,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    dataset = get_dataset(name, root=root, image_size=image_size, train=True)
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0 and persistent_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor from [-1, 1] to [0, 1] for visualization or metrics."""

    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0
