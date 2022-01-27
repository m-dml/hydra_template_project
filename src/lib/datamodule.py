from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class ParentDataloader:
    pin_memory: bool = False
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class MNISTDataset:
    _target_: str = "torchvision.datasets.MNIST"
    download: bool = True
    root: str = MISSING  # path where to save downloaded data


@dataclass
class MNISTDataLoader(ParentDataloader):
    _target_: str = "src.datamodule.MNIST.MNISTDataLoader"
    _recursive_: bool = False

    dataset: Any = MISSING
    train_transforms: Any = MISSING
    valid_transforms: Any = MISSING
    shuffle_train_dataset: bool = True
    shuffle_validation_dataset: bool = False
    shuffle_test_dataset: bool = False
    data_base_path: str = "data/"
