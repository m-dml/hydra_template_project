from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class ResNet:
    _target_: str = "torchvision.models.resnet18"
    num_classes: int = 1000  # has to be 1000 for pretrained model
    pretrained: bool = False


@dataclass
class CustomResnet:
    _target_: str = "src.models.BaseModels.CustomResnet"
    kernel_size: int = 7
    stride: int = 2
    channels: int = 3
    model: Any = None
    maxpool1: bool = True


@dataclass
class FullyConnected:
    _target_: str = "src.models.BaseModels.Classifier"
    hidden_layers: list = (1000, 1000)
    activation: Any = MISSING
    input_features: int = 1000
    num_classes: Any = None
    normalize: bool = False
    bias_in_last_layer: bool = True
