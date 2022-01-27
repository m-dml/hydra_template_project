from dataclasses import dataclass
from typing import Any


@dataclass
class SquarePad:
    _target_: str = "src.utils.SquarePadTransform.SquarePad"


@dataclass
class Resize:
    _target_: str = "torchvision.transforms.Resize"
    size: Any = (520, 520)


@dataclass
class RandomRotation:
    _target_: str = "torchvision.transforms.RandomRotation"
    degrees: float = 360.0


@dataclass
class RandomHorizontalFlip:
    _target_: str = "torchvision.transforms.RandomHorizontalFlip"


@dataclass
class RandomVerticalFlip:
    _target_: str = "torchvision.transforms.RandomVerticalFlip"


@dataclass
class ColorJitter:
    _target_: str = "torchvision.transforms.ColorJitter"
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    hue: float = 0.1


@dataclass
class ToTensor:
    _target_: str = "torchvision.transforms.ToTensor"


@dataclass
class Compose:
    _target_: str = "torchvision.transforms.Compose"
