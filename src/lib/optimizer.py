from dataclasses import dataclass


@dataclass
class Optimizer:
    lr: float = 0.02


@dataclass
class Adam(Optimizer):
    _target_: str = "torch.optim.Adam"


@dataclass
class SGD(Optimizer):
    _target_: str = "torch.optim.SGD"
    nesterov: bool = False
    momentum: float = 0.9
    weight_decay: float = 1e-6
    lr: float = 0.3


@dataclass
class RMSprop(Optimizer):
    _target_: str = "torch.optim.RMSprop"
    momentum: float = 0


@dataclass
class LARS(Optimizer):
    _target_: str = "pl_bolts.optimizers.lars.LARS"
    lr: float = 4.8
    momentum: float = 0.9
    trust_coefficient: float = 0.001
    weight_decay: float = 1e-6
