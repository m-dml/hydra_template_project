from dataclasses import dataclass


@dataclass
class NLLLoss:
    _target_: str = "torch.nn.NLLLoss"
