from dataclasses import dataclass


@dataclass
class BaseLogger:
    save_dir: str = "./logs/"


@dataclass
class TestTubeLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.TestTubeLogger"
    save_dir: str = "./logs/test_tube"


@dataclass
class TensorBoardLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = "./logs/tensorboard"
    default_hp_metric: bool = False
    log_graph: bool = True
    version: str = "all"


@dataclass
class MLFlowLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    save_dir: str = "./logs/ml_flow"
