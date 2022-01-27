from dataclasses import dataclass


@dataclass
class BaseLogger:
    save_dir: str = "./outputs/logger"


@dataclass
class TestTubeLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.TestTubeLogger"
    save_dir: str = "./outputs/test_tube"


@dataclass
class TensorBoardLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    save_dir: str = "./outputs/tensorboard"
    default_hp_metric: bool = False
    log_graph: bool = True
    version: str = "all"


@dataclass
class MLFlowLogger(BaseLogger):
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    save_dir: str = "./outputs/ml_flow"
