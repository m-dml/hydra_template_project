from dataclasses import dataclass
from typing import Any


@dataclass
class Trainer:
    _target_: str = "pytorch_lightning.Trainer"
    enable_checkpointing: bool = True
    default_root_dir: Any = None
    gradient_clip_val: float = 0.0
    process_position: int = 0
    num_nodes: int = 1
    auto_select_gpus: bool = True
    enable_progress_bar: bool = True
    overfit_batches: float = 0.0
    track_grad_norm: int = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: int = 1
    max_epochs: int = 1
    min_epochs: int = 1
    max_steps: Any = -1
    min_steps: Any = None
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    sync_batchnorm: bool = False
    precision: int = 32
    weights_summary: Any = "top"
    weights_save_path: Any = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Any = None
    profiler: Any = None
    benchmark: bool = False
    deterministic: bool = False
    auto_lr_find: bool = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = True
    auto_scale_batch_size: bool = False
    plugins: Any = None
    amp_backend: str = "native"
    amp_level: Any = None
    move_metrics_to_cpu: bool = False
    accelerator: Any = None
    devices: Any = None
