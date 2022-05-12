import copy
import faulthandler
import logging
import os
import platform
import sys
from typing import List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torchvision.transforms import Compose

from src.lib.config import Config, register_configs
from src.utils import utils

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

faulthandler.enable()
# sometimes windows and matplotlib don't play well together. Therefore we have to configure win for plt:
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# register the structured configs:
register_configs()

# set up advanced logging:


@hydra.main(config_name="config", config_path="conf")
def main(cfg: Config):
    # for integration tests main is splitted, so that there exists a not decorated version.
    return inner_main(cfg)


def inner_main(cfg: Config):
    utils.extras(cfg)  # check if debug is activated and if so, change some trainer settings
    utils.set_log_levels(cfg.log_level)
    log = utils.get_logger(cfg.log_level)

    # Pretty print config using Rich library
    if cfg.print_config:
        utils.print_config(cfg, resolve=True)  # prints the complete hydra config to std-out

    torch.manual_seed(cfg.random_seed)  # set random seed
    pl.seed_everything(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Transformations
    train_transforms: Compose = hydra.utils.instantiate(cfg.datamodule.train_transforms)
    valid_transforms: Compose = hydra.utils.instantiate(cfg.datamodule.valid_transforms)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
        train_transforms=train_transforms,
        valid_transforms=valid_transforms,
        dataset=cfg.datamodule.dataset,
        is_ddp=cfg.strategy is not None,
    )
    datamodule.setup()  # manually set up the datamodule here, so an example batch can be drawn

    # generate example input array:
    for batch in datamodule.train_dataloader():
        example_input, _ = batch
        break

    log.info(f"Size of one batch is: {example_input.element_size() * example_input.nelement() / 2**20} mb")

    # Init Lightning model
    log.info(f"Instantiating model <{cfg.lightning_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        model=cfg.model,
        loss=cfg.loss,
        example_input_array=example_input.detach().cpu(),
        batch_size=cfg.datamodule.batch_size,
    )

    # load the state dict if one is provided (has to be provided for finetuning classifier in simclr):
    device = "cuda" if cfg.trainer.accelerator == "gpu" else "cpu"
    if cfg.load_state_dict is not None:
        log.info(f"Loading model weights from {cfg.load_state_dict}")
        net = copy.deepcopy(model.model.cpu())
        # check state dict before loading:
        this_state_dict = model.model.state_dict().copy()
        len_old_state_dict = len(this_state_dict)
        log.info(f"Old state dict has {len_old_state_dict} entries.")
        try:
            new_state_dict = torch.load(cfg.load_state_dict, map_location=torch.device(device))
        except Exception as e:
            log.error()
            raise e

        missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=True)
        log.warning(f"Missing keys: {missing_keys}")
        log.warning(f"Unexpected keys: {unexpected_keys}")
        state_dict_error_count = 0
        for state_key, state in net.state_dict().items():
            if this_state_dict[state_key].allclose(state, atol=1e-12, rtol=1e-12):
                log.error(
                    f"Loaded state dict params for layer '{state_key}' are same as random initialized one ("
                    f"Might be due to caching, if you just restarted the same model twice!)"
                )
                state_dict_error_count += 1
        if state_dict_error_count > 0:
            log.warning(
                f"{state_dict_error_count} state entries are the same after init. "
                f"(From a total of {len_old_state_dict} items)"
            )
        model.model = copy.deepcopy(net.model.to(device))
        del net
        log.info(f"Successfully loaded model weights from {cfg.load_state_dict}")

    # log hparam metrics to tensorboard:
    log.info("Logging hparams to tensorboard")
    hydra_params = utils.log_hyperparameters(config=cfg, model=model)
    for this_logger in logger:
        if "tensorboard" in str(this_logger):
            log.info("Add hparams to tensorboard")
            this_logger.log_hyperparams(hydra_params, {"hp/loss": 0, "hp/epoch": 0})
        else:
            this_logger.log_hyperparams(hydra_params)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to lightning!")
    model.hydra_params = hydra_params

    # Init Trainer:
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(
        cfg.trainer, strategy=cfg.strategy, logger=logger, callbacks=callbacks, _convert_="partial"
    )

    # if activated in the config, start the pytorch lightning automatic batch-size and lr tuning process
    if cfg.auto_tune:
        log.info("Starting tuning the model")
        trainer.tune(model, datamodule)

    log.info("Starting training")
    trainer.fit(model, datamodule)  # the actual training of the NN

    # Print path to best checkpoint
    if trainer.checkpoint_callback.best_model_path is not None:
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # if at some point optuna is used, then some metric has to be returned, which optuna can optimize
    return trainer.callback_metrics["hp/loss"].item()


if __name__ == "__main__":
    log = utils.get_logger()
    log.info("Starting python script")

    main()
