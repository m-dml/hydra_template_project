import os

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

from src.utils import utils


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        example_input_array,
        optimizer,
        scheduler,
        model,
        loss,
        batch_size,
    ):

        super().__init__()

        self.lr = optimizer.lr
        self.cfg_optimizer = optimizer
        self.cfg_loss = loss
        self.cfg_scheduler = scheduler

        self.save_hyperparameters(ignore=["example_input_array", "all_labels"])

        self.example_input_array = example_input_array
        self.loss_func = hydra.utils.instantiate(self.cfg_loss)

        self.model: nn.Module = hydra.utils.instantiate(model)

        self.console_logger = utils.get_logger("LightningBaseModel")
        self.console_logger.debug("Test Debug Message")
        self.batch_size = batch_size

    def forward(self, inputs, *args, **kwargs):
        return self.model(inputs)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=self.model.parameters(), lr=self.lr)
        if self.cfg_scheduler:
            self.console_logger.info("global batch size is {}".format(self.batch_size))
            train_iters_per_epoch = self.trainer.datamodule.len_train_data
            self.console_logger.info(f"train iterations per epoch are {train_iters_per_epoch}")
            warmup_steps = int(train_iters_per_epoch * 10)
            total_steps = int(train_iters_per_epoch * self.trainer.max_epochs)
            self.console_logger.info(f"Total train steps are {total_steps}")

            if self.cfg_scheduler == "linear_warmup_decay":
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(
                        optimizer, linear_warmup_decay(warmup_steps, total_steps, cosine=True)
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "name": "Lars-LR",
                }
            elif self.cfg_scheduler == "cosine":
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0.0),
                    "interval": "step",
                    "frequency": 1,
                    "name": "Cosine LR",
                }
            else:
                raise NotImplementedError(
                    f"The scheduler {self.cfg_scheduler} is not implemented. Please use one of "
                    f"[linear_warmup_decay, cosine] or none."
                )
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self.do_step(batch)

    def training_step_end(self, training_step_outputs, *args, **kwargs):
        predictions, loss = training_step_outputs
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return self.do_step(batch)

    def validation_step_end(self, validation_step_outputs, *args, **kwargs):
        predictions, loss = validation_step_outputs

        # log to tensorboard hp category:
        self.log("hp/loss", loss)
        self.log("hp/epoch", torch.tensor(self.current_epoch).float())

        return {
            "loss": loss,
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        return self.do_step(batch)

    def test_step_end(self, test_step_outputs, *args, **kwargs):
        predictions, loss = test_step_outputs
        return {"loss": loss}

    def do_step(self, batch):
        inputs, labels = batch
        predictions = self(inputs)
        loss = self.loss_func(predictions, labels)
        return predictions, loss

    def validation_epoch_end(self, outputs):
        #  here all plotting of confusion matrices etc. should be done
        pass

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.automatic_optimization and (self.current_epoch == 0):
            return

        # save model to onnx:
        folder = self.trainer.checkpoint_callback.dirpath
        onnx_file_generator = os.path.join(folder, f"model_{self.global_step}.onnx")

        torch.onnx.export(
            model=self.model.to(self.device),
            args=self.example_input_array.to(self.device),
            f=onnx_file_generator,
            opset_version=13,
            do_constant_folding=True,
            verbose=False,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # makes the batch-size variable for inference
                "output": {0: "batch_size"},
            },
        )

        # save the feature_extractor_weights:
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(folder, f"complete_model_{self.global_step}.weights"))
