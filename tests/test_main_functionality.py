import logging
import os
import time
import unittest

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.global_hydra import GlobalHydra

from src.lib.config import register_configs
from src.utils import utils


class TestMainFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # set random seeds:
        pl.seed_everything(7)
        np.random.seed(7)
        torch.manual_seed(7)

        # set up hydra
        register_configs()
        hydra.initialize(config_path="../conf", job_name="unittest_main")

        # set up exact config
        cls.cfg = hydra.compose(
            config_name="config",
            overrides=["+experiment=unit_test"],
        )

        cls.test_log_file = "test.log"
        if os.path.isfile(cls.test_log_file):
            time.sleep(2)
            os.remove(cls.test_log_file)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.test_log_file):
            os.remove(cls.test_log_file)

        GlobalHydra.instance().clear()

    def _get_example_array(self):
        train_transforms = hydra.utils.instantiate(self.cfg.datamodule.train_transforms)
        self.assertIsNotNone(train_transforms)
        valid_transforms = hydra.utils.instantiate(self.cfg.datamodule.valid_transforms)
        self.assertIsNotNone(valid_transforms)

        datamodule = hydra.utils.instantiate(
            self.cfg.datamodule,
            train_transforms=train_transforms,
            valid_transforms=valid_transforms,
            dataset=self.cfg.datamodule.dataset,
            is_ddp=False,
        )
        datamodule.setup()  # manually set up the datamodule here, so an example batch can be drawn

        dataloader = datamodule.train_dataloader()
        for batch in dataloader:
            example_input, example_label = batch
            break

        self.assertIsNotNone(example_label)
        self.assertIsNotNone(example_input)
        self.assertIsInstance(example_input, torch.Tensor)
        return example_input, example_label

    def test_utils_logging(self):
        def _test_log(level="debug"):
            utils.set_log_levels(level)
            logger = utils.get_logger()

            fh = logging.FileHandler(self.test_log_file)
            logger.addHandler(fh)

            logger.error("Test Error")
            logger.warning("Test Warning")
            logger.info("Test Info")
            logger.debug("Test Debug")

            with open(self.test_log_file, "r") as f:
                content = f.readlines()

            print("content ", content)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
            os.remove(self.test_log_file)
            return content

        self.assertEqual(len(_test_log(level="debug")), 4)
        self.assertEqual(len(_test_log(level="info")), 3)
        self.assertEqual(len(_test_log(level="warning")), 2)
        self.assertEqual(len(_test_log(level="error")), 1)

    def test_default_datamodule(self):
        example_input, example_label = self._get_example_array()
        self.assertEqual(example_input.shape[0], self.cfg.datamodule.batch_size)

    def test_lightning_module_init(self):
        example_input, _ = self._get_example_array()
        model = hydra.utils.instantiate(
            self.cfg.lightning_module,
            optimizer=self.cfg.optimizer,
            scheduler=self.cfg.scheduler,
            model=self.cfg.model,
            loss=self.cfg.loss,
            example_input_array=example_input.detach().cpu(),
            batch_size=self.cfg.datamodule.batch_size,
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(model.hparams)

    def test_init_local_trainer(self):
        trainer = hydra.utils.instantiate(
            self.cfg.trainer, strategy=None, logger=None, callbacks=None, _convert_="partial"
        )

        self.assertIsNotNone(trainer)
