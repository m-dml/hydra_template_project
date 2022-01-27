import unittest
import logging
import hydra
import numpy as np
import pytorch_lightning as pl
import torch

from src.lib.config import register_configs
from src.utils import utils
import os

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
            overrides=[
                "datamodule.batch_size=1",
                "datamodule.shuffle_train_dataset=false",
                "strategy=SingleDevice",
                "datamodule/train_transforms=no_transforms",
            ],
        )

        cls.test_log_file = "test.log"

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.test_log_file):
            os.remove(cls.test_log_file)

    def test_utils_logging(self):
        utils.set_log_levels("debug")
        logger = utils.get_logger()

        fh = logging.FileHandler(self.test_log_file)
        logger.addHandler(fh)

        logger.error("Test Error")
        logger.debug("Test Debug")

        with open(self.test_log_file, "r") as f:
            content = f.readlines()

        self.assertEqual(len(content), 2)

        utils.set_log_levels("info")
        logger = utils.get_logger()

        fh = logging.FileHandler(self.test_log_file)
        logger.addHandler(fh)

        logger.error("Test Error")
        logger.debug("Test Debug")

        with open(self.test_log_file, "r") as f:
            content = f.readlines()

        self.assertEqual(len(content), 1)

