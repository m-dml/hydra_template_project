import logging
import os
import time
import unittest

import hydra
import numpy as np
import pytorch_lightning as pl
import torch

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
            overrides=[
                "datamodule.batch_size=1",
                "datamodule.shuffle_train_dataset=false",
                "strategy=SingleDevice",
                "datamodule/train_transforms=no_transforms",
            ],
        )

        cls.test_log_file = "test.log"
        if os.path.isfile(cls.test_log_file):
            time.sleep(2)
            os.remove(cls.test_log_file)

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.test_log_file):
            time.sleep(2)
            os.remove(cls.test_log_file)

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
