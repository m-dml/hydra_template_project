import os
import shutil
import unittest

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.global_hydra import GlobalHydra

from main import inner_main
from src.lib.config import register_configs


class TestIntegration(unittest.TestCase):
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
            overrides=["+experiment=unit_test", "print_config=false"],
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.cfg.output_dir_base_path):
            shutil.rmtree(cls.cfg.output_dir_base_path)

        GlobalHydra.instance().clear()

    def test_integration(self):
        loss = inner_main(cfg=self.cfg)

        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
