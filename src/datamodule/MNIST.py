import logging

import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset


class MNISTDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        data_base_path: str,
        batch_size: int,
        num_workers: int,
        train_transforms,
        valid_transforms,
        shuffle_train_dataset: bool = True,
        shuffle_valid_dataset: bool = False,
        shuffle_test_dataset: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ):
        super(MNISTDataLoader, self).__init__()

        self.cfg_dataset = dataset
        self.data_path = data_base_path

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = train_transforms
        self.val_transforms = valid_transforms
        self.shuffle_train_dataset = shuffle_train_dataset
        self.shuffle_valid_dataset = shuffle_valid_dataset
        self.shuffle_test_dataset = shuffle_test_dataset
        self.pin_memory = pin_memory

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_data: Dataset = instantiate(self.cfg_dataset, transform=self.train_transforms, train=True)
            logging.debug(f"Number of training samples: {len(self.train_data)}")

            self.valid_data: Dataset = instantiate(self.cfg_dataset, transform=self.val_transforms, train=False)
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == "test" or stage is None:
            self.test_data: Dataset = instantiate(self.cfg_dataset, transform=self.val_transforms, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train_dataset,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_valid_dataset,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_test_dataset,
            pin_memory=self.pin_memory,
        )
