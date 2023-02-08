import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from .. import builder

from .. import enums


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = builder.build_dataset(cfg)

    def train_dataloader(self):
        if self.cfg.stage == "unimodal" and self.cfg.decoder.modality == enums.Modality.Language:
            split = "restval"
        else:
            split = "train"
        dataset = self.dataset(self.cfg, split=split)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        dataset = self.dataset(self.cfg, split="val")
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        dataset = self.dataset(self.cfg, split=self.cfg.test_split)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=1,
            num_workers=self.cfg.train.num_workers,
        )

    def all_dataloader(self):
        dataset = self.dataset(self.cfg, split=self.cfg.test_split)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
        )