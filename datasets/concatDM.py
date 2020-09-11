import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision import datasets

import pytorch_lightning as pl

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class ConcatDataModule(pl.LightningDataModule):
    def __init__(self,s_datamodule,t_datamodule):
        super().__init__()
        self.s_datamodule=s_datamodule
        self.t_datamodule=t_datamodule

    def setup(self,stage=None):
        self.s_datamodule.setup()
        self.t_datamodule.setup()
                
    def train_dataloader(self):
        s_train_loader=self.s_datamodule.train_dataloader()
        t_train_loader=self.t_datamodule.train_dataloader()
        concat_dataset=ConcatDataset(
            s_train_loader.dataset,
            t_train_loader.dataset
        )
        loader = torch.utils.data.DataLoader(
            concat_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=8
        )
        return loader

    def val_dataloader(self):
        s_val_loader=self.s_datamodule.val_dataloader()
        t_val_loader=self.t_datamodule.val_dataloader()
        return [s_val_loader,t_val_loader]

    def test_dataloader(self):
        s_test_loader=self.s_datamodule.test_dataloader()
        t_test_loader=self.t_datamodule.test_dataloader()
        return [s_test_loader,t_test_loader]
   