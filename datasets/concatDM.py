import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision import datasets

import pytorch_lightning as pl

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self,*datasets):
        self.datasets=datasets

    def __getitem__(self,i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class ConcatDataModule(pl.LightningDataModule):
    def __init__(self,concat_train_dataset,concat_test_dataset):
        super().__init__()
        self.concat_train=concat_train_dataset
        self.concat_test=concat_test_dataset
        
    def train_dataloader(self):
        return DataLoader(self.concat_train, batch_size=64, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.concat_test, batch_size=64, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.concat_test, batch_size=64, shuffle=False, num_workers=8)
   