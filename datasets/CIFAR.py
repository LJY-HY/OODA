import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,TUNING=False,Training=True,batch_size=64):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.TUNING=TUNING
        self.Training = Training
        self.transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(self.mean, self.std)])
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.CIFAR10(root='./workspace/datasets/cifar10',train=True,download=True, transform=self.transform)
        datasets.CIFAR10(root='./workspace/datasets/cifar10',train=False,download=True, transform=self.transform_test)

    def setup(self, stage=None):
        self.cifar_train = datasets.CIFAR10(root='./workspace/datasets/cifar10',train=True,download=True, transform=self.transform)
        self.cifar_test = datasets.CIFAR10(root='./workspace/datasets/cifar10',train=False,download=True, transform=self.transform_test)
        if self.Training==False:
            tuning_set, test_set = random_split(self.cifar_test,[1000,9000])
            if self.TUNING:
                self.cifar_train = tuning_set
                self.cifar_test = test_set
            else:
                self.cifar_train = test_set
                self.cifar_test = datasets.CIFAR10(root='./workspace/datasets/cifar10',train=False,download=True, transform=self.transform_test)

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8)



class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self,TUNING=False,Training=True,batch_size=64):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.TUNING=TUNING
        self.Training=Training
        self.transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(self.mean, self.std)])
        self.batch_size=batch_size
    def prepare_data(self):
        datasets.CIFAR100(root='./workspace/datasets/cifar100',train=True,download=True, transform=self.transform)
        datasets.CIFAR100(root='./workspace/datasets/cifar100',train=False,download=True, transform=self.transform_test)

    def setup(self, stage=None):
        self.cifar_train = datasets.CIFAR100(root='./workspace/datasets/cifar100',train=True,download=True, transform=self.transform)
        self.cifar_test = datasets.CIFAR100(root='./workspace/datasets/cifar100',train=False,download=True, transform=self.transform_test)
        if self.Training==False:
            tuning_set, test_set = random_split(self.cifar_test,[1000,9000])
            if self.TUNING:
                self.cifar_train = tuning_set
                self.cifar_test = test_set
            else:
                self.cifar_train = test_set
                self.cifar_test = datasets.CIFAR100(root='./workspace/datasets/cifar100',train=False,download=True, transform=self.transform_test)

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return cifar_val

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
