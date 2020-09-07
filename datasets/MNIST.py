import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,batch_size=64):
        super().__init__()
        self.mean = 0.1307
        self.std = 0.3081
        self.transform = transforms.Compose([transforms.Pad(padding=2,padding_mode='symmetric'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.MNIST(root='./workspace/datasets/MNIST',train=True,download=True, transform=self.transform)
        datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform)

    def setup(self, stage=None):
        self.mnist_test = datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform)
        self.mnist_train = datasets.MNIST(root='./workspace/datasets/MNIST',train=True,download=True, transform=self.transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return mnist_val

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=8)