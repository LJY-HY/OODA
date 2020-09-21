import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision import datasets

import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,TUNING=False,Training=True,batch_size=64):
        super().__init__()
        self.mean = 0.1307
        self.std = 0.3081
        self.TUNING=TUNING
        self.Training = Training
        self.transform = transforms.Compose([transforms.Pad(padding=2,padding_mode='edge'),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.MNIST(root='./workspace/datasets/MNIST',train=True,download=True, transform=self.transform)
        datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform)
    
    def setup(self, stage=None):
        self.mnist_train = datasets.MNIST(root='./workspace/datasets/MNIST',train=True,download=True, transform=self.transform)
        self.mnist_test = datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform)
        if self.Training==False:
            tuning_set, test_set = random_split(self.mnist_test,[1000,9000])
            if self.TUNING:
                self.mnist_train = tuning_set
                self.mnist_test = test_set
            else:
                self.mnist_train = test_set
                self.mnist_test = datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return mnist_val

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=8)