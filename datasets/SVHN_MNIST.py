import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision import datasets

import pytorch_lightning as pl
'''
This Datamodule is used when multiple dataset is treated as in-distribution.
'''
class SVHNMNISTDataModule(pl.LightningDataModule):
    def __init__(self,TUNING=False,Training=True,batch_size=64):
        super().__init__()
        self.mean_svhn = [129.3/255, 124.1/255, 112.4/255]
        self.std_svhn = [68.2/255, 65.4/255.0, 70.4/255.0]
        self.mean_mnist = 0.1307
        self.std_mnist = 0.3081
        self.TUNING=TUNING
        self.Training = Training
        self.transform_test_svhn = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(self.mean_svhn, self.std_svhn)])
        self.transform_test_mnist = transforms.Compose([transforms.Pad(padding=2,padding_mode='edge'),
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean_mnist, self.std_mnist)])
        self.batch_size=batch_size

    def prepare_data(self):
        datasets.SVHN(root='./workspace/datasets/SVHN', split ='test',transform=self.transform_test_svhn, download=True)
        datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform_test_mnist)

    def setup(self, stage=None):
        self.SVHN_test = datasets.SVHN(root='./workspace/datasets/SVHN', split='test',transform=self.transform_test_svhn, download=True)
        self.MNIST_test = datasets.MNIST(root='./workspace/datasets/MNIST',train=False,download=True, transform=self.transform_test_mnist)
        self.SVHN_test_5000, self.other = random_split(self.SVHN_test,[5000,21032])
        self.MNIST_test_5000, self.others = random_split(self.MNIST_test,[5000,5000])
        self.concat_test = torch.utils.data.ConcatDataset([self.SVHN_test_5000,self.MNIST_test_5000])
        if self.Training==False:
            tuning_set, test_set = random_split(self.concat_test,[1000,9000])
            if self.TUNING:
                self.SVHN_MNIST_train = tuning_set
                self.SVHN_MNIST_test = test_set
            else:
                self.SVHN_MNIST_train = test_set
                self.SVHN_MNIST_test = test_set
                
    def train_dataloader(self): 
        return DataLoader(self.SVHN_MNIST_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.SVHN_MNIST_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.SVHN_MNIST_test, batch_size=self.batch_size, shuffle=False, num_workers=8)