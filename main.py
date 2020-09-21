from __future__ import print_function
import argparse
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import cal 

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in_dataset', default='CIFAR10',type=str, choices=['LSUN','LSUN_resize','Imagenet','Uniform','Gaussian','SVHN','MNIST','MNIST_M','CIFAR10','CIFAR100'],
                    help='in-distribution dataset')
parser.add_argument('--out_dataset', default='LSUN', type=str, choices=['LSUN','LSUN_resize','Imagenet','Uniform','Gaussian','SVHN','MNIST','MNIST_M','CIFAR10','CIFAR100'],
                    help='out-of-distribution dataset')
parser.add_argument('--target_dataset', default='MNIST', type=str, choices=['LSUN','LSUN_resize','Imagenet','Uniform','Gaussian','SVHN','MNIST','MNIST_M','CIFAR10','CIFAR100'],
                    help='target dataset')
parser.add_argument('--nn', default="VGG", type=str,
                    choices=['VGG','Resnet','WideResnet','Densenet','Densenet_BC'], help='neural network name and training set')
parser.add_argument('--train_mode', default='DA',type=str, choices=['SO','CC','DA','TO'],
                    help='SO : Source Only      CC : Concat     DA : Domain Adaptation      TO : Target Only')
parser.add_argument('--magnitude', default=0, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')
parser.add_argument('--gpu', default = 0, type = int,
		            help='gpu index')
parser.add_argument('--tuning',action='store_true',
                    help='if True, tune parameter first, if False, run ODIN with given parameter')
parser.add_argument('--training',action='store_true',
                    help='if True, datamodule returns [50000,10000] dataset, if False, datamodule returns [1000,9000] dataset')
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()
    cal.test(args)
if __name__ == '__main__':
    main()