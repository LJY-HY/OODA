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

parser.add_argument('--in_dataset', default='CIFAR10',type=str, choices=['CIFAR10','CIFAR100','SVHN','MNIST_M'],
                    help='in-distribution dataset')
parser.add_argument('--out_dataset', default='LSUN', type=str, choices=['LSUN','LSUN_resize','Imagenet','Uniform','Gaussian','SVHN','MNIST_M','CIFAR10','MNIST'],
                    help='out-of-distribution dataset')
parser.add_argument('--nn', default="Densenet_BC", type=str,
                    choices=['VGG','Resnet','WideResnet','Densenet','Densenet_BC'], help='neural network name and training set')
parser.add_argument('--magnitude', default=0, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')
parser.add_argument('--gpu', default = 0, type = int,
		            help='gpu index')
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()
    cal.test(args)

if __name__ == '__main__':
    main()