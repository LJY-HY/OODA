from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.classifiers import *
from datasets.CIFAR import *
from datasets.LSUN import *
from datasets.SVHN import *
from datasets.MNIST_M import *
from datasets.Imagenet import *
from utils.args import *
import calMetric

from ood_method.distribution_detector import DISTRIBUTION_DETECTOR
start = time.time()

def test(args):
    # args              : (defaults)
    # 1) in_dataset     : CIFAR10
    # 2) out_dataset    : LSUN
    # 3) nn             : Densenet
    # 4) magnitude      : 0.0012
    # 5) temperature    : 1000
    # 6) gpu            : 0

    criterion = nn.CrossEntropyLoss()

    in_dataset = args.in_dataset
    out_dataset = args.out_dataset
    NNModels = args.nn
    magnitude = args.magnitude
    temperature = args.temperature
    CUDA_DEVICE = args.gpu

    ##### Datamodule setting #####
    in_dm = globals()[in_dataset+'DataModule'](batch_size=64)
    if out_dataset != "Gaussian" and out_dataset!= "Uniform":
        out_dm = globals()[out_dataset+'DataModule'](batch_size=64)
  
    ##### Pretrained model setting #####
    model_name = in_dataset+'_'+NNModels
    model = globals()[model_name]()                                 # only model module is imported
    modelpath = './workspace/model_ckpts/' + model_name + '/'
    checkpoint_callback=ModelCheckpoint(filepath=modelpath)
    trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=1, num_nodes=1, max_epochs = 1)

    ##### Open files to save confidence score #####
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')


    if os.path.isfile(modelpath + 'final.ckpt'):
        model = model.load_from_checkpoint(checkpoint_path=modelpath + 'final.ckpt')
    else:
        print('No pretrained model.','Execute train.py first',sep='\n')
    
    if out_dataset == "Gaussian":
        calData.testGaussian(model,criterion,CUDA_DEVICE,in_dm,magnitude,temperature)
        calMetric.metric(model_name,out_dataset)
    elif out_dataset == "Uniform":
        calData.testUni(model,criterion,CUDA_DEVICE,in_dm,magnitude,temperature)
        calMetric.metric(model_name,out_dataset)
    else:
        # setting in-dist detector
        detector = DISTRIBUTION_DETECTOR(model,criterion,CUDA_DEVICE,magnitude,temperature,f1,g1)
        trainer.fit(detector,datamodule=in_dm)      
        # setting out-dist detector
        detector = DISTRIBUTION_DETECTOR(model,criterion,CUDA_DEVICE,magnitude,temperature,f2,g2)
        trainer.fit(detector,datamodule=out_dm)
        # calculate metrics
        calMetric.metric(in_dataset,out_dataset,NNModels)

        # TODO : delete validation loop in trainer.fit()
        # TODO : combine calMetric.metric into trainer.test
        # TODO : import Gaussian, Uniform into out_dataset