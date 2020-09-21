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
from scipy import misc

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.classifiers import *
from datasets.CIFAR import *
from datasets.LSUN import *
from datasets.SVHN import *
from datasets.MNIST_M import *
from datasets.MNIST import *
from datasets.Imagenet import *
from utils.args import *
import calMetric

from OOD_method.odin import ODIN
from DA_method.DANN import *
def test(args):
    # args              : (defaults)
    # 1) in_dataset     : CIFAR10
    # 2) out_dataset    : CIFAR100
    # 3) nn             : Densenet_BC
    # 4) magnitude      : 0.
    # 5) temperature    : 1000
    # 6) gpu            : 0
    # 7) tuning         : True
    # 8) training       : False

    criterion = nn.CrossEntropyLoss()

    in_dataset = args.in_dataset
    out_dataset = args.out_dataset
    NNModels = args.nn
    magnitude = args.magnitude
    temperature = args.temperature
    CUDA_DEVICE = args.gpu
    TUNING = args.tuning
    Training = args.training

    ##### Pretrained model setting #####
    model_name = in_dataset+'_'+NNModels
    adapted_model_name = args.train_mode + '_' + 'to'+ '_' +args.target_dataset + '_' + 'final.ckpt'
    model = globals()[model_name]()                                 # only model module is imported
    modelpath = './workspace/model_ckpts/' + model_name + '/'

    ##### Datamodule setting #####
    in_dm = globals()[in_dataset+'DataModule'](TUNING=TUNING,Training=Training,batch_size=1)
    out_dm = globals()[out_dataset+'DataModule'](TUNING=TUNING,Training=Training,batch_size=1)

    os.makedirs(modelpath, exist_ok=True)
    checkpoint_callback=ModelCheckpoint(filepath=modelpath+adapted_model_name)
    trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=1, num_nodes=1, max_epochs = 1)

    if os.path.isfile(modelpath + adapted_model_name):
        DANN_ = DANN(None,model,args.train_mode)
        model = DANN_.load_from_checkpoint(modelpath + adapted_model_name,model,args.train_mode)
    else:
        print('No pretrained model.','Execute train.py first',sep='\n')
        return 0

    
  
    ##### Softmax Scores Path Setting #####
    path = './workspace/softmax_scores/'
    os.makedirs(path,exist_ok=True)

    T_candidate = [1,10,100,1000]
    e_candidate = [0,0.0005,0.001,0.0014,0.002,0.0024,0.005,0.01,0.05,0.1,0.2]

    if TUNING:
        temperature=T_candidate
        magnitude=e_candidate
    else:
        # Make float/int object iterable
        temperature = [args.temperature]
        magnitude = [args.magnitude]

    tnr_best=0.
    T_temp=1
    ep_temp=0
    for T in temperature:
        for ep in magnitude:
            print('T       : ',T)
            print('epsilon : ',ep)
            ##### Open files to save confidence score #####
            f1 = open(path+"confidence_Base_In.txt", 'w')
            f2 = open(path+"confidence_Base_Out.txt", 'w')
            g1 = open(path+"confidence_Odin_In.txt", 'w')
            g2 = open(path+"confidence_Odin_Out.txt", 'w')
            if out_dataset == "Gaussian":  
                calMetric.metric(path)
            elif out_dataset == "Uniform":
                calMetric.metric(path)
            else:
                # setting in-dist detector
                detector = ODIN(model, criterion, CUDA_DEVICE, ep, T, f1, g1)
                trainer.fit(detector,datamodule=in_dm)      
                # setting out-dist detector
                detector = ODIN(model, criterion, CUDA_DEVICE, ep, T, f2, g2)
                trainer.fit(detector,datamodule=out_dm)
                # calculate metrics
                results = calMetric.metric(path)
                if tnr_best<results['Odin']['TNR']:
                    tnr_best=results['Odin']['TNR']
                    results_best = results
                    T_temp=T
                    ep_temp=ep
            f1.close()
            f2.close()
            g1.close()
            g2.close()

    if TUNING:
        TUNING=False            # Tuning Ended. run calMetric with rest 9,000 data with min(T,ep)
        print('\nBest Performance Out-of-Distribution Detection')
        print('T       : ',T_temp)
        print('epsilon : ',ep_temp)
        f1 = open(path+"confidence_Base_In.txt", 'w')
        f2 = open(path+"confidence_Base_Out.txt", 'w')
        g1 = open(path+"confidence_Odin_In.txt", 'w')
        g2 = open(path+"confidence_Odin_Out.txt", 'w')
        in_dm = globals()[in_dataset+'DataModule'](TUNING=TUNING,Training=Training,batch_size=1)
        out_dm = globals()[out_dataset+'DataModule'](TUNING=TUNING,Training=Training,batch_size=1)
        detector = ODIN(model, criterion, CUDA_DEVICE, ep_temp, T_temp, f1, g1)
        trainer.fit(detector,datamodule=in_dm)
        detector = ODIN(model, criterion, CUDA_DEVICE, ep_temp, T_temp, f2, g2)
        trainer.fit(detector,datamodule=out_dm)
        results=calMetric.metric(path)
        f1.close()
        f2.close()
        g1.close()
        g2.close()
   
    print('\nBest Performance Out-of-Distribution Detection')
    print('T       : ',T_temp)
    print('epsilon : ',ep_temp)
    print("{:31}{:>22}".format("Neural network architecture:", NNModels))
    print("{:31}{:>22}".format("In-distribution dataset:", in_dataset))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", out_dataset))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Odin"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("TNR at TPR 95%:",results_best['Base']['TNR']*100, results_best['Odin']['TNR']*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("Accuracy:",results_best['Base']['DTACC']*100, results_best['Odin']['DTACC']*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:",results_best['Base']['AUROC']*100, results_best['Odin']['AUROC']*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:",results_best['Base']['AUIN']*100, results_best['Odin']['AUIN']*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:",results_best['Base']['AUOUT']*100, results_best['Odin']['AUOUT']*100))