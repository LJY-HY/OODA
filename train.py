'''
Train model without any trainig method(DA,OOD).
'''
import os
import torch
import argparse
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.classifiers import *
from datasets.CIFAR import *
from datasets.SVHN import *
from datasets.MNIST_M import *
from datasets.MNIST import *
from datasets.concatDM import *

from utils.args import *

from DA_method.DANN import *

parser = argparse.ArgumentParser(description='Pytorch Domain Adaptation in neural networks')

parser.add_argument('--source_dataset', default='MNIST',type=str, choices=['CIFAR10','CIFAR100','SVHN','MNIST_M','MNIST'],
                    help='source dataset')
parser.add_argument('--target_dataset', default='MNIST_M', type=str, choices=['LSUN','LSUN_resize','Imagenet','Uniform','Gaussian','SVHN','MNIST_M','CIFAR10','MNIST'],
                    help='target dataset')
parser.add_argument('--nn', default="VGG", type=str,
                    choices=['plain','VGG','Resnet','WideResnet','Densenet','Densenet_BC'], help='neural network name and training set')
parser.add_argument('--train_mode', default='DA',type=str, choices=['SO','CC','DA','TO'],
                    help='SO : Source Only      CC : Concat     DA : Domain Adaptation      TO : Target Only')
parser.add_argument('--batch_size',default=64,type=int)
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()
    s_dataset = args.source_dataset
    t_dataset = args.target_dataset
    NNModel = args.nn
    batch_size=args.batch_size

    ##### Pretrained model setting #####
    model_name = s_dataset + '_' + NNModel
    adapted_model_name = args.train_mode + '_' + 'to'+ '_' +args.target_dataset + '_' + 'final.ckpt'
    model = globals()[model_name]()
    modelpath  = './workspace/model_ckpts/' + model_name + '/'

    ##### Datamodule setting #####
    s_dm = globals()[s_dataset+'DataModule'](batch_size=batch_size)
    t_dm = globals()[t_dataset+'DataModule'](batch_size=batch_size)
    concat_dm = ConcatDataModule(s_dm,t_dm)
    
    os.makedirs(modelpath, exist_ok=True)
    checkpoint_callback=ModelCheckpoint(filepath=modelpath+adapted_model_name)
    trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=[1], num_nodes=1, max_epochs = 60)

    if os.path.isfile(modelpath + adapted_model_name):
        # if model trained under args.train_mode exists,
        # load from checkpoint
        DANN_ = DANN(None,model,args.train_mode)
        DA_model = DANN_.load_from_checkpoint(modelpath + adapted_model_name,model,args.train_mode)
    else:
        DA_model = DANN(None,model,args.train_mode)
        trainer.fit(DA_model,concat_dm)
        trainer.save_checkpoint(modelpath + adapted_model_name)

    trainer.test(DA_model,datamodule=concat_dm)
    
if __name__ == '__main__':
    main()