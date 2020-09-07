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

parser = argparse.ArgumentParser(description='Pytorch Domain Adaptation in neural networks')

parser.add_argument('--source_dataset', default='MNIST',type=str, choices=['CIFAR10','CIFAR100','SVHN','MNIST_M','MNIST'],
                    help='source dataset')
parser.add_argument('--target_dataset', default='MNIST_M', type=str, choices=['LSUN','LSUN_resize','Imagenet','Uniform','Gaussian','SVHN','MNIST_M','CIFAR10','MNIST'],
                    help='target dataset')
parser.add_argument('--nn', default="Densenet_BC", type=str,
                    choices=['VGG','Resnet','WideResnet','Densenet','Densenet_BC'], help='neural network name and training set')
parser.add_argument('--mode', default='DA',type=str, choices=['Source_Only','Concat','DA'],
                    help='adaptation mode')
parser.set_defaults(argument=True)

def main():
    global args
    args = parser.parse_args()
    s_dataset = args.source_dataset
    t_dataset = args.target_dataset
    NNModel = args.nn
    model_name = s_dataset + '_' + NNModel
    model = globals()[model_name]()
    modelpath  = './workspace/model_ckpts/' + model_name + '/'
    s_dm = globals()[s_dataset+'DataModule'](batch_size=64);s_dm.setup()
    t_dm = globals()[t_dataset+'DataModule'](batch_size=64);t_dm.setup()
    os.makedirs(modelpath, exist_ok=True)
    checkpoint_callback=ModelCheckpoint(filepath=modelpath)
    trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=1, num_nodes=1, max_epochs = 60)

    if os.path.isfile(modelpath + 'final.ckpt'):
        model = model.load_from_checkpoint(checkpoint_path=modelpath + 'final.ckpt')
    elif args.mode=='Soruce_Only':
        # train model with source data only
        # and then, test with target data
        trainer.fit(model, s_dm)
        trainer.save_checkpoint(modelpath + 'final.ckpt')
    elif args.mode=='Concat':
        # Concat two dataset
        s_train_dataset = s_dm.train_dataloader().dataset
        t_train_dataset = t_dm.train_dataloader().dataset
        s_test_dataset = s_dm.test_dataloader().dataset
        t_test_dataset = t_dm.test_dataloader().dataset
       
        concat_train_dataset = ConcatDataset((s_train_dataset,t_train_dataset))
        concat_test_dataset = ConcatDataset((s_test_dataset,t_test_dataset))

        concat_dm=ConcatDataModule(concat_train_dataset,concat_test_dataset)
        trainer.fit(model, concat_dm)
        trainer.save_checkpoint(modelpath + 'final.ckpt')
    elif args.mode=='DA':
        # Train model using Domain Adaptation
        trainer.save_checkpoint(modelpath + 'final.ckpt')
        pass
    trainer.test(model, datamodule = t_dm)

if __name__ == '__main__':
    main()


# TODO : Concat multiple dataset
# TODO : import DANN into pytorch-lightning
# TODO : fix MNIST datamodule padding. Make it into 3-channel
