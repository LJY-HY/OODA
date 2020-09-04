import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.classifiers import *
from datasets.CIFAR import *
from datasets.SVHN import *
from utils.args import *

if __name__ == '__main__':
    datasets = ['SVHN']
    NNModels = ['Densenet_BC']
    for dataset in datasets:
<<<<<<< HEAD
        if dataset == 'SVHN':
            dm = SVHNDataModule()
            max_epochs = 100
=======
        if dataset == 'CIFAR10':
            dm = CIFAR10DataModule()
            max_epochs = 60
        elif dataset == 'CIFAR100':
            dm = CIFAR100DataModule()
            max_epochs = 180
        elif dataset == 'SVHN':
            dm = SVHNDataModule()
            max_epochs = 30
>>>>>>> 2f6818701ec88a2438c019627a28a8403ca3678e
        for NNModel in NNModels:
            model_name = dataset + '_' + NNModel
            model = globals()[model_name]()
            modelpath  = './workspace/model_ckpts/' + model_name + '/'
            os.makedirs(modelpath, exist_ok=True)
            checkpoint_callback=ModelCheckpoint(filepath=modelpath)
            trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=1, num_nodes=1, max_epochs = max_epochs)
            if os.path.isfile(modelpath + 'final.ckpt'):
                model = model.load_from_checkpoint(checkpoint_path=modelpath + 'final.ckpt')
            else:
                trainer.fit(model, dm)
                trainer.save_checkpoint(modelpath + 'final.ckpt')
            trainer.test(model, datamodule = dm)