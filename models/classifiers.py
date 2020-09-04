import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models.vgg as VGG
import torchvision.models.resnet as Resnet

from models.WideResnet import Wide_ResNet
from models.Densenet import DenseNet
from models.Densenet_BC import DenseNet_BC
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

'''
SVHN_model skeleton
'''
class SVHN_LIGHTNING(pl.LightningModule):
    # Base model is VGG-16
    def __init__(self):
        super(SVHN_LIGHTNING, self).__init__()
        self.model = VGG.vgg16_bn()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        data, target = batch
        loss = F.cross_entropy(self.forward(data), target)
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def validation_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'val_loss':loss,'correct':correct}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('Validation accuracy : ',sum_correct/73257,'\n\n') # self.arg.validation_size
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('Test accuracy :',sum_correct/26032,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

class SVHN_WideResnet(SVHN_LIGHTNING):
    # This Module is based on WideResNet28-10 for dataset SVHN
    def __init__(self):
        super(SVHN_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,10,0.3,10)

class SVHN_Densenet_BC(SVHN_LIGHTNING):
    def __init__(self):
        super(SVHN_Densenet_BC,self).__init__()
        self.model = DenseNet_BC(num_classes=10)
    