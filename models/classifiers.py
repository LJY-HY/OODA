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
datamodule skeleton
'''
class LIGHTNING_Model(pl.LightningModule):
    # Base model is VGG-16
    def __init__(self):
        super(LIGHTNING_Model, self).__init__()
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
        return {'val_loss':loss,'correct':correct, 'batch_size':target.shape[0]}

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        dataset_size = sum([x['batch_size'] for x in outputs])
        tensorboard_logs = {'val_loss':avg_loss}
        print('\nValidation accuracy : ',sum_correct/dataset_size,'\n\n') # self.arg.validation_size
        return {'avg_val_loss':avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(output,target), 'correct':correct,'batch_size':target.shape[0]}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        sum_correct = sum([x['correct'] for x in outputs])
        dataset_size = sum([x['batch_size'] for x in outputs])
        tensorboard_logs = {'test_loss': avg_loss}
        print('\nTest accuracy :',sum_correct/dataset_size,'\n')
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

'''
SVHN trained model skeleton
'''
class SVHN_LIGHTNING(LIGHTNING_Model):
    def __init__(self):
       super(SVHN_LIGHTNING, self).__init__()
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

class SVHN_WideResnet(SVHN_LIGHTNING):
    # This Module is based on WideResNet28-10 for dataset SVHN
    def __init__(self):
        super(SVHN_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,10,0.3,10)

class SVHN_Densenet_BC(SVHN_LIGHTNING):
    def __init__(self):
        super(SVHN_Densenet_BC,self).__init__()
        self.model = DenseNet_BC(num_classes=10)

class SVHN_plain(SVHN_LIGHTNING):
    def __init__(self):
        super(SVHN_plain,self).__init__()
        self.model.feature = nn.Sequential()
        self.model.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.model.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.model.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.model.feature.add_module('f_relu1', nn.ReLU(True))
        self.model.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.model.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.model.feature.add_module('f_drop1', nn.Dropout2d())
        self.model.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.model.feature.add_module('f_relu2', nn.ReLU(True))

class SVHN_VGG(SVHN_LIGHTNING):
    # This Module is based on VGG-16 for dataset SVHN
    def __init__(self):
        super(SVHN_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,10)
        )

class SVHN_Resnet(SVHN_LIGHTNING):
    # This Module is based on Resnet-50 for dataset SVHN
    def __init__(self):
        super(SVHN_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=10)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 10)
        del self.model.maxpool
        self.model.maxpool = lambda x : x
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.model.features = nn.Sequential(
                                self.model.conv1,
                                self.model.bn1,
                                self.model.relu,
                                self.model.layer1,
                                self.model.layer2,
                                self.model.layer3,
                                self.model.layer4,
                                self.model.avgpool
        )
        

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model.features(x)
        x = torch.flatten(x, 1)
        x = self.model.linear(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

'''
CIFAR10 trained model skeleton
'''
class CIFAR10_LIGHTNING(LIGHTNING_Model):
    # This Module is based on VGG-16
    def __init__(self):
        super(CIFAR10_LIGHTNING, self).__init__()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

class CIFAR10_VGG(CIFAR10_LIGHTNING):
    # This Module is based on VGG-16 for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,10)
        )

class CIFAR10_Resnet(CIFAR10_LIGHTNING):
    # This Module is based on Resnet-50 for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=10)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 10)
        del self.model.maxpool
        self.model.maxpool = lambda x : x

class CIFAR10_WideResnet(CIFAR10_LIGHTNING):
    # This Module is based on WideResNet28-10 for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,10,0.3,10)

class CIFAR10_Densenet(CIFAR10_LIGHTNING):
    # This Module is based on Densenet for dataset CIFAR10
    def __init__(self):
        super(CIFAR10_Densenet, self).__init__()
        self.model = DenseNet()

class CIFAR10_Densenet_BC(CIFAR10_LIGHTNING):
    def __init__(self):
        super(CIFAR10_Densenet_BC,self).__init__()
        self.model = DenseNet_BC(num_classes=10)

'''
CIFAR100 trained model skeleton
'''
class CIFAR100_LIGHTNING(LIGHTNING_Model):
    # This Module is based on VGG-16
    def __init__(self):
        super(CIFAR100_LIGHTNING, self).__init__()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

class CIFAR100_VGG(CIFAR100_LIGHTNING):
    # This Module is based on VGG-16 for dataset CIFAR100
    def __init__(self):
        super(CIFAR100_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,100)
        )

class CIFAR100_Resnet(CIFAR100_LIGHTNING):
    # This Module is based on Resnet-50 for dataset CIFAR100
    def __init__(self):
        super(CIFAR100_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=100)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 100)
        del self.model.maxpool
        self.model.maxpool = lambda x : x
    
class CIFAR100_WideResnet(CIFAR100_LIGHTNING):
    # This Module is based on WideResNet 28-20 for dataset CIFAR-100
    def __init__(self):
        super(CIFAR100_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,20,0.3,100)

class CIFAR100_Densenet(CIFAR100_LIGHTNING):
    # This Module is based on VGG-16 for dataset CIFAR100
    def __init__(self):
        super(CIFAR100_Densenet, self).__init__()
        self.model = DenseNet(num_classes=100)

class CIFAR100_Densenet_BC(CIFAR10_LIGHTNING):
    def __init__(self):
        super(CIFAR100_Densenet_BC,self).__init__()
        self.model = DenseNet_BC(num_classes=100)


'''
MNIST trained model skeleton
'''
class MNIST_LIGHTNING(LIGHTNING_Model):
    # This Module is based on VGG-16
    def __init__(self):
        super(MNIST_LIGHTNING, self).__init__()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

class MNIST_plain(MNIST_LIGHTNING):
    def __init__(self):
        super(MNIST_plain,self).__init__()
        self.model.feature = nn.Sequential()
        self.model.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.model.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.model.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.model.feature.add_module('f_relu1', nn.ReLU(True))
        self.model.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.model.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.model.feature.add_module('f_drop1', nn.Dropout2d())
        self.model.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.model.feature.add_module('f_relu2', nn.ReLU(True))

class MNIST_VGG(MNIST_LIGHTNING):
    # This Module is based on VGG-16 for dataset MNIST
    def __init__(self):
        super(MNIST_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,10)
        )

class MNIST_Resnet(MNIST_LIGHTNING):
    # This Module is based on Resnet-50 for dataset MNIST
    def __init__(self):
        super(MNIST_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=10)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 10)
        del self.model.maxpool
        self.model.maxpool = lambda x : x

class MNIST_WideResnet(MNIST_LIGHTNING):
    # This Module is based on WideResNet28-10 for dataset MNIST
    def __init__(self):
        super(MNIST_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,10,0.3,10)

class MNIST_Densenet(MNIST_LIGHTNING):
    # This Module is based on Densenet for dataset MNIST
    def __init__(self):
        super(MNIST_Densenet, self).__init__()
        self.model = DenseNet()

class MNIST_Densenet_BC(MNIST_LIGHTNING):
    def __init__(self):
        super(MNIST_Densenet_BC,self).__init__()
        self.model = DenseNet_BC(num_classes=10)



'''
MNIST-M trained model skeleton
'''
class MNIST_M_LIGHTNING(LIGHTNING_Model):
    # This Module is based on VGG-16
    def __init__(self):
        super(MNIST_M_LIGHTNING, self).__init__()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

class MNIST_M_VGG(MNIST_M_LIGHTNING):
    # This Module is based on VGG-16 for dataset MNIST_M
    def __init__(self):
        super(MNIST_M_VGG, self).__init__()
        self.model = VGG.vgg16_bn()
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512,10)
        )

class MNIST_M_Resnet(MNIST_M_LIGHTNING):
    # This Module is based on Resnet-50 for dataset MNIST_M
    def __init__(self):
        super(MNIST_M_Resnet, self).__init__()
        self.model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=10)
        self.model.inplanes=64
        self.model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.bn1 = nn.BatchNorm2d(64)
        self.model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 10)
        del self.model.maxpool
        self.model.maxpool = lambda x : x

class MNIST_M_WideResnet(MNIST_M_LIGHTNING):
    # This Module is based on WideResNet28-10 for dataset MNIST_M
    def __init__(self):
        super(MNIST_M_WideResnet, self).__init__()
        self.model = Wide_ResNet(28,10,0.3,10)

class MNIST_M_Densenet(MNIST_M_LIGHTNING):
    # This Module is based on Densenet for dataset MNIST_M
    def __init__(self):
        super(MNIST_M_Densenet, self).__init__()
        self.model = DenseNet()

class MNIST_M_Densenet_BC(MNIST_M_LIGHTNING):
    def __init__(self):
        super(MNIST_M_Densenet_BC,self).__init__()
        self.model = DenseNet_BC(num_classes=10)