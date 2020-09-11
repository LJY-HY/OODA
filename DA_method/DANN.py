import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.autograd import Variable
from torch.autograd import Function

class DANN(pl.LightningModule):
    # This module is based on classifier.Densenet_BC.
    # Only ReverseLayerF layer is applied.

    def __init__(self,model=None,mode=None):
        super(DANN, self).__init__()
        self.model=model.model
        self.mode=mode
        self.alpha=0.
        self.model.class_classifier=nn.Sequential()
        self.model.class_classifier.add_module('c_fc1', nn.Linear(342, 100))   # check input size
        self.model.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.model.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.model.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.model.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.model.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.model.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.model.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
       
        self.model.domain_classifier = nn.Sequential()
        self.model.domain_classifier.add_module('d_fc1', nn.Linear(342  , 100))  # check input size
        self.model.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.model.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.model.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
       
    def forward(self, x, alpha):
        output = self.model.conv1(x)
        output = self.model.trans1(self.model.block1(output))
        output = self.model.trans2(self.model.block2(output))
        output = self.model.block3(output)
        output = self.model.relu(self.model.bn1(output))
        output = F.avg_pool2d(output,8)
        output = output.view(-1,342)
        # CHECK this line works
        reverse_feature = ReverseLayerF.apply(output,self.alpha)
        class_output = self.model.class_classifier(output)
        domain_output = self.model.domain_classifier(reverse_feature)

        return class_output, domain_output

    def training_step(self,batch,batch_idx):
        self.len_dataloader=self.train_dataloader().dataset.__len__()
        self.max_epochs=self.trainer.max_epochs
        p = float(batch_idx + self.current_epoch * self.len_dataloader) / self.max_epochs / self.len_dataloader
        self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

        s_batch,t_batch=batch
        s_data, s_label = s_batch
        t_data, t_label = t_batch
        
        # Source Loss
        s_class_output, s_domain_output = self.forward(s_data,alpha=self.alpha)
        s_class_loss = F.cross_entropy(s_class_output, s_label)

        s_domain_label = np.zeros(s_domain_output.shape[0])
        s_domain_label += 0
        s_domain_label = torch.from_numpy(s_domain_label).long().to(s_domain_output.device)
        s_domain_loss = F.cross_entropy(s_domain_output,s_domain_label)
        
        # Target Loss
        t_class_output,t_domain_output = self.forward(t_data,alpha=self.alpha)
        t_class_loss = F.cross_entropy(t_class_output, t_label)                 # Not Used
        t_domain_label = np.zeros(t_domain_output.shape[0])
        t_domain_label += 1
        t_domain_label=torch.from_numpy(t_domain_label).long().to(t_domain_output.device)
        t_domain_loss = F.cross_entropy(t_domain_output,t_domain_label)

        s_class_pred = s_class_output.argmax(dim=1,keepdim=True)
        s_domain_pred = s_domain_output.argmax(dim=1,keepdim=True)
        t_class_pred = t_class_output.argmax(dim=1,keepdim=True)
        t_domain_pred = t_domain_output.argmax(dim=1,keepdim=True)

        if self.mode=='SO':
            loss = s_class_loss
        elif self.mode=='CC':
            loss = s_class_loss + t_class_loss
        elif self.mode=='DA':
            loss = s_class_loss + s_domain_loss + t_domain_loss

        # Use class_correct/domain_correct to check accuracy in train_dataset
        class_correct = s_class_pred.eq(s_label.view_as(s_class_pred)).sum().item() + \
                        t_class_pred.eq(t_label.view_as(t_class_pred)).sum().item()
        domain_correct = s_domain_pred.eq(s_domain_label.view_as(s_domain_pred)).sum().item()  + \
                         t_domain_pred.eq(t_domain_label.view_as(t_domain_pred)).sum().item()
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs, 'batch_idx':batch_idx}

    def configure_optimizers(self):
        # Not used in this model_module
        return optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self,batch,batch_idx,dataloader_idx):
        data, target = batch
        class_output,domain_output = self.forward(data,alpha=self.alpha)
        class_loss = F.cross_entropy(class_output, target)
        
        domain_target=np.zeros(domain_output.shape[0])
        domain_target+=dataloader_idx
        domain_target=torch.from_numpy(domain_target).long().to(domain_output.device)
        domain_loss = F.cross_entropy(domain_output,domain_target)

        loss = class_loss + domain_loss
      
        class_pred = class_output.argmax(dim=1,keepdim=True)
        domain_pred = domain_output.argmax(dim=1,keepdim=True)

        class_correct = class_pred.eq(target.view_as(class_pred)).sum().item()
        domain_correct = domain_pred.eq(domain_target.view_as(domain_pred)).sum().item()
       
        return {'loss':loss, 'correct':class_correct, 'domain_correct':domain_correct,
                'batch_size':target.shape[0]}

    def validation_epoch_end(self,outputs):
        s_avg_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        s_sum_correct = sum([x['correct'] for x in outputs[0]])
        s_dataset_size = sum([x['batch_size'] for x in outputs[0]])

        t_avg_loss = torch.stack([x['loss'] for x in outputs[1]]).mean()
        t_sum_correct = sum([x['correct'] for x in outputs[1]])
        t_dataset_size = sum([x['batch_size'] for x in outputs[1]])
        tensorboard_logs = {'val_loss':s_avg_loss}
        print('\nSource_Validation accuracy : ',s_sum_correct/s_dataset_size)
        print('\nTarget_Validation accuracy : ',t_sum_correct/t_dataset_size,'\n')
        return {'avg_val_loss':s_avg_loss, 'log':tensorboard_logs}    

    def test_step(self,batch,batch_idx,dataloader_idx):
        data, target = batch
        class_output,domain_output = self.forward(data,alpha=self.alpha)
        class_loss = F.cross_entropy(class_output, target)
        
        domain_target=np.zeros(domain_output.shape[0])
        domain_target+=dataloader_idx
        domain_target=torch.from_numpy(domain_target).long().to(domain_output.device)
        domain_loss = F.cross_entropy(domain_output,domain_target)

        loss = class_loss + domain_loss
      
        class_pred = class_output.argmax(dim=1,keepdim=True)
        domain_pred = domain_output.argmax(dim=1,keepdim=True)

        class_correct = class_pred.eq(target.view_as(class_pred)).sum().item()
        domain_correct = domain_pred.eq(domain_target.view_as(domain_pred)).sum().item()
        #domain_correct is not used. Use it if you want to check

        return {'loss':loss, 'correct':class_correct, 'domain_correct':domain_correct,
                'batch_size':target.shape[0]}

    def test_epoch_end(self, outputs):
        s_avg_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        s_sum_correct = sum([x['correct'] for x in outputs[0]])
        s_dataset_size = sum([x['batch_size'] for x in outputs[0]])

        t_avg_loss = torch.stack([x['loss'] for x in outputs[1]]).mean()
        t_sum_correct = sum([x['correct'] for x in outputs[1]])
        t_dataset_size = sum([x['batch_size'] for x in outputs[1]])
        tensorboard_logs = {'val_loss':s_avg_loss}
        print('\nSource_Validation accuracy : ',s_sum_correct/s_dataset_size)
        print('\nTarget_Validation accuracy : ',t_sum_correct/t_dataset_size,'\n')
        return {'avg_val_loss':s_avg_loss, 'log':tensorboard_logs}  

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# TODO : fix loss according to train.py --mode
