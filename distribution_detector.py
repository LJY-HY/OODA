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

'''
LIGHTNING MODULE SKELETON

-inputs
    1)model
    2)criterion
    3)CUDA_DEVICE
    4)in_dm
    5)out_dm
    6)magnitude
    7)temperature
'''

class DISTRIBUTION_DETECTOR(pl.LightningModule):
    def __init__(self,model,criterion,CUDA_DEVICE=0,magnitude=0.0012,temperature=1000,fp1=None,fp2=None):
        super(DISTRIBUTION_DETECTOR, self).__init__()
        self.model = model.cuda(CUDA_DEVICE)
        self.criterion = criterion
        self.CUDA_DEVICE = CUDA_DEVICE
        self.noiseMagnitude = magnitude
        self.temperature = temperature
        self.fp1 = fp1
        self.fp2 = fp2
        self.count=0

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self,batch,batch_idx):
        self.model.freeze()
        images, target = batch
        inputs = Variable(images.cuda(self.CUDA_DEVICE), requires_grad = True)
        del images
        outputs= self.forward(inputs)
        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
        nnOutputs = nnOutputs.transpose()
        for maxvalues in np.max(nnOutputs,axis=1):
            self.fp1.write("{}, {}, {}\n".format(self.temperature, self.noiseMagnitude, maxvalues))
        
        # Using temperature scaling
        outputs = outputs / self.temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs,axis=1)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(self.CUDA_DEVICE))
        loss = self.criterion(outputs, labels[0])
        loss.backward()
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -self.noiseMagnitude, gradient)
        outputs = self.model(Variable(tempInputs))
        outputs = outputs / self.temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
        nnOutputs = nnOutputs.transpose()
        for maxvalues in np.max(nnOutputs,axis=1):
            self.fp2.write("{}, {}, {}\n".format(self.temperature, self.noiseMagnitude, maxvalues))

        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        # Not used in this model_module
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
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
        print('Validation accuracy : ',sum_correct/dataset_size,'\n\n') # self.arg.validation_size
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
        print('Test accuracy :',sum_correct/dataset_size,'\n')            
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def backward(self,trainer,loss,optimizer,optimizer_idx):
        pass
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                        second_order_closure=None, on_tpu=False,
                        using_native_amp=False,using_lbfgs=False):
        pass


# TODO : sum_correct has to be divided by size of test_dataset