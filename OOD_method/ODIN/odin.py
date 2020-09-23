from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from scipy import misc

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

from models.classifiers import *
from datasets.CIFAR import *
from datasets.LSUN import *
from datasets.SVHN import *
from datasets.MNIST_M import *
from datasets.MNIST import *
from datasets.Imagenet import *
from datasets.SVHN_MNIST import *
from utils.args import *
from histogram import *
import calMetric

from DA_method.DANN import *
def odin(args):
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
    target_dataset= args.target_dataset
    NNModels = args.nn
    magnitude = args.magnitude
    temperature = args.temperature
    CUDA_DEVICE = args.gpu
    TUNING = args.tuning
    Training = args.training

    ##### Pretrained model setting #####
    model_name = in_dataset+'_'+NNModels
    adapted_model_name = args.train_mode + '_' + 'to'+ '_' +args.target_dataset+'_' + 'final.ckpt'
    model = globals()[model_name]()                                 # only model module is imported
    modelpath = './workspace/model_ckpts/' + model_name + '/'

    ##### Datamodule setting #####
    if args.in_num!=1:
        in_dm = globals()[in_dataset+target_dataset+'DataModule'](TUNING=TUNING,Training=Training,batch_size=1)
        out_dm = globals()[out_dataset+'DataModule'](TUNING=TUNING,Training=Training,batch_size=1)
    else:
        # train_mode에 상관없이 in-dist 에 들어가는 dataset의 개수가 1개인 경우
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

    ##### distribution saver path Setting #####
    result_path = './OOD_method/distribution_result/'+args.train_mode+'/'+args.in_dataset+str(args.in_num)+args.out_dataset+'/'
    os.makedirs(result_path,exist_ok=True)

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
        results_best=calMetric.metric(path)
        f1.close()
        f2.close()
        g1.close()
        g2.close()
    
    save_histogram(path,result_path)

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


class ODIN(pl.LightningModule):
    def __init__(self,model,criterion,CUDA_DEVICE=0,magnitude=0.0012,temperature=1000,fp1=None,fp2=None):
        super(ODIN, self).__init__()
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
        if len(outputs)==2:
            outputs=outputs[0]
        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
        nnOutputs = nnOutputs.transpose()
        for maxvalues in np.max(nnOutputs,axis=1):
            self.fp1.write("{}\n".format(maxvalues))
        
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
        if len(outputs)==2:
            outputs = outputs[0]
        outputs = outputs / self.temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
        nnOutputs = nnOutputs.transpose()
        for maxvalues in np.max(nnOutputs,axis=1):
            self.fp2.write("{}\n".format(maxvalues))

        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        # Not used in this model_module
        optimizer = optim.SGD(self.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1), 'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def test_step(self,batch,batch_idx):
        data, target = batch
        outputs = self.forward(data)
        if len(outputs)==2:
            outouts = outputs[0]
        pred = outputs.argmax(dim=1,keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return {'test_loss':F.cross_entropy(outputs,target), 'correct':correct,'batch_size':target.shape[0]}

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