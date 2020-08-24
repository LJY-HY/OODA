"""
iSUN 처리
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc

def testData(model,criterion,CUDA_DEVICE,in_dm,out_dm,noiseMagnitude,temperature):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = 10000
    # if out_dm == iSUNDataModule(): N = 8925
    print("Processing in-distribution images")
########################################In-distribution###########################################
#   testloader10    : in_dm
#   testloader      : out_dm
    in_dm.setup()
    out_dm.setup()
    testloader10=in_dm.test_dataloader()
    testloader = out_dm.test_dataloader()

    for j, data in enumerate(testloader10):
        if j<1000: continue
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = model.forward(inputs)
        

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temperature
	
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()
        
        if j == N - 1: break


    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
        if j<1000: continue
        images, _ = data
    
    
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = model(inputs)
        


        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temperature
  
  
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break




def testGaussian(model,criterion,CUDA_DEVICE,in_dm,magnitude,temperature):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
########################################In-Distribution###############################################
    N = 10000
    print("Processing in-distribution images")
    in_dm.setup()
    out_dm.setup()
    testloader10=in_dm.test_dataloader()
    testloader = in_dm.test_dataloader()

    for j, data in enumerate(testloader10):
        
        if j<1000: continue
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = model(inputs)
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temperature
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

        g1.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

    
    
########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    for j, data in enumerate(testloader):
        if j<1000: continue
        
        images = torch.randn(1,3,32,32) + 0.5
        images = torch.clamp(images, 0, 1)
        images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
        images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
        images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
        
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = model(inputs)
        
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temperature
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break




def testUni(model,criterion,CUDA_DEVICE,in_dm,magnitude,temperature):
    t0 = time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
########################################In-Distribution###############################################
    N = 10000
    print("Processing in-distribution images")

    in_dm.setup()
    out_dm.setup()
    testloader10=in_dm.test_dataloader()
    testloader = in_dm.test_dataloader()

    for j, data in enumerate(testloader10):
        if j<1000: continue
        
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = model(inputs)
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temperature
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

        g1.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4}  images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()



########################################Out-of-Distribution######################################
    print("Processing out-of-distribution images")
    for j, data in enumerate(testloader):
        if j<1000: continue
        
        images = torch.rand(1,3,32,32)
        images[0][0] = (images[0][0] - 125.3/255) / (63.0/255)
        images[0][1] = (images[0][1] - 123.0/255) / (62.1/255)
        images[0][2] = (images[0][2] - 113.9/255) / (66.7/255)
        
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = model(inputs)
        
        
        
        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        
        # Using temperature scaling
        outputs = outputs / temperature
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temperature, noiseMagnitude, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break











