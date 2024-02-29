# # Universal Patterns for Revealing Backdoors CIFAR-10
# 
# Here we perform our optimization to obtain the universal pattern that help us reveal the backdoor.

import numpy as np
import torch
from torch import optim

import utils.model as model

import pickle
import time
import glob
from tqdm import tqdm

import os
import sys

import torch

import logging

# sys argv check
if len(sys.argv) != 3:
    print("Usage: python train_ULP.py <num_ULPs> <logfile>")
    sys.exit()

#logging
logfile = sys.argv[2]
if not os.path.exists(os.path.dirname(logfile)):
    os.makedirs(os.path.dirname(logfile))
os.makedirs("results", exist_ok=True)

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
    logging.FileHandler(logfile, "w"),
    logging.StreamHandler()
])

def import_from(module, name):
  module = __import__(module, fromlist=[name])
  return getattr(module, name)

init_num_filters=64
inter_fc_dim=384
nofclasses=10 #CIFAR10
use_cuda=True

N=int(sys.argv[1])

# poisoned
poisoned_models_train = sorted(glob.glob('/home/hegedusi/backdoor_models/R18_cifar10_robust/cifar10-*.pth'))[:12]
poisoned_models_val = sorted(glob.glob('/home/hegedusi/backdoor_models/R18_cifar10_robust/cifar10-*.pth'))[12:]

# clean models
clean_models=glob.glob('/home/hegedusi/backdoor_models/R18_cifar10_robust/cifar10_s*.pth')

# train - 400 clean 400 poisoned
models_train=clean_models[:12] + poisoned_models_train
labels_train=np.concatenate([np.zeros((len(clean_models[:12]),)),np.ones((len(poisoned_models_train),))])

# val - 100 clean 100 poisoned
models_val=clean_models[12:] + poisoned_models_val
labels_val=np.concatenate([np.zeros((len(clean_models[12:]),)),np.ones((len(poisoned_models_val),))])

print(len(models_train), len(models_val))

train_models,val_models,train_labels,val_labels=models_train,models_val,labels_train,labels_val

device = torch.device('cuda:' + str(0))

#IDE jönne a RESNET Kódja:
ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
layers = [2, 2, 2, 2]
# layers = [1, 1, 1, 1]
cnn = ResNet(BasicBlock, layers, nofclasses).to(device)

# ### Perform Optimization

X=torch.rand((N,3,32,32),requires_grad=True,device='cuda')
X.data*=255.
W=torch.randn((N*nofclasses,2),requires_grad=True,device='cuda')
b=torch.zeros((2,),requires_grad=True,device='cuda')

optimizerX = optim.SGD(params=[X],lr=1e+3)                 #1e+2
optimizerWb = optim.Adam(params=[W,b],lr=1e-3)          #1e-3

cross_entropy=torch.nn.CrossEntropyLoss()

batchsize=50
REGULARIZATION=1e-6       #1e-6

Xgrad=list()
Wgrad=list()
bgrad=list()

max_val_accuracy=0
for epoch in range(1000):
    epoch_loss=list()
    randind=np.random.permutation(len(train_models))
    train_models=np.asarray(train_models)[randind]
    train_labels=train_labels[randind]
    for i,model in tqdm(enumerate(train_models)):
        cnn.load_state_dict(torch.load(model))
        cnn.eval()
        label=np.array([train_labels[i]])
        y=torch.from_numpy(label).type(torch.LongTensor).to(device)
        logit=torch.matmul(cnn(X.to(device)).view(1,-1),W)+b

        reg_loss = REGULARIZATION * (torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:])) +
                                     torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :])))

        loss=cross_entropy(logit,y)+reg_loss


        optimizerWb.zero_grad()
        optimizerX.zero_grad()

        loss.backward()
   
        if np.mod(i,batchsize)==0 and i!=0:
            Xgrad=torch.stack(Xgrad,0)
#             Wgrad=torch.stack(Wgrad,0)
#             bgrad=torch.stack(bgrad,0)

            X.grad.data=Xgrad.mean(0)
#             W.grad.data=Wgrad.mean(0)
#             b.grad.data=bgrad.mean(0)

            optimizerX.step()

            X.data[X.data<0.]=0.
            X.data[X.data>255.]=255.

            Xgrad=list()
            Wgrad=list()
            bgrad=list()

        Xgrad.append(X.grad.data)
#         Wgrad.append(W.grad.data)
#         bgrad.append(b.grad.data)
        optimizerWb.step()
        epoch_loss.append(loss.item())

    with torch.no_grad():
        pred=list()
        for i,model in enumerate(train_models):
            cnn.load_state_dict(torch.load(model))
            cnn.eval()
            label=np.array([train_labels[i]])
            logit=torch.matmul(cnn(X.to(device)).view(1,-1),W)+b
            pred.append(torch.argmax(logit,1))
        train_accuracy=(1*(np.asarray(pred)==train_labels.astype('uint'))).sum()/float(train_labels.shape[0])

        pred=list()
        for i,model in enumerate(val_models):
            cnn.load_state_dict(torch.load(model))
            cnn.eval()
            label=np.array([val_labels[i]])
            logit=torch.matmul(cnn(X.to(device)).view(1,-1),W)+b
            pred.append(torch.argmax(logit,1))
        val_accuracy=(1*(np.asarray(pred)==val_labels.astype('uint'))).sum()/float(val_labels.shape[0])

    if val_accuracy>=max_val_accuracy:
        pickle.dump([X.data,W.data,b.data],open('./results/ULP_vggmod_CIFAR-10_N{}.pkl'.format(N),'wb'))
        max_val_accuracy=np.copy(val_accuracy)

    logging.info('Epoch %03d Loss=%f, Train Acc=%f, Val Acc=%f'%(epoch,np.asarray(epoch_loss).mean(),train_accuracy*100.,val_accuracy*100.))
logging.info(max_val_accuracy)