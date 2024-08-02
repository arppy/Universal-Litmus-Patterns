
import numpy as np
import sys
import os

import torch
from torch import optim
import torch.nn.functional as F

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pickle
import time
import glob

import torch
from torch.utils import data
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
import logging
import pdb
import random

avgpool = torch.nn.AdaptiveAvgPool1d(200)

nofclasses = 10  # Tiny-ImageNet
use_cuda = True
seed = 42
N = 10
GPU = 0
device = torch.device('cuda:' + str(GPU))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transformNorm = transforms.Normalize(mean, std)

X, W, b = pickle.load(open('./results/ULP_resnetmod_imagenette_N{}_a.pkl'.format(N), 'rb'))

test_poisoned_models = glob.glob('/home/berta/backdoor_models/R18_imagenette_robust_extended_test/*-?-*.pth')
test_clean_models = glob.glob('/home/berta/backdoor_models/R18_imagenette_robust_extended_test/imagenette_*.pth')
test_models = test_clean_models + test_poisoned_models
test_labels = np.concatenate([np.zeros((len(test_clean_models),)), np.ones((len(test_poisoned_models),))])
random.seed(seed)
random.shuffle(test_models)
random.seed(seed)
random.shuffle(test_labels)

cnn = torchvision_models.resnet18(weights=None)
cnn.fc = torch.nn.Linear(512, nofclasses)
cnn = cnn.to(device)

features = list()
probabilities = list()
pred = list()
for i, model_ in enumerate(test_models):
    cnn.load_state_dict(torch.load(model_, map_location=device))
    cnn.eval()
    label = np.array([test_labels[i]])
    output = avgpool(cnn(transformNorm(X.to(device))).view(1, 1, -1)).squeeze(0)
    logit = torch.matmul(output, W) + b
    # 	logit=torch.matmul(cnn(X.to(device)).view(1,-1),W)+b
    probs = torch.nn.Softmax(dim=1)(logit)
    features.append(logit.detach().cpu().numpy())
    probabilities.append(probs.detach().cpu().numpy())
    pred.append(torch.argmax(logit, 1).cpu().numpy())

features_np = np.stack(features).squeeze()
probs_np = np.stack(probabilities).squeeze()
pred_np = np.stack(pred).squeeze()

fpr, tpr, thresholds = roc_curve(test_labels, probs_np[:, 1])
auc = roc_auc_score(test_labels, probs_np[:, 1])

pickle.dump([fpr, tpr, thresholds, auc], open("./results/ROC_ULP_N{}.pkl".format(N), "wb"))