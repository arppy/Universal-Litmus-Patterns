
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

test_poisoned_models = glob.glob('/home/berta/backdoor_models/R18_imagenette_robust_extended_defpre_minASR025_test/*-?-*.pth')
test_clean_models = glob.glob('/home/berta/backdoor_models/R18_imagenette_robust_extended_defpre_minASR025_test/imagenette_*.pth')
test_models = test_clean_models + test_poisoned_models
test_labels = np.concatenate([np.zeros((len(test_clean_models),)), np.ones((len(test_poisoned_models),))])


cnn = torchvision_models.resnet18(weights=None)
cnn.fc = torch.nn.Linear(512, nofclasses)
cnn = cnn.to(device)


# Initialize accumulators for TP, FP, TN, FN
total_TP = 0
total_FP = 0
total_TN = 0
total_FN = 0

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
    predicted_class = torch.argmax(probs, dim=1)
    TP = ((predicted_class == 1) & (label == 1)).sum().item()
    FP = ((predicted_class == 1) & (label == 0)).sum().item()
    TN = ((predicted_class == 0) & (label == 0)).sum().item()
    FN = ((predicted_class == 0) & (label == 1)).sum().item()
    total_TP += TP
    total_FP += FP
    total_TN += TN
    total_FN += FN

# Calculate the aggregated FPR and FNR after all batches
FPR = total_FP / (total_FP + total_TN) if (total_FP + total_TN) > 0 else 0
FNR = total_FN / (total_FN + total_TP) if (total_FN + total_TP) > 0 else 0
J = 1-(FPR+FNR)

features_np = np.stack(features).squeeze()
probs_np = np.stack(probabilities).squeeze()
pred_np = np.stack(pred).squeeze()
test_accuracy = (pred_np == test_labels.astype('uint')).sum() / float(test_labels.shape[0])

fpr, tpr, thresholds = roc_curve(test_labels, probs_np[:, 1])
auc = roc_auc_score(test_labels, probs_np[:, 1])

pickle.dump([fpr, tpr, thresholds, auc], open("./results/ROC_ULP_N{}.pkl".format(N), "wb"))