# Training clean models
# Architecture - Modified Resnet output classes = 200
# Dataset - tiny-imagenet

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets

import logging
import pickle
import glob
import os
import sys
from resnet import resnet18_mod
from PIL import Image
from torch.utils.data import ConcatDataset

import torch
from torch.utils import data

class ModelTransformWrapper(torch.nn.Module):
  def __init__(self, model, transform):
    super(ModelTransformWrapper, self).__init__()
    self.model = model
    self.transform = transform
    self.parameters = model.parameters

  def forward(self, x):
    return self.model.forward(self.transform(x))

def project(x, original_x, epsilon):
  max_x = original_x + epsilon
  min_x = original_x - epsilon

  x = torch.max(torch.min(x, max_x), min_x)

  return x
class LinfProjectedGradientDescendAttack:
  def __init__(self, model, loss_fn, eps, step_size, steps, random_start=True, reg=lambda: 0.0, bounds=(0.0, 1.0),
               device=None):
    self.model = model
    self.loss_fn = loss_fn

    self.eps = eps
    self.step_size = step_size
    self.bounds = bounds
    self.steps = steps

    self.random_start = random_start

    self.reg = reg

    self.device = device if device else torch.device('cpu')

  '''def perturb(self, original_x, labels, random_start=True):
      model_original_mode = self.model.training
      self.model.eval()
      if random_start:
          rand_perturb = torch.FloatTensor(original_x.shape).uniform_(-self.eps, self.eps)
          rand_perturb = rand_perturb.to(self.device)
          x = original_x + rand_perturb
          x.clamp_(self.bounds[0], self.bounds[1])
      else:
          x = original_x.clone()

      x.requires_grad = True

      with torch.enable_grad():
          for _iter in range(self.steps):
              outputs = self.model(x)

              loss = self.loss_fn(outputs, labels) + self.reg()

              grads = torch.autograd.grad(loss, x)[0]

              x.data += self.step_size * torch.sign(grads.data)

              x = project(x, original_x, self.eps)
              x.clamp_(self.bounds[0], self.bounds[1])

      self.model.train(mode=model_original_mode)
      return x'''

  def perturb(self, original_x, y, eps=None):
    if eps is not None :
      self.eps = eps
      self.step_size = 1.5 * (eps / self.steps)
    if self.random_start:
      rand_perturb = torch.FloatTensor(original_x.shape).uniform_(-self.eps, self.eps)
      rand_perturb = rand_perturb.to(self.device)
      x = original_x.detach() + rand_perturb
      x.clamp_(self.bounds[0], self.bounds[1])
    else:
      x = original_x.detach()

    for _iter in range(self.steps):
      x.requires_grad_()
      with torch.enable_grad():
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y) + self.reg()
      grads = torch.autograd.grad(loss, x)[0]
      x = x.detach() + self.step_size * torch.sign(grads.detach())
      x = project(x, original_x, self.eps)
      x.clamp_(self.bounds[0], self.bounds[1])
    return x

  def __call__(self, *args, **kwargs):
    return self.perturb(*args, **kwargs)


#logging
logfile = sys.argv[2]
if not os.path.exists(os.path.dirname(logfile)):
	os.makedirs(os.path.dirname(logfile))

eps = None
if len(sys.argv) > 3 :
	eps = float(sys.argv[3])

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
	logging.FileHandler(logfile, "w"),
	logging.StreamHandler()
])

# Setting the hyper parameters

use_cuda=True
nofclasses=200 # Tiny ImageNet
nof_epochs=100
batchsize=100


# Load clean data
DATA_ROOT="/home/berta/data/tiny-imagenet-200/"

# Data loading code
traindir = os.path.join(DATA_ROOT, 'train')
valdir = os.path.join(DATA_ROOT, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

transform_list = [ transforms.RandomCrop(56), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
		transforms.ToTensor()]
if eps is None :
	transform_list.append(normalize)

train_dataset = datasets.ImageFolder(
	traindir,
	transforms.Compose(transform_list))

train_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=batchsize, shuffle=True,
	num_workers=8, pin_memory=True)

transform_list_valid = [ transforms.CenterCrop(56), transforms.ToTensor()]
if eps is None :
	transform_list_valid.append(normalize)

val_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder(valdir, transforms.Compose(transform_list_valid)),
	batch_size=batchsize, shuffle=False,
	num_workers=8, pin_memory=True)

saveDir = './clean_models/train/clean_resnetmod_tiny-imagenet_%04d.pt'
saveDirmeta = os.path.join(os.path.dirname(saveDir), 'meta')
if not os.path.exists(os.path.dirname(saveDir)):
	os.makedirs(os.path.dirname(saveDir))

if not os.path.exists(saveDirmeta):
	os.makedirs(saveDirmeta)

crossentropy=torch.nn.CrossEntropyLoss()

count=0
clean_models = []
partition = int(sys.argv[1])
gpu="0"
runs=0
while runs<14:
	n = partition*14+runs
	val_temp=0
	train_accuracy=0
	logging.info('Training model %d'%(n))

	cnn = resnet18_mod(num_classes=200)
	if eps is not None :
		cnn = ModelTransformWrapper(model=cnn, transform=normalize)

	logging.info(cnn)
	# Compute number of parameters
	s  = sum(np.prod(list(p.size())) for p in cnn.parameters())
	print ('Number of params: %d' % s)

	if use_cuda:
		device='cuda:'+gpu
		cnn.to(device)
	else:
		device=torch.device('cpu')

	learning_rate = 0.1
	if eps is not None :
		criterion = torch.nn.CrossEntropyLoss()
		parameter_presets = {'eps': eps, 'step_size': 0.01, 'steps': 3}
		attack = LinfProjectedGradientDescendAttack(cnn, criterion, **parameter_presets, random_start=True, device=device)
		learning_rate = max(learning_rate * batchsize / 256.0, learning_rate)
	optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
	if eps is not None :
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
														total_steps=None, epochs=nof_epochs,
														steps_per_epoch=int(100000 / batchsize) + 1,
														pct_start=0.0025, anneal_strategy='cos',
														cycle_momentum=False, div_factor=1.0,
														final_div_factor=1000000.0, three_phase=False, last_epoch=-1,
														verbose=False)
	else :
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
	for epoch in range(nof_epochs):
		cnn.train()
		# adjust_learning_rate(optimizer, epoch)
		epoch_loss=list()
		epoch_acc=list()
		for i, (x, y) in enumerate(train_loader):
			if x.shape[0]==1:
				break
			x=x.to(device) # CPU or Cuda
			y=y.to(device) # CPU or Cuda
			if eps is not None :
				cnn.eval()
				x_adv = attack.perturb(x, y)
				cnn.train()
				yhat = cnn(x_adv)
			else :
				yhat = cnn(x)
			loss = crossentropy(yhat,y) # Classification loss
			if i%100==0:
				logging.info("Epoch:{}    Iter:{}/{}    Training loss: {:.3f}   Training acc: {:.2f}"
					  .format(epoch, i, len(train_loader), loss.item(), train_accuracy))
			train_pred = torch.argmax(yhat, dim=1)
			epoch_acc.append((1.*(train_pred==y)).sum().item()/float(train_pred.shape[0]))
			optimizer.zero_grad()
			loss.backward() # Backward pass
			optimizer.step() # Take a step
			# Keep track of losses
			epoch_loss.append(loss.item())
			train_accuracy = sum(epoch_acc)/len(epoch_acc)
		scheduler.step()
		with torch.no_grad():
			# Calculate validation accuracy
			acc=list()

			cnn.eval()
			for x,y in val_loader:
				x=x.to(device) # CPU or Cuda
				y=y.to(device) # CPU or Cuda
				val_pred = torch.argmax(cnn(x),dim=1)
				acc.append((1.*(val_pred==y)).sum().item()/float(val_pred.shape[0]))
			val_accuracy=sum(acc)/len(acc)
			# Save the best model on the validation set
			if val_accuracy>=val_temp:
				torch.save(cnn.state_dict(), saveDir%n)
				val_temp=val_accuracy

			logging.info("Max val acc:{:.3f}".format(val_temp))

	clean_models.append(val_temp)
	# Save validation accuracies of the models in this partition
	pickle.dump(clean_models,open(saveDirmeta + '/meta_{:02d}.pkl'.format(partition),'wb'))
	runs+=1

	torch.cuda.empty_cache()

