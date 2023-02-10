import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torchsummary import summary
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiplicativeLR, StepLR
import pickle
import torch.nn.utils.prune as prune
import os
import csv
import pandas as pd
from tabulate import tabulate


from priors import GaussianPrior, GMMPrior
from models import Classifier_BBB, Classifier_ConvBBB
import mirabest
from uncertainty import entropy_MI, overlapping, GMM_logits
from utils import *

from pathlib import Path
from datamodules import MiraBestDataModule


#%%
#vars = parse_args()
config_dict, config = parse_config('config1.txt')
print(config_dict, config)
#%%
#prior
prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])

#training 

#imsize         = config_dict['training']['imsize']
epochs         = config_dict['training']['epochs']


hidden_size    = config_dict['training']['hidden_size']
nclass         = config_dict['training']['num_classes']
learning_rate  = torch.tensor(config_dict['training']['lr0']) # Initial learning rate {1e-3, 1e-4, 1e-5} -- use larger LR with reduction = 'sum' 
momentum       = torch.tensor(config_dict['training']['momentum'])
weight_decay   = torch.tensor(config_dict['training']['decay'])
reduction      = config_dict['training']['reduction']
burnin         = config_dict['training']['burnin']
T              = config_dict['training']['temp']
kernel_size    = config_dict['training']['kernel_size']
pac            = config_dict['training']['pac']

base           = config_dict['model']['base']
early_stopping = config_dict['model']['early_stopping']


#output
filename = config_dict['output']['filename_uncert']
test_data_uncert = config_dict['output']['test_data']
pruning_ = config_dict['output']['pruning']

#%% load data
datamodule = MiraBestDataModule(config_dict, config)
train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
test_loader = datamodule.test_loader()


num_batches_train = len(train_loader)
num_batches_valid = len(validation_loader)
num_batches_test = len(test_loader)
print(num_batches_train,num_batches_valid, num_batches_test)

#%% check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)
#%%

input_ch = 1
out_ch = nclass #y.view(-1)
kernel_size = kernel_size
#model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

#%%
model.load_state_dict(torch.load("model.pt"))
test_err= test(model, test_loader, device, T, burnin, reduction, pac)
print(test_err)  
#%% 
{'MBFRConfident', 'MBFRUncertain', 'MBHybrid'} 
test_data_uncert = 'MBFRConfident'
csvfile = './exp1/model_uncert.csv'


#%%
rows = ['index', 'target', 'entropy', 'entropy_singlepass' , 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','cov0_00','cov0_01','cov0_11','cov1_00','cov1_01','cov1_11', 'data type', 'label', 'pruning']
                        
with open(csvfile, 'w+', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)
#%%
#pruning_        
path = './dataMiraBest'
uncert(model, test_data_uncert, device, T, burnin, reduction, csvfile, pruning_, path)
#%%
csvfile = "./exp1/model_uncert.csv"
data = pd.read_csv(csvfile)

#%%
plt.hist(data["entropy"], color= data["data type"])
#%%
x = "data type" #{"pruning", "label","target", "data type", "type"}
hue = "label" #{"label", "data type", "type", "target"}
#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="entropy",data=data, hue = hue,  palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="var_logits_0",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="entropy_singlepass",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="mutual info",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="softmax_eta",data=data, hue=hue,palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="logits_eta",data=data,  palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="cov0_00",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="cov0_01",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="cov0_11",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="cov1_00",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="cov1_01",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
figure = plt.figure(dpi= 200)
ax = sns.boxplot( x = x,y="cov1_11",data=data, hue = hue, palette="Set3")
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax = sns.swarmplot(x="target", y="logits_eta", data=data, color=".25", size = 3)
