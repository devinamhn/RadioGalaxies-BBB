import sys
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
import wandb

from pathlib import Path
from datamodules import MiraBestDataModule
from ivon._ivon import IVON


def train_ivon(model, train_loader, optimizer, device, T, burnin, reduction, pac):

    train_loss, train_accs=[],[]; acc = 0
    train_loss_c, train_loss_l = [],[]
    
    trainloss_c_conv, trainloss_c_linear = [], []
    
    num_batches_train = len(train_loader)
    for batch, (x_train, y_train) in enumerate(train_loader):
        model.train()

        x_train, y_train = x_train.to(device), y_train.to(device)
        model.zero_grad()

        with optimizer.sampled_params(train=True):
            #conv
            loss, pred, complexity_cost, likelihood_cost, conv_complexity, linear_complexity = model.sample_elbo(x_train, y_train, 1, batch, num_batches_train, samples_batch=len(y_train), T=T, burnin=burnin, reduction=reduction)
            #mlp
            #loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_train, y_train, 1, batch, num_batches_train, samples_batch=len(y_train), T=T, burnin=burnin, reduction=reduction)

            train_loss.append(loss.item()*len(y_train))
            
            #mlp and conv
            train_loss_c.append(complexity_cost.item()*len(y_train))
            train_loss_l.append(likelihood_cost.item()*len(y_train))
            
            #conv
            trainloss_c_conv.append(conv_complexity.item()*len(y_train))
            trainloss_c_linear.append(linear_complexity.item()*len(y_train))
            
            acc = (pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
            train_accs.append(acc.mean().item()*len(y_train))
            loss.backward()
            
      
        
        optimizer.step()
      
    return train_loss, train_loss_c, train_loss_l, train_accs,  trainloss_c_conv,  trainloss_c_linear

def validate_ivon(model, validation_loader, device, T, burnin, reduction, epoch, prior, prior_var, pac, file_path):
    #conv
    num_batches_valid = len(validation_loader)
    input_ch = 1
    out_ch = 2
    kernel_size = 5
    
    '''
    #mlp
    input_size = 150*150
    hidden_size = 200#800
    output_size = 2
    imsize=150
    '''
    #load model checkpoint
    if(epoch==0):
        pass 
    else:
        #mlp
        #model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)
        #conv
        model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
        model.load_state_dict(torch.load(file_path + "model.pt"))
        
    with torch.no_grad():
        
        test_loss, test_accs = [], []; acc = 0
        test_loss_c, test_loss_l = [], []
        
        testloss_c_conv, testloss_c_linear = [], []
        for i, (x_test, y_test) in enumerate(validation_loader):
                
            model.eval()
            x_test, y_test = x_test.to(device), y_test.to(device)
            with optimizer.sampled_params(train=False):
                #conv
                loss, pred, complexity_cost, likelihood_cost, conv_complexity, linear_complexity = model.sample_elbo(x_test, y_test, 1, i, num_batches_valid, samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction)
                #mlp
                #loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_test, y_test, 1, i, num_batches_valid, samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction)
                
                acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean() #why did i do pred.mean(dim=0)?
                
                test_loss.append(loss.item()*len(y_test))
                test_loss_c.append(complexity_cost.item()*len(y_test))
                test_loss_l.append(likelihood_cost.item()*len(y_test))
                
                #conv only
                testloss_c_conv.append(conv_complexity.item()*len(y_test))
                testloss_c_linear.append(linear_complexity.item()*len(y_test))

                test_accs.append(acc.mean().item()*len(y_test))
                
     
        return test_loss, test_loss_c, test_loss_l, test_accs, testloss_c_conv,  testloss_c_linear
    

#vars = parse_args()
config_dict, config = parse_config('config_ngd.txt')
jobid = int(sys.argv[1])

seed = config_dict['training']['seed'] + jobid
data_seed = config_dict['training']['seed_data'] + jobid

torch.manual_seed(seed)

#prior
prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])[1]

augment = config_dict['data']['augment']
#training 

#imsize         = config_dict['training']['imsize']
epochs         = config_dict['training']['epochs']
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
conditioner    = config_dict['model']['conditioner']

path_out = config_dict['output']['path_out']


#output
file_path = path_out + str(jobid) + '/'
# temp_list = {0:5e-1, 1:1e-1, 2:5e-2, 3:1e-2, 4:5e-3, 5:1e-3, 6:5e-4, 7:1e-4, 8: 5e-5, 9:1e-5, 10:2e-1, 11:2e-2,12:2e-3, 13:2e-4, 14:2e-5, 15:1}
# temp_index = int(jobid-1)
# T = temp_list[temp_index]

filename = config_dict['output']['filename_uncert']
# test_data_uncert = config_dict['output']['test_data']
# pruning_ = config_dict['output']['pruning']
epochs = 3000
learning_rate = 5e-5
weight_decay = 0 #1e-5 #used instead of specifying prior
momentum = 0.9
# beta_2 = 1 - 1e-5
# print_mod = 11
# factor = config_dict['training']['factor']
# patience = config_dict['training']['patience']
h0 = 0.01
train_samples = 1
training_data_len = 584
ess_factor = 100

lr_final = 0
warmup = 5

wandb_name = 'IVON' + str(jobid)
wandb.init(
    project= "Evaluating-VI",
    config = {
        "seed": seed,
        "data_seed": data_seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        # "factor": factor,
        # "patience": patience,
        "epochs": epochs,
        "conditioner": conditioner,
        "prior": prior,
        "prior_var": prior_var,
        "augmentation": augment,
        "temp": T,
        "optimiser": "iVON",
        "h0": h0,
        "train_samples": train_samples,
        "ess_factor": ess_factor
    },
    name=wandb_name,
)


#load data
datamodule = MiraBestDataModule(config_dict, config, data_seed)
train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
test_loader = datamodule.test_loader()

#check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# print("Device: ",device)

input_ch = 1
out_ch = nclass #y.view(-1)
kernel_size = kernel_size

model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

for i in range (1):
    model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor=0.95, patience=3, verbose=False)
    optimizer = IVON(model.parameters(), lr=learning_rate, ess=training_data_len*ess_factor, weight_decay=weight_decay, beta1=momentum, hess_init=h0)
    
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #             optimizer,
    #             start_factor=1.0 / warmup,
    #             end_factor=1.0,
    #             total_iters=warmup)
            
    epoch_trainaccs, epoch_testaccs = [], []
    epoch_trainloss, epoch_testloss = [], []
    
    epoch_trainloss_complexity, epoch_testloss_complexity = [], []
    epoch_trainloss_loglike, epoch_testloss_loglike = [], []
    
    epoch_trainloss_complexity_conv, epoch_testloss_complexity_conv = [], []
    epoch_trainloss_complexity_linear, epoch_testloss_complexity_linear = [], []
    
    
    epoch_testerr = []
    epoch_trainerr = []
    
    _bestacc = 0.
            
    for epoch in range(epochs):

        # if(epoch == warmup):
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, eta_min=lr_final, T_max=epochs
        #     )
    
        train_loss, train_loss_c, train_loss_l, train_accs, train_complexity_conv, train_complexity_linear = train_ivon(model, train_loader, optimizer, device, T, burnin, reduction, pac)
            
        # print('Epoch: {}, Train Loss: {}, Train Accuracy: {}, NLL: {}, Complexity: {}'.format(epoch, np.sum(train_loss)/len(train_sampler), np.sum(train_accs)/len(train_sampler), np.sum(train_loss_l)/len(train_sampler), np.sum(train_loss_c)/len(train_sampler)))
        
        
        test_loss, test_loss_c, test_loss_l, test_accs, test_complexity_conv, test_complexity_linear = validate_ivon(model, validation_loader, device, T, burnin, reduction, epoch, prior, prior_var, pac, file_path)
    
        # print('Epoch: {}, Test Loss: {}, Test Accuracy: {}, Test Error: {}, NLL: {}, Complexity:{}'.format(epoch, np.sum(test_loss)/len(valid_sampler), np.sum(test_accs)/len(valid_sampler), 100.*(1 - np.sum(test_accs)/len(valid_sampler)), np.sum(test_loss_l)/len(valid_sampler), np.sum(test_loss_c)/len(valid_sampler)))
        
        epoch_trainaccs.append(np.sum(train_accs)/len(train_sampler))
        epoch_testaccs.append(np.sum(test_accs)/len(valid_sampler))
        epoch_trainerr.append(100.*(1 - np.sum(train_accs)/len(train_sampler)))
        epoch_testerr.append(100.*(1 - np.sum(test_accs)/len(valid_sampler)))
        
        
        epoch_trainloss.append(np.sum(train_loss)/len(train_sampler))
        epoch_testloss.append(np.sum(test_loss)/len(valid_sampler))
        
        epoch_trainloss_complexity.append(np.sum(train_loss_c)/len(train_sampler))
        epoch_trainloss_loglike.append(np.sum(train_loss_l)/len(train_sampler))
    
        epoch_testloss_complexity.append(np.sum(test_loss_c)/len(valid_sampler))
        epoch_testloss_loglike.append(np.sum(test_loss_l)/len(valid_sampler))
        
        epoch_trainloss_complexity_conv.append(np.sum(train_complexity_conv)/len(train_sampler))
        epoch_trainloss_complexity_linear.append(np.sum(train_complexity_linear)/len(train_sampler))
        epoch_testloss_complexity_conv.append(np.sum(test_complexity_conv)/len(valid_sampler))
        epoch_testloss_complexity_linear.append(np.sum(test_complexity_linear)/len(valid_sampler))
        
        # scheduler.step()
        
        accuracy = epoch_testaccs[-1]
        
        # check early stopping criteria:
        if early_stopping and accuracy>_bestacc:
            _bestacc = accuracy
            torch.save(model.state_dict(), file_path + "model.pt")
            torch.save(model.state_dict(), file_path + "model"+str(i)+".pt")
            torch.save(optimizer.state_dict(), file_path + "model_optim.pt")
            best_acc = accuracy
            best_epoch = epoch
        
        wandb.log({"train_loss":epoch_trainloss[epoch], 
                    "train_loglikelihood": epoch_trainloss_loglike[epoch],
                    "train_complexity": epoch_trainloss_complexity[epoch],
                    "train_error": epoch_trainerr[epoch],
                    "val_loss": epoch_testloss[epoch], 
                    "val_loglikelihood": epoch_testloss_loglike[epoch],
                    "val_complexity": epoch_testloss_complexity[epoch],
                    "val_error": epoch_testerr[epoch]
        })

    print('Finished Training')
    print("Final validation error: ",100.*(1 - epoch_testaccs[-1]))
    
    if early_stopping:
        print("Best validation error: ",100.*(1 - best_acc)," @ epoch: "+str(best_epoch))
    
    if not early_stopping: 
        torch.save(model.state_dict(), file_path + "model.pt") 
        


# print(100.*(1 - best_acc)) 
# print(best_epoch)
# best_verr = 100-best_acc
# wandb.log({"best_vloss_epoch": best_epoch, "best_vloss": best_vloss})
wandb.log({"best_err_epoch": best_epoch, "best_err": 100.*(1 - best_acc)})
#calculate test error 
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

model.load_state_dict(torch.load(file_path+"model.pt"))
test_err= test(model, test_loader, device, T, burnin, reduction, pac)

err_arr = []
for i in range(200):
    test_err = test(model, test_loader, device, T, burnin, reduction, pac)
    err_arr.append(test_err)


wandb.log({"Mean test error":np.mean(err_arr), "Std test error": np.std(err_arr)})
wandb.finish()
#
#get_samples(model, n_samples = 10000, n_params = 5, log_space = False)
