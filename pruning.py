#%%
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

#%%
#vars = parse_args()
config_dict, config = parse_config('config1.txt')

#prior
prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])

#training 
batch_size     = config_dict['training']['batch_size']
validation_split = config_dict['training']['frac_val']
epochs         = config_dict['training']['epochs']
imsize         = config_dict['training']['imsize']
hidden_size    = config_dict['training']['hidden_size']
nclass         = config_dict['training']['num_classes']
learning_rate  = torch.tensor(config_dict['training']['lr0']) # Initial learning rate {1e-3, 1e-4, 1e-5} -- use larger LR with reduction = 'sum' 
momentum       = torch.tensor(config_dict['training']['momentum'])
weight_decay   = torch.tensor(config_dict['training']['decay'])
reduction      = config_dict['training']['reduction']
burnin         = config_dict['training']['burnin']
T              = config_dict['training']['temp']
kernel_size    = config_dict['training']['kernel_size']

base           = config_dict['model']['base']
early_stopping = config_dict['model']['early_stopping']

#data
dataset = config_dict['data']['dataset']
path = config_dict['data']['datadir']
datamean = config_dict['data']['datamean']
datastd = config_dict['data']['datastd']

#output
filename_pruning = config_dict['output']['filename_uncert']
test_data_uncert = config_dict['output']['test_data']
pruning_ = config_dict['output']['pruning']

#%%
path = './dataMiraBest' 
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])

#%%

if(dataset =='MBFRConf'):

    test_data_confident = mirabest.MBFRConfident(path, train=False,
                                                 transform=transform, target_transform=None,
                                                 download=False)
    test_data_conf = test_data_confident
    #convert from PIL to torch.tensor
    #trainloader = torch.utils.data.DataLoader(dataset=train_data_conf, batch_size=batch_size,shuffle=True)

    
elif(dataset == 'MBFRConf+Uncert'):

    #confident
    test_data_confident = mirabest.MBFRConfident(path, train=False,
                     transform=transform, target_transform=None,
                     download=False)
    
    #uncertain
    test_data_uncertain = mirabest.MBFRUncertain(path, train=False,
                     transform=transform, target_transform=None,
                     download=False)
    #concatenate datasets
    test_data_conf = torch.utils.data.ConcatDataset([test_data_confident, test_data_uncertain])
    
    
    #convert from PIL to torch.tensor
    #trainloader = torch.utils.data.DataLoader(dataset=train_data_conf, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=batch_size,shuffle=True)

    
test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=batch_size,shuffle=True)

num_batches_test = len(test_loader)
print(num_batches_test)

#%% check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)

#%%  ################ SNR pruning ##########################

#%%
def density_snr_conv(model):
    device = "cpu"
    model = model.to(device="cpu")
    
 
    #get trained posterior on weights for the fully-connected layers
    mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w,  model.out.w_mu.detach().numpy().flatten())

    rho_post_w = np.append(model.h1.w_rho.detach().numpy().flatten(), model.h2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.out.w_rho.detach().numpy().flatten())
   
    #convert rho to sigma
    sigma_post_w = np.exp(rho_post_w)
    
    #calculate SNR = |mu_weight|/sigma_weight
    SNR = abs(mu_post_w)/sigma_post_w
    db_SNR = 10*np.log10(SNR)
    
    #order the weights by SNR
    sorted_SNR = np.sort(db_SNR)[::-1]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return mu_post_w, db_SNR
#%%
class snrPruningMethod(prune.BasePruningMethod):
    """Prune weights based on signal-to-noise ratio
    """
    PRUNING_TYPE = 'unstructured'
    
    def __init__(self,  threshold):
        
        self.threshold = threshold
        

    def compute_mask(self, t, default_mask):
        
        density, db_SNR = density_snr_conv(model)
        sorted_SNR = np.sort(db_SNR)#[::-1]
        threshold_percent = self.threshold
        threshold_snr = sorted_SNR[int(len(sorted_SNR)*threshold_percent)]
        mask = default_mask.clone()
        print(threshold_percent, threshold_snr)
        mask.view(-1)[np.where(db_SNR<=threshold_snr)] = 0
   
        mask.view(-1)[((list(np.where(db_SNR<threshold_snr))[0]+198408 ), )] = 18 
        return mask
    
#%% Fisher pruning
class FisherPruningMethod(prune.BasePruningMethod):
    """Prune weights based on Fisher Information
    """

    PRUNING_TYPE = 'unstructured'
    
    def __init__(self,  threshold, r):
        
        self.threshold = threshold
        self.r = r

    def compute_mask(self, t, default_mask):
        #---------------------------------------------------------------------------------------------------------
        
        #remove params based on FIM and mag
        
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        #print("total params", pytorch_total_params)

        #total_fc_learnable = 120*1568*2 + 120+120 +84*120*2 + 84+ 84+ 84*2*2 + 2 + 2
        #print("total params in fully connected layers", total_fc_learnable)
        #total_w_mus =  120*1568 + 84*120 + 84*2
        #print("total w_mus",total_w_mus)
        
        param = []
        threshold_values = self.threshold
        
        #r = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #r = 0.3 #tunable b/w (0, 1)
        print("r=",self.r)
        mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
        mu_post_w = np.append(mu_post_w,model.out.w_mu.detach().numpy().flatten())
        
        
        #params to remove based on magnitude and FIM
        params_to_remove_mag = threshold_values * len(mu_post_w) * (1-self.r)
        params_to_remove_FIM = threshold_values * len(mu_post_w) * (self.r)
        #print("params_to_remove based on mag", params_to_remove_mag)
        #print("params_to_remove based on mag", params_to_remove_FIM)
        
        sorted_mus = np.sort(abs(mu_post_w))
        threshold_mag = sorted_mus[int(params_to_remove_mag)]
        #print("threshold_mag", threshold_mag)
        threshold_mag = abs(threshold_mag)
        
        indices_mag = np.where((mu_post_w<=threshold_mag) & (mu_post_w>=-threshold_mag))[0]
      
        #print("total params to remove based on mag", len(indices_mag))
        
        indices_FIM1 = np.where((mu_post_w>threshold_mag))[0]
        indices_FIM2 = np.where((mu_post_w<-threshold_mag))[0]
        indices_FIM = list(indices_FIM1) + list(indices_FIM2)
        #print("remianing params ", len(indices_FIM))
        
        for i in ([16,20,24]): #w_mus only    
            param = np.append(param, optimizer.state_dict()['state'][i]['exp_avg_sq'].detach().numpy().flatten())
                    
        sorted_param = np.sort(param[indices_FIM])
        
      
        threshold_FI = sorted_param[int(params_to_remove_FIM)]
        
        FIM_indices = list(np.where(param<=threshold_FI))[0]
        indices_FIM_remove = set(FIM_indices).difference(indices_mag)
      
        #print("threshold_FI_1", threshold_FI_1)
        #print("total params to remove based on mag", (indices_mag))
        #print("total params to remove based on FIM", (indices_FIM_remove))
        
        
        mask = default_mask.clone()
 
        ind = list(indices_mag) + list(indices_FIM_remove)
        #print(ind)
        mask.view(-1)[ind] = 0
        mask.view(-1)[np.array(ind)+198408] = 18
       
        return mask
        
#%% combined pruning function for SNR and fisher
def pruning(model, filename_pruning,threshold, niter, method, r = 0, uncert = False):
    
    mean_arr = []
    stdev_arr = []
    
    for i in threshold:
        
        model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
        #model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load("model.pt",map_location=torch.device('cpu')))
        
        test_err = test(model,test_loader, device, T, burnin, reduction)
        
        parameters_to_prune = (
        (model.h1, 'w_mu'),
        (model.h2, 'w_mu'),
        (model.out, 'w_mu'),
       
        
        (model.h1, 'w_rho'),
        (model.h2, 'w_rho'),
        (model.out, 'w_rho')
        )
        
        if(method == 'SNR'):
            prune.global_unstructured(
            parameters_to_prune,
            pruning_method= snrPruningMethod,
            threshold = i 
            )
        
            
        elif(method == 'Fisher'):
            prune.global_unstructured(
            parameters_to_prune,
            pruning_method= FisherPruningMethod,
            threshold = i, 
            r = r
            )
            
        else:
            print("Pruning method misspecified.")
        
        err_arr = []
        for i in range(niter):
            test_err = test(model, test_loader, device, T, burnin, reduction)
            err_arr.append(test_err)
            
        mean_arr.append(np.mean(err_arr))
        stdev_arr.append(np.std(err_arr))
        
        #print("mean_arr", mean_arr, stdev_arr)
        
        if(uncert == True):
        
            #uncertainty quantification
            
            #test_data_uncert = 'MBHybrid'       #{'MBFRConfident', 'MBFRUncertain', 'MBHybrid'} 
            csvfile = filename_pruning        
            rows = ['index', 'target', 'entropy', 'entropy_singlepass' , 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','cov0_00','cov0_01','cov0_11','cov1_00','cov1_01','cov1_11', 'data type', 'label', 'pruning']
                                    
            with open(csvfile, 'w+', newline="") as f_out:
                    writer = csv.writer(f_out, delimiter=',')
                    writer.writerow(rows)
                    
            uncert(model, test_data_uncert, device, T, burnin, reduction, csvfile, pruning_, path)
        else:
            pass
   
        
    return mean_arr, stdev_arr

#%%
#linear
input_size = imsize*imsize
hidden_size = hidden_size
output_size = nclass

#conv
input_ch = 1
out_ch = nclass
kernel_size = kernel_size

#%%
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
#model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("model.pt",map_location=torch.device('cpu')))
test_err = test(model,test_loader, device, T, burnin, reduction)
print(test_err)

#%% for fisher pruning
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
optimizer.load_state_dict(torch.load("model_optim.pt"))

#%%
threshold_values = [0.1,0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9]
#%%
mean, stddev = pruning(model, filename_pruning, threshold_values, niter=10, method = pruning_, r= 0.3,uncert = False)

print("Mean and stdev for 100 passes through test loop")
print(tabulate(np.column_stack((threshold_values,mean, stddev)), headers=['Threshold','Mean', 'Stdev']))
#%%
#fisher only == set r = 1
#mean_murho_fisher = mean
#std_murho_fisher = stddev

#mag only == set r = 0
#mean_murho_mag = mean
#std_murho_mag = stddev

#mean_r3 = mean
#std_r3= stddev
#%%
threshold_values = np.array(threshold_values)*100

#%% pruning plots
fig, ax = plt.subplots(dpi = 300, figsize=((8,4)))

pl.figure(dpi = 300, figsize=((8,4))) #8,4

c1 = 'tab:red'
c2='tab:orange'
c3 = 'green'
c4 = 'tab:green'
c5 = 'pink'
c6 = 'orange'
c7 = 'magenta'
c8 = 'yellow'
c9 = 'green'



pl.plot(threshold_values, np.array(mean_murho_snr), label = "SNR pruning", color = c1, linestyle='--')
pl.plot(threshold_values, np.array(mean_murho_fisher), label = "Fisher pruning", color = c2, linestyle='-.')
#pl.plot(threshold_values, np.array(mean_murho_mag), label = "mag+fisher; r=0", color = c3) 
pl.plot(threshold_values, np.array(mean_r3), label = "mag+fisher; r=0.3", color = c9) 


plt.axhline(y=12.80, color='black', linestyle=':', label='0% pruning')
plt.fill_between(threshold_values,12.80+2.6,12.80-2.6, alpha=0.5, color='gray')

pl.scatter(threshold_values, np.array(mean_murho_snr), s = 10, color = c1)
pl.scatter(threshold_values, np.array(mean_murho_fisher), s = 10, color = c2)
#pl.scatter(threshold_values, np.array(mean_murho_mag), s = 10, color = c3)
pl.scatter(threshold_values, np.array(mean_r3), s = 10, color = c9)


plt.errorbar(threshold_values, np.array(mean_murho_snr), std_murho_snr, linestyle='None', fmt='-', capsize = 3, color = c1)
plt.errorbar(threshold_values, np.array(mean_murho_fisher), std_murho_fisher1, linestyle='None', fmt='-', capsize = 3, color = c2)
#plt.errorbar(threshold_values, np.array(mean_murho_mag), std_murho_mag, linestyle='None', fmt='-', capsize = 3, color = c3)

plt.errorbar(threshold_values, np.array(mean_r3), std_r3, linestyle='None', fmt='-', capsize = 3, color = c9)


pl.grid(True)
#pl.title("pruning", kwargs)
pl.xlabel("percentage of parameters pruned")
pl.ylabel("test error (%)")
pl.xticks(np.arange(10, 100, 10))
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))