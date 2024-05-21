""" Script for calculating energy scores for different test sets """

import torch
from pathlib import Path
from datamodules import MiraBestDataModule
from models import Classifier_BBB, Classifier_ConvBBB
import utils
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import utils

def energy_function(logits, T = 1):
    
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy


config_dict, config = utils.parse_config('config1.txt')

#
#prior
prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])[1]


kernel_size    = config_dict['training']['kernel_size']
nclass         = config_dict['training']['num_classes']
reduction      = config_dict['training']['reduction']
burnin         = config_dict['training']['burnin']
T              = config_dict['training']['temp']
pac            = config_dict['training']['pac']
#output
test_data_uncert = config_dict['output']['test_data']

#load data
datamodule = MiraBestDataModule(config_dict, config)
train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
test_loader = datamodule.test_loader()
data_path = './dataMiraBest' 

#  check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)
#

input_ch = 1
out_ch = nclass #y.view(-1)
kernel_size = kernel_size
model_path = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps_final/nonlinear_shuffle/gaussian/vi_'
file_path = model_path + str(4) + '/'

model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
model.load_state_dict(torch.load(file_path+"model.pt"))
test_err= utils.test(model, test_loader, device, T, burnin, reduction, pac)


print("MBCONF")
pred_list_mbconf, softmax = utils.get_logits(model, test_data_uncert='MBFRConfident', device=device, path='./dataMiraBest')
mean_energy_conf = energy_function(pred_list_mbconf)

print("MBUNCERT")
pred_list_mbuncert, softmax = utils.get_logits(model, test_data_uncert='MBFRUncertain', device=device, path='./dataMiraBest')
mean_energy_uncert = energy_function(pred_list_mbuncert)

print("MBHYBRID")
pred_list_mbhybrid, softmax = utils.get_logits(model, test_data_uncert='MBHybrid', device=device, path='./dataMiraBest')
mean_energy_hybrid = energy_function(pred_list_mbhybrid)

print("GalaxyMNIST")
pred_list_gal_mnist, softmax = utils.get_logits(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres')
mean_energy_galmnist =energy_function(pred_list_gal_mnist)

print("MIGHTEE")
pred_list_mightee, softmax = utils.get_logits(model, test_data_uncert='mightee', device=device, path='./')
mean_energy_mightee =energy_function(pred_list_mightee)

s1 = pd.Series(mean_energy_conf, name = 'VI MB Conf')
s2 = pd.Series(mean_energy_uncert, name = 'VI MB Uncert')
s3 = pd.Series(mean_energy_hybrid, name = 'VI MB Hybrid')
s4 = pd.Series(mean_energy_galmnist, name = 'VI Galaxy MNIST')
s5 = pd.Series(mean_energy_mightee, name = 'VI MIGHTEE')
energy_score_df = pd.DataFrame([s1, s2, s3, s4, s5]).T



energy_score_df.to_csv('./results/ood/vi_energy_scores.csv')

#load csv and make plot

# energy_score_df = pd.read_csv('./results/ood/vi_energy_scores.csv', index_col=0)

# binwidth = 1#0.10
# ylim_lower = 0
# ylim_upper = 7

# bin_lower = -30 #-80
# bin_upper = 2

# plt.clf()
# plt.figure(dpi=300)
# sns.histplot(energy_score_df[['VI MB Conf', 'VI Galaxy MNIST', 'VI MIGHTEE']], binrange = [bin_lower, bin_upper], 
#              binwidth = binwidth)
# plt.ylim(ylim_lower, ylim_upper)
# plt.xlabel(' Negative Energy')#Negative
# plt.xticks(np.arange(bin_lower, bin_upper, 5))
# plt.savefig('./results/ood/energy_hist_vi.png')

exit()