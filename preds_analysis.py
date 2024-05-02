
import torch
from pathlib import Path
from datamodules import MiraBestDataModule
from models import Classifier_BBB, Classifier_ConvBBB
import utils
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *

from pred import credible_interval, uncert_vi
#vars = parse_args()
config_dict, config = utils.parse_config('config1.txt')
print(config_dict, config)

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
path_out = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps_final/nonlinear_shuffle/gaussian/augment/vi_'
# file_path = model_path + str(4) + '/'


model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
# model.load_state_dict(torch.load(file_path+"model.pt"))
# test_err= test(model, test_loader, device, T, burnin, reduction, pac)

test_data_uncert = 'MBFRConfident'
print(test_data_uncert)
mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)


for i in range(10):
    model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
    model_path = path_out + str(i+1)
    model.load_state_dict(torch.load(model_path+'/model.pt', map_location=torch.device(device) ))

    mean_expected_error[i], std_expected_error[i], uce_pe[i], uce_mi[i], uce_ae[i] = uncert_vi(model, test_data_uncert, device, T, burnin, reduction, path=data_path)

print("Mean of error over seeds", mean_expected_error.mean())
print("Std of error over seeds", mean_expected_error.std())

print("Mean of std over seeds", std_expected_error.mean())
print("Std of std over seeds", std_expected_error.std())


print('Mean of uce pe', uce_pe.mean())
print('Std of uce pe', uce_pe.std())

print('Mean of uce mi', uce_mi.mean())
print('Std of uce mi', uce_mi.std())

print('Mean of uce ae', uce_ae.mean())
print('Std of uce ae', uce_ae.std())

