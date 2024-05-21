
import torch
from pathlib import Path
from datamodules import MiraBestDataModule
from models import Classifier_BBB, Classifier_ConvBBB
import utils
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *

#vars = parse_args()
config_dict, config = utils.parse_config('config_lap.txt')
print(config_dict, config)

prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])


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
out_ch = nclass
kernel_size = kernel_size

n_samples = 200 #1000


temp_list = {0:5e-1, 1:1e-1, 2:5e-2, 3:1e-2, 4:5e-3, 5:1e-3, 6:5e-4, 7:1e-4, 8: 5e-5, 9:1e-5, 10:2e-1, 11:2e-2,12:2e-3, 13:2e-4, 14:2e-5, 15:1}

model_path = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps_final/nonlinear_shuffle/laplace/vi_'

# for getting random weight posterior samples
for i in np.arange(1, 11, 1): #1, 17
    model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
    file_path = model_path + str(i) + '/'
    model.load_state_dict(torch.load(file_path + 'model.pt', map_location=torch.device(device) ))
    T = temp_list[3]
    err_arr = []
    for k in range(10):
        print("temp:", T)
        test_err = test(model, test_loader, device, T, burnin, reduction, pac)
        err_arr.append(test_err)
    print(np.mean(err_arr))
    print(np.std(err_arr))

    conv1, conv2, conv3, conv4, fc1, fc2, out = posterior_samples_random(model, n_samples = n_samples, n_params = 50, log_space = False)
    
    np.save(file_path + 'vi_conv1_weights.npy', conv1)
    np.save(file_path + 'vi_conv2_weights.npy', conv2)
    np.save(file_path + 'vi_conv3_weights.npy', conv3)
    np.save(file_path + 'vi_conv4_weights.npy', conv4)
    np.save(file_path + 'vi_fc1_weights.npy', fc1)
    np.save(file_path + 'vi_fc2_weights.npy', fc2)
    np.save(file_path + 'vi_out_weights.npy', fc2)

# #for getting last layer weight posterior samples
# for i in np.arange(1, 2, 1):    
#     model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
#     file_path = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_' + str(i) + '/'
#     model.load_state_dict(torch.load(file_path + 'model.pt', map_location=torch.device(device) ))

#     err_arr = []
#     for k in range(10):
#         test_err = test(model, test_loader, device, T, burnin, reduction, pac)
#         err_arr.append(test_err)
#     print(np.mean(err_arr))
#     print(np.std(err_arr))

#     sample = posterior_samples(model, n_samples = n_samples, n_params = 168, log_space = False)
#     np.save(file_path + 'vi_' + str(i) +'_1000samples_lap_l7_weights.npy', sample)