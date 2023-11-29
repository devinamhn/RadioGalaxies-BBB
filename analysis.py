
import torch
from pathlib import Path
from datamodules import MiraBestDataModule
from models import Classifier_BBB, Classifier_ConvBBB
import utils
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def energy_function(logits, T = 1):
    
    # print(-torch.logsumexp(pred_list_mbconf, dim = 2))
    mean_energy = torch.mean(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()
    std_energy = torch.std(-torch.logsumexp(logits, dim = 2), dim = 0).detach().numpy()

    return mean_energy

#
#vars = parse_args()
config_dict, config = utils.parse_config('config1.txt')
print(config_dict, config)
#
#prior
prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])


kernel_size    = config_dict['training']['kernel_size']

nclass         = config_dict['training']['num_classes']
reduction      = config_dict['training']['reduction']
burnin         = config_dict['training']['burnin']
T              = config_dict['training']['temp']

#output
test_data_uncert = config_dict['output']['test_data']

#load data
datamodule = MiraBestDataModule(config_dict, config)
train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
test_loader = datamodule.test_loader()


#  check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)
#

input_ch = 1
out_ch = nclass #y.view(-1)
kernel_size = kernel_size
#model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device) ))


pred_list_mbconf = utils.get_logits(model, test_data_uncert='MBFRConfident', device=device, path='./dataMiraBest')
print(pred_list_mbconf)
print(pred_list_mbconf.shape)
mean_energy_conf = energy_function(pred_list_mbconf)
print(mean_energy_conf)

pred_list_gal_mnist = utils.get_logits(model, test_data_uncert='Galaxy_MNIST', device=device, path='./dataGalaxyMNISTHighres')
print(pred_list_gal_mnist)
print(pred_list_gal_mnist.shape)
mean_energy_galmnist =energy_function(pred_list_gal_mnist)
print(mean_energy_galmnist)

energy_score_df = pd.DataFrame({'MB Conf': mean_energy_conf, 
                                'Galaxy MNIST': mean_energy_galmnist,
                                }) #, mean_enerygy_galmnist.reshape(100, 1)])
# energy_score_df['dataset'] = dataset
print(energy_score_df)
energy_score_df.to_csv('./results/ood/dropout_energy_scores.csv')

sns.kdeplot(energy_score_df) #, hue = 'dataset')
plt.xlabel('Negative Energy')
plt.savefig('./energy_kde.png')