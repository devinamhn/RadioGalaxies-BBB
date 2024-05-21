import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as spy
from matplotlib import pyplot as plt
from utils import get_vi_samples
# from radiogalaxies_bnns.eval.posterior_samples.utils import get_hmc_samples, get_vi_samples, get_lla_samples, get_dropout_samples
# from radiogalaxies_bnns.eval.posterior_samples.kld_estimators import scipy_estimator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# path_vi_laplace_1 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_conv1_weights.npy'
# path_vi_laplace_2 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_conv2_weights.npy'
# path_vi_laplace_3 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_conv3_weights.npy'
# path_vi_laplace_4 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_conv4_weights.npy'
# path_vi_laplace_5 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_fc1_weights.npy'
# path_vi_laplace_6 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_fc2_weights.npy'
# path_vi_laplace_7 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_linear_conditioner/vi_1/vi_1_out_weights.npy'


# path_vi_laplace_1 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_conv1_weights.npy'
# path_vi_laplace_2 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_conv2_weights.npy'
# path_vi_laplace_3 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_conv3_weights.npy'
# path_vi_laplace_4 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_conv4_weights.npy'
# path_vi_laplace_5 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_fc1_weights.npy'
# path_vi_laplace_6 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_fc2_weights.npy'
# path_vi_laplace_7 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles/vi_1/vi_1_out_weights.npy'


path_vi_laplace_1 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_conv1_weights.npy'
path_vi_laplace_2 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_conv2_weights.npy'
path_vi_laplace_3 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_conv3_weights.npy'
path_vi_laplace_4 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_conv4_weights.npy'
path_vi_laplace_5 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_fc1_weights.npy'
path_vi_laplace_6 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_fc2_weights.npy'
path_vi_laplace_7 = '/share/nas2/dmohan/bbb/RadioGalaxies-BBB/exps/ensembles_sched/vi_1/vi_1_out_weights.npy'

# n_chains = 1 
# chain_index = 0
param_indices = [232274 + 1, 232274 + 14, 232274 + 36, 232274 + 39, 232274 + 41 ] #[232274 + 6, 232274 + 15, 232274 + 35, 232274 + 39, 232274 + 41 ]
num_params = len(param_indices)
# [1, 14, 36, 46, 68]
# samples_hmc, num_samples = get_hmc_samples(path_hmc, n_chains, param_indices, chain_index)
num_samples = 200 #200
index_highz = [6, 15, 35, 39, 41] #[1, 14, 36, 39, 41]  #
num_params_vi = 50

vi_samples_laplace_1 = get_vi_samples(path_vi_laplace_1, num_samples, num_params_vi, index_highz)
vi_samples_laplace_2 = get_vi_samples(path_vi_laplace_2, num_samples, num_params_vi, index_highz)
vi_samples_laplace_3 = get_vi_samples(path_vi_laplace_3, num_samples, num_params_vi, index_highz)
vi_samples_laplace_4 = get_vi_samples(path_vi_laplace_4, num_samples, num_params_vi, index_highz)
vi_samples_laplace_5 = get_vi_samples(path_vi_laplace_5, num_samples, num_params_vi, index_highz)
vi_samples_laplace_6 = get_vi_samples(path_vi_laplace_6, num_samples, num_params_vi, index_highz)
vi_samples_laplace_7 = get_vi_samples(path_vi_laplace_7, num_samples, num_params_vi, index_highz)




print(vi_samples_laplace_1.shape)
print(vi_samples_laplace_2.shape)


# cov_vi_laplace = torch.cov(torch.from_numpy(vi_samples_laplace).reshape((5, 200)))
# cov_vi_gaussian = torch.cov(torch.from_numpy(vi_samples_gaussian).reshape((5, 200)))
# cov_hmc = torch.cov(torch.from_numpy(samples_hmc).reshape((5, 200)))

# diagonal_shrinkage = cov_hmc/cov_vi_laplace

# corr_hmc = torch.corrcoef(torch.from_numpy(samples_hmc).reshape((5, 200)))
# corr_vi = torch.corrcoef(torch.from_numpy(vi_samples_laplace).reshape((5, 200)))

# # corr_scipy_vi = spy.stats.pearsonr()

layers = [vi_samples_laplace_1, vi_samples_laplace_2, vi_samples_laplace_3, vi_samples_laplace_4, vi_samples_laplace_5, vi_samples_laplace_6, vi_samples_laplace_7]
for i in range(7):
    vi_sample_laplace_df_1 = pd.DataFrame(layers[i], index = ['VI 1']*num_samples) 
    samples_list = [vi_sample_laplace_df_1]

    bnn_samples_df = pd.concat(samples_list).rename(columns=
                                    
                                    {0: r"$w_{1 \_ 6}^7$", 
                                    1: r"$w_{1 \_ 15}^7$",
                                    2: r"$w_{1 \_ 35}^7$",
                                    3: r"$w_{1 \_ 39}^7$",
                                    4: r"$w_{1 \_ 41}^7$"                                  
                                    }
                                    ).reset_index()

    g = sns.PairGrid(bnn_samples_df, hue = 'index', diag_sharey=False) #warn_singular=False
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.set(xlim=(-0.6,0.6), ylim = (-0.6,0.6))


    plt.savefig('/share/nas2/dmohan/bbb/RadioGalaxies-BBB/snspairplt_layer'+ str(i+1)+'.png')


exit()




