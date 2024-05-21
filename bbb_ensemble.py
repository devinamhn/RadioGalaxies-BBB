import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
import wandb
from pathlib import Path
from models import Classifier_BBB, Classifier_ConvBBB
from datamodules import MiraBestDataModule

#vars = parse_args()
config_dict, config = utils.parse_config('config_augment.txt')
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
learning_rate  = torch.tensor(config_dict['training']['lr0'])
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

filename = config_dict['output']['filename_uncert']


wandb_name = 'VI ' + str(jobid)
wandb.init(
    project= "Evaluating-VI",
    config = {
        "seed": seed,
        "data_seed": data_seed,
        "learning_rate": learning_rate,
        # "weight_decay": weight_decay,
        # "factor": factor,
        # "patience": patience,
        "epochs": epochs,
        "conditioner": conditioner,
        "prior": prior,
        "prior_var": prior_var,
        "augmentation": augment,
        "temp": T,
        "optimiser": "Adam",
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
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor=0.95, patience=3, verbose=False)
    
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
    
        train_loss, train_loss_c, train_loss_l, train_accs, train_complexity_conv, 
        train_complexity_linear = utils.train(model, train_loader, optimizer, device, 
                                    T, burnin, reduction, pac)
            
        
        
        test_loss, test_loss_c, test_loss_l, test_accs, test_complexity_conv, 
        test_complexity_linear = utils.validate(model, validation_loader, device, T, burnin,
                                    reduction, epoch, prior, prior_var, pac, file_path)
    
        
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
        
        # scheduler.step(epoch_testloss_loglike[-1])
        
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
test_err= utils.test(model, test_loader, device, T, burnin, reduction, pac)

err_arr = []
for i in range(200):
    test_err = utils.test(model, test_loader, device, T, burnin, reduction, pac)
    err_arr.append(test_err)


wandb.log({"Mean test error":np.mean(err_arr), "Std test error": np.std(err_arr)})
wandb.finish()
#
#get_samples(model, n_samples = 10000, n_params = 5, log_space = False)
