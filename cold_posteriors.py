import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import Classifier_ConvBBB
from utils import *
from datamodules import MiraBestDataModule

import csv
import numpy as np
import os
#%% check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)
#%%
config_dict, config = parse_config('config1.txt')

#%%
#prior
prior = config_dict['priors']['prior']
prior_var = torch.tensor([float(i) for i in config_dict['priors']['prior_init'].split(',')])

#training 

#imsize         = config_dict['training']['imsize']
epochs         = config_dict['training']['epochs']


hidden_size    = config_dict['training']['hidden_size']
out_ch         = config_dict['training']['num_classes'] #previously nclass
learning_rate  = torch.tensor(config_dict['training']['lr0']) # Initial learning rate {1e-3, 1e-4, 1e-5} -- use larger LR with reduction = 'sum' 
momentum       = torch.tensor(config_dict['training']['momentum'])
weight_decay   = torch.tensor(config_dict['training']['decay'])
reduction      = config_dict['training']['reduction']
burnin         = config_dict['training']['burnin']
T              = config_dict['training']['temp']
kernel_size    = config_dict['training']['kernel_size']
pac            = config_dict['training']['pac']
input_ch = 1
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
#%% list of temp values
temp_list = {0:5e-1, 1:1e-1, 2:5e-2, 3:1e-2, 4:5e-3, 5:1e-3, 6:5e-4, 7:1e-4, 8: 5e-5, 9:1e-5, 10:2e-1, 11:2e-2,12:2e-3, 13:2e-4, 14:2e-5, 15:1}

#%% multiple runs saved in csv files
for i in range (16):#16 for all temp values (1, 5):
    
    T =  temp_list[i] #1e-2
    print("Temperature = ", T)
    
    model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
    #model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)#, weight_decay = 1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum = 0.9)

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor=0.95, patience=3, verbose=False)

    epochs = 500
    
    epoch_trainaccs, epoch_testaccs = [], []
    epoch_trainloss, epoch_testloss = [], []
    
    epoch_trainloss_complexity, epoch_testloss_complexity = [], []
    epoch_trainloss_loglike, epoch_testloss_loglike = [], []
    
    epoch_trainloss_complexity_conv, epoch_testloss_complexity_conv = [], []
    epoch_trainloss_complexity_linear, epoch_testloss_complexity_linear = [], []
    
    
    epoch_testerr = []
    epoch_trainerr = []
    
    #early_stopping = True
    _bestacc = 0.
    
    rows = ['epoch', 'epoch_trainloss', 'epoch_trainloss_loglike', 'epoch_trainloss_complexity', 'epoch_trainerr','epoch_testloss', 'epoch_testloss_loglike', 'epoch_testloss_complexity', 'epoch_testerr']
    csvfile = "model_"+ str(i) +".csv"
    
    with open(csvfile, 'w+', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(rows)
            
    for epoch in range(epochs):
    
        train_loss, train_loss_c, train_loss_l, train_accs, train_complexity_conv, train_complexity_linear = train(model, train_loader, optimizer, device, T, burnin, reduction)
        
        test_loss, test_loss_c, test_loss_l, test_accs, test_complexity_conv, test_complexity_linear = validate(model, validation_loader, device, T, burnin, reduction, epoch, prior, prior_var)
  
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
        
        scheduler.step(epoch_testloss_loglike[-1])
        

        accuracy = epoch_testaccs[-1]
        # check early stopping criteria:
        if early_stopping and accuracy>_bestacc:
            _bestacc = accuracy
            torch.save(model.state_dict(), "model.pt")
            torch.save(model.state_dict(), "model"+str(i)+".pt")
            torch.save(optimizer.state_dict(), "model_optim" +str(i)+".pt")
            best_acc = accuracy
            best_epoch = epoch
        
        #create output row:
        _results = [epoch, epoch_trainloss[epoch], epoch_trainloss_loglike[epoch], epoch_trainloss_complexity[epoch], epoch_trainerr[epoch], epoch_testloss[epoch], epoch_testloss_loglike[epoch], epoch_testloss_complexity[epoch], epoch_testerr[epoch]]
        with open(csvfile, 'a', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)
        
        #see how a single param of the n/w changes
        #w_mu = (model.h1.w_mu.to(device=torch.device(device)).detach().numpy().flatten())[10000]
        #w_rho = np.exp((model.h1.w_rho.to(device=torch.device(device)).detach().numpy().flatten())[10000])
        
    print('Finished Training')
    print("Final validation error: ",100.*(1 - epoch_testaccs[-1]))
    
    if early_stopping:
        print("Best validation error: ",100.*(1 - best_acc)," @ epoch: "+str(best_epoch))
    
    if not early_stopping: 
        torch.save(model.state_dict(), "model.pt") 
        
    os.remove("./model.pt")
#%% testing

#model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)


model.load_state_dict(torch.load("model.pt"))
#model2_lenet_corrected_pruned
#model.load_state_dict(torch.load("model3_lenet_corrected_pruned80.pt"))
test_err = test(model, test_loader, device, T, burnin, reduction)
print(test_err)

#%%
#test_err = test(model)
#print(test_err)
err_arr = []
for i in range(100):
    test_err = test(model, test_loader, device, T, burnin, reduction)
    err_arr.append(test_err)
            
print(np.mean(err_arr))
print(np.std(err_arr))

#%%
for i in range (16):#15 for all temp values (1, 5):
    
    T =  temp_list[i] #1e-2
    print(T)
    model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

    model.load_state_dict(torch.load("model"+str(i)+".pt"))
    err_arr = []
    for j in range(100):
        test_err = test(model, test_loader, device, T, burnin, reduction)
        err_arr.append(test_err)
                
    print(np.mean(err_arr))
    print(np.std(err_arr))

#%%