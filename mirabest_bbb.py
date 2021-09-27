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
filename = config_dict['output']['filename_uncert']
test_data_uncert = config_dict['output']['test_data']
pruning_ = config_dict['output']['pruning']
#%%
input_size = imsize*imsize
hidden_size = hidden_size
output_size = nclass
batch_size= batch_size
#learning_rate = torch.tensor(1e-4) # Initial learning rate {1e-3, 1e-4, 1e-5} -- use larger LR with reduction = 'sum' 
#momentum = torch.tensor(9e-1)
burnin = burnin
T = T
reduction = reduction #{"sum", "mean"}
#for now the dropout rates are specified in models.py
#dropout_rate1= 0.0
#dropout_rate2= 0.5
#dropout_rate3= 0.5


#%% full data
'''
train_data = mirabest.MiraBest_full(path, train=True,
                 transform=None, target_transform=None,
                 download=True)

test_data = mirabest.MiraBest_full(path, train=False,
                 transform=None, target_transform=None,
                 download=True)

#full data - contains targets b/w [0, 7]
train_data_conf = mirabest.MiraBest_full(path, train=True,
                 transform=transform, target_transform=None,
                 download=False)

#full data
test_data_conf = test_data = mirabest.MiraBest_full(path, train=False,
                 transform=transform, target_transform=None,
                 download=False)

'''

#%%
path = './dataMiraBest' 
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])

#%%

if(dataset =='MBFRConf'):
    #confident
    train_data_confident = mirabest.MBFRConfident(path, train=True,
                                                  transform=transform, target_transform=None,
                                                  download=False)
    train_data_conf = train_data_confident
    
    test_data_confident = mirabest.MBFRConfident(path, train=False,
                                                 transform=transform, target_transform=None,
                                                 download=False)
    test_data_conf = test_data_confident
    #convert from PIL to torch.tensor
    #trainloader = torch.utils.data.DataLoader(dataset=train_data_conf, batch_size=batch_size,shuffle=True)

    
elif(dataset == 'MBFRConf+Uncert'):
    #confident
    train_data_confident = mirabest.MBFRConfident(path, train=True,
                     transform=transform, target_transform=None,
                     download=False)
    
    #uncertain
    train_data_uncertain = mirabest.MBFRUncertain(path, train=True,
                     transform=transform, target_transform=None,
                     download=False)
    
    #concatenate datasets
    train_data_conf= torch.utils.data.ConcatDataset([train_data_confident, train_data_uncertain])
    
        
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

    

#train valid test
dataset_size = len(train_data_conf)
indices = list(range(dataset_size))
split = int(dataset_size*0.2) #int(np.floor(validation_split * dataset_size))
shuffle_dataset = True

if shuffle_dataset :
    #np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
    
# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
    
train_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=batch_size, sampler=valid_sampler)
#convert from PIL to torch.tensor
#trainloader = torch.utils.data.DataLoader(dataset=train_data_conf, batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=batch_size,shuffle=True)



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
out_ch = nclass
kernel_size = kernel_size

model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

#model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)

print(summary(model, input_size=(1, 150, 150)))
#%%
#model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor=0.95, patience=3, verbose=True)
#%%
learning_rate = torch.tensor(5e-5) # Initial learning rate {1e-3, 1e-4, 1e-5} -- use larger LR with reduction = 'sum' 

#%% multiple runs saved in csv files
for i in range (1):#(1, 5):
    model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
    #model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
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
    csvfile = "model_GMM_exp1_"+ str(i) +".csv"
    
    with open(csvfile, 'w+', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(rows)
            
    for epoch in range(epochs):
    
        train_loss, train_loss_c, train_loss_l, train_accs, train_complexity_conv, train_complexity_linear = train(model, train_loader, optimizer, device, T, burnin, reduction)
            
        print('Epoch: {}, Train Loss: {}, Train Accuracy: {}, NLL: {}, Complexity: {}'.format(epoch, np.sum(train_loss)/len(train_sampler), np.sum(train_accs)/len(train_sampler), np.sum(train_loss_l)/len(train_sampler), np.sum(train_loss_c)/len(train_sampler)))
        
        
        test_loss, test_loss_c, test_loss_l, test_accs, test_complexity_conv, test_complexity_linear = validate(model, validation_loader, device, T, burnin, reduction, epoch, prior, prior_var)
    
        print('Epoch: {}, Test Loss: {}, Test Accuracy: {}, Test Error: {}, NLL: {}, Complexity:{}'.format(epoch, np.sum(test_loss)/len(valid_sampler), np.sum(test_accs)/len(valid_sampler), 100.*(1 - np.sum(test_accs)/len(valid_sampler)), np.sum(test_loss_l)/len(valid_sampler), np.sum(test_loss_c)/len(valid_sampler)))
        
    
        #save density and snr
        if(epoch % 100 ==0 ):
            density, db_SNR = density_snr_conv(model) #density_snr(model)            
            with open('model_'+str(i)+ 'density' + str(epoch) + '.txt', "wb") as fp:
                pickle.dump(density, fp)
            with open('model_'+str(i)+ 'snr' + str(epoch) + '.txt', "wb") as fp:
                pickle.dump(db_SNR, fp)
        
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
            torch.save(optimizer.state_dict(), "model_optim.pt")
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
        
    #os.remove("./model.pt")
#%%
print(100.*(1 - best_acc)) 
print(best_epoch)

#%%
## plots
pl.figure(dpi=200)
#pl.plot(epoch_trainloss, label='train loss')
pl.plot(epoch_testloss, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
#pl.ylim(-0.05,0.2) 
#pl.yscale('log')
#pl.axvline(13, linestyle='--', color='g',label='Early Stopping Checkpoint')
pl.xlabel('Epochs')
pl.ylabel('Loss')
pl.show()
#%%
pl.figure(dpi=200)
#pl.plot(epoch_trainloss_complexity, label='train loss')
pl.plot(epoch_testloss_complexity, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('Weighted complexity cost')
pl.show()
#%%
pl.figure(dpi=200)
#pl.plot(epoch_trainloss_complexity_conv, label='train loss')
pl.plot(epoch_testloss_complexity_conv, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('Weighted complexity cost - conv layers')
pl.show()

pl.figure(dpi=200)
#pl.plot(epoch_trainloss_complexity_linear, label='train loss')
pl.plot(epoch_testloss_complexity_linear, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('Weighted complexity cost - linear layers')
pl.show()
#%%
pl.figure(dpi=200)
pl.plot(epoch_trainloss_loglike, label='train loss')
pl.plot(epoch_testloss_loglike, label='val loss')
pl.legend(loc='upper right')
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('negative log likelihood cost')
pl.show()
#%%
pl.figure(dpi=200)
pl.plot(epoch_testerr, label = "valid error")
pl.plot((1-np.array(epoch_trainaccs))*100, label = "train error")
pl.legend(loc='upper right')
#pl.ylim(1.3,  5)
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('error(%)')
pl.show()
#%%
density, db_SNR = density_snr_conv(model)
#%%
plt.figure(dpi=200)
plt.xlabel('Weight')
plt.grid(True)
sns.set_palette("colorblind")
sns.kdeplot(density, x="weight", fill=True)
#%%
#plot density and CDF of SNR(dB)
plt.figure(dpi=300)
plt.xlabel('Signal-to-Noise (dB)')
plt.ylabel('Density')
#plt.xlim((-35,15))
plt.grid(True)
sns.set_style("whitegrid", {'axes.grid' : True})
sns.kdeplot(db_SNR, x="SNR", fill=True,alpha=0.5)
#%%
plt.figure(dpi=200)
plt.ylabel('CDF')
plt.xlabel('Signal-to-Noise')
sns.kdeplot(db_SNR, x="SNR", fill=False,alpha=0.5, cumulative=True, color= 'black')
#%%
#model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)

model.load_state_dict(torch.load("model.pt"))
test_err = test(model, test_loader, device, T, burnin, reduction)
print(test_err)  
#%%
err_arr = []
for i in range(100):
    test_err = test(model, test_loader, device, T, burnin, reduction)
    err_arr.append(test_err)
            
print(np.mean(err_arr))
print(np.std(err_arr))


#%%####### uncertainty quantification #####

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

#%%  ################ SNR pruning ##########################

#%%
def density_snr_conv(model):
    device = "cpu"
    model = model.to(device="cpu")
    weights = 0
    '''
    weights =  np.append(model.conv1.w.to(device=torch.device(device)).detach().numpy().flatten(), model.conv2.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights,model.conv3.w.to(device=torch.device(device)).detach().numpy().flatten() )
    weights = np.append(weights,model.conv4.w.to(device=torch.device(device)).detach().numpy().flatten() )

    weights = np.append(weights,model.h1.w.to(device=torch.device(device)).detach().numpy().flatten() )
    weights =  np.append(weights, model.h2.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights, model.out.w.to(device=torch.device(device)).detach().numpy().flatten())
    '''
    #get trained posterior on weights for the 2 hidden layers
    mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w,  model.out.w_mu.detach().numpy().flatten())
    '''
    mu_post_w = np.append(mu_post_w, model.conv1.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv2.w_mu.detach().numpy().flatten())
    #mu_post_w = np.append(model.conv1.w_mu.detach().numpy().flatten(),model.conv2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv3.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv4.w_mu.detach().numpy().flatten())
    '''
    rho_post_w = np.append(model.h1.w_rho.detach().numpy().flatten(), model.h2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.out.w_rho.detach().numpy().flatten())
    '''
    rho_post_w = np.append(rho_post_w, model.conv1.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv2.w_rho.detach().numpy().flatten())
    #rho_post_w = np.append(model.conv1.w_rho.detach().numpy().flatten(),model.conv2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv3.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv4.w_rho.detach().numpy().flatten())
    '''
    #convert rho to sigma
    #sigma_post_w = np.log(1+np.exp(rho_post_w))
    sigma_post_w = np.exp(rho_post_w)
    #calculate SNR = |mu_weight|/sigma_weight
    SNR = abs(mu_post_w)/sigma_post_w
    db_SNR = 10*np.log10(SNR)
    #order the weights by SNR
    sorted_SNR = np.sort(db_SNR)[::-1]
    #remove x% of weights with lowest SNR
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #weights = 1
    return weights, db_SNR
#%%

class snrPruningMethod(prune.BasePruningMethod):
    """Prune weights based on signal-to-noise ratio
    """
    #PRUNING_TYPE = 'global'
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

#%%
def pruning(model, threshold, niter):
    
    
    
    mean_arr = []
    stdev_arr = []
    softmax_olap = []
    logits_olap = []
    ent = []
    print(threshold)
    for i in threshold:
        
        model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
        #model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load("model.pt",map_location=torch.device('cpu')))
        #model = torch.load("model.pt",map_location=torch.device('cpu'))
        
        test_err = test(model,test_loader, device, T, burnin, reduction)
        
        parameters_to_prune = (
  
        (model.h1, 'w_mu'),
        (model.h2, 'w_mu'),
        (model.out, 'w_mu'),
        

        (model.h1, 'w_rho'),
        (model.h2, 'w_rho'),
        (model.out, 'w_rho')
    
        )
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method= snrPruningMethod,
            threshold = i 
        )
        
        err_arr = []
        for i in range(niter):
            test_err = test(model,test_loader, device, T, burnin, reduction)
            err_arr.append(test_err)
            
        mean_arr.append(np.mean(err_arr))
        stdev_arr.append(np.std(err_arr))
        
        print("mean_arr", mean_arr, stdev_arr)
        
        #uncertainty quantification
        #{'MBFRConfident', 'MBFRUncertain', 'MBHybrid'} 
        test_data_uncert = 'MBHybrid'       
        csvfile = './model_mb_hybrid_snr.csv'        
        rows = ['index', 'target', 'entropy', 'entropy_singlepass' , 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','cov0_00','cov0_01','cov0_11','cov1_00','cov1_01','cov1_11', 'data type', 'label', 'pruning']
                                
        with open(csvfile, 'w+', newline="") as f_out:
                writer = csv.writer(f_out, delimiter=',')
                writer.writerow(rows)
        
        #pruning_        
        uncert(model, test_data_uncert, device, T, burnin, reduction, csvfile, pruning_, path)
            
    return mean_arr, stdev_arr
    

#%%
model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
#model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("model.pt",map_location=torch.device('cpu')))
test_err = test(model,test_loader, device, T, burnin, reduction)
print(test_err)
#%%
threshold_values = [0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9]
#%%
#threshold_values = [0.4]
mean, stddev = pruning(model, threshold_values, 100)

print("Mean and stdev for 100 passes through test loop")
print(tabulate(np.column_stack((threshold_values,mean, stddev)), headers=['Threshold','Mean', 'Stdev']))
#print(tabulate(np.column_stack((threshold_values,mean, stddev, softmax_olap,logits_olap)), headers=['Threshold','Mean', 'Stdev','Softmax Overlap', 'Logits Overlap']))
#print(tabulate(np.column_stack((np.array(threshold_values)*100,1275200 - np.array(threshold_values)*1275200,mean, stddev)), headers=['Proportion removed(%)','#Weights','Mean', 'Stdev']))
#%% mean and std for SNR pruning 

#mean_murho_snr = mean
#std_murho_snr = stddev 

#%%
threshold_values = np.array(threshold_values)*100


#%% ###################### FISHER PRUNING ###################

#%%
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
optimizer.load_state_dict(torch.load("model_optim.pt"))

#%% Fisher pruning
class FisherPruningMethod(prune.BasePruningMethod):
    """Prune weights based on Fisher Information
    """

    PRUNING_TYPE = 'unstructured'
    
    def __init__(self,  threshold):
        
        self.threshold = threshold
        

    def compute_mask(self, t, default_mask):
        #---------------------------------------------------------------------------------------------------------
        
        #remove params based on FIM and mag
        
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        #print("total params", pytorch_total_params)

        total_fc_learnable = 120*1568*2 + 120+120 +84*120*2 + 84+ 84+ 84*2*2 + 2 + 2
        #print("total params in fully connected layers", total_fc_learnable)
        total_w_mus =  120*1568 + 84*120 + 84*2
        #print("total w_mus",total_w_mus)
        
        param = []
        threshold_values = self.threshold
        
        #r = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        r = 0.3 #tunable b/w (0, 1)
        print("r=",r)
        mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
        mu_post_w = np.append(mu_post_w,model.out.w_mu.detach().numpy().flatten())
        
        
        #params to remove based on magnitude and FIM
        params_to_remove_mag = threshold_values * len(mu_post_w) * (1-r)
        params_to_remove_FIM = threshold_values * len(mu_post_w) * (r)
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
        
        
#%%
def pruning(model, threshold, niter):
    
    
    
    mean_arr = []
    stdev_arr = []
    softmax_olap = []
    logits_olap = []
    ent = []
    #print(threshold)
    for i in threshold:
        
        model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
        #model = Classifier_BBB(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load("model.pt",map_location=torch.device('cpu')))
        #model = torch.load("model.pt",map_location=torch.device('cpu'))
        
        test_err = test(model,test_loader, device, T, burnin, reduction)
        
        parameters_to_prune = (
        
        

        (model.h1, 'w_mu'),
        (model.h2, 'w_mu'),
        (model.out, 'w_mu'),
       
        
        (model.h1, 'w_rho'),
        (model.h2, 'w_rho'),
        (model.out, 'w_rho')
        
    
        )
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method= FisherPruningMethod,
            threshold = i 
        )
        
        err_arr = []
        for i in range(niter):
            test_err = test(model,test_loader, device, T, burnin, reduction)
            err_arr.append(test_err)
            
        mean_arr.append(np.mean(err_arr))
        stdev_arr.append(np.std(err_arr))
        
        print("mean_arr", mean_arr, stdev_arr)
        
        #uncertainty quantification
        #{'MBFRConfident', 'MBFRUncertain', 'MBHybrid'} 
        test_data_uncert = 'MBHybrid'       
        csvfile = './model_mb_hybrid_fisher.csv'        
        rows = ['index', 'target', 'entropy', 'entropy_singlepass' , 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','cov0_00','cov0_01','cov0_11','cov1_00','cov1_01','cov1_11', 'data type', 'label', 'pruning']
                                
        with open(csvfile, 'w+', newline="") as f_out:
                writer = csv.writer(f_out, delimiter=',')
                writer.writerow(rows)
        
        #pruning_        
        uncert(model, test_data_uncert, device, T, burnin, reduction, csvfile, pruning_, path)
        #csvfile = 
        #pruning_ = "45%" #"90%"        
        #uncert(model, test_data_uncert, device, T, burnin, reduction, csvfile, pruning_, path)
        
    return mean_arr, stdev_arr

#%%
threshold_values = [0.1,0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9]
#%%
#threshold_values = [0.6]

#mean, stddev, softmax_olap, logits_olap, ent = pruning(model, threshold_values , 100)
mean, stddev = pruning(model, threshold_values, 100)

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
#%%
