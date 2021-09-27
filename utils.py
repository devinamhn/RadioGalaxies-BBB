import torch
import argparse
import configparser as ConfigParser
import ast
import numpy as np
from models import Classifier_BBB, Classifier_ConvBBB
from uncertainty import entropy_MI, overlapping, GMM_logits
import matplotlib.pyplot as plt
import csv
import mirabest
import torchvision.transforms as transforms
import torch.nn.functional as F

#%%
def density_snr(model):
    device = "cpu"
    model = model.to(device="cpu")
    
    weights =  np.append(model.h1.w.to(device=torch.device(device)).detach().numpy().flatten(), model.h2.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights, model.out.w.to(device=torch.device(device)).detach().numpy().flatten())
    
    #get trained posterior on weights for the 2 hidden layers
    mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w,  model.out.w_mu.detach().numpy().flatten())

    rho_post_w = np.append(model.h1.w_rho.detach().numpy().flatten(), model.h2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.out.w_rho.detach().numpy().flatten())
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
    return weights, db_SNR
#%%
def density_snr_conv(model):
    device = "cpu"
    model = model.to(device="cpu")
    '''
    weights =  np.append(model.conv1.w.to(device=torch.device(device)).detach().numpy().flatten(), model.conv2.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights, model.conv3.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights, model.conv4.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights,model.h1.w.to(device=torch.device(device)).detach().numpy().flatten() )
    weights =  np.append(weights, model.h2.w.to(device=torch.device(device)).detach().numpy().flatten())
    weights = np.append(weights, model.out.w.to(device=torch.device(device)).detach().numpy().flatten())

    '''
    #get trained posterior on weights for the 2 hidden layers
    mu_post_w = np.append(model.h1.w_mu.detach().numpy().flatten(), model.h2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w,  model.out.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv1.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv2.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv3.w_mu.detach().numpy().flatten())
    mu_post_w = np.append(mu_post_w, model.conv4.w_mu.detach().numpy().flatten())
    
    rho_post_w = np.append(model.h1.w_rho.detach().numpy().flatten(), model.h2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.out.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv1.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv2.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv3.w_rho.detach().numpy().flatten())
    rho_post_w = np.append(rho_post_w, model.conv4.w_rho.detach().numpy().flatten())
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
    return mu_post_w, db_SNR

def train(model, train_loader, optimizer, device, T, burnin, reduction):

    train_loss, train_accs=[],[]; acc = 0
    train_loss_c, train_loss_l = [],[]
    
    trainloss_c_conv, trainloss_c_linear = [], []
    
    num_batches_train = len(train_loader)
    for batch, (x_train, y_train) in enumerate(train_loader):
        model.train()
        x_train, y_train = x_train.to(device), y_train.to(device)
        model.zero_grad()
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

def validate(model, validation_loader, device, T, burnin, reduction, epoch, prior, prior_var):
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
    if(epoch==0):
        pass 
    else:
        #mlp
        #model = Classifier_BBB(input_size, hidden_size, output_size, prior_var, prior, imsize).to(device)
        #conv
        model = Classifier_ConvBBB(input_ch, out_ch, kernel_size, prior_var, prior).to(device)
        model.load_state_dict(torch.load("model.pt"))
        
    with torch.no_grad():
        
        test_loss, test_accs = [], []; acc = 0
        test_loss_c, test_loss_l = [], []
        
        testloss_c_conv, testloss_c_linear = [], []
        for i, (x_test, y_test) in enumerate(validation_loader):
                
            model.eval()
            x_test, y_test = x_test.to(device), y_test.to(device)
            #conv
            loss, pred, complexity_cost, likelihood_cost, conv_complexity, linear_complexity = model.sample_elbo(x_test, y_test, 1, i, num_batches_valid, samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction)
            #mlp
            #loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_test, y_test, 1, i, num_batches_valid, samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction)
            
            acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean()
            
            test_loss.append(loss.item()*len(y_test))
            test_loss_c.append(complexity_cost.item()*len(y_test))
            test_loss_l.append(likelihood_cost.item()*len(y_test))
            
            #conv only
            testloss_c_conv.append(conv_complexity.item()*len(y_test))
            testloss_c_linear.append(linear_complexity.item()*len(y_test))

            test_accs.append(acc.mean().item()*len(y_test))
            
     
        return test_loss, test_loss_c, test_loss_l, test_accs, testloss_c_conv,  testloss_c_linear
    
def test(model, test_loader, device, T, burnin, reduction):

    
    num_batches_test = len(test_loader)
    test_sampler = 0
    with torch.no_grad():
        model.eval()        
        test_loss, test_accs = [], []; acc = 0
        test_loss_c, test_loss_l = [], []
        testloss_c_conv, testloss_c_linear = [], []
        for i, (x_test, y_test) in enumerate(test_loader):
           
            x_test, y_test = x_test.to(device), y_test.to(device)
                #samples = 5?
            #conv
            loss, pred, complexity_cost, likelihood_cost, conv_complexity, linear_complexity = model.sample_elbo(x_test, y_test, 1, i, num_batches_test,samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction)
            #mlp
            #loss, pred, complexity_cost, likelihood_cost = model.sample_elbo(x_test, y_test, 1, i, num_batches_test,samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction)
                
            acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean()
            test_loss.append(loss.item()*len(y_test))
            test_loss_c.append(complexity_cost.item()*len(y_test))
            test_loss_l.append(likelihood_cost.item()*len(y_test))
            
            #conv 
            testloss_c_conv.append(conv_complexity.item()*len(y_test))
            testloss_c_linear.append(linear_complexity.item()*len(y_test))
    
            test_accs.append(acc.mean().item()*len(y_test))
            test_sampler = test_sampler + len(y_test)

    
    
    testaccs= np.sum(test_accs)/test_sampler
    testerr = (100.*(1 - np.sum(test_accs)/test_sampler))
    testloss = np.sum(test_loss)/test_sampler
    return testerr
#%%
def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="config_bbb.txt", required=True, help='Name of the input config file')
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config
#%%
from sklearn.model_selection import KFold
import random
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

def uncert(model, test_data_uncert, device, T, burnin, reduction, csvfile, pruning_, path):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
    

    #options for test_data and test_data1
    if(test_data_uncert == 'MBFRConfident'):
        # confident
        test_data = mirabest.MBFRConfident(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
        
        test_data1 = mirabest.MBFRConfident(path, train=False,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'Conf'
    elif(test_data_uncert == 'MBFRUncertain'):
        # uncertain
        
        test_data = mirabest.MBFRUncertain(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
        
        test_data1 = mirabest.MBFRUncertain(path, train=False,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'Uncert'
    elif(test_data_uncert == 'MBHybrid'):
        #hybrid
        test_data = mirabest.MBHybrid(path, train=True,
                         transform=transform, target_transform=None,
                         download=False)
        test_data1 = mirabest.MBHybrid(path, train=True,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'Hybrid'
    else:
        print("Test data for uncertainty quantification misspecified")
    
    logit = True
    indices = np.arange(0, len(test_data), 1)
    num_batches_test = 1
    for index in (indices):
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        #print("target is",y)
        target = y.detach().numpy().flatten()[0]
        samples_iter = 200
        #for a single datapoint
        with torch.no_grad():
            output_ = []
            logits_=[]
            #accs_ =[]
            i = 1
            model.eval()
            for j in range(samples_iter):
                x_test, y_test = x.to(device), y.to(device)
                loss, pred, complexity_cost, likelihood_cost, conv_complexity, linear_complexity, logits = model.sample_elbo(x_test, y_test, 1, i, num_batches_test,samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction, logit= logit)
                #acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32)
                
                output_.append(pred)
                logits_.append(logits)
                #accs_.append(acc)
            
        softmax=[]
        logit_arr = []
        for i in range(samples_iter):
            a = np.exp(np.array(output_[i][0]))
            softmax.append(a[0])
            logit_arr.append(np.array(logits_[i][0]))
            
        x = [0, 1]
        y = softmax[0]
        x = np.tile(x, (samples_iter, 1))
        
        if(target == 0):
            #label = 'FRI'
            label = 'Conf'
        elif(target ==1):
            #label = 'FRII'
            label = 'Uncert'
        else:
            pass
        
        #plt.title("softmax probabilities")
        plt.figure(figsize= (2.6, 4.8), dpi=300)
        plt.rcParams["axes.grid"] = False
        plt.subplot((211))
        plt.scatter(x, softmax, marker='_',linewidth=1,color='b',alpha=0.5)
        plt.title("softmax outputs")
        plt.xticks(np.arange(0, 2, 1.0))
        
        plt.subplot((212))
        plt.imshow(test_data1[index][0])
        plt.axis("off")
        #label = 'target = ' + str(test_data1[index][1])
        plt.title('class'+str(target) +': '+label)
        plt.show()
        
        softmax = np.array(softmax)
        y_logits = np.array(logit_arr)
        
        mean_logits = np.mean(y_logits,axis=0)
        var_logits = np.std(y_logits,axis=0)
        print("Mean of Logits", mean_logits)
        print("Stdev pf Logits", var_logits)
        
        entropy, mutual_info, entropy_singlepass = entropy_MI(softmax, samples_iter)
    
        print("Entropy:", entropy)
        print("Mutual Information:", mutual_info)
        print("Entropy of a single pass:", entropy_singlepass)
        
        softmax_eta = overlapping(softmax[:,0], softmax[:,1])
        print("Softmax Overlap Index", softmax_eta)
        
        logits_eta = overlapping(y_logits[:,0], y_logits[:,1])
        print("Logit-Space Overlap Index", logits_eta)
        
        covs = GMM_logits(y_logits, 2)
    
        plt.figure(dpi=200)
        plt.rcParams["axes.grid"] = False
        plt.axes().set_facecolor('white')
        plt.scatter(x, y_logits, marker='_',linewidth=1,color='b',alpha=0.5)
        plt.xticks(np.arange(0, 2, 1))
        
        
        
        
        pruning_ = 'Fisher' #'Fisher' #SNR#40%'
        data_type = 'MBHybrid'
        # create output row:
        ['index', 'target', 'entropy','entropy_singlepass ', 'mutual info', 'var_logits_0','var_logits_1','softmax_eta', 'logits_eta','GMM_covs', 'data type', 'label', 'pruning']
        _results = [index, target, entropy, entropy_singlepass, mutual_info, var_logits[0],var_logits[1], softmax_eta, logits_eta,covs[0][0][0], covs[0][0][1], covs[0][1][1],covs[1][0][0],covs[1][0][1],covs[1][1][1], data_type, label, pruning_ ]
        
        with open(csvfile, 'a', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)
        
#%%
