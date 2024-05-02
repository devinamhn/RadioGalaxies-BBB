from sklearn.model_selection import KFold
import random
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import torch
import argparse
import configparser as ConfigParser
import ast
import numpy as np
from models import Classifier_BBB, Classifier_ConvBBB
from uncertainty import entropy_MI, overlapping, GMM_logits, calibration
import matplotlib.pyplot as plt
import csv
import mirabest
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez
from pathlib import Path

from cata2data import CataData
from mightee import MighteeZoo
from utils import Path_Handler

def credible_interval(samples, credibility):
    '''
    calculate credible interval - equi-tailed interval instead of highest density interval

    samples values and indices
    '''
    mean_samples = samples.mean()
    sorted_samples = np.sort(samples)
    lower_bound = 0.5 * (1 - credibility)
    upper_bound = 0.5 * (1 + credibility)

    index_lower = int(np.round(len(samples) * lower_bound))
    index_upper = int(np.round(len(samples) * upper_bound))

    return sorted_samples, index_lower, index_upper, mean_samples


def uncert_vi(model, test_data_uncert, device, T, burnin, reduction, path):
    test_data = test_data_uncert
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.0031 ,), (0.0350,))])
    test_data1 = test_data_uncert
    
    if(test_data_uncert == 'MBFRConfident'):
        
    #confident test set
        test_data = mirabest.MBFRConfident(path, train=False,
                            transform=transform, target_transform=None,
                            download=False)
        
        test_data1 = mirabest.MBFRConfident(path, train=False,
                            transform=None, target_transform=None,
                            download=False)
        #uncomment for test set
        indices = np.arange(0, len(test_data), 1)
    
    elif(test_data_uncert == 'MBFRUncertain'):
        # uncertain
        
        test_data = mirabest.MBFRUncertain(path, train=False,
                         transform=transform, target_transform=None,
                         download=False)
        
        test_data1 = mirabest.MBFRUncertain(path, train=False,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'MBFR_Uncert'
    elif(test_data_uncert == 'MBHybrid'):
        #hybrid
        test_data = mirabest.MBHybrid(path, train=True,
                         transform=transform, target_transform=None,
                         download=False)
        test_data1 = mirabest.MBHybrid(path, train=True,
                         transform=None, target_transform=None,
                         download=False)
        data_type = 'MBHybrid'

    elif(test_data_uncert == 'Galaxy_MNIST'):
        transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((150,150), antialias = True), 
        torchvision.transforms.Grayscale(),
        ])
        # 64 pixel images
        train_dataset = GalaxyMNISTHighrez(
            root='./dataGalaxyMNISTHighres',
            download=True,
            train=True,  # by default, or set False for test set
            transform = transform
        )

        test_dataset = GalaxyMNISTHighrez(
            root='./dataGalaxyMNISTHighres',
            download=True,
            train=False,  # by default, or set False for test set
            transform = transform
        )
        gal_mnist_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=104, shuffle = False)

        for i, (x_test_galmnist, y_test_galmnist) in enumerate(gal_mnist_test_loader):
            x_test_galmnist, y_test_galmnist = x_test_galmnist.to(device), y_test_galmnist.to(device)
            y_test_galmnist = torch.zeros(104).to(device)
            if(i==0):
                break
        test_data = x_test_galmnist

    elif(test_data_uncert == 'mightee'):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
                torchvision.transforms.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
            ]
        )
        paths = Path_Handler()._dict()
        set = 'certain'

        data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
        test_loader = DataLoader(data, batch_size=len(data))
        # for i, (x_test, y_test) in enumerate(test_loader):
        #     x_test, y_test = x_test.to(device), y_test.to(device)
        
        test_data = data

    else:
        print("Test data for uncertainty quantification misspecified")
    
    logit = True
    indices = np.arange(0, len(test_data), 1)
    num_batches_test = 1
    
    error_all = []
    entropy_all = []
    mi_all = []
    aleat_all =[]
    avg_error_mean = []
    loss_all = []
    
    for index in (indices):
        
        x = torch.unsqueeze(torch.tensor(test_data[index][0]),0)
        y = torch.unsqueeze(torch.tensor(test_data[index][1]),0)
        #print("target is",y)
        target = y.detach().numpy().flatten()[0]
        samples_iter = 200
        #for a single datapoint
        output_ = []
        logits_=[]
        y_test_all = []
        prediction = []
        errors = []
        with torch.no_grad():
       
            i = 1
            model.eval()
            for j in range(samples_iter):
                x_test, y_test = x.to(device), y.to(device)
                loss, pred, complexity_cost, likelihood_cost, conv_complexity, linear_complexity, logits = model.sample_elbo(x_test, y_test, 1, i, num_batches_test,samples_batch=len(y_test), T=T, burnin=burnin, reduction=reduction, logit= logit)
        
                softmax = torch.exp(pred)
                outputs = logits

                output_.append(softmax.cpu().detach().numpy().flatten())
                logits_.append(outputs.cpu().detach().numpy().flatten())
                

                # prediction.append(pred.cpu().detach().numpy().flatten()[0])
                y_test_all.append(y_test.cpu().detach().numpy().flatten()[0])
             
        softmax = np.array(output_)#.cpu().detach().numpy())
        y_logits = np.array(logits_)#.cpu().detach().numpy())
        

        # print(softmax.shape)

        sorted_softmax_values, lower_index, upper_index, mean_samples = credible_interval(softmax[:, 0].flatten(), 0.64)
        print("90% credible interval for FRI class softmax", sorted_softmax_values[lower_index], sorted_softmax_values[upper_index])
        
        sorted_softmax_values_fr1 = sorted_softmax_values[lower_index:upper_index]
        sorted_softmax_values_fr2 = 1 - sorted_softmax_values[lower_index:upper_index]


        softmax_mean = np.vstack((mean_samples, 1-mean_samples)).T
        softmax_credible = np.vstack((sorted_softmax_values_fr1, sorted_softmax_values_fr2)).T   

        
        entropy, mutual_info, entropy_singlepass = entropy_MI(softmax_credible, 
                                                        samples_iter= len(softmax_credible[:,0]))

        pred = np.argmax(softmax_credible, axis = 1)
        # print(y_test, pred)

        y_test_all = np.tile(target, len(softmax_credible[:,0]))

        errors =  np.mean((pred != y_test_all).astype('uint8'))

       
        pred_mean = np.argmax(softmax_mean, axis = 1)
        error_mean = (pred_mean != target)*1

        # print(error_mean)
        # print(errors)
        
        error_all.append(errors)
        avg_error_mean.append(error_mean)
        
        entropy_all.append(entropy/np.log(2))    
        mi_all.append(mutual_info/np.log(2))
        aleat_all.append(entropy_singlepass/np.log(2))    

        # if(index == 1):
        #     break
    n_bins = 8
    path_out = './'
    uce_pe  = calibration(path_out, np.array(error_all), np.array(entropy_all), n_bins, x_label = 'predictive entropy')
    # print("Predictive Entropy")
    # print("uce = ", np.round(uce, 2))

    uce_mi  = calibration(path_out, np.array(error_all), np.array(mi_all), n_bins, x_label = 'mutual information')
    # print("Mutual Information")
    # print("uce = ", np.round(uce, 2))

    
    uce_ae  = calibration(path_out, np.array(error_all), np.array(aleat_all), n_bins, x_label = 'average entropy')
    # print("Average Entropy")
    # print("uce = ", np.round(uce, 2))

    print("mean and std of error")
    print(error_all)
    print(np.mean(error_all)*100)
    print(np.std(error_all))

    print("Average of expected error")
    print((np.array(avg_error_mean)).mean())
    print((np.array(avg_error_mean)).std())

    mean_expected_error = np.array(avg_error_mean).mean()
    std_expected_error = np.array(avg_error_mean).std()

    return mean_expected_error, std_expected_error, uce_pe, uce_mi, uce_ae