from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from PIL import Image
import mirabest
#config_dict, config = parse_config('config1.txt')

#data
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((datamean ,), (datastd,))])

class MiraBestDataModule():
    
    def __init__(self, config_dict, config):
        
        self.batch_size     = config_dict['training']['batch_size']
        self.validation_split = config_dict['training']['frac_val']
        
        
        self.dataset = config_dict['data']['dataset']
        self.path = Path(config_dict['data']['datadir'])
        self.datamean = config_dict['data']['datamean']
        self.datastd = config_dict['data']['datastd']
        self.augment = config_dict['data']['augment'] #more useful to define while calling train/val loader?
        self.imsize = config_dict['training']['imsize']
    
        
    def transforms(self, aug):
        
        if(aug == 'False'):
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((self.datamean ,), (self.datastd,))])

        else:
            #crop, pad(reflect), rotate, to tensor, normalise
            #change transform to transform_aug for the training and validation set only:train_data_confident
            crop     = transforms.CenterCrop(self.imsize)
            pad      = transforms.Pad((0, 0, 1, 1), fill=0)
            
            
            transform = transforms.Compose([crop,pad,
                                            transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
                                            transforms.ToTensor(),
                                            transforms.Normalize((self.datamean ,), (self.datastd,)),
                                            ])
            
        return transform
    
    def train_val_loader(self):
        
        transform = self.transforms(self.augment)
        
        if(self.dataset =='MBFRConf'):
            train_data_confident = mirabest.MBFRConfident(self.path, train=True,
                                                          transform=transform, target_transform=None,
                                                          download=True)
            train_data_conf = train_data_confident
        
        
        elif(self.dataset == 'MBFRConf+Uncert'):
            train_data_confident = mirabest.MBFRConfident(self.path, train=True,
                             transform=transform, target_transform=None,
                             download=True)
            
            train_data_uncertain = mirabest.MBFRUncertain(self.path, train=True,
                             transform=transform, target_transform=None,
                             download=True)
            
            #concatenate datasets
            train_data_conf= torch.utils.data.ConcatDataset([train_data_confident, train_data_uncertain])
            
                
        #train-valid
        dataset_size = len(train_data_conf)
        indices = list(range(dataset_size))
        split = int(dataset_size*0.2) #int(np.floor(validation_split * dataset_size))
        shuffle_dataset = True
        random_seed = 15
        
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
    
        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
         
        train_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=self.batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(train_data_conf, batch_size=self.batch_size, sampler=valid_sampler)
        
        return train_loader, validation_loader, train_sampler, valid_sampler

    def test_loader(self):           
        #no augmentation for test_loader
        
        transform = self.transforms(aug=False)
        
        if(self.dataset =='MBFRConf'):
 
            test_data_confident = mirabest.MBFRConfident(self.path, train=False,
                                                         transform=transform, target_transform=None,
                                                         download=False)
            test_data_conf = test_data_confident

            
        elif(self.dataset == 'MBFRConf+Uncert'):

            #confident
            test_data_confident = mirabest.MBFRConfident(self.path, train=False,
                             transform=transform, target_transform=None,
                             download=True)
            
            #uncertain
            test_data_uncertain = mirabest.MBFRUncertain(self.path, train=False,
                             transform=transform, target_transform=None,
                             download=True)
            
            #concatenate datasets
            test_data_conf = torch.utils.data.ConcatDataset([test_data_confident, test_data_uncertain])
            
        test_loader = torch.utils.data.DataLoader(dataset=test_data_conf, batch_size=self.batch_size,shuffle=True)
        
        return test_loader

  
    
    
    
    
                
        