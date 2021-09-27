import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from layers import Linear_BBB, Conv_BBB

class Classifier_BBB(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, prior_var, prior_type, imsize):
        super().__init__()
        self.h1  = Linear_BBB(in_dim, hidden_dim, prior_var, prior_type= prior_type)
        self.h2  = Linear_BBB(hidden_dim, hidden_dim, prior_var, prior_type= prior_type)
        self.out = Linear_BBB(hidden_dim, out_dim, prior_var, prior_type= prior_type)
        self.imsize = imsize
        self.out_dim = out_dim
    
    def forward(self, x, logit= False):
        '''
        #MNIST
        x = x.view(-1, 28*28) #flatten
        
        #Mirabest
        #x = x.view(-1, 150*150)
        '''
        x = x.view(-1, self.imsize*self.imsize)
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))    
        x = self.out(x)
        logits = x
        x = F.log_softmax(x ,dim=1)
        if(logit == True):
            return x, logits
        else:
            return x
        
        return x
    
    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.out.log_prior
        
    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.out.log_post
    
    def log_like(self,outputs,target, reduction):
        #log P(D|w)
        return F.nll_loss(outputs, target, reduction=reduction)
        
    # avg cost function over no. of samples = {1, 2, 5, 10}
    def sample_elbo(self, input, target, samples, batch, num_batches, samples_batch, T=1.0, burnin=None, reduction = "sum", logit=False):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        outputs = torch.zeros(samples, target.shape[0], self.out_dim).to(self.device)
        
        log_priors = torch.zeros(samples).to(self.device)
        log_posts = torch.zeros(samples).to(self.device)
        log_likes = torch.zeros(samples).to(self.device)
       
        for i in range(samples):
            
            if(logit == True):
                outputs[i], logits = self(input, logit = True)
               
            else:
                outputs[i] = self(input, logit = False)
            
       
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = self.log_like(outputs[i,:,:], target, reduction)
           
        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
   
        if burnin=="blundell":
            frac = 2**(num_batches - (batch + 1))/2**(num_batches - 1)
        elif burnin==None:
            if reduction == "sum":
                frac = T/(num_batches) # 1./num_batches #
            elif reduction == "mean":
                frac = T/(num_batches*samples_batch)
            else:
                pass
        
        

        complexity_cost = frac*(log_post - log_prior)
        loss = complexity_cost + log_like #or likelihood_cost
        
        if(logit==True):        
            return loss, outputs, complexity_cost, log_like, logits
        else:
            return loss, outputs, complexity_cost, log_like
    

    
class Classifier_ConvBBB(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size, prior_var, prior_type):
        super().__init__()
        
        z = 0.5*(150 +1 - 2)
        z = int(0.5*(z - 2))
        
        
        self.conv1 = Conv_BBB(in_ch, 6, kernel_size, stride=1, padding = 1, prior_var = prior_var,  prior_type=prior_type)
        self.conv2 = Conv_BBB(6, 16, kernel_size, stride=1, padding = 1, prior_var = prior_var, prior_type=prior_type)
        self.conv3 = Conv_BBB(16, 26, kernel_size, stride=1, padding = 1, prior_var = prior_var, prior_type=prior_type)
        self.conv4 = Conv_BBB(26, 32, kernel_size, stride=1, padding = 1, prior_var = prior_var, prior_type=prior_type)
        self.h1  = Linear_BBB(7*7*32, 120, prior_var, prior_type)# --
        self.h2  = Linear_BBB(120, 84, prior_var, prior_type)
        self.out = Linear_BBB(84, out_ch, prior_var, prior_type)
        self.out_dim = out_ch
        
        
    def forward(self, x, logit= False): 
        
     
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        
        #print(x.shape)
        
        x = x.view(-1, 7*7*32)
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = self.out(x)
        logits = x
        x = F.log_softmax(x ,dim=1)
    
        if(logit == True):
            return x, logits
        else:
            return x
    
    def log_prior(self):
        conv_layers = self.conv1.log_prior + self.conv2.log_prior + self.conv3.log_prior + self.conv4.log_prior
        linear_layers = self.h1.log_prior + self.h2.log_prior + self.out.log_prior
        return conv_layers, linear_layers
    
    def log_post(self):
        conv_layers =  self.conv1.log_post +  self.conv2.log_post +  self.conv3.log_post  +self.conv4.log_post
        linear_layers = self.h1.log_post + self.h2.log_post + self.out.log_post
        return conv_layers, linear_layers
    
    def log_like(self,outputs,target, reduction):
        #log P(D|w)
        return F.nll_loss(outputs, target, reduction=reduction)
        
    # avg cost function over no. of samples = {1, 2, 5, 10}
    def sample_elbo(self, input, target, samples, batch, num_batches, samples_batch, T=1.0, burnin=None, reduction = "sum",logit= False):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        outputs = torch.zeros(samples, target.shape[0], self.out_dim).to(self.device)
        #pac
        #outputs1 = torch.zeros(samples, target.shape[0], self.out_dim).to(self.device)
        
        log_priors_conv = torch.zeros(samples).to(self.device)
        log_priors_linear = torch.zeros(samples).to(self.device)
        log_priors = torch.zeros(samples).to(self.device)
        
        log_posts_conv =  torch.zeros(samples).to(self.device)
        log_posts_linear =  torch.zeros(samples).to(self.device)
        log_posts = torch.zeros(samples).to(self.device)
        
        log_likes = torch.zeros(samples).to(self.device)
        #pac
        #log_likes_1 = torch.zeros(samples).to(self.device)
        
        for i in range(samples):
            
            if(logit == True):
                outputs[i], logits = self(input, logit = True)
               
            else:
                outputs[i] = self(input, logit = False)
                #pac
                #outputs1[i] = self(input, logit=False) 
                
            log_priors_conv[i],log_priors_linear[i] = self.log_prior()
            log_priors[i] = log_priors_conv[i] + log_priors_linear[i]
            
            log_posts_conv[i], log_posts_linear[i] = self.log_post()
            log_posts[i] = log_posts_conv[i] +  log_posts_linear[i]
            log_likes[i] = self.log_like(outputs[i,:,:], target, reduction)
            
            #pac
            #log_likes_1[i] = self.log_like(outputs1[i,:,:], target, reduction)
            
        
        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        
        #pac
        #log_like_1 = log_likes_1.mean()
        
        log_prior_conv = log_priors_conv.mean()
        log_prior_linear = log_priors_linear.mean()
        log_post_conv = log_posts_conv.mean()
        log_post_linear = log_posts_linear.mean()
        
     
        if burnin=="blundell":
            frac = 2**(num_batches - (batch + 1))/(2**(num_batches) - 1)
        elif burnin==None:
            if reduction == "sum":
                frac = T/num_batches
            elif reduction == "mean":
                frac = T/(num_batches*samples_batch)
            else:
                pass
   

        loss = frac*(log_post - log_prior) + log_like
        
        #print("loss before variance term", loss)
        
        complexity_cost = frac*(log_post - log_prior)
        likelihood_cost = log_like
        
        #uncomment for pac
        '''
        if(self.training == True):
            #print("training")
            variance=[]
            #replacing all tensors with np arrays
            for i in range(len(target)):
                prob = -F.nll_loss(outputs[0,:,:][i].reshape(1,2), target[i].reshape(1), reduction=reduction).detach().numpy()
                #print(prob)
                prob1 = -F.nll_loss(outputs1[0,:,:][i].reshape(1,2), target[i].reshape(1), reduction=reduction).detach().numpy()
                max_prob = np.maximum(prob, prob1)
                #alpha = np.log(np.exp(prob - max_prob) +np.exp(prob1 - max_prob)) - np.log(2)
                #h_alpha = alpha/(1-np.exp(alpha))**2 + 1/(np.exp(alpha)*(1-np.exp(alpha)))
                var = np.exp(2*prob - 2*max_prob) - np.exp(prob + prob1 - 2*max_prob)
                
                #print(var)
                #print("halpha*var", (h_alpha*var))
                variance = np.append(variance, var)
      
            loss = complexity_cost + log_like - np.sum((variance))
            #loss = complexity_cost + log_like -  np.sum(h_alpha*variance)

            #print("loss after var term", loss)
        
        else:
            pass
           
            
        '''
        
        complexity_conv = frac*(log_post_conv - log_prior_conv)
        complexity_linear = frac*(log_post_linear - log_prior_linear)
        
        if(logit==True):
            return loss, outputs, complexity_cost, likelihood_cost, complexity_conv, complexity_linear, logits
        else:
            return loss, outputs, complexity_cost, likelihood_cost, complexity_conv, complexity_linear