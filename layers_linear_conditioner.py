import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from priors import GaussianPrior, GMMPrior, LaplacePrior, CauchyPrior, LaplaceMixture


#BBB Layer
class Linear_BBB(nn.Module):

    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var, prior_type):
        super().__init__()

        #set dim
        self.input_features = input_features
        self.output_features = output_features
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialize weight params
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.1, 0.1).to(self.device))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features).uniform_(0.006, 0.01).to(self.device)) 
        
        #initialize bias params
        self.b_mu =  nn.Parameter(torch.zeros(output_features).uniform_(-0.1, 0.1).to(self.device))
        self.b_rho = nn.Parameter(torch.zeros(output_features).uniform_(0.006, 0.01).to(self.device))
    
       
        # initialize prior distribution
        if(prior_type == 'Gaussian'):
            self.prior = GaussianPrior(prior_var) #1e-1
        elif(prior_type == 'GaussianMixture'):
            self.prior = GMMPrior(prior_var)
        elif(prior_type == 'Laplacian'):
            self.prior = LaplacePrior(prior_var)
        elif(prior_type == 'LaplaceMixture'):
            self.prior = LaplaceMixture(prior_var)
        elif(prior_type == 'Cauchy'):
            pass
            self.prior = CauchyPrior(prior_var)
        else:
            print("Unspecified prior")
       
        
    def forward(self, input):
        """
          Optimization process
          
        """
      
        #sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).to(self.device)
        self.w = self.w_mu + self.w_rho * w_epsilon
        
        
        #sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(self.device)
        self.b = self.b_mu + self.b_rho * b_epsilon 
        
        
        
        #record prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior) 
        
  
        self.w_post = Normal(self.w_mu.data, self.w_rho)
        self.b_post = Normal(self.b_mu.data, self.b_rho)
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        
    
        
        return F.linear(input, self.w, self.b)

class Conv_BBB(nn.Module):

    """
        Conv Layer of our BNN.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, prior_var=  [torch.tensor(1/2), torch.tensor(1e-1), torch.tensor(1e-3)], prior_type = 'Gaussian Mixture'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        #self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.w_mu = nn.Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size).uniform_(-0.1, 0.1).to(self.device))
        self.w_rho = nn.Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size).uniform_(0.006, 0.01).to(self.device)) 
        
        self.b_mu =  nn.Parameter(torch.zeros(out_channels).uniform_(-0.1, 0.1).to(self.device))
        self.b_rho = nn.Parameter(torch.zeros(out_channels).uniform_(0.006, 0.01).to(self.device))
        
        
        # initialize prior distribution
        if(prior_type == 'Gaussian'):
            self.prior = GaussianPrior(prior_var) #1e-1
        elif(prior_type == 'GaussianMixture'):
            self.prior = GMMPrior(prior_var)
        elif(prior_type == 'Laplacian'):
            self.prior = LaplacePrior(prior_var)
        elif(prior_type == 'LaplaceMixture'):
            self.prior = LaplaceMixture(prior_var)
        elif(prior_type == 'Cauchy'):
            pass
            self.prior = CauchyPrior(prior_var)
        else:
            print("Unspecified prior")
        
      
    def forward(self, input):
        r = abs(torch.randn(1)) 
        
        
        '''                
        size = self.w_mu.shape
        print(size, size[0])
        
        size_b = self.b_mu.shape
        print(size_b, size_b[0])
        '''        
        
        #sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).to(self.device)
        self.w = self.w_mu + self.w_rho * w_epsilon

        #sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(self.device)
        self.b = self.b_mu + self.b_rho* b_epsilon 
        
        #record prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior) 
       
        #record variational_posterior - log q(w|theta)
        self.w_post = Normal(self.w_mu.data, self.w_rho)
        self.b_post = Normal(self.b_mu.data, self.b_rho)
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        
        
        
        return F.conv2d(input, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)
        