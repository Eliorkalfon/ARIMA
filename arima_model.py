import torch
from torch import nn
import numpy as np

class Arima_model(torch.nn.Module):
    #ARIMA(0,1,1) model (Actually I transformed it to MA(1))

 def __init__(self,num_samples):
     super(Arima_model,self).__init__()
     self.theta = nn.Parameter(torch.distributions.Uniform(0, 1).sample((1,)))
     self.drift = nn.Parameter(torch.distributions.Uniform(0, 1).sample((1,)))
     self.noise = torch.zeros(1,num_samples)


 def forward(self,W):
     old_noise = torch.clone(self.noise[0,0:-1])
     old_noise.requires_grad=True
     #update the noise vector according to Arima
     self.noise[0, 1:] = (W + self.theta * self.noise[0, 0:-1] - self.drift).detach()
     out = W + self.theta * old_noise - self.drift
     return out

 def calculate_sigma_sqr(self,output):
    self.sigma_sqr = output/(self.noise.shape[1]-1)
    return self.sigma_sqr

def loss_from_max_likelihood(output):
    #According to the normal distribution of the error vector we need to maximize the sum of square of each error vec
    square_output = torch.pow(output,2)
    loss = torch.sum(square_output)
    return loss

def probabilty_estimator(errors_vec,sigma_sqr):
    #resulted likelihood of the new errors
    prob = np.exp(-loss_from_max_likelihood(errors_vec).detach().numpy()/(2*sigma_sqr))\
           /(np.power(2*np.pi,(errors_vec.shape[1])/2)*np.power(sigma_sqr,(errors_vec.shape[1])))
    return prob

def create_random_arima(num_samples = 20):
    #theta and drift were taken from uniform distribution
    errors_vec  = torch.randn(1, num_samples)
    theta = nn.Parameter(torch.distributions.Uniform(0, 1).sample((1,)))
    drift = nn.Parameter(torch.distributions.Uniform(0, 1).sample((1,)))
    print(theta,drift)
    arima_series = torch.zeros_like(errors_vec)
    for i in range(1,arima_series.shape[1]):
        arima_series[0,i] = arima_series[0,i-1]+errors_vec[0,i]-theta*errors_vec[0,i-1]+drift
    return arima_series.detach(),errors_vec.detach()