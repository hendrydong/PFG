"""
The Bayesian sampling tasks are defined here, such as Bayesian Logistic Regression, Bayesian Neural Networks.
"""


from turtle import forward
import torch
import numpy as np
import random
from torch.distributions import Normal,Bernoulli
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import math

class tasks:
    """
    Implement Large Scale (Bayesian) Sampling Tasks.

    Args:
        X_train: input (feature) tensor
        y_train: output (label) tensor
        batch_size: batch size when querying likelihood 
        num_particles: the number of particle samples
    """
    def __init__(self,X_train, y_train, batch_size, num_particles, **kwargs) -> None:
        pass

    def log_prob(self, theta):
        raise NotImplementedError




class BayesianLR:
    """
    Implement Bayesian Logistic Regression (Gaussian Prior).

    Args:
        X_train: input (feature) tensor
        y_train: output (label) tensor
        batch_size: batch size when querying likelihood 
        num_particles: the number of particle samples
        sigma: std of the prior distribution
    """
    def __init__(self, X_train, y_train, batch_size, num_particles, sigma=1, device='cpu'):

        # PyTorch Gamma is slightly different from numpy Gamma
        self.sigma = sigma
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles

    def log_prob(self, theta):
        model_w = theta
        # w_prior should be decided based on current alpha
        w_prior = Normal(0, self.sigma**0.5)
        if self.batch_size>=self.X_train.shape[0]:
            X_batch = self.X_train
            y_batch = self.y_train
        else:
            random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
            X_batch = self.X_train[random_idx]
            y_batch = self.y_train[random_idx]          
            
        logits = torch.matmul(X_batch, model_w.t())
        y_batch_repeat = y_batch.unsqueeze(1).repeat(1, self.num_particles)  # make y the same shape as logits
        log_p_data = Bernoulli(logits=logits).log_prob(y_batch_repeat).sum(0)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)

        log_p = log_p0 + log_p_data 
        return log_p









class BayesianNN:
    """
    Implement Bayesian Neural Networks (MLP).

    Args:
        X_train: input (feature) tensor
        y_train: output (label) tensor
        batch_size: batch size when querying likelihood 
        num_particles: the number of particle samples
        hidden_dim: the number of MLP hidden dimension 
    """
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim, device='cpu'):

        self.gamma_prior = Gamma(torch.tensor(1., device=device),
                                 torch.tensor(1 / 0.1, device=device))
        self.lambda_prior = Gamma(torch.tensor(1., device=device),
                                  torch.tensor(1 / 0.1, device=device))
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1]
        self.hidden_dim = hidden_dim

    def forward(self, inputs, theta):
        # Unpack theta
        w1 = theta[:, 0:self.n_features *
                   self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features *
                   self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2)
                   * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -3].reshape(-1, 1, 1)

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = F.relu(torch.bmm(inputs, w1) + b1)
        out = torch.bmm(inter, w2) + b2
        out = out.squeeze()
        return out

    def log_prob(self, theta):
        if type(theta)==list:
            theta = theta[0]
        model_gamma = torch.exp(theta[:, -2])
        model_lambda = torch.exp(theta[:, -1])
        model_w = theta[:, :-2]
        w_prior = Normal(0, torch.sqrt(torch.ones_like(model_lambda) / model_lambda))

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs = self.forward(X_batch, theta)  # [num_particles, batch_size]
        model_gamma_repeat = model_gamma.unsqueeze(1).repeat(1, self.batch_size)
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)
        distribution = Normal(outputs, torch.sqrt(
            torch.ones_like(model_gamma_repeat) / model_gamma_repeat))
        log_p_data = distribution.log_prob(y_batch_repeat).sum(dim=1)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0) + self.gamma_prior.log_prob(
            model_gamma) + self.lambda_prior.log_prob(model_lambda)
        log_p = log_p0 + log_p_data * (self.X_train.shape[0] / self.batch_size)  # (8) in paper
        return log_p

    def test(self, theta, X_test, y_test):
        prob = self.forward(X_test, theta)
        y_pred = prob.mean(dim=0)  # Average among outputs from different network parameters(particles)

        model_gamma = torch.exp(theta[:, -2])

        prob = []
        
        model_gamma_repeat = model_gamma.unsqueeze(1).repeat(1, y_pred.shape[0])
        distribution = Normal(y_pred, torch.sqrt(torch.ones_like(model_gamma_repeat) / model_gamma_repeat))
        log_p_data = ((distribution.log_prob(y_test).exp().mean(0)).log()).mean()


        rmse = torch.norm(y_pred - y_test) / math.sqrt(y_test.shape[0])

        return rmse,log_p_data