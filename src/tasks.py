import torch
import numpy as np
import random
import scipy.io
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam,SGD
from torch.distributions import Normal,Bernoulli
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import time
import urllib
import io


class BayesianLR:
    def __init__(self, X_train, y_train, batch_size, num_particles, sigma=1, device='cpu'):
        # PyTorch Gamma is slightly different from numpy Gamma
        self.alpha_prior = Gamma(torch.tensor(1., device=device), torch.tensor(sigma, device=device))
        self.sigma = sigma
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles

    def log_prob(self, theta):
        model_w = theta
        # w_prior should be decided based on current alpha (not sure)
        w_prior = Normal(0, self.sigma**0.5)
        if self.batch_size>=self.X_train.shape[0]:
            X_batch = self.X_train
            y_batch = self.y_train
        else:
            random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
            X_batch = self.X_train[random_idx]
            y_batch = self.y_train[random_idx]          
            

        # Reference https://github.com/wsjeon/SVGD/blob/master/derivations/bayesian_classification.pdf
        # See why we can make use of binary classification loss to represent log_p_data
        logits = torch.matmul(X_batch, model_w.t())
        y_batch_repeat = y_batch.unsqueeze(1).repeat(1, self.num_particles)  # make y the same shape as logits
        log_p_data = Bernoulli(logits=logits).log_prob(y_batch_repeat).sum(0)
        # log_p_data = -BCEWithLogitsLoss(reduction='none')(logits, y_batch_repeat).sum(dim=0)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)

        log_p = log_p0 + log_p_data # (8) in paper
        return log_p


class BNN:
    '''
    A two-layer BNN with N(0,1) prior
    '''
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_unit=256, device='cpu') -> None:
        self.X_train =X_train
        self.y_train = y_train.squeeze()
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.hidden_unit = hidden_unit
    
    def log_prob(self, theta):
        w1, w2 = self._flatten_param(theta)
        p_w1 = Normal(torch.zeros_like(w1), torch.ones_like(w1))
        p_w2 = Normal(torch.zeros_like(w2), torch.ones_like(w2))
        y_pred = self._forward(self.X_train, w1, w2)
        py = Normal(y_pred, 1.0)
        ll = (p_w1.log_prob(w1).sum([-1,-2]) + p_w2.log_prob(w2).sum(-1) +
            py.log_prob(self.y_train[None,:]).sum(-1))
        return ll

    def _forward(self, X, w1, w2):
        z1 = torch.einsum('ij,kju->kiu', X, w1) # (num_particle, batch_size, hidden_unit)
        z1 = torch.relu(z1)
        z2 = torch.einsum('ijk,ik->ij', z1, w2)
        return z2 # (num_particle, batch_size)
    
    def get_bma(self, X, theta):
        w1, w2 = self._flatten_param(theta)
        return self._forward(X, w1, w2).mean(0)

    def _flatten_param(self, theta):
        w1_len = self.X_train.shape[1] * self.hidden_unit
        w1_shape = (self.X_train.shape[1], self.hidden_unit)
        w2_len = self.hidden_unit
        w2_shape = (self.hidden_unit,)
        return (
            theta[:, :w1_len].reshape(-1, *w1_shape),
            theta[:, w1_len:].reshape(-1, *w2_shape)
        )
    
    def get_particle_size(self):
        w1_len = self.X_train.shape[1] * self.hidden_unit
        w2_len = self.hidden_unit
        return w1_len + w2_len



def getGermanCreditData():
    url = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
               'statlog/german/german.data-numeric')
    url = urllib.request.urlopen(url)
    try:
        raw_data = url.read()
    finally:
        url.close()
    a = np.genfromtxt(io.BytesIO(raw_data), delimiter=4)[:, :25]

    # get output
    y = a[:, -1]
    y[y == 1] = 0
    y[y == 2] = 1

    # get inputs and standardise
    x = a[:, :-1]
    x = scipy.stats.zscore(x)
    x1 = np.zeros((x.shape[0], x.shape[1] + 1))
    x1[:, 0] = np.ones(x.shape[0])
    x1[:, 1:] = x
    x = np.copy(x1)
    x = x[:, 1:]
    z = np.zeros((1000, 325))
    zz = np.zeros((z.shape[0], 300))
    k = 0
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            zz[:, k] = np.transpose(x[:, i] * x[:, j])
            k += 1
    zz = np.column_stack([x, zz])
    z[:, 0] = np.ones(1000)
    z[:, 1:] = zz
    
    return z.astype('float32'), y.astype('float32')


class GermanCreditLR():
    def __init__(self, X, y, lamb=0.01):
        self._z = X
        self._y = y
        self._N = X.shape[0]
        self._lambda = lamb

    def log_prob(self, param):
        sigma = torch.exp(param[:, 0])
        beta = param[:, 1:]
        sigma_sq = sigma**2
        logit = torch.matmul(self._z, beta.T).T
        observe_log_prob = torch.distributions.Bernoulli(logits=logit).log_prob(self._y).sum(-1)
        w_log_prob = torch.distributions.Normal(torch.zeros_like(beta),
                torch.ones_like(beta) * sigma[:, None]).log_prob(beta).sum(-1)

        sigma_log_prob = torch.distributions.Gamma(1, 0.01).log_prob(sigma_sq)
        #torch.distributions.Exponential(self._lambda).log_prob(sigma_sq)
        return observe_log_prob + w_log_prob + sigma_log_prob


class ExpectedCalibrationError:
    """Returns the expected calibration error for a given bin size.
    Args:
        y_proba (tensor): Tensor containing returned class probabilities. (NxC)
        y (tensor): Tensor containing integers which corresponds to classes. (Cx1)
    Returns:
       tensor: The expected calibration error
    Code is based on https://github.com/gpleiss/temperature_scaling.
    """

    def __init__(self, n_bins=5, **kwargs):
        super().__init__(**kwargs)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, y_proba, y):
        # device = y_proba.device
        n_samples = y.size(0)
        y_conf, y_pred = y_proba.max(-1)

        # Eq. 3 from Guo paper
        ece = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            idx_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
            n_bin = sum(idx_bin).float()

            if n_bin > 0:
                acc_bin = torch.mean((y_pred[idx_bin] == y[idx_bin]).float())
                mean_conf_bin = torch.mean(y_conf[idx_bin])

                ece += n_bin * torch.abs(acc_bin - mean_conf_bin)

        return ece.item()/n_samples

class ExpectedCalibrationErrorSigmoid:
    """Returns the expected calibration error for a given bin size.
    Args:
        y_proba (tensor): Tensor containing returned class probabilities. (NxC)
        y (tensor): Tensor containing integers which corresponds to classes. (Cx1)
    Returns:
       tensor: The expected calibration error
    Code is based on https://github.com/gpleiss/temperature_scaling.
    """

    def __init__(self, n_bins=5, **kwargs):
        super().__init__(**kwargs)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, y_proba, y):
        # device = y_proba.device
        n_samples = y.size(0)
        y_pred = torch.round(y_proba)
        y_conf = y_proba
        y_conf[y_pred==0] = 1-y_conf[y_pred==0]
        #, y_pred = y_proba.max(-1)

        # Eq. 3 from Guo paper
        ece = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            idx_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
            n_bin = sum(idx_bin).float()

            if n_bin > 0:
                acc_bin = torch.mean((y_pred[idx_bin] == y[idx_bin]).float())
                mean_conf_bin = torch.mean(y_conf[idx_bin])

                ece += n_bin * torch.abs(acc_bin - mean_conf_bin)

        return ece.item()/n_samples


ece_cri = ExpectedCalibrationErrorSigmoid()

def test(theta, X_test, y_test):
    model_w = theta
    logits = torch.matmul(X_test, model_w.t())
    prob = torch.sigmoid(logits).mean(dim=1)  # Average among outputs from different network parameters(particles)
    pred = torch.round(prob)
    ll = torch.log(prob[y_test==1]).sum() + torch.log(1-prob[y_test==0]).sum()
    acc = torch.mean((pred.eq(y_test)).float())
    ece_o = ece_cri(prob,y_test)
    print("Accuracy: {}".format(acc), "NLL: {}".format(-ll/X_test.shape[0]), "ECE: {}".format(ece_o))



def test_hir(theta, X_test, y_test):
    model_w = theta[:, :-1]
    logits = torch.matmul(X_test, model_w.t())
    prob = torch.sigmoid(logits).mean(dim=1)  # Average among outputs from different network parameters(particles)
    pred = torch.round(prob)
    ll = torch.log(prob[y_test==1]).sum() + torch.log(1-prob[y_test==0]).sum()
    acc = torch.mean((pred.eq(y_test)).float())
    ece_o = ece_cri(prob,y_test)
    print("Accuracy: {}".format(acc), "NLL: {}".format(-ll/X_test.shape[0]), "ECE: {}".format(ece_o))

