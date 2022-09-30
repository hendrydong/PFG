import math
from multiprocessing.spawn import import_main_path
import numpy as np
import random
import torch
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split

from torch import nn
import torch.nn.functional as F
import argparse
from torch import optim
from torch.optim import Adam
import sys

sys.path.append('../')
from src.svgd import get_gradient
from src import pfg
from src.precondition import Pred

import pandas

parser = argparse.ArgumentParser('Bayesian Neural Network')
parser.add_argument(
    '--data', type=str, default='boston'
)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--hdim', type=int, default=32)
parser.add_argument('--inner_iter', type=int, default=1)
parser.add_argument('--num_particles', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--iteration', type=int, default=20000)
parser.add_argument('--warmup_iteration', type=int, default=100)

parser.add_argument('--f0_coef', type=float, default=0.1)
parser.add_argument('--exp_alpha', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_f', type=float, default=1e-3)
parser.add_argument('--H_coef', type=float, default=0.1)
parser.add_argument('--sigma0', type=float, default=1)

parser.add_argument('--adam', type=bool, default=0)


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigma0 = args.sigma0

lr = args.lr_f
lr0 = args.lr
h = args.hdim
N = args.inner_iter
num_particles = args.num_particles

data_name = args.data



class BayesianNN:
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
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
        # log_gamma, log_lambda = theta[-2], theta[-1]

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = F.relu(torch.bmm(inputs, w1) + b1)
        out = torch.bmm(inter, w2) + b2
        out = out.squeeze()
        return out

    def log_prob(self, theta):
        model_gamma = torch.exp(theta[:, -2])
        model_lambda = torch.exp(theta[:, -1])
        model_w = theta[:, :-2]
        # w_prior should be decided based on current lambda (not sure)
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


def test(model, theta, X_test, y_test):
    prob = model.forward(X_test, theta)
    y_pred = prob.mean(dim=0)  # Average among outputs from different network parameters(particles)

    model_gamma = torch.exp(theta[:, -2])

    prob = []
    
    model_gamma_repeat = model_gamma.unsqueeze(1).repeat(1, y_pred.shape[0])
    distribution = Normal(y_pred, torch.sqrt(torch.ones_like(model_gamma_repeat) / model_gamma_repeat))
    log_p_data = ((distribution.log_prob(y_test).exp().mean(0)).log()).mean()


    rmse = torch.norm(y_pred - y_test) / math.sqrt(y_test.shape[0])

    print("RMSE: {}, LL: {}".format(rmse,log_p_data))


def main():
    if data_name =='boston':
        data = np.loadtxt('../data/boston_housing')
    elif data_name =='concrete':
        data = pandas.read_excel('../data/Concrete_Data.xls').values
    elif data_name =='energy':
        data = pandas.read_excel("../data/ENB2012_data.xlsx").values
        data = data[:, :-1]
    elif data_name =='naval':
        data = pandas.read_fwf('../data/UCI CBM Dataset/data.txt', header=None).values
        data = data[:, :-1]
    elif data_name=="wine_red":
        data = pandas.read_csv("../data/winequality-red.csv", delimiter=';').values
    elif data_name=="wine_white":
        data = pandas.read_csv("../data/winequality-white.csv", delimiter=';').values
    elif data_name=="yacht":
        data = pandas.read_csv("../data/yacht_hydrodynamics.data",delim_whitespace=True).values
    elif data_name=="year":
        data = pandas.read_csv("../data/YearPredictionMSD.txt",delimiter=',').values
    elif data_name=="protein":
        data = pandas.read_csv("../data/CASP.csv").values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)
    X, y = data[:, :-1], data[:, -1]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    # Normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)
    X_train_mean, X_train_std = torch.mean(X_train, dim=0), torch.std(X_train, dim=0)
    y_train_mean, y_train_std = torch.mean(y_train, dim=0), torch.std(y_train, dim=0)
    X_train_std[X_train_std==0]=1
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std


    num_particles, batch_size, hidden_dim = 100, 200, 50

    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)

    # Random initialization (based on expectation of gamma distribution)
    theta = torch.cat(
        [torch.zeros([num_particles, (X.shape[1] + 2) * hidden_dim + 1], device=device).normal_(0, math.sqrt(10)),
         torch.log(0.1 * torch.ones([num_particles, 2], device=device))], dim=1)

    n_features = theta.shape[1]
    activation = nn.Sigmoid()  # nn.Softsign()#nn.Tanh()
    net = nn.Sequential(nn.Linear(n_features, h), activation, nn.Linear(h, h),
                        activation, nn.Linear(h, n_features)).to(device)
    if not args.adam:
        op1 = Pred([theta], lr=lr0, exp_alpha=args.exp_alpha)
    else:
        op1 = Adam([theta], lr=lr0, betas=(0, 0.999))
    
    op2 = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=1)
    trainer = pfg.PFG(
        model, net, op1, op2, dim=theta.shape[1], f0_coef=args.f0_coef,
        inner_iter=args.inner_iter, H=None, exact_div=False)

    for _ in range(args.warmup_iteration):
        trainer.score_step(theta)
    trainer.optim2 = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=1)

    ITERATION = args.iteration + 1

    for epoch in range(ITERATION+1):    
        trainer.step(theta)
        trainer.H = op1.avg**args.H_coef
        if epoch % 100 == 0:

            test(model, theta, X_test, y_test)


    test(model, theta, X_test, y_test)


if __name__ == '__main__':
    main()
