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
from pfg.tasks import BayesianNN
from pfg import sampler
from pfg.precondition import Pred

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
        opt = "pred"
    else:
        opt = "adam"
    trainer = sampler.PFG(theta, args.lr, net, optim = opt, f_lr = args.lr_f, f_optim = "sgd", f0_coef = args.f0_coef, \
                            inner_iter = args.inner_iter, exact_div = False, exp_alpha = args.exp_alpha)
 
    trainer.init_approximator(model, args.warmup_iteration)


    ITERATION = args.iteration + 1

    for epoch in range(ITERATION+1):    
        trainer.compute_grad(model)
        trainer.step()
        if epoch % 100 == 0:

            rmse,log_p_data = model.test(theta, X_test, y_test)
            print("Epoch: {} RMSE: {}, LL: {}".format(epoch, rmse,log_p_data))

    rmse,log_p_data = model.test(theta, X_test, y_test)
    print("Epoch: {} RMSE: {}, LL: {}".format(epoch, rmse,log_p_data))


if __name__ == '__main__':
    main()
