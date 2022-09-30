from ast import arg
import torch
import numpy as np
import random
import scipy.io
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.distributions import Normal,Bernoulli
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
from torch import optim
from torch import nn
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import time
import sys
sys.path.append('../')
from src.precondition import Pred
from src.tasks import BayesianLR
from src import pfg
import argparse
from torch_two_sample.statistics_diff import MMDStatistic


parser = argparse.ArgumentParser('Bayesian Logistic Regression')
parser.add_argument(
    '--data', type=str, default='sonar_scale.txt'
)
parser.add_argument(
    '--gt', type=int, default=1,
) # provide ground truth if possible
parser.add_argument(
    '--posterior_sample', type=str, default='sonar_sample.npy'
)
parser.add_argument(
    '--posterior_mean', type=str, default='sonar_mean.npy'
)

parser.add_argument('--hdim', type=int, default=32)
parser.add_argument('--inner_iter', type=int, default=5)
parser.add_argument('--num_particles', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--iteration', type=int, default=10000)
parser.add_argument('--warmup_iteration', type=int, default=10)

parser.add_argument('--f0_coef', type=float, default=0.0)
parser.add_argument('--exp_alpha', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_f', type=float, default=1e-3)
parser.add_argument('--sigma0', type=float, default=1)

parser.add_argument('--adam', type=bool, default=0)



args = parser.parse_args()




mmd = MMDStatistic(200,4000)


if args.gt:
    mean = np.load("../data/"+args.posterior_mean)
    sample = torch.from_numpy(np.load("../data/"+args.posterior_sample)).float()
else:
    mean,sample = None,None



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




    
sigma0 = args.sigma0

lr = args.lr_f
lr0 = args.lr
h = args.hdim
N = args.inner_iter
num_particles = args.num_particles



from scipy.stats import gaussian_kde

def elbo(trc_em_sample,em_sample, em_kde_distrib, targ):
    kl = np.log(em_kde_distrib(em_sample.transpose())).mean()
    #print('log posterior density',kl)
    kl -= targ.log_prob(trc_em_sample).mean().item()
    return kl




def main():
    # Prepare data
    l1 = []
    l2 = []
    l3 = []
    X,y = load_svmlight_file("../data/"+args.data)
    #print(y)
    #raise
    #scipy.io.loadmat('data/covertype.mat')
    #X, y = data['covtype'][:, 1:], data['covtype'][:, 0]
    y[y == -1] = 0  # y in {1,2} -> y in {1,0}
    X = csr_matrix.toarray(X)
    X = np.hstack([X, np.ones([X.shape[0], 1])])  # add constant

    n_features = X.shape[1]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

    batch_size = args.batch_size
    if batch_size>= X_train.shape[0]:
        print('Batch size > # of training data, full gradient is used.')
        batch_size = X_train.shape[0]

    

    activation = nn.Sigmoid()#nn.Softsign()#nn.Tanh()

    net = nn.Sequential(nn.Linear(n_features,h),activation,nn.Linear(h,h),activation,nn.Linear(h,n_features)).to(device)
    #net = nn.Linear(n_features+1,n_features+1).to(device)




    # random initialization (expectation of alpha is 0.01)
    theta = torch.zeros([num_particles, n_features], device=device).normal_(0, sigma0**0.5)

    model = BayesianLR(X_train, y_train, batch_size, num_particles,sigma = args.sigma0,device=device)
    if not args.adam:
        op1 = Pred([theta], lr=lr0,exp_alpha=args.exp_alpha)
    else:
        op1 = Adam([theta], lr=lr0,betas = (0,0.999))
    op2 = optim.SGD(net.parameters(), lr=lr, momentum=0.9,nesterov=1)
    trainer = pfg.PFG(model,net,op1,op2,dim = X.shape[1],f0_coef = args.f0_coef, inner_iter = args.inner_iter, H = None, exact_div = False)
 

    for _ in range(args.warmup_iteration):
        trainer.score_step(theta)
    trainer.optim2 = optim.SGD(net.parameters(), lr=lr, momentum=0.9,nesterov=1)

    ITERATION = args.iteration + 1

    t1=time.time()
    for epoch in range(ITERATION+1):

        trainer.step(theta)
             
        if epoch % 100 == 0:
            t2 = time.time()
            X_np = theta.cpu().numpy()
            if args.gt:
                l1.append(mmd(theta,sample,[10**(-i) for i in range(4)]).log().item())
                l2.append(np.sum(X_np.mean(0)-mean)**2)
                l3.append(elbo(theta,X_np,gaussian_kde(X_np.T,1),model))
                print(epoch,l1[-1],l2[-1],l3[-1])
            else:
                l3.append(elbo(theta,X_np,gaussian_kde(X_np.T,1),model))
                print(epoch,l3[-1])
            
            t1 = t2

if __name__ == '__main__':
    main()
