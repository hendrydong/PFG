import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import time
from pfg.tasks import BayesianLR
from pfg import sampler
from torch_two_sample.statistics_diff import MMDStatistic

# Define and parse the command line arguments
parser = argparse.ArgumentParser('Bayesian Logistic Regression')
parser.add_argument('--data', type=str, default='sonar_scale.txt')
parser.add_argument('--gt', type=int, default=1)
parser.add_argument('--posterior_sample', type=str, default='sonar_sample.npy')
parser.add_argument('--posterior_mean', type=str, default='sonar_mean.npy')
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

# Initialize MMD statistic class
mmd = MMDStatistic(200,4000)

# Load the ground truth mean and sample if available
if args.gt:
    mean = np.load("../data/" + args.posterior_mean)
    sample = torch.from_numpy(np.load("../data/" + args.posterior_sample)).float()
else:
    mean, sample = None, None

# Determine the device to run the code on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load other arguments
sigma0 = args.sigma0
h = args.hdim
N = args.inner
num_particles = args.num_particles



from scipy.stats import gaussian_kde

def elbo(trc_em_sample,em_sample, em_kde_distrib, targ):
    kl = np.log(em_kde_distrib(em_sample.transpose())).mean()
    kl -= targ.log_prob(trc_em_sample).mean().item()
    return kl




def main():
    # Load the data and prepares it for training.
    l1 = []
    l2 = []
    l3 = []
    X,y = load_svmlight_file("../data/"+args.data)
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


    # Define the activation function as sigmoid. Define the function class with neural network architecture
    activation = nn.Sigmoid()

    net = nn.Sequential(nn.Linear(n_features,h),activation,nn.Linear(h,h),activation,nn.Linear(h,n_features)).to(device)

    # Initialize particles and the PFG trainer.
    theta = torch.zeros([num_particles, n_features], device=device).normal_(0, sigma0**0.5)

    model = BayesianLR(X_train, y_train, batch_size, num_particles,sigma = args.sigma0,device=device)
    if not args.adam:
        opt = "pred"
    else:
        opt = "adam"
    
    trainer = sampler.PFG(theta, args.lr, net, optim = opt, f_lr = args.lr_f, f_optim = "sgd", f0_coef = args.f0_coef, \
                            inner_iter = args.inner_iter, exact_div = False, exp_alpha = args.exp_alpha)
 
    trainer.init_approximator(model, args.warmup_iteration)

    ITERATION = args.iteration + 1

    t1=time.time()
    
    # Loop through the specified number of iterations, computing the gradient and updating the parameters in each iteration.
    for epoch in range(ITERATION+1):
        trainer.compute_grad(model)
        trainer.step()
             
        if epoch % 100 == 0:
            t2 = time.time()
            X_np = theta.cpu().numpy()

            # Prints the value of ELBO (Evidence Lower Bound) for each epoch, along with logMMD and Mean distance
            if args.gt:
                l1.append(mmd(theta,sample,[10**(-i) for i in range(4)]).log().item())
                l2.append(np.sum(X_np.mean(0)-mean)**2)
                l3.append(elbo(theta,X_np,gaussian_kde(X_np.T,1),model))
                print('Epoch:%d\t logMMD:%.3f\t Mean distance:%.3f\t ELBO:%.3f\t'%(epoch,l1[-1],l2[-1],l3[-1]))
            else:
                l3.append(elbo(theta,X_np,gaussian_kde(X_np.T,1),model))
                print('Epoch:%d\t ELBO:%.3f\t'%(epoch,l3[-1]))
                
            
            t1 = t2

if __name__ == '__main__':
    main()
