import math
import torch

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def divergence_approx(f, y, e=None):
    if e is None:
        e = sample_rademacher_like(y)
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_bf(dx, y):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

class PFG:
  def __init__(self, P, net, optimizer1,optimizer2, dim,f0_coef = 0, inner_iter = 1, H = None, exact_div = False,alpha = 1,device = 'cpu'):
    self.P = P
    self.net = net
    self.optim1 = optimizer1
    self.optim2 = optimizer2
    self.inner_iter = inner_iter

    self.H = H
    self.f0_coef = 0
    self.alpha = alpha
    if exact_div:
        self.div = divergence_bf
    else:
        self.div = divergence_approx
    self.f0_coef = f0_coef

  def phi(self, X):
    phi = 0
    if self.f0_coef>0:
        if self.H is None:
            phi = phi - self.stein_score(X) * self.f0_coef
        else:
            H = self.H
            H = (H/torch.sum(H)*X.shape[1])
            phi = phi -  self.stein_score(X) * self.f0_coef / H

    return (-self.alpha*self.net(X)+phi)/ X.size(0)

  def particle_step(self, X):
    self.optim1.zero_grad()
    X.grad = self.phi(X)
    self.optim1.step()

  def stein_score(self, X):
    X = X.detach().requires_grad_(True)
    log_prob = self.P.log_prob(X)
    score_func = torch.autograd.grad(log_prob.sum(), X)[0].detach()
    return score_func

  def step(self, X):
    for _ in range(self.inner_iter):
        self.score_step(X)
    self.particle_step(X)

  def score_step(self, X):
    #H = torch.std(X,0)
    
    
    
    score_func = self.stein_score(X)
    
    self.net.train()
    
    X = X.detach().requires_grad_(True)
    S = self.net(X)

    H = self.H

    self.optim2.zero_grad()
    if self.H is None:
        Q = 0.5* torch.sum(S**2)
    else:
        H = self.H
        H = (H/torch.sum(H)*X.shape[1])
        Q = 0.5* torch.sum((S*H)**2)
        #Q = 0.5*torch.trace(H.matmul(S.T).matmul(S))

    coef = 1-self.f0_coef

   
    loss = (-coef*torch.sum(score_func*S) - torch.sum(self.div(S,X)) + Q)/S.shape[0]
    loss.backward()
    self.optim2.step()
