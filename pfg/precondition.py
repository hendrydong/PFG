import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np


class Pred(Optimizer):
    """
    Implement Preconditioned algorithm with Pytorch Optimizer class.
    The main difference of Pred and Adam/Adagrad is that Pred use a constant Hessian estimator at each step,
    while Adam/Adagrad estimate Hessian for each input x.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups or tensor
        lr (float): learning rate
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        exp_alpha (float): the exponent of Hessian (diagonal) estimator to perform preconditioning
        alpha (float): decay coefficient of approximator output.
    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, centered=False, exp_alpha=0.5):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered, exp_alpha=exp_alpha)
        super(Pred, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(Pred, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, lr=None):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1
                if group['exp_alpha']>0:
                    # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                    square_avg.mul_(alpha).addcmul_( d_p, d_p,value=1-alpha)
                    
                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.mul_(alpha).add_(1-alpha, d_p)
                        avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                    else:
                        avg = square_avg.sqrt().add_(group['eps'])
                    
                    
                    avg = torch.mean(avg,0).reshape(1,-1)


                    p.data.addcdiv_(d_p, avg**(group['exp_alpha']*2),value=-group['lr'])
                    self.avg = avg
                else:
                    p.data.add_(d_p,alpha=-group['lr'])

                

        return loss




Adam = torch.optim.Adam # support Adam algorithm in torch

