import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np
from pfg.precondition import Adam, Pred
from pfg.utils import kernel_rbf, divergence_bf,divergence_approx


def get_logp_gradient(model, inputs):
    """Get $\nabla \log p$ from model with batch inputs

    Args:
        model: probablistic model with log_prob attribute
        inputs: batch input for model
    Returns:
        gradient: return the log prob gradient for inputs
    """    
    if type(inputs)==list:
        inputs = inputs[0]
    inputs = inputs.detach().requires_grad_(True)

    log_prob = model.log_prob(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs)[0]

    gradient = -log_prob_grad

    return gradient


def get_svgd_gradient(model, inputs):
    """Get functional gradient in SVGD algorithm. 

    Args:
        model: probablistic model with log_prob attribute
        inputs: batch input for model
    Returns:
        gradient: return the functional gradient for inputs (SVGD-like)
    """    
    n = inputs.size(0)
    inputs = inputs.detach().requires_grad_(True)

    log_prob = model.log_prob(inputs)
    log_prob_grad = torch.autograd.grad(log_prob.sum(), inputs)[0]

    kernel = kernel_rbf(inputs)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs)[0]

    gradient = -(kernel.mm(log_prob_grad) + kernel_grad) / n

    return gradient


class SGLD(Optimizer):
    """
    Implement SGLD with Pytorch Optimizer class.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups or tensor
        lr (float): learning rate
    """

    def __init__(self, params, lr=required):
        if type(params)==torch.FloatTensor:
            params = [params]
        defaults = dict(lr=lr)
        super(SGLD, self).__init__(params, defaults)


    def compute_grad(self, model):
        """
        Compute grad for the update.

        Args:
            model: probablistic model with log_prob attribute
        """
        self.params.p = get_logp_gradient(model, self.params)
        


    def step(self, lr=None, add_noise = True):
        """
        Performs a single optimization step.

        Args:
            lr (float): learning rate (default: None). 
                If lr is not None, the learning rate is set as lr.
            add_noise (bool): whether add Brownian motion in the update (default: True). 
                If add_noise = False, the algorithm is equivalent to SGD.
        """

        for group in self.param_groups:
            if lr:
                group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) / np.sqrt(group['lr'])
                    )
                    p.data.add_(-group['lr'],
                                d_p + langevin_noise.sample().to(p))
                else:
                    p.data.add_(-group['lr'], d_p)


    
class pSGLD(Optimizer):
    """
    Implement RMSprop-like preconditioned SGLD (pSGLD) with Pytorch Optimizer class.  

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups or tensor
        lr (float): learning rate
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, centered=False):
        if type(params)==torch.FloatTensor:
            params = [params]
        defaults = dict(lr=lr, alpha=alpha, eps=eps, centered=centered)
        super(pSGLD, self).__init__(params, defaults)
        
    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def compute_grad(self, model):
        """
        Compute grad for the update.

        Args:
            model: probablistic model with log_prob attribute
        """
        self.params.p = get_logp_gradient(model, self.params)
        


    def step(self, lr=None, add_noise = True):
        """
        Performs a single optimization step.

        Args:
            lr (float): learning rate (default: None). 
                If lr is not None, the learning rate is set as lr.
            add_noise (bool): whether add Brownian motion in the update (default: True). 
                If add_noise = False, the algorithm is equivalent to RMSProp.
        """

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
                
                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1-alpha, d_p, d_p)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1-alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    
                
                if group['addnoise']:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size).to(p),
                        torch.ones(size).to(p).div_(group['lr']).div_(avg).sqrt()
                    )
                    p.data.add_(-group['lr'],
                                d_p.div_(avg) + langevin_noise.sample())
                else:
                    p.data.addcdiv_(-group['lr'], d_p, avg)






class SVGD(Optimizer):
    """
    Implement RBF SVGD with Pytorch Optimizer class.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups or tensor
        lr (float): learning rate
        optim [adam or sgd or pred]: optimizer to update particles
    """

    def __init__(self, params, lr=required, optim="adam"):
        if type(params)==torch.FloatTensor:
            params = [params]
        assert optim in ["sgd","adam","pred"]
        defaults = dict(lr=lr)
        self.params = params
        if optim == "sgd":
            self.optim = torch.optim.SGD(params, lr=lr)
        elif optim == "adam":
            self.optim = Adam(params, lr=lr,betas = (0,0.999))
        elif optim == "pred":
            self.optim = Pred(params, lr=lr)
        else:
            raise NotImplementedError("Not supported optimizer.")
        
        super(SVGD, self).__init__(params, defaults)


    def compute_grad(self, model):
        """
        Compute grad for the update.

        Args:
            model: probablistic model with log_prob attribute
        """
        self.params.p = get_svgd_gradient(model, self.params)


    def step(self):
        """
        Performs a single optimization step.
        """
        self.optim.step()




class PFG(Optimizer):
    """
    Implement PFG with Pytorch Optimizer class.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups or tensor
        lr (float): learning rate
        approximator (nn.Module): functional gradient approximator 
        optim [adam or sgd or pred]: optimizer to update particles
        f_lr (float): learning rate to update the approximator
        f_optim [adam or sgd]: optimizer to update the approximator
        f0_coef (float): interpolation coefficient between true $\nabla \log p_*$ and the approximated one
        inner_iter (int): the number of inner loop to train the approximator
        exact_div (bool): whether to compute exact divergence (if false, the randomized estimator is used)
        exp_alpha (float): the exponent of Hessian (diagonal) estimator to perform preconditioning
        alpha (float): decay coefficient of approximator output.
    """
    def __init__(self, params, lr, approximator, optim = "pred", f_lr = 1e-3, f_optim = "adam", f0_coef = 0, inner_iter = 1, exact_div = False, exp_alpha = 0.5, alpha = 1):
        if type(params)==torch.Tensor:
            params = [params]
        if optim == "sgd":
            self.optim1 = torch.optim.SGD(params, lr=lr)
        elif optim == "adam":
            self.optim1 = Adam(params, lr=lr,betas = (0,0.999))
        elif optim == "pred":
            self.optim1 = Pred(params, lr=lr, exp_alpha = exp_alpha)
        else:
            raise NotImplementedError("Not supported optimizer.")

        if f_optim == "sgd":
            self.optim2 = torch.optim.SGD(approximator.parameters(), lr=f_lr, momentum=0.9, nesterov=1)
        elif f_optim == "adam":
            self.optim2 = Adam(approximator.parameters(), lr=f_lr)
        else:
            raise NotImplementedError("Not supported optimizer.")
        
        self.params = params
        self.approximator = approximator
        self.inner_iter = inner_iter
        self.f0_coef = 0
        self.f_optim = f_optim
        self.f_lr = f_lr
        self.alpha = alpha
        if exact_div:
            self.div = divergence_bf
        else:
            self.div = divergence_approx
        self.f0_coef = f0_coef
        defaults = dict(lr=lr, f_lr = f_lr)
        super(PFG, self).__init__(params, defaults)


    def _phi(self, model):
        phi = 0
        X = self.params
        if type(X)==list:
            X = X[0]
        if self.f0_coef>0:
            phi = phi - self.stein_score(model) * self.f0_coef
        return (-self.alpha*self.approximator(X)+phi)/ X.size(0)

    def stein_score(self, model):
        X = self.params
        if type(X)==list:
            X = X[0]
        X = X.detach().requires_grad_(True)
        log_prob = model.log_prob(X)
        score_func = torch.autograd.grad(log_prob.sum(), X)[0].detach()
        return score_func

    def compute_grad(self, model):
        """
        Compute grad for the update.

        Args:
            model: probablistic model with log_prob attribute
        """
        for _ in range(self.inner_iter):
            self.approximator_step(model)
        X = self.params
        if type(X)==list:
            X = X[0]
        X.grad = self._phi(model)


    def step(self):
        self.optim1.step()


    def init_approximator(self, model, iters):
        for _ in range(iters):
            self.approximator_step(model)
        if self.f_optim == "sgd":
            self.optim2 = torch.optim.SGD(self.approximator.parameters(), lr=self.f_lr, momentum=0.9, nesterov=1)
        elif self.f_optim == "adam":
            self.optim2 = Adam(self.approximator.parameters(), lr=self.f_lr)


    def approximator_step(self,model):
        X = self.params
        if type(X)==list:
            X = X[0]
        
        score_func = self.stein_score(model)
        self.approximator.train()

        X = X.detach().requires_grad_(True)
        S = self.approximator(X)
        self.optim2.zero_grad()
        Q = 0.5* torch.sum(S**2)
        coef = 1-self.f0_coef
        loss = (-coef*torch.sum(score_func*S) - torch.sum(self.div(S,X)) + Q)/S.shape[0]
        loss.backward()
        self.optim2.step()
