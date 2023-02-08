import math
import torch





def sample_rademacher_like(y):
    """Rademacher like random variables.

    Args:
        y: pytorch array 
    Returns:
        pytorch array whose elements are Rademacher random variable $u_i$, such that $P(u_i=1) = P(u_i=-1) = 1/2$. The shape is the same as y.
    """    
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    """Gaussian like random variables.

    Args:
        y: pytorch array 
    Returns:
        pytorch array whose elements are gaussian random variable $u_i$, such that $u_i \sim \mathcal{N}(0,1)$. The shape is the same as y.
    """    
    return torch.randn_like(y)


def divergence_approx(fy, y, e=None):
    """Divergence approximation for function $f$ at $y$. Using randomized approximation $e^\top df(y)/dy e$. 

    Args:
        f: differentiable function
        y: the point 
        e: the distribution to be used in trace approximation (default: Rademacher distribution)
    Returns:
        Approximated divergence trace(f(y)/dy)
    """   

    if e is None:
        e = sample_rademacher_like(y)
    e_dzdx = torch.autograd.grad(fy, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_bf(fy, y):
    """Exact divergence computation for function f at y.

    Args:
        fy: f(y), output of a differentiable function f at y
        y: the input y 
    Returns:
        Divergence trace(f(y)/dy)
    """   


    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(fy[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()








def median(tensor):
    """
    Torch version of np.median(). 
    Remark: torch.median() acts differently from np.median(). We want to simulate numpy implementation.

    Args:
        tensor: torch tensor
    Returns:
        median of tensor (equivalent to np.median)
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


def kernel_rbf(inputs):
    """
    RBF kernel matrix, the bandwidth is selected as median.

    Args:
        inputs: torch tensor
    Returns:
        RBF kernel matrix
    """
    n = inputs.shape[0]
    pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    kernel_matrix = torch.exp(-pairwise_distance / h)
    return kernel_matrix


