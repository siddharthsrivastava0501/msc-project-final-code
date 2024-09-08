from torch import Tensor
import torch
import numpy as np

def sig(x, a = 1., thr = 0.):
    a, thr = torch.tensor(a), torch.tensor(thr)
    return 1 / (1 + torch.exp(-a * (x - thr)))

def pairwise_difference_matrix(x):
    '''
    Compute the difference matrix D of an input vector `x` = [x_1, ..., x_n]:
    D = [x_i - x_j] for all i,j.
    '''
    x = torch.as_tensor(x)

    x_col = x.view(-1, 1)
    x_row = x.view(1, -1)

    D = x_col - x_row

    return D.numpy()

def reshape_mlp_params(all_args, a, b):
    dim = a * b + b
    args = all_args[:dim]
    
    # The first a*b params are the weights, the rest of it is the bias
    weights = torch.cat([arg.view(-1) for arg in args[:a*b]], dim=0).view(b, a)
    biases = torch.cat([arg.view(-1) for arg in args[a*b:]], dim=0)
    
    return weights, biases, all_args[dim:]

def Se(x):
    aE = 1.3
    thrE = 4
    return sig(x, thrE, aE) - sig(0, thrE, aE)

def Si(x):
    aI = 2
    thrI = 3.7
    return sig(x, thrI, aI) - sig(0, thrI, aI)

def silu(x) -> Tensor:
    return x / (1 + torch.exp(-x))

def tanh(x) -> Tensor:
    return 2*sig(2*x) - 1

def softplus(x) -> Tensor:
    return torch.log(1 + torch.exp(x))

def dEdt(Ei, Ii, E_ext, ai = 10., bi = 12., P = 0.2, tau_E = 1., act = Se, G = 0.8, r = 0.) -> Tensor:
    de = (-Ei + (1 - r*Ei)*act(ai*Ei - bi*Ii + P + G*E_ext)) / tau_E
    return de

def dIdt(Ei, Ii, I_ext, ci = 9., di = 3., Q = 0.2, tau_I = 2., act = Si, G = 0.8, r =  0) -> Tensor:
    di = (-Ii + (1 - r*Ii)*act(ci*Ei - di*Ii + Q - G*I_ext)) / tau_I
    return di


def h_dXdt(Xt, Yt, a, omega, X_ext, G = 0.8):
    return (a - Xt**2 - Yt**2) * Xt - omega * Yt + G*X_ext

def h_dYdt(Xt, Yt, a, omega, Y_ext, G = 0.8):
    return (a - Xt**2 - Yt**2) * Yt + omega * Xt + G*Y_ext
