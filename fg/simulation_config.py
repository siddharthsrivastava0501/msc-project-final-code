import torch
from torch import Tensor
from .functions import dEdt, dIdt, reshape_mlp_params, Se, Si, h_dXdt, h_dYdt
import numpy as np
from torch.nn.functional import linear, leaky_relu

def _initial_C(nr):
    C = torch.empty((nr, nr)).normal_(0.2, 0.1)
    C.fill_diagonal_(0.)

    return C

def simulate_wc(config : dict) -> tuple[Tensor, Tensor]:
    '''
    Simulates the excitatory and inhibitory dynamics of Wilson-Cowan using the
    equations found here:
    https://en.wikipedia.org/wiki/Wilson–Cowan_model#Simplification_of_the_model_assuming_time_coarse_graining
    '''
    T = config.get('T', 6.)
    dt = config.get('dt', 0.01)
    nr = config.get('nr', 5)

    torch.empty((nr,)).normal_(5.)

    a = config.get('a', torch.empty((nr,)).normal_(3., 1.))
    b = config.get('b', torch.empty((nr,)).normal_(5., 1.))
    c = config.get('c', torch.empty((nr,)).normal_(4., 1.))
    d = config.get('d', torch.empty((nr,)).normal_(3., 1.))
    P = config.get('P', torch.empty((nr,)).normal_(1., 0.))
    Q = config.get('Q', torch.empty((nr,)).normal_(1., 0.))
    tauE = config.get('tauE', torch.full((nr,), 1.))
    tauI = config.get('tauI', torch.full((nr,), 2.))
    C = config.get('C', _initial_C(nr))
    dyn_noise = config.get('dyn_noise', 0.)
    obs_noise = config.get('obs_noise', 0.)
    nn_weights = config.get('nn_weights', False)
    layer_sizes = config.get('layer_sizes', [])
    E_act = config.get('E_act', Se)
    I_act = config.get('E_act', Si)

    simulation_info = (
        f"Running simulation with: "
        f"T = {T}, dt = {dt}, nr = {nr}, a = {a}, b = {b}, c = {c}, d = {d}, "
        f"P = {P}, Q = {Q}, tauE = {tauE}, tauI = {tauI}, nr = {nr}, "
        f"E_act = {E_act}, I_act = {I_act}"
    )
    print(simulation_info)

    time = torch.arange(0, T + dt, dt)
    E = np.zeros((len(time), nr))
    I = np.zeros((len(time), nr))

    E[0] = 0.0
    I[0] = 0.0

    for t in range(len(time) - 1):
        E_input = np.dot(C, E[t])
        I_input = np.dot(C, I[t])

        for r in range(nr):
            E[t+1, r] = E[t, r] + dt*(dEdt(E[t, r], I[t, r], E_input[r], a[r], b[r], P[r], tauE[r], E_act) + np.random.normal(0, dyn_noise))
            I[t+1, r] = I[t, r] + dt*(dIdt(E[t, r], I[t, r], I_input[r], c[r], d[r], Q[r], tauI[r], I_act) + np.random.normal(0, dyn_noise))

    E += np.random.normal(0, obs_noise, E.shape)
    I += np.random.normal(0, obs_noise, I.shape)

    return E, I

def simulate_hopf(config : dict):
    # https://www.nature.com/articles/s41598-017-03073-5
    T = config.get('T', 6.)
    dt = config.get('dt', 0.05)
    nr = config.get('nr', 5)

    torch.empty((nr,)).normal_(5.)

    a = config.get('a', torch.empty((nr,)).normal_(3., 1.))
    omega = config.get('omega', torch.empty((nr,)).normal_(5., 1.))
    beta = config.get('beta', torch.empty((nr,)).normal_(4., 1.))

    C = config.get('C', _initial_C(nr))
    obs_noise = config.get('obs_noise', 0.)

    time = torch.arange(0, T + dt, dt)
    X = np.zeros((len(time), nr))
    Y = np.zeros((len(time), nr))

    simulation_info = (
        f"Running simulation with: "
        f"T = {T}, dt = {dt}, nr = {nr}, a = {a}, omega = {omega}, beta = {beta}"
    )
    print(simulation_info)

    X[0] = 0.0
    Y[0] = 0.0

    for t in range(len(time) - 1):
        X_input = np.dot(C, X[t])
        Y_input = np.dot(C, Y[t])

        for r in range(nr):
            X[t+1, r] = X[t, r] + dt * (h_dXdt(X[t, r], Y[t, r], a[r], omega[r], X_input[r]) + np.random.normal(0, beta[r]))
            Y[t+1, r] = Y[t, r] + dt * (h_dYdt(X[t, r], Y[t, r], a[r], omega[r], Y_input[r]) + np.random.normal(0, beta[r]))

    return X, Y