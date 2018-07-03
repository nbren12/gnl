"""
Lorenz 96 model
From https://en.wikipedia.org/wiki/Lorenz_96_model
"""
from scipy.integrate import odeint
import numpy as np
import torch

# these are our constants
N = 36  # number of variables
F = 8  # forcing


def lorenz96(x, t):

    # compute state derivatives
    d = np.zeros(N)
    # first the 3 edge cases: i=1,2,N
    d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
    d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
    d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]
    # then the general case
    for i in range(2, N - 1):
        d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
    # add the forcing term
    d = d + F

    # return the state derivatives
    return d


x0 = F * np.ones(N)  # initial state (equilibrium)
x0[19] += 0.01  # add small perturbation to 20th variable
t = np.arange(0.0, 3000.0, 0.01)

x = odeint(lorenz96, x0, t)
x = (x - x.mean())/x.std()

torch.save(x, "l96.torch")
