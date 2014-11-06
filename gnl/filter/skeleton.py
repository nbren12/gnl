"""
Example file for filtering the column skeleton model

To run:
    python -m gnl.filter.lorenz63
"""
from gnl.filter import *

from scipy.integrate import odeint
from numpy.random import multivariate_normal, randint
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot, exp, round, log
from IPython import embed


# Define Grid
nx     = 1
ndet   = 3
nstoch = 1

nv = nx * (ndet + nstoch)

# Define evolver

sq = np.array([.022])
st = np.array([.022])
def evolveskel(state_data, tcur, tout):
    """Wrap skel.evolve to fit FilterDriver class

    Args: state_data (ndarray): shape is (ndet + nstoch, nx). The birth-death
        variable is stored as log(eta + 1)

    """
    da = .01

    from skeleton.skel import evolve

    state_data = state_data.copy()

    # state_data is flat so need to reshape
    sh = state_data.shape
    state_data.shape = (ndet + nstoch, nx)
    U = state_data[:ndet,:]
    y = state_data[ndet:,:]

    eta = round(exp(y)/da-1) # Turn y into eta
    # eta = round(y**2) * ( y >0 )
    eta[eta < 0] = 0 # Ensure no negative values
    evolve(tout-tcur, U, eta, sq=sq, st=st, da=da)

    assert eta[0] >= 0
    y[:] = log(da*(eta + 1)) # Turn eta into y
    # y[:] = np.sqrt(eta)
    # Old Shape
    state_data.shape = sh

    return state_data



# Time step
dt = 10.0
nt = 200
tout = np.arange(nt) * dt

# Observation operators

# only observe A
G = np.array([[0,0,0,1]])
Ro = np.diag([.1**2])

# only observe Z
# G = np.array([[0,0,1,0]])
# Ro = np.diag([.01**2])

# observe Z and A
# G = np.array([[0,0,1,0],
#               [0,0,0,1]])
# Ro = np.diag([.01**2, .4**2])

# Full obs
# Ro = np.eye(nv) * .5**2
# Ro = np.diag([.1**2, .1**2, .05**2, 1.0**2])

nobs = G.shape[0]

# Initialize truth
y0 = np.ones(nv) * .1

# Initialize ensemble
ne = 10
ens = np.ones((nv, ne)) * .1 + multivariate_normal(np.ones(nv), np.eye(4)*.1, ne).T

# Analysis
analyzer = SequentialKFAnalysis(G, np.diag(Ro), r=.04)

# Filter driver object
fd = FilterDriver(ens, evolveskel, analyzer)

# Ouput
output = np.zeros((ne, nt, nv))
y_truth = np.zeros((nt, nv))
obs     = np.zeros((nt, nobs))

obs[0,:] = dot(G, y_truth[0,:])
# Assimilate Observations
for i in range(1,nt):
    print('Iteration {i} of {n}'.format(i=i, n=nt), end='\r')

    # Run Truth model
    y_truth[i,:] = evolveskel(y_truth[i-1,:], tout[i-1], tout[i])

    # Observe the truth
    obs[i,:] = dot(G, y_truth[i,:]) \
            + multivariate_normal(np.zeros(nobs), Ro)
    if True:
        # Prediction
        fd.predict(tout[i-1], tout[i])

        # Analysis
        fd.analyze(obs[i,:])

    ens[:] = fd._ensemble
    output[:, i, :]  = ens.T

vr = 0 # Variable to plot

fig,b = plt.subplots()

b.plot(tout, dot(y_truth, G[vr,:]), 'k--', label='Truth')
# ax.plot(tout, ensMu[:,vr], 'k-', label='EnKF')
b.plot(tout, obs[:, vr], 'ko', label='obs')


b.plot(tout, dot(output, G[vr,:]).T, 'b', alpha=.20, label='EAKF');

rms = lambda x: np.sqrt(x**2).mean()

fig,b = plt.subplots()
# vr = vr + nobs
b.plot(tout, y_truth[:,vr], 'k--', label='Truth')
b.plot(tout, output[:,:,vr].T, 'b', alpha=.20, label='EAKF');

print("Filter RMS is %f"%rms(y_truth-output.mean(axis=0)))
# print("Obs RMS is %f"%rms(obs-y_truth))

plt.show()
