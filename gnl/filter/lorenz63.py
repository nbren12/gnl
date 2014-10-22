"""
Example file for filtering the Lorenz63 system

To run:
    python -m gnl.filter.lorenz63
"""
from gnl.filter import *

from scipy.integrate import odeint
from numpy.random import multivariate_normal
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot


# Right hand side of loren63 system
def rhs(U, t, sigma=10, rho=28, b=8/3):
    x = U[0]
    y = U[1]
    z = U[2]

    return np.array([sigma * ( y -x ),
                    rho * x - y - x * z,
                    x * y - b * z])

# Define evolver callabale for the FilterDriver Object
def evolve63(U, tcur, tout):
    return odeint(rhs, U, [tcur, tout])[-1,:]

# Time step
dt = .08
nt = 400

nv = 3

# Run exact model
tout = np.arange(nt) * dt
y_truth = odeint(rhs, [1.0, 1.0, 1.0], tout)

# Generate Observations
G = np.eye(nv)
Ro = np.eye(3) * 4**2
obs = dot(y_truth, G.T) + multivariate_normal(np.ones(nv), Ro, len(tout))

# Initialize ensemble
ne = 10
ens = np.ones((nv, ne)) + multivariate_normal(np.ones(nv), Ro*.1, ne).T

# Analysis
analyzer = SequentialKFAnalysis(G, np.diag(Ro))

# Filter driver object
fd = FilterDriver(ens, evolve63, analyzer)

# Ouput
output = np.zeros((ne, nt, nv))

# Assimilate Observations
for i in range(1,nt):
    print('Iteration {i} of {n}'.format(i=i, n=nt), end='\r')

    # Prediction
    fd.predict(tout[i-1], tout[i])

    # Analysis
    fd.analyze(obs[i,:])

    ens[:] = fd._ensemble
    output[:, i, :]  = ens.T

vr = 0 # Variable to plot

fig,b = plt.subplots()

b.plot(tout, y_truth[:,vr], 'k--', label='Truth')
# ax.plot(tout, ensMu[:,vr], 'k-', label='EnKF')
b.plot(tout, obs[:, vr], 'ko', label='obs')


b.plot(tout, output[:,:,vr].T, 'b', alpha=.20, label='EAKF');


rms = lambda x: np.sqrt(x**2).mean()

print("Filter RMS is %f"%rms(y_truth-output.mean(axis=0)))
print("Obs RMS is %f"%rms(obs-y_truth))

plt.show()
