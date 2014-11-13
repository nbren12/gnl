"""
Testing the linear kalman filter on a vector autoregressive model
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from gnl.util import vdot, nbsubplots
from gnl.filter import kalman_filter


n = 3

F = np.eye(n) * .8          # Dynamical Operator
F[0, -1] = -.9
F[-1, 0] = .1
H = np.diag([1, 1.0, 0.00]) # Observation operator
R = np.eye(n) * .1          # Dynamical noise covariance
Ro = np.eye(n) * .2         # Observation noise covariance

truth = np.ones(n)

nt = 200


def kalmanf(y0):
    """A generator which runs the kalman filter and yields the vars"""

    truth = y0.copy()

    post_mean = np.ones(n)
    post_var  = np.eye(n) * .1

    yield post_mean, post_var, truth, truth

    for i in range(nt):
        print('On iteration {i} of {nt}'.format(i=i+1, nt=nt), end='\r')
        truth = vdot(F, truth) + \
                multivariate_normal(np.zeros(n), R)

        # Observation
        obs = truth + \
              multivariate_normal(np.zeros(H.shape[0]), Ro)

        prior_mean = vdot(F, post_mean)
        prior_var  = vdot(F, post_var, F.T) + R

        post_mean, post_var = kalman_filter(prior_mean, prior_var, obs, Ro, H)
        yield post_mean, post_var, truth, obs


# Run kalman filter and collect output
mu, sig2, truth, obs = [np.vstack(arr) for arr in zip(*kalmanf(truth))]

# Plot output
fig, axs = nbsubplots(n, 1, aspect=.2, h=2)

rms = lambda x: np.sqrt(np.mean(x**2))

for i, ax in enumerate(axs.flat):
    ax.plot(truth[:,i], 'k--')
    ax.plot(obs[:,i], '.')
    ax.plot(mu[:,i], 'b-')
    err = rms(mu[:,i]-truth[:,i])
    oberr = rms(obs[:,i]-truth[:,i])
    ax.set_title('Ob = {oberr:.2f} ; err = {err:.2f}'.format(**locals()))
    
plt.tight_layout()
plt.show()
