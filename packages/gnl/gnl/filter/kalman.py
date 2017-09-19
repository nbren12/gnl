from scipy.linalg import solve
from ..util import vdot
import numpy as np

def conjpose(A):
    return np.conj(A.T)

def kalman_filter(prior_mean, prior_var, obs, obs_var, obs_mat):
    """Kalman filter function

    Args:
        prior_mean: (n,) array_like
        prior_var: (n, n) prior noise covariance
        obs: (k,) observations
        obs_var: (k,k) the observation noise covariance
        obs_mat: (k,n) The linear observation operator

    Returns:
        post_mean: (n,)
        post_var: (n,n)

    """

    # H Sigma H' + R
    tmp1 = vdot(obs_mat, prior_var, conjpose(obs_mat)) + obs_var

    inov = obs - vdot(obs_mat, prior_mean)
    post_mean = prior_mean + \
                vdot(prior_var, conjpose(obs_mat), solve(tmp1, inov, sym_pos=True))
    
    post_var  = prior_var \
                -vdot(prior_var, conjpose(obs_mat), 
                      solve(tmp1, vdot(obs_mat, prior_var), sym_pos=True))


    return post_mean, post_var

