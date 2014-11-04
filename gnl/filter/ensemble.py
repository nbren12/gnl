import numpy as np
from scipy.linalg import solve, sqrtm, svd, lstsq
from numpy.random import multivariate_normal
from numpy import sqrt, dot
from toolz.curried import *

def vdot(*arrs):
    """Variadic numpy dot function"""

    if len(arrs) == 2:
        return dot(**arrs)
    else:
        return vdot(arrs[0], vdot(*arrs[1:]))


class EnKFAnalysis(object):
    """
    Object for performing ensemble kalman filter


    The observation model is given by:
        obs  = G(pred) + eta
    with
        Var (eta) = Ro.


    """
    def __init__(self, G, Ro, r=1.0):
        """TODO: Docstring for __init__.

        Args:
            G (matrix): observation operator possibly nonlinear
            Ro (2d or 1d array): observation noise covariance
        """
        self._G = G
        self._Ro = Ro

        self._r  = r

        if Ro.ndim == 2:
            self._Ro2 =sqrtm(Ro)
        else:
            self._Ro2 = sqrt(Ro)

    def __call__(self, ensemble, obs, r=1.0):
        """Preform EnKF analysis

        :ensemble: ensemble member predictions shape (nvar, nensemble)
        :obs: observations
        :Ro:  observation noise covariance (default 1.0)
        :G:  observation operator (default ident)
        :r:  variance inflation factor
        :returns: analysis ensemble members (shape is same as ensemble)
        """

        G = self._G
        Ro = self._Ro
        r  = self._r

        nvar, ne = ensemble.shape

        anl = np.empty_like(ensemble)

        mu = ensemble.mean(axis=1)
        U  = ensemble-mu[:,None]

        # V = np.apply_along_axis(G, 0, ensemble)
        V = dot(G, ensemble)
        # V = V - G(mu)[:,None]
        V = V - V.mean(axis=1)[:,None]


        U *= sqrt(1+r)
        V *= sqrt(1+r)

        A = V.dot(V.T) + Ro * (ne - 1)

        for k in range(ne):
            dev = obs + np.dot(self._Ro2, np.random.randn(nvar)) \
                    - dot(G, ensemble[:,k])

            anl[:,k] = ensemble[:,k] +  U.dot(V.T.dot(solve(A, dev)))

        return anl


def obs_increment_EAKF(ensemble, observation, obs_error_var):
    """
    Computes increment for the ensemble adjustment filter.

    Note:
        From Yoonsang's code
    """
    # Set error return to default successful
    err = 0

    # Compute prior ensemble mean and variance
    prior_mean = ensemble.mean()
    prior_var = ensemble.var(ddof=1)

    # If both prior and observation error variance are return error
    if prior_var <= 0 and obs_error_var <= 0:
        err = 1
        return

    # Compute the posterior mean and variance
    # If prior variance is 0, posterior mean is prior_mean and variance is 0
    if prior_var == 0:
        post_mean = prior_mean
        post_var = 0
        var_ratio = 0.0
    elif obs_error_var == 0:
        # If obs_error_var is 0, posterior mean is observation and variance is 0
        post_mean = observation
        post_var = 0
        var_ratio = 0.0
    else:
        # Use product of gaussians
        # Compute the posterior variance
        post_var = 1 / (1 / prior_var + 1 / obs_error_var)
        var_ratio = post_var / prior_var

        # Compute posterior mean
        post_mean = post_var * (prior_mean / prior_var + observation / obs_error_var)

    # Shift the prior ensemble to have the posterior mean
    updated_ensemble = ensemble - prior_mean + post_mean

    # Contract the ensemble to have the posterior_variance
    updated_ensemble = sqrt(var_ratio) * (updated_ensemble - post_mean) \
        + post_mean

    # Compute the increments
    obs_increments = updated_ensemble - ensemble

    return obs_increments, err


def get_state_increments(state_ens, obs_ens, obs_incs):
    """
    Computes state increments given observation increments and the state and
    obs prior ensembles
    """


    # Regression model: state_ens = beta * obs_ens + eps
    stateMu = state_ens.mean()
    obsMu = obs_ens.mean()
    sig2ob = np.sum((obs_ens-obsMu)**2)

    if abs(sig2ob) > 1e-10:
        beta = np.dot(state_ens - stateMu, obs_ens - obsMu)\
            / sig2ob
    else:
        beta = 0.0

    # Original
    state_incs = obs_incs * beta
    return state_incs


class SequentialKFAnalysis(EnKFAnalysis):
    """Object for performing ensemble kalman filters on independent
    observations

    A localization interface is implemented, but by default, no localization is
    performed. Subclasses with localization should override `_localized`.

    """


    def _localized(self, idx_obs, idx_state):
        """Return localization information given observation and state indices"""

        cov_factor = 1.0
        cov_factor *= sqrt(1+self._r)
        affected  = True

        return cov_factor, affected

    def __call__(self, ensemble, obs):
        """Calculate analysis given prior ensemble and observations

        Args:
            ensemble (2d array): Array containing ensemble member forecasts.
                ensemble.shape = (nstate, nens). This array is altered by the
                computation.
            obs (1d array): Observations

        Returns:
            analysis ensemble.
        """

        nstate, nens = ensemble.shape

        nobs = obs.shape[0]

        for i in range(nobs):
            g  = self._G[i,:]
            ro = self._Ro[i]

            # Compute increments of ensemble member observations
            obs_ensemble = dot(g, ensemble)
            ob = obs[i]
            obs_inc, errflag = obs_increment_EAKF(obs_ensemble, ob, ro)

            for j in range(nstate):
                cov_factor, affected = self._localized(i, j)
                if affected:
                    # Compute state increments using least squares
                    state_incs = get_state_increments(ensemble[j, :],
                                                      obs_ensemble, obs_inc)

                    # Update ensemble
                    ensemble[j, :] += state_incs * cov_factor

        return ensemble


class Observer(object):

    """Object for generating observations"""

    def __init__(self, G, Ro):
        """
        Args:
            G (callable): observation operator with prototype
                obs = G(state)
            Ro (2d or 1d array): error covariance
        """

        self._G  = G
        self._Ro = Ro

    def  _call__(self, ens):
        nv = ens.shape[0]
        if ens.ndim == 1:
            ens.shape = (nv, 1)

        ne = ens.shape[1]

        # TODO finish

def test_enkf_eafkf():
    """
    Testing EnKF and EaKF on lorenz 63

    Notes:
        Runs and plots the ensemble adjustment and ensemble kalman filters with
        full observations.

    """
    from scipy.integrate import odeint
    from numpy.random import multivariate_normal
    from matplotlib import pyplot as plt


    # Right hand side of loren63 system
    def rhs(U, t, sigma=10, rho=28, b=8/3):
        x = U[0]
        y = U[1]
        z = U[2]

        return np.array([sigma * ( y -x ),
                        rho * x - y - x * z,
                        x * y - b * z])


    # Time step
    dt = .08
    nt = 400

    nv = 3

    # Run exact model
    tout = np.arange(nt) * dt
    y_truth = odeint(rhs, [1.0, 1.0, 1.0], tout)

    # Observations
    Ro = np.eye(3) * 4**2
    obs = y_truth + multivariate_normal(np.ones(nv), Ro, len(tout))

    # Functions for running the filters
    def run_lorenzenkf():
        ne = 10
        nv = 3

        ens = np.ones((nv, ne)) + multivariate_normal(np.ones(nv), Ro*.1, ne).T
        pred= np.empty_like(ens)

        output = np.zeros((ne, nt, nv))
        G = np.eye(nv)

        enkf = EnKFAnalysis(G, Ro)

        for i in range(1,nt):
            print('Iteration {i} of {n}'.format(i=i, n=nt), end='\r')
            # Prediction
            for k in range(ne):
                pred[:, k] = odeint(rhs, ens[:, k], [0.0, dt])[-1,:]
    #             pred[:, k] += rhs(ens[:, k], 0.0) * dt

            # Analysis
            anl = enkf(pred, obs[i,:],  r=0.00)

            ens[:] = anl
            output[:, i, :]  = ens.T

        return output

    def run_lorenzeakf():
        ne = 10
        nv = 3

        ens = np.ones((nv, ne)) + multivariate_normal(np.ones(nv), Ro*.1, ne).T
        pred= np.empty_like(ens)

        output = np.zeros((ne, nt, nv))
        G = np.eye(nv)
        eakf = SequentialKFAnalysis(G, np.diag(Ro), r=0.00)

        for i in range(1,nt):
            print('Iteration {i} of {n}'.format(i=i, n=nt), end='\r')
            # Prediction
            for k in range(ne):
                pred[:, k] = odeint(rhs, ens[:, k], [0.0, dt])[-1,:]
    #             pred[:, k] += rhs(ens[:, k], 0.0) * dt

            # Analysis
            anl = eakf(pred, obs[i,:])
    #         print(anl)

            ens[:] = anl
            output[:, i, :]  = ens.T

        return output


    # Run and plot output for enkf
    vr = 1
    output_enkf = run_lorenzenkf()

    fig, (a,b) = plt.subplots(2, figsize=(10,5), sharex=True)
    a.plot(tout, y_truth[:,vr], 'k--', label='Truth')
    # ax.plot(tout, ensMu[:,vr], 'k-', label='EnKF')
    a.plot(tout, obs[:, vr], 'ko', label='obs')


    a.plot(tout, output_enkf[:,:,vr].T, 'b', alpha=.20, label='EnKF');
    # plt.xlim([10, 20])

    # Run and plot output for eakf
    output_eakf = run_lorenzeakf()
    b.plot(tout, y_truth[:,vr], 'k--', label='Truth')
    # ax.plot(tout, ensMu[:,vr], 'k-', label='EnKF')
    b.plot(tout, obs[:, vr], 'ko', label='obs')


    b.plot(tout, output_eakf[:,:,vr].T, 'b', alpha=.20, label='EAKF');


    rms = lambda x: np.sqrt(x**2).mean()

    print("EAKF RMS is %f"%rms(y_truth-output_eakf.mean(axis=0)))
    print("EnKF RMS is %f"%rms(y_truth-output_enkf.mean(axis=0)))
    print("Obs RMS is %f"%rms(obs-y_truth))

    plt.show()
