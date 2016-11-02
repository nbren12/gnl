""" Module for implementations of integrating continuous time markov chains
"""
import numpy as np
import logging

logger = logging.getLogger(__file__)


def stochastic_integrate_array(scmt, rates, a, b):
    """Preform Gillespie algorithm for an array of data

    Parameters
    ----------
    scmt: (n,)
        array of integers representing the discrete stochastic state
    rates: (p, p, n)
        array of stochastic transition rates. p is the number of allowed states
    a: float
        starting time
    b: float
        ending time
    """

    n = scmt.shape[0]
    time = np.ones(n) * a
    running = np.arange(scmt.shape[0])

    crates = np.cumsum(rates, axis=1)

    while len(running) > 0:
        logger.debug("running has length {0}".format(len(running)))

        lam = crates[scmt[running],
                    :,
                    np.arange(len(running))]

        mask = lam[:,-1] != 0

        running = running[mask]
        lam = lam[mask, :]

        U1 = np.random.rand(len(running))
        tau = -log(U1)/lam[:,-1]
        time[running] = time[running] + tau

        mask = time[running] < b
        running = running[mask]

        if len(running) > 0:
            lam = lam[mask,:]

            U2 = np.random.rand(len(running))

            for i, idx in enumerate(running):
                scmt[idx] = np.searchsorted(lam[i,:], U2[i]*lam[i,-1])


    return scmt
