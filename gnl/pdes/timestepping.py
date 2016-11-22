"""Module for timesteppers and associated utilties"""
import logging

def steps(onestep, q, dt, tbound, *args, **kwargs):
    """Iterator with fixed time step

    Parameters
    ----------
    onestep: callable(soln, t, dt, *args, **kwargs)
    """
    t = tbound[0]

    yield t, q

    while (t < tbound[1] - 1e-10):
        dt = min(dt, tbound[1]-t)
        q, t  = onestep(q, t, dt, *args, **kwargs), t + dt

        logging.debug("t = {t:.2f}".format(t=t))

        yield t, q
