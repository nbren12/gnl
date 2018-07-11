import numpy as np
from .tadmor import Tadmor1DBase
from ..fab import BCMultiFab


def tadmor_error(n):
    g = 2
    uc = np.zeros((1, n + 2*g, 1))

    L = 1.0
    dx = L / n

    x = np.arange(-2, n+2) * dx
    x = x[:,None]

    initcond = lambda x: np.exp(-((x % L - .5) / .10)**2)
    dt = dx / 2

    class AdvecSolver(Tadmor1DBase):

        n_ghost = g

        def fx(self, out, uc):
            out[:] = uc

        def central_scheme(self, uc, dx, dt):
            g = self.n_ghost
            uc[0,:g,:] = uc[0,-2*g:-g,:]
            uc[0,-g:,:] = uc[0,g:2*g,:]
            super(AdvecSolver, self).central_scheme(uc, dx, dt)

    tend = .25
    t = 0

    tad = AdvecSolver()

    uc[0] = initcond(x)

    while (t < tend - 1e-10):
        dt = min(dt, tend - t)
        tad.central_scheme(uc, dx, dt)
        t += dt

    return np.mean(np.abs(initcond(x - t) - uc[0])[g:-g,0])


def test_tadmor_convergence(plot=False):
    """
    Create error convergence plots for 1d advection problem
    """
    from ..testing import test_convergence
    nlist = [16, 32, 64, 128, 256]
    test_convergence(tadmor_error, nlist, order_expected=2.0)


if __name__ == '__main__':
    pass
    test_tadmor_convergence(plot=True)
    # plot_advection2d()
    # test_slopes()
    # test_stagger_avg()
