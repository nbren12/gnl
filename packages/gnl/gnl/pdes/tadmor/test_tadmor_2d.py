import numpy as np
from .tadmor_2d import Tadmor2D
from ..fab import BCMultiFab


def tadmor_error(n):
    uc = np.zeros((1, n + 4, n + 4))

    L = 1.0
    dx = L / n

    x = np.arange(n) * dx
    x, y = np.meshgrid(x, x)

    initcond = lambda x, y: np.exp(-((x % L - .5) / .10)**2) * np.exp(-((y % L - .5) / .10)**2)

    dt = dx / 2

    class AdvecSolver(Tadmor2D):

        n_ghost = 4
        neq = 1
        bcs = None
        sizes = (n, n)

        def fx(self, out, uc):
            out[:] = uc

        def fy(self, out, uc):
            out[:] = uc

        def init_uc(self):
            # create initial data
            uc = BCMultiFab(
                sizes=self.sizes,
                n_ghost=self.n_ghost,
                dof=self.neq,
                bcs=self.bcs)

            return uc

    tend = .25
    t = 0

    tad = AdvecSolver()
    tad.geom.n_ghost = 8

    uc = tad.init_uc()
    uc.validview[:] = initcond(x, y)

    while (t < tend - 1e-10):
        dt = min(dt, tend - t)
        tad.central_scheme(uc, dx, dx, dt)
        t += dt

    return np.mean(np.abs(initcond(x - t, y - t) - uc.validview[0, ...]))


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
