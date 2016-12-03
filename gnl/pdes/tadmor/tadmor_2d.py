""" Python implementation of the Tadmor centered scheme in 2d


Routines
--------
central_scheme - 2d implementation of tadmor centered scheme

TODO
----
Replace all periodic_bc calls with `comm', so that this code can be run in parallel
"""
from functools import partial
import numpy as np
from scipy.ndimage import correlate1d
from numba import jit

try:
    from ..bc import periodic_bc
except:
    from .tadmor_common import periodic_bc

from .tadmor import _slopes, _stagger_avg, _corrector_step


def _roll2d(u):
    return np.roll(np.roll(u, -1, axis=1), -1, axis=2)

class PeriodicGeom(object):
    n_ghost = 4
    def fill_boundaries(self, uc):
        periodic_bc(uc, g=self.n_ghost, axes=(1,2))
        return uc

    def validview(self, uc):
        g = self.n_ghost
        return uc[:, g:-g, g:-g]


class Tadmor2D(object):

    geom = PeriodicGeom()

    def fx(self, uc):
        raise NotImplementedError

    def fy(self, uc):
        raise NotImplementedError

    def _single_step(self, uc, dx, dy, dt):
        uc = self.geom.fill_boundaries(uc)

        ux = np.zeros_like(uc)
        uy = np.zeros_like(uc)
        uc = uc.copy()
        lmd_x = dt / dx
        lmd_y = dt / dy

        ustag = _stagger_avg(uc)

        # predictor: mid-time-step pointewise values at cell-center
        # Eq. (1.1) in Jiand and Tadmor
        ux = _slopes(self.fx(uc), axis=1)
        uy = _slopes(self.fy(uc), axis=2)
        uc -= lmd_x / 2 *ux   + lmd_y/2 * uy

        # corrector
        # Eq (1.2) in Jiang and Tadmor
        # self.fill_boundaries(uc)
        ustag += _corrector_step(self.fx(uc), self.fy(uc), lmd_x, lmd_y)

        return ustag

    def central_scheme(self, uc, dx, dy, dt):
        """ One timestep of centered scheme


        Parameters
        ----------
        fx : callable
            fx(u) calculates the numeric flux in the x-direction
        uc: (neq, n)
            The state vector on the centered grid
        dx: float
            size of grid cell
        dt: float
            Time step

        Returns
        -------
        out: (neq, n)
        state vector on centered grid
        """
        ustag = _roll2d(self._single_step(uc, dx, dy, dt / 2))
        uc = self._single_step(ustag, dx, dy, dt / 2)

        return uc
