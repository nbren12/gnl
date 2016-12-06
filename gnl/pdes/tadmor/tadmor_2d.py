""" Python implementation of the Tadmor centered scheme in 2d


Members
-------
Tadmor2D
"""
from functools import partial
import numpy as np
from scipy.ndimage import correlate1d
from numba import jit
from ..fab import MultiFab

from .tadmor import _slopes, _stagger_avg, _corrector_step


def _roll2d(u):
    return np.roll(np.roll(u, -1, axis=1), -1, axis=2)


class Tadmor2DBase(object):
    def fx(self, uc):
        raise NotImplementedError

    def fy(self, uc):
        raise NotImplementedError

    def _extra_corrector(self, uc, dt):
        pass

    def _single_step(self, uc, dx, dy, dt):

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
        self._extra_corrector(uc, dt/2)
        uc -= lmd_x / 2 * ux + lmd_y / 2 * uy

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

        uc[:] = _stagger_avg(_roll2d(self._single_step(uc, dx, dy, dt)))
        return uc


class Geom(object):
    """This class is provided for compatibility"""

    def validview(self, uc):
        if isinstance(uc, MultiFab):
            return uc.validview
        else:
            return MultiFab(uc, self.n_ghost).validview


class Tadmor2D(Tadmor2DBase):
    """This class is provided for compatibility

    Multifab needs to have at least 3 ghost cells
    """
    geom = Geom()

    def central_scheme(self, vec, dx, dy, dt):
        if isinstance(vec, MultiFab):
            vec.exchange()
            uc = vec.ghostview
            uc[:] = super(Tadmor2D, self).central_scheme(uc, dx, dy, dt)
            return vec
        else:
            fab = MultiFab(data=vec, n_ghost=self.geom.n_ghost)
            fab.exchange()
            uc = fab.ghostview
            return super(Tadmor2D, self).central_scheme(uc, dx, dy, dt)
