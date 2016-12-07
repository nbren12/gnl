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

from .tadmor import Tadmor2DBase, divergence





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
        vec.exchange()
        uc = vec.ghostview
        super(Tadmor2D, self).central_scheme(uc, dx, dy, dt)
        return vec
