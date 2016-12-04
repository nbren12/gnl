from petsc4py import PETSc
from ..tadmor.tadmor_2d import MultiFab

class PETScFab(MultiFab):

    def __init__(self, da):
        "docstring"
        self.da = da
        self._lvec = self.da.createLocalVec()
        self._gvec = self.da.createGlobalVec()

    @property
    def n_ghost(self):
        return self.da.stencil_width

    @property
    def ghostview(self):
        return self.l[:].swapaxes(-1, 0)

    @property
    def validview(self):
        """It is more convenient to make the components the first dimension

        np.swapaxes should not change the underlying memory structures.
        """

        inds = []
        l = self.l
        for (beg, end), start in zip(self.da.ranges, l.starts):
            inds.append(slice(beg-start, end-start))
        return self.l[inds][:].swapaxes(-1, 0)

    @property
    def l(self):
        return self.da.getVecArray(self._lvec)

    @property
    def g(self):
        return self.da.getVecArray(self._gvec)

    def scatter(self):
        self.da.globalToLocal(self._gvec, self._lvec)

    def gather(self):
        self.g[:] = self.validview.swapaxes(0, -1)

    def exchange(self):
        self.gather()
        self.scatter()
