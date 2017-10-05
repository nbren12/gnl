from petsc4py import PETSc
from ..tadmor.tadmor_2d import MultiFab

class PETScFab(MultiFab):

    def __init__(self, da):
        "docstring"
        self.da = da
        self.lvec = self.da.createLocalVec()
        self.gvec = self.da.createGlobalVec()

    def view(self, n_ghost):
        inds = []
        l = self.l
        for (beg, end), start in zip(self.da.ranges, l.starts):
            inds.append(slice(beg-start-n_ghost, end-start+n_ghost))
        return self.l[inds][:].swapaxes(-1, 0)
        return self.valid

    @property
    def n_ghost(self):
        return self.da.stencil_width

    @property
    def dof(self):
        return self.da.dof

    @property
    def ghostview(self):
        return self.l[:].swapaxes(-1, 0)

    @property
    def validview(self):
        """It is more convenient to make the components the first dimension

        np.swapaxes should not change the underlying memory structures.
        """

        return self.view(0)

    @property
    def globalview(self):
        return self.g[:].swapaxes(0, -1)

    @property
    def l(self):
        return self.da.getVecArray(self.lvec)

    @property
    def g(self):
        return self.da.getVecArray(self.gvec)

    def scatter(self):
        self.da.globalToLocal(self.gvec, self.lvec)

    def gather(self):
        self.globalview[:] = self.validview

    def exchange(self):
        self.gather()
        self.scatter()
