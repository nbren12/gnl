import numpy as np
from .bc import periodic_bc, fillboundary

class MultiFab(object):
    """
    >>> a = MultiFab(sizes=[10,10], dof=1, n_ghost=2)
    >>> a.view(1).shape
    (1, 12, 12)
    >>> a.sizes
    (10, 10)
    """
    def __init__(self, data=None, sizes=None, n_ghost=0, dof=None):
        "docstring"

        if data is not None:
            if dof is None:
                dof = -1
            self.data = data[:dof,...]
        elif sizes is not None:
            self.data = np.zeros([dof] + [s + 2 * n_ghost for s in sizes])
        else:
            raise ValueError("Need either data or sizes")

        self.n_ghost = n_ghost

    @property
    def sizes(self):
        return self.view(0).shape[1:]

    @property
    def dof(self):
        return self.data.shape[0]

    def exchange(self):
        periodic_bc(self.data, g=self.n_ghost, axes=(1, 2))

    @property
    def validview(self):
        g = self.n_ghost
        return self.data[:, g:-g, g:-g]

    @property
    def ghostview(self):
        return self.data

    def view(self, g):
        ng = self.n_ghost
        if g < ng:
            return self.data[:, ng-g:-ng+g, ng-g:-ng+g]
        else:
            return self.data

class BCMultiFab(MultiFab):
    def __init__(self, bcs=None, **kwargs):
        "docstring"
        super(BCMultiFab, self).__init__(**kwargs)

        if bcs is None:
            bcs = [('wrap', 'wrap')]*self.dof

        self.bcs = bcs

    def exchange(self):
        bcs = self.bcs

        assert len(self.bcs) == self.dof

        for i, bc in enumerate(self.bcs):
            bcarg = [[b]*2 for b in bc]
            fillboundary(self.ghostview[i],
                         bcs=bcarg,
                         axes=[0,1],
                         g=self.n_ghost)
