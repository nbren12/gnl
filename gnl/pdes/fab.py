import numpy as np
from .bc import periodic_bc

class MultiFab(object):
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
        return self.data[ng-g:-ng-g]
