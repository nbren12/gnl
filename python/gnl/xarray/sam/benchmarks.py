import cProfile
from contextlib import contextmanager
from timeit import default_timer
import numpy as np
import xarray as xr
from . import regrid


@contextmanager
def timer():
    start = default_timer()
    yield
    end = default_timer()
    print(f"Time elapsed: {end-start}")


def benchmark_coarsen():
    x = xr.DataArray(np.random.rand(10000, 10000), dims=['x', 'y'])
    blocks = dict(x=100, y=100)

    with timer():
        y = regrid.coarsen(x, blocks).compute()

    x = x.values
    with timer():
        regrid.coarsen_centered_np(x, blocks)

if __name__ == '__main__':
    cProfile.run("benchmark_coarsen()", filename="benchmarks.txt")