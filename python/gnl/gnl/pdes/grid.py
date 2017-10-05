import numpy as np

def ghosted_grid(size, length, g):

    mesh = [np.r_[-g:s+g]/s*l for s, l in zip(size, length)]
    diff = [m[1]-m[0] for m in mesh]
    mesh = np.meshgrid(*mesh, indexing='ij')

    return mesh, diff
