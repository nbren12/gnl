"""Module for implementing boundary conditions in ghost cell schemes

These functions are kind of like np.pad, but operator inplace

fillboundary
periodic_bc

"""
import numpy as np

def _even(u, g):
    return u[g-1::-1,...]

def _odd(u, g):
    return -u[g-1::-1,...]

def _extrap(u, g):
    return u[0:1,...]

def _wraps(u, g):
    return u[-g:,...]

def fillboundary(u, axes=[1,2], bcs=None, g=1):
    """Fill boundary cells in u

    Parameters
    ----------
    u:
        an array with ghostcells
    bcs: seq
        a sequence of tuples ( 'mode', None ) describing boundary
        conditions for each axis. default is periodic. Acceptable modes include
          - 'even' ('neumann')
          - 'odd' ('dirichlet')
          - 'wrap' ('periodic')
          - 'extrap' 
          - None (do nothing)
    g: int
        number of ghost cells

    Examples
    -----
    >>> x = np.arange(-2,5+3) 
    >>> fillboundary(x, [('even', 'even')], 2)
    array([1, 0, 0, 1, 2, 3, 4, 5, 5, 4])
    >>> fillboundary(x, [('even', 'odd')], 2)
    array([ 1,  0,  0,  1,  2,  3,  4,  5, -5, -4])
    >>> fillboundary(x, [('extrap', 'odd')], 2)
    array([ 0,  0,  0,  1,  2,  3,  4,  5, -5, -4])
    >>> fillboundary(x, [('wrap', 'wrap')], 2)
    array([4, 5, 0, 1, 2, 3, 4, 5, 0, 1])
    """

    name_to_fun ={'even': _even, 'odd': _odd, 'extrap': _extrap,
                  'wrap': _wraps}

    if bcs is None:
        bcs = [['wrap', 'wrap']]*len(axes)

    for axis, bc in zip(axes, bcs):
        uv = u.swapaxes(axis, 0)
        in_views = [uv[g:-g,...], uv[-g-1:g-1:-1,...]]
        out_views = [uv[:g,...], uv[-1:-g-1:-1,...]]

        for ghost_region, valid_region, btype in zip(out_views, in_views, bc):
            if btype is not None:
                try:
                    ghost_region[:] = name_to_fun[btype](valid_region, g)
                except KeyError:
                    raise KeyError("{} is not a valid boundary type".
                                   format(btype))
    return u


def periodic_bc(u, g=2, axes=(1, 2)):
    """periodic bc in arbitrary dimensions

    Provided for convenience

    Examples
    ------
    >>> x = np.arange(-3, 14)
    >>> periodic_bc(x, g=3, axes=(0,))
    array([ 8,  9, 10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  1,  2])
    >>> x.__array_interface__['data'][0] == _.__array_interface__['data'][0]
    True
    """
    return fillboundary(u, axes=axes, g=g)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
