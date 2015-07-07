import xray
import numpy as np

def xargs(z):
    """
    Returns:
    x, y, z
    """

    dims = z.dims
    y = z.coords[dims[0]].values
    x = z.coords[dims[1]].values

    return x, y, z.values

def integrate(x, axis='z'):
    """
    Integrate a dataframe along an axis using np.trapz
    """
    axisnum = list(x.dims).index(axis)
    
    dims = list(x.dims)
    del dims[axisnum]
    
    coords = {key:x.coords[key] for key in dims}
    
    tot = np.trapz(x.values, x=x.coords[axis], axis=axisnum)
    return xray.DataArray(tot, coords, dims)
