import xray

def xargs(z):
    """
    Returns:
    x, y, z
    """

    dims = z.dims
    y = z.coords[dims[0]].values
    x = z.coords[dims[1]].values

    return x, y, z.values
