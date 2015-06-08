def readmat(fname):
    """Read Matlab v4/5/6 and v7.3 using hd5py"""
    from scipy.io import loadmat

    try:
        return loadmat(fname)
    except:
        import h5py
        return h5py.File(fname)
