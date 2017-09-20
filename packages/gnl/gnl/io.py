import logging
import netCDF4 as nc
logger = logging.getLogger(__name__)


def readmat(fname):
    """Read Matlab v4/5/6 and v7.3 using hd5py"""
    from scipy.io import loadmat

    try:
        return loadmat(fname)
    except:
        import h5py
        return h5py.File(fname)


class NetCDF4Writer(object):
    """Object for writing solution output to netcdf

    """

    def __init__(self, grid={}, filename="out.nc"):
        """

        Parameters
        ----------
        grid: dict
            dictionary of grid values
        """
        self._f = nc.Dataset(filename, "w")
        self.filename = filename

        # Create dimension information
        for k in grid:
            self._f.createDimension(k, len(grid[k]))
            gridvar = self._f.createVariable(k, 'f4', (k,))
            gridvar[:]  = grid[k]

        self._f.createDimension('time', None)
        self.times = self._f.createVariable('time', 'f4', ('time',))

        self._iter = 0
        self._f.sync()


    def collect(self, t, variables):
        """Collect variables into netcdf file

        Parameters
        ----------
        variables: seq
            iterable of (varname, dim, arr) tuples
        """
        logger.info("Storing output data at t={0}".format(t))
        self.times[self._iter] = t

        try:
            self.ncvars
        except AttributeError:
            self.ncvars = {}

        for name, dim, arr in variables:
            try:
                ncvar = self.ncvars[name]
            except KeyError:
                dim = ['time'] + list(dim)
                ncvar = self._f.createVariable(name, 'f4', dim)
                self.ncvars[name] = ncvar


            ncvar[self._iter] = arr

        if self._iter % 50 == 0:
            self._f.sync()

        self._iter += 1
        return

    def finish(self):
        logger.info("Closing `{}`".format(self.filename))
        self._f.sync()
        self._f.close()
        pass
