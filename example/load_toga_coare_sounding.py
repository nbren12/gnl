"""
Read data from the TOGA COARE soundings available at
http://tornado.atmos.colostate.edu/togadata/ifa_data.html#DATA
"""
import os
import xray
import datetime
import numpy as np
from numpy import linspace, arange, concatenate
from scipy.interpolate import interp1d


def getsound(sound):
    columns = ['p', 'z', 'T', 'q', 'rh', 'u', 'v']

    tdat = np.fromstring(sound[0], dtype=np.int, sep=' ')

    time  = datetime.datetime(tdat[0]+1900, *tdat[1:])

    arr = np.fromstring(''.join(sound[1:]), sep=' ')\
            .reshape((-1, len(columns)))

    da = xray.Dataset({columns[i]:(['i'], arr[:,i]) for i in range(7)},
                     coords={'time': time,
                             'i': np.arange(arr.shape[0])})

    return  da


def load_sounds(fname):

    f = open(fname)
    lines = f.readlines()

    sounds = [lines[i:i+42] for i in range(0, len(lines), 42)]
    return xray.concat([getsound(sound) for sound in sounds], dim='time')





def main():


    ## Get TOGA profiles
    if not os.path.exists('basic_flds'):
        os.system('curl http://tornado.atmos.colostate.edu/togadata/data/ifa_averaged_v2/basic_flds.ifa_v2.1.gz | gunzip > basic_flds')

    toga = load_sounds('basic_flds')
    toga.to_netcdf("toga.nc")

if __name__ =='__main__':
    main()
