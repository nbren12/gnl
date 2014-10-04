
# madrespek.py
#
# Full of useful things like:
#
# 1. Plotting functions
# 2. acf, pgram and more
# 3. important constants
#
#
# (C) Noah D. Brenowitz. 2014. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi
from numba import f8, void, jit, autojit
# import pandas as pd


day_s = 86400.0
hour_s= 3600.0

def fftdiff(u, L = 4e7, axis=-1):
    """
    Function for calculated the derivative of a periodic signal using fft.

    L is the physical length of the signal (default = 4e7, m aroun earth)
    """
    from numpy.fft import fft, ifft, fftfreq
    nx = u.shape[axis]
    x = np.linspace(0, L, nx, endpoint=False)
    k = fftfreq( nx, 1.0/nx )
    fd = fft(u, axis=axis) * k * 1j * 2 * pi / L
    ud = np.real( ifft(fd, axis=axis) )

    return ud

def pgram(x, fs=1.0):
    """
    A function to plot the periodogram of a signal x

    (C) 2013 Noah D. Brenowitz
    """
    from numpy import fft
    ps  = abs(fft.fft(x**2))**2/x.shape[0]
    print(ps.max())
    fs  = fft.fftfreq(x.shape[0],fs)

    return plt.loglog(fs, ps)

def pdpgram(x):
    fs = np.diff(np.array(x.index))[0]
    return pgram(x-x.mean(), fs=fs)


def scatter_line_plot(x, y, bands = None, alpha=.05):
    """
    Scatter plot with lines

    (C) 2013 Noah D. Brenowitz
    """
    from numpy import polyfit, polyval

    plt.plot(x, y, '.', alpha=alpha)

    if bands is not None:
        for band in bands:
            ind = (x > band[0]) &( x < band[1])
            pf = polyfit( x[ind], y[ind], 1)
            plt.plot(x[ind], polyval(pf, x[ind]))

    else:
        pf = polyfit(x, y, 1)
        plt.plot(x , polyval(pf, x))

def cart2pol(x,y):
    """
    Cartesian coordinates to polar

    (C) 2013 Noah D. Brenowitz
    """
    from pylab import sqrt, arctan, pi
    r = sqrt(x**2 + y**2)
    theta = arctan(y/x)
    theta[x<0] = pi + theta[x<0]
    theta[theta <0 ] = theta[theta<0] +2*pi

    return (r, theta)

def acf(x, axes=(0,1)):
    """
    2D ACF using fft

    Inputs:

    x is ndarray with shape (nt, nx)
    """
    from numpy.fft import fft2, ifft2, fftshift

    if x.ndim == 1:
        x = x[:,None]
    elif x.ndim == 2:
        pass
    else:
        raise NotImplementedError

    nt, nx = x.shape

    padding = np.zeros((nt, nx))

    x = np.concatenate((x, padding))

    fx = fft2(x, axes=axes)
    ac=  np.real(ifft2(fx * np.conj(fx), axes=axes))[:(nt-10),:] / nx / np.arange(nt, 10, -1)[:,None]

    ac = ac[:nt/2, :nx/2]

    return ac

def pdacf(y, nlags=1000):
    import statsmodels.api as sm
    import pandas as pd
    ac = sm.tsa.acf(y, True, nlags=nlags, fft=True)
    t  = np.array(y.index[:nlags+1]) -y.index[0]
    return pd.Series( ac, t )

#######################################################################
#                             Iris Plots                              #
#######################################################################


def cmapfac(dx=5.0, positive=True):
    """
    A convenience function for creating colormap/contour level pairs

    - dx is the spacing between contours
    - positive=True => white-to-black colorscale
      positive=False => diverging colorscale

    """
    import matplotlib.cm as cm
    if positive:
        cmap = 'brewer_Greys_09'
        brewer_cmap = cm.get_cmap(cmap)
        levs = np.arange(brewer_cmap.N+1) * dx
    else:
        cmap = 'brewer_PuOr_11'
        brewer_cmap = cm.get_cmap(cmap)
        N = brewer_cmap.N
        levs = (np.arange(N+2) - ( brewer_cmap.N + 2 ) /2)* dx

    return {'cmap': cmap, 'levs': levs}

def qxt(cube, dx = None, positive = False,
        **kwargs):
    """
    A convenience wrapper for hovmoller, that helps in the creation of contour
    levels and color bars

    - dx is the spacing of the contours
      dx = None means auto find dx
    - see cmapfac for info on `positive` kwarg

    """
    import iris
    import iris.plot as iplt


    if dx is None:
        dx = cube.data.std() / 1.2
        if dx > 1 :
            dx = np.round(dx, 0)
        elif dx < 1 :
            dx = np.round(dx, 1)

    return hovmoller(cube, **cmapfac(dx=dx, positive=positive))

def hovmoller(cube,
        levs= None,
        cmap = 'brewer_PuOr_11',
        **kwargs):
    """
    Plots a 2d hovmoller diagram given an iris cube.
    """
    import iris
    import iris.quickplot as qplt
    import iris.plot as iplt
    import matplotlib.cm as cm


    plotme = cube
    brewer_cmap = cm.get_cmap(cmap)
    if levs is None:
        levs = brewer_cmap.N

    x  = cube.coord('longitude').points
    t  = cube.coord('time').points

    im = plt.contourf(x, t, plotme.data, levs, extend='both', cmap = brewer_cmap, **kwargs)
    title = cube.name() + " (%s)"%str(cube.units)
    std = sqrt(cube.collapsed('time', iris.analysis.VARIANCE).collapsed('longitude', iris.analysis.MEAN).data)
    m   = cube.collapsed(('time' ,'longitude'), iris.analysis.MEDIAN).data
    plt.gca().set_title('%s Rms %.1f Med %.1f'%(title,  std, m))
    plt.gca().axis('tight')
    plt.gca().set_xlabel(cube.coord('longitude').units)
    plt.gca().set_ylabel('Days')
    plt.colorbar(im, ax = plt.gca())

def climatology(cube):
    """
    Summary hovmoller diagrams for various fields in the netcdf file
    """
    import iris
    import iris.quickplot as qplt
    import iris.plot as iplt
    import matplotlib.pyplot as plt
    from math import sqrt


    mu = cube.collapsed(('time',) , iris.analysis.MEAN)
    mmu = mu.collapsed(('time', 'longitude'), iris.analysis.MEAN)
    std = cube.collapsed(('time',) , iris.analysis.STD_DEV)
    qplt.plot(mu)
    iris.plot.plot(mu + std, 'r--')
    iris.plot.plot(mu - std,'r--')
    ax=  plt.gca().twinx()
    iris.plot.plot(std ,'k-')

    title = qplt._title(cube, False)
    plt.gca().set_title(title)
    plt.gca().axis('tight')

#######################################################################
#                             Iris Analysis                           #
#######################################################################

def clim(cube, axis=('time',)):
    import iris
    return cube.collapsed(axis, iris.analysis.MEAN)


def anom(cube, axis=('time',)):
    import iris
    out = cube - cube.collapsed(axis, iris.analysis.MEAN)
    out.name = cube.name
    return out


#######################################################################
#                        Wheeler-Kiladis Plots                        #
#######################################################################

@autojit
def wk_smooth121(ff, axis):
    """
    1-2-1 Filter for smoothing power spectrum data ala Wheeler-Kiladis
    """
    nr, nc = ff.shape
    bak = np.zeros((nr +2, nc+ 2))
    bak[1:-1,1:-1] = ff

    for i in range(1, nr +1):
        for j in range(1, nc + 1):
            if axis == 0:
                ff[i-1, j-1] =(bak[i-1, j ] + 2.0 * bak[i, j] + bak[i +1, j  ]) / 4.0
            elif axis == 1:
                ff[i-1, j-1] =(bak[i , j-1] + 2.0 * bak[i, j] + bak[i, j +1 ]) / 4.0
            else:
                raise NotImplementedError

    return

def wk_plot(x, t, z, cmap = 'hot_r', smooth = True, title = None, colorbar= False, **kwargs):
    """
    Plotting raw frequency-wavenumber power spectrum for x-t data

    Note:

    I had to reorder the k direction to get eastward waves that actually move
    eastward. Necesary because it is exp(kx-om * t)

    fx = -[ -n/2 ... 0 ... n/2]
    """
    from scipy.fftpack import fft2, fftfreq, fftshift
    from matplotlib import mlab

    # Demean data and calculate pow spec using fft
    # z = np.squeeze(cube.data)
    # z = mlab.demean(z)
    #
    # x = cube.coord('longitude').points
    # t = cube.coord('time').points

    nt, nx = z.shape
    dt = np.diff(t).mean()
    dx = 1.0 / nx

    fz = fftshift(np.abs(fft2(z)), axes=1)
    fx = -fftshift(fftfreq(nx, d = dx ))
    ft = fftfreq(nt, d = dt)

    pz = np.abs(fz)**2/(nx * nt)**2 *2
    pz /= np.sum(pz)
    nt2 = nt/2


    # Smooth the data using 1-2-1 filter
    if smooth:
        wk_smooth121(pz, 0)
        wk_smooth121(pz, 1)



    # Make the plot

    levs=  np.arange(-5.0, -1.5, .25)
    plt.contourf(fx, ft[:nt2], np.log10(pz[:nt2, :]), levs, extend='both', cmap = cmap, ymax=None, **kwargs)
    if colorbar:
        plt.colorbar()


    # Plot the gravity wave dispersion lines


    def plot_symmetric_sw_waves(h=50, c=None, x=15, type='cpd'):
        if c is None:
            c = np.sqrt(9.81 * h)
        else:
            h = c**2 / 9.81
        c *= day_s / 4e7

        plt.plot( fx, fx * c, 'k')
        plt.plot( -fx, fx * c, 'k')

        y = c *x
        plt.text(x, y , '%.0f'%(h),
                horizontalalignment='center',
                verticalalignment='center',
                bbox = {'facecolor':'white'})


    plot_symmetric_sw_waves(h=12, x= 18)
    plot_symmetric_sw_waves(h=25, x = 17)
    plot_symmetric_sw_waves(h=50, x= 13)
    plot_symmetric_sw_waves(c = 50, x =6)

    # Add some visual gridline guides

    plt.plot([0, 0], [0, 10], 'k--')


    def day_gridlines(day, xloc=19):
        day = float(day)
        plt.plot([-40, 40], [ 1.0 / day, 1.0/day], 'k--')
        plt.text(xloc, 1.0/day, '%.0f days'%day,
                    horizontalalignment='right',
                )

    for i in [3, 6, 30]:
        day_gridlines(i)


    plt.grid('on')
    if ymax is None:
        plt.ylim([0, min(.8, ft.max())])
    else:
        plt.ylim([0, ymax])
    plt.xlabel('Zonal Wavenumber')
    plt.ylabel('CPD')
    # plt.title(cube.name())


