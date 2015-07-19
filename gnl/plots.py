import numpy as np
import matplotlib.pyplot as plt

article_style = {
    'axes.titlesize': 'medium',
    'axes.labelsize': 'small'
}

def nbsubplots(nrows=1, ncols=1, w=None, h=1.0, aspect=1.0, **kwargs):
    """Make a set of axes with fixed aspect ratio"""
    from matplotlib import pyplot as plt

    if w is not None:
        h = w * aspect
    else:
        w = h / aspect

    return  plt.subplots(nrows,ncols, figsize=(w * ncols ,h * nrows), **kwargs)

def figlabel(*args, fig=None, **kwargs):
    """Put label in figure coords"""
    if fig is None:
        fig = plt.gcf()
    plt.text(*args, transform=fig.transFigure, **kwargs)


def loghist(x, logy=True, gaussian_comparison=True, ax=None,
            lower_percentile=1e-5, upper_percentile=100-1e-5,
            label='Sample'):
    """
    Plot log histogram of given samples with normal comparison using
    kernel density estimation
    """
    from scipy.stats import gaussian_kde, norm
    from numpy import percentile

    if ax is None:
        ax = plt.axes()

    p = gaussian_kde(x)

    npts = 100

    p1 = percentile(x, lower_percentile)
    p2 = percentile(x, upper_percentile)
    xx = np.linspace(p1, p2, npts)

    if logy:
        y = np.log(p(xx))
    else:
        y = p(xx)

    ax.plot(xx, y, label=label)

    if gaussian_comparison:
        mles = norm.fit(x)
        gpdf = norm.pdf(xx, *mles)
        if logy:
            ax.plot(xx, np.log(gpdf),  label='Gauss')
        else:
            ax.plot(xx, gpdf,  label='Gauss')

    ax.set_xlim([p1, p2])


def test_loghist():
    from numpy.random import normal

    x = normal(size=1000)
    loghist(x)
    plt.legend()
    plt.show()

def plot2d(x, y, z, ax=None, cmap='RdGy', **kw):
    """ Plot dataset using NonUniformImage class
    
    Args:
        x (nx,)
        y (ny,)
        z (nx,nz)
        
    """
    from matplotlib.image import NonUniformImage
    if ax is None:
        fig = plt.gcf()
        ax  = fig.add_subplot(111)
   
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max()) 
    
    im = NonUniformImage(ax, interpolation='bilinear', extent=xlim + ylim,
                        cmap=cmap)
   
    im.set_data(x,y,z, **kw)
    ax.images.append(im)
    #plt.colorbar(im)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    return im

def test_plot2d():
    x = np.arange(10)
    y = np.arange(20)
    
    z = x[None,:]**2 + y[:,None]**2
    plot2d(x,y,z)
    plt.show()


def func_plot(df, func, w=1, aspect=1.0, figsize=None, layout=(-1,3),
              sharex=False, sharey=False,
             **kwargs):
    """Plot every column in dataframe with func(series, ax=ax, **kwargs)"""
    ncols = df.shape[1]

    q, r = divmod(ncols, layout[-1])

    nrows = q
    if r> 0:
        nrows +=1

    # Adjust figsize
    if not figsize:
        figsize = (w * layout[-1], w * aspect * nrows)
    fig, axs = plt.subplots(nrows, layout[1], figsize=figsize, sharex=sharex, sharey=sharey)
    lax = axs.ravel().tolist()
    for i in range(ncols):
        ser = df.iloc[:,i]
        ax  = lax.pop(0)
        ax.text(.1,.8, df.columns[i], bbox=dict(fc='white'), transform=ax.transAxes)
        func(ser, ax=ax, **kwargs)

    for ax in lax:
        fig.delaxes(ax)

def pgram(x,ax=None):
    from scipy.signal import welch
    f, Pxx = welch(x.values)
    if not ax:
        ax = plt.gca()

    ax.loglog(f, Pxx)
    ax.grid()
    ax.autoscale(True, tight=True)
