import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages

article_style = {'axes.titlesize': 'medium', 'axes.labelsize': 'small'}


def nbsubplots(nrows=1, ncols=1, w=None, h=1.0, aspect=1.0, **kwargs):
    """Make a set of axes with fixed aspect ratio"""
    from matplotlib import pyplot as plt

    if w is not None:
        h = w * aspect
    else:
        w = h / aspect

    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), **kwargs)


def figlabel(*args, fig=None, **kwargs):
    """Put label in figure coords"""
    if fig is None:
        fig = plt.gcf()
    plt.text(*args, transform=fig.transFigure, **kwargs)


def loghist(x,
            logy=True,
            gaussian_comparison=True,
            ax=None,
            lower_percentile=1e-5,
            upper_percentile=100 - 1e-5,
            label='Sample',
            colors=('k', 'g'),
            cstyle={}):
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

    ax.plot(xx, y, label=label, c=colors[0])

    if gaussian_comparison:
        mles = norm.fit(x)
        gpdf = norm.pdf(xx, *mles)
        if logy:
            ax.plot(xx, np.log(gpdf), label='Gauss', **cstyle)
        else:
            ax.plot(xx, gpdf, label='Gauss', **cstyle)

    ax.set_xlim([p1, p2])


def test_loghist():
    from numpy.random import normal

    x = normal(size=1000)
    loghist(x)
    plt.legend()
    plt.show()


def plot2d(x, y, z, ax=None, cmap='RdGy', norm=None, **kw):
    """ Plot dataset using NonUniformImage class

    Parameters
    ----------
    x : (nx,)
    y : (ny,)
    z : (nx,nz)
        
    """
    from matplotlib.image import NonUniformImage
    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(111)

    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    im = NonUniformImage(ax,
                         interpolation='bilinear',
                         extent=xlim + ylim,
                         cmap=cmap)

    if norm is not None:
        im.set_norm(norm)

    im.set_data(x, y, z, **kw)
    ax.images.append(im)
    #plt.colorbar(im)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    def update(z):
        return im.set_data(x, y, z, **kw)

    return im, update


def test_plot2d():
    x = np.arange(10)
    y = np.arange(20)

    z = x[None, :]**2 + y[:, None]**2
    plot2d(x, y, z)
    plt.show()


def func_plot(df,
              func,
              w=1,
              aspect=1.0,
              figsize=None,
              layout=(-1, 3),
              sharex=False,
              sharey=False,
              **kwargs):
    """Plot every column in dataframe with func(series, ax=ax, **kwargs)"""
    ncols = df.shape[1]

    q, r = divmod(ncols, layout[-1])

    nrows = q
    if r > 0:
        nrows += 1

    # Adjust figsize
    if not figsize:
        figsize = (w * layout[-1], w * aspect * nrows)
    fig, axs = plt.subplots(nrows,
                            layout[1],
                            figsize=figsize,
                            sharex=sharex,
                            sharey=sharey)
    lax = axs.ravel().tolist()
    for i in range(ncols):
        ser = df.iloc[:, i]
        ax = lax.pop(0)
        ax.text(.1,
                .8,
                df.columns[i],
                bbox=dict(fc='white'),
                transform=ax.transAxes)
        func(ser, ax=ax, **kwargs)

    for ax in lax:
        fig.delaxes(ax)


def pgram(x, ax=None):
    from scipy.signal import welch
    f, Pxx = welch(x.values)
    if not ax:
        ax = plt.gca()

    ax.loglog(f, Pxx)
    ax.grid()
    ax.autoscale(True, tight=True)


def labelled_bar(x, ax=None, pad=200, **kw):
    """A bar chart for a pandas series x with labelling


    x.plot(kind='hist') labels the xaxis only of the plots, and it is nice to
    label the actual bars directly.

    """
    locs = np.arange((len(x)))

    if not ax:
        fig, ax = plt.subplots()

    rects = ax.bar(locs, x, **kw)
    ax.set_xticks(locs + .5)
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    def autolabel(rects, labels, ax=None, pad=pad):

        for rect, lab in zip(rects, labels):
            lab = str(lab)
            height = rect.get_height() * (1 if rect.get_y() >= 0 else -1)

            kw = {'ha': 'center'}
            if height < 0:
                kw['va'] = 'top'
                height -= ax.transScale.inverted().transform((0, pad))[1]
            else:
                kw['va'] = 'bottom'
                height += ax.transScale.inverted().transform((0, pad))[1]

            ax.text(rect.get_x() + rect.get_width() / 2., height, lab, **kw)

    autolabel(rects, x.index, ax=ax, pad=pad)
    return rects, ax


class LogP1(colors.Normalize):
    """Logarithmic norm for variables from [0, infty]"""

    def __init__(self, data=None, base=10, **kwargs):
        colors.Normalize.__init__(self, **kwargs)
        if data is not None:
            base = np.percentile(data, 90) + 1
        self.base = base

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.log(1 + value) / np.log(self.base))


def plotiter(l,
             ncol=3,
             yield_axis=False,
             figsize=None,
             w=1,
             aspect=1.0,
             tight_layout=True,
             label_dict={},
             sharex=False,
             sharey=False,
             **kwargs):
    """Return a generator wrapping an iterator with matplotlib subplots

    This function is used in a similar manner to seaborns FacetGrid class, but
    is designed to work with standard python data structures.

    Parameters
    ----------
    l: seq
        An iterator whose which will be yielded
    ncol: int, optional
        The maximum number of columns
    yield_axis: bool, optional
        if True, a matplotlib axes object is also yielded

    Yields
    ------
    obj:
       an element of input iterator
    ax: matplotlib.pyplot.axes, optional
       axes object is returned if yield_axis=True

    """

    # Label_dict defaults:
    label_kwargs = dict(labeltype='alpha', loc=(-.05, 1.1))
    label_kwargs.update(label_dict)

    l = list(l)
    n = len(l)

    ncol = min(n, ncol)

    nrow = np.ceil(n / ncol)

    if figsize is None:
        figsize = (w * ncol, aspect * w * nrow)

    plt.figure(figsize=figsize, **kwargs)

    for i in range(n):

        subplot_kwargs = {}
        if i > 0:
            if sharex:
                subplot_kwargs['sharex'] = ax
            if sharey:
                subplot_kwargs['sharey'] = ax


        ax = plt.axes(plt.subplot(nrow, ncol, i + 1, **subplot_kwargs))

        if label_kwargs['labeltype'] == 'alpha':
            label = string.ascii_uppercase[i]
            args = label_kwargs['loc'] + (label,)

            ax.text(*args, transform=ax.transAxes,
                    fontdict=dict(weight='bold', size='x-large'))

        if yield_axis:
            yield l[i], ax
        else:
            yield l[i]

    if tight_layout:
        plt.tight_layout()

    return
