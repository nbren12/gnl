from sklearn.decomposition import TruncatedSVD
import numpy as np


def exact_dmd(x, center=True, **kwargs):
    """
    Parameters
    ----------
    x : (n, f)
        nd array time by features

    n_components : int
        number of components to use

    Returns
    -------
    dmd_modes:
       :
    """

    svd = TruncatedSVD(**kwargs)

    if center:
        x = x - x.mean(axis=0)

    x, xp = x[:-1], x[1:]
    xp = xp.values

    # fit svd
    svd.fit(x)

    u = svd.components_.T
    sig_inv = np.diag(1 / svd.singular_values_)
    vt = x @ u @ sig_inv

    atilde = sig_inv @ (vt.T @ (xp @ u))

    # left eigenvectors of atilde
    # each row is an eigenvector
    lam, w = np.linalg.eig(atilde.T)
    w = w.T

    # dmd modes
    phi = w @ sig_inv @ (xp.T @ vt).T

    return lam, phi, atilde
