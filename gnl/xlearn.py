"""Sklearn wrappers
"""
import dask.array as da
import numpy as np
from dask.array.linalg import svd_compressed

from .xarray import XRReshaper


class TruncatedSVD(object):
    """Dask friendly TruncatedSVD class
    """
    def __init__(self, n_components=10, **kwargs):
        self.n_components  = n_components
        self.svd_kwargs = kwargs

    def fit(self, A):
        _, _, vt = svd_compressed(A, self.n_components, **self.svd_kwargs)

        self.components_ = vt

    def transform(self, A):
        return A.dot(self.components_.T)

    def inverse_transform(self, A):
        return A.dot(self.components_)

class PCA(TruncatedSVD):
    def fit(self, A):
        self.empirical_mean_ = A.mean(axis=0)
        B = A - self.empirical_mean_
        return super(PCA, self).fit(B)

    def transform(self, A):
        return super(PCA, self).transform(A-self.empirical_mean_)

    def inverse_transform(self, A):
        return super(PCA, self).inverse_transform(A)+self.empirical_mean_


class XTransformerMixin(object):
    """Mixin which enables an sklearn like interface for objects with the
    transformer interface
    """


    # add init method
    def __init__(self, *args, feature_dims=[], weights=1.0, **kwargs):
        self.feature_dims = feature_dims
        self._model = self._parent(*args, **kwargs)
        self.weights = weights

    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self._model, key)

    def fit(self, A):
        feats = self.feature_dims
        rs = XRReshaper(A* np.sqrt(self.weights))
        vals, dims = rs.to(feats)

        self._rs = rs
        return self._model.fit(vals )


    def transform(self, A):
        # sample dims

        sample_dims = [dim for dim in A.dims
                       if dim not in self.feature_dims]

        feats = self.feature_dims
        rs = XRReshaper(A*self.weights)
        vals, dims = rs.to(feats)

        dout = self._model.transform(vals)
        return rs.get(dout, sample_dims + ['mode'])

    def inverse_transform(self, A):

        nmodes = A.shape[-1]
        vals = A.data.reshape((-1, nmodes))

        dout = self._model.inverse_transform(vals)
        return self._rs.get(dout, self._rs.dims)/np.sqrt(self.weights)

    @property
    def components_(self):
        rs = self._rs
        return rs.get(self._model.components_, ['mode'] + self.feature_dims)\
            / np.sqrt(self.weights)

class XSVD(XTransformerMixin):
    _parent = TruncatedSVD

class XPCA(XSVD):
    _parent = PCA
