"""Sklearn wrappers
"""
import xarray as xr
import dask.array as da
import numpy as np
from dask.array.linalg import svd_compressed


class XDataMatrix(object):
    """Class for making data matrices
    """
    def __init__(self, feature_dims=None):
        "docstring"
        self.feature_dims = feature_dims

    def _sample_dims(self, A):
        return [dim for dim in A.dims
                if dim not in self.feature_dims]

    def __call__(self, A):
        out =  A.stack(feature=self.feature_dims, sample=self._sample_dims(A))\
            .transpose('sample', 'feature')

        self.feature_coord = out.feature

        return out

    @property
    def _coords(self):
        return {'feature': self.feature_coord}

    def unstack_feats(self, A):
        if len(self.feature_dims) > 1:
            return A.unstack('feature')
        else:
            return A.rename({'feature': self.feature_dims[0]})

def unstack_array(x, dims, coords):
    new_coords = {}

    for i, dim in enumerate(dims):
        if dim in coords:
            new_coords[dim] = coords[dim]
        else:
            new_coords[dim] = np.arange(x.shape[i])

    return xr.DataArray(x, dims=dims, coords=new_coords)


class TruncatedSVD(object):
    """Dask friendly TruncatedSVD class
    """
    def __init__(self, **kwargs):
        self.svd_kwargs = kwargs

    def fit(self, A):

        if not hasattr(A, 'dask'):
            A = da.from_array(A, A.shape)

        n_comps = self.svd_kwargs.pop('n_components')
        _, _, vt = svd_compressed(A, n_comps, **self.svd_kwargs)

        self.components_ = vt

        return self

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

    TODO remove all stacking/unstacking code to XDataMatrix class
    """


    # add init method
    def __init__(self, *args, feature_dims=[], weights=1.0, **kwargs):

        # class used for massaging data
        self._rs = XDataMatrix(feature_dims)

        self._model = self._parent(*args, **kwargs)

        if hasattr(weights, 'dims'):
            if any(dim not in feature_dims for dim  in weights.dims):
                raise ValueError("Weights must only be defined along feature dimensions.")

        self.weights = weights

    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self._model, key)


    def fit(self, A):
        vals = self._rs(A * np.sqrt(self.weights))
        self._model.fit(vals.data)
        return self


    def transform(self, A):
        # sample dims
        vals = self._rs(A * np.sqrt(self.weights))
        dout = self._model.transform(vals.data)
        return unstack_array(dout, ['sample', 'mode'], vals.coords)

    def inverse_transform(self, A):
        dout = self._model.inverse_transform(A.data)

        coord = {'sample': A.sample, 'feature': self._rs.feature_coord}
        return  unstack_array(dout, ['sample', 'feature'], coord)\
            .pipe(self._rs.unstack_feats)/np.sqrt(self.weights)

    @property
    def components_(self):
        coords = {'feature': self._rs.feature_coord}
        out =  unstack_array(self._model.components_, ['mode', 'feature'], coords)\
               .pipe(self._rs.unstack_feats) /np.sqrt(self.weights)

        return out




class XSVD(XTransformerMixin):
    _parent = TruncatedSVD

class XPCA(XSVD):
    _parent = PCA
