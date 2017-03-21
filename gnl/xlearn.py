import numpy as np
import dask.array as da
from .xarray import XRReshaper


class XTransformer(object):
    """Wraps sklearn transformation objects

    So far this probably only works for SVD objects, but not PCA objects because those carry along the mean of the features.

    """
    def __init__(self, model, feature_dims, w=1.0):
        """
        Parameters
        ----------
        model:
            Instantiated sklearn transformer object
        feature_dims: seq of string
            list of xarray dimensions to be used as feature.
        w: DataArray or float
            weights
        """
        self._model = model
        self.feats = feature_dims

        self.w = w

    def fit(self, A):
        # make sure the data is loaded
        A.load()
        rs = XRReshaper(A * np.sqrt(self.w))
        vals, dims = rs.to(self.feats)
        self._model.fit(vals)
        self._rs = rs
        return self

    @property
    def components_(self):
        rs = self._rs
        return rs.get(self._model.components_, ['mode'] + self.feats) / np.sqrt(self.w)

    def transform(self, A):

        # sample dims
        sample_dims = [dim for dim in A.dims
                       if dim not in self.feats]

        # make sure that chunks of A is contigous along the feature dimensions
        if A.chunks is not None:
            assert all(len(A.chunks[A.get_axis_num(dim)]) == 1
                       for dim in self.feats)

        rs = XRReshaper(A * np.sqrt(self.w))
        vals, dims = rs.to(self.feats)

        # new chunk sizes for transform
        chunks = (vals.chunks[0], (self._model.n_components,))

        # use map blocks to perform the transform
        def f(val):
            return self._model.transform(val)

        pcs =  da.map_blocks(f, vals, dtype=vals.dtype, chunks=chunks)

        return rs.get(pcs, sample_dims + ['mode'])


class XSVD(XTransformer):

    def __init__(self, feature_dims, w=1.0, *args, **kwargs):
        "docstring"
        from sklearn.decomposition import TruncatedSVD

        model = TruncatedSVD(*args, **kwargs)
        return super(XSVD, self).__init__(model, feature_dims, w=w)

    def inverse_transform(self, A):
        """This only works for SVD"""

        return A.dot(self.components_ )
