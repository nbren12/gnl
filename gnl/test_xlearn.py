"""Tests for xlearn module
"""
import numpy as np

import xarray as xr
from xarray.tutorial import load_dataset

from .xlearn import XSVD, XPCA


def test_xsvd():
    air = load_dataset("air_temperature").air.chunk()

    # latitude weighted weights
    weights = np.cos(2*np.pi * air.lat/360) 

    svd = XSVD(feature_dims=['lat', 'lon'], weights=weights, n_components=4)
    svd.fit(air)
    pcs = svd.transform(air)
    recon = svd.inverse_transform(pcs)
    svd.components_


    pca = XPCA(feature_dims=['lat', 'lon'], weights=weights, n_components=4)
    pca.fit(air)
    pca.inverse_transform(pca.transform(air))
    pca.components_
