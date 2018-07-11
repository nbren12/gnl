import numpy as np
import xarray as xr
from sklearn.decomposition import PCA, TruncatedSVD
import argparse
import gnl


def stack(x):
    return x.stack(f=['lat', 'lon'])\
        .transpose('time', 'f')


def unstack(y, coords, dim_name='dim'):
    dims = [dim_name, 'f']
    coords = {
        'f': coords['f'],
        dim_name: np.arange(y.shape[0])
    }
    return xr.DataArray(y, dims=dims, coords=coords)\
        .unstack('f')


def weighted_eofs(da, wgt='coslat', n_components=10,
                  component_dim_name='dim'):
    """

    Parameters
    ----------
    da : DataArray (lat, lon)
        Input DataArray
    wgt : DataArray (lat) or str
        weights. Default: 'coslat'
    n_components : int
        number of EOFs to extract

    Returns
    -------
    eofs_analysis: xr.Dataset
        dataset with eofs, pcs, and singular values

    """

    if set(da.dims) < set(['lat', 'lon']):
        raise NotImplementedError("This function only accepts lat/lon data at "
                                  "the moment.")

    if wgt == 'coslat':
        wgt = np.cos(np.deg2rad(da.lat))

    # flatten data
    flat_data = stack(da * np.sqrt(wgt))

    # fit pca
    pca = PCA(n_components=n_components)
    pca.fit(flat_data)

    # get eofs
    eofs = unstack(pca.components_, flat_data.coords,
                   dim_name=component_dim_name) / np.sqrt(wgt)

    # singular values
    variance = xr.DataArray(pca.explained_variance_,
                            dims=[component_dim_name],
                            coords={component_dim_name: np.arange(n_components)})

    # get pcs
    dims = ['time', component_dim_name]
    coords = {'time': flat_data.time,
              component_dim_name: np.arange(n_components)}
    pcs = pca.transform(flat_data)
    pcs = xr.DataArray(pcs, dims=dims, coords=coords)

    # prepare output dataset
    return xr.Dataset({
        'pcs': pcs,
        'eofs': eofs,
        'wgt': wgt,
        'variance': variance
    })


def weighted_dmd(da, wgt='coslat',
                 component_dim_name='dim',
                 n_components=10,
                 **kwargs):
    """

    Parameters
    ----------
    da : DataArray (lat, lon)
        Input DataArray
    wgt : DataArray (lat) or str
        weights. Default: 'coslat'
    n_components : int
        number of EOFs to extract

    Returns
    -------
    eofs_analysis: xr.Dataset
        dataset with eofs, pcs, and singular values

    """

    if set(da.dims) < set(['lat', 'lon']):
        raise NotImplementedError("This function only accepts lat/lon data at "
                                  "the moment.")

    if wgt == 'coslat':
        wgt = np.cos(np.deg2rad(da.lat))

    # flatten data
    X = stack(da * np.sqrt(wgt))

    kwargs['n_components'] = n_components
    lam, phi, atilde = gnl.exact_dmd(X, **kwargs)

    # get pcs
    dims = ['time', component_dim_name]

    comp_coord = np.arange(n_components)
    dmds = unstack(phi, X.coords, dim_name=component_dim_name) \
           / np.sqrt(wgt)

    lam = xr.DataArray(lam, dims=component_dim_name,
                       coords={component_dim_name: comp_coord})

    return xr.Dataset({
        'dmds': dmds,
        'lam': lam,
    })




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-n", "--n_components", type=int, default=10)
    parser.add_argument("-v", "--variable", type=str, default='olr')

    args = parser.parse_args()

    output = args.output
    data_path = args.input
    n_components = args.n_components

    # prepare data
    ds = xr.open_dataset(data_path)
    olr = ds[args.variable].sel(time=slice('1980', None))

    weighted_eofs(olr, n_components=n_components) \
        .to_netcdf(output)


if __name__ == '__main__':
    main()
