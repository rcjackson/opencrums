import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs

from sklearn import preprocessing
from sklearn.decomposition import PCA
from distributed import LocalCluster, Client

if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/'

    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-105, -85, 25, 35]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    cluster = LocalCluster()
    client = Client(cluster)
    variable = 'SSCMASS'
    ds = xr.open_mfdataset(nc_path + '%s*.nc' % variable)[variable]

    lon_inds = np.argwhere(
            np.logical_and(
                ds.lon.values >= ax_extent[0],
                ds.lon.values <= ax_extent[1])).astype(int)
    lat_inds = np.argwhere(
            np.logical_and(
                ds.lat.values >= ax_extent[2],
                ds.lat.values <= ax_extent[3])).astype(int)
    lat = ds.lat.values[
        int(lat_inds[0]):int(lat_inds[-1])]
    lon = ds.lon.values[
        int(lon_inds[0]):int(lon_inds[-1])]
    ds = ds[:, 
            int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
    print(ds)
    # Convert to BCC anomaly
    vals = ds.values
    scaler = preprocessing.StandardScaler()
    val_shape = vals.shape
    vals = np.reshape(vals, (val_shape[0], val_shape[1] * val_shape[2]))
    scaler = scaler.fit(vals)
    vals = scaler.transform(vals)
    totsum = vals.sum(axis=1)
    valid_inds = np.where(np.isfinite(totsum))
    vals = vals[valid_inds, :]
    vals = np.squeeze(vals)
    coslat = np.cos(np.deg2rad(ds.coords['lat'].values))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    print(wgts)
    n_components=10
    solver = PCA(n_components=n_components)
    solver.fit(vals)
    explained_variance_ = solver.explained_variance_
    explained_variance_ratio_ = solver.explained_variance_ratio_
    components_ = solver.components_
    
    exp_variance = xr.DataArray(explained_variance_, dims='n_components')
    exp_variance_ratio = xr.DataArray(
            explained_variance_ratio_, dims='n_components')
    components = xr.DataArray(
            components_, dims=('num_samples', 'num_features'))

    print(explained_variance_)
    print(explained_variance_ratio_)

    out_path = '/lcrc/group/earthscience/rjackson/opencrums/models/EOFs/'
    out_ds = xr.Dataset({'exp_variance': exp_variance,
        'exp_variance_ratio': exp_variance_ratio,
        'components': components})
    out_ds.to_netcdf(out_path + '%s.nc' % variable)
    fig, ax = plt.subplots(2, 5, figsize=(25, 15),
             subplot_kw={'projection': ccrs.PlateCarree()})
    variance = explained_variance_ratio_
    components = np.reshape(components_,
        (n_components, val_shape[1], val_shape[2]))
    x, y = np.meshgrid(lon, lat)
    c = ax[0,0].pcolormesh(x, y, components[0], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[0,0].set_xlabel('Latitude [deg]')
    ax[0,0].set_ylabel('Longitude [deg]')
    ax[0,0].set_title('PC1 %3.2f' % (variance[0] * 100.))
    ax[0,0].coastlines()
    c = plt.colorbar(c, ax=ax[0,0])
    x, y = np.meshgrid(lon, lat)
    c = ax[1,0].pcolormesh(x, y, components[1], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[1,0].set_xlabel('Latitude [deg]')
    ax[1,0].set_ylabel('Longitude [deg]')
    ax[1,0].set_title('PC2 %3.2f' % (variance[1] * 100.))
    ax[1,0].coastlines()
    plt.colorbar(c, ax=ax[1,0])

    x, y = np.meshgrid(lon, lat)
    c = ax[0,1].pcolormesh(x, y, components[2], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[0,1].set_xlabel('Latitude [deg]')
    ax[0,1].set_ylabel('Longitude [deg]')
    ax[0,1].set_title('PC3 %3.2f' % (variance[2] * 100.))
    ax[0,1].coastlines()
    plt.colorbar(c, ax=ax[0,1])
    c = ax[1,1].pcolormesh(x, y, components[3], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[1,1].set_xlabel('Latitude [deg]')
    ax[1,1].set_ylabel('Longitude [deg]')
    ax[1,1].set_title('PC4 %3.2f' % (variance[3] * 100.))
    ax[1,1].coastlines()
    plt.colorbar(c, ax=ax[1, 1])
    c = ax[0,2].pcolormesh(x, y, components[4], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[0,2].set_xlabel('Latitude [deg]')
    ax[0,2].set_ylabel('Longitude [deg]')
    ax[0,2].set_title('PC5 %5.2f' % (variance[4] * 100.))
    ax[0,2].coastlines()
    plt.colorbar(c, ax=ax[0, 2])
    c = ax[1,2].pcolormesh(x, y, components[5], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[1,2].set_xlabel('Latitude [deg]')
    ax[1,2].set_ylabel('Longitude [deg]')
    ax[1,2].set_title('PC6 %3.2f' % (variance[5] * 100.))
    ax[1,2].coastlines()
    plt.colorbar(c, ax=ax[1, 2])
    c = ax[0,3].pcolormesh(x, y, components[6], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[0,3].set_xlabel('Latitude [deg]')
    ax[0,3].set_ylabel('Longitude [deg]')
    ax[0,3].set_title('PC7 %5.2f' % (variance[6] * 100.))
    ax[0,3].coastlines()
    plt.colorbar(c, ax=ax[0, 3])
    c = ax[1,3].pcolormesh(x, y, components[7], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[1,3].set_xlabel('Latitude [deg]')
    ax[1,3].set_ylabel('Longitude [deg]')
    ax[1,3].set_title('PC8 %3.2f' % (variance[7] * 100.))
    ax[1,3].coastlines()
    plt.colorbar(c, ax=ax[1, 3])
    c = ax[0,4].pcolormesh(x, y, components[8], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[0,4].set_xlabel('Latitude [deg]')
    ax[0,4].set_ylabel('Longitude [deg]')
    ax[0,4].set_title('PC9 %5.2f' % (variance[8] * 100.))
    ax[0,4].coastlines()
    plt.colorbar(c, ax=ax[0, 4])
    c = ax[1,4].pcolormesh(x, y, components[9], cmap='coolwarm', vmin=-0.2,
            vmax=0.2)
    ax[1,4].set_xlabel('Latitude [deg]')
    ax[1,4].set_ylabel('Longitude [deg]')
    ax[1,4].set_title('PC10 %3.2f' % (variance[9] * 100.))
    ax[1,4].coastlines()
    plt.colorbar(c, ax=ax[1, 4])
    fig.savefig(out_path + '/pngs/%s.png' % variable)

    ds.close()

