import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import sys
import cartopy.feature as cfeature

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
    variable = sys.argv[1]
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
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    classification = xr.open_dataset('classification_dust.nc')
    num_clusters = classification.classification.max().values
    fig, ax = plt.subplots(int(num_clusters + 1), 1, figsize=(7, 15),
            subplot_kw={'projection': ccrs.PlateCarree()})
    vmin = ds.min().values
    vmax = ds.max().values
    if variable == "DUCMASS":
        factor = 1e3
        contours = np.linspace(0, 0.2, 30)
        title_label = "DUCMASS [g/m^3]"
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'DUFLUXU*.nc')['DUFLUXU']
        
        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsv = xr.open_mfdataset(nc_path + 'DUFLUXV*.nc')['DUFLUXV']

        dsv = dsv[:,
             int(lat_inds[0]):int(lat_inds[-1]),
             int(lon_inds[0]):int(lon_inds[-1])]
        print(ds)
        streamlines = True
    elif variable == "DUFLUXU" or variable == "DUFLUXV":
        factor = 1e3
        contours = np.linspace(-3, 3, 30)
        title_label = "%s [g/m/s]" % variable 
        cmap = 'coolwarm'
        streamlines = False
    else:
        factor = 1
    for i in range(int(num_clusters + 1)):
        inds = np.argwhere(classification.classification.values == i)
        x, y = np.meshgrid(lon, lat)
        mean = np.squeeze(np.mean(ds.values[inds, :, :], axis=0)) * factor
        
        c = ax[i].contourf(x, y, mean, cmap=cmap,
                levels=contours)
        if streamlines:
            fluxu_mean = np.squeeze(np.mean(dsu.values[inds, :, :], axis=0))
            fluxv_mean = np.squeeze(np.mean(dsv.values[inds, :, :], axis=0))
            ax[i].streamplot(x, y, fluxu_mean, fluxv_mean)
        ax[i].set_xlabel('Latitude [deg]')
        ax[i].set_ylabel('Longitude [deg]')
        ax[i].coastlines()
        ax[i].add_feature(cfeature.BORDERS)
        ax[i].add_feature(states_provinces)
        plt.colorbar(c, ax=ax[i])
        ax[i].set_title(title_label)

    fig.savefig('Clusters_%s.png' % variable)
