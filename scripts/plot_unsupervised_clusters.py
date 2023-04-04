import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import sys
import os
import cartopy.feature as cfeature
import glob
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from distributed import LocalCluster, Client
from datetime import datetime, timedelta

titles = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Hazardous']
if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/'
    
    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-120, -70, 5, 55]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    variable = sys.argv[1]
    regime = sys.argv[2]
    def get_som(timestamp):
        nearest = np.argmin(np.abs(som_class.index - timestamp))
        if np.abs(som_class.index - timestamp)[nearest] > pd.Timedelta('1D'):
            return np.nan
        return som_class.cluster[nearest]

    # Get variable
    ds = xr.open_mfdataset(nc_path + '%s*.nc' % variable)[variable]
    class_ds = xr.open_mfdataset('jja_aerosol_classes.nc')
    class_ds = class_ds.reindex(time=ds.time, method='nearest', tolerance=timedelta(hours=2))
    print(class_ds.label) 
    som_class = pd.read_csv('som_cluster_10yr_700hpa_00utc.csv', 
                    parse_dates=True, index_col='date')
    som_class = som_class.to_xarray().reindex(date=ds.time.values, method='nearest',
        tolerance=timedelta(days=1))
    anti_cyclone = [4, 8, 12, 13]
    pre_trough = [0, 1, 2, 3]
    post_trough = [7, 11, 14, 15]
    transitional = [5, 6, 9, 10]
    all_regimes = np.arange(0, 16, 1)
    # Restrict to domain
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
    ds.load() 

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    num_clusters = 5
    classification = class_ds.label.values
    print(class_ds.label)
    num_clusters = np.nanmax(classification)
    fig, ax = plt.subplots(int(np.ceil(num_clusters/3)), 4, figsize=(15, 15),
            subplot_kw={'projection': ccrs.PlateCarree()})
    vmin = ds.min().values
    vmax = ds.max().values
    # Change plot settings depending on variable
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
        dsu.load()
        dsv.load()
        streamlines = True
    elif variable == "OCCMASS":
        factor = 1e3
        contours = np.linspace(0, 0.05, 40)
        title_label = "OCCMASS [g/m^3]"
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'OCFLUXU*.nc')['OCFLUXU']

        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsv = xr.open_mfdataset(nc_path + 'OCFLUXV*.nc')['OCFLUXV']

        dsv = dsv[:,
             int(lat_inds[0]):int(lat_inds[-1]),
             int(lon_inds[0]):int(lon_inds[-1])]
        print(ds)
        streamlines = True 
    elif variable == "BCCMASS":
        factor = 1e3
        contours = np.linspace(0, 0.005, 50)
        title_label = "BCCMASS [g/m^3]"
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'BCFLUXU*.nc')['BCFLUXU']

        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsu.load()
        dsv = xr.open_mfdataset(nc_path + 'BCFLUXV*.nc')['BCFLUXV']

        dsv = dsv[:,
             int(lat_inds[0]):int(lat_inds[-1]),
             int(lon_inds[0]):int(lon_inds[-1])]
        dsv.load()
        print(ds)
        streamlines = True
    elif variable == "SO2CMASS" or variable == "SO4CMASS":
        streamlines = True     
        factor = 1e3
        contours = np.linspace(0, 0.005, 50)
        title_label = "%s [g/m^3]" % variable
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'SUFLUXU*.nc')['SUFLUXU']

        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsu.load()
        dsv = xr.open_mfdataset(nc_path + 'SUFLUXV*.nc')['SUFLUXV']

        dsv = dsv[:,
             int(lat_inds[0]):int(lat_inds[-1]),
             int(lon_inds[0]):int(lon_inds[-1])]
        dsv.load()
        print(ds)
    elif variable == "DMSCMASS":
        streamlines = True
        factor = 1e3
        contours = np.linspace(0, 0.005, 50)
        title_label = "%s [g/m^3]" % variable
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'SUFLUXU*.nc')['SUFLUXU']

        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsu.load()
        dsv = xr.open_mfdataset(nc_path + 'SUFLUXV*.nc')['SUFLUXV']

        dsv = dsv[:,
             int(lat_inds[0]):int(lat_inds[-1]),
             int(lon_inds[0]):int(lon_inds[-1])]
        dsv.load()
        print(ds)
    elif variable == "DUFLUXU" or variable == "DUFLUXV":
        factor = 1e3
        contours = np.linspace(-3, 3, 30)
        title_label = "%s [g/m/s]" % variable 
        cmap = 'coolwarm'
        streamlines = True
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'DUFLUXU*.nc')['DUFLUXU']

        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsu.load()
        dsv = xr.open_mfdataset(nc_path + 'DUFLUXV*.nc')['DUFLUXV']

        dsv = dsv[:,
             int(lat_inds[0]):int(lat_inds[-1]),
             int(lon_inds[0]):int(lon_inds[-1])]
        dsv.load()
        print(ds)
    else:
        factor = 1
        contours = np.linspace(-3, 3, 30)
        title_label = "%s [g/m/s]" % variable
        cmap = 'coolwarm'
        streamlines = False
    contours = contours
    # Plot subplots
    
    for i in range(int(np.ceil(num_clusters/3))):
        for j in range(3):
            regime_list = globals()[regime]
            print(len(classification), len(som_class["cluster"]))
            inds = np.argwhere(np.logical_and(classification == i*3+j, 
                       np.isin(som_class["cluster"], regime_list)))
            print(len(inds))
            x, y = np.meshgrid(lon, lat)
            mean = np.squeeze(np.mean(ds.values[inds, :, :], axis=0))
            vmax = np.nanpercentile(ds.values, 95)
            c = ax[i, j].pcolormesh(x, y, mean, cmap=cmap, vmin=0, vmax=vmax)
            if streamlines:
                fluxu_mean = np.squeeze(np.mean(dsu.values[inds, :, :], axis=0))
                fluxv_mean = np.squeeze(np.mean(dsv.values[inds, :, :], axis=0))
                ax[i, j].streamplot(x, y, fluxu_mean, fluxv_mean)
            ax[i, j].set_xlabel('Latitude [deg]')
            ax[i, j].set_ylabel('Longitude [deg]')
            ax[i, j].coastlines()
            ax[i, j].add_feature(cfeature.BORDERS)
            ax[i, j].add_feature(states_provinces)
            if j == 2:
                plt.colorbar(c, ax=ax[i, j + 1], label=title_label)
                ax[i, j + 1].axis('off')
            ax[i, j].set_title('State %d %s N = %.2f days' % (i*3 + j, regime, len(inds)/24))
    plt.tight_layout()
    if not os.path.exists('unsupervised_clusters_%s/' % regime):
        os.mkdir('unsupervised_clusters_%s' % regime)
    fig.savefig('unsupervised_clusters_%s/Clusters_%s.png' % (regime, variable))
