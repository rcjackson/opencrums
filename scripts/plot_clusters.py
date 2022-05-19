import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import sys
import cartopy.feature as cfeature
import glob
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from distributed import LocalCluster, Client
from datetime import datetime, timedelta

titles = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Hazardous']
if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/'
    air_now_data = glob.glob(
            '/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
    air_now_df = pd.concat(map(pd.read_csv, air_now_data))
    air_now_df['datetime'] = pd.to_datetime(
            air_now_df['DateObserved'] + ' 00:00:00')
    air_now_df = air_now_df.set_index('datetime')
    air_now_df = air_now_df.sort_index()
    print(air_now_df['CategoryNumber'].values.min())
    
    def get_air_now_label(time):
        if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
            return np.nan
        ind = np.argmin(np.abs(air_now_df.index - time))
        return air_now_df['CategoryNumber'].values[ind]
    
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
    num_clusters = 5
    classification = np.array([get_air_now_label(x) for x in ds.time.values])
    fig, ax = plt.subplots(int(num_clusters), 1, figsize=(6, 14),
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
    elif variable == "OCCMASS":
        factor = 1e3
        contours = np.linspace(0, 0.05, 30)
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
        contours = np.linspace(0, 0.003, 30)
        title_label = "BCCMASS [g/m^3]"
        cmap = 'Reds'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'BCFLUXU*.nc')['BCFLUXU']

        dsu = dsu[:,
             int(lat_inds[0]):int(lat_inds[-1]),
            int(lon_inds[0]):int(lon_inds[-1])]
        dsv = xr.open_mfdataset(nc_path + 'BCFLUXV*.nc')['BCFLUXV']

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
    for i in range(int(num_clusters)):
        inds = np.argwhere(classification == i+1)
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
        ax[i].set_title(titles[i])

    fig.savefig('Clusters_%s.png' % variable)
