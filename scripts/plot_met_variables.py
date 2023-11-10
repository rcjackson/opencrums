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

titles = ['AQI Quintile 1', 'AQI Quintile 2', 'AQI Quintile 3', 
        'AQI Quintile 4', 'AQI Quintile 5']

if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/skin_temp/'
    air_now_data = glob.glob(
            '/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
    air_now_df = pd.concat(map(pd.read_csv, air_now_data))
    air_now_df['datetime'] = pd.to_datetime(
            air_now_df['DateObserved'] + ' 00:00:00')
    air_now_df = air_now_df.set_index('datetime')
    air_now_df = air_now_df.sort_index()
    print(air_now_df['CategoryNumber'].values.min())
   
    som_classes = pd.read_csv('som_cluster_10yr_700hpa_00utc.csv',
        parse_dates=True, index_col="date")

    def get_som(timestamp):
        nearest = np.argmin(np.abs(som_classes.index - timestamp))
        if np.abs(som_classes.index - timestamp)[nearest] > pd.Timedelta('1D'):
            return np.nan
        return som_classes.cluster[nearest]

    def get_air_now_label(time):
        if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
            return np.nan
        ind = np.argmin(np.abs(air_now_df.index - time))
        if air_now_df['AQI'].values[ind] < 34.0:
            return 1
        elif air_now_df['AQI'].values[ind] >= 34.0 and air_now_df['AQI'].values[ind] < 42.0:
            return 2
        elif air_now_df['AQI'].values[ind] >= 42.0 and air_now_df['AQI'].values[ind] < 51.0:
            return 3
        elif air_now_df['AQI'].values[ind] >= 51.0 and air_now_df['AQI'].values[ind] < 57.0:
            return 4
        elif air_now_df['AQI'].values[ind] >= 57.0:
            return 5
 
    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-105, -85, 25, 35]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    regime = ""
    if len(sys.argv) > 2:
        regime = sys.argv[2]

    pre_trough = [0, 1, 2, 3]
    post_trough = [7, 11, 14, 15]
    anticyclonic = [4, 8, 12, 13]
    transitional = [5, 6, 9, 10]
    all_regimes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if regime == "":
        regime = "all_regimes"
    cluster = LocalCluster()
    client = Client(cluster)
    variable = sys.argv[1]
    ds = xr.open_mfdataset(nc_path + '*.nc4')[variable]
    lon = ds.lon.values
    lat = ds.lat.values
    print(ds) 
    soms = np.array([get_som(x) for x in ds.time.values])
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    num_clusters = 5
    classification = np.array([get_air_now_label(x) for x in ds.time.values])
    fig, ax = plt.subplots(int(num_clusters), 1, figsize=(8, 16),
            subplot_kw={'projection': ccrs.PlateCarree()})
    vmin = ds.min().values
    vmax = ds.max().values
    if variable == "SLP":
        contours = np.arange(990, 1020, 0.5)
        title_label = "MSLP [hPA]"
        cmap = 'coolwarm'
        streamlines = False
        ds = ds / 1e2
    elif variable == "TS" or variable == "T2M" or variable == "T10M":
        contours = np.arange(268, 308, 1)
        title_label = "%s [K]" % variable
        cmap = 'coolwarm'
        streamlines = False
    else:    
        streamlines = False
        cmap = 'coolwarm'
        contours = np.logspace(-7, -5, 20)
        title_label = "%s" % variable
    
    for i in range(int(num_clusters)):
        if regime == "":
            inds = np.argwhere(classification == i+1)
        else:
            inds = np.argwhere(
                    np.logical_and(np.isin(soms, globals()[regime]), classification == i+1))
        x, y = np.meshgrid(lon, lat)
        mean = np.squeeze(np.mean(ds.values[inds, :, :], axis=0))
        
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
        cbar = plt.colorbar(c, ax=ax[i])
        ax[i].set_title(titles[i])
        cbar.set_label(title_label)

    fig.savefig('Clusters_%s_%s.png' % (variable, regime), bbox_inches='tight')

