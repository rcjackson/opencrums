import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import sys
import cartopy.feature as cfeature
import glob
import pandas as pd
import matplotlib.colors as colors
import matplotlib.ticker as mticker

from sklearn import preprocessing
from sklearn.decomposition import PCA
from distributed import LocalCluster, Client
from datetime import datetime, timedelta
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

titles = ['AQI Quintile 1', 'AQI Quintile 2', 'AQI Quintile 3', 
        'AQI Quintile 4', 'AQI Quintile 5']

if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/'
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
        return air_now_df['AQI'].values[ind]
 
    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-105, -85, 25, 35]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    regime = ""
    if len(sys.argv) > 1:
        regime = sys.argv[1]

    pre_trough = [0, 1, 2, 3]
    post_trough = [7, 11, 14, 15]
    anticyclonic = [4, 8, 12, 13]
    transitional = [5, 6, 9, 10]
    all_regimes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if regime == "":
        regime = "all_regimes"
    cluster = LocalCluster()
    client = Client(cluster)
    ds = {}
    keys = ["DUCMASS", "BCCMASS", "SO4CMASS"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for k in keys:
        ds[k] = xr.open_mfdataset(nc_path + '%s*.nc' % k)[k]
        ds[k].load()
        lon = ds[k].lon.values
        lat = ds[k].lat.values
        print(ds) 
        #soms = np.array([get_som(x) for x in ds[k].time.values])
        ds_by_season = ds[k].groupby("time.season").mean(dim="time")
        ds[k] = ds[k] - ds_by_season.sel(season="JJA")
        ds[k] = ds[k].sel(lat=29.7604, lon=-95.3698, method='nearest')
        ds[k] = ds[k].sel(time=slice('2015-06-01', '2015-09-01'))
        aqi = np.array([get_air_now_label(x) for x in ds[k].time.values])
    
    for k in keys:
        (ds[k] * 1e6).plot(ax=ax, label=k)
    print(aqi) 
    ax.plot(ds[k].time, aqi, label='AQI', linewidth=4, color='k')
    ax.set_ylabel("Anomaly [$mg m^{-2}$]")
    ax.legend()
    #inds = np.argwhere(np.isin(soms, globals()[regime])) 
    #x_bins = np.linspace(0, 150, 25)
    #y_bins = np.linspace(-contours[-1] / 10, contours[-1] / 10, 50)
    #print(ds.values.shape)
    #hist, x_bins, y_bins = np.histogram2d(np.squeeze(aqi[inds]), np.squeeze(ds.values[inds]) * 1e6, bins=(x_bins, y_bins), normed=True)
    #x_bins = (x_bins[:-1] + x_bins[1:]) / 2.
    #y_bins = (y_bins[:-1] + y_bins[1:]) / 2.
    #x_bins_in, y_bins_in = np.meshgrid(x_bins, y_bins)
    #hist = np.where(hist == 0, np.nan, hist)
    #print(x_bins_in.shape, y_bins_in.shape)
    #c = ax.pcolormesh(x_bins_in, y_bins_in, hist.T, cmap='viridis')
    #ax.set_xlabel('EPA PM2.5 AQI')
    #ax.set_ylabel(title_label)
    #plt.colorbar(c, label='Normalized frequency')
    fig.savefig('Timeseries.png', bbox_inches='tight')

