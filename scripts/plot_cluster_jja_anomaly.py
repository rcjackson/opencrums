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

#titles = ['AQI Quintile 1', 'AQI Quintile 2', 'AQI Quintile 3', 
#        'AQI Quintile 4', 'AQI Quintile 5']
titles = ['Good', 'Moderate', 'Unhealthy Sens.']

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
    ds = xr.open_mfdataset(nc_path + '%s*.nc' % variable)[variable]
    ds.load()
    lon = ds.lon.values
    lat = ds.lat.values
    print(ds) 
    soms = np.array([get_som(x) for x in ds.time.values])
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    num_clusters = 3
    classification = np.array([get_air_now_label(x) for x in ds.time.values])
    fig, ax = plt.subplots(1, int(num_clusters)+1, figsize=(12, 4),
            subplot_kw={'projection': ccrs.PlateCarree()})
    vmin = ds.min().values
    vmax = ds.max().values
    log_scale = False
    ds_by_season = ds.groupby("time.season").mean(dim="time")
    ds = ds - ds_by_season.sel(season="JJA")

    if variable == "DUCMASS":
        factor = 1e3
        contours = np.linspace(0.2e-5, 0.2e-3, 30) * 1e6
        title_label = "DUCMASS JJA anomaly [$mg\ m^{-2}$]"
        cmap = 'coolwarm'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'DUFLUXU*.nc')['DUFLUXU']
        
        dsv = xr.open_mfdataset(nc_path + 'DUFLUXV*.nc')['DUFLUXV']
        streamlines = True
    elif variable == "OCCMASS":
        factor = 1e3
        contours = np.linspace(0.05e-5, 0.05e-3, 30) * 1e6
        title_label = "OCCMASS JJA anomaly [$mg\ m^{-2}$]"
        cmap = 'coolwarm'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'OCFLUXU*.nc')['OCFLUXU']

        dsv = xr.open_mfdataset(nc_path + 'OCFLUXV*.nc')['OCFLUXV']

        print(ds)
        streamlines = True 
    elif variable == "SO2CMASS" or variable == "SO4CMASS":
        streamlines = True
        factor = 1e3
        contours = np.linspace(1e-6, 1e-5, 50) * 1e6
        title_label = "%s JJA anomaly [$mg\ m^{-2}]$" % variable
        log_scale = False
        cmap = 'coolwarm'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'SUFLUXU*.nc')['SUFLUXU']

        dsu.load()
        dsv = xr.open_mfdataset(nc_path + 'SUFLUXV*.nc')['SUFLUXV']

        dsv.load()
        print(ds)
    elif variable == "BCCMASS":
        factor = 1e3
        contours = np.linspace(0.00001, 0.01, 30) * 1e3
        title_label = "BCCMASS JJA anomaly [$mg\ m^{-2}$]"
        cmap = 'coolwarm'
        # Load fluxes for streamline plot
        dsu = xr.open_mfdataset(nc_path + 'BCFLUXU*.nc')['BCFLUXU']

        dsv = xr.open_mfdataset(nc_path + 'BCFLUXV*.nc')['BCFLUXV']

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
        streamlines = False
        cmap = 'coolwarm'
        contours = np.logspace(-8, -6, 20) * 1e6
        title_label = "%s JJA anomaly [$mg\ m^{-2}$]" % variable
    
    for i in range(int(num_clusters)):
        if regime == "":
            inds = np.argwhere(classification == i+1)
        else:
            inds = np.argwhere(
                    np.logical_and(np.isin(soms, globals()[regime]), classification == i+1))
        x, y = np.meshgrid(lon, lat)
        mean = np.squeeze(np.mean(ds.values[inds, :, :], axis=0)) * 1e6
        if log_scale == True: 
            norm = colors.LogNorm(10**np.round(np.log10(contours[0])), 10**np.round(np.log10(contours[-1]))) 
            c = ax[i].pcolormesh(x, y, mean, cmap=cmap, norm=norm)
        else:
            c = ax[i].pcolormesh(x, y, mean, cmap=cmap, vmin=-contours[-1]/10, vmax=contours[-1] / 10)
        if streamlines:
            fluxu_mean = np.squeeze(np.mean(dsu.values[inds, :, :], axis=0))
            fluxv_mean = np.squeeze(np.mean(dsv.values[inds, :, :], axis=0))
            ax[i].streamplot(x, y, fluxu_mean, fluxv_mean)
        gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
        ax[i].set_xlabel('Latitude [deg]')
        ax[i].set_ylabel('Longitude [deg]')
        ax[i].coastlines()
        ax[i].add_feature(cfeature.BORDERS)
        ax[i].add_feature(states_provinces)
        ax[i].set_title(titles[i])
        
        #gl.xlocator = mticker.FixedLocator([-115, -95, -75])
        gl.xlines = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabels_right = False
        if i > 0:
            gl.ylabels_left = False
        gl.ylines = False
    cbar = plt.colorbar(c, ax=ax[-1])
    ax[-1].axis('off')
    cbar.set_label(title_label)

    fig.savefig('Clusters_anomaly_%s_%s_small.png' % (variable, regime), bbox_inches='tight')

