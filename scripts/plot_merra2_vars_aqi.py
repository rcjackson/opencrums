import pandas as pd
import xarray as xr
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

air_now_data = glob('/lcrc/group/earthscience/rjackson/epa_air_now/*.csv')
air_now_df = pd.concat(map(pd.read_csv, air_now_data))
air_now_df['datetime'] = pd.to_datetime(air_now_df['DateObserved'] + ' 00:00:00')
air_now_df = air_now_df.set_index('datetime')
air_now_df = air_now_df.sort_index()
print(air_now_df['CategoryNumber'].values.min())

classification = ['Good', 'Moderate', 'Unhealthy Sens.', 'Unhealthy', 'Hazardous']
batch_size = 16

def get_air_now_label(time):
    if np.min(np.abs((air_now_df.index - time))) > timedelta(days=1):
        return np.nan
    ind = np.argmin(np.abs(air_now_df.index - time))
    return air_now_df['CategoryNumber'].values[ind]

species = "DU"
ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sCMASS*.nc' % species).sortby('time')
times = np.array(list(map(pd.to_datetime, ds.time.values)))
mass = ds["%sCMASS" % species].values
ds.close()
if species == "SO4" or species == "DMS" or species == "SO2":
    inp = "SU"
else:
    inp = species
ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sFLUXU*.nc' % inp).sortby('time')
fluxu = ds["%sFLUXU" % inp].values
ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%sFLUXV*.nc' % inp).sortby('time')
fluxv = ds["%sFLUXV" % inp].values
air_now = np.array([get_air_now_label(x) for x in times])

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

lon = ds["lon"].values
lat = ds["lat"].values
lon_inds = np.argwhere(
            np.logical_and(
                ds.lon.values >= ax_extent[0],
                ds.lon.values <= ax_extent[1])).astype(int)
lat_inds = np.argwhere(
            np.logical_and(
                ds.lat.values >= ax_extent[2],
                ds.lat.values <= ax_extent[3])).astype(int)
lon = lon[lon_inds]
lat = lat[lat_inds]

lon, lat = np.meshgrid(lon, lat)
fig, ax = plt.subplots(5, 1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            figsize=(10, 15))
states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

if species == "BC":
    vmax = 2e-6
if species == "SO4" or species == "SO2":
    vmax = 1e-5
if species == "DU":
    vmax = 1e-4

for l in range(1, 6):
    inds = np.argwhere(air_now == (l))
    mean_mass = np.mean(mass[inds, :, :], axis=0)
    mean_fluxu = np.mean(fluxu[inds, :, :], axis=0)
    mean_fluxv = np.mean(fluxv[inds, :, :], axis=0)
    c = ax[l - 1].pcolormesh(lon, lat, np.squeeze(mean_mass),
            cmap='Reds', vmin=0, vmax=vmax)
    bar = plt.colorbar(c, label='Mean %s mass' % species,
                ax=ax[l - 1])

    ax[l - 1].coastlines()
    ax[l - 1].add_feature(states_provinces)
    ax[l - 1].add_feature(cfeature.BORDERS)
    ax[l - 1].set_title(classification[l-1])
    ax[l - 1].set_xlabel('Latitude')
    ax[l - 1].set_ylabel('Longitude')
fig.savefig('%s.png' % species, bbox_inches='tight', dpi=300)
