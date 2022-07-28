import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import xarray as xr
import pandas as pd
import numpy as np

from datetime import timedelta
pickles = '/lcrc/group/earthscience/rjackson/opencrums/notebooks/'
out_plot_path = '/lcrc/group/earthscience/rjackson/merra_relevances/'
pickle_list = glob.glob(pickles + 'relevance*.pickle')

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]



som_classes = pd.read_csv('som_cluster_10yr_700hpa_00utc.csv',
        parse_dates=True, index_col="date")

def get_som(timestamp):
    nearest = np.argmin(np.abs(som_classes.index - timestamp))
    if np.abs(som_classes.index - timestamp)[nearest] > pd.Timedelta('1D'):
        return np.nan
    print(np.abs(som_classes.index - timestamp)[nearest])
    return som_classes.cluster[nearest]

print(som_classes)
# Get lats, lons for plotting
ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
print(ds.time)
x = ds["DUCMASS"].values
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

time = ds["time"].values
ds.close()

classification = ['Good', 'Moderate', 'Unhealthy Sens.', 'Unhealthy', 'Hazardous']
k = 0
lon, lat = np.meshgrid(lon, lat)
soms = []
pre_trough = [0, 1, 2, 3]
post_trough = [7, 11, 14, 15]
anticyclonic = [4, 8, 12, 13]
transitional = [5, 6, 9, 10]

num_points = np.zeros(5)
mean_relevances = {}
p = open(pickle_list[0], mode="rb")
relevances = pickle.load(p)
input_keys = []
for key in relevances.keys():
    if "input_" in key:
        input_keys.append(key)
        mean_relevances[key] = np.zeros((5, lon.shape[0], lon.shape[1]))

for picks in pickle_list:
    p = open(picks, mode='rb')
    relevances = pickle.load(p)
    true_times = np.array([x in pre_trough for x in soms])

    soms = np.array([get_som(x) for x in relevances['time']])
    classes = relevances['outputs'].argmax(axis=1)
            
    for j in range(len(classes)):
        if not soms[j] in pre_trough:
            continue
        num_points[classes[j]] = num_points[classes[j]] + 1
        sum_all_r = np.squeeze(np.max(np.concatenate(
            [relevances[k][j, :, :].numpy() for k in input_keys])))
        
        for k in input_keys:
            mean_relevances[k][classes[j], :, :] += np.squeeze(
                    relevances[k][j, :, :].numpy()) 
print(num_points)
r_max = -np.inf
for j in range(5):
    for k in input_keys:
        mean_relevances[k][j,:,:] /= num_points[j]
        r_max = np.max([r_max, np.max(mean_relevances[k][j, :, :])])

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

for key in input_keys:
    fig, ax = plt.subplots(5, 1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            figsize=(10, 15))
    for l in range(1, 6):     
        r = np.squeeze(mean_relevances[key][l-1])
        r = r / r_max
        c = ax[l - 1].pcolormesh(lon, lat, r,
            cmap='coolwarm', vmin=-0.1, vmax=0.1)
        bar = plt.colorbar(c, label='Normalized relevance',
                ax=ax[l - 1])
       
        ax[l - 1].coastlines()
        ax[l - 1].add_feature(states_provinces)
        ax[l - 1].add_feature(cfeature.BORDERS)
        ax[l - 1].set_title(classification[l-1])
        ax[l - 1].set_xlabel('Latitude')
        ax[l - 1].set_ylabel('Longitude')
    fig.savefig('relevance_%s_pretrough.png' % key)
    plt.close(fig)

