import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import xarray as xr
import pandas as pd
import numpy as np
import sys
import matplotlib.ticker as mticker

from datetime import timedelta
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

pickles = '/lcrc/group/earthscience/rjackson/opencrums/scripts/relevance_pickles/'
out_plot_path = '/lcrc/group/earthscience/rjackson/merra_relevances/'
pickle_list = glob.glob(pickles + 'rel-0hr*.pickle')


code = 'HOU'
if code == 'HOU':
    ax_extent = [-120, -70, 5, 55]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

if len(sys.argv) > 1:
    regime = sys.argv[1]
else:
    regime = ""
som_classes = pd.read_csv('som_cluster_10yr_700hpa_00utc.csv',
        parse_dates=True, index_col="date")

def get_som(timestamp):
    nearest = np.argmin(np.abs(som_classes.index - timestamp))
    if np.abs(som_classes.index - timestamp)[nearest] > pd.Timedelta('1D'):
        return np.nan
    return som_classes.cluster[nearest]

# Get lats, lons for plotting
ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS2010.nc')
print(ds.time)
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

ds.close()

classification = ['Good', 'Moderate', 'Unhealthy Sens.', 'Unhealthy', 'Hazardous']
k = 0
lon, lat = np.meshgrid(lon, lat)
soms = []
pre_trough = [0, 1, 2, 3]
post_trough = [7, 11, 14, 15]
anticyclonic = [4, 8, 12, 13]
transitional = [5, 6, 9, 10]
all_regimes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

num_points = np.zeros(5)
mean_relevances = {}
p = open(pickle_list[0], mode="rb")
relevances = pickle.load(p)
input_keys = []
for key in relevances.keys():
    if "input_" in key:
        input_keys.append(key)
        mean_relevances[key] = np.zeros((5, lon.shape[0], lon.shape[1]))

max_rel_feature = np.zeros((5, len(input_keys), lon.shape[0], lon.shape[1]))
max_relevance = np.zeros((5, lon.shape[0], lon.shape[1]))

for picks in pickle_list:
    p = open(picks, mode='rb')
    relevances = pickle.load(p)
    true_times = np.array([x in pre_trough for x in soms])

    for k in input_keys:
        relevances[k] = relevances[k].numpy()

    soms = np.array([get_som(x) for x in relevances['time']])
 
    classes = relevances['output'].numpy().argmax(axis=1)
    #aqi = relevances['aqi'].argmax(axis=1)
    for j in range(len(classes)):
        r_max = -np.inf
        r_min = np.inf
        for k in input_keys:
            r_max = np.max([r_max, relevances[k][j, :, :].max()])
            r_min = np.min([r_min, relevances[k][j, :, :].min()])

        for k in input_keys:
            relevances[k][j, :, :] = (relevances[k][j, :, :] - r_min) / (
                    r_max - r_min)

    num_grid_points = np.prod(lat.shape)
    for j in range(len(classes)):
#        if not classes[j] == aqi[j]:
#            continue
        if not regime == "":
            som_list = globals()[regime]
            if soms[j] not in som_list:
                continue
        num_points[classes[j]] = num_points[classes[j]] + 1
        sum_all_r = np.squeeze(np.max(np.concatenate(
            [relevances[k][j, :, :] for k in input_keys])))
        #relevances[k] = np.where(relevances[k] > 1 / (num_grid_points * 17), relevances[k], 0)
        rel_array = np.squeeze(
                    np.stack([relevances[k][j, :, :] for k in input_keys], axis=0))
         
        #rel_array = np.where(rel_array > 0, rel_array, np.nan)
        rel_array = (rel_array - np.nanmin(rel_array)) / (np.nanmax(rel_array) - np.nanmin(rel_array))
        rel_array[~np.isfinite(rel_array)] = 0
        max_rel_features = np.squeeze(np.argmax(rel_array, axis=0))
        for num, k in enumerate(input_keys):
#            print(classes[j])
            mean_relevances[k][classes[j], :, :] += np.squeeze(
                    rel_array[num, :, :]) 
            max_rel_feature[classes[j], num, :, :] += max_rel_features == num * 1
            
is_relevant_feature = max_rel_feature.max(axis=1) > 0
max_rel_feature = max_rel_feature.argsort(axis=1).astype(float)[:, -2, :, :]
#max_rel_feature[~is_relevant_feature] = np.nan

print(num_points)
r_max = -np.inf
for j in range(5):
    for k in input_keys:
        mean_relevances[k][j,:,:] /= num_points[j]
        r_max = np.max([r_max, np.percentile(mean_relevances[k][j, :, :], 95)])

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

rel = {}
rel_std = {}
for key in input_keys:
    rel[key] = np.zeros(5)
    rel_std[key] = np.zeros(5)
    for l in range(1, 6):     
        r = np.squeeze(mean_relevances[key][l - 1])
        print(r.max())
        rel[key][l - 1] = r.mean() 
        rel_std[key][l - 1] = r.std()

for j in range(5):
    relevance = np.squeeze(
        np.stack([mean_relevances[k][j, :, :] for k in input_keys], axis=0))
    for k in range(relevance.shape[1]):
        for l in range(relevance.shape[2]):
            max_relevance[j, k, l] = relevance[int(max_rel_feature[j, k, l]), k, l]

#max_rel_feature = np.where(max_relevance > 0.1, max_rel_feature, np.nan)    
rowLabels = [x[6:] for x in input_keys]
#colLabels = ['Quintile 1 AQI', 'Quintile 2 AQI', 'Quintile 3 AQI', 
#             'Quintile 4 AQI', 'Quintile 5 AQI']
colLabels = classification
cell_text = []
fig, ax = plt.subplots(2, 4, figsize=(10, 6),
        subplot_kw=dict(projection=ccrs.PlateCarree()))
i = 0
j = 0
tab20 = cm.get_cmap('tab20', 20)
tab16 = tab20(np.arange(0, 16, 1))
tab16 = ListedColormap(tab16)
for i, k in enumerate(input_keys):
    if "MASS" in k:
        key_loc = k.index("MASS")
        new_key = k[:key_loc] + "C" + "MASS"
        input_keys[i] = new_key

for j in range(3):
    #max_relevance[j] = (max_relevance[j] - max_relevance[j].min()) / (max_relevance[j].max() - max_relevance[j].min())
    cbar_kwargs = {'ticks': np.arange(0, len(input_keys), 1) + 0.5,
            'tick_labels': [x[6:] for x in input_keys]}
    c = ax[0, j].pcolormesh(lon, lat, max_rel_feature[j], 
            cmap=tab16, vmin=0, vmax=16)
    d = ax[1, j].pcolormesh(lon, lat, max_relevance[j],
            cmap='Reds', vmin=0, vmax=1)
    ax[0, j].coastlines()
    ax[0, j].set_ylabel('Latitude [$^{\circ}]$')
    ax[0, j].set_xlabel('Longitude [$^{\circ}]$)')
    ax[0, j].set_title(colLabels[j])
    gl = ax[0, j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax[0, j].add_feature(states_provinces)
    ax[0, j].add_feature(cfeature.BORDERS)
    ax[0, j].set_xlabel('Latitude')
    ax[0, j].set_ylabel('Longitude')
    gl.xlocator = mticker.FixedLocator([-115, -95, -75])
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    if j > 0:
       gl.ylabels_left = False
    gl.ylines = False

    ax[1, j].coastlines()
    ax[1, j].set_ylabel('Latitude [$^{\circ}]$')
    ax[1, j].set_xlabel('Longitude [$^{\circ}]$)')
    ax[1, j].set_title(colLabels[j])

    
cbar = plt.colorbar(c, ax=ax[0, 5],
                    ticks=cbar_kwargs['ticks'], 
                    label='Most relevant feature')
ax[0, 5].axis('off')
cbar.ax.set_yticklabels(cbar_kwargs['tick_labels'])
cbar2 = plt.colorbar(d, ax=ax[1, 5],
                    label='Maximum relevance')
ax[1, 5].axis('off')

#for k in rel.keys():
#    cell_text.append(['%1.1f' % rel[k][l] for l in range(5)])
#ax.table(cellText=cell_text, rowLabels=rowLabels, colLabels=colLabels,
#        loc='center')

fig.savefig('relevance%s_second.png' % regime, bbox_inches='tight')
plt.close(fig)

