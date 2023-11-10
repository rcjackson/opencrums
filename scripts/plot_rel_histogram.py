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

import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep

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

land_s = sys.argv[1]
if land_s.lower() == "land":
    land_m = 1
else:
    land_m = 0

if len(sys.argv) > 2:
    regime = sys.argv[2]
else:
    regime = "all_regimes"
som_classes = pd.read_csv('som_cluster_10yr_700hpa_00utc.csv',
        parse_dates=True, index_col="date")

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))

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

land_mask = np.zeros_like(lon)
land_mask = np.where(np.logical_and.reduce((lon > -100, lon < -93, lat > 28, lat < 32)), 1, 0)
#for i in range(lon.shape[0]):
#    for j in range(lat.shape[1]):
#        land_mask = is_land(lon[i, j], lat[i, j])
        
print(land_mask)
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
    #classes = relevances['aqi'].argmax(axis=1)
    for j in range(len(classes)):
        r_max = -np.inf
        r_min = np.inf
        for k in input_keys:
            r_max = np.max([r_max, relevances[k][j, :, :].max()])
            r_min = np.min([r_min, relevances[k][j, :, :].min()])


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
        rel_array = np.squeeze(
                    np.stack([relevances[k][j, :, :] for k in input_keys], axis=0))
         
        rel_array = rel_array / np.nanmean(rel_array)
        #rel_array[rel_array < 1] = -np.inf
        max_rel_features = np.squeeze(np.argmax(rel_array, axis=0))
        for num, k in enumerate(input_keys):
            mean_relevances[k][classes[j], :, :] += np.squeeze(
                    rel_array[num, :, :]) 
            max_rel_feature[classes[j], num, :, :] += max_rel_features == num * 1

for i in range(5):
    for j in range(len(input_keys)):
        max_rel_feature[i, j] = np.where(land_mask == land_m, max_rel_feature[i, j], -1)

is_relevant_feature = max_rel_feature.max(axis=1) > 0

max_rel_feature = max_rel_feature.sum(axis=2).sum(axis=2)
rowLabels = [x[6:] for x in input_keys]
colLabels = ['Quintile 1 AQI', 'Quintile 2 AQI', 'Quintile 3 AQI', 
             'Quintile 4 AQI', 'Quintile 5 AQI']
cell_text = []
fig, ax = plt.subplots(1, 5, figsize=(15, 5))
i = 0
j = 0
for i, k in enumerate(input_keys):
    if "MASS" in k:
        key_loc = k.index("MASS")
        new_key = k[:key_loc] + "C" + "MASS"
        input_keys[i] = new_key

which_variables = {"SO4CMASS": 0, "DUCMASS": 1, "BCCMASS": 2, "DMSCMASS": 3, "OCCMASS": 4}
my_hist = np.zeros((5, 5))
for j in range(5):
    l = 0
    for i, k in enumerate(input_keys):
        var_name = k.split("_")[1]
        if var_name in list(which_variables.keys()):
            l = which_variables[var_name]
            my_hist[j, l] = max_rel_feature[j, i] / num_points[j]
            l = l + 1

width = 0.15
fig, ax = plt.subplots(1, 1)
x_labels = classification
x_ticks = [1, 2, 3, 4, 5]
for j, key in enumerate(which_variables):
    offset = width * (j - 2.5)
    rects = ax.bar(np.array(x_ticks) + offset, my_hist[:, j], label=key, width=0.2)
    #ax.bar_label(rects, padding=3)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Total points / number of hours")
ax.set_ylim([0, 50])
ax.set_xlim([0.5, 3.5])
ax.set_xlabel("EPA AirNow PM2.5 Class")
ax.legend()
fig.savefig('relevance_hist_%s_%s.png' % (land_s, regime), bbox_inches='tight')
plt.close(fig)

