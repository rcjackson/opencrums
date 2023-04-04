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
pickles = '/lcrc/group/earthscience/rjackson/opencrums/scripts/relevance_pickles/'
out_plot_path = '/lcrc/group/earthscience/rjackson/merra_relevances/'
pickle_list = glob.glob(pickles + 'rel-0hr*.pickle')


code = 'HOU'
if code == 'HOU':
    ax_extent = [-120, -70, 5, 55]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]



som_classes = pd.read_csv('som_cluster_10yr_700hpa_00utc.csv',
        parse_dates=True, index_col="date")

def get_som(timestamp):
    nearest = np.argmin(np.abs(som_classes.index - timestamp))
    if np.abs(som_classes.index - timestamp)[nearest] > pd.Timedelta('1D'):
        return np.nan
    return som_classes.cluster[nearest]

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

    for k in input_keys:
        relevances[k] = relevances[k].numpy()

    soms = np.array([get_som(x) for x in relevances['time']])
    classes = relevances['output'].numpy().argmax(axis=1)
    for j in range(len(classes)):
        r_max = -np.inf
        r_min = np.inf
        for k in input_keys:
            r_max = np.max([r_max, relevances[k][j, :, :].max()])
            r_min = np.min([r_min, relevances[k][j, :, :].min()])

        for k in input_keys:
            relevances[k][j, :, :] = (relevances[k][j, :, :] - r_min) / (
                    r_max - r_min)

    for j in range(len(classes)):
        num_points[classes[j]] = num_points[classes[j]] + 1
        sum_all_r = np.squeeze(np.max(np.concatenate(
            [relevances[k][j, :, :] for k in input_keys])))
        
        for k in input_keys:
            mean_relevances[k][classes[j], :, :] += np.squeeze(
                    relevances[k][j, :, :]) 
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
    fig, ax = plt.subplots(5, 1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            figsize=(25, 15))
    rel[key] = np.zeros(5)
    rel_std[key] = np.zeros(5)
    for l in range(1, 6):     
        r = np.squeeze(mean_relevances[key][l - 1])
        print(r.max())
        rel[key][l - 1] = r.mean() 
        rel_std[key][l - 1] = r.std()
fig, ax = plt.subplots(1, 5, figsize=(25, 20))
rowLabels = [x[6:] for x in rel.keys()]
colLabels = ['Good', 'Moderate', 'Unhealthy Sens.', 'Unhealthy', 'Hazardous']
cell_text = []
i = 0
j = 0

for j in range(5):
    rel_array = np.array([rel[k][j] for k in rel.keys()])
    inds = np.argsort(rel_array).astype(int)
    rowLabelsSort = [rowLabels[x] for x in inds]
    ax[j].barh(np.arange(len(rowLabels)), rel_array[inds])
    ax[j].set_yticks([x for x in np.arange(len(rowLabels))])
    ax[j].set_yticklabels(rowLabelsSort)
    ax[j].set_xlim([0, 0.6])
    ax[j].set_ylabel('Relevance')
#for k in rel.keys():
#    cell_text.append(['%1.1f' % rel[k][l] for l in range(5)])
#ax.table(cellText=cell_text, rowLabels=rowLabels, colLabels=colLabels,
#        loc='center')

fig.savefig('relevance.png')
plt.close(fig)

