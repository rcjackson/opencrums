import glob
import sys
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
if len(sys.argv) > 0:
    season = sys.argv[1]
    if season == "DJF":
        months = [1, 2, 12]
    elif season == "MAM":
        months = [3, 4, 5]
    elif season == "JJA":
        months = [6, 7, 8]
    elif season == "SON":
        months = [9, 10, 11]
hour = 0
pickle_list = glob.glob(pickles + 'rel-%dhr*.pickle' % hour)

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

ds.close()

classification = ['Good', 'Moderate', 'Unhealthy Sens.', 'Unhealthy', 'Hazardous']
k = 0
lon, lat = np.meshgrid(lon, lat)
soms = []
pre_trough = [0, 1, 2, 3]
post_trough = [7, 11, 14, 15]
anticyclonic = [4, 8, 12, 13]
transitional = [5, 6, 9, 10]
if len(sys.argv) > 3:
    regime = sys.argv[3]
else:
    regime = ""

num_points = np.zeros(5)
mean_relevances = {}
p = open(pickle_list[0], mode="rb")
relevances = pickle.load(p)
input_keys = []
means = {}
maxs = {}
mins = {}
x = {}
for key in relevances.keys():
    if "input_" in key:
        input_keys.append(key)
        mean_relevances[key] = np.zeros((5, lon.shape[0], lon.shape[1]))
        means[key] = np.zeros((5, lon.shape[0], lon.shape[1]))
        maxs[key] = 0
        mins[key] = 0

for k in input_keys:
    relevances[k] = relevances[k].numpy()
    variable = k[6:]
    print("Loading %s" % variable)
    if variable[-4:] == "MASS":
        variable = variable[:-4] + "C" + variable[-4:]
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/%s*.nc' % variable)
    x[k] = ds[variable].values
    mins[k] = x[k].min()
    maxs[k] = x[k].max()
    ds.close()

for picks in pickle_list:
    p = open(picks, mode='rb')
    print(picks)
    relevances = pickle.load(p)
    classes = relevances['output'].numpy().argmax(axis=1)
    aqi = relevances['aqi'].argmax(axis=1)
    for k in input_keys:
        relevances[k] = relevances[k].numpy()
     
    for j in range(len(classes)):
        for k in input_keys:
            r_min = relevances[k][j, :, :].min()
            r_max = relevances[k][j, :, :].max()
            max_v = np.max(np.abs([r_min, r_max]))
            relevances[k][j, :, :] = (relevances[k][j, :, :]) / max_v

    true_times = np.array([x in pre_trough for x in soms])

    soms = np.array([get_som(x) for x in relevances['time']])
    hours = np.array([x.hour for x in relevances['time'].tolist()])
    months = np.array([x.month for x in relevances['time'].tolist()])
    variable = key[6:]

    for j in range(len(classes)):
        sum_all_r = np.squeeze(np.max(np.concatenate(
            [relevances[k][j, :, :] for k in input_keys])))
    
        if regime == "":
            if months[j] in months:
                num_points[classes[j]] = num_points[classes[j]] + 1 
                for k in input_keys:
                    mean_relevances[k][classes[j], :, :] += np.squeeze(
                        relevances[k][j, :, :]) 
                    means[k][classes[j], :, :] += np.squeeze(x[k][j, :, :])
        else:
            if soms[j] in globals()[regime]:
                num_points[classes[j]] = num_points[classes[j]] + 1 
                for k in input_keys:
                    mean_relevances[k][classes[j], :, :] += np.squeeze(
                        relevances[k][j, :, :]) 


print(num_points/24)
r_max = -np.inf
r_mean = 0
i = 0
for j in range(5):
    for k in input_keys:
        print(means[k][j, :, :])
        mean_relevances[k][j,:,:] /= num_points[j]
        means[k][j, :, :] /= num_points[j]
        means[k][j, :, :] = means[k][j, :, :] - x[k].mean(axis=0)
        r_max = np.max([r_max, np.percentile(mean_relevances[k][j, :, :], 95)])
        r_mean += np.mean(mean_relevances[k][j, :, :])
        i += 1

r_mean = r_mean / i

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
        r = np.squeeze(mean_relevances[key][l - 1])
        m = means[key][l - 1, :, :] 
        mmax = np.percentile(means[key], 90)
        #mmin = np.floor(np.log10(means[key][means[key] > 0].min()))
        #mmax = -1
        mmin = np.percentile(means[key], 10)
        mmax = np.abs(np.max([mmax, mmin]))
        mmin = -mmax
        print(r)
        if not "FLUXU" in key:
            c = ax[l - 1].pcolormesh(lon, lat, m,
                    vmin=mmin, vmax=mmax, cmap='coolwarm', label='%s [$kg m^{-2}$]' % key[7:])
            d = ax[l - 1].contourf(lon, lat, r,
                    cmap='coolwarm', alpha=0.5,
                    levels=[-1, -0.25, 0.25, 1])
            bar = plt.colorbar(c, label='%s [$kg m^{-2}$]' % key[6:],
                    ax=ax[l - 1])
        else:
            mv = means[key[:-1] + "V"][l - 1, :, :]
            mag = np.sqrt(m**2 + mv**2)
            c = ax[l - 1].pcolormesh(lon, lat, mag,
                    vmin=mmin, vmax=mag.max(), cmap='Blues', label='%s [$kg m^{-2}$]' % key[7:])
            bar = plt.colorbar(c, label='%s [$kg m s^{-1}$]' % key[6:-1],
                    ax=ax[l - 1])
            ax[l - 1].streamplot(lon, lat, m*1e6, mv*1e6)
            d = ax[l - 1].contourf(lon, lat, r,
                    cmap='coolwarm', alpha=0.2,
                    levels=[-1, -0.25, 0.25, 1])
        ax[l - 1].set_extent(ax_extent)
        ax[l - 1].coastlines()
        ax[l - 1].add_feature(states_provinces)
        ax[l - 1].add_feature(cfeature.BORDERS)
        ax[l - 1].set_title(classification[l - 1])
        ax[l - 1].set_xlabel('Latitude')
        ax[l - 1].set_ylabel('Longitude')

    artists, labels = cs.legend_elements(str_format='{:2.1f}'.format)
    #ax[-1].legend(artists, labels, handleheight=2, framealpha=1)
    fig.savefig('output_relevance_pngs/relevance_%s-%s.png' % (season, key))
    plt.close(fig)

