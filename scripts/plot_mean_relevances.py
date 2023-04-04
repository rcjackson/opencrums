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
pickles = '/lcrc/group/earthscience/rjackson/opencrums/scripts/relevance_pickles_large/'
out_plot_path = '/lcrc/group/earthscience/rjackson/merra_relevances/'
if len(sys.argv) > 0:
    start_hour = int(sys.argv[1])
    end_hour = int(sys.argv[2])
else:
    start_hour = -1
    end_hour = 24
hour = 0
pickle_list = glob.glob(pickles + 'rel-%dhr*.pickle' % hour)



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
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/DUCMASS*.nc')
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
stds = {}
for key in relevances.keys():
    if "input_" in key:
        input_keys.append(key)
        mean_relevances[key] = np.zeros((5, lon.shape[0], lon.shape[1]))
        stds[key] = np.zeros((5, lon.shape[0], lon.shape[1]))
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
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_extended/%s*.nc' % variable)
    x[k] = ds[variable].values
    mins[k] = x[k].min()
    maxs[k] = x[k].max()
    ds.close()

for picks in pickle_list:
    p = open(picks, mode='rb')
    relevances = pickle.load(p)
    classes = relevances['output'].numpy().argmax(axis=1)
    aqi = relevances['aqi'].argmax(axis=1)
    for k in input_keys:
        relevances[k] = relevances[k].numpy()
    
    for j in range(len(classes)):
        for k in input_keys:
            r_min = relevances[k][j, :, :].min()
            r_max = relevances[k][j, :, :].max()
            maxv = np.max(np.abs(relevances[k])[j, :, :])
            relevances[k][j, :, :] = relevances[k][j, :, :] / maxv
    true_times = np.array([x in pre_trough for x in soms])

    soms = np.array([get_som(x) for x in relevances['time']])
    hours = np.array([x.hour for x in relevances['time'].tolist()])
    months = np.array([x.month for x in relevances['time'].tolist()])
    variable = key[6:]

    for j in range(len(classes)):
        sum_all_r = np.squeeze(np.max(np.concatenate(
            [relevances[k][j, :, :] for k in input_keys])))
        
        if regime == "":
            if hours[j] >= start_hour and hours[j] <= end_hour:
                num_points[classes[j]] = num_points[classes[j]] + 1 
                for k in input_keys:
                    mean_relevances[k][classes[j], :, :] += np.squeeze(
                        relevances[k][j, :, :]) 
                    means[k][classes[j], :, :] += np.squeeze(x[k][j, :, :])
        else:
            if soms[j] in globals()[regime]:
                if hours[j] >= start_hour and hours[j] <= end_hour: 
                    num_points[classes[j]] = num_points[classes[j]] + 1 
                    for k in input_keys:
                        mean_relevances[k][classes[j], :, :] += np.squeeze(
                            relevances[k][j, :, :]) 
                        means[k][classes[j], :, :] += np.squeeze(x[k][j, :, :])

print(mean_relevances['input_DUMASS'][1, :, :])
print(num_points)
r_max = -np.inf
r_mean = 0
i = 0
for j in range(5):
    for k in input_keys:
        mean_relevances[k][j,:,:] /= num_points[j]
        means[k][j, :, :] /= num_points[j]
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
    fig, ax = plt.subplots(3, 1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            figsize=(10, 15))
    for l in range(1, 4):     
        r = np.squeeze(mean_relevances[key][l - 1])
        m = means[key][l - 1, :, :] 
        
        if not "FLUXU" in key:
            print(m, key)
            if "DUMASS" in key:
                mmax = 2e-5
            elif "BCMASS" in key:
                mmax = 1e-6
            else:
                mmax = np.nanpercentile(x[key], 95)
            mmin = np.percentile(x[key], 5)
           
            #mmax = np.abs(np.max([mmax, mmin]))
            mmin = np.max([0, mmin])
            if "OCMASS" in key:
                mmin = 1e-6
                mmax = 3e-6
            #mmin = -mmax
            #mmax = -1
            mmin = 0
            c = ax[l - 1].pcolormesh(lon, lat, m,
                    vmin=mmin, vmax=mmax, cmap='Greys', label='%s' % key[7:])
            d = ax[l - 1].contourf(lon, lat, r, cmap='coolwarm', alpha=0.5,
                    levels=[-1, -0.1, 0.1, 1])
            bar = plt.colorbar(c, label='%s [$kg\ m^{-2}$]' % key[6:],
                    ax=ax[l - 1])
        else:
            mv = means[key[:-1] + "V"][l - 1, :, :]
            print(np.mean(m), np.mean(mv))
            mag = np.sqrt(m**2 + mv**2)
            c = ax[l - 1].pcolormesh(lon, lat, mag,
                    vmin=mmin, vmax=mag.max(), cmap='Blues', label='%s [$kg\ m^{-2}$]' % key[7:])
            bar = plt.colorbar(c, label='%s [$kg\ m s^{-1}$]' % key[6:-1],
                    ax=ax[l - 1])
            ax[l - 1].streamplot(lon, lat, m*1e6, mv*1e6)
            d = ax[l - 1].contourf(lon, lat, r, alpha=0.5,
                    cmap='coolwarm',
                    levels=[-1, -0.1, 0.1,  1])
        #ax[l - 1].set_extent(ax_extent)
        ax[l - 1].coastlines()
        ax[l - 1].add_feature(states_provinces)
        ax[l - 1].add_feature(cfeature.BORDERS)
        ax[l - 1].set_title(classification[l - 1] + ' %dhr' % (start_hour))
        ax[l - 1].set_xlabel('Latitude')
        ax[l - 1].set_ylabel('Longitude')
    fig.savefig('output_relevance_pngs/relevance_mean%s%dhr-%s.png' % (regime, start_hour, key), dpi=150,
            bbox_inches='tight')
    plt.close(fig)

    

