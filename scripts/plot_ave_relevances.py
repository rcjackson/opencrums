import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import xarray as xr
import numpy as np

pickles = '/lcrc/group/earthscience/rjackson/opencrums/notebooks/'
out_plot_path = '/lcrc/group/earthscience/rjackson/merra_relevances/'
pickle_list = glob.glob(pickles + 'relevance*.pickle')

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

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

mean_relevances = {}

for picks in pickle_list:
    p = open(picks, mode='rb')
    relevances = pickle.load(p)
    classes = relevances['outputs'].argmax(axis=1)
    input_keys = []
    for key in relevances.keys():
        if "input_" in key:
            input_keys.append(key)
            mean_relevances[key] = np.zeros((5, lon.shape[0], lon.shape[1]))

    for j in range(len(classes)):
        sum_all_r = np.squeeze(np.sum(np.concatenate(
            [relevances[k][j, :, :] for k in input_keys])))
        for k in input_keys:
            mean_relevances[k][classes[j], :, :] += np.squeeze(relevances[k][j, :, :]) / sum_all_r

for j in range(5):
    for k in input_keys:
        mean_relevances[k][j,:,:] /= np.sum(classes == j)  

states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

for key in input_keys:
    fig, ax = plt.subplots(5, 1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            figsize=(10, 10))
    for l in range(1, 6):     
        r = np.squeeze(mean_relevances[key][l-1])
        c = ax[l - 1].contourf(lon, lat, r,
            cmap='coolwarm', levels=np.linspace(-0.001, 0.001, 100))
        ax[l - 1].coastlines()
        ax[l - 1].add_feature(states_provinces)
        ax[l - 1].add_feature(cfeature.BORDERS)
        ax[l - 1].set_title(classification[l-1])
        ax[l - 1].set_xlabel('Latitude')
        ax[l - 1].set_ylabel('Longitude')
    fig.savefig('relevance_%s.png' % key)
    plt.close(fig)

