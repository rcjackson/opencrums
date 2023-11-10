import glob
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import numpy as np

pickles = '/lcrc/group/earthscience/rjackson/opencrums/scripts/relevance_pickles_large_new/'
out_plot_path = '/lcrc/group/earthscience/rjackson/merra_relevances/'
pickle_list = glob.glob(pickles + '*.pickle')
print(pickle_list)
code = 'HOU'
if code == 'HOU':
    ax_extent = [-120, -70, 5, 55]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

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

time = ds["time"].values
ds.close()

classification = ['','Hazardous', 'Good', 'Moderate', 'Unhealthy Sens.', 'Unhealthy']
k = 0
lon, lat = np.meshgrid(lon, lat)
for picks in pickle_list:
    p = open(picks, mode='rb')
    relevances = pickle.load(p)
    classes = relevances['aqi'].argmax(axis=1)
    print(lon.shape)
    input_keys = []
    for key in relevances.keys():
        if "input" in key:
            input_keys.append(key)

    for j in range(10):
        i = 0
        fig, ax = plt.subplots(4, 4,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
            figsize=(13,13))
        sum_all_r = np.sum(np.concatenate(
            [relevances[k][j, :, :] for k in input_keys]))
        r_array = np.stack([relevances[k][j, :, :] for k in input_keys])
        r_array = np.where(np.abs(r_array) < np.nanpercentile(r_array, 99.9), r_array, np.nan)
        for key in input_keys:
            r = np.squeeze(relevances[key][j, :, :])
            #r = np.where(r > 0, r, np.nan)
            #print(np.nanmin(r_array), np.nanmax(r_array))
            r = np.where(np.abs(r) < np.nanpercentile(np.abs(r_array), 99.9), r,
                    np.nanpercentile(np.abs(r_array), 99.9))
            #r = (r - np.nanmin(r_array)) / (np.nanmax(r_array) - np.nanmin(r_array))
            r = r / np.nanmean(r_array)
            #r = np.where(np.abs(r) > 1e-3, r, np.nan)
            #print(np.nanmax(r), np.nanmax(r), np.nanmedian(r), np.nanpercentile(r, 99))

            c = ax[int(i/4), i % 4].pcolormesh(lon, lat, r,
                vmin=0, vmax=10, cmap='coolwarm')
 
            ax[int(i/4), i % 4].coastlines()
            ax[int(i/4), i % 4].set_title(
                str(relevances['time'][j]) + ' :%s \n' % classification[classes[j]] + key.split("_")[1])
            ax[int(i/4), i % 4].set_xlabel('Latitude')
            ax[int(i/4), i % 4].set_ylabel('Longitude')
            plt.colorbar(c, ax=ax[int(i/4), i % 4])
            i += 1
        fig.savefig(out_plot_path + '%d.png' % k)
        k = k + 1            
        plt.close(fig)

