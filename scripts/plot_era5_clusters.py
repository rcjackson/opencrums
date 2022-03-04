import sys
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt




code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/era5-preprocessed/*unscaled*.nc')
ds.load()
ds = ds.sortby('time')
x = np.squeeze(ds["u"].values)
time = ds["time"].values
old_shape = x.shape
lat = ds["latitude"].values
lon = ds["longitude"].values
# Make sure that we can divide our dataset into the given intervals
y = np.squeeze(ds["v"].values)
which_lats = np.where(np.logical_and(ds["latitude"].values >= ax_extent[2],
    ds["latitude"].values <= ax_extent[3]))[0]
which_lons = np.where(np.logical_and(ds["longitude"].values >= ax_extent[0],
    ds["longitude"].values <= ax_extent[1]))[0]
#x = x[:,which_lats[0]:(which_lats[0]+96),which_lons[0]:(which_lons[0]+96)]
#y = y[:,which_lats[0]:(which_lats[0]+96),which_lons[0]:(which_lons[0]+96)]
magnitude = np.sqrt(x**2 + y**2)
old_shape = y.shape
shape = x.shape
ds.close()

cluster_ds = xr.open_dataset('wind_encodings_predict.nc')
cluster = cluster_ds.cluster.values
fig, ax = plt.subplots(5, 3, subplot_kw=dict(projection=ccrs.PlateCarree()),
        figsize=(20, 10))
lon, lat = np.meshgrid(lon, lat)

for i in range(14):
    c = ax[int(i / 3), i % 3].contourf(lon, lat,
            magnitude[cluster == i, :, :].mean(axis=0),
            levels=np.linspace(0, 75, 75), cmap='Greys')
    ax[int(i / 3), i % 3].streamplot(lon, lat,
            x[cluster == i, :, :].mean(axis=0),
        y[cluster == i, :, :].mean(axis=0), color='k')
    #plt.colorbar(c, ax=ax[int(i / 2), i % 2])
    ax[int(i / 3), i % 3].coastlines()
    ax[int(i / 3), i % 3].set_xlabel('Latitude')
    ax[int(i / 3), i % 3].set_ylabel('Longitude')
    ax[int(i / 3), i % 3].set_title('Cluster %d' % i)
plt.colorbar(c, ax=ax[int(i / 3), i % 3])
fig.savefig('ERA5_winds_clusters.png')


