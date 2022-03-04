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
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
print(ds)
x = ds["DUCMASS"].values
lon = ds["lon"].values
lat = ds["lat"].values

ds.close()
ds = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXU*.nc')
y = ds["DUFLUXU"].values
ds.close()
ds = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXV*.nc')
z = ds["DUFLUXV"].values
ds.close()


cluster_ds = xr.open_dataset('dust_encodings_predict.nc')
cluster = cluster_ds.cluster.values
fig, ax = plt.subplots(4, 4,
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        figsize=(13,7))
lon, lat = np.meshgrid(lon, lat)
vmin = 0
vmax = 5e-4
for i in range(14):
    c = ax[int(i/4), i % 4].contourf(lon, lat, x[cluster == i, :, :].mean(
        axis=0),
            cmap='Reds', levels=np.logspace(-6, np.log10(vmax), 40))
    ax[int(i/4), i % 4].streamplot(lon, lat, y[cluster == i, :, :].mean(axis=0),
        z[cluster == i, :, :].mean(axis=0))
    ax[int(i/4), i % 4].coastlines()
    ax[int(i/4), i % 4].set_title('Cluster %d' % i)
    ax[int(i/4), i % 4].set_xlabel('Latitude')
    ax[int(i/4), i % 4].set_ylabel('Longitude')
plt.colorbar(c, ax=ax[3, 3])
fig.savefig('aerosol_clusters.png')


