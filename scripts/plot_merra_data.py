import sys
import xarray as xr
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os

img_path = '/lcrc/group/earthscience/rjackson/merra2_aerosol_plots/'

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
print(ds.time)
x = ds["DUCMASS"].values
lon = ds["lon"].values
lat = ds["lat"].values
time = ds["time"].values
ds.close()
ds = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXU*.nc')
y = ds["DUFLUXU"].values
ds.close()
ds = xr.open_mfdataset(
        '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUFLUXV*.nc')
z = ds["DUFLUXV"].values
ds.close()


for i in range(len(time)):
    fig, ax = plt.subplots(1, ,
        subplot_kw=dict(projection=ccrs.PlateCarree()),
        figsize=(13,7))
    lon, lat = np.meshgrid(lon, lat)
    
    c = ax.contourf(lon, lat, x[i, :, :].mean(
        axis=0),
            cmap='Reds', levels=np.logspace(-6, np.log10(vmax), 40))
    ax.streamplot(lon, lat, y[cluster == i, :, :].mean(axis=0),
        z[cluster == i, :, :].mean(axis=0))
    ax.coastlines()
    ax.set_title('Time %s' % str(time[i]))
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    plt.colorbar(c, ax=ax)
    fig.savefig(img_path + '%s.png' % time[i].strftime('%Y%m%d.%H%M%S'))


