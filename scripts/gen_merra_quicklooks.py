import xarray as xr
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import os
import sys

from glob import glob

merra_path = '/lcrc/group/earthscience/rjackson/MERRA2/%s/' % sys.argv[1]
merra_quicklook_path = '/lcrc/group/earthscience/rjackson/MERRA2/quicklook/'

code = 'HOU'
if code == 'HOU':
   ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
   ax_extent = [-90, -75, 30, 37.5]

state_shapes = cartopy.io.shapereader.Reader(
    '/home/rjackson/.local/share/cartopy/shapefiles/natural_earth/cultural/ne_10m_admin_1_states_provinces_lines.shp')
coast_lines = cartopy.io.shapereader.Reader(
    '/home/rjackson/.local/share/cartopy/shapefiles/natural_earth/physical/50m_coastline.shp')
file_list = sorted(glob(merra_path + '/**/*.nc4', recursive=True))

usa = [country for country in state_shapes.records() 
          if country.attributes['adm0_name'] == 'United States of America']
for fi in file_list:
    my_ds = xr.open_dataset(fi)
    for hour in range(24):
        fig, ax = plt.subplots(2, 2, figsize=(20, 10),
           subplot_kw={'projection': ccrs.PlateCarree()})

        my_ds.isel(time=hour).SO2CMASS.plot(ax=ax[0,0], cmap='Reds', 
            vmin=0, vmax=1e-4)
        for x in usa:
            ax[0,0].add_feature(
                cfeature.ShapelyFeature([x.geometry], crs=ccrs.PlateCarree()),
                                     facecolor='none', edgecolor='black', lw=1)
        for record in coast_lines.records():
            ax[0,0].add_feature(
                cfeature.ShapelyFeature([record.geometry], crs=ccrs.PlateCarree()), 
                facecolor='none', edgecolor='black', lw=1)
        ax[0,0].set_extent(ax_extent)
        my_ds.isel(time=hour).BCCMASS.plot(ax=ax[0,1], cmap='Reds', 
            vmin=0, vmax=1e-4)
        for x in usa:
            ax[0,1].add_feature(
                cfeature.ShapelyFeature([x.geometry], crs=ccrs.PlateCarree()), 
                                     facecolor='none', edgecolor='black', lw=1)
        for record in coast_lines.records():
            ax[0,1].add_feature(
                cfeature.ShapelyFeature([record.geometry], crs=ccrs.PlateCarree()), 
                facecolor='none', edgecolor='black', lw=1)

        ax[0,1].coastlines()
        ax[0,1].set_extent(ax_extent)
        my_ds.isel(time=hour).DUCMASS.plot(ax=ax[1,0], cmap='Reds',  
            vmin=0, vmax=1e-4)
        for x in usa:
            ax[1,0].add_feature(
                cfeature.ShapelyFeature([x.geometry], crs=ccrs.PlateCarree()),
                                     facecolor='none', edgecolor='black', lw=1)
        for record in coast_lines.records():
            ax[1,0].add_feature(
                cfeature.ShapelyFeature([record.geometry], crs=ccrs.PlateCarree()),
                facecolor='none', edgecolor='black', lw=1)

        ax[1,0].coastlines()
        ax[1,0].set_extent(ax_extent)
        my_ds.isel(time=hour).OCCMASS.plot(ax=ax[1,1], cmap='Reds',
            vmin=0, vmax=1e-4)
        for x in usa:
            ax[1,1].add_feature(
                cfeature.ShapelyFeature([x.geometry], crs=ccrs.PlateCarree()),
                                     facecolor='none', edgecolor='black', lw=1)
        for record in coast_lines.records():
            ax[1,1].add_feature(
                cfeature.ShapelyFeature([record.geometry], crs=ccrs.PlateCarree()),
                facecolor='none', edgecolor='black', lw=1)

        ax[1,1].coastlines()
        ax[1,1].set_extent(ax_extent)
        if not os.path.isdir(merra_quicklook_path + my_ds.time[hour].dt.strftime('%Y').values):
            os.makedirs((merra_quicklook_path + my_ds.time[hour].dt.strftime('%Y').values))
        out_file_path = merra_quicklook_path + my_ds.time[hour].dt.strftime('%Y/%Y%m%d-%H%M%S.png').values
        print("Saving " + out_file_path)
        fig.savefig(out_file_path, dpi=150)
        plt.close(fig)
