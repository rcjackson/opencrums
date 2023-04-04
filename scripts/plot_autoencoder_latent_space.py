import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy as np

z_means = np.linspace(0, 20, 10)
z_stds = np.linspace(0, 20, 10)

# Get lats, lons for plotting
ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/DUCMASS*.nc')
code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]

lats = np.linspace(ax_extent[2], ax_extent[3], 32)
lons = np.linspace(ax_extent[0], ax_extent[1], 16)
lats, lons = np.meshgrid(lats, lons)
print(lats.shape, lons.shape)
model = tf.keras.models.load_model('../models/autoencoder-%s' % sys.argv[1])

print("Generating figure...")
fig, ax = plt.subplots(10, 10, figsize=(40, 40), subplot_kw=dict(projection=ccrs.PlateCarree()))
for i in range(10):
    for j in range(10):
        img = model.decoder.predict(np.array([[z_means[i], z_stds[j]]]))
        ax[i, j].pcolormesh(lats, lons, img.squeeze(), vmin=0, vmax=1, cmap='Reds')
        ax[i, j].coastlines()
        ax[i, j].set_title('Mean = %3.2f, std = %3.2f' % (z_means[i], z_stds[j]))
fig.savefig('latent-space-%s.png' % sys.argv[1], bbox_inches='tight')




