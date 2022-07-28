import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import InterclusterDistance, silhouette_visualizer

variable_list = ["U", "V"]
nc_path = '/lcrc/group/earthscience/rjackson/MERRA2_met/*.nc4'

merra_ds = xr.open_mfdataset(nc_path)
merra_ds.load()
encodings = []
for x in variable_list:
    encoder_ds = xr.open_dataset('3dencodings-%s.nc' % x)
    print(encoder_ds)
    encodings.append(encoder_ds["encoding"].values)
    encoder_ds.close()

encodings = np.concatenate(encodings, axis=1)
print(encodings.shape)
pc = PCA(n_components=30)
encodings_reduced = pc.fit_transform(encodings)
print(np.cumsum(pc.explained_variance_ratio_))
var_string = '_'.join(variable_list)

n_clusters=15
km = KMeans(n_clusters)
km.fit(encodings_reduced)
labels = km.predict(encodings_reduced)
pressure = 850.
for i in range(n_clusters):
    magnitude = np.sqrt(merra_ds.U.sel(lev=pressure).values**2 + merra_ds.V.sel.(lev=pressure).values**2)
    wind_mean = np.mean(magnitude[labels == i, :, :], axis=0)
    u_mean = np.mean(merra_ds.U.sel(lev=pressure).values[labels == i, :, :], axis=0)
    v_mean = np.mean(merra_ds.V.sel(lev=pressure).values[labels == i, :, :], axis=0)
    lon_m, lat_m = np.meshgrid(merra_ds.lon.values, merra_ds.lat.values)
    fig, ax = plt.subplots(int(n_clusters/3), 3)
    ax[n_clusters % 3, (n_clusters/3)].streamplot(lon_m, lat_m, merra_ds
    ax[n_clusters % 3, (n_clusters/3)].contourf(lon_m, lat_m, wind_mean)



