import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from distributed import LocalCluster, Client

if __name__ == '__main__':
    nc_path = '/lcrc/group/earthscience/rjackson/MERRA2/hou_reduced/'

    code = 'HOU'
    if code == 'HOU':
        ax_extent = [-105, -85, 25, 35]
    elif code == 'SEUS':
        ax_extent = [-90, -75, 30, 37.5]

    cluster = LocalCluster()
    client = Client(cluster)
    transformed = -1    
    for variable in sys.argv[1:]:
        ds = xr.open_mfdataset(nc_path + '%s*.nc' % variable)[variable]

        lon_inds = np.argwhere(
                np.logical_and(
                    ds.lon.values >= ax_extent[0],
                    ds.lon.values <= ax_extent[1])).astype(int)
        lat_inds = np.argwhere(
                np.logical_and(
                    ds.lat.values >= ax_extent[2],
                    ds.lat.values <= ax_extent[3])).astype(int)
        lat = ds.lat.values[
            int(lat_inds[0]):int(lat_inds[-1])]
        lon = ds.lon.values[
            int(lon_inds[0]):int(lon_inds[-1])]
        ds = ds[:,
                int(lat_inds[0]):int(lat_inds[-1]),
                int(lon_inds[0]):int(lon_inds[-1])]
        print(ds)
        
        # Convert to BCC anomaly
        vals = ds.values
        scaler = preprocessing.StandardScaler()
        val_shape = vals.shape
        vals = np.reshape(vals, (val_shape[0], val_shape[1] * val_shape[2]))
        scaler = scaler.fit(vals)
        vals = scaler.transform(vals)
        totsum = vals.sum(axis=1)
        valid_inds = np.where(np.isfinite(totsum))
        vals = vals[valid_inds, :]
        vals = np.squeeze(vals)
        coslat = np.cos(np.deg2rad(ds.coords['lat'].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]
        print(wgts)
        sum_variance = 0
        eof_path = '/lcrc/group/earthscience/rjackson/opencrums/models/EOFs/'
        eofs = xr.open_dataset(eof_path + '%s.nc' % variable)
 
        n_components = eofs.dims['n_components']
        times = ds.time.values
        eofs.close()
        ds.close()

        solver = PCA(n_components=n_components)
        solver.fit(vals)
        if np.all(transformed == -1):
            transformed = solver.transform(vals)
        else:
            transformed = np.concatenate(
                [transformed, solver.transform(vals)], axis=1)
    #rmse = np.zeros(20)
    #for num_clusters in range(1, 20):
    #    km = KMeans(n_clusters=num_clusters)
    #    km.fit(vals)
    #    rmse[num_clusters] = km.inertia_
    #    print("%d clusters RMSE: %f" % (num_clusters, km.inertia_))
    #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #ax.plot(np.arange(1, 21), rmse)
    #ax.set_xlabel('# clusters')
    #ax.set_ylabel('RMSE')
    #fig.savefig('kmeans_%s.png' % variable)
    
    n_clusters = 5
    solver = PCA(n_components=n_components)
    solver.fit(vals)
    transformed = solver.transform(vals)
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(transformed)
    out_labels = cluster.labels_
    print(out_labels.shape)
    classification = xr.DataArray(out_labels, dims='time')
    classification.attrs["long_name"] = "KMeans class"
    time = xr.DataArray(times, dims='time')
    ds = xr.Dataset({'time': time, 'classification': classification})
    ds.to_netcdf('classification_dust.nc')


