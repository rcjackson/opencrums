import sys
import xarray as xr
import numpy as np

from sklearn.preprocessing import StandardScaler

out_path = '/lcrc/group/earthscience/rjackson/era5-preprocessed/'

def load_data(num_timesteps):
    ds = xr.open_mfdataset(
            '/lcrc/group/earthscience/rjackson/era5-abbreviated/%s/*.grib' % sys.argv[1], engine='cfgrib')
    ds.load()
    ds = ds.sortby('time')
    times = ds.time
    print(ds)
    x = ds["u"].values[:, 3, :160, :272]
    old_shape = x.shape
    # Make sure that we can divide our dataset into the given intervals
    x = x[:(int(old_shape[0] / num_timesteps) * num_timesteps), :, :]
    old_shape = x.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(x, (int(old_shape[0]), old_shape[1] * old_shape[2])))
    x = scaler.transform(np.reshape(x, (old_shape[0], old_shape[1] * old_shape[2])))
    new_shape = (int(old_shape[0]/num_timesteps), num_timesteps,
        old_shape[1], old_shape[2])
    x = np.reshape(x, old_shape)

    y = ds["v"].values[:, 4, :160, :272]
    old_shape = y.shape
    y = y[:(int(old_shape[0] / num_timesteps) * num_timesteps), :, :]
    old_shape = y.shape
    scaler = StandardScaler()
    scaler.fit(np.reshape(y, (int(old_shape[0]), old_shape[1] * old_shape[2])))
    y = scaler.transform(np.reshape(y, (old_shape[0], old_shape[1] * old_shape[2])))
    y = np.reshape(y, old_shape)
    #x = np.stack([x, y, np.zeros_like(x)], axis=-1)
    #print(x.shape)
    #x_dataset = tf.data.Dataset.from_tensor_slices((x, x))
    #shape = x.shape
    lat = ds.latitude[:160]
    lon = ds.longitude[:272]
    u = xr.DataArray(x, dims=('time', 'latitude', 'longitude'))
    v = xr.DataArray(y, dims=('time', 'latitude', 'longitude'))
    ds.close()
    out_name = out_path + '500hPa_winds_preprocessed_%s.nc' % sys.argv[1]
    out_ds = xr.Dataset({'u': u, 'v': v, 'latitude': lat, 'longitude': lon,
        'time': times})
    out_ds.to_netcdf(out_name)
    del ds, x, y
    

load_data(1)
