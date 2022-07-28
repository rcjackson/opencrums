import xarray as xr
import numpy as np
import sys

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

nc_path = '/lcrc/group/earthscience/rjackson/MERRA2_met/*.nc4'

ds = xr.open_mfdataset(nc_path)

variable = sys.argv[1]
if len(sys.argv) > 2:
    epoch_no = int(sys.argv[2])
else:
    epoch_no = 1

if variable == "U" or variable == "V":
    feature_min = -30.
    feature_max = 30.
elif variable == "OMEGA":
    feature_min = -10.
    feature_max = 10.
elif variable == "T":
    feature_min = 230.
    feature_max = 320.
elif variable == "QV":
    feature_min = -6
    feature_max = -2

if isinstance(ds[variable].values, np.ma.MaskedArray):
    input_var = np.nan_to_num(ds[variable].values.filled(np.nan))
else:
    input_var = np.nan_to_num(ds[variable].values, feature_min)
input_var = input_var[:, :23, :, :]
if variable == "QV":
    input_var[input_var == 0] = 1e-6
    input_var = np.log10(input_var)

print(input_var[0,0,:,:])
orig_shape = input_var.shape
lon = ds.lon.values
lat = ds.lat.values
time = ds.time.values

scaler = MinMaxScaler(feature_range=(feature_min, feature_max))
scaler.fit(np.squeeze(input_var.flatten()).reshape(-1,1))
input_var = scaler.transform(np.squeeze(input_var.flatten()).reshape(-1,1)).reshape(orig_shape)

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]


model = tf.keras.models.load_model('../models/3dencoder-merra%s' % variable)
model.summary()
encoder = tf.keras.models.Model(model.input, model.get_layer(index=8).output)
encodings = encoder.predict(input_var)
encodings = np.reshape(encodings, (encodings.shape[0], np.prod(encodings.shape[1:])))
out_ds = xr.Dataset({'time': (['time'], time), 'encoding': (['time', 'encoding_len'], encodings)})
out_ds.to_netcdf('3dencodings-%s.nc' % variable)




