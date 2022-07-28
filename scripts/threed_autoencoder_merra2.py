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
    feature_min = 0
    feature_max = 1.
elif variable == "OMEGA":
    feature_min = 0.
    feature_max = 1.
elif variable == "T":
    feature_min = 0
    feature_max = 1.
elif variable == "QV":
    feature_min = 0
    feature_max = 1

if isinstance(ds[variable].values, np.ma.MaskedArray):
    input_var = np.nan_to_num(ds[variable].values.filled(np.nan))
else:
    input_var = np.nan_to_num(ds[variable].values, feature_min)

if variable == "QV":
    input_var[input_var == 0] = 1e-6
    input_var = np.log10(input_var)

input_var = input_var[:, :23, :, :]
print(input_var[0,0,:,:])
orig_shape = input_var.shape
lon = ds.lon.values
lat = ds.lat.values

scaler = MinMaxScaler(feature_range=(feature_min, feature_max))
scaler.fit(np.squeeze(input_var.flatten()).reshape(-1,1))
input_var = scaler.transform(np.squeeze(input_var.flatten()).reshape(-1,1)).reshape(orig_shape)

code = 'HOU'
if code == 'HOU':
    ax_extent = [-105, -85, 25, 35]
elif code == 'SEUS':
    ax_extent = [-90, -75, 30, 37.5]


def encoder_model(input_var, no_layers=3, target_shape=(32, 48, 48)):
    input_layer = tf.keras.layers.Input(shape=(input_var.shape[1], input_var.shape[2], input_var.shape[3], 1))
    if (target_shape[0] - input_var.shape[1]) % 2 == 1:
        left_pad0 = int((target_shape[0] - input_var.shape[1]) / 2)
        right_pad0 = left_pad0 + 1
    else:
        left_pad0 = int((target_shape[0] - input_var.shape[1]) / 2)
        right_pad0 = left_pad0

    if (target_shape[1] - input_var.shape[2]) % 2 == 1:
        left_pad1 = int((target_shape[1] - input_var.shape[2]) / 2)
        right_pad1 = left_pad1 + 1
    else:
        left_pad1 = int((target_shape[1] - input_var.shape[2]) / 2)
        right_pad1 = left_pad1 

    if (target_shape[2] - input_var.shape[2]) % 2 == 1:
        left_pad2 = int((target_shape[2] - input_var.shape[3]) / 2)
        right_pad2 = left_pad2 + 1
    else:
        left_pad2 = int((target_shape[2] - input_var.shape[3]) / 2)
        right_pad2 = left_pad2 

    
    padding = ((left_pad0, right_pad0), (left_pad1, right_pad1),
        (left_pad2, right_pad2))
    print(padding)
    layer = tf.keras.layers.ZeroPadding3D(padding)(input_layer)
    encoding_dimension = (input_var.shape[1] / 2**no_layers, input_var.shape[2] / 2**no_layers,
            input_var.shape[3] / 2**no_layers, 1)
    for i in range(no_layers - 1):
        x = tf.keras.layers.Conv3D(2 ** (i + 4), (2, 2, 2), padding="same")(layer)
        layer = tf.keras.layers.MaxPooling3D((2,2,2))(x)
    layer = tf.keras.layers.Conv3D(1, (2, 2, 2), padding="same")(layer)
    encoding = tf.keras.layers.Flatten(name="encoding")(layer)
    reshaped = tf.keras.layers.Reshape(layer.shape[1:])(encoding)

    for i in range(no_layers - 1):
        x = tf.keras.layers.Conv3DTranspose(
                2 ** ((no_layers - i) + 4), (2, 2, 2), padding="same")(layer)
        layer = tf.keras.layers.UpSampling3D((2, 2, 2))(x)
    output_conv = tf.keras.layers.Conv3DTranspose(1, (2, 2, 2), padding="same")(layer)
    output = tf.keras.layers.Cropping3D(padding, name="output")(output_conv)
    return tf.keras.Model(input_layer, output)

model = encoder_model(input_var, no_layers=4)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error")
train_var, test_var = train_test_split(input_var, random_state=666, test_size=0.2)
early_stop = tf.keras.callbacks.EarlyStopping(patience=100)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='../models/3dencoder_%s/3dencoder-{epoch:05d}-{val_loss:.4f}.hdf5' % variable,
        mode='max')
csv_logger = tf.keras.callbacks.CSVLogger('3d_encoder_%s.log' % variable)
model.fit(train_var, train_var, validation_data=(test_var, test_var), epochs=10000,
        callbacks=[early_stop, csv_logger, checkpoint])
model.save('../models/3dencoder-merra%s' % variable)




